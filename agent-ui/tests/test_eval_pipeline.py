import ast
import asyncio
import importlib.util
import io
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock


AGENT_DIR = Path(__file__).resolve().parents[1]
DEMO_FILES = sorted(AGENT_DIR.glob("*.py"))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


class _FakeStream:
    def __init__(self, events=()):
        self.events = events

    def __aiter__(self):
        async def iterate():
            for event in self.events:
                yield event

        return iterate()

    async def close(self):
        return None


class _FakeCompletions:
    def __init__(self, events=()):
        self.events = events
        self.calls = []

    async def create(self, **kwargs):
        call = dict(kwargs)
        if "messages" in call:
            call["messages"] = [dict(message) for message in call["messages"]]
        self.calls.append(call)
        return _FakeStream(self.events)


class _FakeOpenAI:
    def __init__(self, events=()):
        self.completions = _FakeCompletions(events)
        self.chat = types.SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self):
        self.closed = True


def _install_sdk_stubs():
    assemblyai = types.ModuleType("assemblyai")
    assemblyai.__path__ = []

    class _Placeholder:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _AsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(
                completions=_FakeCompletions()
            )

    class _Events:
        SpeechStarted = object()
        Turn = object()
        Error = object()

    class _RealTimeTranscriber:
        def __init__(self, **_kwargs):
            pass

        def on(self, *_args):
            pass

        def connect(self, *_args):
            pass

        def stream(self, *_args):
            pass

        def set_params(self, *_args):
            pass

        def disconnect(self, **_kwargs):
            pass

    assemblyai.AsyncTTSSession = _Placeholder
    assemblyai.AsyncLLMGateway = _Placeholder
    assemblyai.Client = _Placeholder
    assemblyai.TTSSession = _Placeholder
    assemblyai.TextToSpeech = _Placeholder
    assemblyai.TTSConfig = _Placeholder
    assemblyai.TTSContainer = types.SimpleNamespace(wav=object())
    assemblyai.TTSError = type("TTSError", (Exception,), {})
    assemblyai.RealTimeTranscriber = _RealTimeTranscriber

    streaming = types.ModuleType("assemblyai.streaming")
    streaming.__path__ = []
    realtime = types.ModuleType("assemblyai.streaming.v3")
    realtime.RealTimeError = type("RealTimeError", (Exception,), {})
    realtime.RealTimeEvents = _Events
    realtime.RealTimeParameters = _Placeholder
    realtime.RealTimeSessionParameters = _Placeholder
    realtime.RealTimeTranscriber = _RealTimeTranscriber
    realtime.SpeechStartedEvent = type("SpeechStartedEvent", (), {})
    realtime.TurnEvent = type("TurnEvent", (), {})
    assemblyai.streaming = streaming

    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = _AsyncOpenAI
    openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
    openai.PermissionDeniedError = type("PermissionDeniedError", (Exception,), {})
    openai.RateLimitError = type("RateLimitError", (Exception,), {})
    sounddevice = types.ModuleType("sounddevice")
    sounddevice.RawInputStream = _Placeholder
    sounddevice.RawOutputStream = _Placeholder

    class _FastAPI:
        def __init__(self, **_kwargs):
            pass

        def get(self, *_args, **_kwargs):
            return lambda function: function

        def websocket(self, *_args, **_kwargs):
            return lambda function: function

    class _HTTPException(Exception):
        def __init__(self, *, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _PlainTextResponse:
        def __init__(self, content, *, headers=None):
            self.body = content
            self.headers = headers or {}

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FastAPI
    fastapi.HTTPException = _HTTPException
    fastapi.WebSocket = type("WebSocket", (), {})
    fastapi.WebSocketDisconnect = type(
        "WebSocketDisconnect", (Exception,), {}
    )
    fastapi_responses = types.ModuleType("fastapi.responses")
    fastapi_responses.HTMLResponse = _Placeholder
    fastapi_responses.FileResponse = _Placeholder
    fastapi_responses.PlainTextResponse = _PlainTextResponse

    sys.modules.update(
        {
            "assemblyai": assemblyai,
            "assemblyai.streaming": streaming,
            "assemblyai.streaming.v3": realtime,
            "openai": openai,
            "sounddevice": sounddevice,
            "fastapi": fastapi,
            "fastapi.responses": fastapi_responses,
        }
    )


def _load_demo_modules():
    os.environ.setdefault("ASSEMBLYAI_API_KEY", "eval-placeholder")
    os.environ.setdefault("OPENAI_API_KEY", "eval-placeholder")
    _install_sdk_stubs()

    modules = {}
    for path in DEMO_FILES:
        name = f"agent_ui_eval_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[path.stem] = module
    return modules


MODULES = _load_demo_modules()


def _event(token):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                delta=types.SimpleNamespace(content=token)
            )
        ]
    )


class _FakeAsyncTTSSession:
    def __init__(self):
        self.generated = []
        self.flush_count = 0
        self.cancel_count = 0

    async def generate(self, text):
        self.generated.append(text)

    async def cancel(self):
        self.cancel_count += 1

    async def flush(self):
        self.flush_count += 1
        return self.flush_count

    async def audio(self, flush_id):
        yield bytes((flush_id, 0))


class _FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.delta_sent = asyncio.Event()

    async def send_json(self, payload):
        self.sent.append(("json", payload))
        if payload.get("type") == "assistant_delta":
            self.delta_sent.set()

    async def send_bytes(self, payload):
        self.sent.append(("bytes", payload))


class _FakeConversationWebSocket(_FakeWebSocket):
    def __init__(self, module, payloads):
        super().__init__()
        self.module = module
        payloads = list(payloads)
        self.payloads = iter(payloads)
        self.expected_finals = sum(
            payload.get("type") == "text"
            for payload in payloads
            if isinstance(payload, dict)
        )
        self.final_count = 0
        self.text_count = 0
        self.first_final = asyncio.Event()
        self.turn_finished = asyncio.Event()
        if self.expected_finals == 0:
            self.turn_finished.set()

    async def accept(self):
        return None

    async def receive(self):
        try:
            payload = next(self.payloads)
        except StopIteration:
            await self.turn_finished.wait()
            return {"type": "websocket.disconnect"}
        if isinstance(payload, dict) and payload.get("type") == "text":
            self.text_count += 1
            if self.text_count > 1:
                await self.first_final.wait()
        if isinstance(payload, bytes):
            return {"type": "websocket.receive", "bytes": payload}
        return {
            "type": "websocket.receive",
            "text": json.dumps(payload),
        }

    async def close(self, **_kwargs):
        return None

    async def send_json(self, payload):
        await super().send_json(payload)
        if payload.get("type") == "assistant_final":
            self.final_count += 1
            self.first_final.set()
            if self.final_count >= self.expected_finals:
                self.turn_finished.set()


class _FakeSyncTTSSession:
    def __init__(self, *_args, **_kwargs):
        self.begin = types.SimpleNamespace(id="eval-session")
        self.generated = []
        self.flush_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def generate(self, text):
        self.generated.append(text)

    def flush(self):
        self.flush_count += 1
        return self.flush_count

    def audio(self, flush_id):
        return [bytes((flush_id, 0))]


class _BlockingOpenAIStream:
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield _event("Partial reply")
        await asyncio.Event().wait()

    async def close(self):
        self.closed = True


def _wav_bytes(pcm=b"\x10\x00"):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(pcm)
    return buffer.getvalue()


class EvalPipelineTests(unittest.TestCase):
    def test_every_demo_parses_and_imports_without_live_dependencies(self):
        self.assertGreaterEqual(len(DEMO_FILES), 10)
        for path in DEMO_FILES:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                self.assertIn(path.stem, MODULES)

    def test_assemblyai_key_uses_environment_before_infisical(self):
        module = MODULES["infisical_secrets"]
        with mock.patch.dict(
            os.environ,
            {"ASSEMBLYAI_API_KEY": "env-key"},
            clear=True,
        ):
            self.assertEqual(module.get_assemblyai_api_key(), "env-key")

    def test_voice_app_exposes_every_saved_conversation_and_message(self):
        module = MODULES["web_app"]
        store_module = MODULES["conversation_store"]
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "history.sqlite")
            store = store_module.ConversationStore(db_path)
            try:
                store.create("older")
                store.append("older", "user", "First question")
                store.append(
                    "older",
                    "assistant",
                    "First answer",
                    interrupted=True,
                )
                store.create("newer")
                store.append("newer", "user", "Another conversation")
            finally:
                store.close()

            with mock.patch.object(module, "HISTORY_DB", db_path):
                conversations = asyncio.run(module.list_conversations())
                history = asyncio.run(module.get_conversation("older"))

            self.assertEqual(
                {item["id"] for item in conversations["conversations"]},
                {"older", "newer"},
            )
            self.assertEqual(
                conversations["conversations"][0]["id"],
                "newer",
            )
            self.assertEqual(
                [item["content"] for item in history["messages"]],
                ["First question", "First answer"],
            )
            self.assertTrue(history["messages"][1]["interrupted"])

    def test_voice_app_history_returns_not_found_for_unknown_conversation(self):
        module = MODULES["web_app"]
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                module,
                "HISTORY_DB",
                str(Path(temp_dir) / "history.sqlite"),
            ):
                with self.assertRaises(module.HTTPException) as raised:
                    asyncio.run(module.get_conversation("missing"))
        self.assertEqual(raised.exception.status_code, 404)

    def test_voice_app_downloads_transcript_with_speakers_and_utc_times(self):
        module = MODULES["web_app"]
        store_module = MODULES["conversation_store"]
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "history.sqlite")
            store = store_module.ConversationStore(db_path)
            store.create("call/one")
            store.append("call/one", "user", "What time is it?")
            store.append(
                "call/one",
                "assistant",
                "It is noon.",
                interrupted=True,
            )
            store.close()

            with mock.patch.object(module, "HISTORY_DB", db_path):
                transcript = asyncio.run(module.download_transcript("call/one"))

            self.assertIn("You:", transcript.body)
            self.assertIn("Assistant [interrupted]:", transcript.body)
            self.assertIn("What time is it?", transcript.body)
            self.assertIn("It is noon.", transcript.body)
            self.assertRegex(
                transcript.body,
                r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z \| You:",
            )
            self.assertEqual(
                transcript.headers["Content-Disposition"],
                'attachment; filename="conversation-call_one.txt"',
            )

    def test_voice_app_transcript_download_returns_not_found(self):
        module = MODULES["web_app"]
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                module,
                "HISTORY_DB",
                str(Path(temp_dir) / "history.sqlite"),
            ):
                with self.assertRaises(module.HTTPException) as raised:
                    asyncio.run(module.download_transcript("missing"))
        self.assertEqual(raised.exception.status_code, 404)

    def test_assemblyai_key_is_loaded_from_infisical(self):
        module = MODULES["infisical_secrets"]
        calls = []

        class FakeClient:
            def __init__(self, *, host):
                calls.append(("host", host))
                self.auth = types.SimpleNamespace(
                    universal_auth=types.SimpleNamespace(
                        login=lambda **kwargs: calls.append(("login", kwargs))
                    )
                )
                self.secrets = types.SimpleNamespace(
                    get_secret_by_name=lambda **kwargs: (
                        calls.append(("get_secret", kwargs))
                        or types.SimpleNamespace(secretValue="infisical-key")
                    )
                )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "INFISICAL_CLIENT_ID": "client-id",
                    "INFISICAL_CLIENT_SECRET": "client-secret",
                    "INFISICAL_PROJECT_ID": "project-id",
                    "INFISICAL_ENVIRONMENT_SLUG": "dev",
                    "INFISICAL_SECRET_PATH": "/",
                },
                clear=True,
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "infisical_sdk": types.SimpleNamespace(
                        InfisicalSDKClient=FakeClient
                    )
                },
            ),
        ):
            self.assertEqual(
                module.get_assemblyai_api_key(),
                "infisical-key",
            )

        self.assertIn(
            (
                "get_secret",
                {
                    "secret_name": "ASSEMBLYAI_API_KEY",
                    "project_id": "project-id",
                    "environment_slug": "dev",
                    "secret_path": "/",
                },
            ),
            calls,
        )

    def test_assemblyai_key_rejects_partial_infisical_configuration(self):
        module = MODULES["infisical_secrets"]
        with mock.patch.dict(
            os.environ,
            {"INFISICAL_CLIENT_ID": "client-id"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "INFISICAL_CLIENT_SECRET"):
                module.get_assemblyai_api_key()

    def test_voice_socket_requires_both_ui_api_keys(self):
        module = MODULES["web_app"]
        websocket = _FakeConversationWebSocket(
            module,
            [
                {
                    "type": "configure",
                    "assemblyAiApiKey": "",
                    "openAiApiKey": "",
                }
            ],
        )
        client_factory = mock.patch.object(module, "get_openai_client")
        with client_factory as create_client:
            asyncio.run(module.voice_socket(websocket))

        errors = [
            item[1]
            for item in websocket.sent
            if item[0] == "json" and item[1].get("type") == "error"
        ]
        self.assertEqual(errors[0]["code"], "invalid_credentials")
        create_client.assert_not_called()

    def test_openai_key_validation_checks_model_access(self):
        module = MODULES["web_app"]
        client = types.SimpleNamespace(
            models=types.SimpleNamespace(
                list=mock.AsyncMock(
                    return_value=types.SimpleNamespace(
                        data=[types.SimpleNamespace(id=module.MODEL)]
                    )
                )
            )
        )
        asyncio.run(module.validate_openai_api_key(client))
        client.models.list.assert_awaited_once_with()

        client.models.list = mock.AsyncMock(
            return_value=types.SimpleNamespace(
                data=[types.SimpleNamespace(id="some-other-model")]
            )
        )
        with self.assertRaises(module.CredentialValidationError) as raised:
            asyncio.run(module.validate_openai_api_key(client))
        self.assertEqual(raised.exception.code, "openai_model_unavailable")

    def test_assemblyai_key_validation_checks_auth_without_transcribing(self):
        module = MODULES["web_app"]
        requests = []

        class FakeAPIClient:
            def __init__(self, *, api_key):
                requests.append(("client", api_key))
                self.http_client = types.SimpleNamespace(
                    get=lambda *args, **kwargs: (
                        requests.append(("get", args, kwargs))
                        or types.SimpleNamespace(status_code=200)
                    )
                )

            def close(self):
                requests.append(("close",))

        with mock.patch.object(module.aai, "Client", FakeAPIClient):
            module.validate_assemblyai_api_key("assemblyai-test-key")

        self.assertEqual(requests[0], ("client", "assemblyai-test-key"))
        self.assertEqual(requests[-1], ("close",))
        self.assertIn(
            (
                "get",
                ("/v2/transcript",),
                {"params": {"limit": 1}, "timeout": 5},
            ),
            requests,
        )

    def test_openai_stream_to_tts_produces_sentences_and_audio(self):
        module = MODULES["openai_demo"]
        module.openai_client = _FakeOpenAI(
            [_event("Hello"), _event(" there. "), _event("Next?")]
        )
        session = _FakeAsyncTTSSession()
        pcm = bytearray()

        asyncio.run(module.stream_reply(session, pcm, asyncio.Lock()))

        self.assertEqual(session.generated, ["Hello there. ", "Next?"])
        self.assertEqual(pcm, b"\x01\x00\x02\x00")

    def test_websocket_agent_streams_sentence_audio_and_bounds_chunks(self):
        module = MODULES["web_app"]
        chunks, remainder = module.split_speech_chunks(
            "First sentence. " + ("long " * 80),
            max_chars=24,
        )
        self.assertEqual(chunks[0], "First sentence.")
        self.assertTrue(all(len(chunk) <= 24 for chunk in chunks))
        self.assertLessEqual(len(remainder), 24)

        websocket = _FakeWebSocket()
        session = _FakeAsyncTTSSession()
        client = _FakeOpenAI(
            [_event("First sentence. "), _event("Second sentence.")]
        )
        messages = [{"role": "user", "content": "Say hello."}]

        reply, interrupted = asyncio.run(
            module.stream_turn(
                websocket,
                session,
                client,
                messages,
                "eval-message",
                asyncio.Event(),
                websocket.send_json,
                websocket.send_bytes,
            )
        )

        self.assertEqual(reply, "First sentence. Second sentence.")
        self.assertFalse(interrupted)
        self.assertEqual(
            session.generated,
            ["First sentence.", "Second sentence."],
        )
        json_events = [
            item[1]["type"]
            for item in websocket.sent
            if item[0] == "json"
        ]
        self.assertEqual(json_events.count("assistant_delta"), 2)
        self.assertIn("state", json_events)
        self.assertEqual(
            [item[1] for item in websocket.sent if item[0] == "bytes"],
            [b"\x01\x00", b"\x02\x00"],
        )

    def test_interrupt_cancels_active_tts_and_preserves_partial_text(self):
        module = MODULES["web_app"]
        session = _FakeAsyncTTSSession()
        websocket = _FakeWebSocket()

        class Completions:
            async def create(self, **_kwargs):
                return _BlockingOpenAIStream()

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=Completions())
        )
        stop = asyncio.Event()

        async def run_turn():
            return await module.stream_turn(
                websocket,
                session,
                client,
                [{"role": "user", "content": "test"}],
                "interrupt-eval",
                stop,
                websocket.send_json,
                websocket.send_bytes,
            )

        async def scenario():
            task = asyncio.create_task(run_turn())
            await asyncio.wait_for(websocket.delta_sent.wait(), timeout=1)
            task.cancel()
            return await asyncio.wait_for(task, timeout=1)

        reply, interrupted = asyncio.run(scenario())

        self.assertEqual(reply, "Partial reply")
        self.assertTrue(interrupted)
        self.assertEqual(session.cancel_count, 1)

    def test_websocket_keeps_one_tts_session_open_across_turns(self):
        module = MODULES["web_app"]
        session = _FakeAsyncTTSSession()
        session.open_count = 0
        session.close_count = 0
        streamed_audio = []
        transcriber_keys = []

        class PersistentSession(_FakeAsyncTTSSession):
            async def __aenter__(self):
                session.open_count += 1
                return session

            async def __aexit__(self, *_args):
                session.close_count += 1

        class FakeTranscriber:
            def __init__(self, **kwargs):
                transcriber_keys.append(kwargs.get("api_key"))
                self.handlers = {}

            def on(self, event, callback):
                self.handlers[event] = callback

            def connect(self, *_args):
                return None

            def stream(self, data):
                streamed_audio.append(data)

            def disconnect(self, **_kwargs):
                return None

        client = _FakeOpenAI([_event("Hello.")])
        websocket = _FakeConversationWebSocket(
            module,
            [
                {
                    "type": "configure",
                    "assemblyAiApiKey": "eval-assemblyai-key",
                    "openAiApiKey": "eval-openai-key",
                },
                {"type": "start", "conversationId": "eval-conversation"},
                b"\x00" * 1_600,
                {"type": "text", "text": "First turn"},
                {"type": "text", "text": "Second turn"},
            ],
        )
        fake_aai = types.SimpleNamespace(
            TTSConfig=lambda **kwargs: kwargs,
            AsyncTTSSession=lambda *_args, **_kwargs: PersistentSession(),
            TTSError=module.aai.TTSError,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "websocket.sqlite")
            openai_client_factory = mock.patch.object(
                module,
                "get_openai_client",
                return_value=client,
            )
            openai_validation = mock.patch.object(
                module,
                "validate_openai_api_key",
                new=mock.AsyncMock(),
            )
            assemblyai_validation = mock.patch.object(
                module,
                "validate_assemblyai_api_key",
            )
            with (
                mock.patch.object(module, "HISTORY_DB", db_path),
                mock.patch.object(module, "aai", fake_aai),
                mock.patch.object(
                    module,
                    "RealTimeTranscriber",
                    FakeTranscriber,
                ),
                openai_client_factory as create_client,
                openai_validation as validate_openai,
                assemblyai_validation as validate_assemblyai,
            ):
                asyncio.run(module.voice_socket(websocket))
            create_client.assert_called_once_with("eval-openai-key")
            validate_openai.assert_awaited_once_with(client)
            validate_assemblyai.assert_called_once_with("eval-assemblyai-key")

            store_module = MODULES["conversation_store"]
            history = store_module.ConversationStore(db_path)
            try:
                context = history.messages_for_prompt("eval-conversation")
            finally:
                history.close()

        self.assertEqual(session.open_count, 1)
        self.assertEqual(session.close_count, 1)
        self.assertTrue(client.closed)
        self.assertEqual(transcriber_keys, ["eval-assemblyai-key"])
        self.assertEqual(session.generated, ["Hello.", "Hello."])
        self.assertEqual(streamed_audio, [b"\x00" * 1_600])
        self.assertEqual(
            client.completions.calls[0]["messages"][-1]["role"], "user"
        )
        self.assertEqual(
            client.completions.calls[1]["messages"][-2],
            {"role": "assistant", "content": "Hello."},
        )
        self.assertEqual(
            [
                item[1]
                for item in websocket.sent
                if item[0] == "bytes"
            ],
            [b"\x01\x00", b"\x02\x00"],
        )
        self.assertEqual(context[-2:], [
            {"role": "user", "content": "Second turn"},
            {"role": "assistant", "content": "Hello."},
        ])

    def test_sync_demo_writes_valid_wav_outputs(self):
        module = MODULES["demo"]
        response = types.SimpleNamespace(
            audio=_wav_bytes(), audio_duration_ms=80
        )
        fake_aai = types.SimpleNamespace(
            TextToSpeech=lambda **_kwargs: types.SimpleNamespace(
                synthesize=lambda *_args: response
            ),
            TTSConfig=lambda **kwargs: kwargs,
            TTSContainer=types.SimpleNamespace(wav=object()),
            TTSSession=_FakeSyncTTSSession,
        )
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                with mock.patch.object(module, "aai", fake_aai):
                    module.generate_wav()
                    module.stream_agent_reply()

                with wave.open("hello.wav", "rb") as generated:
                    self.assertEqual(generated.getnchannels(), 1)
                    self.assertEqual(generated.getframerate(), 24_000)
                with wave.open("agent_reply.wav", "rb") as streamed:
                    self.assertEqual(streamed.getsampwidth(), 2)
                    self.assertEqual(streamed.getnframes(), 2)
            finally:
                os.chdir(old_cwd)

    def test_async_demo_streams_all_sentence_audio(self):
        module = MODULES["async_demo"]

        async def tokens():
            yield "First. "
            yield "Second?"

        module.simulated_llm_tokens = tokens
        session = _FakeAsyncTTSSession()
        pcm = bytearray()

        asyncio.run(module.speak_llm_reply(session, pcm))

        self.assertEqual(session.generated, ["First. ", "Second?"])
        self.assertEqual(pcm, b"\x01\x00\x02\x00")

    def test_barge_in_invalidates_audio_from_old_generation(self):
        speaker = MODULES["interruption_demo"].Speaker()
        generation = speaker.generation
        speaker.write(b"\x01\x00", generation)

        speaker.clear()
        speaker.write(b"\x02\x00", generation)

        self.assertEqual(speaker._buffer, bytearray())

    def test_realtime_stt_callbacks_emit_partial_and_final_text(self):
        module = MODULES["web_app"]
        received = []

        async def record(kind, text):
            received.append((kind, text))

        async def scenario():
            complete = asyncio.Event()

            async def on_partial(text):
                await record("partial", text)
                if len(received) == 2:
                    complete.set()

            async def on_final(text):
                await record("final", text)
                if len(received) == 2:
                    complete.set()

            async def unused(*_args):
                return None

            microphone = module.MicrophoneSTT(
                "eval-key",
                asyncio.get_running_loop(),
                on_partial,
                on_final,
                unused,
                unused,
            )
            await microphone.connect()
            microphone._turn(
                microphone.client,
                types.SimpleNamespace(
                    transcript="hello there",
                    end_of_turn=False,
                ),
            )
            microphone._turn(
                microphone.client,
                types.SimpleNamespace(
                    transcript="hello there",
                    end_of_turn=True,
                ),
            )
            await asyncio.wait_for(complete.wait(), timeout=1)
            await microphone.close()

        asyncio.run(scenario())
        self.assertEqual(
            received,
            [("partial", "hello there"), ("final", "hello there")],
        )

    def test_voice_app_summary_persists_and_emits_protocol_event(self):
        module = MODULES["web_app"]

        class FakeGateway:
            def __init__(self, **_kwargs):
                self.chat_completions = self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def create(self, **_kwargs):
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content="The user prefers concise replies."
                            )
                        )
                    ],
                    model="offline-summary-model",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "voice-summary.sqlite")
            store = MODULES["conversation_store"].ConversationStore(db_path)
            store.create("summary-eval")
            store.append("summary-eval", "user", "Please be concise.")
            store.append("summary-eval", "assistant", "I will.")
            events = []

            async def send_json(payload):
                events.append(payload)

            try:
                with (
                    mock.patch.object(module, "SUMMARY_TRIGGER_MESSAGES", 2),
                    mock.patch.object(module, "RECENT_CONTEXT_MESSAGES", 1),
                    mock.patch.object(
                        module.aai,
                        "AsyncLLMGateway",
                        FakeGateway,
                    ),
                ):
                    asyncio.run(
                        module.send_summary(
                            store,
                            "summary-eval",
                            "eval-key",
                            send_json,
                        )
                    )

                self.assertEqual(
                    events,
                    [
                        {
                            "type": "summary_updated",
                            "version": 1,
                            "text": "The user prefers concise replies.",
                        }
                    ],
                )
                self.assertEqual(
                    store.summary_state("summary-eval")[1],
                    "The user prefers concise replies.",
                )
            finally:
                store.close()

    def test_gateway_summary_update_records_version_and_usage(self):
        module = MODULES["summary_audit_voice_history"]

        class FakeGateway:
            def __init__(self):
                self.requests = []
                message = types.SimpleNamespace(
                    content="The user prefers concise replies."
                )
                self.result = types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)],
                    model="offline-summary-model",
                    usage=types.SimpleNamespace(
                        input_tokens=31,
                        output_tokens=9,
                        total_tokens=40,
                    ),
                )
                self.chat_completions = self

            async def create(self, **kwargs):
                self.requests.append(kwargs)
                return self.result

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "summary.sqlite")
            with (
                mock.patch.object(module, "CONVERSATION_ID", "gateway-eval"),
                mock.patch.object(module, "SUMMARY_TRIGGER_TOKENS", 1),
                mock.patch.object(module, "RECENT_CONTEXT_TOKENS", 1),
                mock.patch.object(module, "MODEL_REGION", None),
            ):
                history = module.HistoryStore(db_path)
                gateway = FakeGateway()
                try:
                    history.append("user", "Please keep replies concise.")
                    history.append("assistant", "Understood.")

                    updated = asyncio.run(
                        module.update_summary(history, gateway)
                    )

                    self.assertTrue(updated)
                    self.assertEqual(len(gateway.requests), 1)
                    self.assertEqual(
                        gateway.requests[0]["model"],
                        module.SUMMARY_MODEL,
                    )
                    summary_state = history.summary_state()
                    self.assertEqual(
                        summary_state[0],
                        "The user prefers concise replies.",
                    )
                    self.assertEqual(summary_state[3], 1)
                    saved = history.get_summary_version(1)
                    self.assertEqual(saved[2], "offline-summary-model")
                    self.assertEqual(saved[7:10], (31, 9, 40))
                finally:
                    history.close()

    def test_summary_history_versions_commit_and_reject_stale_checkpoint(self):
        module = MODULES["summary_audit_voice_history"]
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "history.sqlite")
            with mock.patch.object(module, "CONVERSATION_ID", "eval-conversation"):
                history = module.HistoryStore(db_path)
                try:
                    history.append("user", "Remember this preference.")
                    history.append("assistant", "I will remember it.")
                    batch = history.summary_batch(1)
                    through_id = batch[-1][0]

                    version = history.save_summary_version(
                        summary="The user prefers concise replies.",
                        model="offline-eval-model",
                        from_message_id=batch[0][0],
                        through_message_id=through_id,
                        batch_tokens=sum(row[4] for row in batch),
                        cumulative_tokens=sum(row[4] for row in batch),
                        input_tokens=12,
                        output_tokens=7,
                        total_tokens=19,
                    )

                    self.assertEqual(version, 1)
                    self.assertEqual(len(history.list_summary_versions()), 1)
                    saved = history.get_summary_version(1)
                    self.assertEqual(saved[1], "The user prefers concise replies.")
                    self.assertEqual(history.prompt_messages()[0]["role"], "system")

                    stale_version = history.save_summary_version(
                        summary="Stale summary.",
                        model="offline-eval-model",
                        from_message_id=1,
                        through_message_id=through_id,
                        batch_tokens=1,
                        cumulative_tokens=1,
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                    )
                    self.assertEqual(stale_version, 1)
                    self.assertEqual(len(history.list_summary_versions()), 1)
                finally:
                    history.close()

    def test_summary_store_migrates_legacy_database(self):
        module = MODULES["summary_audit_voice_history"]
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "legacy.sqlite")
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    through_message_id INTEGER NOT NULL DEFAULT 0,
                    source_token_estimate INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO conversations VALUES (
                    'legacy-eval', 'now', 'now'
                );
                INSERT INTO messages(
                    conversation_id, role, content, created_at
                ) VALUES ('legacy-eval', 'user', 'hello', 'now');
                INSERT INTO conversation_summaries VALUES (
                    'legacy-eval', 'older summary', 1, 25, 'now'
                );
                """
            )
            connection.close()

            with mock.patch.object(module, "CONVERSATION_ID", "legacy-eval"):
                history = module.HistoryStore(db_path)
                try:
                    self.assertEqual(history.summary_state()[0], "older summary")
                    legacy_version = history.get_summary_version(1)
                    self.assertIsNotNone(legacy_version)
                    self.assertEqual(legacy_version[2], "legacy-migration")
                    token_estimate = history.db.execute(
                        "SELECT token_estimate FROM messages"
                    ).fetchone()[0]
                    self.assertGreater(token_estimate, 0)
                    self.assertEqual(
                        history.unsummarized_token_count(), 0
                    )
                finally:
                    history.close()


if __name__ == "__main__":
    unittest.main()
