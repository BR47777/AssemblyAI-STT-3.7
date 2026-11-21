import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

import assemblyai as aai
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from openai import (
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from assemblyai.streaming.v3 import (
    RealTimeError,
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriber,
    SpeechStartedEvent,
    TurnEvent,
)
from conversation_store import ConversationStore

logger = logging.getLogger(__name__)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "claude-sonnet-4-6")
SUMMARY_TRIGGER_MESSAGES = int(os.getenv("SUMMARY_TRIGGER_MESSAGES", "30"))
RECENT_CONTEXT_MESSAGES = 20
MIC_SAMPLE_RATE = 16_000
MIC_FRAME_BYTES = 1_600
MAX_MESSAGE_CHARS = 4_000
MAX_API_KEY_CHARS = 4_096
MAX_SPEECH_CHARS = 240
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "marin")
TTS_INSTRUCTIONS = (
    "Speak naturally and clearly, using the language and pronunciation "
    "appropriate for the text."
)
OPENAI_RATE_LIMIT_RETRIES = 2
MAX_OPENAI_RETRY_DELAY = 5
SENTENCE_PATTERN = re.compile(
    r".+?[.!?](?:[\"')\]]+)?(?:\s+|$)",
    re.DOTALL,
)
INDEX_PATH = Path(__file__).parent / "static" / "index.html"
HISTORY_DB = os.getenv("HISTORY_DB", "agent_ui_history.db")
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a concise voice assistant. Reply in the same language as the "
        "user's latest message, switching languages whenever the user does. "
        "Use fluent, idiomatic grammar and natural, complete sentences. Avoid "
        "Markdown unless the user asks for it."
    ),
}

app = FastAPI(title="AssemblyAI Voice Agent")


def get_openai_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key)


class CredentialValidationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def validate_openai_api_key(client: AsyncOpenAI) -> None:
    try:
        models = await client.models.list()
    except AuthenticationError as error:
        raise CredentialValidationError(
            "openai_invalid_key",
            "OpenAI rejected this key. Check that it is active and copied correctly.",
        ) from error
    except PermissionDeniedError as error:
        raise CredentialValidationError(
            "openai_access_denied",
            "This OpenAI key is not permitted to list models.",
        ) from error
    except RateLimitError as error:
        raise CredentialValidationError(
            "openai_rate_limited",
            "OpenAI rate-limited key validation. Check account limits and retry.",
        ) from error
    except Exception as error:
        raise CredentialValidationError(
            "openai_validation_failed",
            "Could not verify the OpenAI key. Check network access and retry.",
        ) from error

    if not any(model.id == MODEL for model in models.data):
        raise CredentialValidationError(
            "openai_model_unavailable",
            f"The configured OpenAI model {MODEL!r} is not available to this key.",
        )


def validate_assemblyai_api_key(api_key: str) -> None:
    client = None
    try:
        client = aai.Client(api_key=api_key)
        response = client.http_client.get(
            "/v2/transcript",
            params={"limit": 1},
            timeout=5,
        )
        if response.status_code == 401:
            raise CredentialValidationError(
                "assemblyai_invalid_key",
                "AssemblyAI rejected this key. Check that it is active and copied correctly.",
            )
        if response.status_code == 403:
            raise CredentialValidationError(
                "assemblyai_access_denied",
                "This AssemblyAI key does not have permission to list transcripts.",
            )
        if response.status_code == 429:
            raise CredentialValidationError(
                "assemblyai_rate_limited",
                "AssemblyAI rate-limited key validation. Wait a moment and retry.",
            )
        if not 200 <= response.status_code < 300:
            raise CredentialValidationError(
                "assemblyai_validation_failed",
                "Could not verify the AssemblyAI key. Check its permissions and retry.",
            )
    except CredentialValidationError:
        raise
    except Exception as error:
        raise CredentialValidationError(
            "assemblyai_validation_failed",
            "Could not reach AssemblyAI to verify the key. Check network access and retry.",
        ) from error
    finally:
        if client is not None:
            client.close()


def openai_rate_limit_code(error: RateLimitError) -> str | None:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None
    details = body.get("error", body)
    if not isinstance(details, dict):
        return None
    code = details.get("code")
    return code if isinstance(code, str) else None


def openai_retry_delay(error: RateLimitError, attempt: int) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    retry_after = headers.get("retry-after") if headers else None
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0), MAX_OPENAI_RETRY_DELAY)
        except (TypeError, ValueError):
            pass
    return min(0.5 * (2 ** attempt), MAX_OPENAI_RETRY_DELAY)


def split_speech_chunks(
    text: str,
    max_chars: int = MAX_SPEECH_CHARS,
) -> tuple[list[str], str]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive.")

    matches = list(SENTENCE_PATTERN.finditer(text))
    complete = [match.group(0).strip() for match in matches]
    remainder = text[matches[-1].end():] if matches else text

    chunks: list[str] = []
    for sentence in complete:
        while len(sentence) > max_chars:
            boundary = sentence.rfind(" ", 0, max_chars + 1)
            if boundary <= 0:
                boundary = max_chars
            chunks.append(sentence[:boundary].strip())
            sentence = sentence[boundary:].strip()
        if sentence:
            chunks.append(sentence)

    while len(remainder) > max_chars:
        boundary = remainder.rfind(" ", 0, max_chars + 1)
        if boundary <= 0:
            boundary = max_chars
        chunk = remainder[:boundary].strip()
        if chunk:
            chunks.append(chunk)
        remainder = remainder[boundary:].lstrip()

    return chunks, remainder


async def send_summary(
    store: ConversationStore,
    conversation_id: str,
    api_key: str,
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
) -> None:
    if store.unsummarized_count(conversation_id) < SUMMARY_TRIGGER_MESSAGES:
        return
    batch = store.summary_batch(conversation_id, RECENT_CONTEXT_MESSAGES)
    if not batch:
        return

    _, previous_summary, _ = store.summary_state(conversation_id)
    transcript = "\n".join(
        f"{role.upper()}{' [interrupted]' if interrupted else ''}: {content}"
        for _, role, content, interrupted in batch
    )
    request: dict[str, Any] = {
        "model": SUMMARY_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Maintain durable memory for a voice conversation. Merge "
                    "the previous summary and new turns. Preserve names, "
                    "preferences, confirmed facts, decisions, promises, "
                    "completed tasks, and unresolved questions. Distinguish "
                    "facts from uncertainty. Note interrupted replies. Do not "
                    "invent information. Return concise plain text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous summary:\n{previous_summary or '(none)'}\n\n"
                    f"New turns:\n{transcript}"
                ),
            },
        ],
    }
    region = os.getenv("LLM_GATEWAY_MODEL_REGION")
    if region:
        if region != "global":
            raise ValueError("LLM_GATEWAY_MODEL_REGION must be 'global'.")
        request["model_region"] = "global"

    async with aai.AsyncLLMGateway(api_key=api_key) as gateway:
        result = await gateway.chat_completions.create(**request)
    summary = (result.choices[0].message.content or "").strip()
    if not summary:
        return

    version = store.save_summary(
        conversation_id,
        summary,
        batch[-1][0],
        getattr(result, "model", None) or SUMMARY_MODEL,
    )
    await send_json(
        {"type": "summary_updated", "version": version, "text": summary}
    )


class MicrophoneSTT:
    def __init__(
        self,
        api_key: str,
        loop: asyncio.AbstractEventLoop,
        on_partial: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
        on_speech_started: Callable[[], Awaitable[None]],
        on_error: Callable[[str], Awaitable[None]],
    ) -> None:
        self.loop = loop
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_speech_started = on_speech_started
        self.on_error = on_error
        self.client = RealTimeTranscriber(api_key=api_key)
        self.client.on(RealTimeEvents.SpeechStarted, self._speech_started)
        self.client.on(RealTimeEvents.Turn, self._turn)
        self.client.on(RealTimeEvents.Error, self._error)
        self.closed = False

    async def connect(self) -> None:
        await asyncio.to_thread(
            self.client.connect,
            RealTimeParameters(
                speech_model="universal-3-6-pro",
                sample_rate=MIC_SAMPLE_RATE,
                mode="balanced",
                interruption_delay=200,
                language_codes=["en"],
            ),
        )

    async def stream(self, audio: bytes) -> None:
        await asyncio.to_thread(self.client.stream, audio)

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            await asyncio.to_thread(self.client.disconnect, terminate=True)

    def _dispatch(
        self,
        callback: Callable[..., Awaitable[None]],
        *args: Any,
    ) -> None:
        if self.closed or self.loop.is_closed():
            return

        def schedule() -> None:
            task = asyncio.create_task(callback(*args))
            task.add_done_callback(self._report_callback_error)

        self.loop.call_soon_threadsafe(schedule)

    @staticmethod
    def _report_callback_error(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Realtime STT callback failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _speech_started(
        self,
        _client: RealTimeTranscriber,
        _event: SpeechStartedEvent,
    ) -> None:
        self._dispatch(self.on_speech_started)

    def _turn(self, _client: RealTimeTranscriber, event: TurnEvent) -> None:
        text = (getattr(event, "transcript", None) or "").strip()
        if not text:
            return
        if getattr(event, "end_of_turn", False):
            self._dispatch(self.on_final, text)
        else:
            self._dispatch(self.on_partial, text)

    def _error(self, _client: RealTimeTranscriber, _error: RealTimeError) -> None:
        self._dispatch(self.on_error, "Realtime transcription failed.")


async def stream_turn(
    websocket: WebSocket,
    client: AsyncOpenAI,
    messages: list[dict[str, str]],
    message_id: str,
    stop: asyncio.Event,
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
    send_bytes: Callable[[bytes], Awaitable[None]],
) -> tuple[str, bool]:
    speech_queue: asyncio.Queue[str | None] = asyncio.Queue()
    stream = None
    reply = ""
    pending = ""
    interrupted = False
    speaking = False

    async def send_audio() -> None:
        nonlocal speaking
        while True:
            speech_text = await speech_queue.get()
            if speech_text is None:
                return
            async with client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=speech_text,
                instructions=TTS_INSTRUCTIONS,
                response_format="pcm",
            ) as response:
                pending_audio = b""
                async for chunk in response.iter_bytes(chunk_size=4_096):
                    if stop.is_set():
                        return
                    audio = pending_audio + chunk
                    complete_length = len(audio) - (len(audio) % 2)
                    if complete_length:
                        if not speaking:
                            speaking = True
                            await send_json(
                                {"type": "state", "state": "speaking"}
                            )
                        await send_bytes(audio[:complete_length])
                    pending_audio = audio[complete_length:]
                if pending_audio:
                    raise ValueError("TTS returned an incomplete PCM sample.")

    audio_task = asyncio.create_task(send_audio())
    try:
        for attempt in range(OPENAI_RATE_LIMIT_RETRIES + 1):
            try:
                stream = await client.chat.completions.create(
                    model=MODEL,
                    stream=True,
                    messages=messages,
                )
                break
            except RateLimitError as error:
                if (
                    openai_rate_limit_code(error) == "insufficient_quota"
                    or attempt == OPENAI_RATE_LIMIT_RETRIES
                ):
                    raise
                await asyncio.sleep(openai_retry_delay(error, attempt))

        async for event in stream:
            if stop.is_set():
                interrupted = True
                break

            token = event.choices[0].delta.content or ""
            if not token:
                continue
            reply += token
            pending += token
            await send_json(
                {
                    "type": "assistant_delta",
                    "messageId": message_id,
                    "text": token,
                }
            )

            chunks, pending = split_speech_chunks(pending)
            for chunk in chunks:
                if stop.is_set():
                    interrupted = True
                    break
                if audio_task.done():
                    await audio_task
                await speech_queue.put(chunk)

        if pending.strip() and not stop.is_set():
            if audio_task.done():
                await audio_task
            await speech_queue.put(pending.strip())
    except asyncio.CancelledError:
        interrupted = True
        stop.set()
        audio_task.cancel()
    finally:
        if stream is not None:
            await stream.close()
        if not audio_task.done():
            if stop.is_set():
                audio_task.cancel()
            else:
                await speech_queue.put(None)
        try:
            await audio_task
        except asyncio.CancelledError:
            if not stop.is_set():
                raise

    return reply.strip(), interrupted or stop.is_set()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/pcm-worklet.js", response_class=FileResponse)
async def pcm_worklet() -> FileResponse:
    return FileResponse(INDEX_PATH.with_name("pcm-worklet.js"))


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/conversations")
async def list_conversations() -> dict[str, list[dict[str, str | int]]]:
    store = ConversationStore(HISTORY_DB)
    try:
        return {"conversations": store.list_conversations()}
    finally:
        store.close()


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
) -> dict[str, str | list[dict[str, str | int | bool]]]:
    store = ConversationStore(HISTORY_DB)
    try:
        if not store.exists(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return {
            "id": conversation_id,
            "messages": store.conversation_messages(conversation_id),
        }
    finally:
        store.close()


@app.get("/api/conversations/{conversation_id}/transcript.txt")
async def download_transcript(conversation_id: str) -> PlainTextResponse:
    store = ConversationStore(HISTORY_DB)
    try:
        if not store.exists(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
        messages = store.conversation_messages(conversation_id)
        lines = [
            "Conversation transcript",
            f"Conversation ID: {conversation_id}",
            "Times are in UTC.",
            "",
        ]
        for message in messages:
            timestamp = str(message["created_at"])
            if timestamp.endswith("+00:00"):
                timestamp = f"{timestamp[:-6]}Z"
            speaker = "You" if message["role"] == "user" else "Assistant"
            interrupted = " [interrupted]" if message["interrupted"] else ""
            lines.extend(
                [
                    f"{timestamp} | {speaker}{interrupted}:",
                    str(message["content"]),
                    "",
                ]
            )
        filename_id = re.sub(r"[^A-Za-z0-9_-]", "_", conversation_id)
        return PlainTextResponse(
            "\n".join(lines),
            headers={
                "Content-Disposition": (
                    f'attachment; filename="conversation-{filename_id}.txt"'
                )
            },
        )
    finally:
        store.close()


@app.websocket("/ws")
async def voice_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    client: AsyncOpenAI | None = None
    while client is None:
        try:
            packet = await websocket.receive()
        except WebSocketDisconnect:
            return

        if packet["type"] == "websocket.disconnect":
            return
        raw_credentials = packet.get("text")
        try:
            credentials = json.loads(raw_credentials or "")
        except json.JSONDecodeError:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "credentials_required",
                    "message": "Submit both API keys to connect.",
                }
            )
            continue

        if (
            not isinstance(credentials, dict)
            or credentials.get("type") != "configure"
        ):
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "credentials_required",
                    "message": "Submit both API keys to connect.",
                }
            )
            continue

        aai_api_key = credentials.get("assemblyAiApiKey")
        openai_api_key = credentials.get("openAiApiKey")
        if (
            not isinstance(aai_api_key, str)
            or not aai_api_key.strip()
            or len(aai_api_key) > MAX_API_KEY_CHARS
            or not isinstance(openai_api_key, str)
            or not openai_api_key.strip()
            or len(openai_api_key) > MAX_API_KEY_CHARS
        ):
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "invalid_credentials",
                    "message": "Enter valid AssemblyAI and OpenAI API keys.",
                }
            )
            continue

        candidate_client = get_openai_client(openai_api_key)
        try:
            await validate_openai_api_key(candidate_client)
            await asyncio.to_thread(validate_assemblyai_api_key, aai_api_key)
        except CredentialValidationError as error:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": error.code,
                    "message": error.message,
                }
            )
            await candidate_client.close()
            continue
        client = candidate_client

    await websocket.send_json({"type": "ready"})

    send_lock = asyncio.Lock()

    async def send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_bytes(payload: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(payload)

    store = ConversationStore(HISTORY_DB)
    loop = asyncio.get_running_loop()
    turn_queue: asyncio.Queue[tuple[str, int, str] | None] = asyncio.Queue()

    conversation_id: str | None = None
    turn_number = 0
    microphone: MicrophoneSTT | None = None
    active_stop: asyncio.Event | None = None
    active_task: asyncio.Task[tuple[str, bool]] | None = None
    active_turn_done: asyncio.Event | None = None

    async def send_error(code: str, message: str) -> None:
        await send_json({"type": "error", "code": code, "message": message})

    async def cancel_active() -> None:
        task = active_task
        stop = active_stop
        done = active_turn_done
        if task is not None and not task.done():
            if stop is not None:
                stop.set()
            task.cancel()
        if done is not None and not done.is_set():
            await done.wait()

    async def close_microphone() -> None:
        nonlocal microphone
        if microphone is not None:
            await microphone.close()
            microphone = None

    async def accept_final(text: str, expected_id: str) -> None:
        nonlocal turn_number
        if expected_id != conversation_id or not text.strip():
            return
        turn_number += 1
        clean_text = text.strip()
        store.append(expected_id, "user", clean_text)
        await send_json(
            {
                "type": "user_final",
                "turn": turn_number,
                "text": clean_text,
            }
        )
        await turn_queue.put((expected_id, turn_number, clean_text))

    async def on_speech_started(expected_id: str) -> None:
        if (
            expected_id == conversation_id
            and active_task is not None
            and not active_task.done()
        ):
            await send_json({"type": "state", "state": "interrupted"})
            await cancel_active()

    async def start_conversation(
        selected_id: str,
        create_if_missing: bool,
    ) -> None:
        nonlocal conversation_id, turn_number, microphone
        if len(selected_id) > 128:
            await send_error("invalid_conversation_id", "Conversation ID is too long.")
            return
        if not create_if_missing and not store.exists(selected_id):
            await send_error("conversation_not_found", "Conversation was not found.")
            return

        await cancel_active()
        await close_microphone()
        if create_if_missing:
            store.create(selected_id)
        conversation_id = selected_id
        turn_number = 0

        async def on_partial(text: str) -> None:
            if conversation_id == selected_id:
                await send_json(
                    {
                        "type": "user_partial",
                        "turn": turn_number + 1,
                        "text": text,
                    }
                )

        async def on_final(text: str) -> None:
            await accept_final(text, selected_id)

        async def on_error(message: str) -> None:
            if conversation_id == selected_id:
                await send_error("stt_error", message)

        async def on_speech() -> None:
            await on_speech_started(selected_id)

        microphone = MicrophoneSTT(
            aai_api_key,
            loop,
            on_partial,
            on_final,
            on_speech,
            on_error,
        )
        try:
            await microphone.connect()
        except Exception as error:
            logger.error(
                "Unable to start realtime transcription (%s)",
                type(error).__name__,
            )
            await close_microphone()
            conversation_id = None
            await send_error("stt_connect_failed", "Unable to start transcription.")
            return

        await send_json({"type": "state", "state": "listening"})
        version, summary, _ = store.summary_state(selected_id)
        if summary:
            await send_json(
                {
                    "type": "summary_updated",
                    "version": version,
                    "text": summary,
                }
            )

    async def process_turns() -> None:
        nonlocal active_stop, active_task, active_turn_done
        while True:
            item = await turn_queue.get()
            if item is None:
                return
            selected_id, _, _ = item
            if selected_id != conversation_id:
                continue

            message_id = str(uuid.uuid4())
            messages = [
                SYSTEM_MESSAGE,
                *store.messages_for_prompt(
                    selected_id,
                    RECENT_CONTEXT_MESSAGES,
                ),
            ]
            stop = asyncio.Event()
            turn_done = asyncio.Event()
            active_stop = stop
            active_turn_done = turn_done
            await send_json({"type": "state", "state": "thinking"})
            task = asyncio.create_task(
                stream_turn(
                    websocket,
                    client,
                    messages,
                    message_id,
                    stop,
                    send_json,
                    send_bytes,
                )
            )
            active_task = task
            try:
                reply, interrupted = await task
                if reply:
                    store.append(
                        selected_id,
                        "assistant",
                        reply,
                        interrupted=interrupted,
                    )
                await send_json(
                    {
                        "type": "assistant_final",
                        "messageId": message_id,
                        "text": reply,
                        "interrupted": interrupted,
                    }
                )
                if interrupted:
                    await send_json(
                        {"type": "state", "state": "interrupted"}
                    )
                if selected_id == conversation_id and microphone is not None:
                    await send_json(
                        {"type": "state", "state": "listening"}
                    )
                try:
                    await send_summary(
                        store,
                        selected_id,
                        aai_api_key,
                        send_json,
                    )
                except Exception as error:
                    logger.error(
                        "Conversation summary update failed (%s)",
                        type(error).__name__,
                    )
                    await send_error(
                        "summary_failed",
                        "Conversation summary could not be updated.",
                    )
            except AuthenticationError:
                await send_error(
                    "openai_invalid_key",
                    "OpenAI rejected this key during response generation. "
                    "Forget the saved keys and reconnect.",
                )
                if selected_id == conversation_id and microphone is not None:
                    await send_json(
                        {"type": "state", "state": "listening"}
                    )
            except PermissionDeniedError:
                await send_error(
                    "openai_access_denied",
                    "This OpenAI key cannot access the configured model.",
                )
                if selected_id == conversation_id and microphone is not None:
                    await send_json(
                        {"type": "state", "state": "listening"}
                    )
            except RateLimitError as error:
                quota_exhausted = (
                    openai_rate_limit_code(error) == "insufficient_quota"
                )
                logger.warning(
                    "OpenAI response rate-limited (%s)",
                    type(error).__name__,
                )
                await send_error(
                    "openai_rate_limited",
                    (
                        "OpenAI API quota or billing limit is exhausted. "
                        "Check usage and billing."
                        if quota_exhausted
                        else "OpenAI rate-limited this request. Wait and "
                        "retry if temporary, or check account limits."
                    ),
                )
                if selected_id == conversation_id and microphone is not None:
                    await send_json(
                        {"type": "state", "state": "listening"}
                    )
            except APIError as error:
                logger.error(
                    "OpenAI voice response failed (%s)",
                    type(error).__name__,
                )
                await send_error(
                    "openai_voice_failed",
                    "OpenAI could not generate the voice response. Check "
                    "audio API access and try again.",
                )
                if selected_id == conversation_id and microphone is not None:
                    await send_json(
                        {"type": "state", "state": "listening"}
                    )
            except Exception as error:
                logger.error("Voice turn failed (%s)", type(error).__name__)
                await send_error(
                    "voice_turn_failed",
                    "Voice response failed. Check server logs.",
                )
                if selected_id == conversation_id and microphone is not None:
                    await send_json(
                        {"type": "state", "state": "listening"}
                    )
            finally:
                active_task = None
                active_stop = None
                active_turn_done = None
                turn_done.set()

    worker_task: asyncio.Task[None] | None = None
    try:
        worker_task = asyncio.create_task(process_turns())
        while True:
            packet = await websocket.receive()
            if packet["type"] == "websocket.disconnect":
                break

            audio = packet.get("bytes")
            if audio is not None:
                if microphone is None:
                    await send_error("not_listening", "Send start before microphone audio.")
                elif len(audio) != MIC_FRAME_BYTES:
                    await send_error(
                        "invalid_audio_frame",
                        f"Expected exactly {MIC_FRAME_BYTES} bytes of PCM audio.",
                    )
                else:
                    try:
                        await microphone.stream(audio)
                    except Exception as error:
                        logger.error(
                            "Unable to stream microphone audio (%s)",
                            type(error).__name__,
                        )
                        await send_error(
                            "stt_stream_failed",
                            "Unable to process microphone audio.",
                        )
                continue

            raw_text = packet.get("text")
            if raw_text is None:
                continue
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                await send_error("invalid_json", "WebSocket message must be JSON.")
                continue
            if not isinstance(payload, dict):
                await send_error("invalid_message", "Expected a JSON object.")
                continue

            message_type = payload.get("type")
            if message_type == "start":
                requested_id = payload.get("conversationId")
                selected_id = (
                    requested_id.strip()
                    if isinstance(requested_id, str) and requested_id.strip()
                    else str(uuid.uuid4())
                )
                await start_conversation(selected_id, create_if_missing=True)
            elif message_type == "select_conversation":
                selected_id = payload.get("conversationId")
                if not isinstance(selected_id, str) or not selected_id.strip():
                    await send_error(
                        "invalid_conversation_id",
                        "A conversationId is required.",
                    )
                else:
                    await start_conversation(
                        selected_id.strip(),
                        create_if_missing=False,
                    )
            elif message_type == "stop":
                await cancel_active()
                await close_microphone()
                conversation_id = None
                await send_json(
                    {"type": "state", "state": "interrupted"}
                )
            elif message_type == "interrupt":
                if active_task is not None and not active_task.done():
                    await send_json(
                        {"type": "state", "state": "interrupted"}
                    )
                    await cancel_active()
            elif message_type == "text":
                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    await send_error("empty_text", "Enter a non-empty message.")
                elif len(text) > MAX_MESSAGE_CHARS:
                    await send_error(
                        "text_too_long",
                        f"Messages must be at most {MAX_MESSAGE_CHARS} characters.",
                    )
                elif conversation_id is None:
                    await send_error(
                        "conversation_not_started",
                        "Send start or select_conversation first.",
                    )
                else:
                    await accept_final(text, conversation_id)
            else:
                await send_error(
                    "unknown_message_type",
                    "Supported types: start, stop, interrupt, text, "
                    "select_conversation.",
                )
    except WebSocketDisconnect:
        logger.info("Voice UI disconnected")
    finally:
        await cancel_active()
        await close_microphone()
        if worker_task is not None:
            await turn_queue.put(None)
            await worker_task
        store.close()
        await client.close()
