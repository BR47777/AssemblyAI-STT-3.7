import asyncio
import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import suppress
from datetime import datetime, timezone

import assemblyai as aai
import sounddevice as sd
from assemblyai.streaming.v3 import (
    RealTimeError,
    RealTimeEvents,
    RealTimeParameters,
    RealTimeSessionParameters,
    RealTimeTranscriber,
    SpeechStartedEvent,
    TurnEvent,
)
from openai import AsyncOpenAI

API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DB_PATH = os.getenv("HISTORY_DB", "conversation_history.db")
CONVERSATION_ID = os.getenv("CONVERSATION_ID", str(uuid.uuid4()))
MAX_HISTORY_MESSAGES = 40
MIC_RATE, TTS_RATE = 16_000, 24_000

openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


class HistoryStore:
    def __init__(self, path: str) -> None:
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )
        self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS messages_conversation_idx
            ON messages(conversation_id, id)
            """
        )
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """
            INSERT OR IGNORE INTO conversations(id, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            (CONVERSATION_ID, now, now),
        )
        self.db.commit()

    def append(
        self,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        content = content.strip()
        if not content:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.execute(
                """
                INSERT INTO messages(
                    conversation_id, role, content, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    CONVERSATION_ID,
                    role,
                    content,
                    json.dumps(metadata or {}),
                    now,
                ),
            )
            self.db.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, CONVERSATION_ID),
            )

    def load_recent(self, limit: int) -> list[dict[str, str]]:
        rows = self.db.execute(
            """
            SELECT role, content, metadata
            FROM (
                SELECT id, role, content, metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (CONVERSATION_ID, limit),
        ).fetchall()

        messages = []
        for role, content, raw_metadata in rows:
            metadata = json.loads(raw_metadata)
            if role == "assistant" and metadata.get("interrupted"):
                content += " [The user interrupted this response.]"
            messages.append({"role": role, "content": content})
        return messages

    def latest_assistant_reply(self) -> str | None:
        row = self.db.execute(
            """
            SELECT content
            FROM messages
            WHERE conversation_id = ? AND role = 'assistant'
            ORDER BY id DESC
            LIMIT 1
            """,
            (CONVERSATION_ID,),
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.db.close()


class Speaker:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.generation = 0

    def write(self, chunk: bytes, generation: int) -> None:
        with self.lock:
            if generation == self.generation:
                self.buffer.extend(chunk)

    def clear(self) -> None:
        with self.lock:
            self.generation += 1
            self.buffer.clear()

    def callback(self, outdata, frames, _time, status) -> None:
        if status:
            print(f"\nSpeaker status: {status}")
        needed = frames * 2
        with self.lock:
            chunk = bytes(self.buffer[:needed])
            del self.buffer[:needed]
        outdata[:] = chunk + b"\0" * (needed - len(chunk))


class MicrophoneSTT:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.turns: asyncio.Queue[str] = asyncio.Queue()
        self.barge_in: asyncio.Event | None = None
        self.client = RealTimeTranscriber(api_key=API_KEY)

        self.client.on(RealTimeEvents.SpeechStarted, self._speech_started)
        self.client.on(RealTimeEvents.Turn, self._turn)
        self.client.on(RealTimeEvents.Error, self._error)
        self.client.connect(
            RealTimeParameters(
                speech_model="universal-3-6-pro",
                sample_rate=MIC_RATE,
                mode="balanced",
                interruption_delay=200,
                language_codes=["en"],
            )
        )

    def _speech_started(
        self,
        _client: RealTimeTranscriber,
        _event: SpeechStartedEvent,
    ) -> None:
        if self.barge_in is not None:
            self.loop.call_soon_threadsafe(self.barge_in.set)

    def _turn(
        self,
        _client: RealTimeTranscriber,
        event: TurnEvent,
    ) -> None:
        text = (event.transcript or "").strip()
        if event.end_of_turn and text:
            self.loop.call_soon_threadsafe(self.turns.put_nowait, text)

    def _error(
        self,
        _client: RealTimeTranscriber,
        error: RealTimeError,
    ) -> None:
        print(f"\nSTT error: {error}")

    def set_agent_context(self, reply: str) -> None:
        self.client.set_params(
            RealTimeSessionParameters(agent_context=reply[-1750:])
        )

    def microphone_callback(self, data, _frames, _time, status) -> None:
        if status:
            print(f"\nMicrophone status: {status}")
        self.client.stream(bytes(data))

    def close(self) -> None:
        self.client.disconnect(terminate=True)


def split_sentences(text: str) -> tuple[list[str], str]:
    pattern = re.compile(r".+?[.!?](?:[\"')\]]+)?(?:\s+|$)", re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        return [], text
    return [m.group(0) for m in matches], text[matches[-1].end():]


async def receive_audio(
    session: aai.AsyncTTSSession,
    flush_id: int,
    speaker: Speaker,
    generation: int,
    stop: asyncio.Event,
) -> None:
    async for chunk in session.audio(flush_id):
        if stop.is_set():
            return
        speaker.write(chunk, generation)


async def speak_turn(
    session: aai.AsyncTTSSession,
    speaker: Speaker,
    microphone: MicrophoneSTT,
    messages: list[dict[str, str]],
) -> tuple[str, bool]:
    stop = asyncio.Event()
    microphone.barge_in = stop
    generation = speaker.generation
    pending = ""
    reply = ""
    audio_tasks: list[asyncio.Task[None]] = []

    stream = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        stream=True,
    )

    try:
        async for event in stream:
            if stop.is_set():
                break

            token = event.choices[0].delta.content or ""
            if not token:
                continue

            reply += token
            pending += token
            sentences, pending = split_sentences(pending)

            for sentence in sentences:
                await session.generate(sentence)
                flush_id = await session.flush()
                audio_tasks.append(
                    asyncio.create_task(
                        receive_audio(
                            session, flush_id, speaker, generation, stop
                        )
                    )
                )

        if pending.strip() and not stop.is_set():
            await session.generate(pending)
            flush_id = await session.flush()
            audio_tasks.append(
                asyncio.create_task(
                    receive_audio(
                        session, flush_id, speaker, generation, stop
                    )
                )
            )

        if audio_tasks and not stop.is_set():
            await asyncio.gather(*audio_tasks)
    finally:
        microphone.barge_in = None
        await stream.close()

        if stop.is_set():
            speaker.clear()
            await session.cancel()
            for task in audio_tasks:
                task.cancel()
            await asyncio.gather(*audio_tasks, return_exceptions=True)

    return reply.strip(), stop.is_set()


async def conversation(
    session: aai.AsyncTTSSession,
    speaker: Speaker,
    microphone: MicrophoneSTT,
    history: HistoryStore,
) -> None:
    system_message = {
        "role": "system",
        "content": (
            "You are a concise voice assistant. Use short, natural "
            "sentences without Markdown. Ask at most one question at a time."
        ),
    }

    previous_reply = history.latest_assistant_reply()
    if previous_reply:
        microphone.set_agent_context(previous_reply)

    print(f"Conversation ID: {CONVERSATION_ID}")
    print("Listening. Say 'goodbye' to exit.")

    while True:
        user_text = await microphone.turns.get()
        print(f"\nUser: {user_text}")

        if user_text.casefold().strip(" .!?") in {
            "goodbye", "quit", "exit", "stop conversation"
        }:
            break

        history.append("user", user_text)
        messages = [
            system_message,
            *history.load_recent(MAX_HISTORY_MESSAGES),
        ]

        print("Assistant: ", end="", flush=True)
        reply, interrupted = await speak_turn(
            session, speaker, microphone, messages
        )

        if reply:
            print(reply)
            history.append(
                "assistant",
                reply,
                {"interrupted": interrupted},
            )
            microphone.set_agent_context(reply)

        if interrupted:
            print("Waiting for the completed interruption...")


async def main() -> None:
    loop = asyncio.get_running_loop()
    history = HistoryStore(DB_PATH)
    speaker = Speaker()
    microphone = MicrophoneSTT(loop)

    try:
        with (
            sd.RawInputStream(
                samplerate=MIC_RATE,
                channels=1,
                dtype="int16",
                blocksize=800,
                callback=microphone.microphone_callback,
            ),
            sd.RawOutputStream(
                samplerate=TTS_RATE,
                channels=1,
                dtype="int16",
                blocksize=1200,
                callback=speaker.callback,
            ),
        ):
            config = aai.TTSConfig(
                voice_id="aai-kyla",
                language="english",
                sample_rate=TTS_RATE,
            )
            async with aai.AsyncTTSSession(
                config,
                api_key=API_KEY,
            ) as session:
                await conversation(
                    session, speaker, microphone, history
                )
    finally:
        speaker.clear()
        microphone.close()
        history.close()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
