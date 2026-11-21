import asyncio
import os
import re
import threading
from contextlib import suppress

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
MIC_RATE, TTS_RATE = 16_000, 24_000

openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


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
        needed = frames * 2  # mono, signed 16-bit PCM
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
        event = self.barge_in
        if event is not None:
            self.loop.call_soon_threadsafe(event.set)

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
            RealTimeSessionParameters(agent_context=reply[:1750])
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
    return [match.group(0) for match in matches], text[matches[-1].end():]


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
) -> str:
    stop = asyncio.Event()
    microphone.barge_in = stop
    generation = speaker.generation
    pending = ""
    full_reply = ""
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

            full_reply += token
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

    return full_reply.strip()


async def conversation(
    session: aai.AsyncTTSSession,
    speaker: Speaker,
    microphone: MicrophoneSTT,
) -> None:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a concise voice assistant. Use short, natural "
                "sentences without Markdown. Ask at most one question at a time."
            ),
        }
    ]

    print("Listening. Speak a request, or say 'goodbye' to exit.")

    while True:
        user_text = await microphone.turns.get()
        print(f"\nUser: {user_text}")

        if user_text.casefold().strip(" .!?") in {
            "goodbye",
            "quit",
            "exit",
            "stop conversation",
        }:
            break

        messages.append({"role": "user", "content": user_text})
        print("Assistant: ", end="", flush=True)

        reply = await speak_turn(
            session,
            speaker,
            microphone,
            messages,
        )

        if reply:
            print(reply)
            messages.append({"role": "assistant", "content": reply})
            microphone.set_agent_context(reply)

        # Bound local LLM history while retaining the system message.
        if len(messages) > 21:
            messages[:] = [messages[0], *messages[-20:]]


async def main() -> None:
    loop = asyncio.get_running_loop()
    speaker = Speaker()
    microphone = MicrophoneSTT(loop)

    try:
        with (
            sd.RawInputStream(
                samplerate=MIC_RATE,
                channels=1,
                dtype="int16",
                blocksize=800,  # 50 ms
                callback=microphone.microphone_callback,
            ),
            sd.RawOutputStream(
                samplerate=TTS_RATE,
                channels=1,
                dtype="int16",
                blocksize=1200,  # 50 ms
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
                await conversation(session, speaker, microphone)
    finally:
        speaker.clear()
        microphone.close()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
