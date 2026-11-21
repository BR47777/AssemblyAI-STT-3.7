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
    def __init__(self):
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.generation = 0

    def write(self, chunk: bytes, generation: int):
        with self.lock:
            if generation == self.generation:
                self.buffer.extend(chunk)

    def clear(self):
        with self.lock:
            self.generation += 1
            self.buffer.clear()

    def callback(self, outdata, frames, _time, _status):
        needed = frames * 2  # mono int16
        with self.lock:
            chunk = bytes(self.buffer[:needed])
            del self.buffer[:needed]
        outdata[:] = chunk + b"\0" * (needed - len(chunk))


class MicrophoneSTT:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
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
                # Lower this toward 0 for faster, more sensitive interruption.
                interruption_delay=200,
                language_codes=["en"],
            )
        )

    def _speech_started(
        self, _client: RealTimeTranscriber, _event: SpeechStartedEvent
    ):
        if self.barge_in:
            self.loop.call_soon_threadsafe(self.barge_in.set)

    def _turn(self, _client: RealTimeTranscriber, event: TurnEvent):
        if event.end_of_turn and event.transcript:
            print(f"\nUser: {event.transcript}")

    def _error(self, _client: RealTimeTranscriber, error: RealTimeError):
        print(f"\nSTT error: {error}")

    def microphone_callback(self, data, _frames, _time, status):
        if status:
            print(f"Microphone status: {status}")
        self.client.stream(bytes(data))

    def close(self):
        self.client.disconnect(terminate=True)


async def openai_tokens(prompt: str):
    stream = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": "Be concise. Use short sentences and no Markdown.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    try:
        async for event in stream:
            token = event.choices[0].delta.content or ""
            if token:
                yield token
    finally:
        await stream.close()


def split_sentences(text: str):
    pattern = re.compile(r".+?[.!?](?:[\"')\]]+)?(?:\s+|$)", re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        return [], text
    return [m.group(0) for m in matches], text[matches[-1].end():]


async def receive_audio(session, flush_id, speaker, generation, stop):
    async for chunk in session.audio(flush_id):
        if stop.is_set():
            return
        speaker.write(chunk, generation)


async def speak_turn(session, speaker, microphone, prompt):
    stop = asyncio.Event()
    microphone.barge_in = stop
    generation = speaker.generation
    pending = ""
    audio_tasks = []

    try:
        async for token in openai_tokens(prompt):
            if stop.is_set():
                break
            pending += token
            sentences, pending = split_sentences(pending)

            for sentence in sentences:
                await session.generate(sentence)
                flush_id = await session.flush()
                audio_tasks.append(asyncio.create_task(
                    receive_audio(
                        session, flush_id, speaker, generation, stop
                    )
                ))

        if pending.strip() and not stop.is_set():
            await session.generate(pending)
            flush_id = await session.flush()
            audio_tasks.append(asyncio.create_task(
                receive_audio(session, flush_id, speaker, generation, stop)
            ))

        if audio_tasks and not stop.is_set():
            await asyncio.gather(*audio_tasks)
    finally:
        microphone.barge_in = None

        if stop.is_set():
            speaker.clear()            # Immediate local silence
            await session.cancel()     # Drop queued/in-flight TTS
            for task in audio_tasks:
                task.cancel()
            await asyncio.gather(*audio_tasks, return_exceptions=True)
            print("\nBarge-in detected; agent stopped.")


async def main():
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
                config, api_key=API_KEY
            ) as tts_session:
                print("Listening. Speak while the agent replies to interrupt.")
                await speak_turn(
                    tts_session,
                    speaker,
                    microphone,
                    "Explain why the sky appears blue in three sentences.",
                )
    finally:
        microphone.close()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
