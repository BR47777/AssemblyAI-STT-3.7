import asyncio
import os
import re
import threading
import wave
from collections.abc import AsyncIterator
from contextlib import suppress

import assemblyai as aai
import sounddevice as sd
from openai import AsyncOpenAI

AAI_API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

SAMPLE_RATE = 24_000
CHANNELS = 1
BLOCK_SIZE = 1_200  # 50 ms of 16-bit mono PCM

openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


class Speaker:
    """Thread-safe PCM player whose buffer can be cleared on barge-in."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._generation = 0
        self.pcm = bytearray()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def write(self, chunk: bytes, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._buffer.extend(chunk)
            self.pcm.extend(chunk)

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self._buffer.clear()

    def callback(self, outdata, frames, _time, status) -> None:
        if status:
            print(f"Audio status: {status}")

        needed = frames * 2  # int16 mono
        with self._lock:
            chunk = bytes(self._buffer[:needed])
            del self._buffer[:needed]

        outdata[:] = chunk + (b"\x00" * (needed - len(chunk)))


async def openai_tokens(
    prompt: str,
    stop: asyncio.Event,
) -> AsyncIterator[str]:
    stream = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise voice assistant. Use short, complete "
                    "sentences without Markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    try:
        async for event in stream:
            if stop.is_set():
                break
            token = event.choices[0].delta.content or ""
            if token:
                yield token
    finally:
        await stream.close()


def split_sentences(text: str) -> tuple[list[str], str]:
    pattern = re.compile(r".+?[.!?](?:[\"')\]]+)?(?:\s+|$)", re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        return [], text
    return [m.group(0) for m in matches], text[matches[-1].end():]


async def interrupt_on_enter(stop: asyncio.Event) -> None:
    await asyncio.to_thread(
        input, "Press Enter while the agent speaks to interrupt...\n"
    )
    stop.set()


async def synthesize_flush(
    session: aai.AsyncTTSSession,
    flush_id: int,
    speaker: Speaker,
    stop: asyncio.Event,
    generation: int,
) -> None:
    async for chunk in session.audio(flush_id):
        if stop.is_set():
            return
        speaker.write(chunk, generation)


async def speak_turn(
    session: aai.AsyncTTSSession,
    speaker: Speaker,
    prompt: str,
) -> bool:
    stop = asyncio.Event()
    generation = speaker.generation
    audio_tasks: list[asyncio.Task[None]] = []
    interrupt_task = asyncio.create_task(interrupt_on_enter(stop))
    pending = ""

    try:
        async for token in openai_tokens(prompt, stop):
            pending += token
            sentences, pending = split_sentences(pending)

            for sentence in sentences:
                await session.generate(sentence)
                flush_id = await session.flush()
                audio_tasks.append(
                    asyncio.create_task(
                        synthesize_flush(
                            session, flush_id, speaker, stop, generation
                        )
                    )
                )

        if pending.strip() and not stop.is_set():
            await session.generate(pending)
            flush_id = await session.flush()
            audio_tasks.append(
                asyncio.create_task(
                    synthesize_flush(
                        session, flush_id, speaker, stop, generation
                    )
                )
            )

        if audio_tasks and not stop.is_set():
            await asyncio.gather(*audio_tasks)

    finally:
        if stop.is_set():
            # Clear playback first so interruption feels immediate.
            speaker.clear()

            # Cancel drops in-flight synthesis, unflushed text, and queued flushes.
            await session.cancel()

            for task in audio_tasks:
                task.cancel()
            await asyncio.gather(*audio_tasks, return_exceptions=True)

            print("Agent interrupted; ready for the user's next turn.")
        else:
            interrupt_task.cancel()
            with suppress(asyncio.CancelledError):
                await interrupt_task

    return stop.is_set()


async def main() -> None:
    speaker = Speaker()
    config = aai.TTSConfig(
        voice_id="aai-kyla",
        language="english",
        sample_rate=SAMPLE_RATE,
    )

    try:
        with sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCK_SIZE,
            callback=speaker.callback,
        ):
            async with aai.AsyncTTSSession(
                config,
                api_key=AAI_API_KEY,
            ) as session:
                print(f"TTS session: {session.begin.id}")
                await speak_turn(
                    session,
                    speaker,
                    "Explain why the sky appears blue in three sentences.",
                )

    except aai.TTSError as error:
        raise SystemExit(
            f"TTS failed: {error}; code={error.error_code}"
        ) from error
    except Exception as error:
        raise SystemExit(f"Voice generation failed: {error}") from error

    with wave.open("openai_reply.wav", "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(speaker.pcm)

    print("Wrote the audio played before interruption to openai_reply.wav")


if __name__ == "__main__":
    asyncio.run(main())
