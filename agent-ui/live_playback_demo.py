import asyncio
import os
import re
import wave
from collections.abc import AsyncIterator

import assemblyai as aai
import sounddevice as sd
from openai import AsyncOpenAI

AAI_API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

SAMPLE_RATE = 24_000
CHANNELS = 1
BLOCK_SIZE = 2_400  # 100 ms at 24 kHz
SENTINEL = object()

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


class Speaker:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[bytes | object] = asyncio.Queue(maxsize=20)
        self.pcm = bytearray()

    async def write(self, chunk: bytes) -> None:
        self.pcm.extend(chunk)
        await self.queue.put(chunk)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        def callback(outdata, frames, _time, status) -> None:
            if status:
                print(f"Audio status: {status}")

            needed = frames * 2  # mono, signed 16-bit samples
            data = bytearray()
            while len(data) < needed:
                try:
                    item = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is SENTINEL:
                    break
                data.extend(item)

            data.extend(b"\x00" * (needed - len(data)))
            outdata[:] = data[:needed]

            # Put any oversized remainder back from the audio thread.
            remainder = bytes(data[needed:])
            if remainder:
                loop.call_soon_threadsafe(self.queue.put_nowait, remainder)

        with sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCK_SIZE,
            callback=callback,
        ):
            while True:
                await asyncio.sleep(0.05)
                if self.queue.qsize() == 1 and self.queue._queue[0] is SENTINEL:
                    self.queue.get_nowait()
                    break

    async def finish(self) -> None:
        await self.queue.put(SENTINEL)


async def openai_tokens() -> AsyncIterator[str]:
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
            {
                "role": "user",
                "content": "Explain why the sky appears blue.",
            },
        ],
    )

    async for event in stream:
        token = event.choices[0].delta.content or ""
        if token:
            yield token


def complete_sentences(text: str) -> tuple[list[str], str]:
    pattern = re.compile(r".+?[.!?](?:[\"')\]]+)?(?:\s+|$)", re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        return [], text
    return [m.group(0) for m in matches], text[matches[-1].end():]


async def speak_openai_reply(
    session: aai.AsyncTTSSession,
    speaker: Speaker,
) -> None:
    pending = ""
    flush_ids: list[int] = []

    async for token in openai_tokens():
        pending += token
        sentences, pending = complete_sentences(pending)

        for sentence in sentences:
            await session.generate(sentence)
            flush_ids.append(await session.flush())

    if pending.strip():
        await session.generate(pending)
        flush_ids.append(await session.flush())

    # Requests arrive in order; consume them in order for gap-free playback.
    for flush_id in flush_ids:
        async for chunk in session.audio(flush_id):
            await speaker.write(chunk)


async def main() -> None:
    speaker = Speaker()
    playback_task = asyncio.create_task(speaker.run())

    config = aai.TTSConfig(
        voice_id="aai-kyla",
        language="english",
        sample_rate=SAMPLE_RATE,
    )

    try:
        async with aai.AsyncTTSSession(
            config,
            api_key=AAI_API_KEY,
        ) as session:
            print(f"TTS session: {session.begin.id}")
            await speak_openai_reply(session, speaker)
    except Exception as error:
        raise SystemExit(f"Voice generation failed: {error}") from error
    finally:
        await speaker.finish()
        await playback_task

    with wave.open("openai_reply.wav", "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(speaker.pcm)

    print("Playback complete; wrote openai_reply.wav")


if __name__ == "__main__":
    asyncio.run(main())
