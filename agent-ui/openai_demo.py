import asyncio
import os
import re
import wave

import assemblyai as aai
from openai import AsyncOpenAI

AAI_API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SAMPLE_RATE = 24_000

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Return complete sentences and any unfinished remainder."""
    matches = list(re.finditer(r".+?[.!?](?:[\"')\]]+)?(?:\s+|$)", buffer))
    if not matches:
        return [], buffer

    sentences = [match.group(0) for match in matches]
    return sentences, buffer[matches[-1].end():]


async def collect_audio(
    session: aai.AsyncTTSSession,
    flush_id: int,
    pcm: bytearray,
    lock: asyncio.Lock,
) -> None:
    # Requests are returned in order, but the lock also protects the WAV buffer.
    async for chunk in session.audio(flush_id):
        async with lock:
            pcm.extend(chunk)
        # Send `chunk` to your playback/WebRTC layer here.


async def stream_reply(
    session: aai.AsyncTTSSession,
    pcm: bytearray,
    lock: asyncio.Lock,
) -> None:
    stream = await openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
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

    pending = ""
    audio_tasks: list[asyncio.Task[None]] = []

    async for event in stream:
        token = event.choices[0].delta.content or ""
        if not token:
            continue

        pending += token
        sentences, pending = split_sentences(pending)

        for sentence in sentences:
            await session.generate(sentence)
            flush_id = await session.flush()
            audio_tasks.append(
                asyncio.create_task(
                    collect_audio(session, flush_id, pcm, lock)
                )
            )

    if pending.strip():
        await session.generate(pending)
        flush_id = await session.flush()
        audio_tasks.append(
            asyncio.create_task(
                collect_audio(session, flush_id, pcm, lock)
            )
        )

    await asyncio.gather(*audio_tasks)


async def main() -> None:
    pcm = bytearray()
    pcm_lock = asyncio.Lock()
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
            await stream_reply(session, pcm, pcm_lock)
    except Exception as error:
        raise SystemExit(f"Voice generation failed: {error}") from error

    # WebSocket TTS returns raw 16-bit mono PCM, so add a WAV header.
    with wave.open("openai_reply.wav", "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)

    print("Wrote openai_reply.wav")


if __name__ == "__main__":
    asyncio.run(main())
