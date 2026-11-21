import asyncio
import os
import re
import wave
from collections.abc import AsyncIterator

import assemblyai as aai

API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
SAMPLE_RATE = 24_000


async def simulated_llm_tokens() -> AsyncIterator[str]:
    """Replace this with your LLM client's async token stream."""
    reply = (
        "I found your account. "
        "Your latest invoice is due next Friday. "
        "Would you like me to explain the charges?"
    )
    for token in re.findall(r"\S+\s*", reply):
        await asyncio.sleep(0.04)
        yield token


async def play_audio(chunk: bytes, output: bytearray) -> None:
    """Replace or extend this with your live audio output."""
    output.extend(chunk)
    await asyncio.sleep(0)  # Yield control to the LLM producer.


async def consume_audio(
    session: aai.AsyncTTSSession,
    flush_id: int,
    output: bytearray,
) -> None:
    async for chunk in session.audio(flush_id):
        await play_audio(chunk, output)


async def speak_llm_reply(
    session: aai.AsyncTTSSession,
    output: bytearray,
) -> None:
    pending_text = ""
    audio_tasks: list[asyncio.Task[None]] = []

    async for token in simulated_llm_tokens():
        pending_text += token

        # Flush complete sentences for low latency and natural phrasing.
        if re.search(r"[.!?][\"')\]]?\s*$", pending_text):
            await session.generate(pending_text)
            flush_id = await session.flush()
            audio_tasks.append(
                asyncio.create_task(
                    consume_audio(session, flush_id, output)
                )
            )
            pending_text = ""

    if pending_text:
        await session.generate(pending_text)
        flush_id = await session.flush()
        audio_tasks.append(
            asyncio.create_task(consume_audio(session, flush_id, output))
        )

    if audio_tasks:
        await asyncio.gather(*audio_tasks)


async def main() -> None:
    config = aai.TTSConfig(
        voice_id="aai-kyla",
        language="english",
        sample_rate=SAMPLE_RATE,
    )
    pcm = bytearray()

    try:
        async with aai.AsyncTTSSession(
            config,
            api_key=API_KEY,
        ) as session:
            print(f"TTS session: {session.begin.id}")
            await speak_llm_reply(session, pcm)

    except aai.TTSError as error:
        raise SystemExit(
            f"TTS failed: {error}; code={error.error_code}"
        ) from error

    # Session audio is raw PCM, so add a WAV header before saving.
    with wave.open("async_agent_reply.wav", "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)

    print("Wrote async_agent_reply.wav")


if __name__ == "__main__":
    asyncio.run(main())
