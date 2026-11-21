import os
import re
import wave

import assemblyai as aai

API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
VOICE_ID = "aai-kyla"
SAMPLE_RATE = 24_000


def generate_wav() -> None:
    tts = aai.TextToSpeech(api_key=API_KEY)
    response = tts.synthesize(
        "Hello from our Text to Speech API.",
        aai.TTSConfig(
            voice_id=VOICE_ID,
            container=aai.TTSContainer.wav,
        ),
    )
    with open("hello.wav", "wb") as output:
        output.write(response.audio)
    print(f"Wrote hello.wav ({response.audio_duration_ms} ms)")


def simulated_llm_tokens():
    reply = "Sure, I can help. What would you like to know next?"
    for token in re.findall(r"\S+\s*", reply):
        yield token


def stream_agent_reply() -> None:
    config = aai.TTSConfig(voice_id=VOICE_ID, language="english")
    pcm = bytearray()
    sentence = ""

    with aai.TTSSession(config, api_key=API_KEY) as session:
        print(f"TTS session: {session.begin.id}")

        for token in simulated_llm_tokens():
            sentence += token
            if re.search(r"[.!?]\s*$", sentence):
                session.generate(sentence)
                flush_id = session.flush()
                for chunk in session.audio(flush_id):
                    pcm.extend(chunk)  # Send each chunk to your audio player here.
                sentence = ""

        if sentence:
            session.generate(sentence)
            for chunk in session.audio(session.flush()):
                pcm.extend(chunk)

    with wave.open("agent_reply.wav", "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)  # 16-bit PCM
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)
    print("Wrote agent_reply.wav")


if __name__ == "__main__":
    generate_wav()
    stream_agent_reply()
