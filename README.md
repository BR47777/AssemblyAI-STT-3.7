# AssemblyAI STT Voice Agent

A collection of AssemblyAI speech demos and a browser-based voice agent. The
main app combines realtime speech-to-text, streamed LLM responses, text-to-speech,
barge-in interruption, and locally persisted conversation history.

## Contents

- [Architecture](#architecture)
- [Requirements and installation](#requirements-and-installation)
- [Run the voice app](#run-the-voice-app)
- [API keys in the UI](#api-keys-in-the-ui)
- [Optional Infisical helper](#optional-infisical-helper)
- [Configuration](#configuration)
- [WebSocket protocol](#websocket-protocol)
- [Conversation persistence](#conversation-persistence)
- [Standalone demos](#standalone-demos)
- [Tests](#tests)
- [Project structure](#project-structure)

## Architecture
![Uploading Bazai Agent Paradise Architecture (1).png…]()

```text
Browser (static/index.html)
  API key form -> browser localStorage -> configure message
  microphone -> AudioWorklet (mono PCM, 16 kHz, 50 ms / 1,600 bytes)
       | WebSocket /ws
       v
FastAPI (web_app.py)
  |-- connection-scoped API keys -> OpenAI and AssemblyAI clients
  |-- realtime audio -> AssemblyAI STT -> partial/final transcripts
  |-- final user text -> conversation queue -> OpenAI streamed response
  |                                          | deltas -> browser chat
  |                                          v
  |                               sentence chunks -> AssemblyAI TTS
  |                                          | PCM audio -> browser playback
  |-- conversation messages and versioned summaries -> SQLite
  `-- threshold-based summaries -> AssemblyAI LLM Gateway
```

### Main voice turn

1. The browser requests microphone access and loads `static/pcm-worklet.js`.
   The worklet converts microphone audio to mono signed 16-bit PCM, 16 kHz,
   sending 50 ms binary frames (800 samples, 1,600 bytes).
2. FastAPI forwards audio frames to an AssemblyAI realtime transcriber.
   Partial and final transcript events are sent to the browser. Final user
   messages are persisted and queued for response generation.
3. A worker streams the response from OpenAI. Text deltas are sent to the UI
   while completed sentence chunks are sent to an AssemblyAI TTS session.
   Long sentences are split at a 240-character maximum.
4. TTS PCM audio is sent to the browser as binary WebSocket frames. Speaking
   over the agent or sending an interrupt control cancels the active response
   and stops scheduled playback.
5. After enough new messages accumulate, the app requests a durable summary
   from the AssemblyAI LLM Gateway and stores it as a new summary version.

The browser UI is served by FastAPI from `static/index.html`; `/pcm-worklet.js`
serves the audio worklet and `/healthz` provides a health check. The user
enters both API keys in the UI. The browser remembers them in `localStorage`;
the backend uses them only for that WebSocket connection and does not write
them to the database or logs. Since browser scripts on this origin can access
`localStorage`, use a trusted device and do not expose this unauthenticated
demo app to an untrusted network.

## Requirements and installation

Python 3.11 or newer is recommended. Node.js is used only for the frontend
check.

From the repository root:

```bash
python -m pip install -r agent-ui/requirements.txt
```

Enter both the AssemblyAI and OpenAI API keys in the app UI. Do not commit
credentials.

## Run the voice app

From the repository root:

```bash
uvicorn web_app:app --app-dir agent-ui --reload
```

Open <http://127.0.0.1:8000>, enter both keys, and select **Connect with keys**.
The values are saved in this browser until **Forget saved keys** is selected.
The app requires an HTTPS origin when used outside localhost so credentials
travel over a secure WebSocket.

## API keys in the UI

Both fields are password inputs. On connect, the browser stores the values in
`localStorage` and sends them in the first WebSocket configuration message.
The backend validates the OpenAI key by listing available models and checks
that the configured model is available. It validates the AssemblyAI key with
a read-only transcript-list request. These checks do not generate text or
transcribe audio. On success, the backend creates an OpenAI client scoped to
that connection and passes the AssemblyAI key to the realtime STT, TTS, and
LLM Gateway clients. The OpenAI client is closed when the WebSocket ends.

Provider validation cannot guarantee later requests will be within quota. If
OpenAI rate-limits a voice response, the UI reports that account limits or
billing should be checked instead of showing only a generic failure.

To remove keys from the browser, select **Forget saved keys**. This closes the
current connection; refresh the page to connect again. Do not use the remember
option on a shared browser profile or device.

## Optional Infisical helper

The standalone [`agent-ui/infisical_secrets.py`](./agent-ui/infisical_secrets.py)
helper can retrieve an AssemblyAI secret for other Python scripts. It is not
used by the web UI, which requires both keys to be entered in the browser.

To use the helper, create an Infisical machine identity with read access to
the desired project and environment, then set its credentials locally:

```bash
export INFISICAL_CLIENT_ID="<machine-identity-client-id>"
export INFISICAL_CLIENT_SECRET="<machine-identity-client-secret>"
export INFISICAL_PROJECT_ID="<project-id>"
export INFISICAL_ENVIRONMENT_SLUG="dev"
export INFISICAL_SECRET_PATH="/"
export INFISICAL_SECRET_NAME="ASSEMBLYAI_API_KEY"
```

The host defaults to `https://app.infisical.com`; set `INFISICAL_HOST` for a
different instance. Use the Infisical project ID rather than its display name.

Call the helper from a script in `agent-ui/`:

```python
from infisical_secrets import get_assemblyai_api_key

api_key = get_assemblyai_api_key()
if api_key is None:
    raise RuntimeError("AssemblyAI API key is not configured.")

# Pass api_key to the AssemblyAI client you are creating.
```

If `ASSEMBLYAI_API_KEY` is set directly, the helper returns it first. If no
direct key or Infisical credentials are configured, it returns `None`.
Incomplete credentials or a failed lookup raise an error.

## Configuration

| Variable | Used by | Default | Description |
| --- | --- | --- | --- |
| `ASSEMBLYAI_API_KEY` | Standalone demos/helper | None | AssemblyAI key; the web UI requires it in the browser. |
| `OPENAI_API_KEY` | Standalone demos | None | OpenAI key; the web UI requires it in the browser. |
| `INFISICAL_CLIENT_ID` | Standalone helper | None | Universal Auth machine identity client ID. |
| `INFISICAL_CLIENT_SECRET` | Standalone helper | None | Universal Auth machine identity client secret. |
| `INFISICAL_PROJECT_ID` | Standalone helper | None | Infisical project ID (not the project name). |
| `INFISICAL_ENVIRONMENT_SLUG` | Standalone helper | `dev` | Infisical environment slug. |
| `INFISICAL_SECRET_PATH` | Standalone helper | `/` | Folder containing the secret. |
| `INFISICAL_SECRET_NAME` | Standalone helper | `ASSEMBLYAI_API_KEY` | Secret key to retrieve. |
| `INFISICAL_HOST` | Standalone helper | `https://app.infisical.com` | Infisical host. |
| `OPENAI_MODEL` | App and selected demos | `gpt-4.1-mini` | Model used for response generation. |
| `SUMMARY_MODEL` | App and summary demos | `claude-sonnet-4-6` | Model used for summary generation. |
| `SUMMARY_TRIGGER_MESSAGES` | Web app | `30` | Unsummarized message threshold. |
| `TTS_VOICE_ID` | Web app | `aai-kyla` | AssemblyAI TTS voice. |
| `HISTORY_DB` | Web app and history demos | `agent_ui_history.db` in app | SQLite database path. |
| `SUMMARY_TRIGGER_TOKENS` | Token-budget demo | `6000` | Token threshold for `token_budget_voice_history.py`. |
| `RECENT_CONTEXT_TOKENS` | Token-budget demo | `2500` | Recent token budget for `token_budget_voice_history.py`. |
| `CONVERSATION_ID` | History demos | Generated UUID | Selects a persisted demo conversation. |
| `LLM_GATEWAY_MODEL_REGION` | Gateway summary demos | None | Only `global` is accepted when set. |

`SUMMARY_TRIGGER_TOKENS`, `RECENT_CONTEXT_TOKENS`, and
`LLM_GATEWAY_MODEL_REGION` apply to the standalone history demos, not the web
app's message-count summary setting.

## WebSocket protocol

The browser connects to `/ws` and uses JSON text frames for controls and
16 kHz PCM binary frames for microphone input.

The first client control must configure both API keys:

```json
{
  "type": "configure",
  "assemblyAiApiKey": "<assemblyai-key>",
  "openAiApiKey": "<openai-key>"
}
```

The server checks that both values are nonempty and within the input size
limit, validates the keys and configured OpenAI model, then responds with
`{"type":"ready"}` before accepting conversation controls. Keys are never
included in server messages.

Other client controls:

| `type` | Fields | Purpose |
| --- | --- | --- |
| `start` | Optional `conversationId` | Start or resume a conversation; generates an ID if omitted. |
| `select_conversation` | `conversationId` | Resume an existing conversation. |
| `text` | `text` | Send a typed user message in the active conversation. |
| `interrupt` | None | Cancel the active response and audio. |
| `stop` | None | Stop microphone capture and active response. |

Server JSON events include `state`, `user_partial`, `user_final`,
`assistant_delta`, `assistant_final`, `summary_updated`, and `error`. TTS audio
is sent as binary frames. Errors carry a `code` and user-facing `message`.

## Conversation persistence

`agent-ui/conversation_store.py` uses SQLite. The app database has:

- `conversations`: conversation IDs and creation/update timestamps.
- `messages`: user and assistant messages, interruption flags, and timestamps.
- `summaries`: versioned summaries and the last message included in each one.

The app defaults to `agent_ui_history.db` in the current working directory.
Use `HISTORY_DB` to change the path. Conversation history and summaries are
local application data; API keys are not stored in these tables.
The sidebar lists all saved conversations and lets you view their complete
message history. The transcript follows new messages only while you are
already at the bottom, so you can scroll up without being pulled back down.
Select a conversation and choose **Download transcript** to save a text file
with each speaker, the message timestamp in UTC, and interruption markers.

## Standalone demos

Run demos from the `agent-ui` directory after installing dependencies and
setting the API keys they require:

```bash
cd agent-ui
python demo.py
python async_demo.py
python openai_demo.py
```

Other examples in `agent-ui/` cover live playback, interruption and barge-in,
realtime voice conversation, persistent history, summarized history,
token-budgeted context, and versioned summary auditing:

- `live_playback_demo.py`, `interruption_demo.py`, `stt_bargein_demo.py`
- `realtime_voice_conversation.py`
- `persistent_voice_history.py`, `summarized_voice_history.py`,
  `gateway_summarized_voice_history.py`
- `token_budget_voice_history.py`, `summary_audit_voice_history.py`

For the token-budgeted history demos:

```bash
export SUMMARY_TRIGGER_TOKENS=6000
export RECENT_CONTEXT_TOKENS=2500
python token_budget_voice_history.py
```

`summary_audit_voice_history.py` supports `summary-history` and
`summary-show <version>` commands. Set `CONVERSATION_ID` and `HISTORY_DB` to
inspect a conversation in a later process.

## Tests

From the repository root:

```bash
python -m unittest discover -s agent-ui/tests -v
node agent-ui/tests/check_frontend.js
```

The Python suite uses fake providers to test imports, audio/TTS behavior,
WebSocket flows, interruption, persistence, and summaries without live API
calls or audio hardware. The Node check validates browser code and confirms
the expected PCM frame shape. Both checks run in GitHub Actions via
`.github/workflows/agent-ui-eval.yml`.

## Project structure

```text
agent-ui/
  web_app.py                 FastAPI WebSocket voice-agent backend
  infisical_secrets.py       Universal Auth secret retrieval helper
  conversation_store.py      SQLite conversations, messages, and summaries
  static/
    index.html               Voice Studio UI and WebSocket client
    pcm-worklet.js           Browser microphone PCM conversion
  tests/
    test_eval_pipeline.py    Mocked end-to-end Python evaluation
    check_frontend.js        Browser script and PCM checks
  requirements.txt           Python dependencies
  *_demo.py                  Focused speech and voice-history examples
```
