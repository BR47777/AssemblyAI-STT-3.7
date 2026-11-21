# AssemblyAI STT Voice Agent

A collection of AssemblyAI speech demos and a browser-based voice agent. The
main app combines realtime speech-to-text, streamed LLM responses, text-to-speech,
barge-in interruption, and locally persisted conversation history.

## Contents

- [Architecture](#architecture)
- [Requirements and installation](#requirements-and-installation)
- [Run the voice app](#run-the-voice-app)
- [Infisical setup](#infisical-setup)
- [Configuration](#configuration)
- [WebSocket protocol](#websocket-protocol)
- [Conversation persistence](#conversation-persistence)
- [Standalone demos](#standalone-demos)
- [Tests](#tests)
- [Project structure](#project-structure)

## Architecture

```text
Browser (static/index.html)
  microphone -> AudioWorklet (mono PCM, 16 kHz, 50 ms / 1,600 bytes)
       | WebSocket /ws
       v
FastAPI (web_app.py)
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
serves the audio worklet and `/healthz` provides a health check. API
credentials remain on the server. The app is a development/demo app and does
not implement user authentication; keep it on a trusted network.

## Requirements and installation

Python 3.11 or newer is recommended. Node.js is used only for the frontend
check.

From the repository root:

```bash
python -m pip install -r agent-ui/requirements.txt
```

Provide `OPENAI_API_KEY` and either `ASSEMBLYAI_API_KEY` or the Infisical
Universal Auth settings described below. Replace example values locally; do
not commit credentials.

## Run the voice app

From the repository root:

```bash
export OPENAI_API_KEY="<your-openai-key>"
export ASSEMBLYAI_API_KEY="<your-assemblyai-key>"
uvicorn web_app:app --app-dir agent-ui --reload
```

Open <http://127.0.0.1:8000>. When an AssemblyAI key is loaded through
Infisical, omit the direct `ASSEMBLYAI_API_KEY` export. A direct
`ASSEMBLYAI_API_KEY`, when present, takes precedence over Infisical.

## Infisical setup

The app uses the `infisicalsdk` Python package and Universal Auth to read one
secret at WebSocket connection time. The fetched value is cached in memory for
the process; it is not written to disk or printed.

### Configure the project in Infisical

At <https://app.infisical.com/>:

1. Open the **BAZAI** project.
2. Select the `dev` environment and `/` folder.
3. Create a secret named `ASSEMBLYAI_API_KEY` and set its value to your
   AssemblyAI API key.
4. Create a machine identity and grant it permission to read the secret in
   that project and environment.
5. Copy the machine identity client ID and client secret. Find the project
   **ID** as well; `BAZAI` is the project name, not necessarily its ID.

### Configure and run locally

Set these variables in the same shell used to run the app:

```bash
export INFISICAL_CLIENT_ID="<machine-identity-client-id>"
export INFISICAL_CLIENT_SECRET="<machine-identity-client-secret>"
export INFISICAL_PROJECT_ID="<project-id>"
export INFISICAL_ENVIRONMENT_SLUG="dev"
export INFISICAL_SECRET_PATH="/"
export INFISICAL_SECRET_NAME="ASSEMBLYAI_API_KEY"
export OPENAI_API_KEY="<your-openai-key>"

uvicorn web_app:app --app-dir agent-ui --reload
```

Defaults are `https://app.infisical.com` for the host, `dev` for the
environment, `/` for the secret path, and `ASSEMBLYAI_API_KEY` for the secret
name. Override the host with `INFISICAL_HOST` for a different Infisical
instance. Keep machine identity credentials out of source control and logs.

The reusable SDK helper is in
[`agent-ui/infisical_secrets.py`](./agent-ui/infisical_secrets.py). It can also
be called from a Python script in `agent-ui/`:

```python
from infisical_secrets import get_assemblyai_api_key

api_key = get_assemblyai_api_key()
if api_key is None:
    raise RuntimeError("AssemblyAI API key is not configured.")

# Pass api_key to the AssemblyAI client you are creating.
```

If no direct key or Infisical credentials are configured, the helper returns
`None`. Partial Universal Auth configuration or a failed Infisical request
raises an error.

## Configuration

| Variable | Used by | Default | Description |
| --- | --- | --- | --- |
| `ASSEMBLYAI_API_KEY` | App and demos | None | AssemblyAI API key; overrides Infisical in the web app. |
| `OPENAI_API_KEY` | App and demos | None | OpenAI API key for response generation. |
| `INFISICAL_CLIENT_ID` | Web app helper | None | Universal Auth machine identity client ID. |
| `INFISICAL_CLIENT_SECRET` | Web app helper | None | Universal Auth machine identity client secret. |
| `INFISICAL_PROJECT_ID` | Web app helper | None | Infisical project ID (not the project name). |
| `INFISICAL_ENVIRONMENT_SLUG` | Web app helper | `dev` | Infisical environment slug. |
| `INFISICAL_SECRET_PATH` | Web app helper | `/` | Folder containing the secret. |
| `INFISICAL_SECRET_NAME` | Web app helper | `ASSEMBLYAI_API_KEY` | Secret key to retrieve. |
| `INFISICAL_HOST` | Web app helper | `https://app.infisical.com` | Infisical host. |
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

Client controls:

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
