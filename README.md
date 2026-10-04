# VivaSensei
Your Personal AI Viva Examiner

## Step 1: local Qwen inference

Uses **Qwen/Qwen2.5-7B-Instruct**, with the official chat template and a viva
examiner system prompt. Dependencies live in the sibling `dlenv` environment.
Run these commands from the `DeepLearning` directory:

```bash
dlenv/bin/python -m pip install -r VivaSensei/requirements.txt
dlenv/bin/python VivaSensei/inference.py --check
dlenv/bin/python VivaSensei/inference.py
```

The default input is `Ask me one viva question about CMOS current mirrors.`
Verified output from Qwen2.5-7B-Instruct on the development GPU:

> What are the key factors to consider when designing a CMOS current mirror for high accuracy in analog circuits?

The question comes from the model and is not hardcoded.

For another topic:

```bash
dlenv/bin/python VivaSensei/inference.py "Ask me one viva question about Miller's theorem."
```

`--check` downloads only tokenizer/config assets and verifies chat formatting.
It does not load weights or generate a response. Full inference downloads roughly
15 GB of model weights into `VivaSensei/.cache/huggingface/` by default; an existing
`HF_HOME` environment variable overrides this location. Use `--device cpu` or
`--device cuda` to select hardware explicitly; the default uses CUDA if available.
On CUDA, the default loads the same model in 4-bit NF4 precision using bitsandbytes,
with double quantization and bfloat16 computation (float16 on older GPUs without
bfloat16 support). This reduces GPU memory use and
can affect the generated question. `--precision bf16` selects full 16-bit weights,
needing about 14.2 GiB for weights plus runtime memory. CPU inference uses bfloat16
and requires at least 16 GiB of available RAM; otherwise the script stops before
downloading weights. Full bfloat16 GPU inference also requires at least 16 GiB of
available VRAM in this script. 4-bit loading still downloads the original weights.

The current machine has an RTX 4050 with 6 GB VRAM, suitable for this short 4-bit
run. The workspace sandbox hides GPU access, so inference must run from a normal
terminal or with approved execution outside the sandbox. Concurrent weight loading
is disabled to keep peak GPU memory within this device's capacity. Once downloaded,
run without Hugging Face network requests using:

```bash
HF_HUB_OFFLINE=1 dlenv/bin/python VivaSensei/inference.py
```

The model runs locally; speech features described below use ElevenLabs.

Reference: [Qwen's official model card and Transformers quickstart](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
Quantization: [Hugging Face bitsandbytes documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

## Same-chat viva memory

Start the chat app from the `DeepLearning` directory in a normal terminal with
GPU access:

```bash
HF_HUB_OFFLINE=1 dlenv/bin/python -m streamlit run VivaSensei/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open the local URL shown by Streamlit. Choose your subject, difficulty, current
topic, and weaknesses in the sidebar, then click **Ask first question**. Answer
in the chat box. Qwen briefly assesses your answer and asks a follow-up.

The current conversation is stored in `st.session_state.messages`, beginning
with one system message. Settings live in `st.session_state.viva`, a plain Python
dictionary. Each Qwen call receives the structured CURRENT VIVA context and all
previous assistant/user messages, followed by your latest answer. Earlier turns
are never silently dropped. **Apply settings** updates the system context and
preserves the chat; **Start new viva** clears the conversation while retaining
the chosen settings. Weaknesses are supplied by you and can be edited in the
sidebar; this step does not automatically extract or score them.

Only the model/tokenizer are cached across Streamlit reruns. Student history
stays per session. Calls to the shared model are serialized so separate browser
sessions do not perform GPU inference simultaneously. If generation fails, the
unanswered message is retained; **Retry response** retries the same turn and
**Edit unanswered message** lets you shorten or correct that turn and resubmit it
without duplicating history. **Discard unanswered message** lets you submit a
different answer.

The input budget is 2,048 tokens per call to keep memory use bounded on the local
6 GB GPU, with up to 192 new output tokens. If the full chat exceeds that budget,
the app retains the history and asks you to start a new viva or shorten your
answer. Browser reloads or server restarts can reset this memory; it is temporary
session context, with no database persistence yet.

Run session lifecycle tests without loading Qwen:

```bash
dlenv/bin/python -m unittest discover -s VivaSensei/tests -v
```

Run the real three-turn context check on the GPU:

```bash
HF_HUB_OFFLINE=1 dlenv/bin/python VivaSensei/tests/smoke_chat.py
```

This smoke check verifies generation and retained conversation formatting; it
does not validate technical correctness. During debugging, the local 4-bit model
gave an incorrect explanation of channel length modulation even with stricter
examiner instructions. Treat its assessments as practice feedback and verify
technical explanations against course material. Factual accuracy remains a model
limitation; passing the application tests does not resolve it.

Streamlit reference: [Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state).

## Voice answers and examiner speech

Install the dependencies in `requirements.txt` and put `ELEVENLABS_API_KEY` in
`VivaSensei/.env`. The helper resolves this file relative to its own location,
so launching Streamlit from `DeepLearning` works. An existing environment
variable takes precedence; optional `VivaSensei/.env.local` overrides `.env`
values when no environment variable is already set. Both files are gitignored.

Expand **Answer by voice**, allow microphone access, and record your answer.
Click **Transcribe recording**, review or edit **Review your transcript**, then
click **Send voice answer**. The reviewed text enters the same conversation as
a typed answer. Recording or transcribing alone does not send an answer to Qwen.
Use **Listen** below any examiner reply to generate an MP3, or enable
**Read new replies aloud** in the sidebar for automatic speech on new replies.
Browsers may require pressing the audio player's play button to begin playback.

Speech uses the helper's `eleven_v4` synthesis model, `scribe_v2` transcription
model, and existing voice ID. Your ElevenLabs account must have access to that
voice. Audio and transcripts are sent to ElevenLabs when the corresponding
controls are used; Qwen inference remains local. Automatic speech is off by
default. Ordinary reruns reuse generated audio within the current session,
and **Start new viva** clears voice drafts and audio along with the chat.
Speech failures can be retried without losing the text conversation.

The normal test suite mocks ElevenLabs and uses no API credits. To run one real
synthesis request and transcribe the result (uses ElevenLabs credits):

```bash
dlenv/bin/python VivaSensei/tests/smoke_speech.py
```

This saves the test MP3 under `VivaSensei/.cache/speech/smoke.mp3`.
References: [Streamlit audio recording](https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input),
[ElevenLabs Python SDK](https://github.com/elevenlabs/elevenlabs-python),
and [ElevenLabs models](https://elevenlabs.io/docs/overview/models).
