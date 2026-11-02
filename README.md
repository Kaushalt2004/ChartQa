# UniChart ChartQA UI

A Gradio web UI for a fine-tuned **UniChart** model (Donut-Swin encoder + mBART decoder, 202M params)
that answers questions about chart images. You don't need OCR: the model reads the chart straight from pixels.

## Features
- **Ask a chart**: upload a bar, line or pie chart and ask several questions in a row about it.
  Shows the prompt, raw decoder output and latency.
- **Benchmark on samples**: runs 12 questions on 4 generated charts with known values and scores
  them with the ChartQA *relaxed accuracy* metric (numbers count as correct within 5%).
- **Other UniChart tasks** *(experimental)*: summarize, extract the data table, open-ended QA.
  This checkpoint was fine-tuned on ChartQA, so these tasks give weak or truncated output.
- **How it works**: an architecture diagram, model details and usage tips.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or /whl/cu124 for an NVIDIA GPU
pip install -r requirements.txt
python make_examples.py         # regenerate the sample charts (they are already included)
```

### Model weights
The model (~800 MB) is not in this repo because it's over GitHub's file size limit.
Put the checkpoint folder next to this repo:

```
parent/
├── ChartQa/                   <- this repo
└── unichart-chartqa-final/    <- config.json, model.safetensors, tokenizer.json, ...
```

or point to it with `UNICHART_MODEL_DIR=/path/to/unichart-chartqa-final`.

## Run
```bash
python app.py      # or double-click run.bat on Windows
```
Open http://127.0.0.1:7860. It runs on GPU (float16) when one is available, otherwise on CPU (about 3–6 s per question).

## Prompt format
```
<chartqa> {question} <s_answer>      -> answer
```

## Limitations
- It only understands **data charts** (bar, line, pie, scatter). Posters, slides, diagrams and documents give meaningless output.
- Short factual questions work best; arithmetic (sums, averages) is less reliable.
- The checkpoint was saved with transformers 5.x. With transformers 4.x, `app.py` falls back to building the processor from `tokenizer.json` itself.
