"""UniChart ChartQA demo UI.

Run:  python app.py      (then open http://127.0.0.1:7860)

Env vars:
  UNICHART_MODEL_DIR  path to the model folder (default: ../unichart-chartqa-final)
  UNICHART_FP32=1     force float32 on GPU (default is float16 on CUDA)
"""
import json
import os
import re
import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

HERE = Path(__file__).parent
MODEL_DIR = Path(os.environ.get("UNICHART_MODEL_DIR", HERE.parent / "unichart-chartqa-final"))
EXAMPLES_DIR = HERE / "examples"

# --------------------------------------------------------------------------- model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" and os.environ.get("UNICHART_FP32") != "1" else torch.float32

print(f"Loading UniChart from {MODEL_DIR} on {DEVICE} ({DTYPE}) ...")
t0 = time.time()
def load_processor(model_dir: Path) -> DonutProcessor:
    try:
        return DonutProcessor.from_pretrained(model_dir)
    except Exception as e:  # checkpoint saved with transformers 5.x, older versions can't parse its config
        print(f"DonutProcessor.from_pretrained failed ({e!r}); building processor manually")
        from transformers import DonutImageProcessor, XLMRobertaTokenizerFast

        img_cfg = json.loads((model_dir / "processor_config.json").read_text())["image_processor"]
        img_cfg.pop("image_processor_type", None)
        tok = XLMRobertaTokenizerFast(
            tokenizer_file=str(model_dir / "tokenizer.json"),
            bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>",
            unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
        )
        return DonutProcessor(image_processor=DonutImageProcessor(**img_cfg), tokenizer=tok)


processor = load_processor(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE, DTYPE).eval()
LOAD_SECONDS = time.time() - t0
N_PARAMS = sum(p.numel() for p in model.parameters())
print(f"Loaded in {LOAD_SECONDS:.1f}s ({N_PARAMS / 1e6:.0f}M params)")

TOK = processor.tokenizer
IMG_H = processor.image_processor.size["height"]
IMG_W = processor.image_processor.size["width"]

TASKS = {
    "Chart QA (fine-tuned task)": "<chartqa> {q} <s_answer>",
    "Summarize chart": "<summarize_chart> <s_answer>",
    "Extract data table": "<extract_data_table> <s_answer>",
    "Open-ended QA": "<opencqa> {q} <s_answer>",
}


@torch.inference_mode()
def run_model(image: Image.Image, prompt: str, num_beams: int = 4, max_new_tokens: int = 64):
    """Single forward pass of UniChart. Returns (answer, debug dict)."""
    image = image.convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)
    decoder_input_ids = TOK(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)

    gen_kwargs = dict(
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=int(max_new_tokens),
        num_beams=int(num_beams),
        pad_token_id=TOK.pad_token_id,
        eos_token_id=TOK.eos_token_id,
        bad_words_ids=[[TOK.unk_token_id]],
        use_cache=True,
    )
    if num_beams > 1:
        gen_kwargs["early_stopping"] = True
    if not prompt.startswith("<chartqa>"):
        # long-form tasks (summary / table / open QA) tend to loop after ChartQA fine-tuning
        gen_kwargs["no_repeat_ngram_size"] = 4
        gen_kwargs["repetition_penalty"] = 1.3

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.generate(pixel_values, **gen_kwargs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - start

    raw = TOK.batch_decode(out)[0]
    cleaned = raw.replace(TOK.eos_token, "").replace(TOK.pad_token, "").replace(TOK.bos_token, "")
    answer = cleaned.split("<s_answer>", 1)[-1].strip()

    debug = {
        "prompt": prompt,
        "prompt_tokens": TOK.convert_ids_to_tokens(decoder_input_ids[0].tolist()),
        "raw_output": raw,
        "generated_tokens": int(out.shape[1] - decoder_input_ids.shape[1]),
        "input_image_size": f"{image.width}x{image.height} -> resized/padded to {IMG_W}x{IMG_H}",
        "latency_s": round(latency, 3),
        "decoding": f"beam search (beams={num_beams})" if num_beams > 1 else "greedy",
        "device": f"{DEVICE} / {str(DTYPE).replace('torch.', '')}",
    }
    return answer, debug


# --------------------------------------------------------------------------- metric
def _to_float(s: str):
    s = s.strip().lower().replace(",", "").replace("%", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return None


def relaxed_match(pred: str, target: str, tol: float = 0.05) -> bool:
    """ChartQA 'relaxed accuracy': numbers within 5%, strings exact (case-insensitive)."""
    p, t = _to_float(pred), _to_float(target)
    if p is not None and t is not None:
        return p == t if t == 0 else abs(p - t) / abs(t) <= tol
    return pred.strip().lower() == target.strip().lower()


# --------------------------------------------------------------------------- examples
def load_examples():
    meta = EXAMPLES_DIR / "examples.json"
    if not meta.exists():
        return []
    data = json.loads(meta.read_text())
    for ex in data:
        ex["path"] = str(EXAMPLES_DIR / ex["file"])
    return data


EXAMPLES = load_examples()


# --------------------------------------------------------------------------- callbacks
def ask(image, question, history, num_beams, max_new_tokens):
    history = history or []
    if image is None:
        raise gr.Error("Upload a chart image first.")
    if not question or not question.strip():
        raise gr.Error("Type a question about the chart.")
    prompt = TASKS["Chart QA (fine-tuned task)"].format(q=question.strip())
    answer, debug = run_model(image, prompt, num_beams, max_new_tokens)
    history = history + [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": f"**{answer}**  \n<sub>{debug['latency_s']}s · {debug['decoding']}</sub>"},
    ]
    return history, "", debug


def reset_chat():
    return [], None


def run_task(image, task, question, num_beams, max_new_tokens):
    if image is None:
        raise gr.Error("Upload a chart image first.")
    template = TASKS[task]
    if "{q}" in template:
        if not question or not question.strip():
            raise gr.Error("This task needs a question.")
        prompt = template.format(q=question.strip())
    else:
        prompt = template
    answer, debug = run_model(image, prompt, num_beams, max_new_tokens)

    table = None
    if task == "Extract data table" and "|" in answer:
        # UniChart serialises tables as  "col | col & cell | cell & ..."
        rows = [[c.strip() for c in r.split("|")] for r in answer.split("&") if r.strip()]
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        table = gr.update(value=rows[1:] if len(rows) > 1 else rows,
                          headers=rows[0] if len(rows) > 1 else None, visible=True)
    else:
        table = gr.update(visible=False)
    return answer, table, debug


def run_benchmark(num_beams, progress=gr.Progress()):
    if not EXAMPLES:
        raise gr.Error("No examples found. Run `python make_examples.py` first.")
    rows, correct, total_time = [], 0, 0.0
    items = [(ex, q, a) for ex in EXAMPLES for q, a in ex["qa"]]
    for ex, q, a in progress.tqdm(items, desc="Running ChartQA"):
        img = Image.open(ex["path"])
        pred, dbg = run_model(img, TASKS["Chart QA (fine-tuned task)"].format(q=q), num_beams, 32)
        ok = relaxed_match(pred, a)
        correct += ok
        total_time += dbg["latency_s"]
        rows.append([ex["file"], q, a, pred, "✅" if ok else "❌", dbg["latency_s"]])
    acc = correct / len(items) * 100
    summary = (
        f"### Relaxed accuracy: **{acc:.1f}%** ({correct}/{len(items)})\n"
        f"Average latency **{total_time / len(items):.2f}s** per question on `{DEVICE}`. "
        "Relaxed accuracy is the official ChartQA metric: numeric answers count as correct "
        "within 5% of the ground truth; text answers need an exact (case-insensitive) match."
    )
    return summary, rows


# --------------------------------------------------------------------------- UI
CSS = """
#title {text-align:center; margin-bottom: 0}
#subtitle {text-align:center; opacity:.75; margin-top:4px}
.stat {text-align:center}
footer {display:none !important}
"""

ABOUT = f"""
## How UniChart answers a question

```
chart image ──► Donut-Swin encoder ──► visual features ─┐
                (960×960, patch 4,                      │ cross-attention
                 window 10, 4 stages)                   ▼
"<chartqa> {{question}} <s_answer>" ──► mBART decoder (4 layers) ──► answer tokens ──► "</s>"
```

1. **Image preprocessing** – the chart is resized (keeping aspect ratio), padded to **{IMG_W}×{IMG_H}**
   and normalised. No OCR step: the model reads text, axes and colours straight from pixels.
2. **Encoder** – a Swin Transformer (Donut variant) turns the image into a grid of visual embeddings.
3. **Task prompt** – the decoder is primed with a task token. This checkpoint was fine-tuned on
   **ChartQA**, so the main prompt is `<chartqa> question <s_answer>`. The tokenizer also contains
   UniChart's pre-training tokens (`<summarize_chart>`, `<extract_data_table>`, `<opencqa>`); they are
   available in the *Other tasks* tab, but after ChartQA fine-tuning their output quality may be poor.
4. **Decoding** – the mBART decoder generates the answer after `<s_answer>` (beam search by default)
   until it emits `</s>`.

### Model card
| | |
|---|---|
| Architecture | `VisionEncoderDecoderModel` (Donut-Swin + mBART) |
| Parameters | {N_PARAMS / 1e6:.0f}M |
| Encoder depths / heads | 2-2-14-2 / 4-8-16-32 |
| Decoder | 4 layers, d_model 1024, vocab 57,531 |
| Running on | `{DEVICE}` · `{str(DTYPE).replace('torch.', '')}` (loaded in {LOAD_SECONDS:.1f}s) |

### Tips for good answers
- Ask **short, factual** questions: "What is the value for X?", "Which category is the highest?",
  "What is the difference between A and B?". ChartQA answers are a word or a number.
- Charts with **data labels** printed on them give much more accurate numbers.
- Crop to a single chart; large dashboards are downscaled to 960 px and fine text gets lost.
- Arithmetic (sums, averages) is the hardest part; check those answers.
"""

with gr.Blocks(title="UniChart ChartQA") as demo:
    gr.Markdown("# 📊 UniChart · Chart Question Answering", elem_id="title")
    gr.Markdown(
        f"Ask questions about any chart image. Local model `{MODEL_DIR.name}` · "
        f"{N_PARAMS / 1e6:.0f}M params · running on **{DEVICE.upper()}**",
        elem_id="subtitle",
    )
    gr.Markdown(
        "> ⚠️ **Use data charts only**: bar, line, pie, or scatter charts with axes, labels and values "
        "(like the samples below). The model can't read posters, slides, architecture "
        "diagrams, infographics or screenshots of documents. On those it gives meaningless answers. "
        "Ask short factual questions; answers are a single word or number."
    )

    with gr.Tabs():
        # ---------------------------------------------------------------- Tab 1
        with gr.Tab("💬 Ask a chart"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    chart = gr.Image(type="pil", label="Chart image", height=420)
                    question = gr.Textbox(
                        label="Question", placeholder="e.g. Which country has the highest value?",
                        lines=1, max_lines=3,
                    )
                    with gr.Row():
                        ask_btn = gr.Button("Ask", variant="primary", scale=3)
                        clear_btn = gr.Button("Clear chat", scale=1)
                    with gr.Accordion("Decoding settings", open=False):
                        beams = gr.Slider(1, 8, value=4, step=1, label="Beams (1 = greedy)")
                        max_tokens = gr.Slider(8, 256, value=64, step=8, label="Max new tokens")
                with gr.Column(scale=5):
                    chat = gr.Chatbot(label="Answers", height=420)
                    with gr.Accordion("🔍 Under the hood (last call)", open=False):
                        debug_out = gr.JSON(label="Prompt, raw decoder output, timing")

            if EXAMPLES:
                gr.Markdown("### Try a sample chart: click a row, then press **Ask**")
                gr.Examples(
                    examples=[[ex["path"], q] for ex in EXAMPLES for q, _ in ex["qa"]],
                    inputs=[chart, question],
                    examples_per_page=12,
                )

            ask_inputs = [chart, question, chat, beams, max_tokens]
            ask_btn.click(ask, ask_inputs, [chat, question, debug_out])
            question.submit(ask, ask_inputs, [chat, question, debug_out])
            clear_btn.click(reset_chat, None, [chat, debug_out])
            chart.change(reset_chat, None, [chat, debug_out])

        # ---------------------------------------------------------------- Tab 2
        with gr.Tab("🧪 Benchmark on samples"):
            gr.Markdown(
                "Runs every sample question through the model and scores it against the known "
                "ground truth of the generated charts, using the ChartQA *relaxed accuracy* metric."
            )
            if EXAMPLES:
                gr.Gallery([ex["path"] for ex in EXAMPLES], columns=4, height=220,
                           label="Sample charts (generated with known data)", allow_preview=True)
            with gr.Row():
                bench_beams = gr.Slider(1, 8, value=4, step=1, label="Beams", scale=3)
                bench_btn = gr.Button("▶ Run benchmark", variant="primary", scale=1)
            bench_summary = gr.Markdown()
            bench_table = gr.Dataframe(
                headers=["chart", "question", "expected", "model answer", "correct", "latency (s)"],
                wrap=True, interactive=False,
            )
            bench_btn.click(run_benchmark, [bench_beams], [bench_summary, bench_table])

        # ---------------------------------------------------------------- Tab 3
        with gr.Tab("🧰 Other UniChart tasks"):
            gr.Markdown(
                "UniChart was pre-trained on several chart tasks, each with its own prompt token. "
                "**This checkpoint was fine-tuned on ChartQA**, so the tasks below are "
                "*experimental*: they show what the prompt tokens do, but quality may be lower."
            )
            with gr.Row(equal_height=False):
                with gr.Column():
                    t_chart = gr.Image(type="pil", label="Chart image", height=380)
                    t_task = gr.Radio(list(TASKS), value="Summarize chart", label="Task")
                    t_question = gr.Textbox(label="Question (QA tasks only)", visible=False)
                    with gr.Row():
                        t_beams = gr.Slider(1, 8, value=4, step=1, label="Beams")
                        t_max = gr.Slider(16, 512, value=256, step=16, label="Max new tokens")
                    t_btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    t_out = gr.Textbox(label="Model output", lines=8)
                    t_table = gr.Dataframe(label="Parsed data table", visible=False, interactive=False)
                    with gr.Accordion("🔍 Under the hood", open=False):
                        t_debug = gr.JSON()
            if EXAMPLES:
                gr.Examples([[ex["path"]] for ex in EXAMPLES], inputs=[t_chart])

            t_task.change(lambda t: gr.update(visible="{q}" in TASKS[t]), t_task, t_question)
            t_btn.click(run_task, [t_chart, t_task, t_question, t_beams, t_max], [t_out, t_table, t_debug])

        # ---------------------------------------------------------------- Tab 4
        with gr.Tab("ℹ️ How it works"):
            gr.Markdown(ABOUT)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CSS,
        server_name="127.0.0.1",
        server_port=int(os.environ.get("PORT", 7860)),
    )
