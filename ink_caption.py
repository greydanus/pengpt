"""Caption pen-stroke drawings with a multimodal Claude call.

Reads a pengpt JSONL (as written by ink_scrape.py), renders each drawing
exactly as the model will learn it -- from the strokes, not the source image,
so the caption describes what survived the pipeline rather than what was in
the photo -- and asks Claude for a short literal description. Output is the
same JSONL with "text" filled in.

Resumable: lines already present in --out are skipped, so a crashed or
rate-limited run picks up where it stopped.

    python ink_caption.py --in data/scraped.jsonl --out data/scraped_captioned.jsonl

Auth comes from the environment (ANTHROPIC_API_KEY or an `ant auth login`
profile). For a large corpus, note --model claude-haiku-4-5 and the Batch API
(50% price) before scaling this loop up.
"""

import argparse
import base64
import io
import json
import os

import numpy as np

PROMPT = (
    "This is a simple black-and-white line drawing made of pen strokes. "
    "Describe it in one short literal phrase of 3 to 12 words, naming the "
    "objects and, if there are several, their spatial arrangement "
    "(e.g. 'a cat curled up next to a ball of yarn'). "
    "If it is unrecognizable, reply exactly: unrecognizable. "
    "Reply with only the phrase, lowercase, no final period."
)


def render_png(points, size=448):
    """Render (N, 3) pen points to PNG bytes, black on white."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(points, dtype=float)
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    starts = np.where(np.r_[1, points[:-1, 2] == 0])[0]
    for a, b in zip(starts, np.r_[starts[1:], len(points)]):
        seg = points[a:b][points[a:b, 2] == 1]
        if len(seg) >= 2:
            ax.plot(seg[:, 0], seg[:, 1], "k-", linewidth=1.6,
                    solid_capstyle="round")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-opus-5": (5.0, 25.0)}
SPENT = [0.0]


def caption(client, model, png_bytes):
    kwargs = {"output_config": {"effort": "low"}} if "opus" in model or "fable" in model else {}
    response = client.messages.create(
        model=model,
        max_tokens=100,
        **kwargs,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png",
                            "data": base64.standard_b64encode(png_bytes).decode()}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    p_in, p_out = PRICES.get(model, (5.0, 25.0))
    SPENT[0] += (response.usage.input_tokens * p_in
                 + response.usage.output_tokens * p_out) / 1e6
    if response.stop_reason == "refusal":
        return None
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip().strip(".").lower() or None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--in", dest="inp", required=True, help="JSONL from ink_scrape.py")
    parser.add_argument("--out", required=True, help="captioned JSONL (appended, resumable)")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--limit", type=int, default=None, help="caption at most N items")
    parser.add_argument("--budget", type=float, default=0.0,
                        help="stop when measured API spend reaches this many dollars")
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    done = 0
    if os.path.exists(args.out):
        done = sum(1 for _ in open(args.out))
        print(f"resuming after {done} already-captioned items")

    items = [json.loads(line) for line in open(args.inp)]
    todo = items[done:]
    if args.limit:
        todo = todo[:args.limit]

    with open(args.out, "a") as f:
        for i, item in enumerate(todo):
            png = render_png(item["points"])
            try:
                text = caption(client, args.model, png)
            except anthropic.APIStatusError as e:
                # SDK already retried 429/5xx; a persistent failure here should
                # stop the run rather than write an uncaptioned line and shift
                # the resume index.
                raise SystemExit(f"API error at item {done + i}: {e.message}")
            # Failures still get a line, so the resume index stays aligned with
            # the input; filter "unrecognizable" downstream before training.
            item.setdefault("meta", {})["label"] = item.get("text", "")
            item["text"] = text or "unrecognizable"
            f.write(json.dumps(item) + "\n")
            f.flush()
            print(f"[{done + i}] {item.get('meta', {}).get('source', '?')}: {item['text']}")
            if args.budget and SPENT[0] >= args.budget:
                print(f"budget reached: ${SPENT[0]:.2f} after {done + i + 1} items")
                break
    print(f"total spend ${SPENT[0]:.2f}")


if __name__ == "__main__":
    main()
