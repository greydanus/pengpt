import argparse, contextlib, glob, io, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def prompts(args):
    import vectors as V
    from strokes import StrokeFile, held_out
    rng, phrases, img, txt = np.random.default_rng(args.seed), [], [], []
    for path in sorted(glob.glob("data/*.sketch.parquet")):
        f, cols = StrokeFile(path), V.columns(path)
        held, short = np.flatnonzero(held_out(f, 50)), f.column("short")
        phrases += [short[i] for i in rng.permutation(held) if short[i]][:args.per_source]
        rows = np.sort(rng.choice(np.flatnonzero(~held_out(f, 50)), 300, replace=False))
        img += [cols["image"][i] for i in rows]
        txt += [cols[c][i] for i in rows for c in ("label", "short", "frame") if cols[c].there(i)]
    phrases = list(dict.fromkeys(phrases))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "phrases.txt"), "w") as f:
        f.write("\n".join(phrases) + "\n")
    shift = (np.mean(img, 0) - np.mean(txt, 0)).astype(np.float32)
    np.save(os.path.join(args.out, "shift.npy"), shift)
    print(f"{len(phrases)} held-out phrases and a shift of length {np.linalg.norm(shift):.3f} -> {args.out}")


def told(args):
    import torch
    import pengpt as P
    device = P.resolve_device("auto")
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, cfg, ck = P.load_for_sampling(args.checkpoint, device)
        test = P.create_datasets(cfg, merges=ck["merges"], sources=ck["sources"])[1]
    model.eval()
    st, full, n = test.stroke_tok, test.told.columns, args.per_source
    rows = np.concatenate([d.indices[:n] for d in test.parts.values()])
    part = np.repeat(list(test.parts), [len(d.indices[:n]) for d in test.parts.values()])
    tokens = [st.encode_word(test.bank_points[i]) for i in rows]
    moves = np.array([st.moves(test.bank_points[i]) for i in rows])

    def score(context_of):
        items = [test.make_item(t, context_of(i), int(test.source_ids[i])) for t, i in zip(tokens, rows)]
        nats = np.zeros(len(rows))
        with torch.inference_mode():
            for k in range(0, len(rows), 50):
                X, C, Y, W = (torch.stack(t).to(device) for t in zip(*items[k:k + 50]))
                logits, _ = model(X, C, source=W)
                loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), Y, ignore_index=-1, reduction="none")
                nats[k:k + 50] = loss.sum(1).cpu().numpy()
        return nats

    scores = {"nothing": score(lambda i: test.encode_text(""))}
    for c in full:
        if full[c].shape[1:] == (1152,):
            test.told.columns = {c: full[c], "image": full["image"]}
            scores[c] = score(lambda i: test.encode_text(test.told.of(i, "")))
    test.told.columns = full
    print(f"nats a pen move on {len(rows)} held-out drawings ({n} a source), the model told one vector\n")
    print("| Told | All | " + " | ".join(test.parts) + " |\n|---|---|" + "---|" * len(test.parts))
    for c, v in sorted(scores.items(), key=lambda kv: kv[1].sum()):
        each = " | ".join(f"{v[part == s].sum() / moves[part == s].sum():.3f}" for s in test.parts)
        print(f"| {c} | {v.sum() / moves.sum():.4f} | {each} |")

    def unit(a):
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    img = np.stack([full["image"][i] for i in rows]).astype(np.float32)
    print("\n| Text | Cosine to its own picture | To another picture | After centring each side | Own text nearest |")
    print("|---|---|---|---|---|")
    for c in ("label", "frame", "phrase", "short", "dense", "sentence"):
        text = np.stack([full[c][i] for i in rows]).astype(np.float32)
        sim, centred = img @ text.T, np.sum(unit(img - img.mean(0)) * unit(text - text.mean(0)), 1).mean()
        other = (sim.sum() - np.trace(sim)) / (sim.size - len(sim))
        nearest = np.mean(sim.argmax(1) == np.arange(len(sim)))
        print(f"| {c} | {np.trace(sim) / len(sim):.3f} | {other:.3f} | {centred:.3f} | {nearest:.1%} |")
    label = np.stack([full["label"][i] for i in rows]).astype(np.float32).mean(0)
    pictures = ((img @ img.T).sum() - len(img)) / (len(img) ** 2 - len(img))
    mean = img.mean(0)
    print(f"\ncosine of one picture to another {pictures:.3f}; length of the mean picture {np.linalg.norm(mean):.3f}; "
          f"cosine of the mean picture to the mean label {float(mean @ label) / np.linalg.norm(mean) / np.linalg.norm(label):.3f}")


def table(args):
    kinds = [("score_labels", "labels"), ("score_phrases", "phrases"), ("score_labels_shift", "labels, shifted"),
             ("score_phrases_shift", "phrases, shifted")]
    print("| Run | Prompts | Guidance | Cosine to its prompt | Its prompt ranked first | In the first five | Strokes |")
    print("|---|---|---|---|---|---|---|")
    for run in args.runs.split(",") if args.runs else sorted(os.listdir(args.out)):
        for folder, name in kinds:
            path = os.path.join(args.out, run, folder, "results.md")
            if os.path.isfile(path):
                for line in open(path):
                    if line[:2] == "| " and line[2].isdigit():
                        print(f"| {run} | {name} {line.strip()}")


def main():
    p = argparse.ArgumentParser(description="what the caption vectors buy a drawing model")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prompts", help="held-out caption phrases to type, and the mean picture vector less the mean text vector")
    a.add_argument("--per_source", type=int, default=30)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="out/ablate_sketch")
    a = sub.add_parser("told", help="held-out drawings scored with the model told each vector column alone")
    a.add_argument("--checkpoint", default="sketch.pt")
    a.add_argument("--per_source", type=int, default=100)
    a = sub.add_parser("table", help="gather the results.md files tools/guidance.py left under each run")
    a.add_argument("--out", default="out/ablate_sketch")
    a.add_argument("--runs", default="")
    args = p.parse_args()
    {"prompts": prompts, "told": told, "table": table}[args.cmd](args)


if __name__ == "__main__":
    main()
