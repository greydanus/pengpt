import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import pengpt as P
from strokes import to_strokes

TEMPERATURES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
CACHE = "paper/figures/unseen_writers.pkl"


def measures(points, texts):
    lifts, turn = [], []
    for p, text in zip(points, texts):
        strokes = [s for s in to_strokes(p) if len(s) > 2]
        letters = sum(ch.isalnum() for ch in text)
        if not strokes or not letters:
            continue
        lifts.append(len(strokes) / letters)
        for s in strokes:
            step = np.diff(s[:, :2], axis=0)
            angle = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))
            turn.append(np.abs(np.diff(angle, 2)).mean() if len(angle) > 2 else 0)
    return float(np.mean(lifts)), float(np.mean(turn))


def writers(cfg, ckpt):
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    test = P.create_datasets(cfg, merges=ckpt["merges"], sources=ckpt["sources"])[1]
    found = []
    for source, d in test.unseen.items():
        for g in d.groups:
            if len(g) >= 4:
                points, texts = [d.bank_points[i] for i in g], [d.bank_texts[i] for i in g]
                found.append(dict(source=source, points=points, texts=texts, measures=measures(points, texts)))
    found.sort(key=lambda w: -w["measures"][0])
    pickle.dump(found, open(CACHE, "wb"))
    return found


def primer(st, writer, rng, budget=105, most=3):
    picked, used = [], 0
    for i in sorted(rng.permutation(len(writer["texts"])), key=lambda i: -min(len(writer["texts"][i]), 5)):
        tokens = st.encode_word(writer["points"][i])
        if writer["texts"][i].isalpha() and used + len(tokens) + 2 <= budget and len(picked) < most:
            picked.append((writer["texts"][i], tokens))
            used += len(tokens) + 2
    return picked


def write(model, dataset, cfg, before, words, temperature, each, seed):
    st, device = dataset.stroke_tok, next(model.parameters()).device
    seq = [st.BOS] + [t for _, tokens in before for t in [*tokens, st.WORD, st.WORD]]
    texts = [" ".join([w for w, _ in before] + [word]) for word in words for _ in range(each)]
    context = torch.from_numpy(np.stack([dataset.encode_text(t) for t in texts])).to(device)
    idx = torch.tensor([seq] * len(texts), device=device)
    out = model.generate(idx, context, max_new_tokens=cfg.max_seq_length - len(seq), temperature=temperature or 1.0, do_sample=temperature > 0,
                         end_token=st.END, pad_token=st.PAD, source=torch.zeros(len(texts), dtype=torch.long, device=device),
                         guidance=2.0 if cfg.label_drop > 0 else 1.0, blank=torch.from_numpy(dataset.encode_text(""))[None],
                         generator=torch.Generator(device).manual_seed(seed))[:, len(seq):].cpu().numpy()
    decoded = []
    for row in out:
        stop = np.flatnonzero((row == st.END) | (row == st.WORD) | (row == st.PAD))
        decoded.append(st.decode_word(row[:stop[0]] if stop.size else row).tolist())
    return [decoded[k * each:(k + 1) * each] for k in range(len(words))]


def main(picks, out="paper/figures/unseen_hands.json", checkpoint="out/ablate_hand/base/best.pt", text="Written by PenGPT", each=4):
    device = P.resolve_device("auto")
    model, dataset, cfg, ckpt = P.load_for_sampling(checkpoint, device)
    known, cells = writers(cfg, ckpt), []
    for k, (j, seed) in enumerate(picks):
        writer, t = known[j], TEMPERATURES[k % len(TEMPERATURES)]
        before = primer(dataset.stroke_tok, writer, np.random.default_rng(seed))
        words = write(model, dataset, cfg, before, text.split(), t, 1 if t == 0 else each, seed)
        cells.append(dict(writer=j, source=writer["source"], temperature=t, primed=[w for w, _ in before], words=words,
                          real=[p.tolist() for p in writer["points"][:3]]))
        print(k, j, writer["source"], t, [w for w, _ in before], flush=True)
    json.dump(cells, open(out, "w"))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]), *sys.argv[2:3])
