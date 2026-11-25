import glob
import json
import os
import shutil

HAND = [("graves", "Graves-style LSTM, our tokens"), ("base", "PenGPT (base)"), (None, None),
        ("no_bpe", "no BPE merges ($\\sim$2x compute)"), ("st_vocab", "ScribeTokens BPE (60x tokens, 3x params, 4x mem)"), ("no_pen_pos", "no pen position"),
        ("no_augment", "no augmentation"), ("no_writer", "no writer vectors"), ("one_word", "one word a sequence"), (None, None),
        ("small", "1.9M param. (63\\% smaller)"), ("big", "11.2M param. (114\\% larger)"), ("tenth", "a tenth of the words"), ("no_csafe", "without CSAFE")]
SKETCH = [("rnn", "LSTM decoder, our tokens"), ("base", "PenGPT (base)"), (None, None), ("no_pen_pos", "no pen position"),
          ("added_pen_pos", "added pen position only"), ("rotary_pen_pos", "rotary pen position only"), ("no_augment", "no augmentation"),
          ("no_order", "no stroke-order alterations"), ("no_guidance", "no label dropout"), ("no_hint", "no ink and stroke hints"),
          (None, None), ("labels", "label bytes, no vectors"), ("image_only", "picture vectors only"),
          ("image_label", "picture and label vectors"), (None, None), ("small", "8 layers, 384 wide"), ("tiny", "6 layers, 256 wide"),
          ("quickdraw", "Quick, Draw! only")]
MISSING = "--"
OTHER_GRID = ("polar", "st_vocab_grid", "st_model")
SEEDS = ("base", "seed", "seed2")


def signed(value, places=3):
    shown = f"{value:+.{places}f}"
    return f"${shown if float(shown) else shown[1:]}$"


def spread(cells, places=None):
    values = [float(c.replace("\\%", "").replace("%", "")) for c in cells if c not in (MISSING, "")]
    if len(values) < 2 or len(values) < len(cells):
        return cells[0]
    mean, sd = sum(values) / len(values), (sum((v - sum(values) / len(values)) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    if places is None:
        places = len(cells[0].replace("\\%", "").replace("%", "").split(".")[-1]) if "." in cells[0] else 0
    unit = "\\%" if "%" in cells[0] else ""
    shown = f"{sd:.{places}f}" if float(f"{sd:.{places}f}") else f"{{<}}{10 ** -places:.{places}f}"
    return f"{mean:.{places}f}{unit}{{\\tiny$\\,\\pm${shown}}}"


def over_seeds(row_of, places={}):
    rows = [r for r in (row_of(name) for name in SEEDS) if r]
    return [spread([r[k] for r in rows], places.get(k)) if rows[0][k] != "" else "" for k in range(len(rows[0]))]


def metrics(run):
    path = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path)]
    return min(rows, key=lambda r: r["test_loss"]), rows[-1]


def rows_of(pattern):
    out = {}
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("|") and len(cells) > 2 and not set(cells[1]) <= set("-") and cells[0] not in ("Run", "Guidance"):
                if not cells[0].startswith("real ink") or "(7 sources)" in cells[0]:
                    out[cells[0].split(" (")[0]] = cells[1:]
    return out


def finished(run, steps):
    found = metrics(run)
    return found if found and found[1]["step"] >= steps else None


def mean_by(out, steps):
    found = [f[0]["word_score_by"] for f in (finished(os.path.join(out, name), steps) for name in SEEDS) if f]
    return {k: sum(f[k] for f in found) / len(found) for k in found[0]}


def compact(cells):
    if cells is None:
        return None
    number = lambda c: float(c.replace("\\%", "").replace("%", ""))
    rank = cells[4] if cells[4] in (MISSING, "") else f"{number(cells[4]):.1f}"
    exact = cells[8] if cells[8] in (MISSING, "") else f"{number(cells[8]):.0f}\\%"
    return [cells[0], cells[1], cells[3], rank, exact]


def wide(cells):
    return cells and [*cells[:6], cells[7], cells[8]]


FULL_ONLY = [("polar", "polar tokens (Cursive Transformer)"), ("st_vocab_grid", "ScribeTokens BPE and grid")]


def hand(out="paper/results/ablate_hand", full=False):
    alias = {"graves": "graves2"}
    base = mean_by(out, 60_000)
    three = (rows_of(f"{out}/transfer_wave*.md") | rows_of(f"{out}/transfer_big.md") | rows_of(f"{out}/transfer_late*.md") | rows_of(f"{out}/transfer_seed*.md")
             | rows_of(f"{out}/transfer_st*.md"))
    one, read = rows_of(f"{out}/transfer_prime1*.md"), rows_of(f"{out}/legible_wave*.md") | rows_of(f"{out}/legible_200*.md")
    sampled, most = rows_of(f"{out}/legible_sampled200_*.md"), rows_of(f"{out}/legible_most_*.md")

    def row(name):
        run = alias.get(name, name)
        found = finished(os.path.join(out, run), 30_000 if name == "graves" else 60_000) or name.startswith("st_") and metrics(os.path.join(out, run))
        cells = [MISSING] * 9
        if not found and name in SEEDS[1:]:
            return None
        if found:
            best = found[0]
            shared = [s for s in best["word_score_by"] if s in base]
            delta = sum(best["word_score_by"][s] - base[s] for s in shared) / len(shared)
            unseen = [sum(u) / len(u) for u in zip(*best["unseen_by"].values())] if best.get("unseen_by") else []
            mark = "$^\\dagger$" if name in OTHER_GRID else ""
            cells[0] = f"{best['word_score']:.3f}" + mark
            cells[1] = "" if name in SEEDS else MISSING if name in OTHER_GRID else signed(delta)
            if unseen:
                cells[2], cells[3] = f"{unseen[0]:.3f}" + mark * (name != "polar"), f"{unseen[-1]:.3f}" + mark * (name != "polar")
        if run in three and not three[run][5].endswith("(1)"):
            cells[4] = three[run][3]
        if run in one:
            cells[5] = one[run][3]
        if run in read:
            cells[6], cells[7] = read[run][0], read[run][1].replace("%", "\\%")
        if run in sampled:
            cells[8] = sampled[run][2].replace("%", "\\%")
        if run in most:
            cells[7] = most[run][2].replace("%", "\\%")
        return cells

    lines = []
    for name, label in (HAND[:1] + FULL_ONLY + HAND[1:] if full else HAND):
        if name is None:
            lines.append("\\midrule")
            continue
        label = label.replace(", 3x params, 4x mem", "") if full else label
        shape = (lambda n: wide(row(n))) if full else (lambda n: compact(row(n)))
        cells = over_seeds(row, {} if full else {4: 1, 8: 0}) if name == "base" else shape(name)
        if name == "base":
            cells = wide(cells) if full else [cells[0], cells[1], cells[3], cells[4], cells[8]]
        if name == "base" and not full:
            label, cells = f"\\textbf{{{label}}}", [f"\\textbf{{{c}}}" if c else c for c in cells]
        if name.startswith("st_") and not finished(os.path.join(out, name), 60_000):
            label, cells = f"\\textcolor{{red}}{{{label}}}", [f"\\textcolor{{red}}{{{c}}}" if c else c for c in cells]
        lines.append(" & ".join([label, *cells]) + " \\\\")
    if "real ink" in read:
        ink = read["real ink"][1].replace("%", "\\%")
        lines += ["\\midrule", " & ".join(["real held-out ink", *[""] * 6, ink, ink] if full else ["real held-out ink", *[""] * 4, compact([""] * 8 + [ink])[4]]) + " \\\\"]
    return "\n".join(lines) + "\n"


def best_recurrent(out, steps=27_000):
    done = [(m[0]["word_score"], os.path.basename(run)) for run in sorted(glob.glob(f"{out}/rnn_lr*"))
            for m in [metrics(run)] if m and m[1]["step"] >= steps and os.path.exists(f"{run}/score_phrases/results.md")]
    return min(done)[1] if done else "rnn"


RECURRENT = {"rnn": best_recurrent}


def final(results="paper/results"):
    def score(run):
        found = metrics(f"{results}/final/{run}")
        return f"{found[0]['word_score']:.3f}" if found else MISSING

    read = {k: v[1:] for k, v in rows_of(f"{results}/ablate_hand/legible_sampled200_*.md").items()}
    packed = json.load(open(f"{results}/final/packed.json")) if os.path.exists(f"{results}/final/packed.json") else {}
    values = dict(FinalHandScore=score("hand3") if score("hand3") != MISSING else score("hand"), FinalSketchScore=score("released_sketch"), LongerSketchScore=score("sketch"),
                  PolarSampled=read["polar"][1].replace("%", "\\%") if "polar" in read else MISSING,
                  BaseSampled=read["base"][1].replace("%", "\\%") if "base" in read else MISSING,
                  FinalHandSize=str(packed.get("hand_mb", MISSING)), FinalSketchSize=str(packed.get("sketch_mb", MISSING)),
                  FinalHandCost=str(packed.get("hand_cost", MISSING)), FinalSketchCost=str(packed.get("sketch_cost", MISSING)),
                  RecurrentRun=best_recurrent(f"{results}/ablate_sketch").replace("rnn_lr", "").replace("rnn", MISSING))
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items())


def sketch(out="paper/results/ablate_sketch"):
    found = [m[0]["word_score_by"] for m in (metrics(os.path.join(out, name)) for name in SEEDS) if m]
    base = {k: sum(f[k] for f in found) / len(found) for k in found[0]} if found else None

    def row(name):
        name = RECURRENT.get(name, lambda out: name)(out)
        found, cells = metrics(os.path.join(out, name)), [MISSING] * 5
        if not found and name in SEEDS[1:]:
            return None
        if found:
            best = found[0]
            cells[0], cells[2] = f"{best['word_score']:.3f}", f"{best['test_loss']:.3f}"
            if base and name not in SEEDS:
                shared = [s for s in best["word_score_by"] if s in base]
                cells[1] = signed(sum(best['word_score_by'][s] - base[s] for s in shared) / len(shared))
            cells[1] = "" if name in SEEDS else cells[1]
        for k, kind in enumerate(("labels", "phrases")):
            got = rows_of(f"{out}/{name}/score_{kind}/results.md").get("1" if name == "no_guidance" else "2")
            if got:
                cells[3 + k] = got[1].replace("%", "\\%") + ("$^\\ddagger$" if name == "no_guidance" else "")
        return cells

    lines = []
    for name, label in SKETCH:
        if name is None:
            lines.append("\\midrule")
            continue
        cells = over_seeds(row) if name == "base" else row(name)
        if name == "base":
            label, cells = f"\\textbf{{{label}}}", [f"\\textbf{{{c}}}" if c else c for c in cells]
        lines.append(" & ".join([label, *cells]) + " \\\\")
    return "\n".join(lines) + "\n"


def guidance(path="paper/results/guidance/results.md"):
    rows = rows_of(path)
    return "\n".join(" & ".join([g, r[0], r[1].replace("%", "\\%"), r[2].replace("%", "\\%"), r[3]]) + " \\\\" for g, r in rows.items()) + "\n"


def keep(src="out", dst="paper/results"):
    for path in glob.glob(f"{src}/ablate_*/*.md") + glob.glob(f"{src}/ablate_*/*/score_*/results.md") + glob.glob(f"{src}/guidance/results.md"):
        os.makedirs(os.path.dirname(dst + path[len(src):]), exist_ok=True)
        shutil.copyfile(path, dst + path[len(src):])
    for path in glob.glob(f"{src}/final/*.json"):
        os.makedirs(f"{dst}/final", exist_ok=True)
        shutil.copyfile(path, dst + path[len(src):])
    for path in glob.glob(f"{src}/ablate_*/*/metrics.jsonl") + glob.glob(f"{src}/final/*/metrics.jsonl"):
        best, last = metrics(os.path.dirname(path))
        os.makedirs(os.path.dirname(dst + path[len(src):]), exist_ok=True)
        with open(dst + path[len(src):], "w") as f:
            f.write("".join(json.dumps(row) + "\n" for row in ([best] if best is last else [best, last])))


def main():
    if os.path.isdir("out/ablate_hand"):
        keep()
    os.makedirs("paper/tables", exist_ok=True)
    for name, text in (("hand", hand()), ("hand_full", hand(full=True)), ("sketch", sketch()), ("guidance", guidance())):
        with open(f"paper/tables/{name}.tex", "w") as f:
            f.write(text + "\\bottomrule\n")
        print(f"paper/tables/{name}.tex\n{text}")
    with open("paper/tables/final.tex", "w") as f:
        f.write(final())
    with open("paper/tables/ablations.tex", "w") as f:
        f.write(hand() + "\\bottomrule\n\\toprule\n\\textbf{Drawings} & Score & $\\Delta$ & Loss & Labels & Captions \\\\\n\\midrule\n"
                + sketch() + "\\bottomrule\n")


if __name__ == "__main__":
    main()
