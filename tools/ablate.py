import argparse
import glob
import json
import os
import shlex
import subprocess
import sys

sys.path.insert(0, ".")
import utils


def files(kind, without=()):
    kept = [p for p in utils.EVERY[kind].split(",") if p and os.path.basename(p).split(".")[0] not in without]
    return ",".join(kept) or None


def ablations(model):
    if model == "hand":
        return {
            "base": "",
            "seed": "--train_seed 1",
            "graves": "--arch graves --n_layer 3 --n_embd 400 --learning_rate 1e-3 --weight_decay 0 --no-compile --no-mixed",
            "polar": "--tokenizer polar --max_seq_length 1024 --pen_pos_bands 0",
            "no_pen_pos": "--pen_pos_bands 0",
            "no_bpe": "--n_merges 0 --max_seq_length 512",
            "no_augment": "--augment 0",
            "no_writer": "--source_dropout 1.0",
            "one_word": "--max_words 1",
            "small": "--n_layer 6 --n_embd 128",
            "big": "--n_layer 10 --n_embd 256",
            "tenth": "--max_train 10000",
            "real_strokes": "--dataset data/bigbank.hand.parquet,data/notes.hand.parquet",
            "no_csafe": files("hand", ["csafe"]) and f"--dataset {files('hand', ['csafe'])}",
        }
    return {
        "base": "",
        "seed": "--train_seed 1",
        "no_pen_pos": "--pen_pos_bands 0 --no-pen_rotary",
        "added_pen_pos": "--no-pen_rotary",
        "rotary_pen_pos": "--pen_pos_bands 0",
        "no_augment": "--augment 0",
        "no_order": "--stroke_swap 0 --stroke_reverse 0 --loop_start 0",
        "no_guidance": "--label_drop 0",
        "no_hint": "--hint 0",
        "labels": "--vectors=",
        "image_only": "--vec_columns image",
        "image_label": "--vec_columns image,label",
        "small": "--n_layer 8 --n_embd 384 --n_head 6",
        "tiny": "--n_layer 6 --n_embd 256 --n_head 4",
        "quickdraw": "--dataset data/quickdraw.sketch.parquet",
    }


def plan(model, profile, only, extra, preset=""):
    recipe = {**utils.recipe(profile, model), **({"preset": preset} if preset else {})}
    head = f"--preset {recipe['preset']} --profile {profile} --model {model}"
    head += f" --gpus {recipe['gpus']}" if recipe.get("gpus", 1) > 1 else ""
    runs = {k: v for k, v in ablations(model).items() if not only or k in only}
    for name in [k for k, v in runs.items() if v is None]:
        print(f"skipping {name}: its data files are not in data/")
    return [(name, " ".join(f"{head} {args} {extra}".split())) for name, args in runs.items() if args is not None]


def read(run):
    path = os.path.join(run, "metrics.jsonl")
    rows = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    return (min(rows, key=lambda r: r["test_loss"]), rows[-1]) if rows else None


def table(out):
    runs = {os.path.basename(d.rstrip("/")): read(d) for d in sorted(glob.glob(os.path.join(out, "*/")))}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        raise SystemExit(f"no metrics.jsonl under {out}")
    base = runs.get("base", (None, None))[0]
    sources = sorted({s for best, _ in runs.values() for s in best["word_score_by"]})
    lines = ["| Run | Step | Word score | Against base | Text rank | Unseen alone / after 1 / after 3 | Last step score | Minutes | "
             + " | ".join(sources) + " |", "|" + "---|" * (8 + len(sources))]
    for name in sorted(runs, key=lambda k: (k != "base", runs[k][0]["word_score"])):
        best, last = runs[name]
        by = best["word_score_by"]
        shared = [s for s in by if base and s in base["word_score_by"]]
        delta = f"{sum(by[s] - base['word_score_by'][s] for s in shared) / len(shared):+.4f}" if shared and name != "base" else ""
        unseen = list(zip(*best.get("unseen_by", {}).values()))
        lines.append(f"| {name} | {best['step']} | {best['word_score']:.4f} | {delta} | {best['text_rank']:.2f} | "
                     + " / ".join(f"{sum(u) / len(u):.4f}" for u in unseen) + f" | {last['word_score']:.4f} | {last['minutes']:.0f} | "
                     + " | ".join(f"{by[s]:.4f}" if s in by else "" for s in sources) + " |")
    text = "\n".join(lines) + "\n\nWord score is at the step with the lowest held-out loss, in nats per pen move; lower is better. " \
           "Against base is the mean difference over the sources both runs hold out. The seed row is the noise to read the rest by.\n"
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    p = argparse.ArgumentParser(description="ablations of either model: write plans for tools/train_cloud.sh, run them here, and tabulate what they scored")
    p.add_argument("command", choices=["plan", "run", "table", "list"])
    p.add_argument("model", choices=["hand", "sketch"])
    p.add_argument("--profile", type=str, default="8xa100")
    p.add_argument("--preset", type=str, default="", help="train this preset in place of the profile's")
    p.add_argument("--nodes", type=int, default=1, help="plan: split the runs over this many nodes")
    p.add_argument("--only", type=str, default="", help="comma-separated names")
    p.add_argument("--extra", type=str, default="", help="arguments added to every run")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    out = args.out or f"out/ablate_{args.model}"
    runs = plan(args.model, args.profile, [k for k in args.only.split(",") if k], args.extra, args.preset)
    if args.command == "list":
        for name, line in runs:
            print(f"{name:14s}{line}")
    if args.command == "plan":
        os.makedirs(out, exist_ok=True)
        for k in range(args.nodes):
            path = os.path.join(out, f"plan_{k + 1}.txt")
            with open(path, "w") as f:
                f.writelines(f"{name} {line}\n" for name, line in runs[k::args.nodes])
            print(f"PLAN={path} OUT={out} NAME=ablate-{args.model}-{k + 1} tools/train_cloud.sh")
    if args.command == "run":
        for name, line in runs:
            where = os.path.join(out, name)
            if read(where):
                print(f"{name}: already has metrics, skipping")
                continue
            os.makedirs(where, exist_ok=True)
            print(f"{name}: {line}")
            with open(os.path.join(where, "train.log"), "w") as log:
                if subprocess.run([sys.executable, "-u", "pengpt.py", "train", "--out_dir", where, *shlex.split(line)], stdout=log,
                                  stderr=subprocess.STDOUT).returncode:
                    print(f"  failed; see {where}/train.log")
    if args.command in ("run", "table"):
        table(out)


if __name__ == "__main__":
    main()
