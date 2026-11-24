import argparse
import collections
import json
import math
import os
import random
import re
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))

SMALL = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

ANIMALS = """cat dog rabbit horse cow pig sheep goat duck chicken rooster goose swan owl eagle parrot penguin flamingo pigeon sparrow crow
bird fish shark whale dolphin octopus squid crab lobster jellyfish seahorse starfish turtle tortoise frog toad snake lizard crocodile
alligator dinosaur dragon lion tiger leopard cheetah bear panda koala kangaroo monkey gorilla elephant giraffe zebra rhino hippo camel
llama deer moose reindeer fox wolf squirrel hedgehog mouse rat hamster bat raccoon skunk beaver otter seal walrus snail spider ant bee
butterfly dragonfly ladybug beetle mosquito fly grasshopper caterpillar worm scorpion unicorn pegasus mermaid kitten puppy pony bull
donkey peacock turkey ostrich pelican toucan hummingbird woodpecker stingray eel goldfish sloth armadillo chameleon gecko iguana bison
buffalo shrimp moth wasp centipede""".split()
THINGS = """house castle church barn lighthouse windmill tent igloo skyscraper bridge tower pyramid cabin hut garage factory school
hospital temple cottage fence gate door window chimney roof stairs ladder bench chair table sofa bed desk lamp clock mirror shelf
bookcase drawer cupboard bathtub toilet sink stove oven fridge microwave toaster kettle teapot cup mug glass bottle jar bowl plate
fork knife spoon pan pot basket bucket box bag backpack suitcase umbrella key lock hammer saw axe screwdriver wrench pliers drill
shovel rake broom scissors needle pencil pen brush crayon eraser ruler book notebook envelope letter map globe telescope microscope
camera phone telephone computer laptop keyboard television radio headphones speaker microphone guitar violin piano drum trumpet
saxophone flute harp bell car truck bus van taxi train tram bicycle motorcycle scooter skateboard tractor bulldozer ambulance
firetruck airplane helicopter rocket spaceship balloon blimp kite boat sailboat ship canoe submarine anchor wheel tire engine
tree flower rose tulip sunflower daisy cactus mushroom leaf bush grass forest mountain volcano hill island river lake waterfall beach
cloud sun moon star rainbow lightning snowflake snowman fire candle lantern lightbulb flashlight apple banana pear orange lemon
strawberry cherry grapes watermelon pineapple peach carrot broccoli corn potato tomato onion pepper pumpkin egg bread cake cupcake
cookie donut pizza hamburger sandwich taco lollipop candy cheese hat cap crown helmet glasses sunglasses scarf tie shirt t-shirt
jacket coat dress skirt pants shorts sock shoe boot glove mitten belt ring necklace watch purse wallet button zipper ball football
basketball baseball trophy medal flag dice puzzle robot doll toy yo-yo swing slide seesaw trampoline face eye nose mouth ear hand
foot arm leg finger tooth heart brain skull bone skeleton person man woman boy girl baby king queen knight wizard witch pirate clown
angel ghost alien astronaut chef doctor farmer firefighter cowboy ninja sword shield bow arrow cannon bomb spear stamp coin diamond
gem crystal magnet battery plug socket fan heater thermometer scale mailbox well fountain statue gravestone feather shell acorn
pinecone log stump rock pebble nest web cocoon horseshoe saddle wheelbarrow cart wagon sled sleigh carriage stroller wheelchair
crutch cane paintbrush palette easel frame vase pillow blanket rug curtain towel soap toothbrush comb hairbrush razor calculator
stapler paperclip tape pushpin calendar compass protractor hourglass sundial pendulum gear spring pulley lever screw nail bolt nut
chain rope hook knot net cage trap""".split()
THINGS += """polar bear,palm tree,pine tree,hot dog,ice cream,bow tie,tennis racket,golf club,hockey stick,chess piece,teddy bear,
police officer,dollar bill,treasure chest,stop sign,traffic light,street lamp,fire hydrant,parking meter,bus stop,paw print,trash can,
recycling bin""".replace("\n", "").split(",")
PARTS = """a face,legs,wings,wheels,a tail,horns,antlers,ears,cat ears,rabbit ears,big eyes,one eye,three eyes,a long neck,a beak,claws,fins,
spots,stripes,a hat,a top hat,a crown,glasses,a bow tie,a scarf,a moustache,a beard,a smile,teeth,arms,hands,feet,a door,windows,a chimney,
a flag,a handle,a lid,a spout,a sail,a propeller,an engine,a ladder,stairs,a roof,a tower,flowers,leaves,branches,roots,thorns,
a long tail,a short tail,a curly tail,big ears,small ears,long legs,short legs,four legs,six legs,eight legs,two heads,a shell,
a mane,whiskers,feathers,scales,fur,a trunk,tusks,a hump,a pouch,a stem,seeds,a core,a string,a ribbon,a bow,buttons,pockets,
a zipper,laces,a buckle,a house on its back,a castle on its back,a sailboat on its back,a bird on its head,a star on top""".replace("\n", "").split(",")
WEAR = """a top hat,a hat,a crown,glasses,sunglasses,a scarf,a bow tie,a tie,a cape,boots,a dress,a sweater,a helmet,a backpack,
a party hat,a cowboy hat,a wizard hat,a witch hat,a pirate hat,headphones,a necklace,mittens,a coat,a raincoat,socks,a tutu,armor""".replace("\n", "").split(",")
ADJ = """small big large tiny huge tall short long fat thin round square flat pointy curly fluffy furry spiky striped spotted old new broken
happy sad angry sleepy scared surprised cute scary funny silly fancy plain simple detailed cartoon realistic little giant baby young
wooden metal stone paper glass empty full open closed upside-down tilted bent twisted melting burning flying floating running walking
sitting standing sleeping jumping dancing swimming smiling crying laughing eating""".split()
ACT = """running,jumping,sitting,standing,sleeping,flying,swimming,walking,dancing,eating,drinking,climbing,crawling,hopping,rolling,
lying down,stretching,yawning,waving,smiling,crying,laughing,singing,reading,fishing,sailing,skating,skiing,surfing,playing,
kicking a ball,chasing its tail,looking up,looking back,curled up,hanging upside down""".replace("\n", "").split(",")
REL = """on,in,under,next to,behind,in front of,above,below,inside,on top of,holding,riding,carrying,eating,chasing,looking at,pulling,
pushing,beside,between two,near,over""".replace("\n", "").split(",")
PLACE = """in a field,on a hill,in the rain,in the snow,on the beach,at night,in a forest,in a garden,on a table,in a box,in a cage,
on a plate,in a bowl,on a shelf,in the sky,on the moon,in space,under a tree,by a river,on a road,in a city,at sea,under water,in a cup,
on a leash""".replace("\n", "").split(",")
VIEW = """side view of,front view of,back view of,top view of,three-quarter view of,close-up of,profile of,rear view of,bird's-eye view of,
front-left three-quarter view of,overhead view of""".replace("\n", "").split(",")
SEEN = """seen from the side,seen from the front,seen from above,seen from behind,seen from below,in profile,facing left,facing right,
facing forward,facing away,upside down,from the side,from the front,from above""".replace("\n", "").split(",")
MANNER = """loose sketch,rough sketch,quick doodle,scribbly doodle,clean outline,bare outline,simple line drawing,childlike drawing,cartoon,
neat line drawing,geometric icon,clean geometric icon,wobbly doodle,detailed drawing,densely detailed sketch,schematic diagram,
lifelike sketch,naturalistic drawing,caricature,angular doodle,sparse outline,careful drawing,messy scribble,icon,diagram,
loose scribbly doodle,minimal line art,contour drawing,stick figure,rough outline""".replace("\n", "").split(",")
ASK = """draw,sketch,draw me,please draw,can you draw,can you do,i want,show me,make,doodle,a drawing of,a sketch of,sketch of,drawing of,
a doodle of,doodle of,a simple sketch of,simple sketch of,simple drawing of,a line drawing of,line drawing of,an outline of,outline of,
a picture of,picture of,a cartoon of,cartoon of,an icon of,icon of,a quick sketch of,a rough sketch of,a detailed drawing of,
how about,give me,i'd like,let's see,one,just""".replace("\n", "").split(",")
KIND = """animal,bird,fish,insect,vehicle,building,tool,fruit,vegetable,plant,flower,tree,person,creature,machine,instrument,toy,
piece of furniture,piece of clothing,container,weapon,food,shape,object,thing,sea creature,farm animal,wild animal,pet,monster,
musical instrument,kitchen tool,garden tool,small boat,big cat,small house,tall building,flying machine""".replace("\n", "").split(",")
VAGUE = "some kind of,a sort of,something like,roughly,one of those,any,a kind of,some sort of,maybe".split(",")
COUNT = "two,three,four,five,six,several,many,a pair of,a group of,a row of,a pile of,a bunch of,lots of".split(",")
MATERIAL = "wood,stone,paper,glass,metal,ice,clouds,bricks,flowers,leaves,cheese,candy,string,blocks,triangles,circles,squares,dots,lines".split(",")


def a(word):
    return ("an " if word[0] in "aeiou" else "a ") + word


def plural(word):
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word[:-1] + "ies" if word.endswith("y") and word[-2] not in "aeiou" else word + "s"


def lines(path):
    return [line.strip() for line in open(path, encoding="utf-8") if line.strip()] if path and os.path.exists(path) else []


def written(n, nouns, words, seed=0):
    R = random.Random(seed)
    noun, animal, of, some = lambda: R.choice(nouns), lambda: R.choice(ANIMALS), R.choice, R.sample
    kinds = [
        (10, lambda: of([noun(), a(noun()), a(noun()), "the " + noun(), plural(noun()), f"{of(COUNT)} {plural(noun())}"])),
        (8, lambda: a(f"{of(ADJ)} {noun()}")),
        (3, lambda: a(f"{of(ADJ)} {of(ADJ)} {noun()}")),
        (10, lambda: f"{a(noun())} with {of(PARTS)}"),
        (3, lambda: f"{a(noun())} with {of(PARTS)} and {of(PARTS)}"),
        (5, lambda: f"{a(of(ANIMALS + nouns))} wearing {of(WEAR)}"),
        (8, lambda: f"{a(noun())} {of(REL)} {a(noun())}"),
        (5, lambda: f"{a(animal())} {of(ACT)}"),
        (3, lambda: f"{a(noun())} {of(PLACE)}"),
        (5, lambda: f"{of(VIEW)} {a(noun())}"),
        (5, lambda: f"{a(noun())} {of(SEEN)}"),
        (3, lambda: f"{a(noun())}, {of(SEEN)}, {of(ACT)}"),
        (9, lambda: f"{of(ASK)} {a(noun())}"),
        (4, lambda: f"{of(ASK)} {a(noun())} with {of(PARTS)}"),
        (3, lambda: f"{of(ASK)} {a(animal())} {of(ACT)}, {of(SEEN)}"),
        (3, lambda: of([a(of(MANNER)), of(MANNER), f"{a(of(MANNER))} of {a(noun())}", f"{a(noun())}; {of(MANNER)}"])),
        (4, lambda: ", ".join([noun(), of(SEEN + VIEW).replace(" of", ""), *some(PARTS, R.randint(1, 3)), of(MANNER)])),
        (3, lambda: of([f"{of(VAGUE)} {of(KIND)}", a(of(KIND)), f"{a(of(KIND))} {of(SEEN)}", f"{a(of(KIND))} with {of(PARTS)}"])),
        (3, lambda: f"{noun()}: " + ", ".join(some(PARTS, R.randint(2, 5)))),
        (4, lambda: f"{of(VIEW).capitalize()} {a(of(ADJ) + ' ' + noun())} {of(ACT)} and {of(SEEN)}, with {of(PARTS)}."),
        (5, lambda: of([f"{a(noun())}-shaped {noun()}", f"{a(noun())} {noun()}", f"half {noun()} half {noun()}", f"{a(noun())} made of {of(MATERIAL)}",
                        f"{a(noun())} that looks like {a(noun())}", f"{a(noun())} shaped like {a(noun())}"])),
        (5, lambda: of([f"{a(noun())} and {a(noun())}", f"{a(noun())}, {a(noun())} and {a(noun())}", f"{a(noun())} with {a(noun())}"])),
        (10 * bool(words), lambda: of([of(words), a(of(words)), f"{a(of(ADJ))} {of(words)}", f"{of(ASK)} {a(of(words))}", f"{a(of(words))} with {of(PARTS)}",
                                       f"{of(words)} {of(words)}"])),
    ]
    weights, out = [w for w, _ in kinds], set()
    while len(out) < n:
        out.add(R.choices(kinds, weights)[0][1]())
    out = sorted(out)
    R.shuffle(out)
    return out


def corpus(args):
    R = random.Random(args.seed)
    captions = list(dict.fromkeys(lines(args.captions)))
    R.shuffle(captions)
    captions = captions[:args.n // 2]
    nouns = sorted({w.lower() for w in ANIMALS + THINGS + lines(args.labels) if len(w) < 30})
    count = collections.Counter(w for c in captions for w in re.findall(r"[a-z]+", c.lower()))
    common = [w for w, k in count.most_common(9000) if len(w) >= 3 and k >= 5]
    rare = sorted({w.lower() for w in lines(args.words) if 4 <= len(w) <= 10 and w.isalpha()})
    made = written(args.n - len(captions), nouns, common + R.sample(rare, min(6000, len(rare))), args.seed)
    both = [t for pair in zip(made, captions) for t in pair] + made[len(captions):]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(both) + "\n")
    print(f"{len(both)} prompts, {len(captions)} of them captions, about {len(nouns)} things -> {args.out}")
    for t in made[:8]:
        print("  " + t)


def unit(v):
    return v / v.norm(dim=-1, keepdim=True)


def small(device=DEVICE):
    from transformers import AutoModel, AutoTokenizer
    return AutoTokenizer.from_pretrained(SMALL), AutoModel.from_pretrained(SMALL).text_model.to(device)


def tokens(tok, texts, device=DEVICE):
    return tok([t.lower() for t in texts], padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)


@torch.inference_mode()
def label(args):
    from words import Words
    words, texts, out = Words(args.teacher), lines(args.prompts), []
    model, remap = words.model.to(DEVICE).to(torch.float32 if DEVICE == "cpu" else torch.float16), words.remap.to(DEVICE)
    for k in range(0, len(texts), 64):
        ids = words.tokenizer([t.lower() for t in texts[k:k + 64]], padding="max_length", truncation=True, max_length=64, return_tensors="pt")["input_ids"]
        out.append(unit(model(input_ids=remap[ids.to(DEVICE)].long()).pooler_output.float()).cpu().numpy().astype(np.float16))
        if k % 6400 == 0:
            print(f"{k}/{len(texts)}", flush=True)
    np.save(args.out, np.concatenate(out))


def fit(args):
    texts, Y = lines(args.prompts), torch.from_numpy(np.load(args.vectors).astype(np.float32))
    texts, held = texts[:len(Y)], max(1, len(Y) // 20)
    mu, (tok, enc) = Y[:-held].mean(0), small()
    head = torch.nn.Linear(enc.config.hidden_size, Y.shape[1]).to(DEVICE)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    told = lambda batch: head(enc(**tokens(tok, batch)).pooler_output) + mu.to(DEVICE)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": args.lr}, {"params": head.parameters(), "lr": 20 * args.lr}], weight_decay=0.01)
    start, rng = [g["lr"] for g in opt.param_groups], np.random.default_rng(0)
    for step in range(args.steps + 1):
        if step % 1000 == 0 or step == args.steps:
            enc.eval()
            with torch.inference_mode():
                P = unit(torch.cat([told(texts[len(Y) - held:][k:k + 256]).float().cpu() for k in range(0, held, 256)]))
            enc.train()
            near = (unit(P - mu) * unit(Y[-held:] - mu)).sum(1).mean()
            first = ((P @ Y[-held:].T).argmax(1) == torch.arange(held)).float().mean()
            print(f"step {step}: held-out prompts, cosine to the teacher with the mean taken away {near:.3f}, own vector nearest {first:.1%}", flush=True)
        if step == args.steps:
            break
        for g, lr in zip(opt.param_groups, start):
            g["lr"] = lr * min(1, (step + 1) / 200) * (0.5 + 0.5 * math.cos(math.pi * step / args.steps))
        rows = rng.integers(0, len(Y) - held, args.batch)
        p, y, m = told([texts[i] for i in rows]), Y[rows].to(DEVICE), mu.to(DEVICE)
        loss = (1 - F.cosine_similarity(p - m, y - m, dim=-1)).mean() + 10 * ((unit(p) - y) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*enc.parameters(), *head.parameters()], 1.0)
        opt.step()
    torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "mu": mu}, args.out)


def pack(args):
    c = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tok, enc = small("cpu")
    t = json.loads(tok.backend_tokenizer.to_str())["model"]
    vocab = [token for token, _ in sorted(t["vocab"].items(), key=lambda kv: kv[1])]
    merges = [m if isinstance(m, str) else " ".join(m) for m in t["merges"]]
    arrays = {}
    for k, w in {**c["enc"], "head.weight": c["head"]["weight"], "head.bias": c["head"]["bias"] + c["mu"]}.items():
        w = w.float().numpy()
        if w.ndim == 2:
            scale = np.maximum(np.abs(w).max(1, keepdims=True), 1e-12) / (2 ** ((args.table if "token_embedding" in k else args.bits) - 1) - 1)
            arrays[k], arrays[k + ".scale"] = np.rint(w / scale).astype(np.int8), scale.astype(np.float32)
        elif "position_ids" not in k:
            arrays[k] = w.astype(np.float32)
    cfg = enc.config
    meta = dict(source=SMALL, width=cfg.hidden_size, n_layer=cfg.num_hidden_layers, n_head=cfg.num_attention_heads, positions=cfg.max_position_embeddings,
                bos=vocab.index("<|startoftext|>"), eos=vocab.index("<|endoftext|>"), eps=cfg.layer_norm_eps, vector=len(c["mu"]))
    packed = np.frombuffer(json.dumps({"vocab": vocab, "merges": merges}, ensure_ascii=False).encode(), np.uint8)
    np.savez_compressed(args.out, meta=json.dumps(meta), tokenizer=packed, **arrays)
    print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB")


def main():
    p = argparse.ArgumentParser(description="a small reader of typed words for the browser: a TinyCLIP text tower taught to give the SigLIP 2 vector a drawing model is "
                                            "told. prompts writes a file of prompts, a line each: half from templates over a list of things and the ways a drawing "
                                            "is asked for, half captions by people; label works out the big model's vectors for them; fit teaches the small one "
                                            "to match them; pack writes the file pen.js reads")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prompts")
    a.add_argument("--n", type=int, default=100_000)
    a.add_argument("--labels", default="", help="more things to draw, a line each, such as the labels of a drawing dataset")
    a.add_argument("--captions", default="", help="captions written by people, a line each; every other prompt is one of these")
    a.add_argument("--words", default="/usr/share/dict/words", help="a word list; a tenth of the written prompts are about its words, so that rare ones are met")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="out/words/prompts.txt")
    a = sub.add_parser("label")
    a.add_argument("--prompts", default="out/words/prompts.txt")
    a.add_argument("--teacher", default="siglip-text.npz", help="downloaded from the GitHub release if it is not there")
    a.add_argument("--out", default="out/words/vectors.npy")
    a = sub.add_parser("fit")
    a.add_argument("--prompts", default="out/words/prompts.txt")
    a.add_argument("--vectors", default="out/words/vectors.npy")
    a.add_argument("--steps", type=int, default=6000)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--lr", type=float, default=5e-5)
    a.add_argument("--out", default="out/words/words.pt")
    a = sub.add_parser("pack")
    a.add_argument("--checkpoint", default="out/words/words.pt")
    a.add_argument("--bits", type=int, default=6)
    a.add_argument("--table", type=int, default=4, help="bits for the table of token vectors, most of the file")
    a.add_argument("--out", default="words.npz")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    {"prompts": corpus, "label": label, "fit": fit, "pack": pack}[args.command](args)


if __name__ == "__main__":
    main()
