# PenGPT in a browser

Both packed models in one page, with no server and no libraries. `pen.js` is `inference/pen.py` again in plain JavaScript, reading
the same `.npz` files, with one 37-line WebAssembly kernel (`matvec.wat`) for the matrix products. It gives the tokens the numpy pen
gives. Handwriting takes about a millisecond a token on a laptop and a drawing about ten.

```bash
cd web && for f in hand.npz sketch.npz words.npz; do curl -LO https://github.com/greydanus/pengpt/releases/download/models-2026-11-16/$f; done
python -m http.server                                                # then open localhost:8000 to write, localhost:8000/#draw to draw
```

The page fetches a model only when its side is first shown, so someone who only writes never downloads the drawing model.

| | Files | Download |
|---|---|---|
| The page itself | `index.html`, `pen.js` | 37 KB (13 KB gzipped) |
| To write | `hand.npz` | 5.8 MB |
| To draw | `sketch.npz` 49.8 MB, `words.npz` 25.9 MB | 75.7 MB |
| Both | | 81.5 MB |

## Typed words

The drawing model is told a SigLIP 2 vector, and SigLIP 2's text half is 375 MB, so the page reads typed words with a small stand-in,
`words.npz`: the text tower of TinyCLIP (19M parameters, MIT licence) with a linear layer on top, tuned to give SigLIP 2's vector
on 100,000 prompts. On prompts held out of that tuning its vector is nearest SigLIP 2's own for the same prompt 96% of the time,
among 5,500. `words.py` makes the file:

```bash
python web/words.py prompts --captions captions.txt --labels labels.txt  # 100,000 prompts: half from templates, half captions by people
python web/words.py label                                                # SigLIP 2's vector for each
python web/words.py fit                                                  # tune the small tower to them: 23 minutes on a laptop
python web/words.py pack                                                 # words.npz, 6 bits a weight and 4 for the token table
```

The released file was tuned on prompts written this way, with the 400,000 COCO captions as `--captions` and the labels of Quick,
Draw!, TU-Berlin and SEVA as `--labels`.

## Files

```
index.html    the page: one side writes, the other draws
pen.js        both models, the reader of typed words, and the worker the page runs them in
matvec.wat    the WebAssembly kernel, kept in pen.js as base64
words.py      makes words.npz
test_web.py   pen.js against the numpy pen, under node
```
