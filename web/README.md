# PenGPT in a browser

Both packed models in a browser tab, with no server and no libraries, in two files: `index.html`, the page and all of its code, and
`pengpt.npz`, the weights. The code is `inference/pen.py` again in plain JavaScript, with one 37-line WebAssembly kernel
(`matvec.wat`) for the matrix products. It gives the tokens the numpy pen gives. Handwriting takes about a millisecond a token on a
laptop and a drawing about ten.

```bash
cd web && curl -LO https://github.com/greydanus/pengpt/releases/download/models-2026-11-16/pengpt.npz
python -m http.server                                                # then open localhost:8000 to write, localhost:8000/#draw to draw
```

`pengpt.npz` holds `hand.npz`, `sketch.npz` and `words.npz` one after another, and the page asks the server for only the bytes of the
model its open side needs: the writing side, where the page opens, fetches 5.8 MB, and the drawing models are fetched only when
someone turns to that side. GitHub Pages serves part of a file when asked. A server that does not, such as `python -m http.server`,
sends the whole file once and the page takes what it needs from it.

| | In `pengpt.npz` | Download |
|---|---|---|
| The page | | 45 KB (16 KB gzipped) |
| To write | the handwriting model | 5.8 MB |
| To draw | the drawing model 49.8 MB, the reader of typed words 25.9 MB | 75.7 MB |
| Both | | 81.5 MB |

## The page

Black on white, in the look of [the blog it is shown on](https://greydanus.github.io/pengpt/): a light sans, grey buttons and blue
for what is active. It opens on the first line of the Iliad in the hand `sam`.

- **Write.** Type anything; pick one of the 1,192 hands or press *another hand*; slide from neat to loose. Each hand writes in an ink
  of its own, taken from a hash of its name along six inks. *Save svg* keeps what is on the page.
  Strokes are smoothed into curves as they are decoded, the same way the numpy pen does it, so the saved file holds curves too.
- **Draw.** Type what to draw, or press *surprise me*, for four drawings. *Literal* is how hard a drawing is pushed towards its words
  and *dense* how much ink it is asked for. Moving either while a drawing is being made changes the strokes still to come and leaves
  the ones already drawn.
- **Pause** holds the pen where it is, on either side, and *resume* carries on from the same stroke. Typing something new, drawing
  again or switching sides lets go of it.

While a model downloads the page fades and a wheel turns over it, with the seconds left worked out again every two seconds from the
rate so far. A model, once fetched, is kept in the browser's Cache storage, so a later visit does not fetch it again. On a narrow
screen the controls stack in even rows.

Inside a frame the page drops its title and its link to this repository, and tells the page around it how tall it is by
`postMessage`, as `{pengpt: height}`, so that the frame can be sized to fit. `index.html?more=/some/path` adds a *Main demo page*
link to that path after the status line.

## Typed words

The drawing model is told a SigLIP 2 vector, and SigLIP 2's text half is 375 MB, so the page reads typed words with a small stand-in:
the text tower of TinyCLIP (19M parameters, MIT licence) with a linear layer on top, tuned to give SigLIP 2's vector on 100,000
prompts. On prompts held out of that tuning its vector is nearest SigLIP 2's own for the same prompt 96% of the time, among 5,500.
`words.py` makes it, and the weights file:

```bash
python web/words.py prompts --captions captions.txt --labels labels.txt  # 100,000 prompts: half from templates, half captions by people
python web/words.py label                                                # SigLIP 2's vector for each
python web/words.py fit                                                  # tune the small tower to them: 23 minutes on a laptop
python web/words.py pack                                                 # words.npz, 6 bits a weight and 4 for the token table
python web/words.py bundle                                               # web/pengpt.npz, from hand.npz, sketch.npz and words.npz
```

The released reader was tuned on prompts written this way, with the 400,000 COCO captions as `--captions` and the labels of Quick,
Draw!, TU-Berlin and SEVA as `--labels`.

## Files

```
index.html    the page and its code: both models, the reader of typed words, and the worker they run in
matvec.wat    the WebAssembly kernel, kept in index.html as base64
words.py      makes the reader of typed words and pengpt.npz
test_web.py   the page's pen against the numpy pen, under node: the same tokens, and the same curves from them
```
