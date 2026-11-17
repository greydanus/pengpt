# Sketch ablations

Fifteen runs of the sketch preset on one A100 each, 54,000 steps, evaluated every 3,000; each differs from base in the flags
`tools/ablate.py list sketch` gives. What the picture, label and caption vectors buy is read in [captions.md](captions.md).

## Held-out pen moves

| Run | Step | Word score | Against base | Text rank | Unseen alone / after 1 / after 3 | Last step score | Minutes | controlsketch | creative | differsketching | icons | photosketch | quickdraw | seva | sketchy | things | tuberlin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 54000 | 0.5914 |  | 1.88 |  | 0.5914 | 197 | 0.5349 | 0.6538 | 0.6210 | 0.3248 | 0.7527 | 0.5202 | 0.6043 | 0.6559 | 0.6322 | 0.6141 |
| quickdraw | 36000 | 0.5218 | +0.0017 | 1.00 |  | 0.5231 | 191 |  |  |  |  |  | 0.5218 |  |  |  |  |
| image_only | 54000 | 0.5800 | -0.0113 | 1.00 |  | 0.5800 | 196 | 0.5294 | 0.6452 | 0.6158 | 0.3081 | 0.7478 | 0.5034 | 0.5875 | 0.6422 | 0.6174 | 0.6034 |
| seed | 54000 | 0.5901 | -0.0012 | 1.38 |  | 0.5901 | 199 | 0.5334 | 0.6525 | 0.6196 | 0.3231 | 0.7519 | 0.5198 | 0.6038 | 0.6538 | 0.6314 | 0.6119 |
| no_hint | 54000 | 0.5903 | -0.0011 | 1.12 |  | 0.5903 | 200 | 0.5338 | 0.6528 | 0.6195 | 0.3239 | 0.7514 | 0.5197 | 0.6040 | 0.6538 | 0.6316 | 0.6125 |
| no_guidance | 54000 | 0.5910 | -0.0004 | 1.38 |  | 0.5910 | 197 | 0.5338 | 0.6535 | 0.6215 | 0.3240 | 0.7528 | 0.5200 | 0.6042 | 0.6547 | 0.6322 | 0.6133 |
| rotary_pen_pos | 54000 | 0.5918 | +0.0004 | 1.50 |  | 0.5918 | 199 | 0.5344 | 0.6546 | 0.6229 | 0.3268 | 0.7529 | 0.5205 | 0.6044 | 0.6558 | 0.6321 | 0.6135 |
| added_pen_pos | 54000 | 0.5926 | +0.0012 | 1.25 |  | 0.5926 | 233 | 0.5358 | 0.6559 | 0.6246 | 0.3256 | 0.7563 | 0.5202 | 0.6045 | 0.6563 | 0.6321 | 0.6147 |
| image_label | 54000 | 0.5956 | +0.0042 | 1.62 |  | 0.5956 | 197 | 0.5343 | 0.6584 | 0.6247 | 0.3356 | 0.7577 | 0.5226 | 0.6091 | 0.6576 | 0.6396 | 0.6166 |
| no_order | 54000 | 0.5957 | +0.0043 | 1.56 |  | 0.5957 | 200 | 0.5416 | 0.6618 | 0.6268 | 0.3276 | 0.7589 | 0.5217 | 0.6055 | 0.6604 | 0.6344 | 0.6183 |
| labels | 54000 | 0.5966 | +0.0053 | 1.06 |  | 0.5966 | 201 | 0.5357 | 0.6576 | 0.6258 | 0.3397 | 0.7581 | 0.5223 | 0.6096 | 0.6592 | 0.6407 | 0.6178 |
| no_pen_pos | 54000 | 0.5985 | +0.0071 | 1.75 |  | 0.5985 | 232 | 0.5401 | 0.6635 | 0.6331 | 0.3345 | 0.7632 | 0.5222 | 0.6075 | 0.6636 | 0.6362 | 0.6213 |
| small | 54000 | 0.5991 | +0.0077 | 1.25 |  | 0.5991 | 161 | 0.5403 | 0.6631 | 0.6330 | 0.3380 | 0.7605 | 0.5242 | 0.6088 | 0.6645 | 0.6371 | 0.6211 |
| no_augment | 9000 | 0.6039 | +0.0125 | 1.12 |  | 0.8946 | 198 | 0.5527 | 0.6967 | 0.6598 | 0.2905 | 0.7832 | 0.4681 | 0.6170 | 0.6864 | 0.6471 | 0.6374 |
| tiny | 54000 | 0.6124 | +0.0210 | 1.44 |  | 0.6124 | 130 | 0.5503 | 0.6798 | 0.6498 | 0.3587 | 0.7733 | 0.5317 | 0.6173 | 0.6809 | 0.6469 | 0.6349 |

Word score is at the step with the lowest held-out loss, in nats per pen move; lower is better. Against base is the mean difference over the sources both runs hold out. The seed row is the noise to read the rest by.


## Typed prompts

Each model draws 345 Quick, Draw! labels and 300 held-out caption phrases, two drawings a prompt at temperature 0.6; SigLIP 2 ranks
each drawing's own prompt among its set. The labels run is told label tokens and no vectors, so phrases mean nothing to it;
quickdraw trained on Quick, Draw! alone, so its pen-move score is over that source only.

| Run | Prompts | Guidance | Cosine to its prompt | Its prompt ranked first | In the first five | Strokes |
|---|---|---|---|---|---|---|
| added_pen_pos | labels | 1 | 0.1178 | 55.9% | 76.7% | 17.0 |
| added_pen_pos | labels | 2 | 0.1306 | 75.7% | 90.6% | 12.1 |
| added_pen_pos | phrases | 1 | 0.1061 | 29.8% | 52.5% | 62.1 |
| added_pen_pos | phrases | 2 | 0.1232 | 44.7% | 71.3% | 45.0 |
| base | labels | 1 | 0.1151 | 54.9% | 73.3% | 21.1 |
| base | labels | 2 | 0.1304 | 75.4% | 89.6% | 14.8 |
| base | phrases | 1 | 0.1022 | 28.8% | 51.2% | 59.5 |
| base | phrases | 2 | 0.1223 | 43.0% | 70.0% | 50.2 |
| image_label | labels | 1 | 0.1186 | 57.5% | 77.4% | 19.4 |
| image_label | labels | 2 | 0.1325 | 77.5% | 91.7% | 15.1 |
| image_label | phrases | 1 | 0.0717 | 6.8% | 16.8% | 74.6 |
| image_label | phrases | 2 | 0.0815 | 9.5% | 24.0% | 64.5 |
| image_only | labels | 1 | 0.0602 | 4.5% | 11.6% | 143.8 |
| image_only | labels | 2 | 0.0625 | 6.1% | 14.2% | 84.1 |
| image_only | phrases | 1 | 0.0666 | 5.0% | 12.2% | 149.7 |
| image_only | phrases | 2 | 0.0725 | 4.3% | 17.2% | 90.6 |
| image_only | labels, shifted | 1 | 0.0814 | 18.8% | 38.0% | 60.9 |
| image_only | labels, shifted | 2 | 0.0839 | 22.3% | 43.8% | 26.3 |
| image_only | phrases, shifted | 1 | 0.0871 | 11.0% | 26.7% | 114.5 |
| image_only | phrases, shifted | 2 | 0.0974 | 17.3% | 36.8% | 54.7 |
| labels | labels | 1 | 0.1170 | 55.1% | 77.4% | 15.3 |
| labels | labels | 2 | 0.1298 | 74.1% | 90.0% | 12.1 |
| labels | phrases | 1 | 0.0280 | 0.8% | 2.8% | 10.6 |
| labels | phrases | 2 | 0.0268 | 0.5% | 2.2% | 14.4 |
| no_augment | labels | 1 | 0.0781 | 14.3% | 28.8% | 71.2 |
| no_augment | labels | 2 | 0.0612 | 3.9% | 11.7% | 3.9 |
| no_augment | phrases | 1 | 0.0926 | 17.3% | 37.5% | 75.8 |
| no_augment | phrases | 2 | 0.0645 | 6.2% | 17.7% | 3.4 |
| no_guidance | labels | 1 | 0.1191 | 57.7% | 77.4% | 14.5 |
| no_guidance | phrases | 1 | 0.1103 | 32.7% | 58.7% | 44.8 |
| no_hint | labels | 1 | 0.1167 | 54.6% | 77.1% | 22.2 |
| no_hint | labels | 2 | 0.1310 | 76.2% | 91.9% | 15.7 |
| no_hint | phrases | 1 | 0.1059 | 30.3% | 54.8% | 67.8 |
| no_hint | phrases | 2 | 0.1209 | 43.5% | 67.8% | 55.5 |
| no_order | labels | 1 | 0.1179 | 54.2% | 77.5% | 16.6 |
| no_order | labels | 2 | 0.1315 | 75.2% | 91.2% | 11.4 |
| no_order | phrases | 1 | 0.1048 | 29.0% | 51.5% | 53.2 |
| no_order | phrases | 2 | 0.1209 | 42.2% | 68.0% | 36.2 |
| no_pen_pos | labels | 1 | 0.1165 | 53.8% | 74.5% | 20.6 |
| no_pen_pos | labels | 2 | 0.1313 | 73.9% | 90.1% | 12.2 |
| no_pen_pos | phrases | 1 | 0.1021 | 26.3% | 46.7% | 62.3 |
| no_pen_pos | phrases | 2 | 0.1209 | 40.3% | 68.2% | 41.2 |
| quickdraw | labels | 1 | 0.1218 | 61.3% | 82.5% | 7.3 |
| quickdraw | labels | 2 | 0.1303 | 73.5% | 89.6% | 6.9 |
| quickdraw | phrases | 1 | 0.1017 | 27.3% | 54.7% | 8.8 |
| quickdraw | phrases | 2 | 0.1137 | 36.7% | 66.3% | 8.7 |
| rotary_pen_pos | labels | 1 | 0.1138 | 51.2% | 73.6% | 21.4 |
| rotary_pen_pos | labels | 2 | 0.1300 | 76.2% | 89.4% | 13.7 |
| rotary_pen_pos | phrases | 1 | 0.1030 | 28.2% | 49.0% | 61.8 |
| rotary_pen_pos | phrases | 2 | 0.1206 | 40.8% | 68.3% | 46.2 |
| seed | labels | 1 | 0.1180 | 57.5% | 76.5% | 23.3 |
| seed | labels | 2 | 0.1304 | 75.1% | 90.7% | 13.0 |
| seed | phrases | 1 | 0.1053 | 31.2% | 52.8% | 59.7 |
| seed | phrases | 2 | 0.1224 | 43.7% | 71.5% | 44.7 |
| small | labels | 1 | 0.1093 | 45.4% | 66.2% | 24.0 |
| small | labels | 2 | 0.1282 | 71.0% | 89.6% | 13.7 |
| small | phrases | 1 | 0.0978 | 21.7% | 43.5% | 70.3 |
| small | phrases | 2 | 0.1166 | 35.8% | 63.8% | 46.7 |
| tiny | labels | 1 | 0.0990 | 34.6% | 54.2% | 26.9 |
| tiny | labels | 2 | 0.1210 | 62.2% | 80.4% | 19.5 |
| tiny | phrases | 1 | 0.0909 | 16.5% | 34.8% | 76.9 |
| tiny | phrases | 2 | 0.1071 | 30.8% | 55.8% | 62.0 |
