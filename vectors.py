import json
import os
import zlib

import numpy as np

from strokes import StrokeFile, stem

MODEL, TOKENS, OPEN = "google/siglip2-so400m-patch14-224", 64, {}


class Column:

    def __init__(self, path, names, there):
        import pyarrow.parquet as pq
        meta = pq.read_metadata(path)
        dim = meta.schema.to_arrow_schema().field(names[0]).type.list_size
        self.path, self.names, self.there, self.group, self.block = path, names, there, max(meta.row_group(0).num_rows, 1), (-1, None)
        self.shape = (meta.num_rows, dim) if len(names) == 1 else (meta.num_rows, len(names), dim)

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        return read(self.path, self.names)

    def __getitem__(self, i):
        g = int(i) // self.group
        if self.block[0] != g:
            import pyarrow.parquet as pq
            if (os.getpid(), self.path) not in OPEN:
                OPEN[os.getpid(), self.path] = pq.ParquetFile(self.path)
            t = OPEN[os.getpid(), self.path].read_row_group(g, columns=self.names, use_threads=False)
            parts = [t[n].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(t.num_rows, -1) for n in self.names]
            self.block = (g, parts[0] if len(parts) == 1 else np.stack(parts, 1))
        return self.block[1][int(i) % self.group]


class Rows:

    def __init__(self, *parts):
        self.parts = list(parts)
        self.end = np.cumsum([len(a) for a in self.parts])
        self.shape = (int(self.end[-1]), *self.parts[0].shape[1:])

    def __len__(self):
        return self.shape[0]

    def at(self, i):
        k = int(np.searchsorted(self.end, i, "right"))
        return self.parts[k], int(i - (self.end[k - 1] if k else 0))

    def there(self, i):
        part, j = self.at(i)
        return bool(part.there[j])

    def __getitem__(self, i):
        part, j = self.at(i)
        v = np.asarray(part[j], np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def together(parts):
    return Rows(*[p for a in parts for p in a.parts])


def said(f, column):
    if column == "image":
        return 0
    return zlib.crc32("\n".join(f.tables["label"][j] for j in f.cols["label"]).encode() if column == "label"
                      else "\n".join(f.column(column)).encode())


def table_path(sketches, tag=""):
    return stem(sketches) + f".vectors{tag}.parquet"


def fields(schema):
    out = {}
    for n in schema.names[1:]:
        out.setdefault(n[:-2] if n[-2:] in ("_1", "_2") else n, []).append(n)
    return out


def read(path, names):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=names)
    parts = [t[n].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(t.num_rows, -1) for n in names]
    return parts[0] if len(parts) == 1 else np.stack(parts, 1)


def fresh(sketches, tag="", quiet=False):
    import pyarrow.parquet as pq
    path = table_path(sketches, tag)
    if not os.path.exists(path):
        return {}
    schema, f = pq.read_schema(path), StrokeFile(sketches)
    assert int(schema.metadata[b"order"]) == order(f), f"{path} was not made from {sketches}"
    told, out = json.loads(schema.metadata[b"said"]), {}
    for column, names in fields(schema).items():
        if told.get(column) == said(f, column):
            out[column] = names
        elif not quiet:
            print(f"{column} in {path} was made from other words than {sketches} now holds", flush=True)
    return out


def columns(sketches, name="siglip", only=None):
    f = StrokeFile(sketches)
    texts = lambda c: [f.tables["label"][j] for j in f.cols["label"]] if c == "label" else f.column(c)
    return {column: Rows(Column(table_path(sketches), names, np.ones(len(f), bool) if column == "image" else np.array([bool(t) for t in texts(column)])))
            for column, names in fresh(sketches).items() if only is None or column in only}


def order(f, n=None):
    return zlib.crc32("\n".join(f.tables["id"][j] for j in f.cols["id"][:n]).encode())


def have(path, name="siglip"):
    return os.path.exists(table_path(path))


class Siglip:

    def __init__(self, name=MODEL, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float32 if self.dev == "cpu" else torch.float16
        self.model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(self.dev).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def unit(self, out):
        out = (out if self.torch.is_tensor(out) else out.pooler_output).float()
        return (out / out.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16)

    def ids(self, texts):
        return self.tokenizer([t.lower() for t in texts], padding="max_length", truncation=True, max_length=TOKENS,
                              return_tensors="pt")["input_ids"].to(self.dev)

    def texts(self, texts, batch=256):
        with self.torch.inference_mode():
            return np.concatenate([self.unit(self.model.get_text_features(input_ids=self.ids(texts[k:k + batch])))
                                   for k in range(0, len(texts), batch)])

    def images(self, images, batch=64):
        with self.torch.inference_mode():
            x = self.torch.from_numpy(images).to(self.dev).to(self.model.dtype).div(127.5).sub(1)[:, None].expand(-1, 3, -1, -1)
            return np.concatenate([self.unit(self.model.get_image_features(pixel_values=x[k:k + batch]))
                                   for k in range(0, len(x), batch)])


def encoder(name="siglip"):
    return Siglip()
