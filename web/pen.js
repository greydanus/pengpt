const DIRECTIONS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
const DOWN = 8, UP = 9, N_BASE = 10, EPS = 1.1920929e-7, FORMAT = 2
const TYPES = {'|i1': Int8Array, '|u1': Uint8Array, '<i2': Int16Array, '<i4': Int32Array, '<i8': BigInt64Array, '<f4': Float32Array, '<f8': Float64Array}

async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

function npy(b) {
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength)
  const start = b[6] === 1 ? 10 : 12, n = b[6] === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
  const head = new TextDecoder().decode(b.subarray(start, start + n))
  const descr = head.match(/'descr': '([^']+)'/)[1]
  const shape = head.match(/'shape': \(([^)]*)\)/)[1].split(',').map(s => s.trim()).filter(Boolean).map(Number)
  const raw = b.slice(start + n).buffer
  if (descr[1] === 'U') {
    const k = +descr.slice(2), u = new Uint32Array(raw), data = []
    for (let i = 0; i < u.length; i += k) data.push(String.fromCodePoint(...Array.from(u.subarray(i, i + k)).filter(c => c)))
    return {data, shape}
  }
  const data = new TYPES[descr](raw)
  return {data: descr === '<i8' ? Int32Array.from(data, Number) : data, shape}
}

export async function npz(buffer) {
  const b = new Uint8Array(buffer), v = new DataView(buffer), out = {}
  let end = b.length - 22
  while (v.getUint32(end, true) !== 0x06054b50) end--
  let p = v.getUint32(end + 16, true)
  for (let k = v.getUint16(end + 10, true); k--;) {
    const method = v.getUint16(p + 10, true), size = v.getUint32(p + 20, true), at = v.getUint32(p + 42, true)
    const lens = [28, 30, 32].map(o => v.getUint16(p + o, true))
    const name = new TextDecoder().decode(b.subarray(p + 46, p + 46 + lens[0])).replace(/\.npy$/, '')
    const from = at + 30 + v.getUint16(at + 26, true) + v.getUint16(at + 28, true)
    const body = b.subarray(from, from + size)
    out[name] = npy(method ? await inflate(body) : body)
    p += 46 + lens[0] + lens[1] + lens[2]
  }
  return out
}

export const backend = {
  matvec(W, x, out = new Float32Array(W.r)) {
    const {q, s, r, c} = W, c4 = c & ~3
    for (let i = 0, o = 0; i < r; i++, o += c) {
      let a = 0, b = 0, d = 0, e = 0, j = 0
      for (; j < c4; j += 4) {
        a += q[o + j] * x[j]
        b += q[o + j + 1] * x[j + 1]
        d += q[o + j + 2] * x[j + 2]
        e += q[o + j + 3] * x[j + 3]
      }
      for (; j < c; j++) a += q[o + j] * x[j]
      out[i] = (a + b + d + e) * s[i]
    }
    return out
  },
}

const MATVEC = 'AGFzbQEAAAABCgFgBn9/f39/fwADAgEABQMBAAEHEwIGbWVtb3J5AgAGbWF0dmVjAAAKoQIBngIDBH8DewF9IAVBeHEhCAJAA0AgBiAETw0B/QwAAAAAAAAAAAAAAAAAAAAAIQr9DAAAAAAAAAAAAAAAAAAAAAAhC0EAIQcCQANAIAcgCE8NASAAIAdq/QEDACEMIAIgB0ECdGohCSAKIAz9pwH9+gEgCf0ABAD95gH95AEhCiALIAz9qAH9+gEgCf0ABBD95gH95AEhCyAHQQhqIQcMAAsLIAogC/3kASEKIAr9HwAgCv0fAZIgCv0fAiAK/R8DkpIhDQJAA0AgByAFTw0BIA0gACAHaiwAALIgAiAHQQJ0aioCAJSSIQ0gB0EBaiEHDAALCyADIAZBAnRqIA0gASAGQQJ0aioCAJQ4AgAgACAFaiEAIAZBAWohBgwACwsL'
const X = 0, OUT = 1 << 15
let simd = null

export async function accelerate(pen) {
  simd = simd || (await WebAssembly.instantiate(Uint8Array.from(atob(MATVEC), c => c.charCodeAt(0)))).instance.exports
  const mats = Object.values(pen.w).filter(W => W.q), need = mats.reduce((n, W) => n + W.q.length + 4 * W.r + 8, 0)
  let p = simd.memory.buffer.byteLength
  simd.memory.grow(Math.ceil(need / 65536))
  const bytes = new Int8Array(simd.memory.buffer), floats = new Float32Array(simd.memory.buffer)
  for (const W of mats) {
    bytes.set(W.q, W.p = p)
    floats.set(W.s, (W.sp = (p + W.q.length + 3) & ~3) >> 2)
    p = W.sp + 4 * W.r
  }
  const js = backend.js = backend.js || backend.matvec
  backend.matvec = (W, x) => {
    if (W.p === undefined) return js(W, x)
    const f = new Float32Array(simd.memory.buffer)
    f.set(x, X)
    simd.matvec(W.p, W.sp, X, OUT, W.r, W.c)
    return f.slice(OUT >> 2, (OUT >> 2) + W.r)
  }
  return pen
}

const matvec = (W, x) => backend.matvec(W, x)

function row(W, i) {
  const out = new Float32Array(W.c), o = i * W.c
  for (let j = 0; j < W.c; j++) out[j] = W.q[o + j] * W.s[i]
  return out
}

function rms(x, from = 0, n = x.length - from) {
  let ss = 0
  for (let i = from; i < from + n; i++) ss += x[i] * x[i]
  const k = 1 / Math.sqrt(ss / n + EPS), out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = x[from + i] * k
  return out
}

function headnorm(x, weight, H) {
  const hd = weight.length
  for (let h = 0; h < H; h++) {
    let ss = 0
    for (let d = 0; d < hd; d++) ss += x[h * hd + d] ** 2
    const k = 1 / Math.sqrt(ss / hd + EPS)
    for (let d = 0; d < hd; d++) x[h * hd + d] *= k * weight[d]
  }
  return x
}

function rotate(x, cos, sin, H) {
  const half = cos.length
  for (let h = 0; h < H; h++) {
    for (let d = 0, o = 2 * h * half; d < half; d++) {
      const a = x[o + d], b = x[o + d + half]
      x[o + d] = a * cos[d] + b * sin[d]
      x[o + d + half] = b * cos[d] - a * sin[d]
    }
  }
}

function attend(q, K, V, n, H) {
  const C = q.length, hd = C / H, out = new Float32Array(C), p = new Float32Array(n), scale = hd ** -0.5
  for (let h = 0; h < H; h++) {
    const o = h * hd
    let top = -Infinity, total = 0
    for (let j = 0; j < n; j++) {
      let a = 0
      for (let d = 0, at = j * C + o; d < hd; d++) a += K[at + d] * q[o + d]
      p[j] = a * scale
      top = Math.max(top, p[j])
    }
    for (let j = 0; j < n; j++) total += p[j] = Math.exp(p[j] - top)
    for (let j = 0; j < n; j++) {
      const w = p[j] / total
      for (let d = 0, at = j * C + o; d < hd; d++) out[o + d] += w * V[at + d]
    }
  }
  return out
}

function add(x, y) {
  for (let i = 0; i < x.length; i++) x[i] += y[i]
}

export function mulberry(seed) {
  let a = seed >>> 0
  return () => {
    a = a + 0x6D2B79F5 | 0
    let t = Math.imul(a ^ a >>> 15, 1 | a)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

const tick = () => new Promise(done => {
  const c = new MessageChannel()
  c.port1.onmessage = () => done(c.port1.close())
  c.port2.postMessage(0)
})

export class Pen {

  static async load(url) {
    const pen = new Pen(await npz(await (await fetch(url)).arrayBuffer()))
    return accelerate(pen).catch(() => pen)
  }

  constructor(z) {
    Object.assign(this, {pen_rotary: false, label_drop: 0, hint: 0, format: 1, vector: 0, encoder: null}, JSON.parse(z.meta.data[0]))
    if (this.format > FORMAT) throw new Error(`this model is in format ${this.format} and this file reads up to format ${FORMAT}`)
    const w = this.w = {}
    for (const [k, a] of Object.entries(z)) {
      if (['meta', 'merges', 'writers'].includes(k) || k.endsWith('.scale')) continue
      w[k] = a.shape.length === 2 ? {q: a.data, s: z[k + '.scale'].data, r: a.shape[0], c: a.shape[1]} : a.data
    }
    this.writers = z.writers.data
    const m = z.merges.data, n = N_BASE + m.length / 3
    Object.assign(this, {PAD: n, END: n + 1, WORD: n + 2, BOS: n + 3})
    this.paths = Array.from({length: n + 4}, (_, i) => i < 8 ? [DIRECTIONS[i]] : [])
    for (let i = 0; i < m.length; i += 3) this.paths[m[i + 2]] = this.paths[m[i]].concat(this.paths[m[i + 1]])
    this.paths = this.paths.map(path => {
      let x = 0, y = 0
      return path.map(([dx, dy]) => [x += dx, y += dy])
    })
    this.deltas = this.paths.map(p => p.length ? p[p.length - 1] : [0, 0])
    this.C = w['tok_emb.weight'].c
    const hd = this.hd = this.C / this.n_head, half = hd / 2, nq = hd >> 2, mq = (half - nq) >> 1
    this.freq = Float32Array.from({length: this.bands}, (_, k) => Math.PI / 2 ** (k + 1))
    this.inv = Float32Array.from({length: half}, (_, d) => 10000 ** -(2 * d / hd))
    this.seq_freq = Float32Array.from({length: nq}, (_, k) => 10000 ** -(k / nq))
    this.pen_freq = Float32Array.from({length: mq}, (_, k) => 2 * Math.PI / (4 * 128 ** (k / Math.max(mq - 1, 1))))
    this.layers = Array.from({length: this.n_layer}, (_, i) => Object.fromEntries(
      ['attn.q', 'attn.kv', 'attn.proj', 'attn.q_norm', 'attn.k_norm', 'cross_attn.q', 'cross_attn.kv', 'cross_attn.proj',
        'cross_attn.q_norm', 'cross_attn.k_norm', 'fc', 'fc_out'].map(k => [k, w[`blocks.${i}.${k}.weight`]])))
  }

  style(writer) {
    return typeof writer === 'string' ? row(this.w['writer_emb.weight'], this.writers.indexOf(writer)) : writer
  }

  told(text) {
    if (typeof text !== 'string') return text
    if (!text) return new Float32Array(this.vector)
    if (!this.encoder) throw new Error('a drawing model needs an encoder to turn typed words into a vector')
    return this.encoder(text)
  }

  rotary(t, px, py) {
    const half = this.hd / 2, cos = new Float32Array(half), sin = new Float32Array(half)
    for (let d = 0; d < half; d++) {
      const nq = this.seq_freq.length, mq = this.pen_freq.length
      const a = !this.pen_rotary ? t * this.inv[d] : d < nq ? t * this.seq_freq[d] : d < nq + mq ? px * this.pen_freq[d - nq] : py * this.pen_freq[d - nq - mq]
      cos[d] = Math.cos(a)
      sin[d] = Math.sin(a)
    }
    return [cos, sin]
  }

  context(text, v, hints) {
    const w = this.w, C = this.C, H = this.n_head, rows = [rms(v)]
    if (this.vector) {
      const told = this.told(text), flags = new Float32Array(40)
      for (const [, i] of hints) flags[i - 258] = 1
      if (told.some(x => x)) {
        const c = matvec(w['ctx_proj.weight'], told)
        add(c, row(w['ctx_slot.weight'], 0))
        for (let k = 0; k < 4; k++) rows.push(rms(c, k * c.length / 4, c.length / 4))
      }
      if (flags.some(x => x)) rows.push(rms(matvec(w['hint_proj.weight'], flags)))
    } else {
      const ids = [...new TextEncoder().encode(text)].map(b => b + 1).slice(0, this.max_text_length - 4 * !!this.hint)
      if (this.text_end) ids.push(257)
      const at = ids.map((_, i) => i).concat(hints.map(h => h[0]))
      ids.push(...hints.map(h => h[1]))
      ids.forEach((id, k) => {
        const e = row(w['ctx_emb.weight'], id)
        add(e, row(w['ctx_pos_emb.weight'], at[k]))
        rows.push(rms(e))
      })
    }
    return this.layers.map(L => {
      const K = new Float32Array(rows.length * C), V = new Float32Array(rows.length * C)
      rows.forEach((c, j) => {
        const kv = matvec(L['cross_attn.kv'], c)
        K.set(headnorm(kv.subarray(0, C), L['cross_attn.k_norm'], H), j * C)
        V.set(kv.subarray(C), j * C)
      })
      return {K, V, n: rows.length}
    })
  }

  *raw(text, writer = '', prefix = [], temperature = 0, topK = null, rng = Math.random, guidance = null, ink = null, strokes = null) {
    const w = this.w, C = this.C, H = this.n_head
    guidance = guidance ?? (this.label_drop ? 2 : 1)
    if (guidance !== 1 && !this.label_drop) throw new Error('guidance needs a model trained with --label_drop; this one was not')
    const clip = n => n == null ? null : Math.min(99, Math.max(0, Math.round(n)))
    const asked = this.hint ? [clip(ink == null ? null : 2 * ink), clip(strokes)] : []
    const digits = asked.flatMap(n => n == null ? [null, null] : [Math.floor(n / 10), n % 10])
    const hints = digits.map((d, k) => [this.max_text_length + this.text_end - 4 + k, 258 + 10 * k + d]).filter((_, k) => digits[k] != null)
    const v = this.style(writer)
    const seq = [this.BOS, ...prefix.flatMap(word => [...word, this.WORD, this.WORD])], given = seq.length
    const n = Math.min(this.max_seq_length, this.block_size)
    const cross = (guidance === 1 ? [text] : [text, '']).map(t => this.context(t, v, hints)), S = cross.length
    const K = this.layers.map(() => cross.map(() => new Float32Array(n * C))), V = this.layers.map(() => cross.map(() => new Float32Array(n * C)))
    const feats = new Float32Array(4 * this.bands)
    let px = 0, py = 0
    for (let t = 0; t < n - 1; t++) {
      px += this.deltas[seq[t]][0]
      py += this.deltas[seq[t]][1]
      const e = row(w['tok_emb.weight'], seq[t])
      if (this.bands) {
        for (let k = 0, B = this.bands; k < B; k++) {
          feats[k] = Math.sin(px * this.freq[k])
          feats[B + k] = Math.cos(px * this.freq[k])
          feats[2 * B + k] = Math.sin(py * this.freq[k])
          feats[3 * B + k] = Math.cos(py * this.freq[k])
        }
        add(e, matvec(w['pen_pos_proj.weight'], feats))
      }
      const x0 = rms(e), [cos, sin] = this.rotary(t, px, py), xs = cross.map(() => x0.slice())
      this.layers.forEach((L, i) => {
        const a0 = w['resid_scale'][i], b0 = w['x0_scale'][i]
        xs.forEach((x, s) => {
          for (let d = 0; d < C; d++) x[d] = a0 * x[d] + b0 * x0[d]
          const a = rms(x), kv = matvec(L['attn.kv'], a)
          const q = headnorm(matvec(L['attn.q'], a), L['attn.q_norm'], H), k = headnorm(kv.subarray(0, C), L['attn.k_norm'], H)
          rotate(q, cos, sin, H)
          rotate(k, cos, sin, H)
          K[i][s].set(k, t * C)
          V[i][s].set(kv.subarray(C), t * C)
          add(x, matvec(L['attn.proj'], attend(q, K[i][s], V[i][s], t + 1, H)))
          const cq = headnorm(matvec(L['cross_attn.q'], rms(x)), L['cross_attn.q_norm'], H), told = cross[s][i]
          add(x, matvec(L['cross_attn.proj'], attend(cq, told.K, told.V, told.n, H)))
          const h = matvec(L['fc'], rms(x))
          for (let d = 0; d < h.length; d++) h[d] = Math.max(h[d], 0) ** 2
          add(x, matvec(L['fc_out'], h))
        })
      })
      if (t < given - 1) continue
      const logp = xs.map(x => {
        const l = matvec(w['head.weight'], rms(x))
        let top = -Infinity, total = 0
        for (let i = 0; i < l.length; i++) top = Math.max(top, l[i] = 15 * Math.tanh(l[i] / 15))
        for (let i = 0; i < l.length; i++) total += Math.exp(l[i] - top)
        return l.map(u => u - top - Math.log(total))
      })
      const mixed = S === 1 ? logp[0] : logp[1].map((u, i) => u + guidance * (logp[0][i] - u))
      let next = 0
      if (temperature > 0) {
        const floor = topK ? Float32Array.from(mixed).sort()[mixed.length - topK] : -Infinity
        let top = -Infinity, total = 0
        for (let i = 0; i < mixed.length; i++) top = Math.max(top, mixed[i] = mixed[i] < floor ? -Infinity : mixed[i] / temperature)
        for (let i = 0; i < mixed.length; i++) total += mixed[i] = Math.exp(mixed[i] - top)
        for (let u = rng() * total; next < mixed.length - 1 && (u -= mixed[next]) >= 0;) next++
      } else {
        for (let i = 1; i < mixed.length; i++) if (mixed[i] > mixed[next]) next = i
      }
      seq.push(next)
      yield next
      if (next === this.END) return
    }
  }

  tokens(...args) {
    const out = [...this.raw(...args)], done = out[out.length - 1] === this.END, words = [[]]
    for (const t of done ? out.slice(0, -1) : out) t === this.WORD ? words.push([]) : words[words.length - 1].push(t)
    return words.filter(u => u.length).slice(0, done ? undefined : -1)
  }

  decode(tokens) {
    const p = []
    let x = 0, y = 0, down = false
    for (const t of tokens) {
      if (t === DOWN) {
        down = true
        p.push(x, y, 1)
      } else if (t === UP) {
        if (down) p.push(x, y, 0)
        down = false
      } else if (this.paths[t].length) {
        if (down) for (const [dx, dy] of this.paths[t]) p.push(x + dx, y + dy, 1)
        x += this.deltas[t][0]
        y += this.deltas[t][1]
      }
    }
    const raw = Float32Array.from(p, (u, i) => i % 3 === 2 ? u : u * this.grid), out = raw.slice()
    for (const [a, b] of spans(raw)) {
      for (let i = a + 1; b - a >= 3 && i < b - 1; i++) {
        out[3 * i] = (raw[3 * i - 3] + raw[3 * i] + raw[3 * i + 3]) / 3
        out[3 * i + 1] = (raw[3 * i - 2] + raw[3 * i + 1] + raw[3 * i + 4]) / 3
      }
    }
    return out
  }

  async *draw(text, source = '', temperature = 0.6, topK = null, rng = Math.random, guidance = null, ink = null, strokes = null, every = 8) {
    const tokens = []
    for (const t of this.raw(text, source, [], temperature, topK, rng, guidance, ink, strokes)) {
      if (t < this.PAD) tokens.push(t)
      if (tokens.length % every === 0) {
        yield this.decode(tokens)
        await tick()
      }
    }
    yield this.decode(tokens)
  }

  async *write(text, writer = '', temperature = 0, topK = null, seed = 0, nAtATime = 2) {
    const rng = mulberry(seed), texts = text.split(/\s+/).filter(Boolean), bytes = s => new TextEncoder().encode(s).length
    let prior = [], priorTokens = []
    for (let i = 0; i < texts.length; i += nAtATime) {
      const chunk = texts.slice(i, i + nAtATime)
      if (bytes([...prior, ...chunk].join(' ')) > this.max_text_length) [prior, priorTokens] = [[], []]
      let tokens = this.tokens([...prior, ...chunk].join(' '), writer, priorTokens, temperature, topK, rng)
      for (const word of chunk.slice(tokens.length)) tokens.push(this.tokens(word, writer, [], temperature, topK, rng)[0] || [])
      tokens = tokens.slice(0, chunk.length)
      for (const t of tokens) yield this.decode(t)
      await tick()
      ;[prior, priorTokens] = tokens.every(t => t.length) ? [chunk.slice(-1), tokens.slice(-1)] : [[], []]
    }
  }
}

export function spans(p) {
  const out = []
  for (let i = 0, a = -1, n = p.length / 3; i <= n; i++) {
    const down = i < n && p[3 * i + 2] === 1
    if (down && a < 0) a = i
    if (!down && a >= 0) {
      out.push([a, i])
      a = -1
    }
  }
  return out
}

export function layout(words, lineWidth = 8, lineHeight = 0.55, space = 0.16) {
  const placed = []
  let x = 0, y = 0
  for (const p of words) {
    if (!p.length) continue
    let lo = Infinity, hi = -Infinity
    for (let i = 0; i < p.length; i += 3) {
      lo = Math.min(lo, p[i])
      hi = Math.max(hi, p[i])
    }
    if (x > 0 && x + hi - lo > lineWidth) [x, y] = [0, y + lineHeight]
    placed.push(p.map((u, i) => i % 3 === 0 ? u - lo + x : i % 3 === 1 ? u + y : u))
    x += hi - lo + space
  }
  return placed
}

export function svg(words, scale = 160, pad = 0.3, stroke = 2.2, color = '#1a237e') {
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity]
  for (const p of words) for (let i = 0; i < p.length; i += 3) for (const k of [0, 1]) {
    lo[k] = Math.min(lo[k], p[i + k] - pad)
    hi[k] = Math.max(hi[k], p[i + k] + pad)
  }
  const paths = words.flatMap(p => spans(p).map(([a, b]) => {
    const xy = Array.from({length: b - a}, (_, i) => [(p[3 * (a + i)] - lo[0]) * scale, (p[3 * (a + i) + 1] - lo[1]) * scale])
    if (b - a === 1) xy.push([xy[0][0] + 0.01, xy[0][1] + 0.01])
    return '<path d="M' + xy.map(([u, v]) => `${u.toFixed(1)} ${v.toFixed(1)}`).join(' L') + '"/>'
  }))
  const [W, H] = [0, 1].map(k => ((hi[k] - lo[k]) * scale).toFixed(0))
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><rect width="100%" height="100%" fill="white"/>` +
    `<g fill="none" stroke="${color}" stroke-width="${stroke}" stroke-linecap="round" stroke-linejoin="round">${paths.join('')}</g></svg>`
}

if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
  let pen, job = 0
  self.onmessage = async ({data: m}) => {
    if (m.load) {
      pen = await Pen.load(m.load)
      postMessage({ready: true, writers: pen.writers.slice(1), vector: pen.vector, fast: !!simd})
      const words = m.words && await (await import('./words.js')).Words.load(m.words).catch(() => null), said = new Map()
      if (words) {
        pen.encoder = text => said.get(text) || said.set(text, words.encode(text)).get(text)
        postMessage({words: true})
      }
      return
    }
    const id = ++job, t0 = performance.now(), rng = mulberry(m.seed || 0)
    for (let k = 0; k < (m.draw || 1); k++) {
      const made = m.draw ? pen.draw(m.text, m.source, m.temperature, m.topK, rng, m.guidance, m.ink, m.strokes)
        : pen.write(m.text, m.writer, m.temperature, m.topK, m.seed)
      for await (const points of made) {
        if (id !== job) return
        postMessage({id: m.id, k, points}, [points.buffer])
      }
    }
    postMessage({id: m.id, done: true, ms: performance.now() - t0})
  }
}
