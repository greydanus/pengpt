import {npz, backend, accelerate} from './pen.js'

const SPLIT = /'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+/gu
const BYTES = (() => {
  const seen = [...Array(256).keys()].filter(b => (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174)
  let n = 0
  return Array.from({length: 256}, (_, b) => String.fromCharCode(seen.includes(b) ? b : 256 + n++))
})()

function erf(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return x < 0 ? -y : y
}

function norm(x, weight, bias, eps) {
  const n = x.length, out = new Float32Array(n)
  let mean = 0, v = 0
  for (let i = 0; i < n; i++) mean += x[i] / n
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2 / n
  const k = 1 / Math.sqrt(v + eps)
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * k * weight[i] + bias[i]
  return out
}

export class Words {

  static async load(url) {
    const words = new Words(await npz(await (await fetch(url)).arrayBuffer()))
    return accelerate(words).catch(() => words)
  }

  constructor(z) {
    Object.assign(this, JSON.parse(z.meta.data[0]))
    const w = this.w = {}, t = JSON.parse(new TextDecoder().decode(z.tokenizer.data))
    for (const [k, a] of Object.entries(z)) {
      if (['meta', 'tokenizer'].includes(k) || k.endsWith('.scale')) continue
      w[k] = a.shape.length === 2 ? {q: a.data, s: z[k + '.scale'].data, r: a.shape[0], c: a.shape[1]} : a.data
    }
    this.ids = new Map(t.vocab.map((token, i) => [token, i]))
    this.ranks = new Map(t.merges.map((pair, i) => [pair, i]))
  }

  tokens(text) {
    const out = [this.bos]
    for (const word of text.normalize('NFC').replace(/\s+/g, ' ').toLowerCase().match(SPLIT) || []) {
      const s = [...new TextEncoder().encode(word)].map(b => BYTES[b])
      s[s.length - 1] += '</w>'
      for (;;) {
        let best = Infinity, at = -1
        for (let i = 0; i < s.length - 1; i++) {
          const r = this.ranks.get(s[i] + ' ' + s[i + 1])
          if (r < best) [best, at] = [r, i]
        }
        if (at < 0) break
        const [a, b] = [s[at], s[at + 1]]
        for (let i = 0; i < s.length - 1; i++) if (s[i] === a && s[i + 1] === b) s.splice(i, 2, a + b)
      }
      out.push(...s.map(u => this.ids.get(u) ?? this.eos))
    }
    return [...out.slice(0, this.positions - 1), this.eos]
  }

  linear(name, x) {
    const out = backend.matvec(this.w[name + '.weight'], x), b = this.w[name + '.bias']
    for (let i = 0; i < out.length; i++) out[i] += b[i]
    return out
  }

  encode(text) {
    const w = this.w, ids = this.tokens(text), C = this.width, H = this.n_head, hd = C / H
    const E = w['embeddings.token_embedding.weight'], P = w['embeddings.position_embedding.weight']
    let x = ids.map((id, t) => Float32Array.from({length: C}, (_, d) => E.q[id * C + d] * E.s[id] + P.q[t * C + d] * P.s[t]))
    for (let l = 0; l < this.n_layer; l++) {
      const p = `encoder.layers.${l}.`
      const h = x.map(u => norm(u, w[p + 'layer_norm1.weight'], w[p + 'layer_norm1.bias'], this.eps))
      const [q, k, v] = ['q', 'k', 'v'].map(n => h.map(u => this.linear(`${p}self_attn.${n}_proj`, u)))
      x = x.map((u, t) => {
        const a = new Float32Array(C)
        for (let j = 0; j < H; j++) {
          const s = Float32Array.from({length: t + 1}, (_, i) => {
            let dot = 0
            for (let d = j * hd; d < (j + 1) * hd; d++) dot += q[t][d] * k[i][d]
            return dot * hd ** -0.5
          })
          const top = Math.max(...s)
          let total = 0
          for (let i = 0; i <= t; i++) total += s[i] = Math.exp(s[i] - top)
          for (let i = 0; i <= t; i++) for (let d = j * hd; d < (j + 1) * hd; d++) a[d] += s[i] / total * v[i][d]
        }
        const y = this.linear(p + 'self_attn.out_proj', a).map((o, d) => o + u[d])
        const m = this.linear(p + 'mlp.fc1', norm(y, w[p + 'layer_norm2.weight'], w[p + 'layer_norm2.bias'], this.eps))
        for (let d = 0; d < m.length; d++) m[d] *= 0.5 * (1 + erf(m[d] / Math.SQRT2))
        return this.linear(p + 'mlp.fc2', m).map((o, d) => o + y[d])
      })
    }
    const out = this.linear('head', norm(x[x.length - 1], w['final_layer_norm.weight'], w['final_layer_norm.bias'], this.eps))
    const length = Math.hypot(...out)
    return out.map(u => u / length)
  }
}
