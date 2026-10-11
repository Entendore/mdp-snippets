/** Transition model: dual-channel (literal + abstract) order-k Markov chain
 *  with language -> family -> universal channels, Kneser-Ney continuation
 *  backoff, decay, merging, per-language drop, worker snapshots. */
import { abstractToken } from './util';
import { canonicalLang, familyOf } from './languages';

const SEP = '\u0000';
const LSEP = '\u0002';
export type Dist = Map<string, number>;
type Level = Map<string, Dist>;

export interface Channel { lang: string; weight: number; }

function bump(level: Level, key: string, token: string, weight: number): boolean {
  let dist = level.get(key);
  if (!dist) { dist = new Map(); level.set(key, dist); }
  const isNew = !dist.has(token);
  dist.set(token, (dist.get(token) ?? 0) + weight);
  return isNew;
}

export class MarkovModel {
  readonly order: number;
  readonly lit: Level[] = [];
  readonly abs: Level[] = [];
  readonly contLit: Map<string, number>[] = [];
  readonly contAbs: Map<string, number>[] = [];
  pairsLearned = 0;

  constructor(order: number) {
    this.order = Math.max(1, Math.min(8, Math.floor(order) || 1));
    for (let i = 0; i <= this.order; i++) {
      this.lit.push(new Map()); this.abs.push(new Map());
      this.contLit.push(new Map()); this.contAbs.push(new Map());
    }
  }

  private key(lang: string, ctx: string[]): string {
    return lang + LSEP + ctx.join(SEP);
  }

  /** Learn into the language channel (all levels) and, at reduced weight,
   *  into the family and universal channels (levels <= 2). */
  learn(lang: string, tokens: string[], weight = 1, universalW = 0, familyW = 0): void {
    if (tokens.length < 2) return;
    const canon = canonicalLang(lang) || lang;
    const family = familyOf(canon);
    const abs = tokens.map((t) => abstractToken(t, canon));
    const channels: Array<[string, number, number]> = [
      [canon, weight, this.order],
      [family, weight * (familyW || 0), Math.min(2, this.order)],
      ['*', weight * (universalW || 0), Math.min(2, this.order)],
    ];
    for (const [cl, cw, maxLevel] of channels) {
      if (cw <= 0) continue;
      for (let i = 1; i < tokens.length; i++) {
        for (let L = 1; L <= maxLevel; L++) {
          if (i - L < 0) break;
          if (bump(this.lit[L], this.key(cl, tokens.slice(i - L, i)), tokens[i], cw)) {
            const ck = cl + LSEP + tokens[i];
            this.contLit[L].set(ck, (this.contLit[L].get(ck) ?? 0) + 1);
          }
          if (bump(this.abs[L], this.key(cl, abs.slice(i - L, i)), tokens[i], cw)) {
            const ck = cl + LSEP + tokens[i];
            this.contAbs[L].set(ck, (this.contAbs[L].get(ck) ?? 0) + 1);
          }
        }
      }
    }
    this.pairsLearned += tokens.length - 1;
  }

  litAt(lang: string, ctx: string[], level: number): Dist | undefined {
    if (level < 1 || level > this.order || ctx.length < level) return undefined;
    const d = this.lit[level].get(this.key(lang, ctx.slice(ctx.length - level)));
    return d && d.size ? d : undefined;
  }

  absAt(lang: string, ctx: string[], level: number): Dist | undefined {
    if (level < 1 || level > this.order || ctx.length < level) return undefined;
    const d = this.abs[level].get(this.key(lang, ctx.slice(ctx.length - level)));
    return d && d.size ? d : undefined;
  }

  /** Interpolated next-token distribution across channels (string = single
   *  weight-1 channel). KN discounting + continuation backoff optional. */
  smoothed(langOrChannels: string | Channel[], litCtx: string[], absCtx: string[], d: number, minCount: number, knDiscount = 0): Dist | undefined {
    const channels: Channel[] = typeof langOrChannels === 'string'
      ? [{ lang: langOrChannels, weight: 1 }]
      : langOrChannels.filter((c) => c.weight > 0.001);
    if (channels.length === 0) return undefined;
    const mix = new Map<string, number>();
    let wTotal = 0;
    const addDist = (dist: Dist | undefined, chW: number): void => {
      if (!dist || chW <= 0) return;
      let c = 0;
      for (const v of dist.values()) c += v;
      if (c < minCount) return;
      const w = chW * (c / (c + d));
      wTotal += w;
      if (knDiscount > 0) {
        for (const [tok, cnt] of dist) {
          const adj = Math.max(0, cnt - knDiscount);
          if (adj > 0) mix.set(tok, (mix.get(tok) ?? 0) + (w * adj) / c);
        }
      } else {
        for (const [tok, cnt] of dist) mix.set(tok, (mix.get(tok) ?? 0) + (w * cnt) / c);
      }
    };
    const maxL = Math.min(this.order, litCtx.length, absCtx.length);
    for (let L = maxL; L >= 1; L--) {
      for (const ch of channels) {
        addDist(this.litAt(ch.lang, litCtx, L), ch.weight);
        addDist(this.absAt(ch.lang, absCtx, L), ch.weight);
      }
    }
    if (knDiscount > 0) {
      const addCont = (cc: Map<string, number> | undefined, chW: number): void => {
        if (!cc || cc.size === 0) return;
        let total = 0;
        for (const n of cc.values()) if (n >= 2) total += n;
        if (total === 0) return;
        const w = 0.2 * chW;
        wTotal += w;
        for (const [tok, n] of cc) if (n >= 2) mix.set(tok, (mix.get(tok) ?? 0) + (w * n) / total);
      };
      for (const ch of channels) {
        addCont(this.filterCont(this.contLit[1], ch.lang), ch.weight);
        addCont(this.filterCont(this.contAbs[1], ch.lang), ch.weight);
      }
    }
    if (wTotal === 0 || mix.size === 0) return undefined;
    const out = new Map<string, number>();
    for (const [tok, m] of mix) out.set(tok, m / wTotal);
    return out;
  }

  private filterCont(cc: Map<string, number>, lang: string): Map<string, number> {
    const prefix = lang + LSEP;
    const out = new Map<string, number>();
    for (const [k, v] of cc) if (k.startsWith(prefix)) out.set(k.slice(prefix.length), v);
    return out;
  }

  /** All channel languages present in the model (incl. families and '*'). */
  languages(): string[] {
    const langs = new Set<string>();
    for (const key of this.lit[1].keys()) {
      const i = key.indexOf(LSEP);
      if (i > 0) langs.add(key.slice(0, i));
    }
    return Array.from(langs);
  }

  bigramsFor(lang: string): number {
    const prefix = lang + LSEP;
    let n = 0;
    for (const [k, dist] of this.lit[1]) if (k.startsWith(prefix)) n += dist.size;
    return n;
  }

  /** Remove one language's data; family/universal channels are kept. */
  dropLanguage(lang: string): void {
    const prefix = lang + LSEP;
    for (const levels of [this.lit, this.abs]) {
      for (const table of levels) {
        for (const k of Array.from(table.keys())) if (k.startsWith(prefix)) table.delete(k);
      }
    }
    for (const ccs of [this.contLit, this.contAbs]) {
      for (const cc of ccs) {
        for (const k of Array.from(cc.keys())) if (k.startsWith(prefix)) cc.delete(k);
      }
    }
  }

  decay(factor: number): void {
    for (const levels of [this.lit, this.abs]) {
      for (const table of levels) {
        for (const [key, dist] of table) {
          for (const [tok, c] of dist) {
            const n = c * factor;
            if (n < 0.25) dist.delete(tok); else dist.set(tok, n);
          }
          if (dist.size === 0) table.delete(key);
        }
      }
    }
  }

  mergeFrom(other: MarkovModel, weight = 1): void {
    for (const [mine, theirs] of [[this.lit, other.lit], [this.abs, other.abs]] as [Level[], Level[]][]) {
      for (let L = 1; L < mine.length && L < theirs.length; L++) {
        for (const [key, dist] of theirs[L]) {
          let m = mine[L].get(key);
          if (!m) { m = new Map(); mine[L].set(key, m); }
          for (const [tok, c] of dist) m.set(tok, (m.get(tok) ?? 0) + c * weight);
        }
      }
    }
    for (const [mine, theirs] of [[this.contLit, other.contLit], [this.contAbs, other.contAbs]] as [Map<string, number>[], Map<string, number>[]][]) {
      for (let L = 1; L < mine.length && L < theirs.length; L++) {
        for (const [k, v] of theirs[L]) mine[L].set(k, Math.max(mine[L].get(k) ?? 0, v));
      }
    }
    this.pairsLearned += other.pairsLearned * weight;
  }

  distinctBigrams(): number {
    let n = 0;
    for (const dist of this.lit[1].values()) n += dist.size;
    return n;
  }

  contextStates(): number {
    let n = 0;
    for (let L = 1; L <= this.order; L++) n += this.lit[L].size + this.abs[L].size;
    return n;
  }

  prune(capPerLevel: number): void {
    for (const levels of [this.lit, this.abs]) {
      for (let L = 1; L < levels.length; L++) {
        const table = levels[L];
        let threshold = 1;
        while (table.size > capPerLevel && threshold < 50) {
          for (const [key, dist] of table) {
            let total = 0;
            for (const c of dist.values()) total += c;
            if (total <= threshold) table.delete(key);
          }
          threshold++;
        }
      }
    }
    for (const ccs of [this.contLit, this.contAbs]) {
      for (const cc of ccs) {
        if (cc.size > 200_000) {
          const top = Array.from(cc.entries()).sort((a, b) => b[1] - a[1]).slice(0, 200_000);
          cc.clear();
          for (const [k, v] of top) cc.set(k, v);
        }
      }
    }
  }

  /** Live-field snapshot for the worker (structured-clone friendly). */
  exportFields(): any {
    return {
      order: this.order,
      lit: this.lit,
      abs: this.abs,
      contLit: this.contLit,
      contAbs: this.contAbs,
      pairsLearned: this.pairsLearned,
    };
  }

  static hydrateFields(f: any): MarkovModel {
    const order = Math.max(1, Math.min(8, Math.floor(Number(f?.order) || 3)));
    const m = new MarkovModel(order);
    const setLevels = (dst: any[], src: any): void => {
      if (Array.isArray(src)) {
        for (let i = 0; i < Math.min(src.length, dst.length); i++) {
          if (src[i] instanceof Map) dst[i] = src[i];
        }
      }
    };
    setLevels(m.lit, f?.lit);
    setLevels(m.abs, f?.abs);
    setLevels(m.contLit, f?.contLit);
    setLevels(m.contAbs, f?.contAbs);
    if (typeof f?.pairsLearned === 'number') m.pairsLearned = f.pairsLearned;
    return m;
  }

  /** Serialize only channels passing `filter` (taste profiles keep
   *  family + universal channels only). */
  exportChannels(filter: (channelLang: string) => boolean): any {
    const chanOf = (k: string): string => {
      const i = k.indexOf(LSEP);
      return i > 0 ? k.slice(0, i) : '';
    };
    const pack = (levels: Level[]) =>
      levels.map((m) =>
        Array.from(m.entries())
          .filter(([k]) => filter(chanOf(k)))
          .map(([k, d]) => [k, Array.from(d.entries()).map(([t, c]) => [t, Math.round(c * 1e3) / 1e3])])
      );
    const packCont = (ccs: Map<string, number>[]) =>
      ccs.map((m) => Array.from(m.entries()).filter(([k]) => filter(chanOf(k))));
    return { order: this.order, pairsLearned: 0, lit: pack(this.lit), abs: pack(this.abs), contLit: packCont(this.contLit), contAbs: packCont(this.contAbs) };
  }

  serialize(): any {
    const r3 = (x: number): number => Math.round(x * 1e3) / 1e3;
    const pack = (levels: Level[]) =>
      levels.map((m) => Array.from(m.entries()).map(([k, d]) => [k, Array.from(d.entries()).map(([t, c]) => [t, r3(c)])]));
    const packCont = (ccs: Map<string, number>[]) => ccs.map((m) => Array.from(m.entries()));
    return { order: this.order, pairsLearned: this.pairsLearned, lit: pack(this.lit), abs: pack(this.abs), contLit: packCont(this.contLit), contAbs: packCont(this.contAbs) };
  }

  static deserialize(data: any): MarkovModel | null {
    try {
      if (!data || typeof data.order !== 'number') return null;
      const model = new MarkovModel(data.order);
      const unpack = (levels: Level[], raw: any) => {
        if (!Array.isArray(raw)) return;
        for (let L = 0; L < raw.length && L < levels.length; L++) {
          for (const entry of raw[L]) {
            if (!Array.isArray(entry) || typeof entry[0] !== 'string') continue;
            const dist = new Map<string, number>();
            for (const pair of entry[1]) {
              if (Array.isArray(pair) && typeof pair[0] === 'string' && typeof pair[1] === 'number') dist.set(pair[0], pair[1]);
            }
            if (dist.size) levels[L].set(entry[0], dist);
          }
        }
      };
      unpack(model.lit, data.lit ?? data.tables);
      unpack(model.abs, data.abs);
      const unpackCont = (ccs: Map<string, number>[], raw: any) => {
        if (!Array.isArray(raw)) return;
        for (let L = 0; L < raw.length && L < ccs.length; L++) {
          for (const e of raw[L]) {
            if (Array.isArray(e) && typeof e[0] === 'string' && typeof e[1] === 'number') ccs[L].set(e[0], e[1]);
          }
        }
      };
      unpackCont(model.contLit, data.contLit);
      unpackCont(model.contAbs, data.contAbs);
      model.pairsLearned = typeof data.pairsLearned === 'number' ? data.pairsLearned : 0;
      return model;
    } catch {
      return null;
    }
  }
}
