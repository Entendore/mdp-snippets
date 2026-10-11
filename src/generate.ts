/** Pure candidate-generation core, shared by the extension host (sync
 *  fallback) and the worker thread: hierarchical dist resolver (KN smoothing
 *  + family/universal channels + cache boost), beam search with MMR slate
 *  selection, rollouts, and the Q-state key builder. */
import { abstractToken, countNewlines, mixDistributions } from './util';
import { familyOf, profileFor } from './languages';
import { MarkovModel, type Dist, type Channel } from './markov';

export interface StateKeyOpts {
  chain?: boolean;
  mode?: 'C' | 'M' | 'R';
  region?: string;
  ws?: string;
}

export function qStateKeyOf(lang: string, litTokens: string[], order: number, indentBucket: number, atEol: boolean, opts: StateKeyOpts = {}): string {
  const t = litTokens.length > order ? litTokens.slice(litTokens.length - order) : litTokens;
  const mode = opts.mode && opts.mode !== 'C' ? opts.mode : '';
  const region = opts.region && opts.region !== 'O' ? opts.region : '';
  return `${lang}\u0002${t.map((x) => abstractToken(x, lang)).join('\u0000')}\u0001${indentBucket}${atEol ? 'E' : 'M'}${opts.chain ? 'A' : ''}${mode}${region}${opts.ws ? '\u0004' + opts.ws : ''}`;
}

export function successorMeta(tokens: string[], atEol: boolean, bucket: number): { bucket: number; atEol: boolean } {
  const endsNl = /\n[ \t]*$/.test(tokens.join(''));
  const lastNl = tokens.reduce((acc, tok, i) => (tok.includes('\n') ? i : acc), -1);
  if (lastNl < 0) return { bucket, atEol };
  let ind = 0;
  for (let i = lastNl + 1; i < tokens.length; i++) {
    const t = tokens[i];
    if (/^[ \t]+$/.test(t)) ind += t.length; else break;
  }
  return { bucket: Math.min(3, Math.floor(ind / 2)), atEol: endsNl };
}

export interface ResolverConfig {
  smoothingD: number;
  minCount: number;
  smoothing: 'interpolated' | 'kneser-ney';
  knDiscount: number;
  transferWeight: number;
  familyTransferWeight: number;
  cacheD: number;
}

export class DistResolver {
  constructor(private readonly markov: MarkovModel, private readonly cfg: ResolverConfig, private readonly channelW: Map<string, number[]>) {}

  channelsFor(lang: string): Channel[] {
    const w = this.channelW.get(lang) ?? [1, this.cfg.familyTransferWeight, this.cfg.transferWeight];
    return [
      { lang, weight: w[0] },
      { lang: familyOf(lang), weight: w[1] },
      { lang: '*', weight: w[2] },
    ].filter((c) => c.weight > 0.001);
  }

  distAt(lang: string, litCtx: string[], absCtx: string[], boost?: Map<string, number> | null, minCount?: number): Dist | undefined {
    const kn = this.cfg.smoothing === 'kneser-ney' ? this.cfg.knDiscount : 0;
    const mc = minCount ?? this.cfg.minCount;
    let d = this.markov.smoothed(this.channelsFor(lang), litCtx, absCtx, this.cfg.smoothingD, mc, kn);
    if (!d) {
      for (const ch of [familyOf(lang), '*']) {
        d = this.markov.smoothed([{ lang: ch, weight: 1 }], litCtx, absCtx, this.cfg.smoothingD, Math.max(1, mc - 1), kn);
        if (d) break;
      }
    }
    if (d && boost && boost.size > 0) d = mixDistributions(d, boost, this.cfg.cacheD);
    return d;
  }
}

export function successorPrior(resolver: DistResolver, lang: string, boost: Map<string, number> | null, baseCtx: string[], tokens: string[]): number {
  const ctx = baseCtx.concat(tokens);
  const dist = resolver.distAt(lang, ctx, ctx.map((t) => abstractToken(t, lang)), boost, 1);
  let best = 0;
  if (dist) for (const v of dist.values()) if (v > best) best = v;
  return best;
}

export interface TokenCand {
  tokens: string[];
  prior: number;
  nextPrior: number;
  isRollout: boolean;
}

export interface TokenGenParams {
  lang: string;
  litBefore: string[];
  absBefore: string[];
  prefix: string;
  atEol: boolean;
  maxLines: number;
  minCount: number;
  minConfidence: number;
  beamWidth: number;
  candidates: number;
  effMax: number;
  suggestOnNewline: boolean;
  rolloutSteps: number;
  diversityLambda: number;
  maxTokensTotal: number;
  cacheBoost: Map<string, number> | null;
}

export type TokenGenRequest = Omit<TokenGenParams, 'cacheBoost'> & { cacheBoost: [string, number][] | null };

export function requestFromParams(p: TokenGenParams): TokenGenRequest {
  const { cacheBoost, ...rest } = p;
  return { ...rest, cacheBoost: cacheBoost ? Array.from(cacheBoost.entries()) : null };
}

export function paramsFromRequest(r: TokenGenRequest): TokenGenParams {
  return { ...r, cacheBoost: r.cacheBoost ? new Map(r.cacheBoost) : null };
}

function jaccard(a: string[], b: string[]): number {
  const sa = new Set(a), sb = new Set(b);
  let inter = 0;
  for (const t of sa) if (sb.has(t)) inter++;
  const uni = sa.size + sb.size - inter;
  return uni === 0 ? 0 : inter / uni;
}

export function generateTokens(resolver: DistResolver, p: TokenGenParams): TokenCand[] {
  const profile = profileFor(p.lang);
  const boost = p.cacheBoost ?? null;

  const firstDist = resolver.distAt(p.lang, p.litBefore, p.absBefore, boost, p.minCount);
  if (!firstDist) return [];

  const acceptable = (token: string): boolean => {
    if (p.prefix.length > 0) return token.length > p.prefix.length && token.startsWith(p.prefix);
    if (token.includes('\n')) return p.atEol && p.suggestOnNewline;
    return true;
  };
  const firsts = Array.from(firstDist.entries())
    .filter(([tok, pr]) => pr >= p.minConfidence && acceptable(tok))
    .sort((a, b) => b[1] - a[1])
    .slice(0, p.beamWidth);
  if (firsts.length === 0) return [];

  const bracketDelta = (tok: string) => (tok === '{' || tok === '(' || tok === '[' ? 1 : tok === '}' || tok === ')' || tok === ']' ? -1 : 0);
  const isTerm = (tok: string, depth: number): boolean =>
    profile.terminators.has(tok) || (profile.eolTerminates && tok.includes('\n') && depth <= 0);

  interface BeamItem { tokens: string[]; logP: number; newlines: number; depth: number; done: boolean; }
  const items: BeamItem[] = firsts.map(([tok, pr]) => {
    const d = bracketDelta(tok);
    const nl = countNewlines(tok);
    return { tokens: [tok], logP: Math.log(pr + 1e-12), newlines: nl, depth: d, done: isTerm(tok, d) || d < 0 || nl >= p.maxLines };
  });

  for (let step = 0; step < p.effMax && items.some((i) => !i.done); step++) {
    const next: BeamItem[] = [];
    for (const item of items) {
      if (item.done) { next.push(item); continue; }
      const ctxLit = p.litBefore.concat(item.tokens);
      const ctxAbs = p.absBefore.concat(item.tokens.map((t) => abstractToken(t, p.lang)));
      const dist = resolver.distAt(p.lang, ctxLit, ctxAbs, boost, p.minCount);
      const top = dist
        ? Array.from(dist.entries()).filter(([, pr]) => pr >= p.minConfidence).sort((a, b) => b[1] - a[1]).slice(0, 2)
        : [];
      if (top.length === 0) { item.done = true; next.push(item); continue; }
      for (const [tok, pr] of top) {
        const depth = item.depth + bracketDelta(tok);
        const newlines = item.newlines + countNewlines(tok);
        const len = item.tokens.length + 1;
        next.push({
          tokens: [...item.tokens, tok],
          logP: item.logP + Math.log(pr + 1e-12),
          newlines,
          depth,
          done: isTerm(tok, depth) || depth < 0 || newlines >= p.maxLines || len >= p.effMax,
        });
      }
    }
    next.sort((a, b) => b.logP / b.tokens.length - a.logP / a.tokens.length);
    items.length = 0;
    items.push(...next.slice(0, p.beamWidth));
  }

  // MMR slate selection
  const pool = items.slice().sort((a, b) => b.logP / b.tokens.length - a.logP / a.tokens.length);
  const chosen: BeamItem[] = [];
  const usedIdx = new Set<number>();
  while (chosen.length < Math.min(p.candidates, pool.length)) {
    let bestI = -1;
    let bestS = -Infinity;
    for (let i = 0; i < pool.length; i++) {
      if (usedIdx.has(i)) continue;
      let s = pool[i].logP / pool[i].tokens.length;
      if (p.diversityLambda > 0 && chosen.length > 0) {
        let maxSim = 0;
        for (const c of chosen) maxSim = Math.max(maxSim, jaccard(pool[i].tokens, c.tokens));
        s -= p.diversityLambda * maxSim;
      }
      if (s > bestS) { bestS = s; bestI = i; }
    }
    usedIdx.add(bestI);
    chosen.push(pool[bestI]);
  }

  const out: TokenCand[] = [];
  const seen = new Set<string>();
  for (const item of chosen) {
    const tokens = item.tokens.slice();
    while (tokens.length > 1 && /^[ \t\r\n]+$/.test(tokens[tokens.length - 1])) tokens.pop();
    const raw = tokens.join('');
    if (raw.trim().length === 0 || raw.length < 2 || seen.has(raw)) continue;
    seen.add(raw);
    out.push({
      tokens,
      prior: firstDist.get(tokens[0]) ?? 1e-6,
      nextPrior: successorPrior(resolver, p.lang, boost, p.litBefore, tokens),
      isRollout: false,
    });
  }

  if (p.rolloutSteps > 0 && p.suggestOnNewline) {
    const roll = rolloutTokens(resolver, p);
    if (roll && !seen.has(roll.tokens.join(''))) {
      seen.add(roll.tokens.join(''));
      out.push(roll);
    }
  }
  return out;
}

function rolloutTokens(resolver: DistResolver, p: TokenGenParams): TokenCand | null {
  const profile = profileFor(p.lang);
  const boost = p.cacheBoost ?? null;
  const bracketDelta = (tok: string) => (tok === '{' || tok === '(' || tok === '[' ? 1 : tok === '}' || tok === ')' || tok === ']' ? -1 : 0);
  const isTerm = (tok: string, depth: number): boolean =>
    profile.terminators.has(tok) || (profile.eolTerminates && tok.includes('\n') && depth <= 0);

  const tokens: string[] = [];
  const ctxLit = p.litBefore.slice();
  const ctxAbs = p.absBefore.slice();
  let depth = 0, lines = 0, segments = 0, firstPrior = 0;
  const maxLinesTotal = Math.max(1, p.maxLines * p.rolloutSteps);

  while (segments < p.rolloutSteps && tokens.length < p.maxTokensTotal && lines < maxLinesTotal) {
    const dist = resolver.distAt(p.lang, ctxLit, ctxAbs, boost, p.minCount);
    if (!dist) break;
    let bestTok = '', bestP = 0;
    for (const [tok, pr] of dist) {
      if (pr <= bestP) continue;
      if (tokens.length === 0) {
        if (p.prefix.length > 0) { if (!(tok.length > p.prefix.length && tok.startsWith(p.prefix))) continue; }
        else if (tok.includes('\n') && !p.atEol) continue;
      }
      bestTok = tok; bestP = pr;
    }
    if (!bestTok || bestP < p.minConfidence) break;
    if (tokens.length === 0) firstPrior = bestP;
    tokens.push(bestTok);
    ctxLit.push(bestTok);
    ctxAbs.push(abstractToken(bestTok, p.lang));
    depth += bracketDelta(bestTok);
    lines += countNewlines(bestTok);
    if (isTerm(bestTok, depth)) segments++;
  }
  while (tokens.length > 1 && /^[ \t\r\n]+$/.test(tokens[tokens.length - 1])) tokens.pop();
  if (tokens.length === 0 || tokens.join('').trim().length === 0) return null;
  return { tokens, prior: firstPrior, nextPrior: successorPrior(resolver, p.lang, boost, p.litBefore, tokens), isRollout: true };
}
