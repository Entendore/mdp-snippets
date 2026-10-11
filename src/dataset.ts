/** Corpus learning: parsing (jsonl/json/markdown fences/txt), deterministic
 *  splitting, held-out evaluation (perplexity, top-k, OOV), and hyperparameter
 *  model selection. Corpora only ever inform the TRANSITION model. */
import { tokenize, abstractToken } from './util';
import { canonicalLang } from './languages';
import { DistResolver, type ResolverConfig } from './generate';
import { MarkovModel } from './markov';

export interface CorpusItem { lang: string; text: string; }

const TEXT_FIELDS = ['text', 'content', 'code', 'body', 'snippet', 'source'];
const LANG_FIELDS = ['lang', 'language', 'languageId', 'programmingLanguage'];

function field(obj: any, names: string[]): any {
  for (const n of names) if (obj && obj[n] !== undefined) return obj[n];
  return undefined;
}

export function parseJsonlDataset(content: string, defaultLang: string): CorpusItem[] {
  const out: CorpusItem[] = [];
  for (const line of content.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    let obj: any;
    try { obj = JSON.parse(s); } catch { continue; }
    const text = field(obj, TEXT_FIELDS);
    if (typeof text !== 'string' || !text.trim()) continue;
    const rawLang = field(obj, LANG_FIELDS);
    const lang = typeof rawLang === 'string' && rawLang ? canonicalLang(rawLang) : defaultLang;
    out.push({ lang, text });
  }
  return out;
}

export function parseJsonDataset(content: string, defaultLang: string): CorpusItem[] {
  let data: any;
  try { data = JSON.parse(content); } catch { return []; }
  const arr = Array.isArray(data) ? data
    : Array.isArray(data?.data) ? data.data
    : Array.isArray(data?.items) ? data.items
    : Array.isArray(data?.snippets) ? data.snippets
    : null;
  if (!arr) return [];
  const out: CorpusItem[] = [];
  for (const obj of arr) {
    if (typeof obj === 'string') { if (obj.trim()) out.push({ lang: defaultLang, text: obj }); continue; }
    const text = field(obj, TEXT_FIELDS);
    if (typeof text !== 'string' || !text.trim()) continue;
    const rawLang = field(obj, LANG_FIELDS);
    const lang = typeof rawLang === 'string' && rawLang ? canonicalLang(rawLang) : defaultLang;
    out.push({ lang, text });
  }
  return out;
}

const FENCE_MAP: Record<string, string> = {
  js: 'javascript', javascript: 'javascript', jsx: 'javascript', ts: 'typescript', typescript: 'typescript', tsx: 'typescript',
  py: 'python', python: 'python', rb: 'ruby', ruby: 'ruby', go: 'go', golang: 'go', rs: 'rust', rust: 'rust',
  java: 'java', c: 'c', cpp: 'cpp', 'c++': 'cpp', cs: 'csharp', csharp: 'csharp', swift: 'swift', kt: 'kotlin', kotlin: 'kotlin',
  php: 'php', scala: 'scala', dart: 'dart', sh: 'shellscript', bash: 'shellscript', shell: 'shellscript', zsh: 'shellscript',
  sql: 'sql', html: 'html', css: 'css', scss: 'scss', json: 'json', yaml: 'yaml', yml: 'yaml',
  clojure: 'clojure', clj: 'clojure', haskell: 'haskell', hs: 'haskell', elm: 'elm', fsharp: 'fsharp', fs: 'fsharp',
};

export function parseMarkdownFences(content: string, defaultLang: string): CorpusItem[] {
  const out: CorpusItem[] = [];
  const re = /```([\w+#-]*)\r?\n([\s\S]*?)```/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(content)) !== null) {
    const tag = (m[1] || '').toLowerCase();
    const lang = tag ? FENCE_MAP[tag] ?? canonicalLang(tag) : defaultLang;
    if (m[2].trim()) out.push({ lang, text: m[2] });
  }
  return out;
}

export function parseCorpusContent(filename: string, content: string, defaultLang: string): CorpusItem[] {
  const lower = filename.toLowerCase();
  if (lower.endsWith('.jsonl') || lower.endsWith('.ndjson')) return parseJsonlDataset(content, defaultLang);
  if (lower.endsWith('.json')) return parseJsonDataset(content, defaultLang);
  if (lower.endsWith('.md') || lower.endsWith('.markdown')) {
    const fenced = parseMarkdownFences(content, defaultLang);
    return fenced.length > 0 ? fenced : [{ lang: defaultLang, text: content }];
  }
  return [{ lang: defaultLang, text: content }];
}

export function datasetNeedsLanguage(filename: string, content: string): boolean {
  return parseCorpusContent(filename, content, '').some((it) => it.lang === '');
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function splitTrainVal<T>(items: T[], valFrac = 0.2, seed = 42): { train: T[]; val: T[] } {
  const rng = mulberry32(seed);
  const arr = items.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  if (arr.length < 2) return { train: arr, val: arr.slice() };
  const nVal = Math.max(1, Math.round(arr.length * valFrac));
  return { train: arr.slice(0, arr.length - nVal), val: arr.slice(arr.length - nVal) };
}

export interface EvalMetrics {
  predictions: number; logLoss: number; perplexity: number;
  top1: number; top3: number; top5: number; oovRate: number;
}

const EVAL_EPS = 1e-6;

export function evaluateSequences(resolver: DistResolver, lang: string, seqs: string[][], budget: number): EvalMetrics {
  let n = 0, ll = 0, t1 = 0, t3 = 0, t5 = 0, oov = 0;
  outer: for (const seq of seqs) {
    for (let i = 1; i < seq.length; i++) {
      if (n >= budget) break outer;
      const ctx = seq.slice(0, i);
      const dist = resolver.distAt(lang, ctx, ctx.map((t) => abstractToken(t, lang)), null, 1);
      const actual = seq[i];
      n++;
      const p = dist?.get(actual);
      if (!dist || p === undefined || p <= 0) { oov++; ll += Math.log(EVAL_EPS); continue; }
      ll += Math.log(p);
      let rank = 0;
      for (const [tok, pr] of dist) if (pr > p && tok !== actual) rank++;
      if (rank < 1) t1++;
      if (rank < 3) t3++;
      if (rank < 5) t5++;
    }
  }
  return {
    predictions: n,
    logLoss: n ? -ll / n : 0,
    perplexity: n ? Math.exp(-ll / n) : Infinity,
    top1: n ? t1 / n : 0, top3: n ? t3 / n : 0, top5: n ? t5 / n : 0,
    oovRate: n ? oov / n : 0,
  };
}

export interface TuneResult {
  order: number; smoothingD: number; knDiscount: number;
  perplexity: number; top1: number; predictions: number;
  tempModel: boolean;
}

export interface TuneOutcome {
  current: TuneResult;
  results: TuneResult[];
  best: TuneResult;
  valItems: number; trainTokens: number;
}

export function gridSearchTune(args: {
  items: CorpusItem[];
  currentModel: MarkovModel;
  currentOrder: number;
  channelW: Map<string, number[]>;
  baseCfg: ResolverConfig;
  orders: number[];
  smoothingDs: number[];
  knDiscounts: number[];
  tokenBudget: number;
  evalBudget: number;
  valFrac?: number;
}): TuneOutcome {
  const { train, val } = splitTrainVal(args.items, args.valFrac ?? 0.2);

  const trainByLang = new Map<string, string[][]>();
  let trainTokens = 0;
  for (const it of train) {
    if (trainTokens >= args.tokenBudget) break;
    const seq = tokenize(it.text, it.lang);
    if (seq.length < 2) continue;
    trainTokens += seq.length;
    let arr = trainByLang.get(it.lang);
    if (!arr) { arr = []; trainByLang.set(it.lang, arr); }
    arr.push(seq);
  }
  const valByLang = new Map<string, string[][]>();
  for (const it of val) {
    const seq = tokenize(it.text, it.lang);
    if (seq.length < 2) continue;
    let arr = valByLang.get(it.lang);
    if (!arr) { arr = []; valByLang.set(it.lang, arr); }
    arr.push(seq);
  }

  const mkResolver = (markov: MarkovModel, smoothingD: number, knDiscount: number): DistResolver =>
    new DistResolver(markov, { ...args.baseCfg, smoothingD, knDiscount, smoothing: knDiscount > 0 ? 'kneser-ney' : 'interpolated' }, args.channelW);

  const evalCombo = (markov: MarkovModel, order: number, smoothingD: number, knDiscount: number, tempModel: boolean): TuneResult => {
    let ll = 0, n = 0, t1 = 0;
    let budget = args.evalBudget;
    for (const [lang, seqs] of valByLang) {
      if (budget <= 0) break;
      const m = evaluateSequences(mkResolver(markov, smoothingD, knDiscount), lang, seqs, budget);
      ll += m.logLoss * m.predictions;
      n += m.predictions;
      t1 += m.top1 * m.predictions;
      budget -= m.predictions;
    }
    return { order, smoothingD, knDiscount, perplexity: n ? Math.exp(ll / n) : Infinity, top1: n ? t1 / n : 0, predictions: n, tempModel };
  };

  const results: TuneResult[] = [];
  for (const d of args.smoothingDs) {
    for (const k of args.knDiscounts) results.push(evalCombo(args.currentModel, args.currentOrder, d, k, false));
  }
  const bestA = results.reduce((a, b) => (b.perplexity < a.perplexity ? b : a));

  for (const order of args.orders) {
    if (order === args.currentOrder) continue;
    const m = new MarkovModel(order);
    for (const [lang, seqs] of trainByLang) {
      for (const seq of seqs) m.learn(lang, seq, 1, args.baseCfg.transferWeight, args.baseCfg.familyTransferWeight);
    }
    results.push(evalCombo(m, order, bestA.smoothingD, bestA.knDiscount, true));
  }

  results.sort((a, b) => a.perplexity - b.perplexity);
  const current = evalCombo(args.currentModel, args.currentOrder, args.baseCfg.smoothingD, args.baseCfg.knDiscount > 0 ? args.baseCfg.knDiscount : 0, false);
  return { current, results, best: results[0], valItems: val.length, trainTokens };
}
