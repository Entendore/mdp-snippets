/** Decision-making core: TD(lambda) mean values (fixed / 1-N / Kalman-gain
 *  alpha), per-(s,a) reward histograms (P(win), CVaR, Beta Thompson),
 *  double-Q planning, eligibility traces, advantage learning, per-language
 *  preference heads, low-rank factor value model, risk-aware ranking,
 *  prioritized sweeping, drift detection, exploit lock, off-policy
 *  propensities, semantic-shape action keys with per-literal residuals,
 *  taste-profile export/merge, pre-v9 key migration, and offline fitted-Q. */
import { shapeKeyOf } from './util';

const STAT_KEYS = ['accepts', 'dismissals', 'partials', 'updates', 'explorations', 'gated', 'gatedProbes', 'chains', 'escapes', 'overridden', 'replays', 'dyna', 'sweeps', 'driftAlarms', 'factorTrainings', 'undos', 'corrections', 'confirmations', 'opeChecks', 'exploitLocks', 'thumbUps', 'thumbDowns', 'fittedQ', 'shapeMerges', 'gateRelaxations'] as const;
export type StatKey = (typeof STAT_KEYS)[number];
export type LearnerStats = Record<StatKey, number>;

function zeroStats(): LearnerStats {
  const s = {} as LearnerStats;
  for (const k of STAT_KEYS) s[k] = 0;
  return s;
}

export interface LearnerParams {
  alpha: number;
  alphaMode: 'fixed' | 'adaptive' | 'kalman';
  gamma: number;
  lambda: number;
  epsilon: number;
  beta: number;
  exploration: 'epsilon' | 'ucb' | 'softmax' | 'thompson';
  ucbC: number;
  temperature: number;
  thompsonSigma: number;
  weightAlpha: number;
  valueObjective: 'discounted' | 'average';
  rhoAlpha: number;
  processNoise: number;
  obsNoise: number;
  factorDim: number;
  factorEta: number;
  riskKappa: number;
  driftThreshold: number;
  sweepPerUpdate: number;
  rankBy?: 'value' | 'pwin' | 'cvar';
  cvarAlpha?: number;
  distEta?: number;
  literalBonus?: boolean;
}

export interface Scorable {
  key: string;
  prior: number;
  features: number[];
  lang?: string;
  literalKey?: string;
}

export interface Explanation {
  tabular: number;
  shadow: number | null;
  factor: number | null;
  head: number;
  value: number;
  sigma: number;
  visits: number;
  fallback: boolean;
  bonus: number;
}

/** A stored transition (persisted replay buffer / offline batch phase). */
export interface OfflineTransition {
  stateKey: string;
  actionKey: string;
  reward: number;
  nextStateKey: string;
  steps: number;
  features?: number[];
  literalKey?: string;
}

interface Emb { v: number[]; n: number; }
interface PairMeta { stateKey: string; actionKey: string; nextStateKey: string; steps: number; lastReward: number; }

const QSEP = '\u0001';
const REPLAY_ALPHA = 0.15;
const VAR0 = 1.0;
const UNIFORM = [0.2, 0.2, 0.2, 0.2, 0.2];
const BIN_EDGES = [-0.35, -0.02, 0.02, 0.6];
const BIN_CENTERS = [-0.8, -0.15, 0.0, 0.3, 1.0];

function binOf(r: number): number {
  if (r < BIN_EDGES[0]) return 0;
  if (r < BIN_EDGES[1]) return 1;
  if (r < BIN_EDGES[2]) return 2;
  if (r < BIN_EDGES[3]) return 3;
  return 4;
}

function cvarHist(h: number[], alpha: number): number {
  let rem = alpha, acc = 0;
  for (let b = 0; b < 5 && rem > 1e-9; b++) {
    const take = Math.min(h[b], rem);
    acc += take * BIN_CENTERS[b];
    rem -= take;
  }
  return acc / alpha;
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

function gauss(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function gammaRand(shape: number): number {
  if (shape < 1) return gammaRand(shape + 1) * Math.pow(Math.random(), 1 / shape);
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * shape);
  for (;;) {
    let x: number, v: number;
    do { x = gauss(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function randnVec(d: number, scale: number): number[] {
  const v = new Array<number>(d);
  for (let i = 0; i < d; i++) v[i] = gauss() * scale;
  return v;
}

function clipNorm(v: number[], maxNorm: number): void {
  let s = 0;
  for (const x of v) s += x * x;
  if (s > maxNorm * maxNorm) {
    const k = maxNorm / Math.sqrt(s);
    for (let i = 0; i < v.length; i++) v[i] *= k;
  }
}

class PriorityHeap {
  private items: { k: string; p: number }[] = [];
  get size(): number { return this.items.length; }
  push(k: string, p: number): void {
    if (this.items.length >= 4000) {
      let minI = 0;
      for (let i = 1; i < this.items.length; i++) if (this.items[i].p < this.items[minI].p) minI = i;
      if (p <= this.items[minI].p) return;
      this.items[minI] = this.items[this.items.length - 1];
      this.items.pop();
      this.down(minI);
    }
    this.items.push({ k, p });
    let i = this.items.length - 1;
    while (i > 0) {
      const par = (i - 1) >> 1;
      if (this.items[par].p >= this.items[i].p) break;
      [this.items[par], this.items[i]] = [this.items[i], this.items[par]];
      i = par;
    }
  }
  pop(): { k: string; p: number } | undefined {
    if (!this.items.length) return undefined;
    const top = this.items[0];
    const last = this.items.pop()!;
    if (this.items.length) { this.items[0] = last; this.down(0); }
    return top;
  }
  private down(i: number): void {
    const n = this.items.length;
    for (;;) {
      const l = 2 * i + 1, r = l + 1;
      let m = i;
      if (l < n && this.items[l].p > this.items[m].p) m = l;
      if (r < n && this.items[r].p > this.items[m].p) m = r;
      if (m === i) return;
      [this.items[i], this.items[m]] = [this.items[m], this.items[i]];
      i = m;
    }
  }
}

export class QLearner {
  readonly q = new Map<string, number>();
  readonly qShadow = new Map<string, number>();
  readonly dist = new Map<string, number[]>();
  readonly visits = new Map<string, number>();
  readonly stateVisits = new Map<string, number>();
  readonly varOf = new Map<string, number>();
  readonly stateValue = new Map<string, number>();
  readonly weightsByLang = new Map<string, number[]>();
  readonly stateEmb = new Map<string, Emb>();
  readonly actionEmb = new Map<string, Emb>();
  readonly factorSteps = new Map<string, number>();
  readonly literalBonus = new Map<string, number>();
  weights: number[] = [];
  rho = 0;
  exploitLock = false;
  stats: LearnerStats = zeroStats();

  private readonly traces = new Map<string, number>();
  private readonly pairMeta = new Map<string, PairMeta>();
  private readonly predecessors = new Map<string, string[]>();
  private readonly heap = new PriorityHeap();
  private nRewards = 0;
  private emaFast = 0;
  private emaSlow = 0;
  private noiseBoost = 0;
  private updatesSinceAlarm = 0;
  private sweepParity = true;

  constructor(private params: LearnerParams) {}

  updateParams(p: LearnerParams): void { this.params = p; }

  get driftActive(): boolean { return this.noiseBoost > 0.05; }

  getQ(state: string, action: string): number {
    return this.q.get(state + QSEP + action) ?? 0;
  }

  private get useBonus(): boolean { return this.params.literalBonus !== false; }

  private bonusOf(a: Scorable): number {
    return this.useBonus && a.literalKey ? this.literalBonus.get(a.literalKey) ?? 0 : 0;
  }

  private histOf(state: string, actionKey: string, fallbackState?: string): number[] {
    const h = this.dist.get(state + QSEP + actionKey);
    if (h) return h;
    if (fallbackState && fallbackState !== state) return this.dist.get(fallbackState + QSEP + actionKey) ?? UNIFORM;
    return UNIFORM;
  }

  pWinOf(state: string, actionKey: string, fallbackState?: string): number {
    const h = this.histOf(state, actionKey, fallbackState);
    return h[3] + h[4];
  }

  cvarOf(state: string, actionKey: string, alpha?: number, fallbackState?: string): number {
    const h = this.histOf(state, actionKey, fallbackState);
    return cvarHist(h, alpha ?? this.params.cvarAlpha ?? 0.25);
  }

  weightsFor(lang?: string): number[] {
    if (!lang) return this.weights;
    let w = this.weightsByLang.get(lang);
    if (!w) { w = this.weights.slice(); this.weightsByLang.set(lang, w); }
    return w;
  }

  private headValue(a: Scorable): number {
    return dot(this.weightsFor(a.lang), a.features);
  }

  factorPredict(state: string, actionKey: string): number | null {
    const zs = this.stateEmb.get(state);
    const za = this.actionEmb.get(actionKey);
    return zs && za ? dot(zs.v, za.v) : null;
  }

  private explainRaw(state: string, a: Scorable): Explanation {
    const k = state + QSEP + a.key;
    const n = this.visits.get(k) ?? 0;
    const tab = this.q.get(k) ?? 0;
    const sh = this.qShadow.get(k);
    let mu = 0;
    let prec = 0;
    if (n > 0) {
      const v = Math.max(0.02, this.varOf.get(k) ?? VAR0);
      mu = sh !== undefined ? (2 * tab + sh) / 3 : tab;
      prec = 1 / v;
    }
    const zs = this.stateEmb.get(state);
    const za = this.actionEmb.get(a.key);
    let factor: number | null = null;
    if (zs && za && zs.n > 0 && za.n > 0) {
      factor = dot(zs.v, za.v);
      const fp = 0.5 * Math.min(1, Math.min(zs.n, za.n) / 8);
      mu = (mu * prec + factor * fp) / (prec + fp);
      prec += fp;
    }
    const head = this.headValue(a);
    const sigma = prec > 0 ? Math.min(2, 1 / Math.sqrt(prec)) : 1;
    return { tabular: tab, shadow: sh ?? null, factor, head, value: mu + head, sigma, visits: n, fallback: false, bonus: 0 };
  }

  /** Workspace conditioning: unvisited suffixed states read through to the
   *  unsuffixed global state with 1.5x inflated sigma. */
  explain(state: string, a: Scorable, fallbackState?: string): Explanation {
    const e = this.explainRaw(state, a);
    const bonus = this.bonusOf(a);
    if (e.visits > 0 || !fallbackState || fallbackState === state) {
      return { ...e, value: e.value + bonus, bonus };
    }
    const f = this.explainRaw(fallbackState, a);
    return { ...f, value: f.value + bonus, sigma: Math.min(2, f.sigma * 1.5), visits: 0, fallback: true, bonus };
  }

  estimate(state: string, a: Scorable, fallbackState?: string): { value: number; sigma: number; visits: number } {
    const e = this.explain(state, a, fallbackState);
    return { value: e.value, sigma: e.sigma, visits: e.visits };
  }

  qhat(state: string, a: Scorable): number {
    return this.estimate(state, a).value;
  }

  /** Dyna's reward model: baseline + preference head. */
  surrogateReward(stateKey: string, a: Scorable): number {
    return (this.stateValue.get(stateKey) ?? 0) + this.headValue(a);
  }

  update(state: string, action: Scorable, reward: number, maxNextQ: number, actionSteps = 1, nextStateKey?: string): number {
    const k = state + QSEP + action.key;
    const n = (this.visits.get(k) ?? 0) + 1;
    this.visits.set(k, n);
    this.stateVisits.set(state, (this.stateVisits.get(state) ?? 0) + 1);

    // distributional channel
    const h = this.dist.get(k) ?? UNIFORM.slice();
    const eta = this.params.distEta ?? 0.15;
    for (let i = 0; i < 5; i++) h[i] *= 1 - eta;
    h[binOf(reward)] += eta;
    this.dist.set(k, h);

    // Kalman uncertainty channel
    const vPred = Math.min(4, (this.varOf.get(k) ?? VAR0) + this.params.processNoise * (1 + 9 * this.noiseBoost));
    const gain = vPred / (vPred + this.params.obsNoise);
    this.varOf.set(k, (1 - gain) * vPred);

    const alpha =
      this.params.alphaMode === 'kalman' ? Math.max(0.05, Math.min(0.9, gain)) :
      this.params.alphaMode === 'adaptive' ? 1 / n :
      this.params.alpha;

    // replacing traces
    for (const [tk, tv] of this.traces) {
      const nv = tv * this.params.gamma * this.params.lambda;
      if (nv < 0.01) this.traces.delete(tk); else this.traces.set(tk, nv);
    }
    this.traces.set(k, 1);
    if (this.traces.size > 200) {
      let worstK = '', worstV = Infinity;
      for (const [tk, tv] of this.traces) if (tv < worstV) { worstV = tv; worstK = tk; }
      if (worstK) this.traces.delete(worstK);
    }

    const average = this.params.valueObjective === 'average';
    const disc = average ? 1 : Math.pow(this.params.gamma, Math.max(1, actionSteps));
    const rhoTerm = average ? -this.rho : 0;
    const bonus = this.bonusOf(action);
    const delta = reward + rhoTerm + disc * maxNextQ - ((this.q.get(k) ?? 0) + bonus);
    for (const [tk, tv] of this.traces) this.q.set(tk, (this.q.get(tk) ?? 0) + alpha * delta * tv);

    // per-literal residual on top of the shared shape value
    if (action.literalKey && this.useBonus && delta !== 0) {
      const b = this.literalBonus.get(action.literalKey) ?? 0;
      this.literalBonus.set(action.literalKey, Math.max(-2, Math.min(2, b + 0.05 * delta)));
    }

    // advantage learning
    const b0 = this.stateValue.get(state) ?? 0;
    this.headUpdate(action, reward - b0);
    this.stateValue.set(state, b0 + 0.1 * (reward - b0));

    if (average) this.rho += this.params.rhoAlpha * (reward - this.rho);

    this.trainFactor(state, action.key, reward + rhoTerm + disc * maxNextQ);
    this.watchDrift(reward);

    if (nextStateKey) this.registerTransition(state, action.key, nextStateKey, actionSteps, reward);
    this.pushPreds(state, Math.abs(delta));
    if (this.params.sweepPerUpdate > 0) this.sweep(this.params.sweepPerUpdate);

    this.stats.updates++;
    return this.q.get(k) ?? 0;
  }

  private headUpdate(action: Scorable, advantage: number): void {
    if (!action.features.length) return;
    if (this.weights.length < action.features.length) {
      this.weights = [...this.weights, ...new Array(action.features.length - this.weights.length).fill(0)];
    }
    const step = (w: number[], rate: number): void => {
      const err = advantage - dot(w, action.features);
      for (let i = 0; i < w.length; i++) w[i] = Math.max(-10, Math.min(10, w[i] + rate * err * action.features[i]));
    };
    const local = action.lang ? this.weightsFor(action.lang) : this.weights;
    step(local, this.params.weightAlpha);
    if (local !== this.weights) step(this.weights, this.params.weightAlpha * 0.25);
  }

  private trainFactor(state: string, actionKey: string, target: number): void {
    const d = this.params.factorDim;
    if (d <= 0) return;
    let zs = this.stateEmb.get(state);
    if (!zs) { if (this.stateEmb.size >= 20_000) return; zs = { v: randnVec(d, 0.1), n: 0 }; this.stateEmb.set(state, zs); }
    let za = this.actionEmb.get(actionKey);
    if (!za) { if (this.actionEmb.size >= 20_000) return; za = { v: randnVec(d, 0.1), n: 0 }; this.actionEmb.set(actionKey, za); }
    const err = target - dot(zs.v, za.v);
    const zsOld = zs.v.slice();
    const zaOld = za.v.slice();
    for (let i = 0; i < d; i++) {
      zs.v[i] = zsOld[i] + this.params.factorEta * err * zaOld[i];
      za.v[i] = zaOld[i] + this.params.factorEta * err * zsOld[i];
    }
    clipNorm(zs.v, 5);
    clipNorm(za.v, 5);
    zs.n++; za.n++;
    const k = state + QSEP + actionKey;
    this.factorSteps.set(k, (this.factorSteps.get(k) ?? 0) + 1);
    this.stats.factorTrainings++;
  }

  private watchDrift(reward: number): void {
    this.nRewards++;
    this.updatesSinceAlarm++;
    this.emaFast += 0.2 * (reward - this.emaFast);
    this.emaSlow += 0.01 * (reward - this.emaSlow);
    this.noiseBoost *= 0.99;
    if (this.nRewards > 40 && this.updatesSinceAlarm > 20 && Math.abs(this.emaFast - this.emaSlow) > this.params.driftThreshold) {
      this.stats.driftAlarms++;
      this.noiseBoost = 1;
      this.emaFast = this.emaSlow;
      this.updatesSinceAlarm = 0;
      this.exploitLock = false; // world changed — re-explore
    }
  }

  private registerTransition(stateKey: string, actionKey: string, nextStateKey: string, steps: number, reward: number): void {
    const k = stateKey + QSEP + actionKey;
    this.pairMeta.set(k, { stateKey, actionKey, nextStateKey, steps, lastReward: reward });
    if (this.pairMeta.size > 30_000) {
      const first = this.pairMeta.keys().next().value;
      if (first !== undefined) this.pairMeta.delete(first);
    }
    let preds = this.predecessors.get(nextStateKey);
    if (!preds) { preds = []; this.predecessors.set(nextStateKey, preds); }
    if (!preds.includes(k)) { preds.push(k); if (preds.length > 24) preds.shift(); }
    if (this.predecessors.size > 30_000) {
      const first = this.predecessors.keys().next().value;
      if (first !== undefined) this.predecessors.delete(first);
    }
  }

  private pushPreds(state: string, priority: number): void {
    if (priority < 0.01) return;
    const preds = this.predecessors.get(state);
    if (!preds) return;
    for (const pairKey of preds) this.heap.push(pairKey, priority);
  }

  private maxQOf(table: Map<string, number>, state: string): number | null {
    const prefix = state + QSEP;
    let best: number | null = null;
    for (const [k, v] of table) if (k.startsWith(prefix) && (best === null || v > best)) best = v;
    return best;
  }

  /** Double-Q prioritized sweeping: alternate tables, bootstrap off the other. */
  sweep(budget: number): void {
    while (budget-- > 0) {
      const top = this.heap.pop();
      if (!top) return;
      const meta = this.pairMeta.get(top.k);
      if (!meta) continue;
      this.sweepParity = !this.sweepParity;
      const table = this.sweepParity ? this.qShadow : this.q;
      const other = this.sweepParity ? this.q : this.qShadow;
      const average = this.params.valueObjective === 'average';
      const disc = average ? 1 : Math.pow(this.params.gamma, Math.max(1, meta.steps));
      const boot = this.maxQOf(other, meta.nextStateKey) ?? this.maxQAtState(meta.nextStateKey);
      const target = meta.lastReward + (average ? -this.rho : 0) + disc * boot;
      const cur = table.get(top.k) ?? 0;
      const delta = target - cur;
      if (Math.abs(delta) < 1e-4) continue;
      table.set(top.k, cur + REPLAY_ALPHA * delta);
      this.stats.sweeps++;
      this.pushPreds(meta.stateKey, Math.abs(delta) * disc);
    }
  }

  /** Planning update (replay / Dyna): TD(0), double-Q alternation, no visit counting. */
  updateReplay(state: string, actionKey: string, reward: number, maxNextQ: number, steps = 1, nextStateKey?: string): void {
    const k = state + QSEP + actionKey;
    const average = this.params.valueObjective === 'average';
    const disc = average ? 1 : Math.pow(this.params.gamma, Math.max(1, steps));
    let target: number;
    let table = this.q;
    if (nextStateKey !== undefined) {
      this.sweepParity = !this.sweepParity;
      table = this.sweepParity ? this.qShadow : this.q;
      const other = this.sweepParity ? this.q : this.qShadow;
      const boot = this.maxQOf(other, nextStateKey) ?? this.maxQAtState(nextStateKey);
      target = reward + (average ? -this.rho : 0) + disc * boot;
    } else {
      target = reward + (average ? -this.rho : 0) + disc * maxNextQ;
    }
    table.set(k, (table.get(k) ?? 0) + REPLAY_ALPHA * (target - (table.get(k) ?? 0)));
    this.trainFactor(state, actionKey, target);
    if (nextStateKey !== undefined) this.registerTransition(state, actionKey, nextStateKey, steps, reward);
    this.stats.replays++;
  }

  /** Offline fitted-Q update (warm start): table + factor model toward the
   *  bootstrapped target; head/baseline toward the immediate reward. No
   *  traces, no visit counting — offline consolidation must not fake online
   *  confidence. */
  fitExperience(state: string, actionKey: string, reward: number, features: number[] | undefined, steps: number, maxNextQ: number, lr = 0.25, literalKey?: string): void {
    const k = state + QSEP + actionKey;
    const average = this.params.valueObjective === 'average';
    const disc = average ? 1 : Math.pow(this.params.gamma, Math.max(1, steps));
    const target = reward + (average ? -this.rho : 0) + disc * maxNextQ;
    this.q.set(k, (this.q.get(k) ?? 0) + lr * (target - (this.q.get(k) ?? 0)));
    this.trainFactor(state, actionKey, target);
    if (literalKey && this.useBonus) {
      const b = this.literalBonus.get(literalKey) ?? 0;
      this.literalBonus.set(literalKey, Math.max(-2, Math.min(2, b + 0.05 * (target - ((this.q.get(k) ?? 0) + b)))));
    }
    if (features && features.length) {
      const base = this.stateValue.get(state) ?? 0;
      this.headUpdate({ key: actionKey, prior: 0, features }, reward - base);
      this.stateValue.set(state, base + 0.1 * (reward - base));
    }
    this.stats.fittedQ++;
  }

  /** Offline batch phase: shuffled epochs of fitExperience over a buffer. */
  fittedQ(transitions: OfflineTransition[], opts: { epochs: number; lr: number }): { passes: number; updates: number } {
    let updates = 0;
    const order = transitions.slice();
    for (let e = 0; e < opts.epochs; e++) {
      for (let i = order.length - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        [order[i], order[j]] = [order[j], order[i]];
      }
      for (const t of order) {
        this.fitExperience(t.stateKey, t.actionKey, t.reward, t.features, t.steps, this.maxQAtState(t.nextStateKey), opts.lr, t.literalKey);
        updates++;
      }
    }
    return { passes: opts.epochs, updates };
  }

  ingestReplay(transitions: { stateKey: string; actionKey: string; reward: number; nextStateKey: string; steps: number }[]): void {
    for (const t of transitions) this.registerTransition(t.stateKey, t.actionKey, t.nextStateKey, t.steps, t.reward);
  }

  maxQAtState(state: string): number {
    const prefix = state + QSEP;
    const vals = new Map<string, number>();
    for (const [k, v] of this.q) if (k.startsWith(prefix)) vals.set(k, v);
    for (const [k, v] of this.qShadow) {
      if (!k.startsWith(prefix)) continue;
      const qv = vals.get(k);
      vals.set(k, qv !== undefined ? (2 * qv + v) / 3 : v);
    }
    let best = 0;
    for (const v of vals.values()) if (v > best) best = v;
    return best;
  }

  /** Action selection with exact behavior propensities (piB) for OPE. */
  choose(state: string, actions: Scorable[], fallbackState?: string): { index: number; explored: boolean; bestScore: number; piB: number; argmaxIndex: number } {
    if (actions.length === 0) return { index: -1, explored: false, bestScore: -Infinity, piB: -1, argmaxIndex: -1 };
    const est = actions.map((a) => this.estimate(state, a, fallbackState));
    const rankBy = this.params.rankBy ?? 'value';
    const base = actions.map((a, i) => {
      if (rankBy === 'value') return est[i].value + this.params.beta * Math.log(1 + a.prior);
      const h = this.histOf(state, a.key, fallbackState);
      if (rankBy === 'pwin') return h[3] + h[4] + 0.1 * est[i].value + this.params.beta * Math.log(1 + a.prior);
      return cvarHist(h, this.params.cvarAlpha ?? 0.25) + this.params.beta * Math.log(1 + a.prior);
    });
    let argmaxIndex = 0;
    for (let i = 1; i < base.length; i++) if (base[i] > base[argmaxIndex]) argmaxIndex = i;
    const bestScore = base[argmaxIndex];

    const lock = this.exploitLock;
    const eps = lock ? 0 : this.params.epsilon;
    const ucbC = lock ? this.params.ucbC * 0.25 : this.params.ucbC;
    const tau = Math.max(0.05, lock ? this.params.temperature * 0.25 : this.params.temperature);
    const tSig = lock ? this.params.thompsonSigma * 0.25 : this.params.thompsonSigma;

    if (this.params.exploration === 'thompson') {
      let pick = argmaxIndex;
      if (rankBy === 'pwin') {
        const samples = actions.map((a, i) => {
          const h = this.histOf(state, a.key, fallbackState);
          const m = 4 + Math.min(this.visits.get(state + QSEP + a.key) ?? 0, 40);
          const ga = gammaRand(Math.max(0.1, m * (h[3] + h[4])));
          const gb = gammaRand(Math.max(0.1, m * (1 - h[3] - h[4])));
          return (ga / (ga + gb)) + 0.1 * est[i].value + this.params.beta * Math.log(1 + a.prior);
        });
        pick = 0;
        for (let i = 1; i < samples.length; i++) if (samples[i] > samples[pick]) pick = i;
      } else if (tSig > 0) {
        const noisy = base.map((s, i) => s + gauss() * tSig * est[i].sigma);
        pick = 0;
        for (let i = 1; i < noisy.length; i++) if (noisy[i] > noisy[pick]) pick = i;
      }
      return { index: pick, explored: pick !== argmaxIndex, bestScore, piB: tSig === 0 && rankBy !== 'pwin' ? 1 : -1, argmaxIndex };
    }

    const scores = this.params.riskKappa > 0 ? base.map((s, i) => s - this.params.riskKappa * est[i].sigma) : base.slice();
    let best = argmaxIndex;
    if (this.params.riskKappa > 0) {
      best = 0;
      for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
    }

    if (this.params.exploration === 'softmax') {
      const maxS = scores[best];
      const w = scores.map((s) => Math.exp((s - maxS) / tau));
      const z = w.reduce((x, y) => x + y, 0);
      let r = Math.random() * z, pick = w.length - 1;
      for (let i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) { pick = i; break; } }
      return { index: pick, explored: pick !== argmaxIndex, bestScore, piB: w[pick] / z, argmaxIndex };
    }

    if (this.params.exploration === 'ucb') {
      const n = this.stateVisits.get(state) ?? 0;
      for (let i = 0; i < actions.length; i++) {
        const na = this.visits.get(state + QSEP + actions[i].key) ?? 0;
        scores[i] += ucbC * Math.sqrt(Math.log(1 + n) / (1 + na));
      }
      best = 0;
      for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
      if (eps > 0 && Math.random() < eps) {
        this.stats.explorations++;
        return { index: Math.floor(Math.random() * actions.length), explored: true, bestScore, piB: eps / actions.length, argmaxIndex };
      }
      return { index: best, explored: best !== argmaxIndex, bestScore, piB: 1 - eps + eps / actions.length, argmaxIndex };
    }

    if (actions.length > 1 && eps > 0 && Math.random() < eps) {
      this.stats.explorations++;
      return { index: Math.floor(Math.random() * actions.length), explored: true, bestScore, piB: eps / actions.length, argmaxIndex };
    }
    return { index: best, explored: best !== argmaxIndex, bestScore, piB: 1 - eps + eps / actions.length, argmaxIndex };
  }

  decayQ(factor: number): void {
    for (const [k, v] of this.q) {
      const n = v * factor;
      if (Math.abs(n) < 1e-3) this.q.delete(k); else this.q.set(k, n);
    }
  }

  /** Remove one language's entries (states start lang\u0002, actions lang\u0003). */
  dropLanguage(lang: string): void {
    const p1 = lang + '\u0002';
    const p2 = lang + '\u0003';
    const wipe = (m: Map<string, any>): void => {
      for (const k of Array.from(m.keys())) if (k.startsWith(p1) || k.startsWith(p2)) m.delete(k);
    };
    wipe(this.q); wipe(this.qShadow); wipe(this.dist); wipe(this.visits);
    wipe(this.varOf); wipe(this.stateValue); wipe(this.stateVisits);
    wipe(this.stateEmb); wipe(this.actionEmb); wipe(this.factorSteps); wipe(this.literalBonus);
    this.traces.clear();
  }

  prune(cap: number): void {
    if (this.q.size > cap) {
      const entries = Array.from(this.q.entries()).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, cap);
      this.q.clear();
      for (const [k, v] of entries) this.q.set(k, v);
      for (const k of Array.from(this.visits.keys())) if (!this.q.has(k)) this.visits.delete(k);
    }
    const trimEmb = (m: Map<string, Emb>): void => {
      if (m.size <= 5000) return;
      const top = Array.from(m.entries()).sort((a, b) => b[1].n - a[1].n).slice(0, 5000);
      m.clear();
      for (const [k, e] of top) m.set(k, e);
    };
    trimEmb(this.stateEmb);
    trimEmb(this.actionEmb);
    if (this.literalBonus.size > 20_000) {
      const top = Array.from(this.literalBonus.entries()).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 20_000);
      this.literalBonus.clear();
      for (const [k, v] of top) this.literalBonus.set(k, v);
    }
  }

  mergeFrom(other: QLearner): void {
    for (const [k, v] of other.q) {
      const n1 = (this.visits.get(k) ?? 0) + 1;
      const n2 = (other.visits.get(k) ?? 0) + 1;
      this.q.set(k, ((this.q.get(k) ?? 0) * n1 + v * n2) / (n1 + n2));
      this.visits.set(k, (this.visits.get(k) ?? 0) + (other.visits.get(k) ?? 0));
      const v1 = this.varOf.get(k), v2 = other.varOf.get(k);
      if (v1 !== undefined && v2 !== undefined) this.varOf.set(k, 1 / ((1 / v1 + 1 / v2) / 2));
      else if (v2 !== undefined) this.varOf.set(k, v2);
      const hd = other.dist.get(k);
      if (hd) {
        const mine = this.dist.get(k);
        if (!mine) this.dist.set(k, hd.slice());
        else for (let i = 0; i < 5; i++) mine[i] = (mine[i] + hd[i]) / 2;
      }
      const sv = other.qShadow.get(k);
      if (sv !== undefined) this.qShadow.set(k, (this.qShadow.get(k) ?? sv) * 0.5 + sv * 0.5);
    }
    for (const [k, v] of other.visits) if (!this.q.has(k)) this.visits.set(k, (this.visits.get(k) ?? 0) + v);
    for (const [k, v] of other.stateVisits) this.stateVisits.set(k, (this.stateVisits.get(k) ?? 0) + v);
    for (const [k, v] of other.stateValue) this.stateValue.set(k, ((this.stateValue.get(k) ?? 0) + v) / 2);
    for (let i = 0; i < other.weights.length; i++) this.weights[i] = ((this.weights[i] ?? 0) + other.weights[i]) / 2;
    for (const [lang, w] of other.weightsByLang) {
      const mine = this.weightsByLang.get(lang);
      if (!mine) this.weightsByLang.set(lang, w.slice());
      else for (let i = 0; i < w.length; i++) mine[i] = ((mine[i] ?? 0) + w[i]) / 2;
    }
    for (const [k, v] of other.literalBonus) this.literalBonus.set(k, ((this.literalBonus.get(k) ?? 0) + v) / 2);
    this.rho = (this.rho + other.rho) / 2;
    for (const key of STAT_KEYS) this.stats[key] += other.stats[key] ?? 0;
  }

  /** Re-key literal action keys to semantic shapes, merging collisions. */
  migrateToShape(): number {
    const rekey = (k: string): string => {
      const i = k.indexOf('\u0003');
      if (i < 0) return k;
      return shapeKeyOf(k.slice(0, i), k.slice(i + 1).split('\u0000'));
    };
    const oldVisits = new Map(this.visits);
    const remapSum = (m: Map<string, number>): number => {
      const out = new Map<string, number>();
      let merges = 0;
      for (const [k, v] of m) {
        const nk = rekey(k);
        if (out.has(nk)) merges++;
        out.set(nk, (out.get(nk) ?? 0) + v);
      }
      m.clear();
      for (const [k, v] of out) m.set(k, v);
      return merges;
    };
    const remapQ = (m: Map<string, number>): number => {
      const out = new Map<string, number>();
      let merges = 0;
      for (const [k, v] of m) {
        const nk = rekey(k);
        if (out.has(nk)) {
          merges++;
          const n1 = oldVisits.get(k) ?? 1;
          const n2 = oldVisits.get(nk) ?? 1;
          out.set(nk, (out.get(nk)! * n2 + v * n1) / (n1 + n2));
        } else out.set(nk, v);
      }
      m.clear();
      for (const [k, v] of out) m.set(k, v);
      return merges;
    };
    let merges = remapSum(this.visits) + remapSum(this.factorSteps);
    merges += remapQ(this.q) + remapQ(this.qShadow);
    const varOut = new Map<string, number>();
    let vm = 0;
    for (const [k, v] of this.varOf) {
      const nk = rekey(k);
      if (varOut.has(nk)) { vm++; varOut.set(nk, Math.min(varOut.get(nk)!, v)); }
      else varOut.set(nk, v);
    }
    this.varOf.clear();
    for (const [k, v] of varOut) this.varOf.set(k, v);
    merges += vm;
    const distOut = new Map<string, number[]>();
    let dm = 0;
    for (const [k, h] of this.dist) {
      const nk = rekey(k);
      const ex = distOut.get(nk);
      if (ex) { dm++; for (let i = 0; i < 5; i++) ex[i] = (ex[i] + h[i]) / 2; }
      else distOut.set(nk, h.slice());
    }
    this.dist.clear();
    for (const [k, h] of distOut) this.dist.set(k, h);
    merges += dm;
    const embOut = new Map<string, Emb>();
    let em = 0;
    for (const [k, e] of this.actionEmb) {
      const nk = rekey(k);
      const ex = embOut.get(nk);
      if (ex) {
        em++;
        const n = ex.n + e.n;
        for (let i = 0; i < ex.v.length && i < e.v.length; i++) ex.v[i] = (ex.v[i] * ex.n + e.v[i] * e.n) / n;
        ex.n = n;
      } else embOut.set(nk, { v: e.v.slice(), n: e.n });
    }
    this.actionEmb.clear();
    for (const [k, e] of embOut) this.actionEmb.set(k, e);
    merges += em;
    this.traces.clear();
    this.stats.shapeMerges += merges;
    return merges;
  }

  private actionOfPairKey(k: string): string {
    const i = k.indexOf('\u0003');
    return i < 0 ? '' : k.slice(i);
  }

  /** Taste-profile subset: global (unsuffixed) states, safe actions only. */
  exportTaste(safeAction: (actionKey: string) => boolean): any {
    const r4 = (x: number): number => Math.round(x * 1e4) / 1e4;
    const globalState = (k: string): boolean => !k.includes('\u0004');
    const subPair = (m: Map<string, number>): [string, number][] =>
      Array.from(m.entries())
        .filter(([k]) => globalState(k) && safeAction(this.actionOfPairKey(k)))
        .map(([k, v]) => [k, r4(v)]);
    return {
      weights: this.weights.map(r4),
      weightsByLang: Array.from(this.weightsByLang.entries()).map(([l, w]) => [l, w.map(r4)]),
      stateVisits: Array.from(this.stateVisits.entries()).filter(([k]) => globalState(k)).map(([k, v]) => [k, v]),
      visits: subPair(this.visits),
      q: subPair(this.q),
      qShadow: subPair(this.qShadow),
      dist: Array.from(this.dist.entries())
        .filter(([k]) => globalState(k) && safeAction(this.actionOfPairKey(k)))
        .map(([k, h]) => [k, h.map(r4)]),
      stateEmb: Array.from(this.stateEmb.entries()).filter(([k]) => globalState(k)).map(([k, e]) => [k, e.v.map(r4), e.n]),
      actionEmb: Array.from(this.actionEmb.entries()).filter(([k]) => safeAction(k)).map(([k, e]) => [k, e.v.map(r4), e.n]),
    };
  }

  /** Merge a taste profile at reduced confidence. */
  mergeTaste(other: any, foreignWeight = 0.5): void {
    const r = (x: unknown): number => (typeof x === 'number' && Number.isFinite(x) ? x : 0);
    if (Array.isArray(other.weights)) {
      for (let i = 0; i < other.weights.length; i++) this.weights[i] = ((this.weights[i] ?? 0) + r(other.weights[i])) / 2;
    }
    if (Array.isArray(other.weightsByLang)) {
      for (const [lang, w] of other.weightsByLang) {
        if (typeof lang !== 'string' || !Array.isArray(w)) continue;
        const mine = this.weightsByLang.get(lang);
        if (!mine) this.weightsByLang.set(lang, w.map(r));
        else for (let i = 0; i < w.length; i++) mine[i] = ((mine[i] ?? 0) + r(w[i])) / 2;
      }
    }
    const mergeQ = (mine: Map<string, number>, raw: any): void => {
      if (!Array.isArray(raw)) return;
      for (const e of raw) {
        if (!Array.isArray(e) || typeof e[0] !== 'string') continue;
        const n1 = (this.visits.get(e[0]) ?? 0) + 1;
        mine.set(e[0], ((mine.get(e[0]) ?? 0) * n1 + r(e[1]) * foreignWeight) / (n1 + foreignWeight));
      }
    };
    mergeQ(this.q, other.q);
    mergeQ(this.qShadow, other.qShadow);
    const mergeSum = (mine: Map<string, number>, raw: any): void => {
      if (!Array.isArray(raw)) return;
      for (const e of raw) {
        if (Array.isArray(e) && typeof e[0] === 'string') mine.set(e[0], (mine.get(e[0]) ?? 0) + r(e[1]) * foreignWeight);
      }
    };
    mergeSum(this.visits, other.visits);
    mergeSum(this.stateVisits, other.stateVisits);
    if (Array.isArray(other.dist)) {
      for (const e of other.dist) {
        if (!Array.isArray(e) || typeof e[0] !== 'string' || !Array.isArray(e[1]) || e[1].length !== 5) continue;
        const mine = this.dist.get(e[0]);
        if (!mine) this.dist.set(e[0], e[1].map(r));
        else for (let i = 0; i < 5; i++) mine[i] = (mine[i] + r(e[1][i])) / 2;
      }
    }
    const mergeEmb = (raw: any, target: Map<string, Emb>): void => {
      if (!Array.isArray(raw)) return;
      for (const e of raw) {
        if (!Array.isArray(e) || typeof e[0] !== 'string' || !Array.isArray(e[1])) continue;
        const ex = target.get(e[0]);
        if (!ex) target.set(e[0], { v: e[1].map(r), n: Math.max(1, Math.round(r(e[2]))) });
        else {
          const n2 = Math.max(1, Math.round(r(e[2]))) * foreignWeight;
          const n = ex.n + n2;
          for (let i = 0; i < ex.v.length && i < e[1].length; i++) ex.v[i] = (ex.v[i] * ex.n + r(e[1][i]) * n2) / n;
          ex.n = n;
        }
      }
    };
    mergeEmb(other.stateEmb, this.stateEmb);
    mergeEmb(other.actionEmb, this.actionEmb);
  }

  serialize(): any {
    const r4 = (x: number): number => Math.round(x * 1e4) / 1e4;
    return {
      q: Array.from(this.q.entries()).map(([k, v]) => [k, r4(v)]),
      qShadow: Array.from(this.qShadow.entries()).map(([k, v]) => [k, r4(v)]),
      dist: Array.from(this.dist.entries()).map(([k, h]) => [k, h.map((x) => r4(x))]),
      visits: Array.from(this.visits.entries()),
      stateVisits: Array.from(this.stateVisits.entries()),
      varOf: Array.from(this.varOf.entries()).map(([k, v]) => [k, r4(v)]),
      stateValue: Array.from(this.stateValue.entries()).map(([k, v]) => [k, r4(v)]),
      weights: this.weights.map(r4),
      weightsByLang: Array.from(this.weightsByLang.entries()).map(([l, w]) => [l, w.map(r4)]),
      stateEmb: Array.from(this.stateEmb.entries()).map(([k, e]) => [k, e.v.map(r4), e.n]),
      actionEmb: Array.from(this.actionEmb.entries()).map(([k, e]) => [k, e.v.map(r4), e.n]),
      factorSteps: Array.from(this.factorSteps.entries()),
      literalBonus: Array.from(this.literalBonus.entries()).map(([k, v]) => [k, r4(v)]),
      rho: r4(this.rho),
      exploitLock: this.exploitLock,
      stats: this.stats,
    };
  }

  static deserialize(data: any): QLearner | null {
    try {
      if (!data || !Array.isArray(data.q)) return null;
      const learner = new QLearner({
        alpha: 0.3, alphaMode: 'kalman', gamma: 0.9, lambda: 0.7, epsilon: 0.05, beta: 1,
        exploration: 'epsilon', ucbC: 0.3, temperature: 0.35, thompsonSigma: 0.5, weightAlpha: 0.05,
        valueObjective: 'discounted', rhoAlpha: 0.05, processNoise: 0.01, obsNoise: 0.5,
        factorDim: 8, factorEta: 0.05, riskKappa: 0, driftThreshold: 0.35, sweepPerUpdate: 2,
        rankBy: 'value', cvarAlpha: 0.25, distEta: 0.15,
      });
      const loadNumMap = (raw: any, target: Map<string, number>): void => {
        if (Array.isArray(raw)) {
          for (const e of raw) if (Array.isArray(e) && typeof e[0] === 'string' && typeof e[1] === 'number') target.set(e[0], e[1]);
        }
      };
      loadNumMap(data.q, learner.q);
      loadNumMap(data.qShadow, learner.qShadow);
      loadNumMap(data.visits, learner.visits);
      loadNumMap(data.stateVisits, learner.stateVisits);
      loadNumMap(data.varOf, learner.varOf);
      loadNumMap(data.stateValue, learner.stateValue);
      loadNumMap(data.factorSteps, learner.factorSteps);
      loadNumMap(data.literalBonus, learner.literalBonus);
      if (Array.isArray(data.dist)) {
        for (const e of data.dist) {
          if (Array.isArray(e) && typeof e[0] === 'string' && Array.isArray(e[1]) && e[1].length === 5) {
            learner.dist.set(e[0], e[1].map((x: unknown) => Number(x) || 0));
          }
        }
      }
      if (Array.isArray(data.weights)) learner.weights = data.weights.filter((w: unknown) => typeof w === 'number');
      if (Array.isArray(data.weightsByLang)) {
        for (const [lang, w] of data.weightsByLang) {
          if (typeof lang === 'string' && Array.isArray(w)) learner.weightsByLang.set(lang, w.filter((x: unknown) => typeof x === 'number'));
        }
      }
      const loadEmb = (raw: any, target: Map<string, Emb>): void => {
        if (!Array.isArray(raw)) return;
        for (const e of raw) {
          if (Array.isArray(e) && typeof e[0] === 'string' && Array.isArray(e[1])) {
            target.set(e[0], { v: e[1].filter((x: unknown) => typeof x === 'number'), n: Number(e[2]) || 0 });
          }
        }
      };
      loadEmb(data.stateEmb, learner.stateEmb);
      loadEmb(data.actionEmb, learner.actionEmb);
      if (typeof data.rho === 'number') learner.rho = data.rho;
      if (typeof data.exploitLock === 'boolean') learner.exploitLock = data.exploitLock;
      if (data.stats && typeof data.stats === 'object') {
        for (const key of STAT_KEYS) learner.stats[key] = Number((data.stats as any)[key]) || 0;
      }
      return learner;
    } catch {
      return null;
    }
  }
}
