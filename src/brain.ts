/** The brain: world model (dual-channel Markov chain, KN smoothing, family
 *  hierarchy, doc cache) + agent (TD(lambda), distributional, double-Q,
 *  factors, heads) + options/PUCT planning + audited gating + shape action
 *  keys + corpus learning + taste profiles + persistence. */
import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { tokenize, hashText, tail, abstractToken, reindent, shapeKeyOf } from './util';
import { canonicalLang, familyOf, profileFor, FAMILIES, langForFilename } from './languages';
import { MarkovModel } from './markov';
import { QLearner, type Explanation, type OfflineTransition } from './learner';
import { lookaheadSearch } from './planner';
import { GenerationPool } from './workerPool';
import {
  DistResolver, generateTokens, requestFromParams, successorPrior,
  qStateKeyOf, successorMeta, type TokenCand, type TokenGenParams,
} from './generate';
import {
  parseCorpusContent, evaluateSequences, gridSearchTune,
  type CorpusItem, type EvalMetrics, type TuneOutcome,
} from './dataset';
import { FEATURE_NAMES, type DashboardData } from './dashboard';

export { qStateKeyOf } from './generate';

export interface MdpConfig {
  enabled: boolean; order: number; candidates: number; maxTokens: number; maxLines: number;
  minCount: number; minConfidence: number; smoothingD: number; beamWidth: number;
  smoothing: 'interpolated' | 'kneser-ney'; knDiscount: number; transferWeight: number;
  familyTransferWeight: number; transferLearn: boolean; transferEta: number;
  cache: boolean; cacheD: number; diversityLambda: number;
  alpha: number; alphaMode: 'fixed' | 'adaptive' | 'kalman'; gamma: number; lambda: number;
  epsilon: number; beta: number;
  exploration: 'epsilon' | 'ucb' | 'softmax' | 'thompson'; ucbC: number;
  temperature: number; thompsonSigma: number; weightAlpha: number;
  rankBy: 'value' | 'pwin' | 'cvar'; cvarAlpha: number; distEta: number;
  valueObjective: 'discounted' | 'average'; rhoAlpha: number;
  processNoise: number; obsNoise: number; factorDim: number; factorEta: number;
  riskKappa: number; driftThreshold: number; sweepPerUpdate: number;
  rewardAccept: number; rewardDismiss: number; rewardEscape: number; rewardOverridden: number;
  partialReward: number; timeReward: boolean; utilityReward: boolean; shaping: number;
  correctionReward: number; watchWindow: number; confirmReward: number;
  minScore: number; minVisitsGate: number; gateProbe: number;
  adaptiveGate: boolean; targetAcceptRate: number; gateStep: number; refactorBias: boolean;
  gateAudit: boolean;
  countDecay: number; decayEvery: number; qDecay: number;
  replayBuffer: number; replayBatch: number; dynaBatch: number; dynaEvery: number;
  rolloutSteps: number;
  lookahead: boolean; lookaheadBudget: number; lookaheadDepth: number; lookaheadCPuct: number;
  rewardThumbUp: number; rewardThumbDown: number;
  undoWindow: number; undoReward: number; undoFrac: number; adaptiveLength: boolean;
  autoExploit: boolean; sessionLog: boolean;
  worker: boolean; workerTimeout: number; workerModelRefreshMs: number;
  workspaceConditioning: boolean; fittedQ: boolean; fittedQEpochs: number;
  actionKeyMode: 'hybrid' | 'shape' | 'literal'; literalBonus: boolean;
  debounceMs: number; maxDebounceMs: number;
  datasetWeight: number; datasetMaxFiles: number; datasetMaxFileMb: number;
  evalTokenBudget: number; tuneTokenBudget: number; tuneOrders: number[];
  languages: string[]; langOverrides: Record<string, any>;
  suggestOnNewline: boolean; persist: boolean; debug: boolean; maxFileKb: number;
}

const OVERRIDE_KEYS = ['enabled', 'candidates', 'maxTokens', 'maxLines', 'minCount', 'minConfidence', 'beamWidth', 'suggestOnNewline', 'rolloutSteps', 'diversityLambda', 'lookahead', 'lookaheadBudget', 'lookaheadDepth', 'lookaheadCPuct'] as const;

const CORPUS_IGNORE = new Set(['node_modules', '.git', 'dist', 'out', 'build', '.next', 'coverage', 'vendor', '__pycache__', '.venv', 'target', '.idea', '.svn']);

export function readConfig(): MdpConfig {
  const c = vscode.workspace.getConfiguration('mdpSnippets');
  const oneOf = <T extends string>(v: string | undefined, allowed: readonly T[], dflt: T): T =>
    (allowed as readonly string[]).includes(v ?? '') ? (v as T) : dflt;
  const actionKeyMode = oneOf(c.get<string>('actionKeyMode', 'hybrid'), ['hybrid', 'shape', 'literal'] as const, 'hybrid');
  return {
    enabled: c.get<boolean>('enabled', true),
    order: c.get<number>('order', 3),
    candidates: c.get<number>('candidates', 3),
    maxTokens: c.get<number>('maxTokens', 24),
    maxLines: c.get<number>('maxLines', 1),
    minCount: c.get<number>('minCount', 2),
    minConfidence: c.get<number>('minConfidence', 0.15),
    smoothingD: c.get<number>('smoothingD', 6),
    beamWidth: c.get<number>('beamWidth', 4),
    smoothing: oneOf(c.get<string>('smoothing', 'kneser-ney'), ['kneser-ney', 'interpolated'] as const, 'kneser-ney'),
    knDiscount: c.get<number>('knDiscount', 0.75),
    transferWeight: c.get<number>('transferWeight', 0.25),
    familyTransferWeight: c.get<number>('familyTransferWeight', 0.4),
    transferLearn: c.get<boolean>('transferLearn', true),
    transferEta: c.get<number>('transferEta', 0.3),
    cache: c.get<boolean>('cache', true),
    cacheD: c.get<number>('cacheD', 8),
    diversityLambda: c.get<number>('diversityLambda', 0.15),
    alpha: c.get<number>('alpha', 0.3),
    alphaMode: oneOf(c.get<string>('alphaMode', 'kalman'), ['fixed', 'adaptive', 'kalman'] as const, 'kalman'),
    gamma: c.get<number>('gamma', 0.9),
    lambda: c.get<number>('lambda', 0.7),
    epsilon: c.get<number>('epsilon', 0.05),
    beta: c.get<number>('beta', 1.0),
    exploration: oneOf(c.get<string>('exploration', 'epsilon'), ['epsilon', 'ucb', 'softmax', 'thompson'] as const, 'epsilon'),
    ucbC: c.get<number>('ucbC', 0.3),
    temperature: c.get<number>('temperature', 0.35),
    thompsonSigma: c.get<number>('thompsonSigma', 0.5),
    weightAlpha: c.get<number>('weightAlpha', 0.05),
    rankBy: oneOf(c.get<string>('rankBy', 'value'), ['value', 'pwin', 'cvar'] as const, 'value'),
    cvarAlpha: c.get<number>('cvarAlpha', 0.25),
    distEta: c.get<number>('distEta', 0.15),
    valueObjective: oneOf(c.get<string>('valueObjective', 'discounted'), ['discounted', 'average'] as const, 'discounted'),
    rhoAlpha: c.get<number>('rhoAlpha', 0.05),
    processNoise: c.get<number>('processNoise', 0.01),
    obsNoise: c.get<number>('obsNoise', 0.5),
    factorDim: c.get<number>('factorDim', 8),
    factorEta: c.get<number>('factorEta', 0.05),
    riskKappa: c.get<number>('riskKappa', 0),
    driftThreshold: c.get<number>('driftThreshold', 0.35),
    sweepPerUpdate: c.get<number>('sweepPerUpdate', 2),
    rewardAccept: c.get<number>('rewardAccept', 1.0),
    rewardDismiss: c.get<number>('rewardDismiss', -0.1),
    rewardEscape: c.get<number>('rewardEscape', -0.2),
    rewardOverridden: c.get<number>('rewardOverridden', -0.05),
    partialReward: c.get<number>('partialReward', 0.25),
    timeReward: c.get<boolean>('timeReward', true),
    utilityReward: c.get<boolean>('utilityReward', true),
    shaping: c.get<number>('shaping', 0.5),
    correctionReward: c.get<number>('correctionReward', -0.3),
    watchWindow: c.get<number>('watchWindow', 30),
    confirmReward: c.get<number>('confirmReward', 0.15),
    minScore: c.get<number>('minScore', -0.25),
    minVisitsGate: c.get<number>('minVisitsGate', 6),
    gateProbe: c.get<number>('gateProbe', 0.1),
    adaptiveGate: c.get<boolean>('adaptiveGate', true),
    targetAcceptRate: c.get<number>('targetAcceptRate', 0.35),
    gateStep: c.get<number>('gateStep', 0.05),
    refactorBias: c.get<boolean>('refactorBias', true),
    gateAudit: c.get<boolean>('gateAudit', true),
    countDecay: c.get<number>('countDecay', 0.998),
    decayEvery: Math.max(1, c.get<number>('decayEvery', 300)),
    qDecay: c.get<number>('qDecay', 0.999),
    replayBuffer: c.get<number>('replayBuffer', 300),
    replayBatch: c.get<number>('replayBatch', 16),
    dynaBatch: c.get<number>('dynaBatch', 8),
    dynaEvery: Math.max(1, c.get<number>('dynaEvery', 150)),
    rolloutSteps: c.get<number>('rolloutSteps', 2),
    lookahead: c.get<boolean>('lookahead', true),
    lookaheadBudget: c.get<number>('lookaheadBudget', 12),
    lookaheadDepth: c.get<number>('lookaheadDepth', 10),
    lookaheadCPuct: c.get<number>('lookaheadCPuct', 1.0),
    rewardThumbUp: c.get<number>('rewardThumbUp', 0.5),
    rewardThumbDown: c.get<number>('rewardThumbDown', -0.5),
    undoWindow: c.get<number>('undoWindow', 10),
    undoReward: c.get<number>('undoReward', -1.0),
    undoFrac: c.get<number>('undoFrac', 0.6),
    adaptiveLength: c.get<boolean>('adaptiveLength', true),
    autoExploit: c.get<boolean>('autoExploit', true),
    sessionLog: c.get<boolean>('sessionLog', false),
    worker: c.get<boolean>('worker', true),
    workerTimeout: c.get<number>('workerTimeout', 120),
    workerModelRefreshMs: c.get<number>('workerModelRefreshMs', 2500),
    workspaceConditioning: c.get<boolean>('workspaceConditioning', false),
    fittedQ: c.get<boolean>('fittedQ', true),
    fittedQEpochs: c.get<number>('fittedQEpochs', 3),
    actionKeyMode,
    literalBonus: actionKeyMode === 'hybrid',
    debounceMs: c.get<number>('debounceMs', 25),
    maxDebounceMs: c.get<number>('maxDebounceMs', 100),
    datasetWeight: c.get<number>('datasetWeight', 0.5),
    datasetMaxFiles: c.get<number>('datasetMaxFiles', 2000),
    datasetMaxFileMb: c.get<number>('datasetMaxFileMb', 2),
    evalTokenBudget: c.get<number>('evalTokenBudget', 20000),
    tuneTokenBudget: c.get<number>('tuneTokenBudget', 150000),
    tuneOrders: (c.get<number[]>('tuneOrders', [2, 3, 4]) ?? [2, 3, 4]).filter((o) => o >= 1 && o <= 8),
    languages: (c.get<string[]>('languages', []) ?? []).map((l) => canonicalLang(l)),
    langOverrides: c.get<Record<string, any>>('langOverrides', {}) ?? {},
    suggestOnNewline: c.get<boolean>('suggestOnNewline', true),
    persist: c.get<boolean>('persist', true),
    debug: c.get<boolean>('debug', false),
    maxFileKb: c.get<number>('maxFileKb', 512),
  };
}

const phi = (p: number): number => Math.log1p(p);

export interface Candidate {
  key: string;          // semantic shape key (or literal in literal mode)
  literalKey: string;   // exact token-sequence key (pin/ban/bonus identity)
  lang: string; tokens: string[]; text: string;
  prior: number; nextPrior: number; nextStateKey: string;
  features: number[]; reindented: boolean; isRollout: boolean; isLookahead: boolean;
}

interface PendingEntry { candidate: Candidate; insertText: string; prefixLen: number; }
interface PendingProposal {
  uri: string; lang: string; stateKey: string; version: number; line: number; character: number;
  cursorOffset: number; beforeLen: number; shownAt: number; indentBucket: number;
  litBefore: string[]; atEol: boolean; indentStr: string;
  piB: number; greedyShown: boolean; probe: boolean; probeScore: number;
  primaryId: number; entries: Map<number, PendingEntry>; order: number[];
}
interface MdpEvent { t: number; kind: 'accept' | 'dismiss' | 'partial' | 'escape' | 'undo' | 'correction' | 'confirm' | 'thumbUp' | 'thumbDown'; }
interface SnippetRecord { text: string; lang: string; accepts: number; dismissals: number; }
interface StateContext { lang: string; uri: string; lit: string[]; abs: string[]; indent: string; atEol: boolean; bucket: number; }
interface Watch { uri: string; start: number; end: number; len: number; at: number; stateKey: string; candidate: Candidate; }
interface OpeLog { piB: number; reward: number; greedyShown: boolean; }
interface LangStat { accepts: number; dismissals: number; partials: number; predHits: number; predTotal: number; }

export class Brain {
  markov: MarkovModel;
  learner: QLearner;
  cfg: MdpConfig;
  readonly modelPath: string;
  readonly sessionLogPath: string;
  readonly stats = { filesLearned: 0 };
  corpusDocs = 0;
  corpusTokens = 0;

  private pendingProposal: PendingProposal | null = null;
  private idSeq = 1;
  private saveTimer: ReturnType<typeof setTimeout> | null = null;
  private learnedHashes = new Map<string, number>();
  private dirty = false;
  private learnEvents = 0;
  private lastAcceptAt = 0;
  private events: MdpEvent[] = [];
  private topSnippets = new Map<string, SnippetRecord>();
  private replay: OfflineTransition[] = [];
  private stateContexts = new Map<string, StateContext>();
  private keystrokesSaved = 0;
  private predHits = 0;
  private predTotal = 0;
  private predEMA = 0.5;

  private docCaches = new Map<string, Map<string, number>>();
  private editIns = 0;
  private editDel = 0;
  private watches: Watch[] = [];
  private gateWindow: number[] = [];
  private gateOffset = 0;
  private opeLogs: OpeLog[] = [];
  private opeStreak = 0;
  private opeEstimate: { greedy: number; onPolicy: number; n: number } | null = null;

  private channelW = new Map<string, number[]>();
  private langStats = new Map<string, LangStat>();
  private pinned = new Set<string>();
  private banned = new Set<string>();
  private curatedInfo = new Map<string, { text: string; lang: string }>();
  private lastPlan: { text: string; key: string; prior: number; pinned: boolean; chosen: boolean; parts: Explanation; score: number }[] | null = null;
  private lastTopKeys: string[] = [];
  private lastBannedKeys: string[] = [];
  private shapeLiterals = new Map<string, Set<string>>();
  private probeOutcomes: number[] = [];
  private probePredicted: number[] = [];

  private pool: GenerationPool | null = null;
  private _resolver: DistResolver | null = null;
  private genWorker = 0;
  private genLocal = 0;

  private acceptLenEMA: number;
  private dismissLenEMA: number;

  constructor(storageDir: string, private readonly logFn: (msg: string) => void) {
    this.cfg = readConfig();
    this.markov = new MarkovModel(this.cfg.order);
    this.learner = new QLearner(this.cfg);
    this.modelPath = path.join(storageDir, 'mdp-model.json');
    this.sessionLogPath = path.join(storageDir, 'session-log.jsonl');
    this.acceptLenEMA = this.cfg.maxTokens;
    this.dismissLenEMA = this.cfg.maxTokens;
    this.load();
  }

  get hasPending(): boolean { return this.pendingProposal !== null; }
  get pendingVersion(): number { return this.pendingProposal ? this.pendingProposal.version : -1; }
  get chainActive(): boolean { return Date.now() - this.lastAcceptAt < 2500; }

  get sessionMode(): 'C' | 'M' | 'R' {
    const total = this.editIns + this.editDel;
    if (total < 40) return 'C';
    const share = this.editDel / total;
    return share > 0.6 ? 'R' : share > 0.35 ? 'M' : 'C';
  }

  /** 4-char workspace identity; '' when conditioning off / no folder. */
  get wsTag(): string {
    if (!this.cfg.workspaceConditioning) return '';
    const folders = vscode.workspace.workspaceFolders;
    return folders?.length ? hashText(folders[0].uri.toString()).toString(36).slice(0, 4) : '';
  }

  refreshConfig(): { orderChanged: boolean } {
    const prevOrder = this.cfg.order;
    const prevPersist = this.cfg.persist;
    this._resolver = null;
    this.cfg = readConfig();
    this.learner.updateParams(this.cfg);
    if (this.cfg.order !== prevOrder) {
      this.markov = new MarkovModel(this.cfg.order);
      this.learner = new QLearner(this.cfg);
      this.events = [];
      this.topSnippets.clear();
      this.replay = [];
      this.stateContexts.clear();
      this.watches = [];
      this.opeLogs = [];
      this.docCaches.clear();
      this.channelW.clear();
      this.shapeLiterals.clear();
      this.acceptLenEMA = this.cfg.maxTokens;
      this.dismissLenEMA = this.cfg.maxTokens;
      return { orderChanged: true };
    }
    if (!prevPersist && this.cfg.persist && this.markov.pairsLearned === 0) this.load();
    return { orderChanged: false };
  }

  // ------------------------------------------------- per-language config

  cfgFor(lang: string): MdpConfig {
    const o = this.cfg.langOverrides?.[canonicalLang(lang)];
    if (!o) return this.cfg;
    const merged = { ...this.cfg };
    for (const k of OVERRIDE_KEYS) {
      if (o[k] !== undefined) (merged as any)[k] = o[k];
    }
    return merged;
  }

  enabledFor(lang: string): boolean {
    if (!this.cfg.enabled) return false;
    return this.cfgFor(lang).enabled;
  }

  // ------------------------------------------------- learning (documents)

  shouldLearn(doc: vscode.TextDocument): boolean {
    if (doc.uri.scheme !== 'file' && doc.uri.scheme !== 'untitled') return false;
    const lang = canonicalLang(doc.languageId);
    if (this.cfg.languages.length > 0 && !this.cfg.languages.includes(lang)) return false;
    return doc.getText().length <= this.cfg.maxFileKb * 1024;
  }

  learnDocument(doc: vscode.TextDocument, force = false): boolean {
    if (!this.shouldLearn(doc)) return false;
    const text = doc.getText();
    const hash = hashText(text);
    const uri = doc.uri.toString();
    if (!force && this.learnedHashes.get(uri) === hash) return false;
    this.learnedHashes.set(uri, hash);
    if (this.learnedHashes.size > 500) {
      const excess = Array.from(this.learnedHashes.keys()).slice(0, this.learnedHashes.size - 400);
      for (const k of excess) this.learnedHashes.delete(k);
    }
    const lines = text.split('\n');
    if (lines.length > 50 && text.length / lines.length > 400) return false;
    const lang = canonicalLang(doc.languageId);
    const tokens = tokenize(text, lang);
    this.markov.learn(lang, tokens, 1, this.cfg.transferWeight, this.cfg.familyTransferWeight);
    if (this.cfg.cache) this.touchCache(uri, tokens.slice(-800), 0.5);
    this.stats.filesLearned++;
    this.noteLearnEvent();
    this.dirty = true;
    this.log(`learned snapshot [${lang}]: ${path.basename(doc.fileName)} (${lines.length} lines)`);
    this.scheduleSave();
    return true;
  }

  noteSaved(doc: vscode.TextDocument): void {
    if (!this.shouldLearn(doc)) return;
    this.learnedHashes.set(doc.uri.toString(), hashText(doc.getText()));
  }

  noteClosed(uri: string): void {
    this.docCaches.delete(uri);
  }

  learnInsertedText(doc: vscode.TextDocument, offset: number, inserted: string): void {
    if (!inserted.trim() || !this.shouldLearn(doc)) return;
    if (inserted.length > 20_000) inserted = inserted.slice(0, 20_000);
    const uri = doc.uri.toString();
    const lang = canonicalLang(doc.languageId);
    const start = Math.max(0, offset - 200);
    const before = doc.getText(new vscode.Range(doc.positionAt(start), doc.positionAt(offset)));
    const seq = tokenize(before + inserted, lang);
    this.notePrediction(lang, seq, tokenize(before, lang), inserted);
    this.markov.learn(lang, seq, 1, this.cfg.transferWeight, this.cfg.familyTransferWeight);
    if (this.cfg.cache) this.touchCache(uri, tokenize(inserted, lang), 1);
    this.noteLearnEvent();
    this.dirty = true;
    this.log(`learned insertion [${lang}] (${inserted.length} chars)`);
  }

  noteEditVolume(ins: number, del: number): void {
    this.editIns = 0.98 * this.editIns + Math.min(ins, 2000);
    this.editDel = 0.98 * this.editDel + Math.min(del, 2000);
  }

  private touchCache(uri: string, tokens: string[], weight: number): void {
    let cache = this.docCaches.get(uri);
    if (!cache) {
      cache = new Map();
      this.docCaches.set(uri, cache);
      if (this.docCaches.size > 20) {
        const first = this.docCaches.keys().next().value;
        if (first !== undefined) this.docCaches.delete(first);
      }
    }
    for (const [tok, w] of cache) {
      const nw = w * 0.98;
      if (nw < 0.05) cache.delete(tok); else cache.set(tok, nw);
    }
    for (const tok of tokens) {
      if (!/^[#@]?[A-Za-z_$][A-Za-z0-9_$]*$/.test(tok)) continue;
      cache.set(tok, Math.min(4, (cache.get(tok) ?? 0) + weight));
    }
    if (cache.size > 3000) {
      const top = Array.from(cache.entries()).sort((a, b) => b[1] - a[1]).slice(0, 3000);
      cache.clear();
      for (const [k, w] of top) cache.set(k, w);
    }
  }

  private notePrediction(lang: string, seq: string[], beforeTokens: string[], inserted: string): void {
    if (inserted.length > 12 || seq.length < 2 || seq.length - beforeTokens.length < 1) return;
    const ctx = seq.slice(0, -1);
    const absCtx = ctx.map((t) => abstractToken(t, lang));
    const dist = this.resolver().distAt(lang, ctx, absCtx);
    const actual = seq[seq.length - 1];
    let hit = 0;
    if (dist) {
      let bestTok = '';
      let bestP = -1;
      for (const [tok, p] of dist) if (p > bestP) { bestP = p; bestTok = tok; }
      hit = bestTok === actual ? 1 : 0;
    }
    this.predTotal++;
    this.predHits += hit;
    this.predEMA = 0.98 * this.predEMA + 0.02 * hit;
    const ls = this.langStatFor(lang);
    ls.predTotal++;
    ls.predHits += hit;
    this.updateTransferWeights(lang, ctx, absCtx, actual);
  }

  /** Hedge update of [self, family, universal] per language, driven by each
   *  channel's probability of the token you actually typed. */
  private updateTransferWeights(lang: string, litCtx: string[], absCtx: string[], actual: string): void {
    if (!this.cfg.transferLearn) return;
    const w = (this.channelW.get(lang) ?? [1, this.cfg.familyTransferWeight, this.cfg.transferWeight]).slice();
    const kn = this.cfg.smoothing === 'kneser-ney' ? this.cfg.knDiscount : 0;
    const chanLangs = [lang, familyOf(lang), '*'];
    for (let i = 0; i < 3; i++) {
      if (w[i] <= 0.001) continue;
      const dist = this.markov.smoothed(chanLangs[i], litCtx, absCtx, this.cfg.smoothingD, 1, kn);
      const p = Math.max(1e-4, dist?.get(actual) ?? 1e-4);
      w[i] *= Math.pow(p, this.cfg.transferEta);
    }
    const mx = Math.max(w[0], w[1], w[2]);
    if (mx > 0) for (let i = 0; i < 3; i++) w[i] = Math.max(0.02, w[i] / mx);
    this.channelW.set(lang, w);
    this.pool?.markDirty();
    this.dirty = true;
  }

  private langStatFor(lang: string): LangStat {
    let ls = this.langStats.get(lang);
    if (!ls) { ls = { accepts: 0, dismissals: 0, partials: 0, predHits: 0, predTotal: 0 }; this.langStats.set(lang, ls); }
    return ls;
  }

  private noteLearnEvent(): void {
    this.learnEvents++;
    this.pool?.markDirty();
    if (this.cfg.countDecay < 1 && this.learnEvents % this.cfg.decayEvery === 0) {
      this.markov.decay(this.cfg.countDecay);
      this.learner.decayQ(this.cfg.qDecay);
      this.dirty = true;
      this.log(`decayed model (counts x${this.cfg.countDecay}, Q x${this.cfg.qDecay})`, true);
    }
    if (this.learnEvents % this.cfg.dynaEvery === 0) this.maybePlan();
  }

  private maybePlan(): void {
    const cfg = this.cfg;
    if (cfg.replayBatch > 0 && this.replay.length >= 10) {
      for (let i = 0; i < cfg.replayBatch; i++) {
        const t = this.replay[(Math.random() * this.replay.length) | 0];
        this.learner.updateReplay(t.stateKey, t.actionKey, t.reward, this.learner.maxQAtState(t.nextStateKey), t.steps, t.nextStateKey);
      }
    }
    if (cfg.dynaBatch > 0 && this.stateContexts.size > 0 && this.learner.weights.some((w) => w !== 0)) {
      const entries = Array.from(this.stateContexts.entries());
      for (let done = 0, tries = 0; done < cfg.dynaBatch && tries < cfg.dynaBatch * 3; tries++) {
        const [stateKey, sc] = entries[(Math.random() * entries.length) | 0];
        const cand = this.singleTokenCandidate(sc);
        if (!cand) continue;
        const shaped = cfg.valueObjective === 'average'
          ? 0
          : cfg.shaping * (Math.pow(cfg.gamma, cand.tokens.length) * phi(cand.nextPrior) - phi(cand.prior));
        const r = this.learner.surrogateReward(stateKey, cand) + shaped;
        this.learner.updateReplay(stateKey, cand.key, r, this.learner.maxQAtState(cand.nextStateKey), cand.tokens.length, cand.nextStateKey);
        this.learner.stats.dyna++;
        done++;
      }
    }
    this.learner.sweep(cfg.dynaBatch * 4);
    this.dirty = true;
  }

  private singleTokenCandidate(sc: StateContext): Candidate | null {
    const resolver = this.resolver();
    const boost = this.cacheBoostFor(sc.uri);
    const dist = resolver.distAt(sc.lang, sc.lit, sc.abs, boost);
    if (!dist) return null;
    let bestTok = '';
    let bestP = 0;
    for (const [tok, p] of dist) {
      if (tok.includes('\n')) continue;
      if (p > bestP) { bestP = p; bestTok = tok; }
    }
    if (!bestTok) return null;
    const nextPrior = successorPrior(resolver, sc.lang, boost, sc.lit, [bestTok]);
    return this.makeCandidate(sc.lang, sc.lit, [bestTok], bestP, nextPrior, '', sc.atEol, sc.indent, sc.bucket, false, false, '');
  }

  // ------------------------------------------------- resolver / pool

  private resolverCfg() {
    return {
      smoothingD: this.cfg.smoothingD, minCount: this.cfg.minCount, smoothing: this.cfg.smoothing,
      knDiscount: this.cfg.knDiscount, transferWeight: this.cfg.transferWeight,
      familyTransferWeight: this.cfg.familyTransferWeight, cacheD: this.cfg.cacheD,
    };
  }

  private resolver(): DistResolver {
    if (!this._resolver) this._resolver = new DistResolver(this.markov, this.resolverCfg(), this.channelW);
    return this._resolver;
  }

  private cacheBoostFor(uri?: string): Map<string, number> | null {
    return this.cfg.cache && uri ? this.docCaches.get(uri) ?? null : null;
  }

  setPool(pool: GenerationPool | null): void {
    this.pool = pool;
    if (pool) pool.bind(() => this.poolModel());
  }

  private poolModel(): { markov: any; channelW: [string, number[]][]; resolverCfg: any } | null {
    try {
      return {
        markov: this.markov.exportFields(),
        channelW: Array.from(this.channelW.entries()),
        resolverCfg: this.resolverCfg(),
      };
    } catch {
      return null;
    }
  }

  flushPool(force = false): void {
    this.pool?.markDirty();
    this.pool?.pushModel(force);
  }

  // ------------------------------------------------- acting

  effectiveMaxTokens(lang?: string): number {
    const cap = lang ? this.cfgFor(lang).maxTokens : this.cfg.maxTokens;
    if (!this.cfg.adaptiveLength) return cap;
    let base = this.acceptLenEMA + 3;
    if (this.dismissLenEMA > this.acceptLenEMA + 4) base = this.acceptLenEMA + 1;
    return Math.max(4, Math.min(cap, Math.round(base)));
  }

  async generateCandidates(langRaw: string, uri: string, stateKey: string, litBefore: string[], prefix: string, atEol: boolean, indentStr: string, regionTag: string): Promise<Candidate[]> {
    const lang = canonicalLang(langRaw);
    const lcfg = this.cfgFor(lang);
    const eff = this.effectiveMaxTokens(lang);
    const absBefore = litBefore.map((t) => abstractToken(t, lang));
    const bucket = Math.min(3, Math.floor(indentStr.length / 2));

    const params: TokenGenParams = {
      lang, litBefore, absBefore, prefix, atEol,
      maxLines: lcfg.maxLines, minCount: lcfg.minCount, minConfidence: lcfg.minConfidence,
      beamWidth: lcfg.beamWidth, candidates: lcfg.candidates, effMax: eff,
      suggestOnNewline: lcfg.suggestOnNewline, rolloutSteps: lcfg.rolloutSteps,
      diversityLambda: lcfg.diversityLambda,
      maxTokensTotal: Math.max(8, Math.min(96, eff * lcfg.rolloutSteps)),
      cacheBoost: this.cacheBoostFor(uri),
    };

    let tokenCands: TokenCand[] | null = null;
    if (this.cfg.worker && this.pool && this.pool.alive) {
      try {
        tokenCands = await this.pool.generate(requestFromParams(params));
        if (tokenCands !== null) this.genWorker++;
      } catch {
        tokenCands = null;
      }
    }
    if (tokenCands === null) {
      tokenCands = generateTokens(this.resolver(), params);
      this.genLocal++;
    }

    const candidates: Candidate[] = [];
    const seen = new Set<string>();
    for (const tc of tokenCands) {
      const raw = tc.tokens.join('');
      if (raw.trim().length === 0 || raw.length < 2 || seen.has(raw)) continue;
      const cand = this.makeCandidate(lang, litBefore, tc.tokens, tc.prior, tc.nextPrior, prefix, atEol, indentStr, bucket, tc.isRollout, false, regionTag);
      if (this.banned.has(cand.literalKey)) continue;
      seen.add(raw);
      candidates.push(cand);
    }

    if (lcfg.lookahead && lcfg.suggestOnNewline) {
      const la = this.lookaheadCandidate(lang, uri, stateKey, litBefore, absBefore, prefix, atEol, indentStr, bucket, eff, regionTag);
      if (la && !this.banned.has(la.literalKey) && !seen.has(la.tokens.join(''))) candidates.push(la);
    }
    return candidates;
  }

  private makeCandidate(lang: string, litBefore: string[], tokens: string[], prior: number, nextPrior: number, prefix: string, atEol: boolean, indentStr: string, bucket: number, isRollout: boolean, isLookahead: boolean, regionTag: string): Candidate {
    const raw = tokens.join('');
    let text = raw;
    let shifted = false;
    if (raw.includes('\n')) {
      const r = reindent(raw, indentStr);
      if (r !== raw) { text = r; shifted = true; }
    }
    const succ = successorMeta(tokens, atEol, bucket);
    const succCtx = litBefore.concat(tokens);
    const nextStateKey = qStateKeyOf(lang, succCtx, this.cfg.order, succ.bucket, succ.atEol);
    const features = this.featuresOf(lang, raw, tokens, prior, prefix, atEol, bucket, shifted, isRollout, isLookahead, regionTag);
    const literalKey = lang + '\u0003' + tokens.join('\u0000');
    const key = this.cfg.actionKeyMode === 'literal' ? literalKey : shapeKeyOf(lang, tokens);
    if (key !== literalKey) {
      let set = this.shapeLiterals.get(key);
      if (!set) {
        if (this.shapeLiterals.size >= 3000) {
          const first = this.shapeLiterals.keys().next().value;
          if (first !== undefined) this.shapeLiterals.delete(first);
        }
        set = new Set();
        this.shapeLiterals.set(key, set);
      }
      if (!set.has(literalKey)) {
        set.add(literalKey);
        if (set.size > 1) this.learner.stats.shapeMerges++;
      }
    }
    return { key, literalKey, lang, tokens, text, prior, nextPrior, nextStateKey, features, reindented: shifted, isRollout, isLookahead };
  }

  private featuresOf(lang: string, raw: string, tokens: string[], prior: number, prefix: string, atEol: boolean, bucket: number, shifted: boolean, isRollout: boolean, isLookahead: boolean, regionTag: string): number[] {
    return [
      1,
      Math.log1p(prior),
      Math.min(tokens.length, 20) / 20,
      raw.includes('\n') ? 1 : 0,
      shifted ? 1 : 0,
      /;\s*$/.test(raw) || /[})\]]\s*$/.test(raw) ? 1 : 0,
      prefix.length > 0 ? 1 : 0,
      atEol ? 1 : 0,
      bucket / 3,
      profileFor(lang).keywords.has(tokens[0]) ? 1 : 0,
      isRollout ? 1 : 0,
      isLookahead ? 1 : 0,
      regionTag === 'C' ? 1 : 0,
      regionTag === 'F' || regionTag === 'S' ? 1 : 0,
      regionTag === 'I' ? 1 : 0,
    ];
  }

  /** PUCT lookahead: plan with the learned value ensemble as the heuristic. */
  private lookaheadCandidate(lang: string, uri: string, rootStateKey: string, litBefore: string[], absBefore: string[], prefix: string, atEol: boolean, indentStr: string, bucket: number, eff: number, regionTag: string): Candidate | null {
    const lcfg = this.cfgFor(lang);
    const profile = profileFor(lang);
    const resolver = this.resolver();
    const boost = this.cacheBoostFor(uri);
    const firstDist = resolver.distAt(lang, litBefore, absBefore, boost, lcfg.minCount);
    if (!firstDist) return null;
    const priorOf = (tokens: string[]): number => (tokens.length ? firstDist.get(tokens[0]) ?? 1e-6 : 0);
    const valueFn = (tokens: string[]): number => {
      const prior = priorOf(tokens);
      return this.learner.estimate(rootStateKey, {
        key: lang + '\u0003' + tokens.join('\u0000'),
        prior,
        features: this.featuresOf(lang, tokens.join(''), tokens, prior, prefix, atEol, bucket, false, false, true, regionTag),
        lang,
      }).value;
    };
    const distFn = (seq: string[]): Map<string, number> | undefined =>
      resolver.distAt(lang, litBefore.concat(seq), absBefore.concat(seq.map((t) => abstractToken(t, lang))), boost, lcfg.minCount);
    const isTerminal = (tokens: string[]): boolean => {
      const last = tokens[tokens.length - 1] ?? '';
      return profile.terminators.has(last) || last === '}' || last === ')' || last === ']';
    };
    const planned = lookaheadSearch({
      distFn,
      valueFn,
      isAcceptable: (t) => (prefix.length > 0 ? t.length > prefix.length && t.startsWith(prefix) : !(t.includes('\n') && !atEol)),
      isTerminal,
      budget: lcfg.lookaheadBudget,
      maxDepth: Math.min(lcfg.lookaheadDepth, eff),
      cPuct: lcfg.lookaheadCPuct,
      topK: 3,
    });
    if (!planned) return null;
    if (planned.join('').trim().length === 0) return null;
    const prior = priorOf(planned);
    return this.makeCandidate(lang, litBefore, planned, prior, successorPrior(resolver, lang, boost, litBefore, planned), prefix, atEol, indentStr, bucket, false, true, regionTag);
  }

  plan(stateKey: string, candidates: Candidate[], fallbackState?: string): { ordered: Candidate[]; topScore: number; piB: number; greedyShown: boolean; primaryScore: number } {
    const { index, bestScore, piB, argmaxIndex } = this.learner.choose(stateKey, candidates, fallbackState);

    this.lastPlan = candidates.map((c, i) => {
      const parts = this.learner.explain(stateKey, c, fallbackState);
      let score: number;
      if (this.cfg.rankBy === 'value') score = parts.value + this.cfg.beta * Math.log(1 + c.prior);
      else if (this.cfg.rankBy === 'pwin') score = this.learner.pWinOf(stateKey, c.key, fallbackState) + 0.1 * parts.value + this.cfg.beta * Math.log(1 + c.prior);
      else score = this.learner.cvarOf(stateKey, c.key, undefined, fallbackState) + this.cfg.beta * Math.log(1 + c.prior);
      return { text: c.text, key: c.literalKey, prior: c.prior, pinned: this.pinned.has(c.literalKey), chosen: i === index, parts, score };
    });
    const primaryScore = this.lastPlan.find((e) => e.chosen)?.score ?? bestScore;

    const ordered = candidates.slice();
    if (index > 0) {
      const [picked] = ordered.splice(index, 1);
      ordered.unshift(picked);
    }
    const pinnedCands = ordered.filter((c) => this.pinned.has(c.literalKey));
    if (pinnedCands.length > 0) {
      for (let i = ordered.length - 1; i >= 0; i--) if (this.pinned.has(ordered[i].literalKey)) ordered.splice(i, 1);
      ordered.unshift(...pinnedCands);
    }
    const topScore = pinnedCands.length > 0 ? Infinity : bestScore;
    return { ordered, topScore, piB, greedyShown: index === argmaxIndex, primaryScore };
  }

  private effectiveMinScore(): number {
    return this.cfg.minScore + this.gateOffset;
  }

  /** The "stay silent" action with a counterfactual audit trail. */
  gateDecision(stateKey: string, top: Candidate, topScore: number, fallbackKey?: string): { suppress: boolean; probe: boolean } {
    let score = topScore;
    if (this.cfg.refactorBias) {
      if (this.sessionMode === 'R') score *= 0.6;
      else if (this.sessionMode === 'M') score *= 0.8;
    }
    if (score >= this.effectiveMinScore()) return { suppress: false, probe: false };
    const visits = (this.learner.stateVisits.get(stateKey) ?? 0) +
      (fallbackKey && fallbackKey !== stateKey ? this.learner.stateVisits.get(fallbackKey) ?? 0 : 0);
    if (visits < this.cfg.minVisitsGate) return { suppress: false, probe: false };
    if (this.cfg.gateProbe > 0 && Math.random() < this.cfg.gateProbe) {
      this.learner.stats.gatedProbes++;
      return { suppress: false, probe: true };
    }
    this.learner.stats.gated++;
    this.logGated(stateKey, top, score);
    return { suppress: true, probe: false };
  }

  private logGated(stateKey: string, top: Candidate, score: number): void {
    if (!this.cfg.sessionLog) return;
    try {
      fs.appendFileSync(this.sessionLogPath, JSON.stringify({
        ts: Date.now(), lang: top.lang, kind: 'gated', r: 0,
        score: +score.toFixed(3), steps: top.tokens.length,
        s: hashText(stateKey).toString(36), a: hashText(top.literalKey).toString(36),
      }) + '\n');
    } catch { /* ignore */ }
  }

  beginProposal(meta: Omit<PendingProposal, 'primaryId' | 'entries' | 'order'>): void {
    this.pendingProposal = { ...meta, primaryId: -1, entries: new Map(), order: [] };
    if (this.stateContexts.size >= 2000) {
      const first = this.stateContexts.keys().next().value;
      if (first !== undefined) this.stateContexts.delete(first);
    }
    this.stateContexts.set(meta.stateKey, {
      lang: meta.lang,
      uri: meta.uri,
      lit: meta.litBefore.slice(-60),
      abs: meta.litBefore.slice(-60).map((t) => abstractToken(t, meta.lang)),
      indent: meta.indentStr,
      atEol: meta.atEol,
      bucket: meta.indentBucket,
    });
  }

  addProposalEntry(candidate: Candidate, insertText: string, prefixLen: number): number {
    const proposal = this.pendingProposal;
    if (!proposal) return -1;
    const id = this.idSeq++;
    proposal.entries.set(id, { candidate, insertText, prefixLen });
    proposal.order.push(id);
    if (proposal.primaryId === -1) proposal.primaryId = id;
    return id;
  }

  // ------------------------------------------------- rewards

  accept(id: number): void {
    const proposal = this.pendingProposal;
    if (!proposal) return;
    const entry = proposal.entries.get(id);
    if (!entry) return;
    const rank = proposal.order.indexOf(id);
    for (let j = 0; j < rank; j++) {
      const passed = proposal.entries.get(proposal.order[j]);
      if (passed) this.applyOverride(proposal.stateKey, passed.candidate);
    }
    this.registerAccept(proposal, entry);
  }

  detectAcceptance(document: vscode.TextDocument, changes: readonly vscode.TextDocumentContentChangeEvent[]): boolean {
    const proposal = this.pendingProposal;
    if (!proposal || changes.length !== 1) return false;
    const change = changes[0];
    const primary = proposal.entries.get(proposal.primaryId);
    if (!primary) return false;
    if (document.offsetAt(change.range.start) !== proposal.cursorOffset - primary.prefixLen) return false;
    if (change.rangeLength !== primary.prefixLen) return false;

    if (change.text === primary.insertText) {
      this.registerAccept(proposal, primary);
      return true;
    }
    if (change.text.length >= 3 && change.text.length < primary.insertText.length && primary.insertText.startsWith(change.text)) {
      this.pendingProposal = null;
      const frac = change.text.length / primary.insertText.length;
      this.keystrokesSaved += change.text.length;
      this.noteProbe(proposal, true);
      this.applyUpdate(proposal.stateKey, primary.candidate, this.cfg.partialReward * frac, primary.candidate.nextStateKey, 'partial', { piB: proposal.piB, greedyShown: proposal.greedyShown });
      return true;
    }
    return false;
  }

  escapeDismiss(): boolean {
    const proposal = this.pendingProposal;
    if (!proposal) return false;
    const primary = proposal.entries.get(proposal.primaryId);
    this.pendingProposal = null;
    if (!primary) return false;
    this.noteProbe(proposal, false);
    this.applyUpdate(proposal.stateKey, primary.candidate, this.cfg.rewardEscape, primary.candidate.nextStateKey, 'escape', { piB: proposal.piB, greedyShown: proposal.greedyShown });
    return true;
  }

  private registerAccept(proposal: PendingProposal, entry: PendingEntry): void {
    this.pendingProposal = null;
    const charsSaved = Math.max(0, entry.candidate.text.length - entry.prefixLen);
    const utilScale = this.cfg.utilityReward ? 0.5 + 0.5 * Math.min(1, charsSaved / 24) : 1;
    const reward = this.cfg.rewardAccept * utilScale * this.timeMultiplier(proposal.shownAt);
    if (Date.now() - this.lastAcceptAt < 2500) this.learner.stats.chains++;
    this.lastAcceptAt = Date.now();
    this.keystrokesSaved += charsSaved;
    this.noteProbe(proposal, true);

    const startOffset = proposal.cursorOffset - entry.prefixLen;
    this.watches.unshift({
      uri: proposal.uri,
      start: startOffset,
      end: startOffset + entry.candidate.text.length,
      len: entry.candidate.text.length,
      at: Date.now(),
      stateKey: proposal.stateKey,
      candidate: entry.candidate,
    });
    if (this.watches.length > 3) this.watches.length = 3;

    this.applyUpdate(proposal.stateKey, entry.candidate, reward, entry.candidate.nextStateKey, 'accept', { piB: proposal.piB, greedyShown: proposal.greedyShown });
  }

  private noteProbe(p: PendingProposal, good: boolean): void {
    if (!p.probe) return;
    this.probeOutcomes.push(good ? 1 : 0);
    this.probePredicted.push(p.probeScore ?? 0);
    if (this.probeOutcomes.length > 20) { this.probeOutcomes.shift(); this.probePredicted.shift(); }
  }

  private expireWatch(): void {
    if (this.watches.length === 0) return;
    const now = Date.now();
    const survivors: Watch[] = [];
    for (const w of this.watches) {
      if (now - w.at > this.cfg.watchWindow * 1000) {
        if (this.cfg.confirmReward !== 0) {
          this.applyUpdate(w.stateKey, w.candidate, this.cfg.confirmReward, w.candidate.nextStateKey, 'confirm');
        }
      } else survivors.push(w);
    }
    this.watches = survivors;
  }

  /** Watch accepted spans: undo / correction; shift offsets on earlier edits. */
  noteDocumentChange(document: vscode.TextDocument, changes: readonly vscode.TextDocumentContentChangeEvent[]): boolean {
    this.expireWatch();
    if (this.watches.length === 0 || changes.length === 0) return false;
    const uri = document.uri.toString();
    let handled = false;
    const survivors: Watch[] = [];
    for (const w of this.watches) {
      if (w.uri !== uri) { survivors.push(w); continue; }
      let alive = true;
      for (const change of changes) {
        if (!alive) break;
        if (change.rangeOffset === w.start && change.text === w.candidate.text) continue;
        const cStart = change.rangeOffset;
        const cEnd = cStart + change.rangeLength;
        if (cEnd <= w.start) {
          const d = change.text.length - change.rangeLength;
          w.start += d; w.end += d;
          continue;
        }
        if (cStart >= w.end) continue;
        const overlap = Math.min(cEnd, w.end) - Math.max(cStart, w.start);
        const frac = Math.max(0, Math.min(1, overlap / Math.max(1, w.len)));
        if (change.text.length === 0 && change.rangeLength > 0 && frac >= this.cfg.undoFrac && Date.now() - w.at <= this.cfg.undoWindow * 1000) {
          this.keystrokesSaved = Math.max(0, this.keystrokesSaved - change.rangeLength);
          this.applyUpdate(w.stateKey, w.candidate, this.cfg.undoReward * frac, w.candidate.nextStateKey, 'undo');
        } else {
          this.applyUpdate(w.stateKey, w.candidate, this.cfg.correctionReward * Math.max(0.3, frac), w.candidate.nextStateKey, 'correction');
        }
        alive = false;
        handled = true;
      }
      if (alive) survivors.push(w);
    }
    this.watches = survivors;
    return handled;
  }

  private timeMultiplier(shownAt: number): number {
    if (!this.cfg.timeReward) return 1;
    const dt = Date.now() - shownAt;
    return dt < 1500 ? 1.25 : dt > 6000 ? 0.75 : 1;
  }

  settle(document: vscode.TextDocument, position: vscode.Position, textBeforeNow: string): void {
    const proposal = this.pendingProposal;
    if (!proposal) return;
    const sameSpot = document.version === proposal.version && position.line === proposal.line && position.character === proposal.character;
    if (sameSpot) return;
    this.pendingProposal = null;

    const primary = proposal.entries.get(proposal.primaryId);
    if (!primary) return;
    const text = primary.candidate.text;
    const mode = this.sessionMode;

    const typedChars = textBeforeNow.length - proposal.beforeLen;
    if (typedChars > 0) {
      const n = Math.min(typedChars + primary.prefixLen, textBeforeNow.length);
      const typedTail = tail(textBeforeNow, Math.max(n, 1));
      let m = 0;
      while (m < typedTail.length && m < text.length && typedTail[m] === text[m]) m++;
      const frac = m / text.length;
      if (m >= 2 && frac >= 0.5) {
        const nextTokens = tokenize(tail(textBeforeNow, 400), proposal.lang);
        const nextStateKey = qStateKeyOf(proposal.lang, nextTokens, this.cfg.order, proposal.indentBucket, false, { chain: false, mode });
        this.noteProbe(proposal, true);
        this.applyUpdate(proposal.stateKey, primary.candidate, this.cfg.partialReward * frac, nextStateKey, 'partial', { piB: proposal.piB, greedyShown: proposal.greedyShown });
        return;
      }
      if (m > 0) return; // heading toward the snippet -> neutral
    }

    const nextTokens = tokenize(tail(textBeforeNow, 400), proposal.lang);
    const nextStateKey = qStateKeyOf(proposal.lang, nextTokens, this.cfg.order, proposal.indentBucket, false, { chain: false, mode });
    this.noteProbe(proposal, false);
    this.applyUpdate(proposal.stateKey, primary.candidate, this.cfg.rewardDismiss, nextStateKey, 'dismiss', { piB: proposal.piB, greedyShown: proposal.greedyShown });
  }

  private applyUpdate(
    stateKey: string,
    candidate: Candidate,
    reward: number,
    nextStateKey: string,
    kind: 'accept' | 'partial' | 'dismiss' | 'escape' | 'undo' | 'correction' | 'confirm' | 'thumbUp' | 'thumbDown',
    ope?: { piB: number; greedyShown: boolean }
  ): void {
    const shaped = this.cfg.valueObjective === 'average'
      ? 0
      : this.cfg.shaping * (Math.pow(this.cfg.gamma, candidate.tokens.length) * phi(candidate.nextPrior) - phi(candidate.prior));
    const total = reward + shaped;

    const maxNext = this.learner.maxQAtState(nextStateKey);
    const before = this.learner.getQ(stateKey, candidate.key);
    const after = this.learner.update(stateKey, candidate, total, maxNext, candidate.tokens.length, nextStateKey);

    if (kind === 'accept') this.learner.stats.accepts++;
    else if (kind === 'dismiss') this.learner.stats.dismissals++;
    else if (kind === 'partial') this.learner.stats.partials++;
    else if (kind === 'escape') this.learner.stats.escapes++;
    else if (kind === 'undo') this.learner.stats.undos++;
    else if (kind === 'correction') this.learner.stats.corrections++;
    else if (kind === 'thumbUp') this.learner.stats.thumbUps++;
    else if (kind === 'thumbDown') this.learner.stats.thumbDowns++;
    else this.learner.stats.confirmations++;

    const ls = this.langStatFor(candidate.lang);
    if (kind === 'accept') ls.accepts++;
    else if (kind === 'partial') ls.partials++;
    else if (kind === 'dismiss' || kind === 'escape') ls.dismissals++;

    if (this.cfg.adaptiveLength && candidate.tokens.length > 1) {
      if (kind === 'accept' || kind === 'confirm') this.acceptLenEMA = 0.85 * this.acceptLenEMA + 0.15 * candidate.tokens.length;
      else if (kind === 'dismiss' || kind === 'escape' || kind === 'undo' || kind === 'correction') this.dismissLenEMA = 0.9 * this.dismissLenEMA + 0.1 * candidate.tokens.length;
    }

    if (kind === 'accept' || kind === 'partial' || kind === 'dismiss' || kind === 'escape') {
      this.gateWindow.push(kind === 'accept' || kind === 'partial' ? 1 : 0);
      if (this.gateWindow.length > 30) this.gateWindow.shift();
      if (this.cfg.adaptiveGate && this.gateWindow.length === 30) {
        const rate = this.gateWindow.reduce((a, b) => a + b, 0) / 30;
        if (rate < this.cfg.targetAcceptRate - 0.05) this.gateOffset = Math.min(1.5, this.gateOffset + this.cfg.gateStep);
        else if (rate > this.cfg.targetAcceptRate + 0.05) this.gateOffset = Math.max(-0.5, this.gateOffset - this.cfg.gateStep);
        if (this.cfg.gateAudit && this.probeOutcomes.length >= 8) {
          const probeRate = this.probeOutcomes.reduce((a, b) => a + b, 0) / this.probeOutcomes.length;
          if (probeRate >= this.cfg.targetAcceptRate) {
            this.gateOffset = Math.max(-0.75, this.gateOffset - this.cfg.gateStep * 1.5);
            this.learner.stats.gateRelaxations++;
            this.log(`gate audit: probe accept rate ${(probeRate * 100).toFixed(0)}% >= target — relaxing threshold`, true);
            this.probeOutcomes.length = 0;
            this.probePredicted.length = 0;
          }
        }
      }
    }

    if (kind === 'accept' || kind === 'dismiss' || kind === 'escape' || kind === 'undo' || kind === 'correction') {
      const rec = this.topSnippets.get(candidate.literalKey) ?? { text: candidate.text, lang: candidate.lang, accepts: 0, dismissals: 0 };
      if (kind === 'accept') rec.accepts++; else rec.dismissals++;
      this.topSnippets.set(candidate.literalKey, rec);
      if (this.topSnippets.size > 300) {
        let worstK = '', worstV = Infinity;
        for (const [k, r] of this.topSnippets) {
          const v = r.accepts + r.dismissals;
          if (v < worstV) { worstV = v; worstK = k; }
        }
        if (worstK) this.topSnippets.delete(worstK);
      }
    }

    if (this.cfg.replayBuffer > 0) {
      this.replay.push({ stateKey, actionKey: candidate.key, reward: total, nextStateKey, steps: candidate.tokens.length, features: candidate.features.slice(), literalKey: candidate.literalKey });
      if (this.replay.length > this.cfg.replayBuffer) this.replay.splice(0, this.replay.length - this.cfg.replayBuffer);
    }

    if (ope && ope.piB > 0 && (kind === 'accept' || kind === 'partial' || kind === 'dismiss' || kind === 'escape')) {
      this.opeLogs.push({ piB: ope.piB, reward: total, greedyShown: ope.greedyShown });
      if (this.opeLogs.length > 200) this.opeLogs.splice(0, this.opeLogs.length - 200);
      if (this.opeLogs.length >= 30 && this.opeLogs.length % 30 === 0) this.checkOPE();
    }

    if (this.cfg.sessionLog) {
      try {
        fs.appendFileSync(this.sessionLogPath, JSON.stringify({
          ts: Date.now(), lang: candidate.lang, kind, r: +reward.toFixed(3), shaped: +shaped.toFixed(3),
          steps: candidate.tokens.length, s: hashText(stateKey).toString(36), a: hashText(candidate.key).toString(36),
        }) + '\n');
      } catch { /* ignore */ }
    }

    this.events.push({ t: Date.now(), kind });
    if (this.events.length > 3000) this.events.splice(0, this.events.length - 3000);
    this.dirty = true;
    this.log(`[${kind}][${candidate.lang}] r=${reward.toFixed(2)} shaped=${shaped.toFixed(3)}  Q: ${before.toFixed(3)} -> ${after.toFixed(3)}`);
    this.scheduleSave();
  }

  /** Weighted importance sampling for the greedy policy; exploit lock. */
  private checkOPE(): void {
    this.learner.stats.opeChecks++;
    let sw = 0, swr = 0, cnt = 0, onSum = 0;
    for (const l of this.opeLogs) {
      onSum += l.reward;
      if (!l.greedyShown || l.piB <= 0) continue;
      const w = 1 / l.piB;
      sw += w;
      swr += w * l.reward;
      cnt++;
    }
    const onPolicy = onSum / this.opeLogs.length;
    if (sw >= 5) {
      const greedy = swr / sw;
      this.opeEstimate = { greedy, onPolicy, n: cnt };
      if (greedy - onPolicy > 0.05) this.opeStreak++;
      else this.opeStreak = 0;
      if (this.opeStreak >= 3 && this.cfg.autoExploit && !this.learner.exploitLock) {
        this.learner.exploitLock = true;
        this.learner.stats.exploitLocks++;
        this.log(`OPE: greedy policy +${(greedy - onPolicy).toFixed(2)} vs on-policy — exploit lock engaged`, true);
      }
    } else {
      this.opeEstimate = { greedy: 0, onPolicy, n: cnt };
    }
    if (!this.learner.exploitLock && this.opeStreak >= 3) this.opeStreak = 0;
  }

  opeSummary(): { greedy: number; onPolicy: number; n: number } | null {
    return this.opeEstimate;
  }

  private applyOverride(stateKey: string, candidate: Candidate): void {
    if (this.cfg.rewardOverridden === 0) return;
    const maxNext = this.learner.maxQAtState(candidate.nextStateKey);
    this.learner.update(stateKey, candidate, this.cfg.rewardOverridden, maxNext, candidate.tokens.length, candidate.nextStateKey);
    this.learner.stats.overridden++;
    this.dirty = true;
  }

  // ------------------------------------------------- curated + thumbs

  pinSnippet(key: string): void { this.pinned.add(key); this.banned.delete(key); this.dirty = true; this.saveNow(); }
  unpinSnippet(key: string): void { this.pinned.delete(key); this.dirty = true; this.saveNow(); }

  banSnippet(key: string): void {
    this.banned.add(key); this.pinned.delete(key);
    if (this.pendingProposal) {
      for (const [id, e] of Array.from(this.pendingProposal.entries)) {
        if (e.candidate.literalKey === key) this.pendingProposal.entries.delete(id);
      }
    }
    if (this.lastPlan) {
      const e = this.lastPlan.find((x) => x.key === key);
      if (e) this.curatedInfo.set(key, { text: e.text, lang: e.key.slice(0, e.key.indexOf('\u0003')) });
    }
    this.dirty = true;
    this.saveNow();
  }

  unbanSnippet(key: string): void { this.banned.delete(key); this.dirty = true; this.saveNow(); }

  thumb(up: boolean): boolean {
    const kind = up ? 'thumbUp' : 'thumbDown';
    const r = up ? this.cfg.rewardThumbUp : this.cfg.rewardThumbDown;
    if (this.pendingProposal) {
      const p = this.pendingProposal;
      const primary = p.entries.get(p.primaryId);
      if (primary) {
        this.applyUpdate(p.stateKey, primary.candidate, r, primary.candidate.nextStateKey, kind, { piB: p.piB, greedyShown: p.greedyShown });
        return true;
      }
    }
    const w = this.watches[0];
    if (w) { this.applyUpdate(w.stateKey, w.candidate, r, w.candidate.nextStateKey, kind); return true; }
    return false;
  }

  explainLastPlan() { return this.lastPlan; }
  currentTopKeys(): string[] { return this.lastTopKeys; }
  currentBannedKeys(): string[] { return this.lastBannedKeys; }

  // ------------------------------------------------- corpus learning (v1.0)

  /** Learn corpus items into the world model ONLY, at datasetWeight. */
  learnCorpusItems(items: CorpusItem[]): { learned: number; skipped: number; tokens: number } {
    let learned = 0, skipped = 0, tokens = 0;
    const w = this.cfg.datasetWeight;
    for (const it of items) {
      const lang = canonicalLang(it.lang);
      if (!lang) { skipped++; continue; }
      if (this.cfg.languages.length > 0 && !this.cfg.languages.includes(lang)) { skipped++; continue; }
      const lines = it.text.split('\n');
      if (lines.length > 50 && it.text.length / lines.length > 400) { skipped++; continue; }
      const seq = tokenize(it.text, lang);
      if (seq.length < 2) { skipped++; continue; }
      this.markov.learn(lang, seq, w, w * this.cfg.transferWeight, w * this.cfg.familyTransferWeight);
      this.corpusDocs++;
      this.corpusTokens += seq.length;
      tokens += seq.length;
      learned++;
    }
    if (learned > 0) { this.dirty = true; this.pool?.markDirty(); }
    return { learned, skipped, tokens };
  }

  async collectCorpusFiles(root: vscode.Uri, cancel?: { isCancelled(): boolean }): Promise<vscode.Uri[]> {
    const out: vscode.Uri[] = [];
    const stack: string[] = [root.fsPath];
    while (stack.length) {
      if (out.length >= this.cfg.datasetMaxFiles) break;
      if (cancel?.isCancelled()) break;
      const dir = stack.pop()!;
      let entries: fs.Dirent[];
      try { entries = await fs.promises.readdir(dir, { withFileTypes: true }); } catch { continue; }
      for (const e of entries) {
        if (out.length >= this.cfg.datasetMaxFiles) break;
        const full = path.join(dir, e.name);
        if (e.isDirectory()) {
          if (!CORPUS_IGNORE.has(e.name) && !e.name.startsWith('.')) stack.push(full);
        } else if (e.isFile() && langForFilename(e.name)) {
          out.push(vscode.Uri.file(full));
        }
      }
    }
    return out;
  }

  async learnFromFolder(root: vscode.Uri, report: (msg: string) => void, cancel?: { isCancelled(): boolean }): Promise<{ learned: number; skipped: number; tokens: number }> {
    const files = await this.collectCorpusFiles(root, cancel);
    let learned = 0, skipped = 0, tokens = 0;
    for (let i = 0; i < files.length; i++) {
      if (cancel?.isCancelled()) break;
      const uri = files[i];
      try {
        const stat = await fs.promises.stat(uri.fsPath);
        if (stat.size > this.cfg.datasetMaxFileMb * 1024 * 1024) { skipped++; continue; }
        const content = await fs.promises.readFile(uri.fsPath, 'utf8');
        const key = uri.toString();
        const hash = hashText(content);
        if (this.learnedHashes.get(key) === hash) { skipped++; continue; }
        this.learnedHashes.set(key, hash);
        const res = this.learnCorpusItems([{ lang: langForFilename(path.basename(uri.fsPath)), text: content }]);
        learned += res.learned; skipped += res.skipped; tokens += res.tokens;
      } catch { skipped++; }
      if (i % 20 === 0) report(`${i + 1}/${files.length}`);
    }
    this.noteLearnEvent();
    this.dirty = true;
    this.flushPool(true);
    this.scheduleSave();
    return { learned, skipped, tokens };
  }

  async learnFromDatasetFile(uri: vscode.Uri, defaultLang: string): Promise<{ learned: number; skipped: number; tokens: number; items: number }> {
    const content = await fs.promises.readFile(uri.fsPath, 'utf8');
    const key = uri.toString();
    const hash = hashText(content);
    if (this.learnedHashes.get(key) === hash) return { learned: 0, skipped: 0, tokens: 0, items: 0 };
    this.learnedHashes.set(key, hash);
    const items = parseCorpusContent(path.basename(uri.fsPath), content, defaultLang);
    const res = this.learnCorpusItems(items);
    this.noteLearnEvent();
    this.dirty = true;
    this.flushPool(true);
    this.scheduleSave();
    return { ...res, items: items.length };
  }

  /** Pure read (no learning) for evaluation/tuning. */
  async loadCorpusFromUris(uris: vscode.Uri[], defaultLang: string): Promise<CorpusItem[]> {
    const out: CorpusItem[] = [];
    for (const uri of uris) {
      const stat = await fs.promises.stat(uri.fsPath).catch(() => null);
      if (!stat) continue;
      if (stat.isDirectory()) {
        for (const f of await this.collectCorpusFiles(uri)) {
          try {
            const st = await fs.promises.stat(f.fsPath);
            if (st.size > this.cfg.datasetMaxFileMb * 1024 * 1024) continue;
            const content = await fs.promises.readFile(f.fsPath, 'utf8');
            out.push({ lang: langForFilename(path.basename(f.fsPath)), text: content });
          } catch { /* skip */ }
        }
      } else {
        try {
          const content = await fs.promises.readFile(uri.fsPath, 'utf8');
          for (const it of parseCorpusContent(path.basename(uri.fsPath), content, defaultLang)) {
            if (it.lang) out.push(it);
          }
        } catch { /* skip */ }
      }
    }
    return out;
  }

  evaluateOnCorpus(items: CorpusItem[]): { overall: EvalMetrics; perLang: Array<{ lang: string; metrics: EvalMetrics }>; items: number } {
    const byLang = new Map<string, string[][]>();
    let count = 0;
    for (const it of items) {
      const lang = canonicalLang(it.lang);
      if (!lang) continue;
      if (this.cfg.languages.length > 0 && !this.cfg.languages.includes(lang)) continue;
      const seq = tokenize(it.text, lang);
      if (seq.length < 2) continue;
      let arr = byLang.get(lang);
      if (!arr) { arr = []; byLang.set(lang, arr); }
      arr.push(seq);
      count++;
    }
    const resolver = new DistResolver(this.markov, this.resolverCfg(), this.channelW);
    const perLang: Array<{ lang: string; metrics: EvalMetrics }> = [];
    let ll = 0, n = 0, t1 = 0, t3 = 0, t5 = 0, oov = 0;
    let budget = this.cfg.evalTokenBudget;
    for (const [lang, seqs] of byLang) {
      if (budget <= 0) break;
      const m = evaluateSequences(resolver, lang, seqs, budget);
      perLang.push({ lang, metrics: m });
      ll += m.logLoss * m.predictions; n += m.predictions;
      t1 += m.top1 * m.predictions; t3 += m.top3 * m.predictions; t5 += m.top5 * m.predictions;
      oov += m.oovRate * m.predictions;
      budget -= m.predictions;
    }
    const overall: EvalMetrics = {
      predictions: n, logLoss: n ? ll / n : 0, perplexity: n ? Math.exp(ll / n) : Infinity,
      top1: n ? t1 / n : 0, top3: n ? t3 / n : 0, top5: n ? t5 / n : 0, oovRate: n ? oov / n : 0,
    };
    return { overall, perLang, items: count };
  }

  tuneOnCorpus(items: CorpusItem[]): TuneOutcome {
    return gridSearchTune({
      items, currentModel: this.markov, currentOrder: this.cfg.order, channelW: this.channelW,
      baseCfg: this.resolverCfg(), orders: this.cfg.tuneOrders,
      smoothingDs: [2, 4, 6, 10], knDiscounts: [0, 0.5, 0.75, 1.0],
      tokenBudget: this.cfg.tuneTokenBudget, evalBudget: this.cfg.evalTokenBudget,
    });
  }

  // ------------------------------------------------- views

  knownLanguages(): { id: string; family: string; bigrams: number; accepts: number }[] {
    const ids = new Set<string>(this.langStats.keys());
    for (const l of this.markov.languages()) {
      if (l !== '*' && !FAMILIES.has(l)) ids.add(l);
    }
    return Array.from(ids)
      .map((id) => ({ id, family: familyOf(id), bigrams: this.markov.bigramsFor(id), accepts: this.langStats.get(id)?.accepts ?? 0 }))
      .sort((a, b) => b.bigrams - a.bigrams);
  }

  resetLanguage(langRaw: string): void {
    const lang = canonicalLang(langRaw);
    this.markov.dropLanguage(lang);
    this.learner.dropLanguage(lang);
    this.langStats.delete(lang);
    this.channelW.delete(lang);
    for (const k of Array.from(this.topSnippets.keys())) if (k.startsWith(lang + '\u0003')) this.topSnippets.delete(k);
    for (const k of Array.from(this.shapeLiterals.keys())) if (k.startsWith(lang + '\u0003')) this.shapeLiterals.delete(k);
    this.replay = this.replay.filter((t) => !t.stateKey.startsWith(lang + '\u0002'));
    this.watches = this.watches.filter((w) => !w.stateKey.startsWith(lang + '\u0002'));
    if (this.pendingProposal && this.pendingProposal.stateKey.startsWith(lang + '\u0002')) this.pendingProposal = null;
    this.dirty = true;
    this.saveNow();
    this.log(`reset language model: ${lang} (family/universal knowledge kept)`, true);
  }

  summary() {
    const s = this.learner.stats;
    return {
      bigrams: this.markov.distinctBigrams(),
      states: this.markov.contextStates(),
      q: this.learner.q.size,
      qShadow: this.learner.qShadow.size,
      accepts: s.accepts, dismissals: s.dismissals, partials: s.partials, chains: s.chains,
      gated: s.gated, gatedProbes: s.gatedProbes, explorations: s.explorations, escapes: s.escapes,
      overridden: s.overridden, undos: s.undos, corrections: s.corrections, confirmations: s.confirmations,
      replays: s.replays, dyna: s.dyna, sweeps: s.sweeps, factorTrainings: s.factorTrainings,
      driftAlarms: s.driftAlarms, driftActive: this.learner.driftActive,
      opeChecks: s.opeChecks, exploitLocks: s.exploitLocks, exploitLock: this.learner.exploitLock,
      files: this.stats.filesLearned, languages: this.langStats.size,
      keystrokesSaved: this.keystrokesSaved,
      predAccuracy: this.predTotal >= 20 ? this.predEMA : null,
      rho: this.cfg.valueObjective === 'average' ? this.learner.rho : null,
      objective: this.cfg.valueObjective, rankBy: this.cfg.rankBy,
      effMaxTokens: this.effectiveMaxTokens(), sessionMode: this.sessionMode, gateOffset: this.gateOffset,
      workerGen: this.genWorker, localGen: this.genLocal, fittedQ: s.fittedQ,
      ws: this.wsTag || null, workspaceConditioning: this.cfg.workspaceConditioning,
      probeRate: this.probeOutcomes.length >= 5 ? this.probeOutcomes.reduce((a, b) => a + b, 0) / this.probeOutcomes.length : null,
      gateRelaxations: s.gateRelaxations, shapeMerges: s.shapeMerges, actionKeyMode: this.cfg.actionKeyMode,
      thumbUps: s.thumbUps, thumbDowns: s.thumbDowns,
      pinned: this.pinned.size, banned: this.banned.size, lookahead: this.cfg.lookahead,
      corpusDocs: this.corpusDocs, corpusTokens: this.corpusTokens,
    };
  }

  dashboardData(): DashboardData {
    const s = this.summary();
    const buckets: { rate: number }[] = [];
    const size = 25;
    for (let i = Math.max(0, this.events.length - 48 * size); i < this.events.length; i += size) {
      const chunk = this.events.slice(i, i + size);
      const good = chunk.filter((e) => e.kind === 'accept' || e.kind === 'partial' || e.kind === 'confirm').length;
      buckets.push({ rate: good / chunk.length });
    }
    const ope = this.opeSummary();
    this.lastTopKeys = [];
    const top = Array.from(this.topSnippets.entries())
      .sort((a, b) => b[1].accepts - a[1].accepts)
      .slice(0, 12)
      .map(([key, r]) => { this.lastTopKeys.push(key); return { key, text: r.text, lang: r.lang, accepts: r.accepts, dismissals: r.dismissals, pinned: this.pinned.has(key) }; });
    this.lastBannedKeys = Array.from(this.banned.keys());
    const banned = this.lastBannedKeys.map((key) => ({ key, text: this.curatedInfo.get(key)?.text ?? key.slice(key.indexOf('\u0003') + 1).replace(/\u0000/g, ' ') }));
    const langs = this.knownLanguages().map((l) => {
      const st = this.langStats.get(l.id);
      const w = this.channelW.get(l.id) ?? [1, this.cfg.familyTransferWeight, this.cfg.transferWeight];
      const t = w[0] + w[1] + w[2];
      return {
        lang: l.id, family: l.family, bigrams: l.bigrams,
        accepts: st?.accepts ?? 0, dismissals: st?.dismissals ?? 0, partials: st?.partials ?? 0,
        accuracy: st && st.predTotal >= 20 ? st.predHits / st.predTotal : null,
        transfer: [w[0] / t, w[1] / t, w[2] / t] as [number, number, number],
      };
    });
    return {
      stats: {
        ...s,
        opeGreedy: ope ? ope.greedy : null, opeOnPolicy: ope ? ope.onPolicy : null, opeN: ope ? ope.n : 0,
        weights: this.learner.weights.map((w) => +w.toFixed(3)),
      },
      buckets, langs, top, banned,
      order: this.cfg.order, exploration: this.cfg.exploration, objective: this.cfg.valueObjective,
    };
  }

  /** Human-readable digest of everything the agent has learned. */
  learningReport(): string {
    const s = this.summary();
    const L: string[] = [];
    const pct = (x: number): string => `${Math.round(x * 100)}%`;
    L.push('# MDP Snippets — learning report', '', `_Generated ${new Date().toLocaleString()}_`, '');
    L.push('## World model (the transition function)', '');
    L.push(`- Markov order **${this.cfg.order}** · smoothing **${this.cfg.smoothing}** (D = ${this.cfg.smoothingD}${this.cfg.smoothing === 'kneser-ney' ? `, KN discount = ${this.cfg.knDiscount}` : ''})`);
    L.push(`- **${s.bigrams.toLocaleString()}** distinct bigrams · **${s.states.toLocaleString()}** context states`);
    L.push(`- Sources: **${s.files}** open files · **${s.corpusDocs.toLocaleString()}** corpus docs (**${s.corpusTokens.toLocaleString()}** tokens, learned at weight ${this.cfg.datasetWeight})`);
    L.push(`- Action keys: ${s.actionKeyMode} (${s.shapeMerges.toLocaleString()} shape merges)`, '');
    L.push('### Per-language', '', '| language | family | bigrams | accepts | next-token acc. | transfer self/family/* |', '|---|---|---|---|---|---|');
    for (const l of this.knownLanguages()) {
      const st = this.langStats.get(l.id);
      const w = this.channelW.get(l.id) ?? [1, this.cfg.familyTransferWeight, this.cfg.transferWeight];
      const t = w[0] + w[1] + w[2];
      const acc = st && st.predTotal >= 20 ? pct(st.predHits / st.predTotal) : '—';
      L.push(`| ${l.id} | ${l.family} | ${l.bigrams.toLocaleString()} | ${st?.accepts ?? 0} | ${acc} | ${pct(w[0] / t)} / ${pct(w[1] / t)} / ${pct(w[2] / t)} |`);
    }
    const w = this.learner.weights;
    if (w.some((x) => Math.abs(x) >= 0.01)) {
      L.push('', '## What it learned about your taste', '', 'Preference weights (positive = you accept it more):', '');
      const named = w.map((v, i) => ({ name: FEATURE_NAMES[i] ?? `phi${i}`, v })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      for (const n of named) {
        if (Math.abs(n.v) < 0.01) continue;
        L.push(`- **${n.name}**: ${n.v >= 0 ? '+' : ''}${n.v.toFixed(2)}`);
      }
    }
    L.push('', '## Feedback ledger (the reward function)', '');
    L.push(`- Accepted **${s.accepts}** · dismissed ${s.dismissals} · Esc ${s.escapes} · partial ${s.partials} · undo ${s.undos} · corrections ${s.corrections} · confirmations ${s.confirmations}`);
    L.push(`- Thumbs: ${s.thumbUps} up / ${s.thumbDowns} down · Tab-chains ${s.chains} · overridden ${s.overridden}`);
    const total = s.accepts + s.dismissals + s.partials;
    if (total > 0) L.push(`- Accept rate: **${pct(s.accepts / total)}** (gate target ${pct(this.cfg.targetAcceptRate)})`);
    L.push('', '## Decision machinery', '');
    L.push(`- Gate: offset ${s.gateOffset >= 0 ? '+' : ''}${s.gateOffset.toFixed(2)} · probe accept ${s.probeRate === null ? '—' : pct(s.probeRate)} · ${s.gateRelaxations} relaxation(s) · ${s.gated} suppressed (${s.gatedProbes} probed)`);
    L.push(`- Exploration: ${this.cfg.exploration}${s.exploitLock ? ' · **exploit lock ON**' : ''} · ranking by ${s.rankBy} · objective ${s.objective}`);
    L.push(`- Planning: ${s.replays.toLocaleString()} replays · ${s.dyna.toLocaleString()} Dyna · ${s.sweeps.toLocaleString()} sweeps · ${s.fittedQ.toLocaleString()} fitted-Q · ${s.factorTrainings.toLocaleString()} factor steps`);
    const ope = this.opeSummary();
    L.push(`- OPE: ${ope && ope.n > 0 ? `greedy ${ope.greedy.toFixed(2)} vs current ${ope.onPolicy.toFixed(2)} (n=${ope.n})` : 'collecting data'} · drift alarms ${s.driftAlarms}`);
    L.push(`- Generation: ${s.workerGen} worker / ${s.localGen} local · effective cap ${s.effMaxTokens} tokens`);
    L.push('', '## Results', '');
    L.push(`- Keystrokes saved: **${s.keystrokesSaved.toLocaleString()}** · live next-token accuracy: ${s.predAccuracy === null ? 'warming up' : pct(s.predAccuracy)}`);
    const top = Array.from(this.topSnippets.values()).sort((a, b) => b.accepts - a.accepts).slice(0, 5);
    if (top.length) {
      L.push('', 'Top snippets:', '');
      for (const t of top) L.push(`- \`${t.text.replace(/`/g, "'").replace(/\n/g, ' ⏎ ').slice(0, 80)}\` — ${t.accepts} up ${t.dismissals} down (${t.lang})`);
    }
    return L.join('\n');
  }

  // ------------------------------------------------- transfer

  importModel(uri: vscode.Uri): { bigrams: number; qEntries: number } {
    const raw = JSON.parse(fs.readFileSync(uri.fsPath, 'utf8')) as any;
    const otherMarkov = MarkovModel.deserialize(raw?.markov);
    if (!otherMarkov) throw new Error('not an MDP Snippets model file');
    if (otherMarkov.order !== this.markov.order) {
      throw new Error(`model order ${otherMarkov.order} does not match current order ${this.markov.order}`);
    }
    this.markov.mergeFrom(otherMarkov, 1);
    const otherLearner = QLearner.deserialize(raw?.learner);
    if (otherLearner) this.learner.mergeFrom(otherLearner);
    this.dirty = true;
    this.saveNow();
    return { bigrams: this.markov.distinctBigrams(), qEntries: this.learner.q.size };
  }

  /** Privacy-safe taste profile: family/universal channels + preference
   *  weights + abstract state data + values for PURE-SYNTAX actions only. */
  buildTasteProfile(): { profile: any; safeActions: number; channels: string[] } {
    const safeAction = (actionKey: string): boolean => {
      const i = actionKey.indexOf('\u0003');
      if (i < 0) return false;
      const lang = actionKey.slice(0, i);
      return actionKey.slice(i + 1).split('\u0000').every((t) => {
        if (t === '\n' || t === '\r\n' || t === ' ') return true;
        const a = abstractToken(t, lang);
        return a !== '#I' && a !== '#N' && a !== '#S' && a !== '#C';
      });
    };
    const channels = this.markov.languages().filter((l) => l === '*' || FAMILIES.has(l));
    const learnerTaste = this.learner.exportTaste(safeAction);
    const markovTaste = this.markov.exportChannels((ch) => ch === '*' || FAMILIES.has(ch));
    const profile = {
      type: 'mdp-taste-profile',
      v: 1,
      createdAt: new Date().toISOString(),
      order: this.cfg.order,
      markov: markovTaste,
      learner: learnerTaste,
      channelW: Array.from(this.channelW.entries()),
    };
    return { profile, safeActions: learnerTaste.q.length + learnerTaste.actionEmb.length, channels };
  }

  importTasteProfile(uri: vscode.Uri): { channels: number; safeActions: number } {
    const raw = JSON.parse(fs.readFileSync(uri.fsPath, 'utf8')) as any;
    if (raw?.type !== 'mdp-taste-profile') throw new Error('not an MDP Snippets taste profile');
    const other = MarkovModel.deserialize(raw.markov);
    if (!other) throw new Error('corrupt taste profile');
    if (other.order !== this.markov.order) throw new Error(`profile order ${other.order} does not match current order ${this.markov.order}`);
    this.markov.mergeFrom(other, 1);
    if (raw.learner) this.learner.mergeTaste(raw.learner, 0.5);
    if (Array.isArray(raw.channelW)) {
      for (const e of raw.channelW) {
        if (Array.isArray(e) && typeof e[0] === 'string' && Array.isArray(e[1]) && e[1].length === 3) {
          const mine = this.channelW.get(e[0]);
          if (!mine) this.channelW.set(e[0], e[1].map((x: unknown) => (typeof x === 'number' ? x : 0.25)));
          else for (let i = 0; i < 3; i++) mine[i] = (mine[i] + (typeof e[1][i] === 'number' ? e[1][i] : mine[i])) / 2;
        }
      }
    }
    this.dirty = true;
    this.flushPool(true);
    this.saveNow();
    return {
      channels: other.languages().filter((l) => l === '*' || FAMILIES.has(l)).length,
      safeActions: Array.isArray(raw.learner?.q) ? raw.learner.q.length : 0,
    };
  }

  // ------------------------------------------------- lifecycle

  warmStart(): { passes: number; updates: number } | null {
    if (!this.cfg.fittedQ || this.replay.length < 20) return null;
    const res = this.learner.fittedQ(this.replay, { epochs: this.cfg.fittedQEpochs, lr: 0.25 });
    this.log(`fitted-Q warm start: ${res.updates} updates over ${this.replay.length} transitions (${res.passes} epochs)`, true);
    return res;
  }

  reset(): void {
    this.markov = new MarkovModel(this.cfg.order);
    this.learner = new QLearner(this.cfg);
    this.learnedHashes.clear();
    this.pendingProposal = null;
    this.stats.filesLearned = 0;
    this.events = [];
    this.topSnippets.clear();
    this.replay = [];
    this.stateContexts.clear();
    this.docCaches.clear();
    this.watches = [];
    this.gateWindow = [];
    this.gateOffset = 0;
    this.opeLogs = [];
    this.opeStreak = 0;
    this.opeEstimate = null;
    this.editIns = 0;
    this.editDel = 0;
    this.channelW.clear();
    this.langStats.clear();
    this.shapeLiterals.clear();
    this.probeOutcomes = [];
    this.probePredicted = [];
    this.pinned.clear();
    this.banned.clear();
    this.curatedInfo.clear();
    this.corpusDocs = 0;
    this.corpusTokens = 0;
    this.keystrokesSaved = 0;
    this.predHits = 0;
    this.predTotal = 0;
    this.predEMA = 0.5;
    this.lastAcceptAt = 0;
    this.acceptLenEMA = this.cfg.maxTokens;
    this.dismissLenEMA = this.cfg.maxTokens;
    this._resolver = null;
    this.genWorker = 0;
    this.genLocal = 0;
    if (this.cfg.persist) this.saveNow();
    else {
      try { if (fs.existsSync(this.modelPath)) fs.unlinkSync(this.modelPath); } catch { /* ignore */ }
    }
  }

  tick(): void {
    this.expireWatch();
    this.flushPool();
    this.saveIfDirty();
  }

  // ------------------------------------------------- persistence

  private load(): void {
    if (!this.cfg.persist) return;
    try {
      if (!fs.existsSync(this.modelPath)) return;
      const raw = JSON.parse(fs.readFileSync(this.modelPath, 'utf8')) as any;
      if (raw?.markov) {
        const model = MarkovModel.deserialize(raw.markov);
        if (model && model.order === this.cfg.order) this.markov = model;
        else if (model) this.log(`ignoring persisted model (order ${model.order} != configured ${this.cfg.order})`, true);
      }
      if (raw?.learner) {
        const learner = QLearner.deserialize(raw.learner);
        if (learner) { learner.updateParams(this.cfg); this.learner = learner; }
      }
      const loadNumMap = (m: Map<string, number>, arr: any): void => {
        if (Array.isArray(arr)) {
          for (const pair of arr) {
            if (Array.isArray(pair) && typeof pair[0] === 'string' && typeof pair[1] === 'number') m.set(pair[0], pair[1]);
          }
        }
      };
      if (Array.isArray(raw?.learned)) loadNumMap(this.learnedHashes, raw.learned);
      if (Array.isArray(raw?.topSnippets)) {
        for (const pair of raw.topSnippets) {
          if (Array.isArray(pair) && typeof pair[0] === 'string' && pair[1] && typeof pair[1] === 'object') {
            this.topSnippets.set(pair[0], {
              text: String(pair[1].text ?? ''), lang: String(pair[1].lang ?? ''),
              accepts: Number(pair[1].accepts) || 0, dismissals: Number(pair[1].dismissals) || 0,
            });
          }
        }
      }
      if (Array.isArray(raw?.replay)) {
        for (const t of raw.replay) {
          if (t && typeof t.stateKey === 'string' && typeof t.actionKey === 'string' &&
              typeof t.reward === 'number' && typeof t.nextStateKey === 'string' && typeof t.steps === 'number') {
            this.replay.push({
              stateKey: t.stateKey, actionKey: t.actionKey, reward: t.reward,
              nextStateKey: t.nextStateKey, steps: t.steps,
              features: Array.isArray(t.features) ? t.features.filter((x: unknown) => typeof x === 'number' && Number.isFinite(x)) : undefined,
              literalKey: typeof t.literalKey === 'string' ? t.literalKey : undefined,
            });
          }
        }
      }
      if (Array.isArray(raw?.channelW)) {
        for (const e of raw.channelW) {
          if (Array.isArray(e) && typeof e[0] === 'string' && Array.isArray(e[1]) && e[1].length === 3) {
            this.channelW.set(e[0], e[1].map((x: unknown) => (typeof x === 'number' && x >= 0 ? x : 0.02)));
          }
        }
      }
      if (Array.isArray(raw?.langStats)) {
        for (const e of raw.langStats) {
          if (Array.isArray(e) && typeof e[0] === 'string' && e[1] && typeof e[1] === 'object') {
            this.langStats.set(e[0], {
              accepts: Number(e[1].accepts) || 0, dismissals: Number(e[1].dismissals) || 0,
              partials: Number(e[1].partials) || 0, predHits: Number(e[1].predHits) || 0, predTotal: Number(e[1].predTotal) || 0,
            });
          }
        }
      }
      if (Array.isArray(raw?.pinned)) for (const k of raw.pinned) if (typeof k === 'string') this.pinned.add(k);
      if (Array.isArray(raw?.banned)) for (const k of raw.banned) if (typeof k === 'string') this.banned.add(k);
      if (Array.isArray(raw?.curatedInfo)) {
        for (const e of raw.curatedInfo) {
          if (Array.isArray(e) && typeof e[0] === 'string' && e[1] && typeof e[1] === 'object') {
            this.curatedInfo.set(e[0], { text: String(e[1].text ?? ''), lang: String(e[1].lang ?? '') });
          }
        }
      }
      if (typeof raw?.keystrokesSaved === 'number') this.keystrokesSaved = raw.keystrokesSaved;
      if (typeof raw?.corpusDocs === 'number') this.corpusDocs = raw.corpusDocs;
      if (typeof raw?.corpusTokens === 'number') this.corpusTokens = raw.corpusTokens;
      // migrate pre-v9 literal action keys to semantic shapes, then rebuild the index
      if ((raw?.v ?? 0) < 9 && this.cfg.actionKeyMode !== 'literal') {
        const merges = this.learner.migrateToShape();
        for (const t of this.replay) {
          const i = t.actionKey.indexOf('\u0003');
          if (i >= 0) t.actionKey = shapeKeyOf(t.actionKey.slice(0, i), t.actionKey.slice(i + 1).split('\u0000'));
        }
        this.log(`migrated ${merges} literal action keys to semantic shapes`, true);
      }
      this.learner.ingestReplay(this.replay);
      this.log(`loaded model: ${this.markov.distinctBigrams()} bigrams, ${this.learner.q.size} Q-entries, ${this.replay.length} replayed transitions`, true);
    } catch (err) {
      this.log(`failed to load model: ${String(err)}`, true);
    }
  }

  saveNow(): void {
    if (!this.cfg.persist) return;
    try {
      this.markov.prune(150_000);
      this.learner.prune(50_000);
      const payload = {
        v: 9,
        savedAt: new Date().toISOString(),
        markov: this.markov.serialize(),
        learner: this.learner.serialize(),
        learned: Array.from(this.learnedHashes.entries()),
        topSnippets: Array.from(this.topSnippets.entries()),
        replay: this.replay.slice(-this.cfg.replayBuffer),
        channelW: Array.from(this.channelW.entries()),
        langStats: Array.from(this.langStats.entries()),
        pinned: Array.from(this.pinned),
        banned: Array.from(this.banned),
        curatedInfo: Array.from(this.curatedInfo.entries()),
        keystrokesSaved: this.keystrokesSaved,
        corpusDocs: this.corpusDocs,
        corpusTokens: this.corpusTokens,
      };
      fs.mkdirSync(path.dirname(this.modelPath), { recursive: true });
      const tmp = this.modelPath + '.tmp';
      fs.writeFileSync(tmp, JSON.stringify(payload));
      fs.renameSync(tmp, this.modelPath);
      this.dirty = false;
    } catch (err) {
      this.log(`failed to save model: ${String(err)}`, true);
    }
  }

  saveIfDirty(): void { if (this.dirty) this.saveNow(); }

  private scheduleSave(): void {
    this.dirty = true;
    if (this.saveTimer) return;
    this.saveTimer = setTimeout(() => { this.saveTimer = null; this.saveNow(); }, 5_000);
  }

  private log(msg: string, force = false): void {
    if (force || this.cfg.debug) this.logFn(msg);
  }
}
