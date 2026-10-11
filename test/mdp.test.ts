import { describe, it, expect } from 'bun:test';
import { tokenize, abstractToken, shapeKeyOf, mixDistributions } from '../src/util';
import { canonicalLang, familyOf, profileFor, langForFilename } from '../src/languages';
import { MarkovModel } from '../src/markov';
import { QLearner } from '../src/learner';
import { DistResolver, generateTokens, qStateKeyOf, type TokenGenParams } from '../src/generate';
import { regionAt } from '../src/region';
import { lookaheadSearch } from '../src/planner';
import { parseJsonlDataset, parseMarkdownFences, datasetNeedsLanguage, splitTrainVal, evaluateSequences, gridSearchTune } from '../src/dataset';

const P = {
  alpha: 0.5, alphaMode: 'fixed' as const, gamma: 0.9, lambda: 0.8, epsilon: 0, beta: 1,
  exploration: 'epsilon' as const, ucbC: 0.3, temperature: 0.35, thompsonSigma: 0.5, weightAlpha: 0.5,
  valueObjective: 'discounted' as const, rhoAlpha: 0.1, processNoise: 0.01, obsNoise: 0.5,
  factorDim: 0, factorEta: 0.05, riskKappa: 0, driftThreshold: 2, sweepPerUpdate: 0,
};
const CFG = { smoothingD: 2, minCount: 1, smoothing: 'interpolated' as const, knDiscount: 0, transferWeight: 0, familyTransferWeight: 0, cacheD: 8 };

describe('languages & tokenization', () => {
  it('canonicalizes aliases and maps families/profiles', () => {
    expect(canonicalLang('typescriptreact')).toBe('typescript');
    expect(canonicalLang('Python')).toBe('python');
    expect(familyOf('go')).toBe('c-like');
    expect(profileFor('python').terminators.size).toBe(0);
    expect(profileFor('python').eolTerminates).toBe(true);
    expect(profileFor('typescript').terminators.has(';')).toBe(true);
    expect(profileFor('typescript').hashComments).toBe(false);
    expect(profileFor('python').hashComments).toBe(true);
  });
  it('tokenizes: whitespace verbatim, operators, per-language comments', () => {
    expect(tokenize('a\n  b')).toEqual(['a', '\n', '  ', 'b']);
    expect(tokenize('a=>b')).toEqual(['a', '=>', 'b']);
    expect(tokenize('let #foo = 1', 'typescript')).toEqual(['let', ' ', '#foo', ' ', '=', ' ', '1']);
    expect(tokenize('x = 1 # note', 'python')).toContain('# note');
    expect(tokenize('a // hi\nb')).toEqual(['a', ' ', '// hi', '\n', 'b']);
  });
  it('abstracts keywords per language', () => {
    expect(abstractToken('foo')).toBe('#I');
    expect(abstractToken('return')).toBe('return');
    expect(abstractToken('end', 'ruby')).toBe('end');
    expect(abstractToken('end', 'typescript')).toBe('#I');
    expect(abstractToken('42')).toBe('#N');
    expect(abstractToken('"s"')).toBe('#S');
  });
  it('maps filenames to languages', () => {
    expect(langForFilename('main.TS')).toBe('typescript');
    expect(langForFilename('app.py')).toBe('python');
    expect(langForFilename('Dockerfile')).toBe('');
  });
});

describe('semantic action keys', () => {
  it('folds renamed variants, index-links identifiers, normalizes whitespace', () => {
    expect(shapeKeyOf('typescript', ['loadUserData', '(', 'user', ')', ';'])).toBe(shapeKeyOf('typescript', ['loadAccountData', '(', 'account', ')', ';']));
    expect(shapeKeyOf('typescript', ['x', '+', 'x'])).toBe(shapeKeyOf('typescript', ['y', '+', 'y']));
    expect(shapeKeyOf('typescript', ['x', '+', 'y'])).not.toBe(shapeKeyOf('typescript', ['x', '+', 'x']));
    expect(shapeKeyOf('typescript', ['a', '  ', '+', ' ', 'b'])).toBe(shapeKeyOf('typescript', ['a', ' ', '+', 'b']));
    expect(shapeKeyOf('typescript', ['a', '\n', '  ', 'b'])).not.toBe(shapeKeyOf('typescript', ['a', ' ', 'b']));
  });
});

describe('markov model', () => {
  it('KN continuation counts favor broadly-continued tokens', () => {
    const m = new MarkovModel(2);
    for (let i = 0; i < 8; i++) m.learn('typescript', ['x', 'a']);
    m.learn('typescript', ['y', 'a']);
    m.learn('typescript', ['p', 'b']); m.learn('typescript', ['q', 'b']); m.learn('typescript', ['r', 'b']);
    const dist = m.smoothed('typescript', [';'], [';'], 6, 1, 0.75);
    expect(dist).toBeDefined();
    expect(dist!.get('b')!).toBeGreaterThan(dist!.get('a')!);
  });
  it('family channel serves sibling languages', () => {
    const m = new MarkovModel(3);
    m.learn('typescript', ['fetch', 'Data'], 1, 0, 1);
    const fam = m.smoothed([{ lang: 'c-like', weight: 1 }], ['fetch'], ['#I'], 6, 1);
    expect(fam?.get('Data')).toBeGreaterThan(0);
    const go = m.smoothed([{ lang: 'go', weight: 1 }, { lang: 'c-like', weight: 1 }, { lang: '*', weight: 1 }], ['fetch'], ['#I'], 6, 1);
    expect(go?.get('Data')).toBeGreaterThan(0);
  });
  it('universal channel learned via 4th arg', () => {
    const m = new MarkovModel(2);
    m.learn('typescript', ['fetch', 'Data'], 1, 1);
    expect(m.smoothed('*', ['fetch'], ['#I'], 6, 1)?.get('Data')).toBeGreaterThan(0);
  });
  it('dropLanguage keeps family and universal knowledge', () => {
    const m = new MarkovModel(2);
    m.learn('typescript', ['a', ' ', 'b'], 1, 1, 1);
    m.learn('python', ['a', ' ', 'c'], 1, 1, 1);
    m.dropLanguage('typescript');
    expect(m.bigramsFor('typescript')).toBe(0);
    expect(m.bigramsFor('python')).toBeGreaterThan(0);
    expect(m.bigramsFor('c-like')).toBeGreaterThan(0);
    expect(m.bigramsFor('*')).toBeGreaterThan(0);
  });
  it('decay forgets weak counts', () => {
    const m = new MarkovModel(2);
    m.learn('typescript', ['a', ' ', 'b']);
    m.decay(0.1);
    expect(m.smoothed('typescript', ['a'], ['a'], 6, 1)).toBeUndefined();
  });
  it('worker snapshot round-trips', () => {
    const m = new MarkovModel(2);
    m.learn('typescript', ['a', ' ', 'b'], 1, 0.2, 0.3);
    const h = MarkovModel.hydrateFields(m.exportFields());
    expect(h.bigramsFor('typescript')).toBe(m.bigramsFor('typescript'));
    expect(h.bigramsFor('*')).toBe(m.bigramsFor('*'));
  });
  it('exportChannels keeps only family/universal data', () => {
    const m = new MarkovModel(2);
    m.learn('typescript', ['a', ' ', 'b'], 1, 0.2, 0.3);
    const h = MarkovModel.deserialize(m.exportChannels((c) => c === '*' || c === 'c-like' || c === 'scripting'))!;
    expect(h.bigramsFor('typescript')).toBe(0);
    expect(h.bigramsFor('c-like')).toBeGreaterThan(0);
    expect(h.bigramsFor('*')).toBeGreaterThan(0);
  });
});

describe('learner: values', () => {
  it('TD(lambda) traces propagate reward back', () => {
    const q = new QLearner(P);
    const a1 = { key: 'a', prior: 0, features: [] };
    const a2 = { key: 'b', prior: 0, features: [] };
    q.update('s1', a1, 0, 0, 1);
    q.update('s2', a2, 1, 0, 1);
    expect(q.getQ('s2', 'b')).toBeCloseTo(0.5);
    expect(q.getQ('s1', 'a')).toBeCloseTo(0.5 * 0.9 * 0.8);
  });
  it('semi-MDP bootstrap discounts by token length', () => {
    const q = new QLearner(P);
    q.update('s', { key: 'a', prior: 0, features: [] }, 1, 2, 3);
    expect(q.getQ('s', 'a')).toBeCloseTo(0.5 * (1 + 0.9 ** 3 * 2));
  });
  it('adaptive (Robbins-Monro) alpha = 1/N', () => {
    const q = new QLearner({ ...P, alphaMode: 'adaptive' });
    const a = { key: 'a', prior: 0, features: [] };
    q.update('s', a, 1, 0, 1);
    expect(q.getQ('s', 'a')).toBeCloseTo(1);
    q.update('s', a, 0, 0, 1);
    expect(q.getQ('s', 'a')).toBeCloseTo(0.5);
  });
  it('Kalman uncertainty shrinks with data', () => {
    const q = new QLearner({ ...P, alphaMode: 'kalman' });
    const a = { key: 'a', prior: 0, features: [] };
    for (let i = 0; i < 20; i++) q.update('s', a, 0.5, 0, 1);
    const s1 = q.estimate('s', a).sigma;
    for (let i = 0; i < 300; i++) q.update('s', a, 0.5, 0, 1);
    expect(q.estimate('s', a).sigma).toBeLessThan(s1);
  });
  it('low-rank factors complete values for never-tried pairs', () => {
    const q = new QLearner({ ...P, factorDim: 4, factorEta: 0.08 });
    const states = ['s1', 's2', 's3', 's4'];
    const actions = ['a1', 'a2', 'a3', 'a4'];
    const u = [0.8, -0.6, 0.5, 1.0];
    const v = [0.9, 0.7, -0.8, 0.4];
    const T = (s: number, a: number) => u[s] * v[a];
    const pairs: [number, number][] = [];
    for (let s = 0; s < 4; s++) for (let a = 0; a < 4; a++) if (!(s === 1 && a === 2)) pairs.push([s, a]);
    for (let iter = 0; iter < 500; iter++) for (const [s, a] of pairs) q.updateReplay(states[s], actions[a], T(s, a), 0, 1);
    expect(Math.abs(q.factorPredict('s2', 'a3')! - T(1, 2))).toBeLessThan(0.15);
    expect(Math.abs(q.estimate('s2', { key: 'a3', prior: 0, features: [] }).value - T(1, 2))).toBeLessThan(0.25);
  });
  it('prioritized sweeping backs value up through predecessors', () => {
    const q = new QLearner({ ...P, lambda: 0 });
    const a1 = { key: 'a1', prior: 0, features: [] };
    const a2 = { key: 'a2', prior: 0, features: [] };
    q.update('s2', a2, 0, 0, 1, 's1');
    q.update('s1', a1, 1, 0, 1, 'sT');
    expect(q.getQ('s2', 'a2')).toBe(0);
    q.sweep(5);
    expect(q.getQ('s2', 'a2')).toBeGreaterThan(0);
  });
  it('R-learning: differential values and rho', () => {
    const q = new QLearner({ ...P, valueObjective: 'average', lambda: 0 });
    const aA = { key: 'a', prior: 0, features: [] };
    const aB = { key: 'b', prior: 0, features: [] };
    for (let i = 0; i < 400; i++) {
      q.update('sA', aA, 1, q.maxQAtState('sB'), 1, 'sB');
      q.update('sB', aB, 0, q.maxQAtState('sA'), 1, 'sA');
    }
    expect(q.rho).toBeCloseTo(0.5, 1);
    expect(q.getQ('sA', 'a') - q.getQ('sB', 'b')).toBeCloseTo(0.5, 1);
  });
  it('baseline-conditioned surrogate reward (advantage learning)', () => {
    const q = new QLearner({ ...P, weightAlpha: 0.5 });
    const a = { key: 'a', prior: 0, features: [1] };
    for (let i = 0; i < 50; i++) q.update('s', a, 1, 0, 1);
    const warm = q.surrogateReward('s', { key: 'x', prior: 0, features: [1] });
    const cold = q.surrogateReward('sZ', { key: 'x', prior: 0, features: [1] });
    expect(warm).toBeGreaterThan(0.8);
    expect(warm - cold).toBeGreaterThan(0.5);
  });
});

describe('learner: exploration, distributions, OPE', () => {
  it('double-Q replay trains the shadow table', () => {
    const q = new QLearner(P);
    for (let i = 0; i < 200; i++) q.updateReplay('s', 'a', 1, 0, 1, 's2');
    expect(q.getQ('s', 'a')).toBeGreaterThan(0.9);
    expect(q.qShadow.get('s\u0001a') ?? 0).toBeGreaterThan(0.9);
  });
  it('replay never touches visit counters', () => {
    const q = new QLearner(P);
    for (let i = 0; i < 60; i++) q.updateReplay('s', 'a', 1, 0, 1);
    expect(q.getQ('s', 'a')).toBeGreaterThan(0.99);
    expect(q.visits.get('s\u0001a')).toBeUndefined();
  });
  it('choose() reports behavior propensities', () => {
    const a1 = { key: 'a1', prior: 0.1, features: [] };
    const a2 = { key: 'a2', prior: 0.9, features: [] };
    const greedy = new QLearner({ ...P, epsilon: 0 }).choose('s', [a1, a2]);
    expect(greedy.index).toBe(1);
    expect(greedy.piB).toBeCloseTo(1);
    expect(greedy.argmaxIndex).toBe(1);
    expect(new QLearner({ ...P, epsilon: 1 }).choose('s', [a1, a2]).piB).toBeCloseTo(0.5);
    const soft = new QLearner({ ...P, exploration: 'softmax', temperature: 0.5 }).choose('s', [a1, a2]);
    expect(soft.piB).toBeGreaterThan(0);
    expect(soft.piB).toBeLessThanOrEqual(1);
  });
  it('distributional values: P(win) and CVaR', () => {
    const q = new QLearner(P);
    const steady = { key: 'steady', prior: 0, features: [] };
    const risky = { key: 'risky', prior: 0, features: [] };
    for (let i = 0; i < 40; i++) {
      q.update('s', steady, 0.25, 0, 1);
      q.update('s', risky, i % 5 < 2 ? -0.4 : 1.4, 0, 1);
    }
    expect(q.pWinOf('s', 'steady')).toBeGreaterThan(0.9);
    expect(q.pWinOf('s', 'risky')).toBeGreaterThan(0.2);
    expect(q.pWinOf('s', 'risky')).toBeLessThan(0.75);
    expect(q.cvarOf('s', 'risky', 0.5)).toBeLessThan(-0.1);
  });
  it('rankBy pwin prefers reliability over higher mean value', () => {
    const build = (rankBy?: string) => {
      const q = new QLearner({ ...P, rankBy: rankBy as any });
      const steady = { key: 'steady', prior: 0, features: [] };
      const risky = { key: 'risky', prior: 0, features: [] };
      for (let i = 0; i < 40; i++) {
        q.update('s', steady, 0.25, 0, 1);
        q.update('s', risky, i % 5 < 2 ? -0.4 : 1.4, 0, 1);
      }
      return { q, steady, risky };
    };
    const byValue = build('value');
    expect(byValue.q.choose('s', [byValue.steady, byValue.risky]).index).toBe(1); // higher value wins
    const byPwin = build('pwin');
    expect(byPwin.q.choose('s', [byPwin.steady, byPwin.risky]).index).toBe(0);    // reliability wins
  });
  it('risk-aware ranking prefers the well-estimated action', () => {
    const a1 = { key: 'a1', prior: 0, features: [] };
    const a2 = { key: 'a2', prior: 0, features: [] };
    const neutral = new QLearner({ ...P, epsilon: 0 });
    const risky = new QLearner({ ...P, epsilon: 0, riskKappa: 6 });
    for (const q of [neutral, risky]) {
      q.update('s', a1, 2, 0, 1);
      q.update('s', a1, 2, 0, 1);
      for (let i = 0; i < 150; i++) q.update('s', a2, 0.8, 0, 1);
    }
    expect(neutral.choose('s', [a1, a2]).index).toBe(0);
    expect(risky.choose('s', [a1, a2]).index).toBe(1);
  });
  it('exploit lock suspends exploration; a drift alarm releases it', () => {
    const q = new QLearner({ ...P, epsilon: 1, driftThreshold: 0.3 });
    const a1 = { key: 'a1', prior: 0.1, features: [] };
    const a2 = { key: 'a2', prior: 0.9, features: [] };
    q.exploitLock = true;
    for (let i = 0; i < 20; i++) expect(q.choose('s', [a1, a2]).index).toBe(1);
    for (let i = 0; i < 60; i++) q.update('s', a2, 1, 0, 1);
    expect(q.stats.driftAlarms).toBeGreaterThanOrEqual(1);
    expect(q.exploitLock).toBe(false);
  });
});

describe('learner: shape keys, migration, taste, conditioning', () => {
  it('hybrid literal bonus: shared shape Q + per-literal residual', () => {
    const q = new QLearner(P);
    const shape = shapeKeyOf('typescript', ['foo', '(', ')', ';']);
    const foo = { key: shape, prior: 0, features: [], lang: 'typescript', literalKey: 'typescript\u0003foo\u0000(\u0000)\u0000;' };
    const bar = { ...foo, literalKey: 'typescript\u0003bar\u0000(\u0000)\u0000;' };
    for (let i = 0; i < 10; i++) q.update('s', foo, 1, 0, 1);
    for (let i = 0; i < 10; i++) q.update('s', bar, -0.2, 0, 1);
    expect(q.q.size).toBe(1);
    expect(q.estimate('s', foo).value).toBeGreaterThan(q.estimate('s', bar).value);
    const e = q.explain('s', foo);
    expect(e.bonus).toBeGreaterThan(0);
    expect(e.value).toBeCloseTo(e.tabular + e.head + e.bonus, 10);
  });
  it('literalBonus disabled when off', () => {
    const q = new QLearner({ ...P, literalBonus: false });
    const shape = shapeKeyOf('typescript', ['foo', '(', ')', ';']);
    const foo = { key: shape, prior: 0, features: [], literalKey: 'typescript\u0003foo\u0000(\u0000)\u0000;' };
    for (let i = 0; i < 5; i++) q.update('s', foo, 1, 0, 1);
    expect(q.explain('s', foo).bonus).toBe(0);
    expect(q.literalBonus.size).toBe(0);
  });
  it('migrateToShape re-keys and merges collisions', () => {
    const q = new QLearner(P);
    q.update('s', { key: 'typescript\u0003foo\u0000(\u0000)\u0000;', prior: 0, features: [] }, 1, 0, 1);
    q.update('s', { key: 'typescript\u0003bar\u0000(\u0000)\u0000;', prior: 0, features: [] }, 1, 0, 1);
    expect(q.q.size).toBe(2);
    const merges = q.migrateToShape();
    expect(q.q.size).toBe(1);
    expect(merges).toBeGreaterThanOrEqual(1);
    expect(q.getQ('s', shapeKeyOf('typescript', ['foo', '(', ')', ';']))).toBeGreaterThan(0);
  });
  it('exportTaste keeps only pure-syntax actions on global states', () => {
    const q = new QLearner(P);
    q.update('s', { key: shapeKeyOf('typescript', ['return', ' ', ';']), prior: 0, features: [] }, 1, 0, 1);
    q.update('s', { key: shapeKeyOf('typescript', ['foo', '(', ')', ';']), prior: 0, features: [] }, 1, 0, 1);
    q.update('s', { key: shapeKeyOf('typescript', ['return', ' ', '42', ';']), prior: 0, features: [] }, 1, 0, 1);
    const safe = (actionKey: string): boolean => {
      const i = actionKey.indexOf('\u0003');
      if (i < 0) return false;
      const lang = actionKey.slice(0, i);
      return actionKey.slice(i + 1).split('\u0000').every((t) => {
        if (t === '\n' || t === '\r\n' || t === ' ') return true;
        const a = abstractToken(t, lang);
        return a !== '#I' && a !== '#N' && a !== '#S' && a !== '#C';
      });
    };
    const taste = q.exportTaste(safe);
    expect(taste.q.length).toBe(1);
    expect(taste.q[0][0]).toContain('return');
  });
  it('mergeTaste shifts values partway (reduced confidence)', () => {
    const b = new QLearner(P);
    const safeKey = shapeKeyOf('typescript', ['return', ' ', ';']);
    for (let i = 0; i < 10; i++) b.update('s', { key: safeKey, prior: 0, features: [] }, 1, 0, 1);
    const a = new QLearner(P);
    a.update('s', { key: safeKey, prior: 0, features: [] }, 0, 0, 1);
    const safe = (): boolean => true;
    a.mergeTaste(b.exportTaste(safe), 0.5);
    const v = a.getQ('s', safeKey);
    expect(v).toBeGreaterThan(0);
    expect(v).toBeLessThan(b.getQ('s', safeKey));
  });
  it('workspace-conditioned read-through borrows global values with inflated sigma', () => {
    const q = new QLearner(P);
    const a = { key: 'a', prior: 0, features: [] };
    for (let i = 0; i < 10; i++) q.update('global', a, 1, 0, 1);
    const through = q.estimate('ws1', a, 'global');
    expect(through.value).toBeGreaterThan(0.8);
    expect(through.visits).toBe(0);
    expect(q.estimate('ws1', a).value).toBe(0);
    expect(q.explain('ws1', a, 'global').fallback).toBe(true);
    const b = { key: 'b', prior: 0, features: [] };
    expect(q.choose('ws1', [b, a], 'global').index).toBe(1);
  });
  it('fittedQ (offline method) converges to the bootstrap fixed point', () => {
    const q = new QLearner({ ...P, gamma: 0.9 });
    q.fittedQ([
      { stateKey: 's1', actionKey: 'a', reward: 0, nextStateKey: 's2', steps: 1 },
      { stateKey: 's2', actionKey: 'b', reward: 1, nextStateKey: 'sT', steps: 1 },
    ], { epochs: 20, lr: 1 });
    expect(q.getQ('s2', 'b')).toBeCloseTo(1, 5);
    expect(q.getQ('s1', 'a')).toBeCloseTo(0.9, 2);
    expect(q.stats.fittedQ).toBe(40);
  });
});

describe('generation core', () => {
  const params = (lang: string, litBefore: string[], absBefore: string[]): TokenGenParams => ({
    lang, litBefore, absBefore, prefix: '', atEol: true, maxLines: 1, minCount: 1, minConfidence: 0.05,
    beamWidth: 4, candidates: 2, effMax: 12, suggestOnNewline: true, rolloutSteps: 0,
    diversityLambda: 0, maxTokensTotal: 12, cacheBoost: null,
  });
  it('python terminates at newline; c-like at semicolon', () => {
    const m = new MarkovModel(2);
    m.learn('python', ['a', ' ', 'bb', '\n', '  ', 'c']);
    const out = generateTokens(new DistResolver(m, CFG, new Map()), params('python', ['a', ' '], ['#I', ' ']));
    expect(out[0].tokens).toEqual(['bb']);
    const m2 = new MarkovModel(2);
    m2.learn('typescript', ['a', ' ', 'b', ';']);
    const out2 = generateTokens(new DistResolver(m2, CFG, new Map()), params('typescript', ['a', ' '], ['#I', ' ']));
    expect(out2[0].tokens).toEqual(['b', ';']);
  });
  it('per-document cache boost mixes in', () => {
    const m = new MarkovModel(2);
    m.learn('typescript', ['a', ' ', 'b']);
    const d = new DistResolver(m, CFG, new Map()).distAt('typescript', ['a', ' '], ['#I', ' '], new Map([['zzz', 8]]));
    expect(d?.get('zzz')).toBeGreaterThan(0.3);
  });
  it('mixDistributions blends by adaptive weight', () => {
    const base = new Map([['a', 0.9], ['b', 0.1]]);
    const mixed = mixDistributions(base, new Map([['zzz', 8]]), 8);
    expect(mixed.get('zzz')).toBeCloseTo(0.5);
    expect(mixed.get('a')).toBeCloseTo(0.45);
  });
  it('state keys append suffixes only when set', () => {
    expect(qStateKeyOf('ts', ['x'], 3, 0, true)).not.toContain('\u0004');
    const k = qStateKeyOf('ts', ['x'], 3, 0, true, { chain: true, mode: 'R', region: 'C', ws: 'ab12' });
    expect(k.endsWith('EARC\u0004ab12')).toBe(true);
  });
});

describe('outline regions', () => {
  const doc = (lines: string[]) => ({ lineCount: lines.length, lineAt: (i: number) => ({ text: lines[i] ?? '' }) });
  it('classifies class body, function body, signature, imports, top level', () => {
    expect(regionAt(doc(['class Foo {', '  x = 1;']), 1, 5, 'typescript').tag).toBe('C');
    expect(regionAt(doc(['function f(a) {', '  return a;']), 1, 4, 'typescript').tag).toBe('F');
    expect(regionAt(doc(['function foo(', '  a, b']), 1, 2, 'typescript').tag).toBe('S');
    expect(regionAt(doc(['import a from "b";', 'import c from "d";']), 1, 7, 'typescript').tag).toBe('I');
    expect(regionAt(doc(['let a = 1;', 'let b = 2;']), 1, 5, 'typescript').tag).toBe('');
  });
  it('ignores braces inside strings', () => {
    expect(regionAt(doc(['const s = "}";', 'let x = 1;']), 1, 5, 'typescript').tag).toBe('');
  });
});

describe('PUCT lookahead', () => {
  const distFn = (ctx: string[]): Map<string, number> | undefined => {
    if (ctx.length === 0) return new Map([['a', 0.7], ['b', 0.3]]);
    if (ctx[ctx.length - 1] === 'a' || ctx[ctx.length - 1] === 'b') return new Map([[';', 1]]);
    return undefined;
  };
  const term = (t: string[]): boolean => t[t.length - 1] === ';';
  it('follows the value function over the model prior', () => {
    const out = lookaheadSearch({ distFn, valueFn: (t) => (t[0] === 'b' ? 1 : 0), isAcceptable: () => true, isTerminal: term, budget: 20, maxDepth: 4, cPuct: 1, topK: 3 });
    expect(out!.join('')).toBe('b;');
  });
  it('falls back to the prior when values are flat; respects filters; null on empty', () => {
    const out = lookaheadSearch({ distFn, valueFn: () => 0, isAcceptable: () => true, isTerminal: term, budget: 20, maxDepth: 4, cPuct: 1, topK: 3 });
    expect(out![0]).toBe('a');
    const filtered = lookaheadSearch({ distFn, valueFn: () => 0, isAcceptable: (t) => t === 'b', isTerminal: term, budget: 5, maxDepth: 4, cPuct: 1, topK: 3 });
    expect(filtered![0]).toBe('b');
    expect(lookaheadSearch({ distFn: () => undefined, valueFn: () => 0, isAcceptable: () => true, isTerminal: term, budget: 5, maxDepth: 4, cPuct: 1, topK: 3 })).toBeNull();
  });
});

describe('dataset learning', () => {
  it('parses JSONL with flexible field names and languages', () => {
    const items = parseJsonlDataset('{"text":"let a = 1","lang":"ts"}\n{"content":"def f(): pass","language":"python"}\nnot json\n\n', 'javascript');
    expect(items).toEqual([
      { lang: 'typescript', text: 'let a = 1' },
      { lang: 'python', text: 'def f(): pass' },
    ]);
  });
  it('maps markdown fences and flags language-less datasets', () => {
    expect(parseMarkdownFences('# t\n\n```py\nx = 1\n```\n\n```js\nlet a\n```\n', 'plaintext').map((i) => i.lang)).toEqual(['python', 'javascript']);
    expect(datasetNeedsLanguage('a.md', '```\ncode\n```')).toBe(true);
    expect(datasetNeedsLanguage('a.md', '```python\nx = 1\n```')).toBe(false);
    expect(datasetNeedsLanguage('a.jsonl', '{"text":"x","lang":"go"}')).toBe(false);
  });
  it('deterministic, disjoint, complete split', () => {
    const a = splitTrainVal([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.2, 7);
    const b = splitTrainVal([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.2, 7);
    expect(a).toEqual(b);
    expect(new Set([...a.train, ...a.val]).size).toBe(10);
    expect(a.val.length).toBe(2);
  });
  it('evaluates a deterministic model at top-1 / PP 1, counts OOV, caps budget', () => {
    const m = new MarkovModel(2);
    m.learn('typescript', ['a', ' ', 'b', ';']);
    const r = evaluateSequences(new DistResolver(m, CFG, new Map()), 'typescript', [['a', ' ', 'b', ';']], 100);
    expect(r.predictions).toBe(3);
    expect(r.top1).toBe(1);
    expect(r.perplexity).toBeCloseTo(1);
    const oov = evaluateSequences(new DistResolver(m, CFG, new Map()), 'typescript', [['a', ' ', 'zzz']], 100);
    expect(oov.oovRate).toBeGreaterThan(0);
    expect(evaluateSequences(new DistResolver(m, CFG, new Map()), 'typescript', [['a', ' ', 'zzz']], 1).predictions).toBe(1);
  });
  it('auto-tune detects when a higher order resolves ambiguity', () => {
    // 6 tokens of history distinguish 'a' from 'b'; 5 or fewer cannot.
    const pattern = 'a y w b y w '.repeat(6);
    const items = [{ lang: 'typescript', text: pattern }];
    const current = new MarkovModel(3);
    current.learn('typescript', ['q', ' ', 'q']);
    const outcome = gridSearchTune({
      items, currentModel: current, currentOrder: 3, channelW: new Map(),
      baseCfg: CFG, orders: [4, 5, 6], smoothingDs: [1], knDiscounts: [0],
      tokenBudget: 5000, evalBudget: 500,
    });
    const o5 = outcome.results.find((r) => r.order === 5 && r.tempModel)!;
    const o6 = outcome.results.find((r) => r.order === 6 && r.tempModel)!;
    expect(o6.perplexity).toBeLessThan(o5.perplexity);
    expect(outcome.best.order).toBe(6);
    expect(outcome.best.tempModel).toBe(true);
    expect(outcome.current.perplexity).toBeGreaterThan(outcome.best.perplexity);
  });
});
