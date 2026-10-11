/** Shallow PUCT lookahead over token continuations. `distFn` is the
 *  transition/prior model, `valueFn` the learned value of the snippet-so-far,
 *  `isTerminal` the option-termination predicate. Pure module. */
import type { Dist } from './markov';

export interface LookaheadOptions {
  distFn: (ctxTokens: string[]) => Dist | undefined;
  valueFn: (tokens: string[]) => number;
  isAcceptable: (token: string) => boolean;
  isTerminal: (tokens: string[]) => boolean;
  budget: number;
  maxDepth: number;
  cPuct: number;
  topK: number;
}

interface Node {
  token: string;
  prior: number;
  pv: number;
  n: number;
  w: number;
  children: Map<string, Node>;
  done: boolean;
}

export function lookaheadSearch(o: LookaheadOptions): string[] | null {
  const root: Node = { token: '', prior: 1, pv: 0, n: 0, w: 0, children: new Map(), done: false };

  for (let iter = 0; iter < o.budget; iter++) {
    const path: Node[] = [root];
    const seq: string[] = [];
    let node = root;
    while (node.children.size > 0 && !node.done && seq.length < o.maxDepth) {
      let best: Node | null = null;
      let bestScore = -Infinity;
      for (const ch of node.children.values()) {
        const q = ch.n > 0 ? ch.w / ch.n : ch.pv;
        const score = q + (o.cPuct * ch.prior * Math.sqrt(node.n + 1)) / (1 + ch.n);
        if (score > bestScore) { bestScore = score; best = ch; }
      }
      if (!best) break;
      node = best;
      path.push(node);
      seq.push(node.token);
    }

    let value = 0;
    if (seq.length > 0 && (node.done || seq.length >= o.maxDepth || o.isTerminal(seq))) {
      node.done = true;
      value = o.valueFn(seq);
    } else {
      const dist = o.distFn(seq);
      const cands = dist
        ? Array.from(dist.entries())
            .filter(([t]) => seq.length > 0 || o.isAcceptable(t))
            .sort((a, b) => b[1] - a[1])
            .slice(0, o.topK)
        : [];
      if (cands.length === 0) {
        node.done = true;
        value = seq.length > 0 ? o.valueFn(seq) : 0;
      } else {
        value = seq.length > 0 ? o.valueFn(seq) : 0;
        for (const [tok, p] of cands) {
          const childSeq = seq.concat([tok]);
          node.children.set(tok, {
            token: tok,
            prior: p,
            pv: o.valueFn(childSeq),
            n: 0,
            w: 0,
            children: new Map(),
            done: o.isTerminal(childSeq) || childSeq.length >= o.maxDepth,
          });
        }
      }
    }

    for (const nd of path) { nd.n++; nd.w += value; }
  }

  const out: string[] = [];
  let node = root;
  while (node.children.size > 0 && out.length < o.maxDepth) {
    let best: Node | null = null;
    for (const ch of node.children.values()) {
      if (!best || ch.n > best.n || (ch.n === best.n && ch.w / Math.max(1, ch.n) > best.w / Math.max(1, best.n))) best = ch;
    }
    if (!best || best.n === 0) break;
    node = best;
    out.push(node.token);
    if (node.done || o.isTerminal(out)) break;
  }
  while (out.length > 1 && /^[ \t\r\n]+$/.test(out[out.length - 1])) out.pop();
  return out.length > 0 ? out : null;
}
