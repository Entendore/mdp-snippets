/** Generation worker: holds a snapshot of the Markov chain + transfer weights
 *  (never the learner — ranking stays live on the main thread). */
import { parentPort } from 'worker_threads';
import { MarkovModel } from './markov';
import { DistResolver, generateTokens, paramsFromRequest } from './generate';

let resolver: DistResolver | null = null;

parentPort?.on('message', (m: any) => {
  try {
    if (m?.type === 'model') {
      const markov = MarkovModel.hydrateFields(m.markov);
      const channelW: Map<string, number[]> = new Map(m.channelW ?? []);
      resolver = new DistResolver(markov, m.resolverCfg ?? {}, channelW);
      parentPort?.postMessage({ type: 'ready' });
    } else if (m?.type === 'generate') {
      if (!resolver) { parentPort?.postMessage({ type: 'candidates', id: m.id, candidates: [] }); return; }
      const cands = generateTokens(resolver, paramsFromRequest(m.req));
      parentPort?.postMessage({ type: 'candidates', id: m.id, candidates: cands });
    }
  } catch (err) {
    if (m && m.id !== undefined) parentPort?.postMessage({ type: 'error', id: m.id, message: String(err) });
  }
});
