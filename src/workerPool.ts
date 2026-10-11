/** Single-worker pool: lazy spawn, debounced model pushes, per-request
 *  timeout, permanent fallback to sync generation on failure. */
import { Worker } from 'worker_threads';
import * as fs from 'fs';
import * as path from 'path';

export interface GenerationPoolOptions {
  timeoutMs: number;
  refreshMs: number;
}

type ModelProvider = () => { markov: any; channelW: [string, number[]][]; resolverCfg: any } | null;

export class GenerationPool {
  private worker: Worker | null = null;
  private failed = false;
  private ready = false;
  private seq = 0;
  private pending = new Map<number, { resolve: (c: any[] | null) => void; timer: ReturnType<typeof setTimeout> }>();
  private provider: ModelProvider | null = null;
  private lastPush = 0;
  private dirty = true;

  constructor(private readonly opts: GenerationPoolOptions) {}

  bind(provider: ModelProvider): void { this.provider = provider; }

  get alive(): boolean { return !this.failed; }

  private ensureWorker(): Worker | null {
    if (this.failed) return null;
    if (this.worker) return this.worker;
    const file = path.join(__dirname, 'worker.js');
    if (!fs.existsSync(file)) { this.failed = true; return null; }
    try {
      const w = new Worker(file);
      w.on('message', (m: any) => this.onMessage(m));
      w.on('error', () => this.kill());
      w.on('exit', (code: number) => {
        if (code !== 0) this.kill();
        else { this.worker = null; this.ready = false; }
      });
      this.worker = w;
      this.pushModel(true);
      return w;
    } catch {
      this.failed = true;
      return null;
    }
  }

  markDirty(): void { this.dirty = true; }

  pushModel(force = false): void {
    if (!this.worker || !this.provider) return;
    const now = Date.now();
    if (!force && (now - this.lastPush < this.opts.refreshMs || !this.dirty)) return;
    const snap = this.provider();
    if (!snap) return;
    this.lastPush = now;
    this.dirty = false;
    this.ready = false;
    this.worker.postMessage({ type: 'model', ...snap });
  }

  /** null => caller falls back to sync generation. */
  async generate(req: any): Promise<any[] | null> {
    const w = this.ensureWorker();
    if (!w || !this.ready) return null;
    if (this.dirty) {
      this.pushModel(true);
      if (!this.ready) return null;
    }
    const id = ++this.seq;
    return new Promise<any[] | null>((resolve) => {
      const timer = setTimeout(() => { this.pending.delete(id); resolve(null); }, this.opts.timeoutMs);
      this.pending.set(id, { resolve: (c) => { clearTimeout(timer); resolve(c); }, timer });
      w.postMessage({ type: 'generate', id, req });
    });
  }

  private onMessage(m: any): void {
    if (!m || typeof m !== 'object') return;
    if (m.type === 'ready') { this.ready = true; return; }
    if (m.type === 'candidates' || m.type === 'error') {
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      p.resolve(m.type === 'candidates' && Array.isArray(m.candidates) ? m.candidates : null);
    }
  }

  private kill(): void {
    this.failed = true;
    this.worker = null;
    this.ready = false;
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.resolve(null); }
    this.pending.clear();
  }

  dispose(): void {
    try { void this.worker?.terminate(); } catch { /* ignore */ }
    this.worker = null;
  }
}
