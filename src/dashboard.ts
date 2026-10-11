/** Dashboard webview: stat cards, acceptance curve, per-language table,
 *  top snippets with pin/ban, banned list, preference weights. */

export const FEATURE_NAMES = ['bias', 'logPrior', 'len', 'multiLine', 'reindent', 'endsStmt', 'wordCmpl', 'atEol', 'indent', 'kwStart', 'rollout', 'lookahead', 'inClass', 'inFunc', 'inImport'];

export interface DashboardData {
  stats: {
    bigrams: number; states: number; q: number; qShadow: number;
    accepts: number; dismissals: number; partials: number; chains: number;
    gated: number; gatedProbes: number; explorations: number; escapes: number;
    overridden: number; undos: number; corrections: number; confirmations: number;
    replays: number; dyna: number; sweeps: number; factorTrainings: number;
    driftAlarms: number; driftActive: boolean; opeChecks: number; exploitLocks: number; exploitLock: boolean;
    files: number; languages: number; keystrokesSaved: number; predAccuracy: number | null;
    rho: number | null; objective: string; rankBy: string; effMaxTokens: number;
    sessionMode: string; gateOffset: number;
    workerGen: number; localGen: number; fittedQ: number; ws: string | null; workspaceConditioning: boolean;
    probeRate: number | null; gateRelaxations: number; shapeMerges: number; actionKeyMode: string;
    thumbUps: number; thumbDowns: number; pinned: number; banned: number; lookahead: boolean;
    corpusDocs: number; corpusTokens: number;
    opeGreedy: number | null; opeOnPolicy: number | null; opeN: number;
    weights: number[];
  };
  buckets: { rate: number }[];
  langs: { lang: string; family: string; bigrams: number; accepts: number; dismissals: number; partials: number; accuracy: number | null; transfer: [number, number, number] }[];
  top: { key: string; text: string; lang: string; accepts: number; dismissals: number; pinned: boolean }[];
  banned: { key: string; text: string }[];
  order: number;
  exploration: string;
  objective: string;
}

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c] as string));
}

export function dashboardHtml(d: DashboardData): string {
  const s = d.stats;
  const total = s.accepts + s.dismissals + s.partials + s.escapes;
  const rate = total > 0 ? s.accepts / total : 0;
  const n = Math.max(1, d.buckets.length - 1);
  const pts = d.buckets.map((b, i) => `${((i / n) * 520 + 20).toFixed(1)},${(104 - b.rate * 88).toFixed(1)}`).join(' ');

  const cards: [string, string][] = [
    ['Bigrams', s.bigrams.toLocaleString()],
    ['Context states', s.states.toLocaleString()],
    ['Q-entries', s.q.toLocaleString()],
    ['Shadow Q', s.qShadow.toLocaleString()],
    ['Accepts', s.accepts.toLocaleString()],
    ['Dismissals', s.dismissals.toLocaleString()],
    ['Partials', s.partials.toLocaleString()],
    ['Esc dismissals', s.escapes.toLocaleString()],
    ['Undo reversals', s.undos.toLocaleString()],
    ['Corrections', s.corrections.toLocaleString()],
    ['Confirmations', s.confirmations.toLocaleString()],
    ['Overridden', s.overridden.toLocaleString()],
    ['Tab-chains', s.chains.toLocaleString()],
    ['Gated (silent)', `${s.gated.toLocaleString()} (+${s.gatedProbes} probes)`],
    ['Gate probe accept', s.probeRate === null ? '—' : `${Math.round(s.probeRate * 100)}%`],
    ['Gate relaxations', s.gateRelaxations.toLocaleString()],
    ['Explorations', s.explorations.toLocaleString()],
    ['Replays', s.replays.toLocaleString()],
    ['Dyna updates', s.dyna.toLocaleString()],
    ['Sweeps', s.sweeps.toLocaleString()],
    ['Fitted-Q updates', s.fittedQ.toLocaleString()],
    ['Factor trainings', s.factorTrainings.toLocaleString()],
    ['Drift alarms', s.driftAlarms.toLocaleString()],
    ['OPE greedy vs current', s.opeGreedy === null ? '—' : `${s.opeGreedy >= s.opeOnPolicy! ? '+' : ''}${(s.opeGreedy - s.opeOnPolicy!).toFixed(2)} (n=${s.opeN})`],
    ['Exploit lock', s.exploitLock ? 'ON' : 'off'],
    ['Session mode', s.sessionMode === 'C' ? 'composing' : s.sessionMode === 'M' ? 'mixed' : 'refactoring'],
    ['Gate offset', `${s.gateOffset >= 0 ? '+' : ''}${s.gateOffset.toFixed(2)}`],
    ['Thumbs up/down', `${s.thumbUps} / ${s.thumbDowns}`],
    ['Pinned', s.pinned.toLocaleString()],
    ['Banned', s.banned.toLocaleString()],
    ['Lookahead', s.lookahead ? 'on' : 'off'],
    ['Shape merges', s.shapeMerges.toLocaleString()],
    ['Action keys', s.actionKeyMode],
    ['Gen worker/local', `${s.workerGen} / ${s.localGen}`],
    ['Workspace', s.ws ?? 'global'],
    ['Languages', s.languages.toLocaleString()],
    ['Corpus docs', s.corpusDocs.toLocaleString()],
    ['Corpus tokens', s.corpusTokens.toLocaleString()],
    ['Keystrokes saved', s.keystrokesSaved.toLocaleString()],
    ['Next-token accuracy', s.predAccuracy === null ? '—' : `${Math.round(s.predAccuracy * 100)}%`],
    ['Avg reward rho', s.rho === null ? '—' : s.rho.toFixed(2)],
    ['Accept rate', `${Math.round(rate * 100)}%`],
  ];
  const cardHtml = cards.map(([k, v]) => `<div class="card"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');

  const langRows = d.langs.length
    ? d.langs.map((l) => `<tr><td>${esc(l.lang)}</td><td>${esc(l.family)}</td><td>${l.bigrams.toLocaleString()}</td><td>${l.accepts}</td><td>${l.dismissals}</td><td>${l.accuracy === null ? '—' : `${Math.round(l.accuracy * 100)}%`}</td><td>${Math.round(l.transfer[0] * 100)}/${Math.round(l.transfer[1] * 100)}/${Math.round(l.transfer[2] * 100)}%</td></tr>`).join('')
    : '<tr><td colspan="7" class="muted">No languages yet.</td></tr>';

  const rows = d.top.length
    ? d.top.map((t, i) => `<tr><td><code>${esc(t.text).replace(/\n/g, ' ⏎ ')}</code>${t.pinned ? ' 📌' : ''}</td><td>${esc(t.lang)}</td><td>${t.accepts}</td><td>${t.dismissals}</td><td><button class="mini" onclick="act('${t.pinned ? 'unpin' : 'pin'}', ${i})">${t.pinned ? 'Unpin' : 'Pin'}</button> <button class="mini" onclick="act('ban', ${i})">Ban</button></td></tr>`).join('')
    : '<tr><td colspan="5" class="muted">No accepted snippets yet — keep Tab-ing.</td></tr>';

  const bannedRows = d.banned.length
    ? d.banned.map((b, i) => `<tr><td><code>${esc(b.text).slice(0, 50)}</code></td><td><button class="mini" onclick="act('unban', ${i})">Unban</button></td></tr>`).join('')
    : '<tr><td colspan="2" class="muted">Nothing banned.</td></tr>';

  const weights = s.weights.length
    ? s.weights.map((w, i) => `<span class="w" title="${FEATURE_NAMES[i] ?? `phi${i}`}">${FEATURE_NAMES[i] ?? `phi${i}`}: ${w.toFixed(2)}</span>`).join('')
    : '<span class="muted">weights appear after the first resolved suggestion</span>';

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  body { font-family: var(--vscode-font-family); padding: 16px; color: var(--vscode-foreground); }
  h2 { margin: 0 0 12px; font-size: 15px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 8px; margin-bottom: 18px; }
  .card { background: var(--vscode-editor-background); border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 8px; }
  .card .v { font-size: 18px; font-weight: 600; }
  .card .k { font-size: 11px; opacity: 0.75; }
  svg { background: var(--vscode-editor-background); border: 1px solid var(--vscode-panel-border); border-radius: 6px; }
  table { border-collapse: collapse; width: 100%; margin-top: 6px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--vscode-panel-border); font-size: 12px; }
  code { font-family: var(--vscode-editor-font-family); }
  .muted { opacity: 0.6; }
  .w { display: inline-block; margin: 2px 6px 2px 0; padding: 2px 6px; border: 1px solid var(--vscode-panel-border); border-radius: 10px; font-size: 11px; }
  .mini { font-size: 11px; padding: 1px 6px; }
  button { margin: 10px 0; }
  .meta { font-size: 11px; opacity: 0.7; margin-top: 14px; }
</style></head><body>
  <h2>MDP Snippets — agent dashboard</h2>
  <div class="cards">${cardHtml}</div>
  <h2>Acceptance rate (recent suggestions → older)</h2>
  <svg width="560" height="120" viewBox="0 0 560 120">
    <line x1="20" y1="60" x2="540" y2="60" stroke="currentColor" stroke-opacity="0.15" stroke-dasharray="4 4"/>
    <polyline points="${pts}" fill="none" stroke="var(--vscode-charts-lines, #4fc1ff)" stroke-width="2"/>
  </svg>
  <h2 style="margin-top:18px">Per-language breakdown</h2>
  <table><tr><th>Language</th><th>Family</th><th>Bigrams</th><th>Accepts</th><th>Dismissed</th><th>Next-token acc.</th><th>Transfer self/family/*</th></tr>${langRows}</table>
  <h2 style="margin-top:18px">Top accepted snippets</h2>
  <table><tr><th>Snippet</th><th>Lang</th><th>Accepts</th><th>Dismissals</th><th>Curate</th></tr>${rows}</table>
  <h2 style="margin-top:18px">Banned snippets</h2>
  <table>${bannedRows}</table>
  <h2 style="margin-top:18px">Learned preference weights (linear head)</h2>
  <div>${weights}</div>
  <button onclick="notify()">Refresh</button>
  <div class="meta">Markov order ${d.order} · exploration: ${d.exploration} · objective: ${d.objective} · ranking: ${s.rankBy} · gate offset ${s.gateOffset.toFixed(2)} · session ${s.sessionMode} · curve buckets = accept+partial share per 25 resolved suggestions</div>
  <script>const vscode = acquireVsCodeApi(); function notify(){ vscode.postMessage('refresh'); } function act(cmd, i){ vscode.postMessage({ cmd, i }); }</script>
</body></html>`;
}
