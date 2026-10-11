/** Extension wiring: provider, commands, events, status bar, worker pool,
 *  dashboard, dataset/corpus commands, taste profiles, learning report. */
import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { Brain } from './brain';
import { MdpInlineProvider, ACCEPT_CMD, getTextBefore } from './provider';
import { dashboardHtml } from './dashboard';
import { GenerationPool } from './workerPool';
import { KNOWN_LANGS } from './languages';
import { datasetNeedsLanguage, type TuneResult } from './dataset';

let activeBrain: Brain | null = null;
let activePool: GenerationPool | null = null;

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel('MDP Snippets');
  const brain = new Brain(context.globalStorageUri.fsPath, (msg) => output.appendLine(msg));
  activeBrain = brain;

  // ---- worker pool + offline warm start ----
  const genCfg = vscode.workspace.getConfiguration('mdpSnippets');
  const pool = new GenerationPool({
    timeoutMs: genCfg.get<number>('workerTimeout', 120),
    refreshMs: genCfg.get<number>('workerModelRefreshMs', 2500),
  });
  brain.setPool(pool);
  activePool = pool;
  context.subscriptions.push({ dispose: () => pool.dispose() });
  brain.warmStart();

  // ---- status bar ----
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 200);
  status.name = 'MDP Snippets';
  status.command = 'mdpSnippets.showStats';
  context.subscriptions.push(status);
  const updateStatus = (): void => {
    const s = brain.summary();
    status.text = `$(symbol-snippet) MDP ${s.bigrams.toLocaleString()}`;
    const acc = s.predAccuracy === null ? 'warming up' : `${Math.round(s.predAccuracy * 100)}%`;
    const ope = brain.opeSummary();
    const opeTxt = ope && ope.n > 0 ? `greedy ${ope.greedy.toFixed(2)} vs current ${ope.onPolicy.toFixed(2)} (n=${ope.n})` : 'collecting data';
    status.tooltip = new vscode.MarkdownString(
      `**MDP Snippets**\n\n- Objective: ${s.objective} · rank by: ${s.rankBy} · exploration: ${brain.cfg.exploration}${s.exploitLock ? ' **(exploit lock ON)**' : ''} · session: ${s.sessionMode === 'C' ? 'composing' : s.sessionMode === 'M' ? 'mixed' : 'refactoring'}\n` +
        `- Order: ${brain.cfg.order} · smoothing: ${brain.cfg.smoothing} · eff. cap: ${s.effMaxTokens} tokens · states: ${s.states.toLocaleString()} · Q: ${s.q.toLocaleString()} (+${s.qShadow} shadow)\n` +
        `- Accepted: ${s.accepts} · dismissed: ${s.dismissals} · Esc: ${s.escapes} · partial: ${s.partials} · undo: ${s.undos} · corrections: ${s.corrections}\n` +
        `- Gate offset: ${s.gateOffset >= 0 ? '+' : ''}${s.gateOffset.toFixed(2)} · probe accept: ${s.probeRate === null ? '—' : `${Math.round(s.probeRate * 100)}%`} · chains: ${s.chains} · overridden: ${s.overridden} · gated: ${s.gated} (probed ${s.gatedProbes})\n` +
        `- Planning: ${s.replays.toLocaleString()} replays · ${s.dyna.toLocaleString()} Dyna · ${s.sweeps.toLocaleString()} sweeps · ${s.fittedQ.toLocaleString()} fitted-Q\n` +
        `- OPE: ${opeTxt} · locks taken: ${s.exploitLocks} · drift alarms: ${s.driftAlarms}${s.driftActive ? ' **(re-exploring)**' : ''}\n` +
        `- Generation: ${s.workerGen} worker / ${s.localGen} local · action keys: ${s.actionKeyMode} (shape merges: ${s.shapeMerges.toLocaleString()}) · lookahead: ${s.lookahead ? 'on' : 'off'}\n` +
        `- Corpus: ${s.corpusDocs.toLocaleString()} docs / ${s.corpusTokens.toLocaleString()} tokens (weight ${brain.cfg.datasetWeight}) · languages modeled: ${s.languages}\n` +
        `- Next-token accuracy: ${acc} · keystrokes saved: ${s.keystrokesSaved.toLocaleString()}\n\nClick for stats.`
    );
  };

  // ---- provider + internal commands ----
  const provider = new MdpInlineProvider(brain);
  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider([{ scheme: 'file' }, { scheme: 'untitled' }], provider),
    vscode.commands.registerCommand(ACCEPT_CMD, (id: number) => {
      brain.accept(id);
      updateStatus();
    }),
    vscode.commands.registerCommand('mdpSnippets._dismissed', () => {
      if (brain.escapeDismiss()) updateStatus();
      void vscode.commands.executeCommand('setContext', 'mdpSnippets.suggestionActive', false);
      void vscode.commands.executeCommand('editor.action.inlineSuggest.hide');
    })
  );

  // ---- document events ----
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((doc) => {
      if (brain.learnDocument(doc)) updateStatus();
    }),
    vscode.workspace.onDidSaveTextDocument((doc) => brain.noteSaved(doc)),
    vscode.workspace.onDidCloseTextDocument((doc) => brain.noteClosed(doc.uri.toString()))
  );
  for (const editor of vscode.window.visibleTextEditors) brain.learnDocument(editor.document);

  interface QueuedInsert { offset: number; text: string; }
  const learnQueue = new Map<string, QueuedInsert[]>();
  const learnTimers = new Map<string, ReturnType<typeof setTimeout>>();
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((event) => {
      const doc = event.document;
      for (const change of event.contentChanges) brain.noteEditVolume(change.text.length, change.rangeLength);
      if (brain.noteDocumentChange(doc, event.contentChanges)) { updateStatus(); return; }
      if (brain.detectAcceptance(doc, event.contentChanges)) { updateStatus(); return; }
      if (!brain.shouldLearn(doc)) return;
      for (const change of event.contentChanges) {
        if (!change.text) continue;
        const key = doc.uri.toString();
        const queue = learnQueue.get(key) ?? [];
        queue.push({ offset: doc.offsetAt(change.range.start), text: change.text });
        learnQueue.set(key, queue);
        if (!learnTimers.has(key)) {
          learnTimers.set(
            key,
            setTimeout(() => {
              const items = learnQueue.get(key) ?? [];
              learnQueue.delete(key);
              learnTimers.delete(key);
              for (const item of items) brain.learnInsertedText(doc, item.offset, item.text);
              updateStatus();
            }, 500)
          );
        }
      }
    })
  );

  // ---- reward attribution: cursor moved away ----
  context.subscriptions.push(
    vscode.window.onDidChangeTextEditorSelection((event) => {
      if (!brain.hasPending) return;
      if (event.textEditor.document.version !== brain.pendingVersion) return;
      const pos = event.selections[0]?.active;
      if (!pos) return;
      brain.settle(event.textEditor.document, pos, getTextBefore(event.textEditor.document, pos));
    })
  );

  // ---- commands ----
  context.subscriptions.push(
    vscode.commands.registerCommand('mdpSnippets.showStats', async () => {
      const s = brain.summary();
      const choice = await vscode.window.showInformationMessage(
        `MDP Snippets — ${s.bigrams.toLocaleString()} bigrams · ${s.states.toLocaleString()} states · ` +
          `${s.q.toLocaleString()} Q-entries · accepted ${s.accepts} / dismissed ${s.dismissals} · files ${s.files} · corpus ${s.corpusDocs}`,
        'Learn Workspace',
        'Reset Model'
      );
      if (choice === 'Learn Workspace') void vscode.commands.executeCommand('mdpSnippets.learnFromWorkspace');
      else if (choice === 'Reset Model') void vscode.commands.executeCommand('mdpSnippets.reset');
    }),

    vscode.commands.registerCommand('mdpSnippets.toggle', async () => {
      const config = vscode.workspace.getConfiguration('mdpSnippets');
      const enabled = config.get<boolean>('enabled', true);
      await config.update('enabled', !enabled, vscode.ConfigurationTarget.Global);
      void vscode.window.showInformationMessage(`MDP Snippets: ${enabled ? 'disabled' : 'enabled'}.`);
    }),

    vscode.commands.registerCommand('mdpSnippets.reset', () => {
      brain.reset();
      updateStatus();
      void vscode.window.showInformationMessage('MDP Snippets: model and Q-table reset.');
    }),

    vscode.commands.registerCommand('mdpSnippets.learnFromActiveFile', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const learned = brain.learnDocument(editor.document, true);
      updateStatus();
      void vscode.window.showInformationMessage(
        learned ? `MDP Snippets: learned ${editor.document.fileName}.` : 'MDP Snippets: file skipped (language filter or size limit).'
      );
    }),

    vscode.commands.registerCommand('mdpSnippets.learnFromWorkspace', () => learnFromWorkspace(brain, updateStatus)),

    vscode.commands.registerCommand('mdpSnippets.exportModel', async () => {
      brain.saveNow();
      if (!fs.existsSync(brain.modelPath)) {
        void vscode.window.showWarningMessage('MDP Snippets: nothing to export yet.');
        return;
      }
      const target = await vscode.window.showSaveDialog({ defaultUri: vscode.Uri.file('mdp-model.json'), filters: { 'JSON model': ['json'] } });
      if (!target) return;
      await vscode.workspace.fs.copy(vscode.Uri.file(brain.modelPath), target, { overwrite: true });
      void vscode.window.showInformationMessage(`MDP Snippets: model exported to ${target.fsPath}.`);
    }),

    vscode.commands.registerCommand('mdpSnippets.importModel', async () => {
      const uris = await vscode.window.showOpenDialog({ filters: { 'MDP model': ['json'] }, canSelectMany: false });
      if (!uris?.[0]) return;
      try {
        const res = brain.importModel(uris[0]);
        updateStatus();
        void vscode.window.showInformationMessage(`MDP Snippets: merged model — ${res.bigrams.toLocaleString()} bigrams, ${res.qEntries.toLocaleString()} Q-entries.`);
      } catch (err) {
        void vscode.window.showErrorMessage(`MDP Snippets: import failed — ${String(err)}`);
      }
    }),

    vscode.commands.registerCommand('mdpSnippets.showDashboard', () => {
      const panel = vscode.window.createWebviewPanel('mdpDashboard', 'MDP Snippets — Dashboard', vscode.ViewColumn.Beside, { enableScripts: true });
      const render = () => { panel.webview.html = dashboardHtml(brain.dashboardData()); };
      render();
      panel.webview.onDidReceiveMessage((msg: any) => {
        if (msg === 'refresh') return render();
        if (msg && typeof msg.cmd === 'string' && typeof msg.i === 'number') {
          const key = msg.cmd === 'unban' ? brain.currentBannedKeys()[msg.i] : brain.currentTopKeys()[msg.i];
          if (!key) return;
          if (msg.cmd === 'pin') brain.pinSnippet(key);
          else if (msg.cmd === 'unpin') brain.unpinSnippet(key);
          else if (msg.cmd === 'ban') brain.banSnippet(key);
          else if (msg.cmd === 'unban') brain.unbanSnippet(key);
          render();
        }
      }, undefined, context.subscriptions);
    }),

    vscode.commands.registerCommand('mdpSnippets.resetLanguage', async () => {
      const langs = brain.knownLanguages();
      if (langs.length === 0) {
        void vscode.window.showInformationMessage('MDP Snippets: no languages learned yet.');
        return;
      }
      const pick = await vscode.window.showQuickPick(
        langs.map((l) => ({ label: l.id, description: `${l.family} · ${l.bigrams.toLocaleString()} bigrams · ${l.accepts} accepts`, id: l.id })),
        { placeHolder: 'Reset the learned model for which language?' }
      );
      if (!pick) return;
      brain.resetLanguage(pick.id);
      updateStatus();
      void vscode.window.showInformationMessage(`MDP Snippets: reset the ${pick.id} model. Family and cross-language knowledge were kept.`);
    }),

    vscode.commands.registerCommand('mdpSnippets.explainSuggestion', () => {
      const plan = brain.explainLastPlan();
      if (!plan || plan.length === 0) {
        void vscode.window.showInformationMessage('MDP Snippets: no current suggestion to explain.');
        return;
      }
      const s = brain.summary();
      output.appendLine(`\n=== Suggestion breakdown (${new Date().toLocaleTimeString()}) ===`);
      output.appendLine(`exploration: ${brain.cfg.exploration}${s.exploitLock ? ' (exploit lock ON)' : ''} · rank by: ${s.rankBy} · session: ${s.sessionMode} · gate threshold: ${(brain.cfg.minScore + s.gateOffset).toFixed(2)}`);
      plan.forEach((e, i) => {
        const mark = e.chosen ? '▶ SHOWN' : `  alt ${i + 1}`;
        const pin = e.pinned ? ' 📌' : '';
        output.appendLine(`${mark}${pin}  "${e.text.replace(/\n/g, ' ⏎ ').slice(0, 60)}"`);
        output.appendLine(
          `    score ${e.score.toFixed(3)} = value ${e.parts.value.toFixed(3)} (Q ${e.parts.tabular.toFixed(3)}` +
          `${e.parts.shadow !== null ? ` ⊕ shadow ${e.parts.shadow.toFixed(3)}` : ''}` +
          `${e.parts.factor !== null ? ` ⊕ factor ${e.parts.factor.toFixed(3)}` : ''}` +
          ` + head ${e.parts.head.toFixed(3)}` +
          `${Math.abs(e.parts.bonus) >= 0.001 ? ` + bonus ${e.parts.bonus.toFixed(3)}` : ''})` +
          ` + β·log(1+prior) ${(brain.cfg.beta * Math.log(1 + e.prior)).toFixed(3)}` +
          `  · σ ${e.parts.sigma.toFixed(2)} · visits ${e.parts.visits}` +
          `${e.parts.fallback ? ' ⟵ global prior (workspace state unvisited)' : ''}`
        );
      });
      output.show(true);
    }),

    vscode.commands.registerCommand('mdpSnippets.thumbsUp', () => { brain.thumb(true); updateStatus(); }),
    vscode.commands.registerCommand('mdpSnippets.thumbsDown', () => { brain.thumb(false); updateStatus(); }),

    vscode.commands.registerCommand('mdpSnippets.openSessionLog', () => {
      if (!fs.existsSync(brain.sessionLogPath)) {
        void vscode.window.showInformationMessage('MDP Snippets: enable mdpSnippets.sessionLog first — no log yet.');
        return;
      }
      void vscode.window.showTextDocument(vscode.Uri.file(brain.sessionLogPath), { preview: true });
    }),

    vscode.commands.registerCommand('mdpSnippets.exportTasteProfile', async () => {
      const { profile, safeActions, channels } = brain.buildTasteProfile();
      const target = await vscode.window.showSaveDialog({ defaultUri: vscode.Uri.file('mdp-taste-profile.json'), filters: { 'MDP taste profile': ['json'] } });
      if (!target) return;
      await vscode.workspace.fs.writeFile(target, Buffer.from(JSON.stringify(profile, null, 2), 'utf8'));
      void vscode.window.showInformationMessage(
        `MDP Snippets: taste profile exported — ${safeActions} pure-syntax action values, channels: ${channels.join(', ') || 'none'}. No identifiers, strings, or numbers are included.`
      );
    }),

    vscode.commands.registerCommand('mdpSnippets.importTasteProfile', async () => {
      const uris = await vscode.window.showOpenDialog({ filters: { 'MDP taste profile': ['json'] }, canSelectMany: false });
      if (!uris?.[0]) return;
      try {
        const res = brain.importTasteProfile(uris[0]);
        updateStatus();
        void vscode.window.showInformationMessage(`MDP Snippets: merged taste profile — ${res.channels} transfer channel(s), ${res.safeActions} shared action values (applied at 0.5 confidence).`);
      } catch (err) {
        void vscode.window.showErrorMessage(`MDP Snippets: taste import failed — ${String(err)}`);
      }
    }),

    vscode.commands.registerCommand('mdpSnippets.learnFromFolder', async () => {
      const dirs = await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectMany: false });
      if (!dirs?.[0]) return;
      const res = await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'MDP Snippets: learning folder', cancellable: true },
        (progress, token) => brain.learnFromFolder(dirs[0], (m) => progress.report({ message: m }), { isCancelled: () => token.isCancellationRequested })
      );
      updateStatus();
      void vscode.window.showInformationMessage(
        `MDP Snippets: learned ${res.learned} file(s) — ${res.tokens.toLocaleString()} tokens at weight ${brain.cfg.datasetWeight} (datasets teach the model, not your preferences). Skipped ${res.skipped}.`
      );
    }),

    vscode.commands.registerCommand('mdpSnippets.learnFromDataset', async () => {
      const uris = await vscode.window.showOpenDialog({ canSelectMany: false, filters: { 'Dataset': ['jsonl', 'ndjson', 'json', 'txt', 'md'] } });
      if (!uris?.[0]) return;
      const uri = uris[0];
      let defaultLang = '';
      if (/\.(txt|md|markdown|jsonl|ndjson|json)$/i.test(uri.fsPath)) {
        try {
          const content = fs.readFileSync(uri.fsPath, 'utf8');
          if (datasetNeedsLanguage(path.basename(uri.fsPath), content)) {
            const pick = await vscode.window.showQuickPick(KNOWN_LANGS, { placeHolder: 'Default language for records/blocks without one' });
            if (!pick) return;
            defaultLang = pick;
          }
        } catch { /* fall through */ }
      }
      const res = await brain.learnFromDatasetFile(uri, defaultLang);
      updateStatus();
      void vscode.window.showInformationMessage(
        res.items === 0
          ? 'MDP Snippets: dataset unchanged since last learn — nothing to do.'
          : `MDP Snippets: learned ${res.learned}/${res.items} record(s) — ${res.tokens.toLocaleString()} tokens at weight ${brain.cfg.datasetWeight}.`
      );
    }),

    vscode.commands.registerCommand('mdpSnippets.evaluateModel', async () => {
      const uris = await pickCorpus('Evaluate on');
      if (!uris?.length) return;
      const defaultLang = await maybeAskLanguage(uris);
      const items = await brain.loadCorpusFromUris(uris, defaultLang);
      if (items.length === 0) { void vscode.window.showWarningMessage('MDP Snippets: no evaluable records found.'); return; }
      const res = brain.evaluateOnCorpus(items);
      const pct = (x: number): string => `${Math.round(x * 100)}%`;
      output.appendLine(`\n=== Corpus evaluation (${new Date().toLocaleTimeString()}) ===`);
      output.appendLine(`items: ${items.length} · predictions: ${res.overall.predictions}`);
      output.appendLine(`perplexity ${res.overall.perplexity.toFixed(2)} · top-1 ${pct(res.overall.top1)} · top-3 ${pct(res.overall.top3)} · top-5 ${pct(res.overall.top5)} · OOV ${pct(res.overall.oovRate)}`);
      for (const p of res.perLang) {
        output.appendLine(`  ${p.lang}: PP ${p.metrics.perplexity.toFixed(2)} · top-1 ${pct(p.metrics.top1)} · OOV ${pct(p.metrics.oovRate)} (${p.metrics.predictions} preds)`);
      }
      output.show(true);
      void vscode.window.showInformationMessage(
        `MDP Snippets: PP ${res.overall.perplexity.toFixed(2)} · top-1 ${pct(res.overall.top1)} · top-5 ${pct(res.overall.top5)} over ${res.overall.predictions} predictions — details in the output channel.`
      );
    }),

    vscode.commands.registerCommand('mdpSnippets.autoTune', async () => {
      const uris = await pickCorpus('Tune on');
      if (!uris?.length) return;
      const defaultLang = await maybeAskLanguage(uris);
      const items = await brain.loadCorpusFromUris(uris, defaultLang);
      if (items.length < 10) { void vscode.window.showWarningMessage('MDP Snippets: need ≥ 10 records to tune.'); return; }
      const outcome = brain.tuneOnCorpus(items);
      output.appendLine(`\n=== Auto-tune (${new Date().toLocaleTimeString()}) ===`);
      output.appendLine(`train ≤ ${outcome.trainTokens.toLocaleString()} tokens · ${outcome.valItems} val items`);
      for (const r of outcome.results.slice(0, 12)) {
        output.appendLine(`  order ${r.order} · D ${r.smoothingD} · KN ${r.knDiscount} → PP ${r.perplexity.toFixed(2)} · top-1 ${Math.round(r.top1 * 100)}%${r.tempModel ? ' (fresh model)' : ''}`);
      }
      output.appendLine(`  — current config → PP ${outcome.current.perplexity.toFixed(2)} · top-1 ${Math.round(outcome.current.top1 * 100)}%`);
      output.show(true);
      if (outcome.best.perplexity >= outcome.current.perplexity * 0.98) {
        void vscode.window.showInformationMessage('MDP Snippets: your current config is already near-optimal on this corpus.');
        return;
      }
      const best = outcome.best;
      const pick = await vscode.window.showInformationMessage(
        `Best config: order ${best.order}, D=${best.smoothingD}${best.knDiscount ? `, KN=${best.knDiscount}` : ''} — PP ${best.perplexity.toFixed(2)} vs current ${outcome.current.perplexity.toFixed(2)}.`,
        'Apply', 'Apply & learn corpus', 'No'
      );
      if (!pick || pick === 'No') return;
      await applyTuneResult(best, brain);
      if (pick === 'Apply & learn corpus') {
        const res = brain.learnCorpusItems(items);
        brain.flushPool(true);
        void vscode.window.showInformationMessage(`MDP Snippets: corpus learned (${res.learned} records, ${res.tokens.toLocaleString()} tokens).`);
      } else if (best.order !== brain.cfg.order) {
        void vscode.window.showWarningMessage('MDP Snippets: order changed — the model was reset; run Learn from Folder/Dataset to refill it.');
      }
      updateStatus();
    }),

    vscode.commands.registerCommand('mdpSnippets.showReport', async () => {
      const file = path.join(context.globalStorageUri.fsPath, 'mdp-report.md');
      fs.mkdirSync(context.globalStorageUri.fsPath, { recursive: true });
      fs.writeFileSync(file, brain.learningReport(), 'utf8');
      await vscode.commands.executeCommand('markdown.showPreviewToSide', vscode.Uri.file(file));
    })
  );

  // ---- configuration changes ----
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (!event.affectsConfiguration('mdpSnippets')) return;
      const { orderChanged } = brain.refreshConfig();
      updateStatus();
      if (orderChanged) {
        void vscode.window.showWarningMessage('MDP Snippets: `order` changed — the learned model was reset.');
      }
    })
  );

  // ---- first-run hint ----
  if (brain.summary().bigrams === 0 && !context.globalState.get('mdpSnippets.introShown')) {
    void vscode.window
      .showInformationMessage(
        'MDP Snippets: run "Learn from Workspace" so the Markov chain has code to generate snippets from.',
        'Learn Workspace'
      )
      .then((pick) => {
        if (pick === 'Learn Workspace') void vscode.commands.executeCommand('mdpSnippets.learnFromWorkspace');
      });
    void context.globalState.update('mdpSnippets.introShown', true);
  }

  // ---- periodic tick: watch expiry, pool refresh, persistence ----
  const saveInterval = setInterval(() => brain.tick(), 60_000);
  context.subscriptions.push({ dispose: () => clearInterval(saveInterval) });

  updateStatus();
  status.show();
}

async function pickCorpus(openLabel: string): Promise<vscode.Uri[] | undefined> {
  return vscode.window.showOpenDialog({
    canSelectMany: true, canSelectFolders: true, openLabel,
    filters: { 'Datasets & code': ['jsonl', 'ndjson', 'json', 'txt', 'md'], 'All files': ['*'] },
  });
}

async function maybeAskLanguage(uris: vscode.Uri[]): Promise<string> {
  for (const uri of uris) {
    if (/\.(txt|md|markdown|jsonl|ndjson|json)$/i.test(uri.fsPath)) {
      let content: string;
      try { content = fs.readFileSync(uri.fsPath, 'utf8'); } catch { continue; }
      if (datasetNeedsLanguage(path.basename(uri.fsPath), content)) {
        return (await vscode.window.showQuickPick(KNOWN_LANGS, { placeHolder: 'Default language for records/blocks without one (Esc = skip them)' })) ?? '';
      }
    }
  }
  return '';
}

async function applyTuneResult(r: TuneResult, brain: Brain): Promise<void> {
  const cfg = vscode.workspace.getConfiguration('mdpSnippets');
  await cfg.update('smoothingD', r.smoothingD, vscode.ConfigurationTarget.Global);
  await cfg.update('knDiscount', r.knDiscount, vscode.ConfigurationTarget.Global);
  await cfg.update('smoothing', r.knDiscount > 0 ? 'kneser-ney' : 'interpolated', vscode.ConfigurationTarget.Global);
  if (r.order !== brain.cfg.order) {
    const ok = await vscode.window.showWarningMessage(
      `Changing the Markov order to ${r.order} resets the learned model (world model and values). Continue?`,
      'Yes, reset & apply'
    );
    if (ok) {
      await cfg.update('order', r.order, vscode.ConfigurationTarget.Global);
      brain.refreshConfig();
    }
  }
}

async function learnFromWorkspace(brain: Brain, refresh: () => void): Promise<void> {
  if (!vscode.workspace.workspaceFolders?.length) {
    void vscode.window.showWarningMessage('MDP Snippets: no workspace folder is open.');
    return;
  }
  const uris = await vscode.workspace.findFiles(
    '**/*',
    '**/{node_modules,.git,dist,out,build,.next,coverage,vendor,__pycache__,.venv}/**',
    1500
  );
  let learned = 0;
  let skipped = 0;
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: 'MDP Snippets: learning workspace', cancellable: true },
    async (progress, cancelToken) => {
      for (let i = 0; i < uris.length; i++) {
        if (cancelToken.isCancellationRequested) break;
        try {
          const doc = await vscode.workspace.openTextDocument(uris[i]);
          if (brain.learnDocument(doc)) learned++;
          else skipped++;
        } catch {
          skipped++;
        }
        if (i % 20 === 0) {
          progress.report({ message: `${i + 1}/${uris.length}` });
          refresh();
        }
      }
    }
  );
  brain.saveNow();
  brain.flushPool(true);
  refresh();
  void vscode.window.showInformationMessage(`MDP Snippets: learned ${learned} file(s), skipped ${skipped}.`);
}

export function deactivate(): void {
  activePool?.dispose();
  activeBrain?.saveNow();
}
