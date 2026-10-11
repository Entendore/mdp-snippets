/** Inline completion provider: full state (tokens, indent, EOL, chain bit,
 *  session belief, outline region, workspace tag), burst debouncing, audited
 *  gating, fallback-state read-through, reward-carrying items. */
import * as vscode from 'vscode';
import { tokenize, tail } from './util';
import { canonicalLang } from './languages';
import { qStateKeyOf } from './generate';
import { regionAt } from './region';
import { Brain } from './brain';

export const ACCEPT_CMD = 'mdpSnippets._accepted';

export function getTextBefore(document: vscode.TextDocument, position: vscode.Position): string {
  const firstLine = Math.max(0, position.line - 60);
  const parts: string[] = [];
  for (let line = firstLine; line < position.line; line++) parts.push(document.lineAt(line).text, '\n');
  parts.push(document.lineAt(position.line).text.slice(0, position.character));
  return parts.join('');
}

export class MdpInlineProvider implements vscode.InlineCompletionItemProvider {
  private cacheKey = '';
  private cacheItems: vscode.InlineCompletionItem[] = [];
  private lastRequestAt = 0;

  constructor(private readonly brain: Brain) {}

  public async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ): Promise<vscode.InlineCompletionItem[]> {
    const cfg = this.brain.cfg;
    if (token.isCancellationRequested) return [];
    if (document.uri.scheme !== 'file' && document.uri.scheme !== 'untitled') return [];
    const lang = canonicalLang(document.languageId);
    if (!this.brain.enabledFor(lang)) return [];
    if (cfg.languages.length > 0 && !cfg.languages.includes(lang)) return [];

    const uri = document.uri.toString();
    const cacheKey = `${uri}|${document.version}|${position.line}|${position.character}`;
    if (cacheKey === this.cacheKey && this.brain.hasPending) return this.cacheItems;

    // Settle promptly (reward attribution uses this invocation's document state).
    const textBefore = getTextBefore(document, position);
    this.brain.settle(document, position, textBefore);
    void vscode.commands.executeCommand('setContext', 'mdpSnippets.suggestionActive', false);

    if (!(await this.debounce(token))) return [];
    if (token.isCancellationRequested) return [];

    const lineText = document.lineAt(position.line).text;
    const atEol = position.character >= lineText.length;
    const typed = lineText.slice(0, position.character);
    const indentStr = /^[ \t]*/.exec(typed)![0];
    const prefix = /[A-Za-z0-9_$]+$/.exec(typed)?.[0] ?? '';

    const tokens = tokenize(tail(textBefore, 1200), lang);
    const litBefore = prefix && tokens.length > 0 && tokens[tokens.length - 1] === prefix ? tokens.slice(0, -1) : tokens;
    if (litBefore.length === 0 && prefix.length === 0) { this.remember(cacheKey, []); return []; }

    const region = regionAt(document, position.line, position.character, lang);
    const bucket = Math.min(3, Math.floor(indentStr.length / 2));
    const ws = this.brain.wsTag;
    const keyOpts = { chain: this.brain.chainActive, mode: this.brain.sessionMode, region: region.tag };
    const stateKey = qStateKeyOf(lang, litBefore, cfg.order, bucket, atEol, ws ? { ...keyOpts, ws } : keyOpts);
    const fallbackKey = ws ? qStateKeyOf(lang, litBefore, cfg.order, bucket, atEol, keyOpts) : stateKey;

    const candidates = await this.brain.generateCandidates(lang, uri, stateKey, litBefore, prefix, atEol, indentStr, region.tag);
    if (candidates.length === 0) { this.remember(cacheKey, []); return []; }

    const { ordered, topScore, piB, greedyShown, primaryScore } = this.brain.plan(stateKey, candidates, fallbackKey);
    const gate = this.brain.gateDecision(stateKey, ordered[0], topScore, fallbackKey);
    if (gate.suppress) { this.remember(cacheKey, []); return []; }

    this.brain.beginProposal({
      uri, lang, stateKey,
      version: document.version,
      line: position.line,
      character: position.character,
      cursorOffset: document.offsetAt(position),
      beforeLen: textBefore.length,
      shownAt: Date.now(),
      indentBucket: bucket,
      litBefore,
      atEol,
      indentStr,
      piB,
      greedyShown,
      probe: gate.probe,
      probeScore: primaryScore,
    });

    const items: vscode.InlineCompletionItem[] = [];
    for (const candidate of ordered) {
      const replacesWord = prefix.length > 0 && candidate.tokens[0].startsWith(prefix);
      const range = replacesWord
        ? new vscode.Range(position.line, position.character - prefix.length, position.line, position.character)
        : undefined;
      const id = this.brain.addProposalEntry(candidate, candidate.text, replacesWord ? prefix.length : 0);
      items.push(
        new vscode.InlineCompletionItem(candidate.text, range, {
          title: 'MDP Snippets: record acceptance',
          command: ACCEPT_CMD,
          arguments: [id],
        })
      );
    }
    this.remember(cacheKey, items);
    return items;
  }

  /** Burst debounce: sleep until the burst quiets or the cap elapses. */
  private async debounce(token: vscode.CancellationToken): Promise<boolean> {
    const cfg = this.brain.cfg;
    if (cfg.debounceMs <= 0) return true;
    const now = Date.now();
    const inBurst = now - this.lastRequestAt < cfg.debounceMs;
    this.lastRequestAt = now;
    if (!inBurst) return true;
    const deadline = now + Math.max(cfg.debounceMs, cfg.maxDebounceMs);
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, cfg.debounceMs));
      if (token.isCancellationRequested) return false;
      const t = Date.now();
      if (t - this.lastRequestAt >= cfg.debounceMs || t >= deadline) return true;
    }
  }

  private remember(key: string, items: vscode.InlineCompletionItem[]): void {
    this.cacheKey = key;
    this.cacheItems = items;
    void vscode.commands.executeCommand('setContext', 'mdpSnippets.suggestionActive', items.length > 0);
  }
}
