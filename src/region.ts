/** Syntactic-region heuristic: classify the cursor's enclosing structure with
 *  a bounded bracket/declaration scan (no language server). Approximate by
 *  design; unclear cases degrade to 'O' (other) / '' (top level). */
import { profileFor } from './languages';

export interface RegionDoc {
  lineCount: number;
  lineAt(line: number): { text: string };
}
export type RegionTag = '' | 'C' | 'F' | 'S' | 'I' | 'O';
export interface Region { tag: RegionTag; depth: number }

const DECL_CLASS = /\b(class|interface|struct|enum|trait|impl|namespace|module|object)\b/;
const DECL_FUNC = /\b(function|fn|func|def|constructor|method)\b|=>/;
const IMPORT_RE = /^\s*(import\b|from\s+[\w.]+\s+import\b|#include|use\s+\w|require\s*[\s(]|@import)/;

function cleanLine(text: string, lang?: string): string {
  const p = profileFor(lang);
  let s = text;
  s = s.replace(/"(?:[^"\\]|\\.)*"/g, '""').replace(/'(?:[^'\\]|\\.)*'/g, "''");
  if (p.backtickStrings) s = s.replace(/`(?:[^`\\]|\\.)*`/g, '``');
  if (p.slashComments) s = s.replace(/\/\/.*$/, '');
  if (p.hashComments) s = s.replace(/#.*$/, '');
  if (p.dashComments) s = s.replace(/--.*$/, '');
  if (p.lispComments) s = s.replace(/;.*$/, '');
  if (p.blockComments) s = s.replace(/\/\*.*?\*\//g, '');
  return s;
}

export function regionAt(doc: RegionDoc, line: number, character: number, lang?: string): Region {
  const start = Math.max(0, line - 500);
  let depth = 0;
  const frames: { tag: 'C' | 'F'; depth: number }[] = [];
  const brackets: { ch: string; line: number }[] = [];

  const pushBracket = (ch: string, i: number): void => {
    brackets.push({ ch, line: i });
    if (brackets.length > 256) brackets.shift();
  };

  for (let i = start; i < line; i++) {
    const clean = cleanLine(doc.lineAt(i).text, lang);
    let opens = 0, closes = 0;
    for (const ch of clean) {
      if (ch === '{' || ch === '(' || ch === '[') { pushBracket(ch, i); opens++; }
      else if (ch === '}' || ch === ')' || ch === ']') { brackets.pop(); closes++; }
    }
    const lineStartDepth = depth;
    depth += opens - closes;
    if (DECL_CLASS.test(clean) && opens > closes) frames.push({ tag: 'C', depth: lineStartDepth });
    else if (DECL_FUNC.test(clean) && opens > closes) frames.push({ tag: 'F', depth: lineStartDepth });
    while (frames.length && depth <= frames[frames.length - 1].depth) frames.pop();
  }

  const cursorText = doc.lineAt(line).text.slice(0, character);
  if (IMPORT_RE.test(cursorText)) return { tag: 'I', depth: Math.min(3, depth) };
  const cursorClean = cleanLine(cursorText, lang);
  for (const ch of cursorClean) {
    if (ch === '{' || ch === '(' || ch === '[') pushBracket(ch, line);
    else if (ch === '}' || ch === ')' || ch === ']') brackets.pop();
  }

  const top = brackets[brackets.length - 1];
  if (top && top.ch === '(') {
    const openLine = cleanLine(doc.lineAt(top.line).text, lang);
    if (DECL_CLASS.test(openLine) || DECL_FUNC.test(openLine)) return { tag: 'S', depth: Math.min(3, depth) };
  }
  const frame = frames[frames.length - 1];
  if (frame) return { tag: frame.tag, depth: Math.min(3, depth) };
  if (depth > 0 || brackets.length > 0) return { tag: 'O', depth: Math.min(3, depth) };
  return { tag: '', depth: 0 };
}
