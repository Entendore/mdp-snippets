/** Language registry: canonical ids, families, per-language tokenizer /
 *  generation profiles. Unknown languages fall back to the legacy profile. */
export type Family = 'c-like' | 'scripting' | 'lisp' | 'ml' | 'shell' | 'sql' | 'web' | 'data' | 'legacy';

export const FAMILIES: ReadonlySet<string> = new Set(['c-like', 'scripting', 'lisp', 'ml', 'shell', 'sql', 'web', 'data', 'legacy']);

export interface LangProfile {
  family: Family;
  keywords: Set<string>;
  slashComments: boolean;
  hashComments: boolean;
  dashComments: boolean;
  lispComments: boolean;
  blockComments: boolean;
  backtickStrings: boolean;
  tripleQuotes: boolean;
  eolTerminates: boolean;
  terminators: Set<string>;
}

const kw = (s: string): Set<string> => new Set(s.split(/\s+/).filter(Boolean));

const LEGACY_KW =
  'const let var function return if else for while do switch case break continue class extends implements interface enum type namespace new delete typeof instanceof in of import export from as default async await yield try catch finally throw this super static get set public private protected readonly declare def lambda pass None True False self print range len elif except with global nonlocal raise assert not and or is null undefined true false void struct impl fn pub mut use match package func go defer';

const FAMILY_BASE: Partial<Record<Family, string>> = {
  'c-like': 'if else for while return break continue new this try catch finally throw true false null class import export',
  scripting: 'if else for while return break continue true false import from as',
  ml: 'let in match with type rec module open fun function if then else',
  shell: 'if then else fi for do done while case esac in function return local export',
  sql: 'select from where group by order having insert into values update set delete join on as and or not null distinct create table primary key foreign references',
  web: 'if else return import',
};

interface LangDef {
  family: Family;
  kw?: string;
  slash?: boolean; hash?: boolean; dash?: boolean; lisp?: boolean; block?: boolean;
  backtick?: boolean; triple?: boolean; eol?: boolean; terms?: string;
}

const LANGS: Record<string, LangDef> = {
  typescript: { family: 'c-like', backtick: true, kw: 'const let var function async await yield typeof instanceof in of switch case default do delete as from static get set public private protected readonly declare type namespace void undefined super satisfies keyof infer asserts' },
  javascript: { family: 'c-like', backtick: true, kw: 'const let var function async await yield typeof instanceof in of switch case default do delete as from static get set void undefined super' },
  java: { family: 'c-like', kw: 'public private protected static final abstract synchronized package boolean int long float double char byte short extends implements instanceof throws assert var record sealed' },
  c: { family: 'c-like', kw: 'int char float double void unsigned signed struct union typedef sizeof const inline restrict' },
  cpp: { family: 'c-like', kw: 'int char float double void unsigned signed struct union typedef sizeof const inline constexpr namespace using template typename virtual override friend operator auto nullptr decltype' },
  csharp: { family: 'c-like', kw: 'using namespace public private protected internal static readonly override virtual abstract sealed async await var dynamic string int bool double object get set value when where' },
  go: { family: 'c-like', kw: 'func package var const type struct interface map chan defer select range fallthrough nil bool string int' },
  rust: { family: 'c-like', kw: 'fn let mut pub use mod struct enum impl trait where as dyn ref move crate unsafe extern match loop Self super' },
  swift: { family: 'c-like', kw: 'func let var guard defer struct class enum protocol extension where as in is Self init deinit weak lazy' },
  kotlin: { family: 'c-like', kw: 'fun val var when object companion data sealed open override lateinit inline reified suspend' },
  php: { family: 'c-like', kw: 'function echo foreach as require include namespace trait instanceof' },
  scala: { family: 'c-like', kw: 'def val var object trait implicit match case yield lazy sealed' },
  dart: { family: 'c-like', kw: 'void final late async await extension mixin' },
  python: { family: 'scripting', triple: true, kw: 'def lambda pass None True False self print range len elif except with global nonlocal raise assert not and or is del async await' },
  ruby: { family: 'scripting', kw: 'def end module elsif unless until begin rescue ensure raise yield when next redo retry nil require do then and or not' },
  clojure: { family: 'lisp', kw: 'def defn defn- let fn if cond do loop recur ns declare' },
  scheme: { family: 'lisp', kw: 'define let lambda if cond do set!' },
  commonlisp: { family: 'lisp', kw: 'defun let lambda if cond do setq' },
  fsharp: { family: 'ml' },
  ocaml: { family: 'ml', dash: true },
  haskell: { family: 'ml', dash: true },
  elm: { family: 'ml' },
  shellscript: { family: 'shell' },
  powershell: { family: 'shell' },
  bat: { family: 'shell' },
  sql: { family: 'sql', dash: true, block: true },
  html: { family: 'web' },
  css: { family: 'web' },
  scss: { family: 'web', slash: true },
  less: { family: 'web', slash: true },
  vue: { family: 'web' },
  svelte: { family: 'web' },
  json: { family: 'data' },
  yaml: { family: 'data', hash: true, eol: true },
  toml: { family: 'data', hash: true },
  ini: { family: 'data', hash: true },
  xml: { family: 'data' },
};

const CANONICAL: Record<string, string> = {
  typescriptreact: 'typescript', tsx: 'typescript', ts: 'typescript',
  javascriptreact: 'javascript', jsx: 'javascript', js: 'javascript',
  bash: 'shellscript', sh: 'shellscript', zsh: 'shellscript', shell: 'shellscript',
  python3: 'python', py: 'python', ipython: 'python',
  'c++': 'cpp', 'objective-c++': 'cpp', 'objective-c': 'c',
  golang: 'go', 'rust-lang': 'rust', rs: 'rust',
  clojurescript: 'clojure', cljs: 'clojure',
  postcss: 'css', sass: 'scss',
  yml: 'yaml', 'github-actions-workflow': 'yaml',
  dockerfile: 'shellscript', makefile: 'shellscript',
};

const FAMILY_DEFAULTS: Record<Family, Partial<LangDef>> = {
  'c-like': { slash: true, block: true, terms: ';' },
  scripting: { hash: true, eol: true },
  lisp: { lisp: true },
  ml: { slash: true },
  shell: { hash: true, eol: true },
  sql: { dash: true, block: true, terms: ';' },
  web: { block: true, terms: ';' },
  data: { eol: true },
  legacy: { slash: true, hash: true, block: true, backtick: true, terms: ';' },
};

export function canonicalLang(id: string | undefined): string {
  if (!id) return '';
  const lower = id.toLowerCase();
  return CANONICAL[lower] ?? lower;
}

const cache = new Map<string, LangProfile>();

export function profileFor(id: string | undefined): LangProfile {
  const key = canonicalLang(id);
  if (key === '' || !LANGS[key]) {
    return {
      family: 'legacy', keywords: kw(LEGACY_KW),
      slashComments: true, hashComments: true, dashComments: false, lispComments: false,
      blockComments: true, backtickStrings: true, tripleQuotes: false,
      eolTerminates: false, terminators: new Set([';']),
    };
  }
  let p = cache.get(key);
  if (!p) {
    const def = LANGS[key];
    const fam = FAMILY_DEFAULTS[def.family];
    const terms = def.terms ?? fam.terms ?? '';
    p = {
      family: def.family,
      keywords: kw((FAMILY_BASE[def.family] ?? '') + ' ' + (def.kw ?? '')),
      slashComments: def.slash ?? fam.slash ?? false,
      hashComments: def.hash ?? fam.hash ?? false,
      dashComments: def.dash ?? fam.dash ?? false,
      lispComments: def.lisp ?? fam.lisp ?? false,
      blockComments: def.block ?? fam.block ?? false,
      backtickStrings: def.backtick ?? fam.backtick ?? false,
      tripleQuotes: def.triple ?? fam.triple ?? false,
      eolTerminates: def.eol ?? fam.eol ?? false,
      terminators: new Set(terms.split(/\s+/).filter(Boolean)),
    };
    cache.set(key, p);
  }
  return p;
}

export function familyOf(id: string | undefined): Family {
  return profileFor(id).family;
}

const EXT_TO_LANG: Record<string, string> = {
  ts: 'typescript', tsx: 'typescript', mts: 'typescript', js: 'javascript', jsx: 'javascript', mjs: 'javascript', cjs: 'javascript',
  py: 'python', pyi: 'python', rb: 'ruby', go: 'go', rs: 'rust', java: 'java', c: 'c', h: 'c',
  cpp: 'cpp', cc: 'cpp', cxx: 'cpp', hpp: 'cpp', hh: 'cpp', cs: 'csharp', swift: 'swift', kt: 'kotlin', kts: 'kotlin',
  php: 'php', scala: 'scala', dart: 'dart', clj: 'clojure', cljs: 'clojure', cljc: 'clojure',
  scm: 'scheme', ss: 'scheme', lisp: 'commonlisp', el: 'commonlisp', fs: 'fsharp', fsi: 'fsharp',
  ml: 'ocaml', mli: 'ocaml', hs: 'haskell', elm: 'elm',
  sh: 'shellscript', bash: 'shellscript', zsh: 'shellscript', fish: 'shellscript', ps1: 'powershell', bat: 'bat', cmd: 'bat',
  sql: 'sql', html: 'html', htm: 'html', css: 'css', scss: 'scss', sass: 'scss', less: 'less',
  vue: 'vue', svelte: 'svelte', json: 'json', yaml: 'yaml', yml: 'yaml', toml: 'toml', ini: 'ini', xml: 'xml',
};

export function langForFilename(name: string): string {
  const ext = name.toLowerCase().split('.').pop() ?? '';
  return EXT_TO_LANG[ext] ?? '';
}

export const KNOWN_LANGS: string[] = Object.keys(LANGS);
