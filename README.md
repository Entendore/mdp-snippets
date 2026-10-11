# MDP Snippets

Inline code completions driven by a Markov Decision Process. A token Markov
chain (dual literal/abstract channels, Kneser-Ney smoothing, language-family
hierarchical transfer) learns what your code looks like; a TD(λ),
distributional, double-Q agent with a linear preference head and a low-rank
value model learns which suggestions *you* accept.

- **Tab** accepts · **Esc** dismisses (rewarded) · **Ctrl+Alt+U/D** thumbs.
- **Learn from Workspace / Folder / Dataset File** teach the world model
  (datasets never teach your preferences — rewards come only from you).
- **Show Dashboard**, **Auto-Tune on Dataset**, **Show Learning Report**.
- Generation runs in a worker thread; all learning is local and persisted.

Build (requires bun, https://bun.sh): `bun install && bun run compile && bun test`.
