# AGENTS.md

## Repo notes
- Manuscript sources live under `paper/`.
- Current manuscript has one large `.tex` file and one `.bib` file.
- Goal is to reorganize the manuscript into a cleaner multi-file structure.

## Task constraints
- Preserve exact scientific content, equations, citations, labels, bibliography behavior, and final compiled PDF content.
- Do not rewrite prose unless necessary to keep compilation working.
- Do not change notation, theorem numbering, figure numbering, table numbering, equation numbering, or citation keys.
- Prefer splitting by top-level sections into `paper/sections/`.
- Keep `paper/main.tex` as the entry point after refactor.
- If macros/preamble are large, they may stay in `main.tex` unless there is a clear safe split.
- After changes, run a compile check and report any LaTeX errors.