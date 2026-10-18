# Notes (development branches only)

- This folder **never ships to `main`**, like `CLAUDE.md`. It lives on the development
  branches (milestone `feat/m<k>-*`, research `research/*`), which will be deleted once the
  project is stable. Shipped files (README, `docs/`, code, tests, benchmarks) must not
  reference anything in here.
- Milestones merge into `main` with `notes/merge_to_main.sh <branch>` (drops `notes/`, checks
  that no shipped file references it, opens a PR; merge after CI is green). New milestone
  branches fork from the previous milestone branch, not from `main`.
- `roadmap.md` (milestones and handoff), `validation.md` (every bug and fix, for the paper),
  `experiments.md` (append-only run log), `algorithms.md`, `offload.md` (design surveys).
- Research: `sota.md`, `gaps.md` are shared by the research lines. Each line has its branch
  `research/<id>-<slug>`, forked from the latest milestone branch, with its own
  `plan_<id>.md` and `experiments.md`. When a line updates `sota.md`/`gaps.md`, commit that on
  its own (`docs(notes): ...`) so it can be cherry-picked into the other branches. Abandoned
  lines are kept, marked in `gaps.md` as `status: abandoned (<reason>)`.
