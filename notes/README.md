# Research notes

- `sota.md`, `gaps.md` are **shared**: they live on `main` and every research branch inherits them.
  When a branch updates them, commit that change on its own (`docs(notes): ...`) so it can be
  cherry-picked back to `main` and merged into the other lines.
- Each research line has its own local branch `research/<id>-<slug>` (e.g. `research/g1-trust-ft`)
  forked from `main`. Line-specific notes live only on that branch:
  `plan_<id>.md` (design) and `experiments.md` (append-only run log).
- Nothing is pushed for now; branches are local. Abandoned lines are kept (not deleted) so negative
  results stay on record — mark them in `gaps.md` as `status: abandoned (<reason>)`.
