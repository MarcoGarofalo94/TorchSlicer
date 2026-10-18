#!/usr/bin/env bash
# Merge a milestone branch into main WITHOUT notes/ (notes live only on the branches, like
# CLAUDE.md, which isn't committed at all). Opens a PR; it does not merge it.
#
#   notes/merge_to_main.sh feat/m11-data-scale
#
# Works in a separate worktree (../TorchSlicer-merge), so the current checkout is untouched.
# Shipped files (README, docs, code, tests) must not reference notes/: the script stops if
# they do. Every merge drops notes/ again, so later merges only conflict on notes/ deletions,
# which it resolves.
set -euo pipefail
BRANCH=${1:?usage: notes/merge_to_main.sh <branch>}
WORK=merge/${BRANCH//\//-}
DIR=$(git rev-parse --show-toplevel)/../TorchSlicer-merge

git fetch -q origin
git worktree add -q -B "$WORK" "$DIR" origin/main
cd "$DIR"
git merge --no-ff --no-commit "origin/$BRANCH" || true
git rm -r -q -f --ignore-unmatch notes             # notes never reach main
for f in $(git diff --name-only --diff-filter=U); do
  case "$f" in notes/*) git rm -q -f "$f" ;; *) echo "conflict outside notes/: $f"; exit 1 ;; esac
done
if git grep -n -I -E "notes/|validation\.md|roadmap\.md|experiments\.md|sota\.md|gaps\.md" -- . ; then
  echo "shipped files reference the notes (above): fix them on $BRANCH first"; exit 1
fi
git commit -q -m "Merge $BRANCH into main (without notes/)"
git push -q -u origin "$WORK"
URL=$(git remote get-url origin | sed -E 's#^git@github.com:#https://github.com/#; s#\.git$##')
GH=$(command -v gh || echo "$HOME/.local/bin/gh")
if [ -x "$GH" ] && "$GH" auth status >/dev/null 2>&1; then
  "$GH" pr create --base main --head "$WORK" --title "Merge $BRANCH into main" \
    --body "Merges \`$BRANCH\` into \`main\` without \`notes/\` (notes stay on the milestone branches)."
else
  echo "open the PR: $URL/compare/main...$WORK?expand=1"
fi
echo "worktree at $DIR (remove with: git worktree remove $DIR)"
