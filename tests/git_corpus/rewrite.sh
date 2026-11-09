mkdir proj && cd proj && git init -q
echo base > a.txt && git add . && git commit -qm base
git switch -q -c feature
for i in 1 2 3; do echo "wip $i" >> a.txt; git commit -qam "wip $i"; done
git switch -q main
echo other > b.txt && git add b.txt && git commit -qm "main work"
git switch feature
git rebase main
git log --oneline --graph --all
GIT_SEQUENCE_EDITOR="sed -i -e '2,3s/^pick/squash/'" GIT_EDITOR=true git rebase -i HEAD~3 #~ (pipe ordering of stdout vs stderr)
git log --oneline
git show --stat HEAD
git switch -q main
echo fix > fix.txt && git add fix.txt && git commit -qm "fix: typo"
git switch -q feature
git cherry-pick main
git log --oneline -2
echo stash-me >> a.txt
git stash
git stash list
git status -s
git stash pop
git status -s
git stash push -m "wip feature"
git stash list
git stash show
git stash drop
git stash list
printf 'venv/\n*.pt\n!keep.pt\n' > .gitignore
mkdir venv && touch venv/x model.pt keep.pt
git status -s
git check-ignore -v model.pt venv/x keep.pt
git add -A
git status -s
git rm --cached keep.pt
git status -s
