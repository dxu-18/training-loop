git status
git log
mkdir proj && cd proj && git init -q
git log
git commit -m "nothing"
echo a > a.txt
git commit -m "nothing staged"
git add nope.txt
git add a.txt && git commit -qm "first"
git commit --amend --no-edit
git show --stat --oneline HEAD
git log --oneline
git checkout nope
git switch nope
git log nope
git diff HEAD~5
git branch -d main #~ (git 2.43 wording; also the real run's /private path)
git checkout -b topic
echo b > b.txt && git add b.txt && git commit -qm "topic work"
git switch -
git switch -
git checkout HEAD~1
git status
git switch main
git branch -D topic
git tag v1
git tag v1
git tag -d v1
git rm nope.txt
git mv a.txt docs.txt
git status -s
git commit -qm "rename"
git log --oneline --name-status
echo x >> docs.txt
git diff --staged
git diff --stat
git add docs.txt
git diff --staged
git reset -- docs.txt
git status -s
git config user.name
git config --list --show-origin #~ (macOS adds core.ignorecase / precomposeunicode)
git config --global alias.lg "log --oneline --graph --decorate --all"
git lg
git frobnicate
git stauts
