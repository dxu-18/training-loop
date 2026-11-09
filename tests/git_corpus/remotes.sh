git init -q --bare server.git
git clone server.git work
cd work
echo hello > README.md && git add . && git commit -qm "first"
git push origin main
git push
git status
git log --oneline --decorate
cd ..
git clone server.git other
cd other && git log --oneline && echo more >> README.md && git commit -qam "second" && git push && cd ..
cd work
git fetch
git status
git pull
git log --oneline
git switch -c feature
echo f > f.txt && git add f.txt && git commit -qm "feature"
git push -u origin feature
git branch -vv
git remote -v
git remote show origin
git ls-remote origin
git push origin --delete feature
git branch -r
git fetch --prune
git branch -r
git tag v1.0 && git push origin v1.0
git ls-remote --tags origin
cd ../other
echo diverge >> README.md && git commit -qam "other side"
cd ../work && git switch -q main && echo mine >> README.md && git commit -qam "my side" && git push
cd ../other && git push
git pull --no-rebase --no-edit
git log --oneline --graph
git push
