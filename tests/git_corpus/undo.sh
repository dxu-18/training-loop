mkdir proj && cd proj && git init -q
for i in 1 2 3 4; do echo "step $i" >> log.txt; git add log.txt; git commit -qm "step $i"; done
git log --oneline
git reset --soft HEAD~1
git status -s
git reset HEAD~1
git status
git reset --hard HEAD
git status -s
cat log.txt
git reset --hard HEAD~1
git log --oneline
git reflog
git reset --hard HEAD@{2}
git log --oneline
echo junk >> log.txt
git restore log.txt
git status -s
echo staged >> log.txt
git add log.txt
git restore --staged log.txt
git status -s
git checkout -- log.txt
git revert --no-edit HEAD
git log --oneline
git show --stat HEAD
git commit --amend -m "Revert the last step"
git log --oneline -2
touch tmp1.txt tmp2.txt
mkdir build && touch build/out.o
git clean -n
git clean -fd
git status -s
