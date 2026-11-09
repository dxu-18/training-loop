mkdir proj && cd proj
git init -q
echo "print('hi')" > train.py
printf 'a\nb\nc\n' > notes.txt
git status
git add train.py
git status
git status -s
git commit -m "feat: add train script"
git add notes.txt
git commit -m "docs: notes"
echo "lr = 1e-3" >> train.py
git status
git diff
git commit -am "tune lr"
git log --oneline
git log
git log --stat
git show HEAD~1
git rev-parse HEAD
git cat-file -p HEAD
git ls-files -s
