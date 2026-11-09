mkdir proj && cd proj && git init -q
printf 'lr = 1e-2\nepochs = 10\nbatch = 32\n' > config.py
git add . && git commit -qm "init config"
git branch feature
git branch
git switch feature
printf 'lr = 1e-2\nepochs = 20\nbatch = 32\n' > config.py
git commit -qam "more epochs"
git switch main
git merge feature
git log --oneline --graph --all
git switch -c exp
echo "dropout = 0.1" >> config.py
git commit -qam "add dropout"
git switch main
echo "seed = 0" > seed.py
git add seed.py && git commit -qm "add seed"
git merge --no-ff --no-edit exp
git log --oneline --graph
git branch -v
git branch --merged
git branch -d exp
git branch -d feature
git switch -c conflict-a
printf 'lr = 3e-4\nepochs = 20\nbatch = 32\ndropout = 0.1\n' > config.py
git commit -qam "lr 3e-4"
git switch main
printf 'lr = 1e-3\nepochs = 20\nbatch = 32\ndropout = 0.1\n' > config.py
git commit -qam "lr 1e-3"
git merge conflict-a
git status
cat config.py
git diff --name-only --diff-filter=U
git merge --abort
git status -s
git merge -X theirs conflict-a
cat config.py
git log --oneline -3
