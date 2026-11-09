# worktrees, submodules, sparse checkout
mkdir lib && cd lib && git init -q && echo lib > l.txt && git add . && git commit -qm lib && cd ..
mkdir app && cd app && git init -q && mkdir -p src docs && echo a > src/a.c && echo d > docs/d.md && echo r > README && git add . && git commit -qm app
git worktree add ../hotfix
git worktree add -b feat ../feat-wt
git worktree list
git branch
cd ../feat-wt && echo x > x && git add x && git commit -qm x && git log --oneline && cd ../app
git switch feat
git worktree remove ../hotfix
git worktree list
git submodule add ../lib vendor/lib
git -c protocol.file.allow=always submodule add ../lib vendor/lib
git status --short
git commit -qm "add lib"
git submodule status
cd .. && git clone -q app app2 && cd app2
git submodule status
git -c protocol.file.allow=always submodule update --init
git submodule status
ls vendor/lib
git submodule foreach 'git log --oneline'
cd ../app
git sparse-checkout set src
ls
git status
git sparse-checkout list
git sparse-checkout disable
ls
