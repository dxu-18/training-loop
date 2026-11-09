mkdir proj && cd proj && git init -q
printf 'def forward(x):\n    return x\n' > model.py
printf 'import model\n' > train.py
git add . && git commit -qm "feat: model and train"
printf 'def forward(x):\n    return x * 2\n\ndef lr_scheduler(step):\n    return 1e-3\n' > model.py
git commit -qam "feat: add lr_scheduler"
git mv train.py run.py
git commit -qm "refactor: rename train to run"
printf 'import model\nprint(model.forward(3))\n' > run.py
git commit -qam "fix: print result"
git log --oneline
git log --format='%h %an %s'
git log -p -1
git log --stat --oneline
git log --follow --oneline -- run.py
git log -S lr_scheduler --oneline
git log --grep=fix --oneline
git log --oneline -- model.py
git diff HEAD~2 HEAD --stat
git diff HEAD~2..HEAD --name-status
git diff HEAD~3 -- model.py
git show HEAD~1:run.py
git show --name-only HEAD
git blame model.py
git shortlog -sn HEAD
git tag v0.1 HEAD~2
git tag -a v0.2 -m "second release"
git tag
git describe --tags
git describe
git log --oneline --decorate
git rev-parse --short HEAD
git rev-parse --abbrev-ref HEAD
git rev-list --count HEAD
git ls-tree HEAD
git ls-tree -r --name-only HEAD
