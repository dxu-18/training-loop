mkdir proj && cd proj && git init -q
echo 0 > f && git add f && git commit -qm c0
git switch -qc a && echo a1 > a && git add a && git commit -qm a1
git switch -qc b main && echo b1 > b && git add b && git commit -qm b1
git switch -q a && echo a2 >> a && git commit -qam a2
git switch -q main && echo m1 >> f && git commit -qam m1
git merge -q --no-edit a
git merge -q --no-edit b
git log --oneline --graph --all #~ (graph.c collapse on the commit row: TODO)
git log --graph --format=%s #~ (graph.c collapse on the commit row: TODO)
git log --oneline --first-parent
git log --oneline --merges
git log --oneline --no-merges
git log --oneline main..b
git log --oneline a...b
