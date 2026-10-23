### cut sort uniq
printf 'id,label,score\n3,cat,0.5\n1,dog,0.9\n2,cat,0.7\n4,bird,0.9\n' > data.csv
cut -d, -f1,3 data.csv
cut -c1-3 data.csv
cut -d, -f2- data.csv
cut -d, --complement -f2 data.csv
cut -d, -f2 data.csv | tail -n +2 | sort | uniq -c | sort -rn
sort data.csv; sort -r data.csv; sort -t, -k3,3nr data.csv | head -3
sort -t, -k2,2 -k1,1n data.csv
printf '10\n9\n100\n1\n' | sort -n; printf '10\n9\n100\n1\n' | sort
printf '1K\n2M\n512\n1G\n3K\n' | sort -h
printf 'b\na\nb\nc\na\n' | sort -u
printf 'B\na\nC\nb\n' | sort -f
printf 'x 3\ny 1\nz 2\n' | sort -k2n
printf 'v1.10\nv1.2\nv1.9\n' | sort -V
printf 'a\na\nb\nc\nc\nc\n' | uniq -d; printf 'a\na\nb\nc\nc\nc\n' | uniq -u; printf 'a\nA\nb\n' | uniq -i -c
printf '  b 2\n a 1\nc 3\n' | sort -k1,1
printf '3\n1\n2\n' | sort -c; echo "rc=$?"
printf 'a 1\nb 2\n' | sort -k2,2r
### tr wc head tail
echo "Hello World" | tr 'a-z' 'A-Z'
printf 'a\r\nb\r\n' | tr -d '\r' | od -c | head -2
echo "too    many   spaces" | tr -s ' '
echo "a,b,c" | tr ',' '\t'
echo "hello" | tr -d 'l'; echo "hello" | tr -cd 'el\n'; echo "abc" | tr 'abc' 'x'
echo "Hello" | tr '[:upper:]' '[:lower:]'; echo "a1b2" | tr -d '[:digit:]'
printf 'one two\nthree\n' | wc; printf 'one two\nthree\n' | wc -l; printf 'one two\nthree\n' | wc -w -c
printf 'a\nb\n' > f1.txt; printf 'x\ny\nz\n' > f2.txt; wc -l f1.txt f2.txt; wc f1.txt
seq 1 30 > nums.txt
head nums.txt | tr '\n' ' '; echo; head -n 3 nums.txt; head -c 5 nums.txt; echo; head -n -27 nums.txt
tail -n 3 nums.txt; tail -n +28 nums.txt; tail -c 6 nums.txt; tail -3 nums.txt
head -n 2 f1.txt f2.txt
sed -n '10,12p' nums.txt; head -n 12 nums.txt | tail -n 3
### cat nl tac paste join comm split
printf 'a\n\n\nb\n' > g.txt; cat -n g.txt; cat -b g.txt; cat -s g.txt; cat -A g.txt
nl f2.txt; tac f2.txt
paste -d, f1.txt f2.txt; paste -sd+ nums.txt | bc 2>/dev/null || true; paste -s f1.txt
printf '1,ada\n2,bob\n3,cy\n' > users.csv; printf '1,admin\n3,guest\n4,root\n' > roles.csv
join -t, users.csv roles.csv; join -t, -a 1 users.csv roles.csv; join -t, -v 2 users.csv roles.csv
printf 'apple\nbanana\ncherry\n' > s1; printf 'banana\ncherry\ndate\n' > s2
comm s1 s2; comm -12 s1 s2; comm -3 s1 s2; comm -23 s1 s2
split -l 10 nums.txt chunk_; ls chunk_*; wc -l chunk_ab
split -l 12 -d nums.txt part_; ls part_*
### grep
printf 'epoch 1 loss=0.9 val_acc=0.5\nDEBUG noise\nepoch 2 loss=0.7 val_acc=0.6\nERROR boom\nwarning: low\nepoch 3 loss=0.5 val_acc=0.8\nTraceback (most recent call last):\n  File x\nValueError: bad\n' > train.log
grep "loss" train.log; grep -i "error" train.log; grep -n "epoch" train.log; grep -v "DEBUG" train.log | head -2
grep -c "epoch" train.log; grep -w "lr" train.log; echo "rc=$?"; grep -o "loss=[0-9.]*" train.log
grep -E "ERROR|warning" train.log; grep -A 2 -B 1 "Traceback" train.log; grep -F "(most" train.log
grep -oP "loss=\K[0-9.]+" train.log; grep -C1 DEBUG train.log
grep -m 1 epoch train.log; grep -q epoch train.log && echo found; grep -x "DEBUG noise" train.log
grep -h val_acc train.log | sort -t= -k3 -n | tail -1
mkdir -p src/sub .venv; printf 'import torch\n# TODO fix\n' > src/a.py; printf 'import pandas as pd\n' > src/sub/b.py; printf 'TODO\n' > src/notes.txt; printf 'import torch\n' > .venv/x.py
grep -r "import torch" --include="*.py" . | sort; grep -rn "TODO" src/ --exclude-dir=.venv | sort; grep -l "TODO" src/*.py
grep -rl "pandas" src; grep -L torch src/*.py src/sub/*.py; grep -r torch src --include='*.txt'; echo "rc=$?"
echo "a.b[0] x" | grep -F "a.b[0]"; echo "abc" | grep "b*c"; echo "aaa" | grep -o "a\{2\}"
grep nosuchfile_pattern nosuch.txt; echo "rc=$?"; grep -s pat nosuch.txt; echo "rc=$?"
printf 'x\0y\n' > bin.dat; grep x bin.dat; echo "rc=$?"
grep -e loss -e DEBUG train.log | wc -l; echo "epoch" | grep -E "^ep(o|x)ch$"
grep -n -o "val_acc=[0-9.]*" train.log
### sed
printf 'lr: 0.1\nbatch_size: 32\n# comment\nepochs: 10\n' > config.yaml
sed 's/0/X/' config.yaml; sed 's/0/X/g' config.yaml
sed -i 's/lr: 0.1/lr: 0.01/' config.yaml; cat config.yaml
sed -n '2,3p' config.yaml; sed '1d' config.yaml; sed '/^#/d' config.yaml
sed -E 's/([0-9]+)/<\1>/g' config.yaml; sed 's/\(batch\)_\(size\)/\2_\1/' config.yaml
sed -i.bak 's/batch_size: 32/batch_size: 64/' config.yaml; cat config.yaml; cat config.yaml.bak | head -2
sed '$d' config.yaml; sed -n '$p' config.yaml; sed '2q' config.yaml; sed -n '/epochs/=' config.yaml
sed 's/lr/LR/;s/epochs/EPOCHS/' config.yaml; sed -e 's/a/A/' -e 's/b/B/' config.yaml
sed '/batch/a\inserted after' config.yaml; sed '1i\header line' config.yaml; sed '/comment/c\replaced' config.yaml
sed 'y/abc/ABC/' config.yaml; sed -n 'l' <<< $'tab\there'
sed 's/.*/\U&/' config.yaml; sed -E 's/(\w+):/\u\1:/' config.yaml
printf '1\n2\n3\n4\n5\n6\n' | sed -n '1~2p'; printf '1\n2\n3\n4\n5\n6\n' | sed '2,+2d'; printf 'a\nb\nc\n' | sed '0,/b/d'
printf 'a\nb\nc\n' | sed -n '/a/,/b/p'; printf 'a\nb\nc\nd\n' | sed 'N;s/\n/-/'; printf 'x\ny\n' | sed '$!d'
printf 'a b\n' | sed 's/ /\n/'; printf 'aaa\n' | sed 's/a/b/2'; printf 'abc\n' | sed 's/x*/-/g'
printf 'one\ntwo\n' | sed -n 'h;n;G;p'; printf 'l1\nl2\nl3\n' | sed '1!G;h;$!d'
sed 's/unterminated config.yaml; echo "rc=$?"; sed 'k' config.yaml; echo "rc=$?"
### awk
printf 'name score lr\nada 0.91 1e-3\nbob 0.85 3e-4\ncy 0.97 1e-3\n' > res.txt
awk '{print $1}' res.txt; awk '{print $2, $NF}' res.txt; awk 'NR > 1 {sum += $2; n++} END {print sum / n}' res.txt
awk '$2 > 0.9' res.txt; awk 'NR>1 {count[$3]++} END {for (k in count) print k, count[k]}' res.txt | sort
awk -v OFS='\t' '{print $1, $2}' res.txt; awk 'length($0) > 12' res.txt; awk '{printf "%-10s %.3f\n", $1, $2}' res.txt
awk 'BEGIN {print 1/3}'; awk 'BEGIN { printf "%d %5.2f %s %c %x %e\n", 3.9, 2.5, "s", 65, 255, 1234.5 }'
awk -F: '{print NF}' <<< "a:b::c"; awk 'BEGIN{FS=","} {print $2}' <<< "x,y,z"; awk '{$2=""; print}' <<< "a b c"
awk '{ n = split($0, parts, "-"); print n, parts[1], parts[n] }' <<< "a-b-c-d"
awk '{ sub(/o/, "0"); print }' <<< "foo boo"; awk '{ gsub(/o/, "0"); print }' <<< "foo boo"; awk '{ print toupper($0), length() }' <<< "abc"
awk '{ print substr($0, 2, 3), index($0, "c") }' <<< "abcdef"; awk 'BEGIN { if (match("xxabcx", /ab+c/)) print RSTART, RLENGTH }'
awk 'NR==2, NR==3' res.txt; awk '/ada/,/bob/ {print NR}' res.txt; awk 'END {print NR}' res.txt
awk '{ for (i = NF; i > 0; i--) printf "%s ", $i; print "" }' <<< "1 2 3"
awk 'BEGIN { x["a"]=1; x["b"]=2; delete x["a"]; for (k in x) print k; print length(x) }'
awk 'function fact(n) { return n <= 1 ? 1 : n * fact(n - 1) } BEGIN { print fact(10) }'
awk 'BEGIN { s = "a,b;c"; n = split(s, arr, /[,;]/); print n, arr[3] }'
awk '{ $1 = $1; print }' OFS=- <<< "a b c"; awk 'BEGIN { print 2^10, 7%3, -7%3, int(-3.7), int(3.7) }'
awk 'BEGIN { print length("hello"), substr("hello", 0, 2), substr("hello", -1, 3), substr("hello", 4) }'
awk '{ print ($1 == 10), ($1 < 9), ("10" < "9") }' <<< "10"; awk 'BEGIN { x; print (x == 0), (x == "") }'
awk '{ printf "%s|", $0 } END { print "" }' RS=";" <<< "a;b;c"; awk 'NF' <<< $'a\n\nb'
awk 'BEGIN { while (i < 3) { i++; if (i == 2) continue; print i } }'
awk 'BEGIN { for (i = 0; i < 5; i++) { if (i == 3) break; print i } }'
awk 'BEGIN { do { k++ } while (k < 5); print k }'; awk 'BEGIN { print index("abc", "") }'
awk '{ print NR ": " $0 }' f1.txt f2.txt 2>/dev/null || printf 'a\nb\n' | awk '{print NR": "$0}'
awk 'BEGIN { printf "%s %s %s\n", "a", "b" }'; awk 'BEGIN { print -0.0 + 0, 1e6, 1e17, 0.1 + 0.2, 100000 * 100000 }'
awk 'BEGIN { CONVFMT = "%.2f"; x = 3.14159; y = x ""; print y; OFMT = "%.1f"; print x }'
echo "x" | awk '{ print > "out.txt" } END { close("out.txt"); while ((getline line < "out.txt") > 0) print "read:", line }'
awk 'BEGIN { "echo hi" | getline v; print v; system("echo from-system") }'
printf 'b\na\nc\n' | awk '{ print | "sort" }'
awk 'BEGIN { printf "%5s|%-5s|%05d\n", "ab", "cd", 42 }'; awk '{ print $0 > "/dev/stderr" }' <<< "to-stderr" 2>&1
awk 'BEGIN { x = "3x"; print x + 1, x * 2 }'; awk '{ print $1 + $2 }' <<< "1.5 2.25"
awk '{ a[$1] += $2 } END { for (k in a) print k, a[k] }' <<< $'x 1\ny 2\nx 3' | sort
awk -F'\t' '{ print $2 }' <<< $'a\tb\tc'; awk '{ print $(NF-1) }' <<< "a b c"
awk 'BEGIN { getline; print "first:", $0 } { print "rest:", $0 }' <<< $'one\ntwo'
awk '{print}' nosuch.txt; echo "rc=$?"; awk '{print $1' <<< x; echo "rc=$?"
### find xargs
mkdir -p proj/src/pkg proj/.venv/lib proj/build proj/data; touch proj/src/a.py proj/src/pkg/b.py proj/src/pkg/__init__.py proj/README.md proj/readme.txt proj/.venv/lib/x.py proj/data/img.jpg proj/data/model.ckpt
mkdir -p proj/src/__pycache__ proj/src/pkg/__pycache__; touch proj/src/__pycache__/a.pyc proj/src/pkg/__pycache__/b.pyc
find proj -name "*.py" | sort; find proj -iname "*readme*" | sort; find proj -type d -name "__pycache__" | sort
find proj -maxdepth 2 -name "*.ckpt"; find proj -name "*.py" -not -path "*/.venv/*" | sort
find proj -name "*.pyc" -delete; find proj -name "*.pyc"; find proj -name "__pycache__" -type d -exec rm -rf {} +; find proj -name "__pycache__"
printf 'a\nb\nc\n' > proj/src/a.py; find proj/src -name "*.py" -exec wc -l {} + | sort
find proj -type f -size +0 | sort; find proj -empty -type f | sort; find proj -type f -name "*.md" -o -name "*.txt" | sort
find proj -path "*/pkg/*" | sort; find proj/src -mindepth 1 -maxdepth 1 | sort; find proj/data -type f -printf "%f %s\n" | sort
find proj -name "*.py" -exec echo found {} \; | sort; find nosuch; echo "rc=$?"
find proj -type d \( -name .venv -o -name build \) -prune -o -type f -print | sort
seq 1 5 | xargs -I{} echo "run {}"; printf '%s\n' a b c d e | xargs -n 2 echo; echo "a b c" | xargs -t echo 2>&1
printf 'x y\nz\n' | xargs -L 1 echo line; printf 'a\0b c\0' | xargs -0 -n1 echo; echo | xargs echo empty; echo | xargs -r echo never
find proj -name "*.jpg" -print0 | xargs -0 ls; echo "'quoted arg' plain" | xargs -n1 echo
seq 1 3 | xargs -P 2 -I{} sh -c 'echo job {}' | sort; echo "a" | xargs false; echo "rc=$?"
### diff cmp column
printf 'a\nb\nc\nd\n' > v1.txt; printf 'a\nB\nc\nd\ne\n' > v2.txt
diff v1.txt v2.txt; echo "rc=$?"; diff v1.txt v1.txt; echo "rc=$?"
diff -q v1.txt v2.txt; diff -i v1.txt v2.txt
printf 'x\ny\nz\n' > w1; printf 'y\nz\n' > w2; diff w1 w2; diff w2 w1
mkdir -p d1 d2; printf 'same\n' > d1/s; printf 'same\n' > d2/s; printf 'one\n' > d1/only1; printf 'old\n' > d1/c; printf 'new\n' > d2/c
diff -r d1 d2; echo "rc=$?"
cmp v1.txt v2.txt; echo "rc=$?"; cmp v1.txt v1.txt; echo "rc=$?"
printf 'name,score\nada,0.91\nbob,0.85\n' | column -s, -t
printf 'a b c\nlonger x y\n' | column -t
printf 'aa\nbb\ncc\n' | rev; echo hello | fold -w 2; printf 'b\na\n' | tee teed.txt | sort; cat teed.txt
echo -n hello | md5sum; echo -n hello | sha256sum; printf 'hi\n' > h.txt; md5sum h.txt; sha256sum h.txt > sums; sha256sum -c sums
echo hello | base64; echo aGVsbG8K | base64 -d; echo hello | xxd; printf 'hello' | xxd -p; printf 'ab\x01cdefg\n' | strings
printf 'line1\nline2\n' | od -c | head -3
