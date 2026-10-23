### echo and quoting
name="Ada"
echo "$name" "${name}_v2" '$name'
echo 'single "quotes"' "double 'quotes'" \$escaped
echo -n "no newline"; echo
echo -e "tab\there\nnewline" ; echo -E 'raw\t'
printf "%s has %d rows\n" "data.csv" 120
printf "%05.2f\n" 3.14159
printf "[%-6s][%6s]\n" ab cd
printf "%s-%s\n" 1 2 3
printf "%x %o %X %e %g\n" 255 8 255 1234.5 0.0001
printf "%q\n" "a b" 'it'"'"'s'
printf '%b\n' 'x\ty'
### parameter expansion
s="experiments/run_042/model.pt"
echo "${#s}"
echo "${s:0:11}"
echo "${s#*/}"
echo "${s##*/}"
echo "${s%/*}"
echo "${s%.pt}.onnx"
echo "${s/run/RUN}" "${s//_/-}" "${s^^}" "${s,,}"
echo "${s: -3}" "${s:12}" "${s:4:-3}"
t="hello"; echo "${t^}" "${t^^[lo]}"
unset u; echo "[${u:-default}]" "[${u-dflt}]" "[${u:+alt}]"
v=""; echo "[${v:-default}]" "[${v-dflt}]" "[${v:+alt}]"
: "${OUT_DIR:=./out}"; echo "$OUT_DIR"
ref=name; name=Grace; echo "${!ref}"
echo "${s/#exp/EXP}" "${s/%pt/PT}"
### default errors
echo "${REQUIRED:?must be set}"
echo after
### arrays
arr=(alpha beta gamma); arr+=(delta)
echo "${arr[0]}" "${arr[-1]}" "${arr[@]}" "${#arr[@]}" "${!arr[@]}" "${arr[@]:1:2}"
for x in "${arr[@]}"; do echo "$x"; done
arr[10]=ten; echo "${!arr[@]}"; unset 'arr[1]'; echo "${arr[@]}" "${#arr[@]}"
declare -A lr=([small]=1e-3 [large]=3e-4)
echo "${lr[small]}"; for k in "${!lr[@]}"; do echo "$k=${lr[$k]}"; done
declare -p lr arr
b=("a b" c); printf '<%s>\n' "${b[@]}" ${b[@]} "${b[*]}"
IFS=,; echo "${b[*]}"; unset IFS
mapfile -t lines <<< $'l1\nl2\nl3'; echo "${#lines[@]} ${lines[1]}"
### arithmetic
echo $((3 + 4 * 2)) $((17 / 5)) $((17 % 5)) $((2 ** 10)) $((-17 / 5)) $((-17 % 5))
i=0; ((i++)); ((i += 5)); echo "$i"
echo $(( i > 3 ? 100 : 200 )) $(( 1 << 4 )) $(( 0x1f )) $(( 010 )) $(( 2#101 ))
(( 5 > 3 )) && echo yes
n=7; if (( n > 5 )); then echo big; fi
let "x = 2 * 3" "y = x + 1"; echo $x $y
declare -i z=4; z+=3; echo $z; z="z * 2"; echo $z
echo $(( 1 / 0 ))
echo after
### brace expansion
echo {1..5} {a..e} {0..10..2} {01..03}
echo pre{a,b,c}post {x,y}{1,2}
echo {5..1} {e..a} {-2..2}
echo a{b,c{d,e}}f
echo {a..c}{1..2}
echo {single} {} {a,}
### word splitting and globbing
touch a.csv b.csv c.txt data_01.csv data_02.csv .hidden
ls *.csv; echo data_??.csv; echo [ab]*; echo *.nomatch
x="a   b  c"; set -- $x; echo $#; set -- "$x"; echo $#
IFS=: read -r p q r <<< "1:2:3"; echo "$p|$q|$r"
y="one,two,,three"; IFS=,; for f in $y; do echo "[$f]"; done; unset IFS
shopt -s nullglob; echo *.nomatch; shopt -u nullglob
mkdir -p d1/d2; touch d1/x.py d1/d2/y.py; shopt -s globstar; echo **/*.py; echo d1/**
echo .*
### conditionals
for p in a.txt nope dir; do
  if [[ -f $p ]]; then echo "$p file"; elif [[ -d $p ]]; then echo "$p dir"; else echo "$p missing"; fi
done
touch a.txt; mkdir -p dir
[[ -f a.txt && ! -f b.txt ]] || echo "no"
[[ -f a.txt && ! -f b.txt ]] && echo "yes"
f=data.csv; [[ $f == *.csv ]] && echo glob-match
s=run_042; [[ $s =~ ^run_([0-9]+)$ ]] && echo "${BASH_REMATCH[1]}"
[[ "abc" < "abd" ]] && echo less
[ 3 -lt 10 ] && echo num-lt
[ "a" = "a" -a 1 -eq 1 ] && echo and
test -z "" && echo empty
[[ -n $undefined ]] || echo unset-empty
[ abc -gt 1 ]; echo "rc=$?"
[[ 2+3 -eq 5 ]] && echo arith-in-cond
### case
for a in train eval test other; do
case "$a" in
  train) echo "training" ;;
  eval|test) echo "evaluating" ;;
  *) echo "usage: $0 {train|eval}" ;;
esac
done
case abc in a*) echo first ;& b*) echo fall ;; *) echo no ;; esac
case x in [a-z]) echo lower;; esac
### loops
for seed in 0 1 2; do echo "seed $seed"; done
for ((i = 0; i < 5; i++)); do echo "$i"; done
printf 'l1\nl2 x\n' > file.txt
while read -r line; do echo "$line"; done < file.txt
printf 'id,label\n1,cat\n2,dog\n' > labels.csv
while IFS=, read -r id label; do echo "$id -> $label"; done < <(tail -n +2 labels.csv)
for i in {1..10}; do [[ $i -eq 3 ]] && continue; [[ $i -eq 6 ]] && break; echo "$i"; done
n=0; until (( n >= 3 )); do echo "n=$n"; ((n++)); done
for i in 1 2; do for j in a b; do [[ $j == b ]] && continue 2; echo "$i$j"; done; done
for i in 1 2 3; do for j in 1 2 3; do (( j == 2 )) && break 2; echo "$i.$j"; done; done
for w in $(echo one two); do echo "w=$w"; done
### functions
usage() { echo "usage: $0 [-e epochs]" >&2; return 1; }
log() { printf '[%s] %s\n' "LOG" "$*"; }
log start "two words"; usage; echo "rc=$?"
counter() { local count=$1; count=$((count + 1)); echo "$count"; }
count=10; counter 5; echo "$count"
outer() { local v=outer; inner; }; inner() { echo "inner sees $v"; }; outer
fact() { (( $1 <= 1 )) && { echo 1; return; }; local r; r=$(fact $(( $1 - 1 ))); echo $(( $1 * r )); }
fact 6
function kw { echo "keyword function $1"; }; kw x
args() { echo "$#: $@"; shift; echo "after shift: $*"; }; args a "b c" d
type log | head -3
### special variables
set -- a b c
echo "$#" "$1" "$@" "${@:2}" "$*"
echo "$0" | sed 's/.*\///'
false; echo "$?"; true; echo "$?"
echo "$$" | grep -q '^[0-9]*$' && echo pid-ok
sleep 0 & wait $!; echo "bg-status $?"
echo "$-"
f() { echo "${FUNCNAME[0]} line $LINENO"; }; f
### command substitution and redirection
today=$(echo 2026-10-20); n=$(printf 'a\nb\nc\n' | wc -l); echo "$today $n"
echo "out"; echo "err" >&2
echo stdout > out.txt 2> err.txt; cat out.txt
ls nonexistent > all.txt 2>&1; cat all.txt
ls nonexistent &> all2.txt; cat all2.txt
echo a >> app.txt; echo b >> app.txt; cat app.txt
cat < app.txt | wc -l
cat > config.yaml <<EOF
lr: 0.001
batch_size: ${BATCH:-32}
EOF
cat config.yaml
cat > script.sh <<'EOF'
echo "quoted delimiter: $HOME is NOT expanded here"
EOF
cat script.sh
text="xyz"; grep "x" <<< "$text"
printf 'b\na\n' > a.txt; printf 'a\nc\n' > b.txt
diff <(sort a.txt) <(sort b.txt)
comm -12 <(sort a.txt) <(sort b.txt)
{ echo one; echo two; } > grouped.txt; cat grouped.txt
(cd /tmp && pwd); pwd | sed 's/.*\///'
echo $(echo "  spaced   out  ")
echo "$(printf 'trailing\n\n\n')|"
exec 3> fd3.txt; echo via-fd3 >&3; exec 3>&-; cat fd3.txt
cat <<- TABS
	indented heredoc
	TABS
### pipefail and errexit
false | true; echo "plain $?"
set -o pipefail; false | true; echo "pipefail $?"; set +o pipefail
true | false | true; echo "${PIPESTATUS[@]}"
set -e
false || echo "or-ok"
if false; then :; fi
echo "still running"
false
echo "not reached"
### nounset
set -u
echo "${maybe:-ok}"
echo "$nope"
echo "not reached"
### traps
cleanup() { echo "cleanup ran"; }
trap cleanup EXIT
trap 'echo "ERR at $LINENO"' ERR
false
echo body
### getopts
parse() {
  local OPTIND=1 opt epochs=10 gpu=0
  while getopts "e:g:h" opt; do
    case $opt in
      e) epochs=$OPTARG ;;
      g) gpu=$OPTARG ;;
      *) echo "bad option"; return 1 ;;
    esac
  done
  shift $((OPTIND - 1))
  echo "epochs=$epochs gpu=$gpu rest=$*"
}
parse -e 5 -g1 config.yaml
parse config.yaml
parse -x 2>/dev/null
parse -e 2>&1
### read builtin
read -r a b <<< "  one   two three  "; echo "[$a][$b]"
read -r <<< "  x y  "; echo "[$REPLY]"
IFS=, read -r p q <<< "1,2,3"; echo "[$p][$q]"
read -a parts <<< "a b c"; echo "${#parts[@]} ${parts[2]}"
read -r -d ';' first <<< "one;two"; echo "$first"
read -n 3 three <<< "abcdef"; echo "$three"
printf 'back\\slash\n' | { read -r raw; echo "$raw"; }
printf 'back\\slash\n' | { read cooked; echo "$cooked"; }
echo "x y" | while read -r one two; do echo "$two $one"; done
read nothing < /dev/null; echo "eof rc=$?"
### local and declare
x=global
f() { local x=local; echo "in f: $x"; g; }
g() { echo "in g: $x"; }
f; echo "after: $x"
declare -r ro=1; ro=2; echo "ro=$ro"
declare -l lower=MiXeD; declare -u upper=MiXeD; echo "$lower $upper"
declare -x EXPORTED=yes; bash -c 'echo "child sees $EXPORTED"'
export WANDB_PROJECT=demo; bash -c 'echo "$WANDB_PROJECT"'
NOT_EXPORTED=1; bash -c 'echo "[$NOT_EXPORTED]"'
VAR=temp bash -c 'echo "temp=$VAR"'; echo "after=[${VAR-}]"
readonly PI=3.14; unset PI; echo "$PI"
### eval source and subshells
cmd='echo evaluated'; eval "$cmd"
x=1; (x=2; echo "sub $x"); echo "main $x"
echo 'MSG="from env.sh"' > env.sh; source env.sh; echo "$MSG"
printf 'A=1\nB=two\n' > .env; set -a; source .env; set +a; bash -c 'echo "$A $B"'
v=$(exit 3); echo "cmdsub status $?"
( exit 7 ); echo "subshell status $?"
bash -c 'exit 5'; echo "child $?"
### printf -v and misc builtins
printf -v pad "%03d" 7; echo "$pad"
type cd echo ls if 2>&1
command -v ls; command -v cd
type -t cd if ls nosuch
hash -r; echo hashed
pushd /tmp > /dev/null; echo "$(pwd)"; popd > /dev/null; pwd | sed 's/.*\///'
alias ll='ls -l'; alias ll
umask; umask 077; umask; umask -S
echo $(( RANDOM >= 0 && RANDOM < 32768 ))
### string tests and edge cases
x="  padded  "; echo "[${x}]" "[${x// /}]"
path=/data/runs/exp7/ckpt_best.pt
echo "${path##*/}" "${path%/*}" "$(basename "$path" .pt)" "$(dirname "$path")"
file="ckpt_best.pt"; echo "${file%.*}" "${file##*.}"
echo "${path//\//:}"
str=abcabc; echo "${str/b/X}" "${str//b/X}" "${str/#a/S}" "${str/%c/E}"
echo "${#}"; set -- 1 2 3; echo "${#}"
list="a b c"; echo ${list// /,}
