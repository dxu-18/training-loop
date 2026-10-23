### jq basics
echo '{"items":[{"name":"a","score":0.95,"tags":["x"]},{"name":"b","score":0.5,"tags":[]}],"n":2,"meta":null}' > data.json
jq '.' data.json; jq '.items[] | {name, score}' data.json; jq -r '.items[].name' data.json
jq '.items | map(select(.score > 0.9))' data.json; jq -c '.items[]' data.json; jq 'length' data.json
printf '{"a":1}\n{"a":2}\n' > rows.jsonl; jq -s '.' rows.jsonl; jq -c '.[]' <<< '[1,[2],{"x":3}]'
jq '.items[0].name, .n' data.json; jq '[.items[] | .score] | add / length' data.json
jq -r '.items[] | "\(.name): \(.score)"' data.json; jq -r '.items[] | [.name, .score] | @csv' data.json
jq -r '.items[] | [.name, .score] | @tsv' data.json
jq '.items |= map(.score *= 2) | .items[].score' data.json; jq 'del(.items) | keys' data.json
jq --arg who b '.items[] | select(.name == $who) | .score' data.json; jq --argjson k 2 '.n * $k' data.json
jq '.meta // "none"' data.json; jq '.items[] | .tags | length' data.json; jq 'keys' data.json; jq 'has("n")' data.json
jq '[.items[].name] | join(",")' data.json; jq '.items | sort_by(.score) | .[0].name' data.json
jq '.items | group_by(.score > 0.7) | length' data.json; jq '[.items[] | .score] | max' data.json
jq 'to_entries | map(.key)' data.json; jq '.items[] | select(.name | test("^a"))' data.json
jq -n '[range(5)] | map(. * 2)'; jq -n '{"a": 1} + {"b": 2}'; jq -n '"abc" | ascii_upcase'; jq -n '[1,2,3] | reverse'
jq -n '[3,1,2] | sort | first, last'; jq -n '"a,b,c" | split(",")'; jq -n '[1,[2,[3]]] | flatten'; jq -n '{a:1} | .b = 2'
jq -n 'reduce range(1;5) as $i (0; . + $i)'; jq -n '[limit(3; range(10))]'; jq -n '"x" * 3'; jq -n '1 / 3'
jq -n '[.[]?]'; jq -n 'if 1 > 2 then "a" elif 2 > 1 then "b" else "c" end'; jq -n '[1,2,3] | map(select(. != 2))'
jq -n '{"k": [1,2]} | .k[1]'; jq -n '[1,2,3] | .[1:]'; jq -n '"hello" | .[1:3]'; jq -n 'null | length'
jq -n '[{"a":1,"b":2},{"a":3}] | map(.a) | add'; jq -n '"2" | tonumber + 1'; jq -n '42 | tostring | length'
jq -n '[1,2,2,3] | unique'; jq -n '{"a":{"b":1}} | paths' | jq -sc .; jq -n '[1,2] | any(. > 1), all(. > 1)'
jq -n '"test 123" | capture("(?<word>[a-z]+) (?<num>[0-9]+)")'; jq -n '"aXbXc" | gsub("X"; "-")'
jq -n 'try error("bad") catch .'; jq -n '[.[] | numbers]' <<< 'null'; jq -c -n '[{"x":1},{"x":2}] | INDEX(.x)' 2>/dev/null || true
jq '.nosuch.deeper' data.json; jq '.items[5]' data.json; jq -e '.meta' data.json; echo "rc=$?"
jq '.n | .x' data.json; echo "rc=$?"; echo 'not json' | jq .; echo "rc=$?"; jq '.[' data.json; echo "rc=$?"
### bc
echo "scale=4; 1/3" | bc; echo "2^10" | bc; echo "17/5; 17%5" | bc; seq 1 10 | paste -sd+ | bc
echo "scale=2; 22/7" | bc; echo "1.5 * 2.25" | bc; echo "3 > 2; 2 == 3" | bc; echo "x = 5; x * x" | bc
echo "scale=3; sqrt(2)" | bc; echo "-7 / 2" | bc; echo "scale=1; -0.25 * 2" | bc; echo "0.1 + 0.2" | bc
echo "define sq(x) { return x * x; }; sq(12)" | bc; echo "for (i = 1; i <= 3; i++) i" | bc
echo "obase=2; 10" | bc; echo "ibase=16; FF" | bc; echo "1/3" | bc -l; echo "s(0); c(0)" | bc -l
echo "1/0" | bc; echo "rc=$?"
