#!/bin/bash
# M6 6f on the fabric: tool calls and reasoning through the four-node
# service — the M6 exit criterion "streamed multi-turn chat and tool calls
# work on TP=4", as a reproducible procedure. Boots the world through
# scripts/serve_run.sh (the recommended serving mode unless
# DGPP_SERVE_KNOBS says otherwise), sends twelve requests — plain chat, a
# two-tool question, tool_choice none / required / named, a follow-up
# turn carrying the tool result, reasoning_effort low, and the streamed
# tool call — prints what came back (finish_reason, reasoning_content,
# content, the parsed calls; the stream's chunk sequence), then stops the
# world and prints the four-way op-stream md5 and the pace. Since M6 6g/6h
# the constrained shapes ride too: parallel_tool_calls false, required +
# single, and response_format json_object / json_schema (the content is
# parsed and checked against the schema here).
#
# Usage: scripts/serve_tools_check.sh [OUT_DIR]
#   OUT_DIR defaults to build-ci/fabric-runs/tools_gate_<date>; the
#   responses land there as <name>.json / stream.sse beside the serve logs.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/scripts/cluster_env.sh" || exit 1
OUT=${1:-$ROOT/build-ci/fabric-runs/tools_gate_$(date +%Y-%m-%d_%H%M%S)}
export DGPP_SERVE_LOG=$OUT/serve
export DGPP_SERVE_KNOBS="${DGPP_SERVE_KNOBS:---max-concurrency 2 --kv-capacity 4096 --default-max-tokens 512 --queue-limit 8 --decode-graph --mtp --seed 20260904}"
HOST=$(dgpp_client_host) || exit 1
mkdir -p "$DGPP_SERVE_LOG"
cd "$ROOT" || exit 1
scripts/serve_run.sh up || exit 1
sleep 2
URL=http://$HOST:$(dgpp_http_port)/v1/chat/completions
M=$(dgpp_served_model) || exit 1
TOOLS='"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]},"days":{"type":"integer","description":"Forecast days"}},"required":["city"]}}},{"type":"function","function":{"name":"get_time","description":"Current local time in a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

post() {  # post NAME BODY — one-shot request, summarized
  curl -s --max-time 600 "$URL" -H 'Content-Type: application/json' -d "$2" > "$OUT/$1.json"
  echo "--- $1"
  python3 "$ROOT/scripts/serve_response_report.py" "$OUT/$1.json"
}

post plain '{"model":"'$M'","messages":[{"role":"user","content":"In one sentence, what is the capital of France?"}],"max_tokens":200}'
post tools_auto '{"model":"'$M'","messages":[{"role":"user","content":"What is the weather in Paris right now, in celsius? Also what time is it there?"}],'"$TOOLS"',"max_tokens":512}'
post tools_none '{"model":"'$M'","messages":[{"role":"user","content":"What is the weather in Paris right now?"}],'"$TOOLS"',"tool_choice":"none","max_tokens":200}'
post tools_required '{"model":"'$M'","messages":[{"role":"user","content":"Tell me about Tokyo."}],'"$TOOLS"',"tool_choice":"required","max_tokens":512}'
post tools_named '{"model":"'$M'","messages":[{"role":"user","content":"I am travelling to Oslo next week for 3 days."}],'"$TOOLS"',"tool_choice":{"type":"function","function":{"name":"get_weather"}},"max_tokens":512}'
post tools_single '{"model":"'$M'","messages":[{"role":"user","content":"What is the weather in Paris right now, in celsius? Also what time is it there?"}],'"$TOOLS"',"parallel_tool_calls":false,"max_tokens":512}'
post tools_required_single '{"model":"'$M'","messages":[{"role":"user","content":"Tell me about Tokyo and what time it is there."}],'"$TOOLS"',"tool_choice":"required","parallel_tool_calls":false,"max_tokens":512}'
post tools_followup '{"model":"'$M'","messages":[{"role":"user","content":"What is the weather in Paris right now, in celsius?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\", \"unit\": \"celsius\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"{\"temperature\": 18, \"condition\": \"partly cloudy\"}"}],'"$TOOLS"',"max_tokens":300}'
post effort_low '{"model":"'$M'","messages":[{"role":"user","content":"What is 17*23?"}],"reasoning_effort":"low","max_tokens":300}'
post json_object '{"model":"'$M'","messages":[{"role":"user","content":"Give me three facts about Lisbon as JSON with keys city, country and facts (a list of strings)."}],"response_format":{"type":"json_object"},"max_tokens":512}'
post json_schema '{"model":"'$M'","messages":[{"role":"user","content":"Give a short structured description of Kyoto: its country, population and up to three landmarks."}],"response_format":{"type":"json_schema","json_schema":{"name":"place","strict":true,"schema":{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"},"population":{"type":"integer"},"landmarks":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":3}},"required":["city","country","population","landmarks"],"additionalProperties":false}}},"max_tokens":1024}'

# The stream: raw SSE saved, then summarized as a chunk-kind sequence.
curl -s --max-time 600 -N "$URL" -H 'Content-Type: application/json' -d '{"model":"'$M'","messages":[{"role":"user","content":"What is the weather in Paris right now, in celsius?"}],'"$TOOLS"',"stream":true,"stream_options":{"include_usage":true},"max_tokens":512}' > "$OUT/stream.sse"
echo "--- stream"
python3 "$ROOT/scripts/serve_streams.py" tools "$OUT/stream.sse"

scripts/serve_run.sh down
md5sum "$DGPP_SERVE_LOG"/serve_rank*.ops
grep -h "sampling summary\|tool calls" "$DGPP_SERVE_LOG/serve_r0.log" | head -5
python3 scripts/serve_pace.py "$DGPP_SERVE_LOG/serve_r0.log" 2>/dev/null
echo "=== tools check done: $OUT"
