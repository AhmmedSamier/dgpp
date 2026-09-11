# Shared site configuration. Source this before reading DGPP_* settings.
# .env is parsed as data by site_env.py, never sourced as shell code.
# Exported settings override .env. Credentials are not loaded.

_DGPP_SITE_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/site_env.py"

dgpp_load_site() {
  local site_file key value
  site_file=$(mktemp) || return 1
  if ! python3 "$_DGPP_SITE_HELPER" shell > "$site_file"; then
    rm -f "$site_file"
    return 1
  fi
  while IFS= read -r -d '' key && IFS= read -r -d '' value; do
    export "$key=$value"
  done < "$site_file"
  rm -f "$site_file"
}

dgpp_load_site || return 1
dgpp_config() { python3 "$_DGPP_SITE_HELPER" config; }
dgpp_nodes() { python3 "$_DGPP_SITE_HELPER" nodes "$@"; }
dgpp_head() { python3 "$_DGPP_SITE_HELPER" head; }
dgpp_client_host() { python3 "$_DGPP_SITE_HELPER" client-host; }
dgpp_rank_prefix() { python3 "$_DGPP_SITE_HELPER" rank-prefix --rank "$1"; }
dgpp_run_rank() { local rank=$1; shift; python3 "$_DGPP_SITE_HELPER" run-rank --rank "$rank" -- "$@"; }
dgpp_require_world() { python3 "$_DGPP_SITE_HELPER" require-world --world "$1"; }
dgpp_model() { python3 "$_DGPP_SITE_HELPER" model; }
dgpp_log_dir() { python3 "$_DGPP_SITE_HELPER" log-dir; }
dgpp_stage_dir() { python3 "$_DGPP_SITE_HELPER" stage-dir; }
dgpp_served_model() { python3 "$(dirname "$_DGPP_SITE_HELPER")/serve_client.py" model; }
dgpp_peers() { python3 "$_DGPP_SITE_HELPER" peers "$@"; }
dgpp_ssh_user() { python3 "$_DGPP_SITE_HELPER" user; }
dgpp_http_port() { python3 "$_DGPP_SITE_HELPER" http-port; }
dgpp_resolve_config() { python3 "$_DGPP_SITE_HELPER" resolve "$@"; }
