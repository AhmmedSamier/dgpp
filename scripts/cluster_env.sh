# The site's fabric, derived from its cluster config (2026-09-10). Sourced by
# the evidence and serving scripts so that NO node address and NO login name
# is ever written into a script: both live in the site-local cluster config
# (deploy/cluster.json, git-ignored; copy deploy/cluster.example.json), and
# the environment still overrides per run.
#
#   . "$ROOT/scripts/cluster_env.sh"
#   HOST=${DGPP_SERVE_HOST:-$(dgpp_head)}      # rank 0, else localhost
#   PEERS=($(dgpp_peers))                      # ranks 1.., space separated
#   NODES=$(dgpp_nodes)                        # every node, rank order
#   USER_=$(dgpp_ssh_user)                     # the config's, else the caller's
#
# DGPP_CLUSTER_CONFIG names another config. A missing or unreadable file
# leaves the head at localhost and the peer list empty — callers report that
# rather than guessing an address.

dgpp_config() {
  if [ -n "${DGPP_CLUSTER_CONFIG:-}" ]; then
    echo "$DGPP_CLUSTER_CONFIG"
  else
    echo "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/deploy/cluster.json"
  fi
}

dgpp_nodes() {
  jq -r '.nodes[]?' "$(dgpp_config)" 2>/dev/null | tr '\n' ' ' | sed 's/ $//'
}

dgpp_head() {
  local h
  h=$(jq -r '.nodes[0] // empty' "$(dgpp_config)" 2>/dev/null)
  echo "${h:-127.0.0.1}"
}

dgpp_peers() {
  jq -r '.nodes[1:][]?' "$(dgpp_config)" 2>/dev/null | tr '\n' ' ' | sed 's/ $//'
}

dgpp_ssh_user() {
  local u
  u=$(jq -r '.ssh_user // empty' "$(dgpp_config)" 2>/dev/null)
  echo "${u:-$(id -un)}"
}
