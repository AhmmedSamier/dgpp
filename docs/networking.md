# Network and cache configuration

`DGPP_NODES` identifies hosts in rank order; it is separate from RDMA device
selection. Rank 0 runs locally. Peers must resolve and reach its address,
and SSH must use the selected `DGPP_SSH_USER` without an interactive prompt.
Keep fabric/journal TCP ports restricted to cluster members. The defaults
are 29970 and 29971; set them in `.env`. These internal protocols have no
authentication and must not be exposed to an untrusted network.

## RoCE lanes

For multi-node runs, set `DGPP_ROCE_DEVICES` to a space-separated ordered
list of local verbs devices. Discover the actual names on your machines:

```bash
python3 scripts/discover_roce.py --config deploy/cluster.json
```

Run this on rank 0 after setting node addresses and SSH access. Without
`--config`, it inspects only the local machine. It prints the verbs-to-Linux
interface mapping, addresses, MTUs, usable GID indices, and candidate `.env`
settings without changing files or networking. On Spark, names can look like
`"rocep1s0f0 roceP2p1s0f0"`; use the discovered values for your hardware.
Match corresponding lanes by subnet before copying the suggestions. If
several GIDs or more than two devices are usable, choose the intended fabric.

If `DGPP_ROCE_DEVICES` is omitted,
DGPP discovers active Ethernet RDMA devices and sorts their names. Explicit
selection is preferable on hosts with several fabrics. The native transport
uses port 1 of each selected device; multi-port HCA selection is not supported.

Use one or two lanes, with the same count and corresponding network order on each rank. Device
names need not match between hosts. Without explicit GID indices, the verbs
layer selects a non-link-local RoCE-v2 GID. `DGPP_ROCE_GID_INDICES="3 3"`
pins one index per explicitly named device. Check link state, IP/GID
assignment, routing and consistent MTU on the selected network interfaces.
Discovery does not prove that corresponding lanes can exchange RDMA traffic.

Read-only starting points are `rdma link`, `ip -br address`, `ip link`, and
`scripts/dgpp-cluster doctor --config deploy/cluster.json`.
Only run the fabric probes on an idle test allocation;
they open QPs, use GPU memory and generate traffic.

## Per-node differences

Shared defaults and overrides live in `.env`, not each deployment JSON:

```dotenv
DGPP_ROCE_DEVICES="rocep1s0f0 roceP2p1s0f0"
HF_HUB_CACHE="~/models/hub"
DGPP_RESIDENT_CACHE_DIR="~/dgpp/resident-cache"
DGPP_NODE_OVERRIDES='{"192.0.2.12":{"DGPP_ROCE_DEVICES":"rocep1s0f0 roceP2p1s0f0","HF_HUB_CACHE":"/srv/models/hub"}}'
```

Override keys must match a host in `DGPP_NODES` exactly. Only RoCE devices,
GID indices, HF hub cache and resident cache directory are accepted in each
node override. `~/` expands on the destination host; `$VARIABLE` substitution
and shell commands are not supported. `HF_HUB_CACHE` takes precedence over
`HF_HOME/hub`. The launcher forwards only these allowlisted site values, not
`.env` or its credentials. Direct native programs do not parse `.env`; use
the wrappers, a resolved config, or explicitly export their settings.

## HTTP exposure

HTTP defaults to `127.0.0.1:18080`. Set `http.bind_host` and `http.port` in
the deployment JSON to override site defaults. A specific LAN IPv4 address
restricts the listening interface; `0.0.0.0` accepts traffic on every IPv4
interface. The server does not enforce API keys and does not implement TLS.

For individual remote access, use an SSH tunnel. For shared access, keep
DGPP on localhost behind an authenticated TLS reverse proxy. A starting
configuration is [deploy/nginx.example.conf](../deploy/nginx.example.conf).
Replace its hostname, certificate paths and password file before installation;
it is not installed or enabled automatically. Disable proxy response buffering
so SSE tokens arrive as produced, and set timeouts for long prefill requests.
See the NGINX documentation for [proxy buffering](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering)
and [basic authentication](https://nginx.org/en/docs/http/ngx_http_auth_basic_module.html).

The bundled diagnostic clients use plain HTTP to a host/port. Run them on
rank 0 or through a tunnel; they do not implement proxy authentication or TLS.
