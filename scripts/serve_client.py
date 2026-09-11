"""Endpoint defaults and model discovery shared by serving clients."""
import argparse
import http.client
import json
from pathlib import Path
import sys

from site_env import config_path, default_host, http_port, run_paths, settings


def default_url():
    return f"http://{default_host()}:{http_port()}"


def default_log():
    import os
    return str(Path(run_paths(config_path(), log_dir=os.environ.get("DGPP_SERVE_LOG"))["log_dir"]).expanduser() / "serve_r0.log")


def served_model(host, port):
    connection = http.client.HTTPConnection(host, port, timeout=30)
    try:
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"model discovery at {host}:{port} returned HTTP {response.status}")
        data = json.loads(response.read())
        models = data.get("data", [])
        if not models or not isinstance(models[0].get("id"), str):
            raise ValueError(f"no model ID advertised at {host}:{port}")
        return models[0]["id"]
    finally:
        connection.close()


def model_at_url(url):
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("this diagnostic client expects an http:// endpoint")
    return served_model(parsed.hostname, parsed.port or 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("model", "url", "log"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.command == "model":
        print(served_model(args.host or default_host(), args.port or http_port()))
    elif args.command == "url":
        print(default_url())
    else:
        print(default_log())


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"service client: {error}", file=sys.stderr)
        raise SystemExit(2)
