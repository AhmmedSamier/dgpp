"""Download once, then finish all independent peer copies before deployment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from cache_sync import sync_snapshot
from cluster_doctor import cache_root, cached_snapshot
from download_model import main as download_main
from site_env import config_argument, resolve_config, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=config_argument, required=True)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    command = ['--config', str(args.config), '--revision', args.revision, '--local-only']
    if args.activate:
        command.append('--activate')
    download_main(command)
    config = resolve_config(args.config, settings())
    root = cache_root({**os.environ, **config['node_env'][0]})
    models = [config['model']]
    table = config.get('engine', {}).get('ngram_table_model')
    if table:
        models.append(table)

    def copy_peer(rank):
        target = config['ssh_user'] + '@' + config['nodes'][rank]
        for model in models:
            sync_snapshot(cached_snapshot(model, root), target, config['node_env'][rank])

    # Exiting the context waits for every worker, including after a failed
    # copy. The caller must not restore production while copies still run.
    peers = range(1, len(config['nodes']))
    with ThreadPoolExecutor(max_workers=max(1, len(config['nodes']) - 1)) as pool:
        futures = [pool.submit(copy_peer, rank) for rank in peers]
        for future in futures:
            future.result()
    print('All peer snapshots copied and verified; transfer workers drained.', flush=True)


if __name__ == '__main__':
    main()
