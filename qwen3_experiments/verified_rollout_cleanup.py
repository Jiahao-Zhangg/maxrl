"""Remove closed training rollouts only after matching a public Hub commit."""

import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def persist(config, name, payload):
    saved = False
    for directory in config['receipt_directories']:
        try:
            folder = Path(directory)
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / name
            temporary = target.with_name(f'.{name}.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(payload, indent=2) + '\n')
            temporary.replace(target)
            saved = True
        except OSError as exc:
            if exc.errno not in (errno.ENOSPC, errno.EDQUOT, errno.EIO):
                raise
    return saved


def cleanup(config, api):
    source = Path(config['rollout_directory'])
    manifest = read(source / 'rollout_manifest.json')
    if manifest['num_steps'] != config['steps'] or manifest['num_rollouts'] != config['rollouts']:
        raise ValueError('Training rollout count is incomplete')
    info = api.repo_info(config['repo_id'], repo_type='dataset', revision=config.get('revision'), files_metadata=True)
    if info.private or not info.sha:
        raise ValueError('Expected a public immutable dataset commit')
    remote = {item.rfilename: item for item in info.siblings}
    records = []
    allowed = [Path(path).resolve() for path in config['allowed_data_roots']]
    for step, metadata in manifest['steps'].items():
        relative = metadata['file']
        path = source / relative
        if '..' in Path(relative).parts or not relative.startswith('data/'):
            raise ValueError('Invalid rollout shard path')
        if relative not in remote:
            raise ValueError(f'Rollout is absent from Hub: {relative}')
        if not path.exists():
            continue
        real = path.resolve()
        if not any(real.is_relative_to(folder) for folder in allowed):
            raise ValueError('Rollout symlink is outside the authorized data roots')
        stat = real.stat()
        if stat.st_uid != os.getuid() or stat.st_size != remote[relative].size:
            raise ValueError('Rollout ownership or size mismatch')
        sha256 = checksum(real)
        item = remote[relative]
        if item.lfs:
            if sha256 != item.lfs.sha256:
                raise ValueError(f'Rollout SHA256 mismatch: {relative}')
        else:
            sha1 = hashlib.sha1(f'blob {stat.st_size}\0'.encode() + real.read_bytes()).hexdigest()
            if sha1 != item.blob_id:
                raise ValueError(f'Rollout Git hash mismatch: {relative}')
        records.append({'step': step, 'relative': relative, 'path': str(path), 'target': str(real),
                        'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino,
                        'sha256': sha256})
    receipt = {'state': 'verified_before_deletion', 'repo_id': config['repo_id'], 'revision': info.sha,
               'files': records, 'bytes': sum(r['bytes'] for r in records), 'time': time.time()}
    if not persist(config, 'cleanup_receipt.json', receipt):
        raise OSError(errno.ENOSPC, 'Cannot retain a cleanup receipt')
    for record in records:
        path, real = Path(record['path']), Path(record['target'])
        stat = real.stat()
        if ((stat.st_size, stat.st_mtime_ns, stat.st_ino) !=
                (record['bytes'], record['mtime_ns'], record['inode']) or path.resolve() != real):
            raise ValueError('Rollout changed after verification')
        real.unlink()
        if path.is_symlink():
            path.unlink()
    receipt.update(state='uploaded_verified_and_deleted', finished_at=time.time())
    persist(config, 'cleanup_receipt.json', receipt)
    return receipt


def ready(config):
    path = Path(config['success_record'])
    if not path.exists():
        return False
    record = read(path)
    return all(record.get(key) == value for key, value in config['success_fields'].items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    if config.get('node') and config['node'] != socket.gethostname().split('.')[0]:
        raise ValueError('This cleanup must run on its assigned compute node')
    from huggingface_hub import HfApi
    api = HfApi(token=False)
    lock_dir = Path(config['receipt_directories'][0])
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / 'cleanup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = lock_dir / 'cleanup_receipt.json'
        if existing.exists() and read(existing).get('state') == 'uploaded_verified_and_deleted':
            return 0
        while True:
            try:
                if ready(config):
                    result = cleanup(config, api)
                    print(json.dumps({'state': result['state'], 'deleted_bytes': result['bytes']}), flush=True)
                    return 0
                persist(config, 'status.json', {'state': 'waiting_for_verified_training_completion',
                                                'pid': os.getpid(), 'updated_at': time.time()})
                if not args.watch:
                    return 75
            except Exception as exc:
                persist(config, 'status.json', {'state': 'retrying', 'error': str(exc),
                                                'pid': os.getpid(), 'updated_at': time.time()})
                if not args.watch:
                    raise
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
