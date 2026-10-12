"""Inventory, stream/export and verify scientific assets without rewriting history.

Run export on the scientific server, not on the small local control disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import re
import shutil
import sys
import tarfile

# Only the completed Core12 experiment lineage; optimization diagnostics excluded.
ASSET_ROOTS = (
    'runs/agent-memory', 'runs/profiles', 'runs/aiops-continuation-20260921',
    'runs/research-final-20260921', 'runs/research-final-20260921-runtime-repair',
    'runs/research-continuation-20260921', 'runs/research-credit-recovery-20260922',
    'runs/research-identity-recovery-20260922',
    'docs/demo/TimeSigma_AIOPS_Leaderboard_3Schema.html',
    'docs/operations/server-storage-cleanup-20260921/provenance-metadata.tar.xz',
)
TEXT_SUFFIXES = {'.json', '.jsonl', '.log', '.txt', '.py', '.ts', '.js', '.html', '.toml', '.yaml', '.yml', '.md', '.csv'}


def safe_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(x in ('..', '.') for x in path.parts):
        raise ValueError('Unsafe archive member path')
    target = root / str(path)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Archive path escapes asset root')
    return target


def credentials(root):
    # Compare actual configured key bytes, never print or serialize them.
    values = [v.encode() for k, v in os.environ.items() if re.search(r'(API_KEY|TOKEN|SECRET)$', k) and len(v) >= 12]
    env = root / 'pi-host/.env'
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.partition('=')
            value = value.strip().strip('\"\'')
            if sep and re.search(r'(API_KEY|TOKEN|SECRET)$', key.strip()) and len(value) >= 12:
                values.append(value.encode())
    return values


def digest(path, secrets=()):
    h = hashlib.sha256()
    tail = b''
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
            if secrets:
                check = tail + chunk
                if any(secret in check for secret in secrets):
                    raise ValueError('Configured credential found in export candidate; review before sharing')
                tail = check[-max(map(len, secrets)):]
    return h.hexdigest()


def inventory(root, output, roots=ASSET_ROOTS):
    root = root.resolve()
    secrets = credentials(root)
    files, inode_hashes = {}, {}
    for name in roots:
        path = safe_path(root, name)
        if not path.exists():
            raise FileNotFoundError(f'Required scientific asset missing: {name}')
        for item in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if item.is_symlink():
                raise ValueError(f'Symlink requires explicit dependency resolution: {item.relative_to(root)}')
            if not item.is_file():
                continue
            rel = item.relative_to(root)
            if '__pycache__' in rel.parts or item.suffix in ('.pyc', '.pyo'):
                continue
            if item.name.startswith('.env') or item.suffix in ('.pem', '.key'):
                raise ValueError(f'Secret-like file must not be exported: {rel}')
            stat = item.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if identity not in inode_hashes:
                inode_hashes[identity] = digest(item, secrets if item.suffix in TEXT_SUFFIXES else ())
            files[rel.as_posix()] = {'bytes': stat.st_size, 'sha256': inode_hashes[identity]}
    report = {'format': 'timesigma-scientific-assets-1', 'roots': list(roots),
              'files': files, 'file_count': len(files),
              'logical_bytes': sum(x['bytes'] for x in files.values()),
              'unique_inode_bytes': sum(k[2] for k in inode_hashes),
              'note': 'Paths inside immutable historical JSON retain original provenance; do not execute those historical launch files. No credentials or environments are included.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return {k: v for k, v in report.items() if k != 'files'}


def verify(root, manifest):
    doc = json.loads(manifest.read_text())
    for name, expected in doc['files'].items():
        path = safe_path(root, name)
        if path.is_symlink() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f'Asset differs: {name}')
    return {'status': 'PASS', 'files': len(doc['files'])}


def pack(root, manifest, output):
    doc = json.loads(manifest.read_text())
    # Stream to a recipient or gzip on server. No staging copy of model trees.
    target = sys.stdout.buffer if str(output) == '-' else output.open('xb')
    try:
        with tarfile.open(fileobj=target, mode='w|gz', compresslevel=1) as archive:
            archive.add(manifest, arcname='asset-manifest.json', recursive=False)
            for name, expected in doc['files'].items():
                path = safe_path(root, name)
                if path.is_symlink() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                    raise ValueError(f'Asset changed since inventory: {name}')
                if str(output) != '-' and shutil.disk_usage(output.parent).free < 10 * 1024**3:
                    raise OSError('Keep 10 GiB free; partial archive retained, not a completed release')
                archive.add(path, arcname=name, recursive=False)
    finally:
        if target is not sys.stdout.buffer:
            target.close()


def unpack(archive, output):
    if output.exists():
        raise FileExistsError('Extract into a NEW directory; never overwrite Memory')
    output.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            safe_path(output, member.name)
            if not (member.isfile() or member.isdir() or member.islnk()):
                raise ValueError('Only regular files/directories/internal hard links are allowed')
            if member.islnk():
                safe_path(output, member.linkname)
            stream.extract(member, output, filter='data')
    return verify(output, output / 'asset-manifest.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('inventory', 'pack', 'verify', 'unpack'))
    p.add_argument('--root', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--archive', type=Path)
    a = p.parse_args()
    if a.stage == 'unpack':
        if not a.archive or not a.output:
            p.error('unpack requires --archive and --output')
        result = unpack(a.archive, a.output.resolve())
    else:
        if not a.root or not a.manifest:
            p.error('requires --root and --manifest')
        root = a.root.resolve()
        if a.stage == 'inventory':
            result = inventory(root, a.manifest.resolve())
        elif a.stage == 'verify':
            result = verify(root, a.manifest.resolve())
        else:
            if a.output is None:
                p.error('pack requires --output FILE or --output - to stream')
            pack(root, a.manifest.resolve(), a.output)
            return
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
