"""Enumerate Git upload files and test precisely that tree at a different path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ('.gitignore', '.python-version', 'README.md', 'pyproject.toml', 'uv.lock')
TREES = ('timesigma', 'third_party', 'examples', 'experiments', 'scripts', 'tests', 'docs/demo',
         'pi-host/src', 'pi-host/test')
PI_FILES = ('package.json', 'package-lock.json', 'tsconfig.json', 'tsconfig.build.json', '.env.example')
EXCLUDE = {'__pycache__', '.pytest_cache', 'upstreams'}


def files(root=ROOT):
    selected = [root / p for p in ROOT_FILES] + [root / 'pi-host' / p for p in PI_FILES]
    for tree in TREES:
        selected.extend(p for p in (root / tree).rglob('*') if p.is_file()
                        and not EXCLUDE.intersection(p.relative_to(root).parts)
                        and p.suffix not in ('.pyc', '.pyo'))
    for path in selected:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or linked upload file: {path}')
        if path.name.startswith('.env') and path.name != '.env.example':
            raise ValueError('Real environment file must never be uploaded')
        if path.stat().st_size >= 100 * 1024**2:
            raise ValueError('Large scientific files belong in release assets')
    return sorted(set(selected))


def run(output, test):
    if output.exists():
        raise FileExistsError('Use a new release-check output')
    output.mkdir(parents=True)
    selected = files()
    report = {'files': [str(p.relative_to(ROOT)) for p in selected],
              'file_count': len(selected), 'bytes': sum(p.stat().st_size for p in selected),
              'tests': {}, 'paid_calls': 0, 'detector_training': False}
    if test:
        stage = output / 'relocated-source'
        for path in selected:
            dest = stage / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        # Reuse installed dependencies only. No source tree or dataset symlinks.
        (stage / '.venv').symlink_to(ROOT / '.venv', target_is_directory=True)
        (stage / 'pi-host/node_modules').symlink_to(ROOT / 'pi-host/node_modules', target_is_directory=True)
        cases = {
            'python': ([sys.executable, '-I', '-c',
                        'import sys;sys.path.insert(0,sys.argv[1]);import pytest;raise SystemExit(pytest.main(["tests","-q"]))', str(stage)], stage),
            'pi_typecheck': (['npm', 'run', 'typecheck'], stage / 'pi-host'),
            'pi_tests': (['npm', 'test', '--', '--no-file-parallelism'], stage / 'pi-host'),
            'pi_build': (['npm', 'run', 'build'], stage / 'pi-host'),
            'memory_example': ([sys.executable, '-I', '-c',
                                'import sys,runpy;sys.path.insert(0,sys.argv[1]);runpy.run_path("examples/handoff/inspect_memory.py",run_name="__main__")', str(stage)], stage),
        }
        for name, (argv, cwd) in cases.items():
            with (output / f'{name}.log').open('x') as log:
                result = subprocess.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            report['tests'][name] = result.returncode
        report['status'] = 'PASS' if not any(report['tests'].values()) else 'FAILED'
    else:
        report['status'] = 'INVENTORIED_NOT_TESTED'
    (output / 'upload-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
    return report['status'] != 'FAILED'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--test', action='store_true')
    a = p.parse_args()
    if not run(a.output.resolve(), a.test):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
