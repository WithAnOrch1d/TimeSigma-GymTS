"""Restore the pinned OmniAnomaly source and Python 3.6 CPU environment."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request

RUNTIME = Path(__file__).resolve().parent
REPO = RUNTIME.parents[2]
CONFIG = REPO / "third_party/adapters/omnianomaly"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", default="conda")
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    args = parser.parse_args()
    for source in json.loads((CONFIG / "sources.json").read_text()):
        root = REPO / "third_party/upstreams/omnianomaly" / source["name"]
        missing = [name for name in source["files"] if not (root / name).exists()]
        archive = None
        if missing and args.source_cache is None:
            slug = source["repository"].removeprefix("https://github.com/")
            url = "https://codeload.github.com/" + slug + "/tar.gz/" + source["revision"]
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if "archive_sha256" in source and hashlib.sha256(data).hexdigest() != source["archive_sha256"]:
                raise ValueError("source archive digest differs: " + url)
            archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        for name, expected in source["files"].items():
            path = root / name
            if path.exists():
                data = path.read_bytes()
            elif args.source_cache:
                data = (args.source_cache / source["name"] / name).read_bytes()
            else:
                members = [m for m in archive.getmembers() if m.isfile() and m.name.partition("/")[2] == name]
                if len(members) != 1:
                    raise ValueError("missing or ambiguous source member: " + name)
                data = archive.extractfile(members[0]).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("pinned source digest differs: " + str(path))
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(data)
        if archive:
            archive.close()
    runtime = REPO / ".runtimes/omnianomaly"
    python = runtime / "bin/python"
    if not python.exists():
        subprocess.run([args.conda, "create", "--yes", "--no-default-packages", "--prefix", str(runtime),
                        "--file", str(RUNTIME / "conda-explicit.txt")], check=True)
    command = [str(python), "-m", "pip", "install", "-r", str(RUNTIME / "requirements.txt")]
    if args.wheelhouse:
        command += ["--no-index", "--find-links", str(args.wheelhouse.resolve())]
    subprocess.run(command, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
