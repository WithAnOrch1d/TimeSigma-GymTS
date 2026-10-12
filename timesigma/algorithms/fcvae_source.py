"""Exact, in-memory compatibility edits to the pinned FCVAE dependency source."""
from hashlib import sha256
from importlib import import_module
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder, SourceFileLoader
from importlib.util import spec_from_file_location
import sys


SOURCE_EDITS = {
    "TSDataset": (
        "5e63bb3472bf1d4686808d57d26faa147e495176b0a058d0c9615c2af9b8fd94",
        (("torch.ones_like(y_aug[i, time_start[i] :]", "torch.ones_like(y_aug[i, time_start[i] :])"),
         ('.to("cuda")', '.to(x.device)')),
    ),
    "Model": (
        "794d16eff0faabc1c1cf48e7c782738e5cf86273970f0348790648d910948337",
        (('.to("cuda")', '.to(x.device)'),),
    ),
}


def compatible_source(name: str, source: bytes) -> str:
    digest, edits = SOURCE_EDITS[name]
    if sha256(source).hexdigest() != digest:
        raise ValueError(f"FCVAE {name} differs from the audited EasyTSAD source")
    text = source.decode("utf-8")
    for old, new in edits:
        assert text.count(old) == 1
        text = text.replace(old, new, 1)
    return text


class _SourceLoader(SourceFileLoader):
    def get_code(self, fullname):
        source = compatible_source(fullname.rsplit(".", 1)[1], self.get_data(self.path))
        return compile(source, self.path, "exec")


class _SourceFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in {f"EasyTSAD.Methods.FCVAE.{name}" for name in SOURCE_EDITS}:
            return None
        original = PathFinder.find_spec(fullname, path)
        return spec_from_file_location(fullname, original.origin, loader=_SourceLoader(fullname, original.origin))


def import_fcvae_source():
    finder = _SourceFinder()
    sys.meta_path.insert(0, finder)
    try:
        return import_module("EasyTSAD.Methods.FCVAE")
    finally:
        sys.meta_path.remove(finder)
