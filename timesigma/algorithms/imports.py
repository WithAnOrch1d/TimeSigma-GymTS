"""Import native Methods with the small compatibility surface they actually use."""
from importlib import import_module
from importlib.util import find_spec
import sys
from types import ModuleType


def _flatten(items):
    # DCdetector imports tkinter only for this tuple/list operation. Preserve
    # leaf objects (attention tensors), tuple order and None handling exactly.
    return tuple(leaf for item in items if item is not None
                 for leaf in (_flatten(item) if isinstance(item, (tuple, list)) else (item,)))


def import_method(module: str) -> ModuleType:
    if module == 'EasyTSAD.Methods.FCVAE':
        from .fcvae import import_fcvae

        return import_fcvae()
    if module == 'EasyTSAD.Methods.DCdetector' and find_spec('tkinter') is None:
        helper = ModuleType('tkinter')
        helper._flatten = _flatten
        sys.modules['tkinter'] = helper
        try:
            return import_module(module)
        finally:
            del sys.modules['tkinter']
    return import_module(module)
