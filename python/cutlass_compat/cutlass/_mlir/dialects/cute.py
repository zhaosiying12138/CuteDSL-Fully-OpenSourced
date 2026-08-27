"""Minimal CuTe dialect type tokens used by FlashInfer runtime pointers."""
from __future__ import annotations

import enum

from .. import ir


class AddressSpace(enum.IntEnum):
    generic = 0
    gmem = 1
    smem = 3


class PtrType:
    @staticmethod
    def get(element_type, address_space=AddressSpace.generic, alignment=None):
        del element_type, alignment
        space = int(address_space)
        return ir.Type("!llvm.ptr" if space == 0 else f"!llvm.ptr<{space}>")
