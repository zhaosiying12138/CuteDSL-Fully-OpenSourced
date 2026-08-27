"""Compatibility record used by direct persistent-scheduler fast paths."""
from __future__ import annotations


class WorkTileInfo:
    def __init__(self, tile_idx, is_valid_tile):
        self.tile_idx = tile_idx
        self.is_valid_tile = is_valid_tile
