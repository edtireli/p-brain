"""Compatibility wrapper.

`utils.loading.load_dce_4d` is the canonical DCE NIfTI loader in p-brain.
This module exists only to avoid breaking older imports.
"""

from utils.loading import load_dce_4d  # noqa: F401
