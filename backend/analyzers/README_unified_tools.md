# Unified tool wrappers

This folder contains:
- `base_tool.py`: shared `BaseTool` ABC.
- `legacy_*_tool.py`: adapters wrapping the existing legacy analyzers.
- `unified_*_tool.py`: unified wrappers (API-safe) around legacy adapters.

NOTE:
The repository previously had naming collisions when wrapper files used the same module names as legacy analyzers.
Keep wrapper modules prefixed with `unified_` and legacy adapters prefixed with `legacy_`.

