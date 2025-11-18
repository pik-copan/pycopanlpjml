# Cursor Workspace Context

This file encodes the conventions and expectations for collaborating on the
InSEEDS–LPJmL coupling stack inside Cursor. Keep it updated when the workflow
evolves.

---

## Repositories & Environment
- Always develop against the editable repos in this tree:
  1. `/p/projects/copan/users/jannesbr/repos/pycopancore`
  2. `/p/projects/copan/users/jannesbr/repos/pycopanlpjml`
  3. `/p/projects/copan/users/jannesbr/repos/inseeds`
- Prepend the paths above to `sys.path` (or add them before project imports in
  bootstrap scripts) so we never fall back to stale site‑packages copies.
- When running tools manually, remember that `pycopancore` version is tracked in
  `setup.py`; keep it at `0.8.7.dev4` or later so it satisfies downstream
  requirements.
- Use NumPy-style docstrings for all new or updated functions/methods.
- Default to ASCII in code files unless the file already uses Unicode.

## Parallel/Dask Standards
- `run_inseeds_dask_mpi.py` is the authoritative coupled-run entry point.
  - Guard module entry with `if __name__ == "__main__":`.
  - Acquire `/tmp/inseeds_coupling_lock.pid` before doing any work.
  - Instantiate the InSEEDS `Model` (and let it connect to LPJmL) **before**
    starting the Dask cluster.
  - Build the `LocalCluster` with `processes=True`, `threads_per_worker=1`,
    `memory_limit="auto"`, and a node-local `local_directory`
    (`$TMPDIR` fallback `/tmp`).
  - Wait for ≥90 % of requested workers; log counts every 10 seconds.
- In `pycopanlpjml.component.Component.update_countries`, dispatch countries via
  `client.submit(..., pure=False)` to bypass Dask tokenization of complex
  pycopancore entities.
- `pycopancore`’s mixin metaclass must yield real `__qualname__` values so
  cloudpickle sees the correct class names.

## Zarr & Data Integrity
- All Zarr writes should run under the combined thread/file lock provided by
  `pycopanlpjml.zarr_backend`.
- Refresh world-level caches (`refresh_cached_arrays()`) at the start of each
  model year before country updates.
- Clean up `/tmp/inseeds_dask_*` directories after runs to avoid disk bloat.

## Testing & Validation
- Fast checks:
  - `python3 tests/test_update_countries_mock.py` → verifies Dask path and
    serialization.
  - `python3 run_inseeds.py` → local serial sanity run.
  - `python3 submit_inseeds.py` + `./monitor_job.sh` → full coupled run on the
    cluster; watch stderr for worker counts and yearly progress.
- Clear `.pyc` files / `__pycache__` directories when switching branches or
  after structural refactors.

## Logging & Diagnostics
- Scripts should emit structured, human-readable logs to stderr (✓, ⚠️, ✗
  patterns are fine).
- Key checkpoints to log: lock acquisition, model creation, Dask worker counts,
  cache refresh progress, each LPJmL/InSEEDS year, and cleanup.

## Collaboration Guidelines
- Never revert or overwrite user edits unless explicitly asked.
- Prefer targeted tests and log inspection to confirm fixes before running
  multi-hour jobs.
- Document significant findings in Markdown under
  `inseeds_regions/` (e.g., `FIXES_YYYY-MM-DD.md`, `PERFORMANCE_*.md`).
- Update this context file whenever we adopt new standards.

For a more narrative reference, see `inseeds_regions/CURSOR_COLLAB_GUIDE.md`.

