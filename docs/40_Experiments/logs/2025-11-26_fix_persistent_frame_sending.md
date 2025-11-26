# Fix: Persistent Frame Sending & Duplicate Logic

**Date:** 2025-11-26
**Author:** AI Agent (Antigravity)
**Status:** Fixed

## Context
The user reported that the simulation continued to send frames (or at least generate visualization data and logs) even when the Live Feed was disabled and the simulation was in "full speed" mode. The logs showed repeated `map_data stats` messages.

## Problem
Upon investigation of `src/pipelines/core/simulation_loop.py`, two critical issues were found:
1. **Duplicate Visualization Logic:** There was a second, redundant block of visualization logic at the end of the `simulation_loop` function (lines ~567+). This block was likely a leftover from a previous merge or refactor.
2. **Unconditional Execution:** In a previous attempt to clean up the file, the guard clause `if not should_send_frame: continue` was removed, but the duplicate block itself was not successfully removed. This caused the duplicate visualization logic to execute unconditionally in every loop iteration, ignoring the `should_send_frame` flag and `live_feed_enabled` status.

## Solution
1. **Removed Duplicate Code:** Deleted the entire duplicate visualization block (lines 532-876) from `simulation_loop.py`.
2. **Cleaned Up Logic:** Consolidated the simulation loop logic to ensure a single, clear flow for step execution and frame generation.
3. **Added Debug Logging:** Added debug logs to the `should_send_frame` decision logic to help diagnose future issues.
4. **Verified Syntax:** Ran a syntax check (`python -m py_compile`) to ensure the file is valid after the large deletion.

## Modified Files
- `src/pipelines/core/simulation_loop.py`: Removed duplicate logic and cleaned up control flow.

## Verification
- **Syntax Check:** Passed.
- **Logic Review:** The `should_send_frame` flag is now correctly calculated based on `effective_steps_interval` (which respects `live_feed_enabled`), and the only visualization logic in the file is guarded by this flag.

## Next Steps
- Monitor logs to ensure `map_data stats` only appears when Live Feed is enabled.
- Verify that "full speed" mode works as expected (no frames sent, high FPS).
