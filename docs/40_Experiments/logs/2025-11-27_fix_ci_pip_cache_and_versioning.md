# 2025-11-27: Fix CI Pip Cache and Refine Versioning

## 📝 Summary
Fixed a CI error where the pip cache cleanup failed because the cache folder was never created. Also refined the frontend versioning strategy to disable automatic bumps on frontend changes, requiring explicit version tags instead.

## 🐛 CI Pip Cache Fix
- **Issue:** The `actions/setup-python` step with `cache: 'pip'` was running unconditionally. When backend changes were skipped, `pip install` was never run, so the pip cache folder wasn't created. The post-job cleanup step of `setup-python` then failed trying to save a non-existent cache.
- **Fix:** Moved `actions/setup-python` to run **after** the backend change detection and made it conditional on `steps.check-backend.outputs.skip != 'true'`.
- **File:** `.github/workflows/ci.yml`

## 🔄 Frontend Versioning Strategy
- **Issue:** The `version-bump.yml` workflow was automatically bumping the version (defaulting to patch) whenever *any* file in `frontend/` changed. This was deemed "weird" and undesirable by the user.
- **Change:** Modified the workflow to **only** trigger a version bump if:
    1. A commit message contains a tag like `[version:bump:patch]`.
    2. A PR has a label like `version:patch`.
- **Outcome:** Frontend changes no longer trigger automatic bumps. Explicit intent is now required.
- **Files:**
    - `.github/workflows/version-bump.yml`
    - `docs/30_Components/VERSIONING_SYSTEM.md`

## 🔗 Related
- [[30_Components/VERSIONING_SYSTEM]]
