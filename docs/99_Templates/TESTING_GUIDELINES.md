# Testing Guidelines

Best practices para testing manual y automated en el proyecto Atheria, optimizados para RAG y desarrollo futuro.

---

## 🎯 Philosophy

**Test Early, Test Often, Document Everything**

- Tests detectan bugs antes de merge
- Documentación de tests sirve como especificación
- Tests manuales complementan (no reemplazan) automated tests

---

## 🧪 Testing Pyramid

```
        /\
       /  \  E2E Tests (Manual - High confidence)
      /____\
     /      \  Integration Tests (Automated - Medium coverage)
    /________\
   /          \  Unit Tests (Automated - High coverage)
  /__________  \
```

---

## ✅ Automated Testing

### Frontend (TypeScript/React)

**Pre-commit Checks:**
```bash
# Lint
cd frontend && npm run lint

# Type checking
npm run type-check  # or tsc --noEmit

# Build (catches import/syntax errors)
npm run build
```

**What to Test:**
- Component mounting/unmounting
- State updates
- WebSocket message handling
- User interactions (clicks, inputs)

### Backend (Python)

**Pre-commit Checks:**
```bash
# Syntax verification
python3 -m py_compile src/**/*.py

# Type checking (if using mypy)
mypy src/

# Run test suite
pytest tests/
```

**What to Test:**
- Buffer operations (history_manager)
- Data transformations (viz pipeline)
- WebSocket handlers
- Edge cases (empty buffers, invalid inputs)

---

## 🖱️ Manual Testing

### When to Test Manually

1. **UI/UX Changes** - Visual appearance, responsiveness
2. **Performance** - FPS, memory usage, latency
3. **Integration** - Multiple components working together
4. **Edge Cases** - Uncommon user behaviors

### Testing Checklist Template

**Para cada feature, verificar:**

```markdown
## Feature: [Nombre del Feature]

### Functional Testing
- [ ] Feature activates correctly
- [ ] Expected output/behavior
- [ ] Edge case 1: [describe]
- [ ] Edge case 2: [describe]

### Performance Testing
- [ ] No lag/stuttering
- [ ] FPS stable (>30 target, >60 ideal)
- [ ] Memory usage stable

### Integration Testing
- [ ] Works with feature A
- [ ] Works with feature B
- [ ] No conflicts with existing features

### Error Handling
- [ ] Invalid input handled gracefully
- [ ] Error messages are clear
- [ ] No crashes/exceptions
```

**Ejemplo:** Ver `docs/40_Experiments/PHASE_3_TESTING_CHECKLIST.md`

---

## 🎨 UI/Visual Testing

### Color Verification

**For Visualizations:**
1. Open DevTools → Color Picker
2. Sample pixels from visualization
3. Verify colormap correctness:
   - HSV Phase: Red(0°) → Green(120°) → Blue(240°)
   - Density: Blue(low) → Yellow(high)
   - Real/Imag: Blue(negative) → Yellow(positive)

### Layout Verification

**Check for:**
- [ ] Z-index issues (overlapping elements)
- [ ] Responsive design (resize browser)
- [ ] Text readability (contrast ratios)
- [ ] Alignment/spacing consistency

**Tools:**
- Browser DevTools → Inspect Element
- Browser zoom (50%, 100%, 150%)
- Multiple screen sizes (if available)

---

## ⚡ Performance Testing

### FPS Measurement

**Method 1: DevTools Performance Tab**
```
1. Open DevTools → Performance
2. Start recording
3. Perform action (e.g., run simulation)
4. Stop recording
5. Check FPS graph (should be ~60 FPS)
```

**Method 2: Manual FPS Counter**
- Look for FPS indicator in UI
- Verify: >30 FPS acceptable, >60 FPS ideal

### Memory Profiling

**Method: DevTools Memory Tab**
```
1. Open DevTools → Memory
2. Take heap snapshot (baseline)
3. Perform memory-intensive action (e.g., fill buffer with 1000 frames)
4. Take second heap snapshot
5. Compare sizes
```

**Red Flags:**
- Memory grows indefinitely (leak)
- Sudden spikes (GC thrashing)
- OOM errors

**Expected:** Memory stabilizes after initial growth.

---

## 🔗 Integration Testing

### Testing Multiple Features Together

**Pattern:**
1. Enable Feature A
2. Verify Feature A works
3. Enable Feature B (while A is active)
4. Verify both A and B work
5. Disable A, verify B still works
6. Re-enable A, verify both work again

**Example:** History Buffer + Advanced Visualizations
```
1. Start simulation (History Buffer)
2. Verify timeline appears
3. Change to HSV Phase visualization
4. Rewind to frame 50
5. Verify HSV visualization updates for frame 50
6. Change to Real visualization (still at frame 50)
7. Verify Real visualization shows frame 50
```

---

## 🐛 Bug Detection

### What to Look For

**Visual Bugs:**
- Flickering/flashing
- Misaligned elements
- Wrong colors
- Missing elements

**Functional Bugs:**
- Features not responding
- Incorrect calculations/data
- State inconsistencies
- Unexpected behavior

**Performance Bugs:**
- Slowdowns/lag
- Memory leaks
- High CPU/GPU usage
- Crashes/freezes

### Reporting Bugs

Use template from `docs/90_Troubleshooting/COMMON_BUGS.md`:

```markdown
### [Bug Name]

**Severity:** 🔴 CRITICAL / 🟡 MEDIUM / 🟢 LOW  
**Symptom:** [What you see]  
**Steps to Reproduce:**
1. Step 1
2. Step 2
3. Bug appears

**Expected:** [What should happen]  
**Actual:** [What actually happens]  
**Screenshot:** ![bug](path/to/screenshot.png)
```

---

## 📊 Test Coverage Goals

### Minimum Coverage (Required for Merge)

- [ ] **Automated Tests:** Build passes, no lint errors
- [ ] **Smoke Test:** Basic functionality works (app loads, no crashes)
- [ ] **Critical Path:** Main user flow tested manually
- [ ] **No Critical Bugs:** All 🔴 bugs fixed

### Ideal Coverage (Nice to Have)

- [ ] **Edge Cases:** Uncommon scenarios tested
- [ ] **Performance:** Benchmarks collected
- [ ] **Integration:** All feature combinations tested
- [ ] **Documentation:** Test results documented

---

## ⏱️ Time Estimates

| Test Type              | Time Estimate | When to Do                  |
|------------------------|---------------|-----------------------------|
| Automated (lint+build) | 2-5 min       | Every commit                |
| Smoke Test             | 5 min         | Every feature               |
| Critical Path          | 10-15 min     | Every feature               |
| Full Manual Checklist  | 30-45 min     | Before merge to main        |
| Performance Profiling  | 15-30 min     | After performance changes   |

---

## 🚀 Pre-Merge Checklist

Antes de crear PR o merge a `main`:

```markdown
- [ ] Automated tests pass (lint, build, unit tests)
- [ ] Manual smoke test completed
- [ ] Critical bugs fixed (🔴 severity)
- [ ] Performance acceptable (no major regressions)
- [ ] Documentation updated
- [ ] Commit messages clear
- [ ] Branch up-to-date with main
```

---

## 🔗 References

- [[PHASE_3_TESTING_CHECKLIST]] - Example comprehensive checklist
- [[COMMON_BUGS]] - Known bugs database
- [[CI_CD_WORKFLOWS]] - Automated testing setup

---

## 💡 Pro Tips

1. **Test in Production-Like Conditions:** Use production grid sizes, real data
2. **Test Incrementally:** Don't wait until the end to test
3. **Document Unexpected Behavior:** Even if not a bug, may be useful
4. **Take Screenshots:** Visual evidence helps debugging
5. **Test on Multiple Browsers:** Chrome, Firefox, Safari (if available)
6. **Clear Cache:** Sometimes fixes "ghost bugs"
7. **Use Incognito Mode:** Eliminates extensions interference
