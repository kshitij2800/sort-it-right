# Project notes — Sort It Right

Running log of what was built, what changed, and **why**. Newest entries at the bottom of each section.

---

## Decisions

| Date | Decision | Why | Alternatives considered |
|---|---|---|---|
| 10 Nov | Pin exact versions of the 12 top-level packages in `requirements.txt` | A package update before the demo can't silently break the app | `pip freeze` (153 lines incl. dependencies, unreadable); no pins (risky) |
| 10 Nov | Don't commit `index/`, `*.pt` model weights, `.venv/` | All can be regenerated: index by `build_index.py`, weights auto-download, venv from `requirements.txt` | Committing them (large, redundant) |
| 19 Nov | Test photos named `<item>_<NN>.jpeg` in `data/test/` | Name shows the true item; two digits keep sort order | Free names (no ground truth in the name) |
| 26 Nov | Material (glass/plastic/metal/deposit) is a **separate step** from detection | COCO has one `bottle` and one `cup` class for all materials, but the bin depends on material and deposit (pantti). YOLO answers "what object", the app asks "what material" | YOLOE open-vocabulary prompts (can't tell clear plastic from glass, can't read deposit label); training a material classifier (no time, no data) |
| 26 Nov | Plan: ask the user with material buttons (brick 8) only when `item_map.csv` lists more than one material | One tap is cheap; a wrong bin is not. Banana etc. need no question | Fully automatic guess |
| 26 Nov | A **wrong label is worse than a miss** → lean towards a higher confidence threshold | A miss → user picks item manually. A wrong label → app confidently gives the wrong bin | Treat both errors equally |
| 26 Nov | Run YOLO once at conf 0.05, apply thresholds afterwards | Any threshold (0.1–0.9) can be tested without re-running the model; test uses the same `pick_best()` as the app | Re-running YOLO per threshold (slower, 3 thresholds only) |
| 26 Nov | Run heavy compute on the Apple GPU (`device="mps"`, CPU fallback) | M2 GPU is faster for YOLO and embeddings; Ollama uses Metal automatically | CPU only |
| 26 Nov | Move project from `~/Documents` to `~/Developer` | iCloud had offloaded parts of `.venv` and a test photo; imports hung. Demo must run offline | `.venv.nosync` (photos/index still at risk); changing iCloud settings (affects whole Mac) |
| 26 Nov | Test items: plastic bottle, energy drink can (metal, pantti), glass cup, carrot (bio), metal spoon (small metal) | Everyday household items available at home, 5 different bins. 4 are COCO classes (bottle, cup/wine glass, carrot, spoon); the can is kept on purpose as one honest hard case (most-returned deposit item in Finland, but no COCO class) | Banana + paper coffee cup (not available); egg carton (dropped, see next row) |
| 26 Nov | Swap egg carton for a metal spoon | Carton has no COCO class either, so it would only repeat what the can already shows. Spoon is a real COCO class and contrasts with the can: both metal, but spoon -> small-metal collection, can -> deposit return. Cartons are still handled in the app by manual pick and the Ask tab | Keep egg carton (two undetectable items, nothing new learned); fork (spoon shape is more distinctive) |
| 26 Nov | Add `carrot` and `spoon` to `WASTE_CLASSES` | Both are in the test set; without them the filter throws the detections away | Add all possibly-waste COCO classes (phones, cutlery, food…) — more classes = more chances of a wrong label, and no photos to measure them |
| 26 Nov | Can photos: `true_label = none\|bottle` | YOLO is **not** expected to recognise a can. The app recovers in both cases: `none` -> "not sure, pick the item" buttons -> user taps *can*; `bottle` -> material buttons include *metal can (pantti)*. Any other label (cup, bowl, …) sends the user the wrong way -> counts as wrong. **Side effect:** because `none` is in the answer key, an undetected can is graded *correct*, not *missed* — for a can, "nothing found" is the desired behaviour. So can photos never add to the missed count; report this next to the results so the can doesn't look like it was detected well | `none` only (punishes `bottle`, though it still leads to the right bin); `bottle` only (would count the desired fallback as a miss) |
| 26 Nov | Keep the 4 `none` photos as taken (curtain, texture close-up, empty hand, blank wall) | Time; they still check the basic case. **Limitation:** they are easy scenes with no object-like shapes, so they say little about false alarms at low thresholds — mention this with the results and in LIMITATIONS.md | Retake 2 with a cluttered desk and a tricky non-waste object (remote, etc.) |
| 26 Nov | **Threshold decision rule** (set *before* seeing results): pick the threshold with the fewest wrong labels while still detecting ≥ 75% of the 24 detectable item photos (bottle, glass, carrot, spoon). Tie -> more correct wins | 75% is the detection target already in the testing plan. Cans excluded from the % (graded correct even when not detected, would inflate it); `none` photos excluded (nothing to detect), but their wrong count still counts | 50% (too many scans would end in manual pick); picking the threshold by eye after seeing results (not defensible) |
| 26 Nov | **Confidence threshold = 0.3** (`DEFAULT_CONF` in `src/detect.py`) | Highest detection rate of any threshold (14/24 = 58%, tied with 0.1–0.25) and the only one of those with **0 wrong labels** (0.1–0.25 call `bottle_06` a banana at 0.28). Higher thresholds don't reduce wrong labels further (already 0) and only add misses. The 75% rule could not be met by any threshold, so 0.3 was chosen as the best trade-off; it would win under any % rule. Accepted 58% for this project because of the code freeze; the app's manual-pick fallback covers the misses | 0.25 (same detections, 1 wrong label); 0.4 (2 fewer detections, no gain); trying a bigger model `yolo26s` (no time before code freeze) |
| 26 Nov | Catch-up plan: drop brick 11 (YOLOE), scale testing to ~34 photos and 15 RAG questions | 4 days to code freeze; testing must not be dropped | Dropping testing (not allowed by brief) |

**Open decisions**
_(none right now)_

---

## Work log

### Brick 0 — repo, venv, packages, folders (10–11 Nov) 
- Created `src/ data/raw/ data/test/ notebooks/ results/ docs/` with `.gitkeep` (git doesn't track empty folders).
- `.gitignore`: added `index/`, `*.pt`, `.DS_Store`, `CLAUDE.md`, `.claude/`.
- `requirements.txt`: 12 packages grouped by pipeline stage, pinned after install.
- Check passed: `import ultralytics, chromadb, streamlit` (first import ~18 s: PyTorch size + bytecode compile + macOS scan).

### Brick 1 — YOLO on one image (19 Nov) 
- `src/detect.py`: load `yolo26n.pt`, run on an image from the command line, print `label conf`, save boxed image.
- Sample `bus.jpg`: bus 0.88, people 0.65–0.87 (half-visible person scored lowest).
- Own photo `bottle_01.jpeg`: **bottle 0.91**. Also detected chair 0.63, dining table 0.63 → motivated the waste filter.

### Brick 2 — waste filter + threshold experiment (26 Nov) 
- `src/detect.py` split into `all_detections()` → `pick_best(detections, conf)` → `detect(image, conf)`.
  - `WASTE_CLASSES` = bottle, cup, wine glass, bowl, banana, apple, orange, book, carrot, spoon.
  - `DEFAULT_CONF = 0.3`, chosen by the experiment (was a 0.4 placeholder).
- `src/threshold_test.py`: reads `data/test/labels.csv` (`file,item,true_label,scenario`), tries thresholds 0.1–0.9,
  classifies each photo as correct / missed / wrong, saves `results/threshold_photos_<date>.csv` and `results/threshold_summary_<date>.csv`.
- Planned photo set: 6 scenarios (clutter, easy, dim, far, held, odd pose) × 5 items (plastic bottle, energy drink can,
  glass cup, carrot, metal spoon) + 4 "none" photos = 34. Names `<item>_<NN>.jpeg`, NN = scenario number (01 clutter … 06 odd pose).
- 26 Nov: 30 item photos taken (bottle, can, glass, carrot, spoon × 6 scenarios); `labels.csv` filled. 4 `none` photos added later (curtain, texture, empty hand, blank wall).
  - File/`item` name is our name for the object (`glass`); `true_label` uses YOLO's class names (`cup|wine glass`) — COCO has no "glass" class.
  - The 01 "clutter" photos for can/glass/carrot/spoon are *light* clutter (desk, chair, one bowl), less busy than `bottle_01`.
  - `carrot_01` and `spoon_01` contain a black bowl; `bowl` is a waste class, so YOLO may pick it -> would count as wrong (realistic trap, kept on purpose).
- Check on `bottle_01.jpeg`: chair/table filtered out, returns `bottle 0.91`.
- Ran `python src/threshold_test.py` on all 34 photos -> results in the Experiments section below; threshold set to 0.3.

---

## Problems and fixes

| Date | Problem | Cause | Fix |
|---|---|---|---|
| 10 Nov | Import check looked "stuck" | First import of PyTorch-based libraries is slow (~18 s), not a hang | Waited; later imports are faster |
| 26 Nov | `detect.py` hung forever at 0% CPU | Project was in iCloud-synced `~/Documents`; macOS had offloaded OpenCV libraries in `.venv` ("dataless" files), Python waited for iCloud to download them | Cloned repo from GitHub to `~/Developer/sort-it-right`, rebuilt `.venv` from pip cache |

---

## Experiments

### Confidence threshold (brick 2)
Run: 26 Nov, model `yolo26n.pt`, device mps, 34 photos (`data/test/labels.csv`), thresholds 0.1–0.9.
Files: `results/threshold_photos_2026-11-26.csv` (per photo), `results/threshold_summary_2026-11-26.csv`.

| Threshold | Correct (34) | Missed | Wrong | Detected of 24 detectable items* |
|---|---|---|---|---|
| 0.10–0.25 | 24 | 9 | 1 | 14 = 58% |
| **0.30** | **24** | **10** | **0** | **14 = 58%** |
| 0.40 | 22 | 12 | 0 | 12 = 50% |
| 0.50 | 18 | 16 | 0 | 8 = 33% |
| 0.60 | 17 | 17 | 0 | 7 = 29% |
| 0.70 | 16 | 18 | 0 | 6 = 25% |
| 0.80 | 14 | 20 | 0 | 4 = 17% |
| 0.90 | 11 | 23 | 0 | 1 = 4% |

\*bottle, glass, carrot, spoon (6 each). Cans and `none` photos are left out of the %: cans are graded correct even when
nothing is detected (`none|bottle`), and `none` photos have nothing to detect.

**Chosen: 0.3.** Decision rule (set before the run): fewest wrong labels while detecting ≥ 75%. **No threshold reached 75%**
(max 58%), so the rule had no valid answer. 0.3 still dominates every other threshold: same detections as 0.1–0.25 but
0 wrong instead of 1, and more detections than 0.4+ with the same 0 wrong. Accepted 58% (below the 75% target) because of the code freeze.

Observations:
- **Only wrong label:** `bottle_06` (bottle lying on its side) -> `banana` 0.28. Removed by 0.3.
- **10 detectable photos are never found, at any threshold** (no waste class even at 0.05): far (glass_04, carrot_04, spoon_04),
  held / odd pose (bottle_05, glass_05, glass_06, spoon_06), light clutter (carrot_01, spoon_01). A threshold can't fix these;
  it is a limit of the nano model on small, tilted or hand-held objects.
- Easy shots score low too: bottle_02 0.31, glass_02 0.48 -> the nano model is borderline even on clean photos.
- **Cans:** 2/6 detected as `bottle` (easy 0.40, dim 0.63), 4/6 `none` -> all 6 recoverable by the app (material buttons or manual pick).
  Graded correct by design, so they do not show up as misses.
- The bowl in carrot_01 / spoon_01 was **not** picked up, so the "bowl trap" caused no wrong label.
- `none` photos: 0 false alarms, but they are easy scenes (see decisions).

For LIMITATIONS.md: 58% detection on own photos (target was 75%); far, hand-held and tilted items are often missed;
cans and cartons are not COCO classes; `none` test set is easy. Possible fixes (future work): bigger model (`yolo26s`),
YOLOE-26 open-vocabulary, fine-tuning on own photos; for the demo: hold the item close and upright.
