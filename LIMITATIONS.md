# Limitations

Known limitations of Sort It Right, found while building and testing it. Each entry says what the limitation is,
the evidence, what it means for a user, and what is done about it (now or as future work).
Details of every experiment are in [notes.md](notes.md); raw results are in [results/](results/).

Last updated: 27 Nov 2026 (after bricks 1–4: detection, sources, chunking). Retrieval and answer generation
have not been tested yet; their limitations will be added after bricks 6, 7 and 10.

---

## 1. Object detection (YOLO26n)

### 1.1 Detection rate is below the target
- **What:** On 24 own photos of detectable items (bottle, glass, carrot, spoon; 6 scenarios each), YOLO26n found
  the right item in **14/24 = 58%** at the chosen threshold of 0.3. The target in the project plan was 75%.
  No threshold reached 75%: even at 0.1 the result is 58%.
- **Evidence:** `results/threshold_summary_2026-11-26.csv`, notes.md (Experiments).
- **Impact:** About 4 in 10 scans of a supported item end with "not sure, pick the item" instead of an automatic answer.
- **Handled by:** manual item pick and the typed Ask tab, so the user still gets an answer.
- **Future work:** a bigger model (`yolo26s`), fine-tuning on own photos, or YOLOE-26 open-vocabulary detection.

### 1.2 Far, hand-held and tilted items are often missed completely
- **What:** 10 of the 24 photos were never detected at any threshold (no waste class even at confidence 0.05):
  all far shots except the bottle (glass, carrot, spoon), held or odd-pose shots (bottle held, glass held, glass on its side,
  spoon on its side) and two light-clutter shots (carrot, spoon).
- **Impact:** Lowering the threshold cannot fix these; it is a limit of the smallest ("nano") model on small, tilted
  or partly covered objects. Even easy shots scored low (bottle 0.31, glass 0.48).
- **Handled by:** demo instruction: hold the item close to the camera, upright, against a plain background.

### 1.3 Wrong labels are possible
- **What:** A bottle lying on its side was labelled **banana** (confidence 0.28). The 0.3 threshold removes this case,
  but a confident wrong label on another object cannot be ruled out with 34 photos.
- **Impact:** A wrong label is the worst error: the app would confidently show the wrong bin.
- **Handled by:** threshold chosen to give 0 wrong labels on the test set; the detected label is shown to the user,
  who can correct it.

### 1.4 Many waste items are not YOLO classes at all
- **What:** YOLO26n is trained on the 80 COCO classes. There is no class for **can, carton, paper, cardboard box,
  plastic bag, battery or jar**. The app's filter keeps 10 classes that can be household waste
  (bottle, cup, wine glass, bowl, banana, apple, orange, book, carrot, spoon).
- **Evidence:** the energy drink can was detected as "bottle" in 2/6 photos and as nothing in 4/6.
- **Impact:** These items cannot be recognised by the camera.
- **Handled by:** manual item pick and the Ask tab. In the test, a can counts as correct if the result is "none"
  (manual pick follows) or "bottle" (the material question offers "metal can"), because the app recovers in both cases.
  This means can photos never count as "missed"; the 58% above is measured without cans for that reason.

### 1.5 The camera cannot tell the material or the deposit
- **What:** COCO has one `bottle` and one `cup` class for every material. The bin depends on material
  (plastic, glass, metal) and on whether the package has a deposit (pantti), which YOLO cannot see.
- **Handled by:** the app asks the user for the material with buttons when an item can be more than one material.

---

## 2. Detection test set

### 2.1 Small and from one setting
- **What:** 34 photos: 5 items × 6 scenarios + 4 photos with no item. One example object per item, one home,
  one camera, one person, taken on one day.
- **Impact:** The 58% is a rough estimate. Different objects (a PET bottle with a label, a different can),
  lighting or backgrounds can give different results.

### 2.2 The "no item" photos are easy
- **What:** The 4 "none" photos are a curtain, a texture close-up, an empty hand and a blank wall. None of them
  contains an object that looks like waste (e.g. a TV remote on a cluttered desk).
- **Impact:** 0 false alarms in this set says little about false alarms in real, cluttered scenes.

### 2.3 Test items were partly chosen for availability
- **What:** Banana and a paper coffee cup from the original plan were replaced (items not available);
  an egg carton was replaced by a metal spoon because cartons have no COCO class.
- **Impact:** Paper and carton items are not covered by the detection test.

---

## 3. Knowledge sources (RAG documents)

### 3.1 Only valid for Mikkeli
- **What:** Sorting rules differ between Finnish municipalities. Example: in Mikkeli a drinking glass goes to
  **glass and ceramics** (eco-point), while some other regions put it in mixed waste. The main sources are
  Metsäsairila (Mikkeli's waste company); Rinki, Palpa and InfoFinland are national.
- **Impact:** Answers can be wrong for users outside the Mikkeli region.

### 3.2 Small document set, English only
- **What:** 13 English pages (Metsäsairila guide + 9 pages, Rinki metal packaging, Palpa FAQ, InfoFinland).
  Finnish-only pages and the Metsäsairila A–Ö search tool are not included. Hazardous waste, electronics, batteries
  and textiles have no dedicated page in the set (only short mentions in the household guide).
- **Impact:** Questions about items not in these pages should get "I'm not sure – check the guide".

### 3.3 Snapshot of the rules, possibly outdated
- **What:** Pages were downloaded on 26 Nov 2026. Some show old dates in the footer (Metsäsairila © 2020,
  Palpa FAQ © 2014–2015), so parts may not be up to date, and English pages may lag behind the Finnish ones.
- **Impact:** If a rule changes, the app keeps answering with the old rule until the sources are downloaded again.

### 3.4 Deposit rule for cans comes from national sources
- **What:** Metsäsairila's pages only say that **non-deposit** cans go to metal. The rule "return deposit cans to
  the shop's reverse vending machine" comes from Rinki and Palpa.
- **Impact:** The answer for the most common recycled item relies on retrieving the right national page.

### 3.5 Copyright: documents are not in the repository
- **What:** All sites say "© … All rights reserved", so the repository stores only the links (`data/sources.csv`).
  The documents and the chunks made from them are downloaded and rebuilt locally
  (`src/download_sources.py`, `src/ingest.py`).
- **Impact:** Setting up the project needs internet once. If a page is moved or removed, it can no longer be downloaded.

---

## 4. Text extraction and chunking

### 4.1 Extraction depends on each website's layout
- **What:** The main text is cut out of each page with a CSS selector per site (`CONTENT_SELECTORS` in `ingest.py`).
- **Impact:** If a site changes its HTML, extraction can silently return little or no text (or menus again).
  The output must be checked after every new download.

### 4.2 Chunks can lose their context
- **What:** Chunks from the middle of the long Metsäsairila "Sorting instructions" page can list items
  ("Yes: Food cans") without the bin heading ("Metal") that came before them.
- **Impact:** Retrieval may find a chunk that does not say which bin it is about. To be measured in the retrieval test (brick 6).
- **Possible fix:** put the page title in front of each chunk when embedding.

### 4.3 Smaller index than planned
- **What:** 61 chunks instead of the planned 100–400, because the 13 pages contain only about 22,000 characters of text.
  Some chunks are short tails (110–214 characters) that mostly repeat the previous chunk; the Palpa text keeps
  line breaks in the middle of sentences.
- **Impact:** Small effect expected; noted for the chunk-size experiment.

---

## 5. Hardware and setup

### 5.1 Runs on one laptop, offline
- **What:** Developed and demonstrated on an Apple M2 with 8 GB RAM. YOLO and embeddings use the Apple GPU (`mps`)
  with a CPU fallback; the language model runs locally with Ollama (a small 3B model).
- **Impact:** A small local model may give weaker answers than a large cloud model; speed on other machines is untested.

### 5.2 Project must not live in an iCloud-synced folder
- **What:** When the project was in `~/Documents`, macOS offloaded parts of the virtual environment to iCloud
  and imports hung forever.
- **Handled by:** the project was moved to `~/Developer`. Anyone running it on a Mac should avoid iCloud-synced folders.

---

## Still to be tested

To be added after the remaining bricks:
- Retrieval: how often the right chunk is in the top 4 (brick 6).
- Answers: correct bin, citations, "I'm not sure" for out-of-scope questions (brick 7).
- End to end with real items on the camera (brick 10).
