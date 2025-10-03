
# PyMetaFlow-HPLC

Interactive Streamlit app for importing, preprocessing, referencing, aligning, and modeling HPLC/PDA chromatograms — with PCA/PLS tooling, loadings, UV matrix helpers, and exports.

 **Live demo:** [https://pymetaflow-hplc.streamlit.app/](https://pymetaflow-hplc.streamlit.app/)

---

## What it does

* **Import**

  * 2D LabSolutions ASCII (`R.Time (min) \t Intensity`)
  * 3D PDA ASCII (choose a wavelength, build a 2D matrix)
* **Preprocess**

  * resampling to grid, smoothing, baseline, normalization/scaling
* **Reference & Align**

  * reference to anchor/target; **Icoshift** (or robust fallback) alignment
* **Model**

  * PCA scores with % variance in axes, interactive **loadings by PC**
  * PLS / PLS-DA (scores, X-loadings, performance bars)
* **Visualize & Export**

  * overlays, stacked plots, peak counts, region integration (AUC), UV spectra/contours
  * MetaboAnalyst export

---

## Quick start

```bash
# clone your repo
git clone <your-repo-url>
cd <your-repo>

# (recommended) create venv and activate it
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# install
pip install -r requirements.txt

# run
streamlit run app_pymetaflow.py
```

> If you plan to use Icoshift alignment, ensure `pyicoshift` installs successfully (it’s in `requirements.txt`).

---

## Data formats

### 2D LabSolutions ASCII

* A plain-text file containing a header line like:

  ```
  R.Time (min)    Intensity
  ```
* The app detects the header and parses two columns into:

  * `RT(min)` and a **sample column** named from the file stem.

### 3D PDA ASCII

* The export has **wavelengths** in a row following the header.
* In the app, choose a **target wavelength (nm)**; the importer extracts the closest recorded wavelength column for each file and builds a 2D matrix (`RT(min)` + one column per file).

---

## Typical workflow

1. **Upload chromatograms**
   In the sidebar, pick a mode:

   * **2D LabSolutions ASCII (uploads)**
     Upload one or more `.txt` files → the app builds a combined matrix by outer-joining on `RT(min)`.
   * **3D PDA (folder of .txt)**
     Provide a local folder and a target wavelength → the app extracts the wavelength trace per file and builds a matrix.
   * **3D PDA (uploaded .txt)**
     Upload 3D `.txt` files and choose target wavelength → the app extracts and builds a matrix.

2. **Optional metadata & bioactivity**

   * Upload **metadata** (`;` delimiter). Map:

     * *Sample ID column* (what chromatogram columns will be renamed to)
     * *HPLC filename stems* (e.g., `HPLC_filename`) so columns can be renamed from file stems → sample IDs
   * Upload **bioactivity** if you plan to fuse external responses.

3. **Preprocessing**

   * Resample to a grid (step size, min/max RT)
   * Smooth (moving average)
   * Baseline (median or rolling-min)
   * Normalize/scale (max=1, area=1, z-score, etc.)

4. **Referencing & Alignment**

   * **Reference**: shift to a common anchor peak / target position
   * **Alignment**:

     * *Preferred*: **Icoshift** (`pyicoshift`) with selectable segments and reference (median/mean/index/maxcorr)
     * *Fallback*: robust correlation-based alignment when `pyicoshift` is not available

5. **Modeling (PCA / PLS)**

   * **PCA**:

     * Standardize (optional)
     * **Scores plot** with **axes labeled by % explained variance** (PC1/PC2)
     * **Loadings plot**: select any PC and the plot updates; optional |loadings| and smoothing
     * Table of top RTs by |loading|
   * **PLS / PLS-DA**:

     * Choose label column from metadata
     * Scores, X-loadings (p[1], p[2]), per-component R²X / R²Y / Q² bar chart

6. **Visualizations & Exports**

   * Overlay/stacked chromatograms
   * Peak counting & per-sample plots
   * Region selection & **AUC integration** (per-sample HTMLs + summary table)
   * UV spectrum @ RT and UV contour from PDA matrices
   * **MetaboAnalyst**-ready CSV

---

## Alignment notes (Icoshift)

* We try multiple calling conventions to be compatible with different `pyicoshift` versions.
* If `pyicoshift` isn’t present (or fails), the app uses a correlation-based fallback so users can continue working.
* On Streamlit Cloud, confirm `pyicoshift` is installed; if you still see errors, reduce `segments`, verify data isn’t all-NaN, and ensure preprocessing produced finite values.

---

## PCA details

* Uses either scikit-learn PCA or a NIPALS implementation for advanced workflows.
* **Score axes show `% explained variance`** so interpretation is immediate.
* The **Loadings** panel lets you:

  * Pick any PC (1..N)
  * Toggle absolute loadings
  * Smooth with a centered moving average
  * Inspect a ranked table of RTs by |loading|

---

## PLS / PLS-DA details

* Numeric `y` → PLS Regression
* Categorical `y` → PLS-DA (one-hot encoded)
* Visuals:

  * Scores (LV1 vs LV2)
  * X-loadings (p[1], p[2])
  * Bars of R²X, R²Y, Q² per number of components

---

## Repository layout (key files)

* `app_pymetaflow.py` — Streamlit UI & orchestration
* `data_processing_HPLC.py` — parsing, PDA extractors, preprocessing, referencing, alignment (Icoshift wrapper + fallback), modeling (PCA/PLS), visuals, exports
* `requirements.txt` — pinned deps including `pyicoshift` (optional but recommended)

---

## Troubleshooting

* **Icoshift error / not found**

  * Ensure `pyicoshift` installed. On Streamlit Cloud, include it in `requirements.txt`.
  * Try fewer segments; verify there are no NaNs (the app already fills NaNs with 0 for alignment).
* **No metadata labels appear**

  * Confirm your *Sample ID* column actually matches chromatogram column names (after renaming by stems).
* **PCA axes don’t show variance**

  * Make sure you’re in the PCA panel of the Modeling section; the app computes and renders `%` in axis titles.

---

## License

Choose and add your preferred license here (e.g., MIT).

---

## Citation / Acknowledgements

...

---
