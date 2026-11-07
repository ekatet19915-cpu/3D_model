# Plausible 3D Model from a Single Photo

University project (topic 04/11): creating a textured 3D model of an object from a single photograph.

The project is built on two modern neural image-to-3D models: **Stable Fast 3D (SF3D)** and **TripoSR**. Both models run in the final Gradio app, where their results can be compared on the same photo.

Reconstruction runs in Google Colab on a T4 GPU. The remaining steps (postprocessing, export) can be run locally on CPU.

## Features

- background removal and preparation of the photo for reconstruction;
- 3D model generation with two models (SF3D and TripoSR) and a side-by-side comparison;
- interactive model viewer in the browser;
- postprocessing: mesh cleaning and smoothing, simplification, color correction, material adjustment;
- export to GLB and OBJ.

## Model Comparison

Both models were run on the same 15 test photos (Google Colab, T4 GPU).

| | TripoSR | Stable Fast 3D (SF3D) |
|---|---|---|
| Shape | good, plausible back side (e.g. mug handle) | good |
| Color / texture | colors stored in vertices, noise visible on the surface | real UV texture and material |
| Material | none | makes every model glossy |
| Output | `mesh.obj` | `mesh.glb` |
| Setup notes | `transformers==4.35.0`, `torchmcubes` replaced by a scikit-image shim | patches for transformers 5.x, float16 on T4, license accepted on Hugging Face |
| Timing | `docs/triposr_timing.csv` | `docs/sf3d_timing.csv` |

**Conclusion.** SF3D is the main model of the pipeline because it provides a UV texture and a material. TripoSR is kept as a backup and for comparison. Details are in `docs/model_choice.md`, result screenshots in `docs/screenshots/`.

## Pipeline

Processing follows this chain:

**photo → background removal → input preparation → SF3D → postprocessing → export**

The functions are in `src/pipeline.py`:

| Function | What it does |
|---|---|
| `remove_background(img)` | removes the background with rembg (model `isnet-general-use`), keeps only the largest object |
| `prepare_input(img, as_rgba=False)` | crops to the object, centers it, resizes to 512×512, gray background (or transparent with `as_rgba=True`) |
| `reconstruct(img)` | runs SF3D in Colab, takes PIL RGBA, returns a `trimesh.Trimesh` with UV texture |
| `postprocess(mesh)` | keeps the largest component, centers the model, normalizes the scale |
| `export(mesh, path)` | saves `path.glb` and `path.obj` |

For processing several photos at once there is `src/batch.py` (`run_batch`); results are written to `docs/pipeline_results.csv`.

### Technical Findings

- GLB duplicates vertices along UV seams, so looking for components by vertex index found hundreds of fake pieces (371 for the bottle). Vertices are merged by position first.
- The rembg model `u2net` cuts off white objects on a light background; `isnet-general-use` performs clearly better.
- T4 has no native bfloat16 support, and with it SF3D fails with a CUDA out-of-memory error, so float16 is used.
- transformers 5.x is incompatible with SF3D (see `docs/setup_notes.md`).

## Gradio App

The app is in the notebook `Colab_notebooks/SPMM Gradio_app_FINAL.ipynb`. It runs in Colab and provides a public link (`share=True`). Each model has its own Conda environment (`sf3d_env`, `triposr_env`) because their dependencies conflict.

What the app can do:

- photo upload;
- preprocessing: background removal with alpha matting, edge erosion, mask thresholds, padding, rotation, brightness, contrast, sharpness, color temperature and tint, choice of background type and input size;
- generation of two models (SF3D and TripoSR) shown side by side; for TripoSR the marching cubes resolution can be set (128, 256 or 320);
- interactive 3D view (`gr.Model3D`) of the raw and the processed model;
- download of the result as GLB.

## Quality Improvements

Postprocessing runs separately for each model:

- **Mesh cleaning:** vertex welding, removal of small components, hole filling, orientation fix.
- **Mesh simplification** (decimation) to a target number of faces.
- **Taubin smoothing:** reduces the surface noise of TripoSR without shrinking the model.
- **New UV unwrap** (xatlas) and texture re-baking at 1024, 2048 or 4096 px.
- **Color correction:** brightness, contrast and saturation (texture for SF3D, vertex colors for TripoSR).
- **Material adjustment:** metallic and glossiness sliders (roughness = 1 − glossiness). This fixes the SF3D problem of making every object glossy.

## Results and Limitations

All 15 photos were run through the pipeline. Results and screenshots are in `docs/screenshots/` and `Pictures/screenshots/`, final photos in `Pictures/final_photos/`. Heavy files (GLB, OBJ, MP4) are stored on Google Drive, not in Git.

Known limitations:

- a light object on a light background loses part of the mask (e.g. the white sneaker);
- only the largest object in the photo is reconstructed;
- the back side of the object is the model's guess, not exact geometry;
- SF3D sometimes gives a patchy texture (gold bird) and an overly glossy look, TripoSR gives only vertex colors and a noisy surface;
- the environment is sensitive to library versions, and a GPU in Colab is required.

## How to Run (Google Colab)

1. Open the notebook `Colab_notebooks/SPMM Gradio_app_FINAL.ipynb` and enable a T4 GPU.
2. Run the first cell (condacolab) **twice**, until it prints "Everything looks OK!".
3. Install SF3D (about 13 minutes) and TripoSR (about 4 minutes), each into its own Conda environment.
4. Log in to Hugging Face: add the token `HF_TOKEN` to Colab secrets and accept the license of `stabilityai/stable-fast-3d` on its model page.
5. Run the cell `!python /content/app.py` and open the public Gradio link.

The first run takes longer because the models download their weights.

> **Note about TripoSR.** Do not run `pip install -r requirements.txt` as is; use the workarounds from `docs/setup_notes.md`.

## Repository Structure

| Folder | Contents |
|---|---|
| `src/` | pipeline and batch processing code |
| `Colab_notebooks/`, `notebooks/` | Colab notebooks |
| `data/test/` | 15 test photos |
| `source_photos/`, `Pictures/` | source photos and result screenshots |
| `docs/` | notes, timings, screenshots |
| `tests/` | tests |
