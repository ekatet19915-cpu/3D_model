# Plausible 3D model from a single photo

University project (topic 04/11): textured 3D model from one photograph. Reconstruction runs in Google Colab (GPU), the rest can run locally on CPU.

SK: Školský projekt (téma 04/11): textúrovaný 3D model z jednej fotografie. Rekonštrukcia beží v Google Colabe (GPU), zvyšok sa dá robiť lokálne.

## Interface agreements (src/pipeline.py)

| Function | Input | Output |
|---|---|---|
| `remove_background(img, keep_largest=True, model_name="isnet-general-use")` | PIL RGB/RGBA | PIL RGBA |
| `prepare_input(img, as_rgba=False)` | PIL RGBA | PIL RGB 512x512, centered, gray background (with `as_rgba=True`: PIL RGBA 512x512, transparent background) |
| `reconstruct(img)` | PIL RGBA (transparent background) | `trimesh.Trimesh` |
| `postprocess(mesh)` | `trimesh.Trimesh` | `trimesh.Trimesh` |
| `export(mesh, path)` | `trimesh.Trimesh`, path without extension | `.glb` and `.obj` |

`reconstruct` always uses SF3D. TripoSR was only used for the comparison in phase 1.

## Rules

- Reconstruction (CUDA) runs only in Colab.
- Heavy files (GLB, OBJ, MP4) stay on Google Drive, not in Git.
- `git pull` before work, commit and push after. Names and commits in English.
- Notebooks: `A_name.ipynb` / `B_name.ipynb`.

SK: Pred prácou git pull, po práci commit a push. Názvy súborov a commity po anglicky. Ťažké súbory patria na Google Drive, nie do Gitu.

## Phase 1 results (model choice)

We tested two image-to-3D models on the same 15 test photos (`data/test/`).

| | TripoSR | Stable Fast 3D (SF3D) |
|---|---|---|
| Shape | good, plausible back side (e.g. mug handle) | good |
| Color / texture | colors stored in vertices; adds noise to the surface texture | real UV texture and material |
| Material | none | makes every model glossy, which does not fit all objects (e.g. matte items) |
| Output | `mesh.obj` | `mesh.glb` |
| Timing | `docs/triposr_timing.csv` | `docs/sf3d_timing.csv` |

Screenshots of the results for both models: `docs/screenshots/` (`<photo>_triposr.png`, `<photo>_sf3d.png`). Sample meshes: `outputs/examples/`.

**Conclusion:** neither model is perfect. TripoSR gives a noisy surface, SF3D a uniformly glossy look. Both problems can be reduced in our own steps (surface smoothing in `postprocess`, material correction in phase 4). The final choice for `reconstruct` is described in `docs/model_choice.md`.

SK: Vyskúšali sme dva modely na rovnakých 15 fotkách. TripoSR pridáva do textúry šum, SF3D robí všetky modely lesklé, čo sa nehodí pre každý predmet (napr. matné). Výsledky a snímky obrazovky sú v `docs/screenshots/`.

## Phase 2 results (core pipeline)

The whole chain works: photo -> background removed -> prepared input -> SF3D -> postprocess -> GLB and OBJ.

| Function | Where it runs | What it does |
|---|---|---|
| `remove_background(img)` | CPU | rembg, model `isnet-general-use`, keeps only the largest object |
| `prepare_input(img, as_rgba=False)` | CPU | crops to the object, centers it, 512x512. Gray background by default, transparent with `as_rgba=True` |
| `reconstruct(img)` | Colab GPU | SF3D, the model is loaded once. Input: PIL RGBA with transparent background. Output: `trimesh.Trimesh` with UV texture |
| `postprocess(mesh)` | CPU | keeps the largest connected part, centers the mesh, longest side = 1, texture is kept |
| `export(mesh, path)` | CPU | saves `path.glb` and `path.obj` |

Batch run on several photos: `src/batch.py` (`run_batch`). For every photo it saves `input.png`, `nobg.png` and the model to Google Drive and writes `docs/pipeline_results.csv` (status, time, number of faces).

### Changes to the interface agreements

- `reconstruct` takes PIL RGBA (transparent background), not RGB with gray background. SF3D takes the shape of the object from the alpha channel. The gray RGB version is only for showing the prepared input in the app.
- `reconstruct` has no `model` argument any more, it always uses SF3D.
- `remove_background` has the optional argument `model_name`.

### What we found out

- GLB files duplicate vertices along UV seams. Looking for components by vertex index finds hundreds of fake pieces (371 for the bottle), so `postprocess` merges vertices by position first. The separate bottle cap is removed, the texture stays.
- rembg model `u2net` cuts off white objects on a light background (white sneaker on a light sweater). `isnet-general-use` is clearly better, but the sneaker mask is still not perfect. This goes to the limitations.
- T4 has no native bfloat16. With bfloat16 SF3D fails with CUDA out of memory, so `reconstruct` uses float16 on T4.
- transformers 5.x does not work with SF3D (see `docs/setup_notes.md`, section "SF3D in Colab").

### Known problems for phase 4 and 5

- SF3D makes the surface too glossy, some textures are patchy (shiny gold bird).
- Light objects on a light background lose a part of the mask (sneaker).
- Only the largest object is kept, so a scene with two objects gives one model.
- `postprocess` does not smooth the surface yet and does not fix the orientation.

SK: Celý reťazec funguje: fotografia -> odstránenie pozadia -> príprava vstupu -> SF3D -> postprocessing -> GLB a OBJ. `reconstruct` teraz berie RGBA obrázok s priehľadným pozadím a vždy používa SF3D. Na viac fotiek naraz slúži `src/batch.py`. Známe problémy: lesklý materiál SF3D, svetlý predmet na svetlom pozadí (tenisky) a len jeden najväčší objekt v zábere. Inštalácia SF3D v Colabe je v `docs/setup_notes.md`.

## How to run (Colab)

1. Open a Colab notebook with a T4 GPU and install everything as described in `docs/setup_notes.md` (section "SF3D in Colab"). Restart the runtime when Colab asks.
2. Clone the repository, log in to Hugging Face (token `HF_TOKEN` in Colab secrets) and mount Google Drive.
3. Run one photo:

```python
import sys
sys.path.insert(0, "/content/3D_model")

from PIL import Image
from src.pipeline import remove_background, prepare_input, reconstruct, postprocess, export

photo = Image.open("/content/3D_model/foto-bottle-medium.jpeg")
rgba = remove_background(photo)
mesh = reconstruct(prepare_input(rgba, as_rgba=True))
mesh = postprocess(mesh)
export(mesh, "/content/drive/MyDrive/3D_model_outputs/bottle")
```

4. Run several photos at once:

```python
from src.batch import run_batch
run_batch(["/content/3D_model/foto-bird-hard.jpg"],
          "/content/drive/MyDrive/3D_model_outputs/pipeline_batch",
          csv_path="/content/3D_model/docs/pipeline_results.csv")
```

`postprocess` and `export` also work locally on CPU (install `trimesh numpy scipy pillow`), `remove_background` additionally needs `rembg[cpu]`.

## Notes about TripoSR (phase 1 only)

- TripoSR environment is fragile. Do not run `pip install -r requirements.txt` as is. Required workarounds: `transformers==4.35.0`, and a `torchmcubes.py` shim based on scikit-image placed in the TripoSR folder. See `docs/setup_notes.md`.
- TripoSR output: `mesh.obj` with vertex colors, `input.png` (background removed). TripoSR removes the background itself (rembg) inside `run.py`.
- Sample meshes from phase 1 are on Google Drive: `3D_model_outputs/triposr/<photo_name>/0/mesh.obj`.

## Open tasks

- Compare rembg vs BiRefNet for `remove_background` (result goes to `docs/bg_comparison.md`). `isnet-general-use` is used for now.
- Material and texture improvements, smoothing, orientation (phase 4).

## Next: phase 3 (Gradio app)

Upload a photo, show intermediate steps (original -> no background -> prepared input -> depth map), interactive 3D view, download of GLB and OBJ, turntable animation.
