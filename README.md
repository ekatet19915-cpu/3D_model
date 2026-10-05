# Plausible 3D model from a single photo

University project (topic 04/11): textured 3D model from one photograph.
Reconstruction runs in Google Colab (GPU), the rest can run locally on CPU.

> **SK:** Školský projekt (téma 04/11): textúrovaný 3D model z jednej fotografie.
> Rekonštrukcia beží v Google Colabe (GPU), zvyšok sa dá robiť lokálne.

## Interface agreements (`src/pipeline.py`)

| Function | Input | Output |
|---|---|---|
| `remove_background(img)` | PIL RGB/RGBA | PIL RGBA |
| `prepare_input(img)` | PIL RGBA | PIL RGB 512x512, centered, gray background |
| `reconstruct(img, model="triposr" or "sf3d")` | PIL RGB | `trimesh.Trimesh` |
| `postprocess(mesh)` | `trimesh.Trimesh` | `trimesh.Trimesh` |
| `export(mesh, path)` | `trimesh.Trimesh`, path without extension | `.glb` and `.obj` |

## Rules
- Reconstruction (CUDA) runs only in Colab.
- Heavy files (GLB, OBJ, MP4) stay on Google Drive, not in Git.
- `git pull` before work, commit and push after. Names and commits in English.
- Notebooks: `A_name.ipynb` / `B_name.ipynb`.

> **SK:** Pred prácou `git pull`, po práci commit a push. Názvy súborov a commity po anglicky.
> Ťažké súbory patria na Google Drive, nie do Gitu.


## Phase 1 results (model choice)

We tested two image-to-3D models on the same 15 test photos (`data/test/`).

| | TripoSR | Stable Fast 3D (SF3D) |
|---|---|---|
| Shape | good, plausible back side (e.g. mug handle) | good |
| Color / texture | colors stored in vertices; **adds noise to the surface texture** | real UV texture and material |
| Material | none | **makes every model glossy, which does not fit all objects** (e.g. matte items) |
| Output | `mesh.obj` | `mesh.glb` |
| Timing | `docs/triposr_timing.csv` | `docs/sf3d_timing.csv` |

Screenshots of the results for both models: `docs/screenshots/`
(`<photo>_triposr.png`, `<photo>_sf3d.png`). Sample meshes: `outputs/examples/`.

**Conclusion:** neither model is perfect. TripoSR gives a noisy surface, SF3D a uniformly glossy
look. Both problems can be reduced in our own steps (surface smoothing in `postprocess`,
material correction in phase 4). The final choice for `reconstruct` is described in
`docs/model_choice.md`.

> **SK:** Vyskúšali sme dva modely na rovnakých 15 fotkách. TripoSR pridáva do textúry šum,
> SF3D robí všetky modely lesklé, čo sa nehodí pre každý predmet (napr. matné). Výsledky a
> snímky obrazovky sú v `docs/screenshots/`.

## Notes for phase 2

1. **Reconstruction runs only in Colab (GPU).** `postprocess` and `export` work on any
   `trimesh.Trimesh`, so they can be developed locally on CPU using sample meshes
   (`mesh.obj`) from Google Drive: `3D_model_outputs/triposr/<photo_name>/0/mesh.obj`.
2. **TripoSR environment is fragile.** Do not run `pip install -r requirements.txt` as is.
   Required workarounds: `transformers==4.35.0`, and a `torchmcubes.py` shim based on
   scikit-image placed in the TripoSR folder. See `docs/setup_notes.md`.
3. **TripoSR output:** `mesh.obj` with vertex colors, `input.png` (background removed).
   TripoSR removes the background itself (rembg) inside `run.py`.
4. **Known problems to solve in postprocess:** noisy surface (smooth it), several
   disconnected components (keep the largest), no UV texture (planned for phase 4).
5. **Interfaces must stay as agreed** (see "Interface agreements"): PIL images and
   `trimesh.Trimesh`, names of functions in `src/pipeline.py`.
6. **Open task:** compare `rembg` vs `BiRefNet` for `remove_background`
   (result goes to `docs/bg_comparison.md`).
7. Heavy files (GLB, OBJ, MP4) stay on Google Drive, not in Git.

> **SK:** Rekonštrukcia beží len v Colabe. `postprocess` a `export` môžeš vyvíjať lokálne
> na CPU na vzorových sieťach z Google Drive. Neinštaluj TripoSR podľa
> `requirements.txt` (pozri `docs/setup_notes.md`). Otvorená úloha: porovnanie
> `rembg` a `BiRefNet`.
