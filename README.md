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

- **Main model: TripoSR.** Fast (about 7 s per photo on a T4), MIT license, good shape
  on the tested mug (handle and back side are plausible). Colors are stored in vertices
  (no UV texture yet) and the surface is slightly noisy.
- **Stable Fast 3D: not used.** We got access to the weights, but its code is incompatible
  with the newest Colab libraries (transformers 5.x, torch 2.11, Python 3.13).
- Details: `docs/model_choice.md`, `docs/setup_notes.md`, timing in `docs/triposr_timing.csv`.

> **SK:** Hlavný model je TripoSR (rýchly, licencia MIT, dobrý tvar). Farba je zatiaľ vo
> vrcholoch (nie UV textúra), povrch je mierne zrnitý. Stable Fast 3D sme nepoužili
> kvôli nekompatibilite s najnovšími knižnicami v Colabe.

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
