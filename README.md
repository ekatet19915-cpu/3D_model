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
