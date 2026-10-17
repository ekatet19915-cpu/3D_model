"""Run the whole pipeline on several photos (step 3.6). Colab (GPU) only.

For every photo it saves, into <out_root>/<photo_name>/:
  input.png   - prepared input (gray background, 512x512)
  nobg.png    - photo with the background removed
  model.glb / model.obj (+ texture)
and appends one row to <csv_path> (time, faces, status).
"""
import csv
import os
import time

from PIL import Image

from src import pipeline


def run_batch(photo_paths, out_root, csv_path="docs/pipeline_results.csv",
              reconstruct_fn=None):
    reconstruct_fn = reconstruct_fn or pipeline.reconstruct
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    rows = []

    for path in photo_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        row = {"photo": name, "status": "ok", "seconds": "", "faces": "", "error": ""}
        t0 = time.time()
        try:
            photo = Image.open(path)
            rgba = pipeline.remove_background(photo)
            rgba.save(os.path.join(out_dir, "nobg.png"))
            pipeline.prepare_input(rgba).save(os.path.join(out_dir, "input.png"))

            mesh = reconstruct_fn(pipeline.prepare_input(rgba, as_rgba=True))
            mesh = pipeline.postprocess(mesh)
            pipeline.export(mesh, os.path.join(out_dir, "model"))
            row["faces"] = len(mesh.faces)
        except Exception as e:  # one bad photo must not stop the whole run
            row["status"] = "FAILED"
            row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(row["photo"], row["status"], row["seconds"], "s", row["error"])

    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["photo", "status", "seconds", "faces", "error"])
        if new_file:
            w.writeheader()
        w.writerows(rows)
    return rows