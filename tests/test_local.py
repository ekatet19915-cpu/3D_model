import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trimesh
from src.pipeline import postprocess, export

os.makedirs("outputs", exist_ok=True)
mesh = trimesh.load("docs/screenshots/bottle.glb", force="mesh")
result = postprocess(mesh)
export(result, "outputs/bottle")
print("Done:", len(mesh.faces), "->", len(result.faces), "faces")
