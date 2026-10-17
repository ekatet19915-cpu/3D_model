import trimesh
from src.pipeline import postprocess, export

mesh = trimesh.load("bottle.glb", force="mesh")
result = postprocess(mesh)
export(result, "outputs/bottle")
print("Done:", len(mesh.faces), "->", len(result.faces), "faces")