"""Core pipeline functions (step 3).

Local CPU part: remove_background, prepare_input, postprocess, export.
reconstruct (SF3D, GPU) is added later in Colab.
"""
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

_REMBG_SESSIONS = {}  # rembg models are loaded once and reused
_SF3D_MODEL = None     # SF3D is loaded once and reused


def remove_background(img: Image.Image, keep_largest: bool = True,
                      model_name: str = "isnet-general-use") -> Image.Image:
    """PIL RGB/RGBA -> PIL RGBA with transparent background.

    keep_largest: keep only the biggest object (removes stray bits of
    background and secondary objects, e.g. jeans next to a sneaker).
    """
    from rembg import new_session, remove

    if model_name not in _REMBG_SESSIONS:
        _REMBG_SESSIONS[model_name] = new_session(model_name)

    out = remove(img.convert("RGB"), session=_REMBG_SESSIONS[model_name]).convert("RGBA")

    if keep_largest:
        alpha = np.array(out.getchannel("A"))
        labels, n = ndimage.label(alpha > 128)
        if n > 1:
            sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
            keep = labels == (np.argmax(sizes) + 1)
            alpha = np.where(keep, alpha, 0).astype(np.uint8)
            out.putalpha(Image.fromarray(alpha))
    return out


def prepare_input(img: Image.Image, size: int = 512, fill: float = 0.85,
                  bg=(127, 127, 127), as_rgba: bool = False) -> Image.Image:
    """PIL RGBA -> PIL RGB (size x size), object centered on gray background.

    fill: share of the square that the longest side of the object takes.
    as_rgba: return RGBA with transparent background instead (this is the
    form SF3D needs; the gray RGB version is for showing in the app).
    """
    img = img.convert("RGBA")
    alpha = np.array(img.getchannel("A"))
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        raise ValueError("No object found in the image (empty mask).")

    # crop to the object and scale so the longest side = size * fill
    obj = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    scale = size * fill / max(obj.size)
    new_size = (max(1, round(obj.width * scale)), max(1, round(obj.height * scale)))
    obj = obj.resize(new_size, Image.LANCZOS)

    # paste into the center of a gray square (alpha used as mask)
    offset = ((size - obj.width) // 2, (size - obj.height) // 2)
    if as_rgba:
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(obj, offset)
        return canvas
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(obj, offset, obj.getchannel("A"))
    return canvas


def _largest_component_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    """Boolean mask of faces that belong to the largest connected part.

    GLB meshes duplicate vertices along UV seams, so we first look at
    vertices merged by position; otherwise one object falls apart into
    hundreds of fake pieces.
    """
    # map every vertex to an id based on its position
    rounded = np.round(mesh.vertices, 5)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    faces = inverse[mesh.faces]  # faces using merged vertex ids

    # connected components over merged vertices (graph of triangle edges)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = inverse.max() + 1
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)

    face_labels = labels[faces[:, 0]]
    # largest part = the one with the biggest surface area
    areas = np.bincount(face_labels, weights=mesh.area_faces)
    return face_labels == np.argmax(areas)


def reconstruct(img: Image.Image, sf3d_path: str = "/content/stable-fast-3d",
                texture_size: int = 1024) -> trimesh.Trimesh:
    """PIL RGBA (object on transparent background) -> trimesh.Trimesh with texture.

    Runs ONLY in Colab with a GPU. sf3d_path = folder where the
    stable-fast-3d repository was cloned.
    """
    global _SF3D_MODEL
    import sys
    import torch

    if _SF3D_MODEL is None:
        if sf3d_path not in sys.path:
            sys.path.insert(0, sf3d_path)
        from sf3d.system import SF3D

        _SF3D_MODEL = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        _SF3D_MODEL.to("cuda")
        _SF3D_MODEL.eval()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mesh, _ = _SF3D_MODEL.run_image(
                [img.convert("RGBA")],
                bake_resolution=texture_size,
                remesh="none",
            )
    return mesh


def postprocess(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Keep the largest part, center it, normalize size, keep texture."""
    mesh = mesh.copy()

    # 1) keep only the largest connected part (texture is preserved)
    mesh.update_faces(_largest_component_mask(mesh))
    mesh.remove_unreferenced_vertices()

    # 2) center at origin
    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    # 3) normalize scale: longest side = 1
    scale = 1.0 / mesh.extents.max()
    mesh.apply_scale(scale)
    return mesh


def export(mesh: trimesh.Trimesh, path: str) -> None:
    """Save mesh as <path>.glb (main) and <path>.obj."""
    mesh.export(f"{path}.glb")
    mesh.export(f"{path}.obj")