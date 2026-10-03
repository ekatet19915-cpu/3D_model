# Setup notes (Colab, Python 3.13, torch 2.11, CUDA 13)

- `torchmcubes` does not build -> replaced by a scikit-image shim `TripoSR/torchmcubes.py`.
- `transformers` must be 4.35.0 for TripoSR (newer versions rename DINO weight keys).
- Packages pinned in requirements.txt do not build on Python 3.13 -> install unpinned versions.
- numba conflict warnings (cuml, cudf, pytensor) can be ignored.
- Stable Fast 3D: texture_baker built OK, but dinov2.py is incompatible with transformers 5.x.
