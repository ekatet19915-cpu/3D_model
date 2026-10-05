# Setup notes (Colab, Python 3.13, torch 2.11, CUDA 13)

- `torchmcubes` does not build -> replaced by a scikit-image shim `TripoSR/torchmcubes.py`.
- `transformers` must be 4.35.0 for TripoSR (newer versions rename DINO weight keys).
- Packages pinned in requirements.txt do not build on Python 3.13 -> install unpinned versions.
- numba conflict warnings (cuml, cudf, pytensor) can be ignored.
- Stable Fast 3D: texture_baker built OK, but dinov2.py is incompatible with transformers 5.x.

## Stable Fast 3D setup (Colab, Python 3.13, torch 2.11)
- `texture_baker` and `uv_unwrapper` build with `pip install --no-build-isolation ./texture_baker/ ./uv_unwrapper/`.
- Pinned versions in requirements.txt do not build -> install unpinned. `gpytoolbox` does not build -> empty stub package (remeshing not used).
- Gated model: accept the license on huggingface.co/stabilityai/stable-fast-3d and log in with HF_TOKEN.
- `sf3d/models/tokenizers/dinov2.py`: stubs for removed `find_pruneable_heads_and_indices`, `prune_linear_layer`; `get_head_mask` replaced by `[None] * num_hidden_layers`.
- `run.py`: `torch.bfloat16` -> `torch.float16` (T4 has no bfloat16; otherwise CUDA out of memory).
