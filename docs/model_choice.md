# Model choice

## Models tested
Both models were run on all 15 test photos in Google Colab (T4 GPU).

| Model | Strengths | Weaknesses |
|---|---|---|
| TripoSR | fast, MIT license, simple setup, good shape | vertex colors only; adds noise to the surface texture |
| Stable Fast 3D | UV texture and material (roughness/metallic) | makes every model glossy, not suitable for all objects; fragile setup (needed patches for transformers 5.x, float16 on T4) |

## Decision
(to be filled in with the team after rating the results together)

## Notes
- Mesh noise (TripoSR) can be reduced by smoothing in `postprocess`.
- Glossy look (SF3D) can be corrected by adjusting material values (phase 4).

> **SK:** Oba modely sme vyskúšali na všetkých 15 fotkách. TripoSR pridáva šum, SF3D robí
> všetko lesklé. Konečný výber doplníme spolu po vyhodnotení.
## SF3D in Colab (works, T4 GPU, Python 3.13)

1. `pip install trimesh rembg onnxruntime`, restart the runtime when Colab asks.
2. `git clone https://github.com/Stability-AI/stable-fast-3d.git /content/stable-fast-3d`
3. Install the SF3D requirements one by one, skipping torch, numpy and
   transformers: jaxtyping, open_clip_torch, pynanoinstantmeshes, gpytoolbox,
   ./texture_baker/, ./uv_unwrapper/
4. `pip install "transformers==4.46.3" "tokenizers>=0.20,<0.21" "huggingface_hub<1.0" --no-deps`,
   then restart the runtime.
   - transformers 5.x breaks SF3D (missing find_pruneable_heads_and_indices).
   - transformers 4.42.3 cannot be built on Python 3.13 (tokenizers).
5. Hugging Face login: token HF_TOKEN in Colab secrets. The license of
   stabilityai/stable-fast-3d must be accepted on its Hugging Face page.
6. T4 has no native bfloat16, so reconstruct uses float16
   (with bfloat16 it fails with CUDA out of memory).
7. Background removal uses the rembg model isnet-general-use
   (u2net cuts off white objects on a light background, e.g. the sneaker).