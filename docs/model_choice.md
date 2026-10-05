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
