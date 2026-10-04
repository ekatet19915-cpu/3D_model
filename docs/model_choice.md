# Model choice

## Decision
**Main model: TripoSR.** Fast (about 7 s per photo on a T4), MIT license,
simple installation, good shape on simple tested objects (e.g. mug with handle).

## Candidates
| Model | Result |
|---|---|
| TripoSR | Works. Colors stored in vertices (no UV texture yet), surface is slightly noisy. |
| Stable Fast 3D | Not used. Weights access obtained, but the code is incompatible with the newest Colab libraries (transformers 5.x, torch 2.11, Python 3.13); we stopped to save time. |

## Notes
- Mesh noise will be reduced in postprocessing.
- UV texture baking is planned for phase 4.

> **SK:** Hlavný model je TripoSR (rýchly, licencia MIT, dobrý tvar). Stable Fast 3D
> sme nepoužili kvôli nekompatibilite s najnovšími knižnicami v Colabe.
