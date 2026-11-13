# Example: AI-generated image detection (GenImage + CNNDetection)

Reproduces the protocol of GFRE (He et al., IJCAI-26) / VIB-Net (Zhang et al., CVPR 2025) and tests a
multi-layer variant of feature-reconstruction detection.

**Training settings.**
- **sd14:** 16k fake + 16k real images sampled (seed 0) from GenImage SDv1.4 train. Only `*_sdv4_*` fakes are used; some copies of that folder also contain `*_sdv5_*` files, which are excluded. Validation is 1k + 1k held out from the same pool.
- **progan:** CNNDetection ProGAN, all 20 classes. Validation is the official CNNDetection val set.

**Test sets (18).**
- 11 from CNNDetection: ProGAN, BigGAN, CycleGAN, StarGAN, StyleGAN, GauGAN, CRN, IMLE, SITD, SAN, Deepfake.
- 7 GenImage val splits: ADM, GLIDE, SDv1.4, SDv1.5, VQDM, Midjourney, Wukong.

| File | What |
|---|---|
| `src/make_splits.py` | Writes `frozen/splits/{train,test}/*.tsv` from `paths.json`, with leakage checks and a manifest |
| `src/unpack_progan.sh` | Unpacks `progan_train.7z.001-007` (a 7z holding a zip) into 20 class folders |
| `src/extract_features.py` | Caches CLIP ViT-L/14 CLS tokens from all 24 blocks plus the final embedding (fp16 memmaps) |
| `src/extract_all.sh` | Runs the extractor for every frozen split |
| `src/train_recon.py` | GFRE-style detector: per-source autoencoder trained on fake features only, then an MLP on the residuals; writes `scores/` only |
| `src/parse_gfre_tables.py` | Copies GFRE Tables 1–2 into `frozen/baselines.json` and flags rows whose printed average differs from the mean of their columns |
| `paths.example.json` | Dataset roots to fill in |

Typical sequence (on the server, inside the project):

```bash
python harness/launch.py splits --tag util --gpu none -- bash -c '$PY src/make_splits.py --part sd14 && $PY src/make_splits.py --part test'
python harness/freeze.py add frozen/splits
python harness/launch.py feats --tag util --gpu auto -- bash src/extract_all.sh
python harness/launch.py gfre_repro_s0 --tag gfre_repro --setting sd14 -- python src/train_recon.py --setting sd14 --source embed
python harness/launch.py ml_a --tag multilayer --setting sd14 -- python src/train_recon.py --setting sd14 --source layers --layers 12,16,20,24
python harness/gate.py --setting sd14
```
