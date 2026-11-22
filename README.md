# CoCO-DGR (Diff-Gaussian-Rasterization Fork)

Modified CUDA differentiable rasterizer for [CoCO-Train](https://github.com/RareIcubu/CoCO-Train), forked from [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) (`3dgs_accel` branch).

## Modifications

- **Opacity-Aware Tight AABB (`COCO_TIGHT_AABB`):** Replaces the default isotropic 3-sigma square screen-space bounding box in the CUDA forward pass (`cuda_rasterizer/forward.cu`, `rasterizer_impl.cu`) with a tight axis-aligned rectangular bounding box scaled by each splat's opacity ($\sqrt{-2 \ln(\tau_\alpha / o)}$). Reduces false-positive tile-Gaussian pairs during binning and sorting without altering rasterization quality.
- **Runtime Toggle:** Controlled via the `COCO_TIGHT_AABB` environment variable (`1` = enabled, `0` = legacy square bounding box), allowing direct A/B ablation without recompiling CUDA kernels.

## Build & Usage

Intended to be compiled as a submodule of `CoCO-Train` inside the `gaussian_splatting` Conda environment:

```bash
pip install --no-build-isolation -e .
```

Enable tight bounding boxes at runtime (handled automatically by `CoCO-Train`'s `--tight_aabb` flag):

```bash
COCO_TIGHT_AABB=1 python train.py ...
```

## Attribution

Based on the original differentiable Gaussian rasterizer by Kerbl et al. (2023) and the `3dgs_accel` backend from Mallick et al. (*Taming 3DGS*, 2024). Opacity-aware tight rectangular bounding boxes follow Radl et al. (*StopThePop*, 2024), Hanson et al. (*Speedy-Splat*, 2025), and Hahlbohm et al. (*Faster-GS*, 2026). Original license is preserved in [LICENSE.md](LICENSE.md).
