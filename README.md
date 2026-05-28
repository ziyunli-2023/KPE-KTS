# A Kinetic Energy Perspective of Flow Matching

Reference implementation of **Kinetic Trajectory Shaping (KTS)** from *"A Kinetic Energy Perspective of Flow Matching"* (ICML 2026 Spotlight, top 2.2%) by Ziyun Li, Huancheng Hu, Soon Hoe Lim, *et al.*


## Algorithm

KTS modifies the flow-matching ODE trajectory with a time-dependent gain `η(t)`:

- **Launch phase** (`t < τ_split`):     `η(t) = 1 + α(t)` — increase kinetic energy
- **Soft-landing phase** (`t ≥ τ_split`): `η(t) = 1 − β(t)` — decrease kinetic energy

ODE update: `x_{i+1} = x_i + η(t_i) · dt · v_θ(x_i, t_i)`. Default schedules: linear `α(t)`, exponential `β(t)`, `k = 3.0`, `τ_split = 0.6`. 

## Repository structure

```
Experiments/
├── environment.yml / requirements.txt
└── src/
    ├── Utils/
    │   ├── FlowMatching.py    # Flow matching + KTS samplers (core algorithm)
    │   ├── Unet.py            # U-Net architecture (Ho et al. 2020 style)
    │   └── cfg.py / loader.py / Plot.py / calc.py
    ├── Generation/
    │   └── generate.py        # apply KTS to a trained FM model
    └── Evaluation/
        ├── compute_FID.py
        ├── compute_fmem.py
        ├── compute_kinetic_energy.py
        └── generate_fid_stats.py
```

## Installation

```bash
cd Experiments
conda env create -f environment.yml      # GPU (CUDA 11.8)
conda activate memorization
# or:  pip install -r requirements.txt
```

## Prerequisites

This release ships the algorithm only. Users provide their own:

- **Trained Flow Matching checkpoint.** Any U-Net velocity network compatible with `Experiments/src/Utils/Unet.py`. The training infrastructure we used is built on top of [Bonnaire et al. — *Why Diffusion Models Don't Memorize*](https://arxiv.org/abs/2505.17638).
- **Preprocessed dataset tensor** at the path expected by `Experiments/src/Utils/cfg.py` (e.g. `CelebA32_all.pt`).
- **FID reference statistics** — `generate_fid_stats.py` will build them from raw CelebA test images.
- **Output directory.** All scripts write to `Experiments/Saves/<model_dir>/{Models, Samples-*, FID, Memorization}`. Create the directory (or symlink it to external storage) wherever you want; it is gitignored.

## Citation

```bibtex
@inproceedings{li2026kinetic,
  title     = {A Kinetic Energy Perspective of Flow Matching},
  author    = {Li, Ziyun and Hu, Huancheng and Lim, Soon Hoe and
               Li, Xuyu and Gao, Fei and Diao, Enmao and Ding, Zezhen and
               Vazirgiannis, Michalis and Bostr{\"o}m, Henrik},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgement

The U-Net, data loaders, and the F<sub>mem</sub> / FID evaluation utilities are built on top of [Bonnaire et al. — *Why Diffusion Models Don't Memorize*](https://github.com/tbonnair/Why-Diffusion-Models-Don-t-Memorize ). The Flow Matching trainer/sampler and KTS algorithm are new contributions from this work.

## License

MIT 

