
# Primate vision reveals a missing principle for robust dynamic AI

Source code accompanying the paper investigating how artificial neural networks and primate visual cortex process dynamic visual information during object motion discrimination tasks.

## Overview

This repository contains implementations for comparing human behavior, macaque neural recordings, and image/video-based neural networks on dynamic visual perception tasks. We analyze appearance-invariant motion coding in the primate inferior temporal cortex and identify gaps between state-of-the-art video models and biological vision systems.

## Project Structure

```
src/
├── compute_scores/    # Score computation
└── plotting/          # Visualization of results
```

## Features

- **Behavior, Neural Recording, & Model Analysis**: Compute performance scores across human, macaque IT datasets, and artificial neural networks
- **Visualizations**: Generate comparative plots of human predictions, model predictions vs. neural responses

## Usage


### Reproducing Figures 1–8

The plotting notebooks use pre-computed results. Download `scores.zip` from the
[scores download link](SCORES_ZIP_DOWNLOAD_URL), place the archive in the
repository's main folder, and extract it there:

```bash
unzip scores.zip
```

After extraction, the repository should contain the following directories:

```text
dyn-align/
├── scores/
│   ├── figure1/
│   ├── figure2/
│   ├── ...
│   └── figure8/
└── src/
    └── plotting/
        ├── figure1.ipynb
        ├── figure2.ipynb
        ├── ...
        └── figure8.ipynb
```

Install [Anaconda](https://www.anaconda.com) if Conda is not already available. From the repository root, create
the `np` environment using the included `environment.yml` file:

```bash
conda env create --file environment.yml
```

Activate the environment:

```bash
conda activate np
```

With the `np` environment active, start Jupyter from the repository root:

```bash
jupyter lab
```

In Jupyter, open `src/plotting/figure1.ipynb` through
`src/plotting/figure8.ipynb` and run all cells in each notebook. Keep the
notebooks in `src/plotting/` when running them: their paths are relative to that
directory and expect the extracted results at `../../scores/`.


### Computing scores

Activate the Conda environment and change to the score-computation directory.
Run the scripts from this directory because they use relative paths:

```bash
conda activate np
cd src/compute_scores
```

Before running a script, open it and replace its bracketed input-path
placeholders, such as `[your file containing ANN model features]` and
`[your label file]`, with paths to the required HDF5 feature or neural-data
files and label files. Inputs are currently configured in the source files
rather than through command-line arguments.

Run the scripts associated with each figure:

| Figure | Commands |
| --- | --- |
| Figure 1 | `python decode_ann_obj_cat_linear.py` |
| Figure 2 | `python decode_ann_obj_dir_linear.py`<br>`python decode_ann_obj_speed_linear.py` |
| Figures 3–4 | `python decode_ann_obj_dir_linear_rgb2afv.py`<br>`python decode_IT_obj_dir_linear_rgb2afv.py` |
| Figures 6–7 | `python temporal_change_IT.py`<br>`python temporal_change_ANN.py` |
| Figure 8 | `python appearance_motion_factors_IT.py`<br>`python appearance_motion_factors_ANN.py` |

The scripts save HDF5 results beneath the repository-level `scores/`
directory. Some analyses use multiprocessing and can take substantial time.
On a SLURM system, set `SLURM_CPUS_PER_TASK` to the number of CPUs allocated to
the job. For local multiprocessing in scripts that define an `hpc` setting,
change `hpc = True` to `hpc = False`.



## Citation

[Add paper citation here]

## License

This project is licensed under the [MIT License](LICENSE).
