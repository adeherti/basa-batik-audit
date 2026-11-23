# BASA — Supplementary Data and Code

Supplementary material for the manuscript submitted to *IEEE Access*:

> **Availability Is Not Exploitation: A Two-Stage Audit of Acquisition Shortcuts
> in Batik Image Classification**
> Herti Yani, Hindriyanto Dwi Purnomo, Irwan Sembiring, Yessica Nataliani.

This package contains the audit code and the derived index, split, and result
files behind the tables and figures in the paper: Stage 1 (cue availability),
Stage 2 (cue exploitation across four backbones), and the duplicate/leakage audit. Raw batik images are
**not** redistributed. The public datasets should be downloaded from their
original sources; the private Batik Jambi collection is not shared. All CSV
files reference images by filename and MD5 only.

## Datasets

| Name in paper | Type | Source |
|---|---|---|
| Batik Nitik 960 | public, single-source, controlled acquisition | Mendeley Data; Minarno et al., *Data* 8(4):63, 2023, doi:10.3390/data8040063 |
| DION (Indonesian Batik Motifs) | public, assembled from web sources | Kaggle: `dionisiusdh/indonesian-batik-motifs` (983 catalogued, 981 readable) |
| Batik Jambi | private, assembled | not redistributed (third-party copyright) |

## Contents

```
code/
  leakage_audit.py            duplicate / near-duplicate and split-leakage audit
  validate_leakage_audit.py   generator for the synthetic benchmark with planted duplicates
  basa_leakage_audit.ipynb    notebook walkthrough of the audit
  basa_cue_reliance_multiarch.ipynb   Stage 2: training + test-time corruption, 4 backbones (with run outputs)
data/
  index/                      per-image index: filename, label, size, bytes, MD5 (+ dup_group in *_grouped)
  splits/                     leakage-free grouped train/test splits, 5 seeds per dataset
  duplicate_audit/            duplicate edges, cross-label pairs, per-seed leakage under a random split
results/
  probe_a_*, table2_*, table_new_A_*   Stage 1 (cue availability): acquisition-statistics classifier
  cue_reliance_multiarch_*             Stage 2 (cue exploitation): 4 backbones x 3 datasets x 3 seeds
figures/
  fig_cue_reliance_multiarch.png       retention ratios per condition, dataset and backbone
validation/                   outputs of the audit on the synthetic benchmark
```

## File-to-paper mapping

| File | What it backs |
|---|---|
| `results/probe_a_raw.csv` | Probe A per-seed accuracies (all features, colour only, resolution only, permutation control) |
| `results/table2_probe_a.csv` | Probe A mean and SD per dataset, including DION FULL / LOOSE / STRICT |
| `results/table_new_A_cue_availability.csv` | Cue availability as multiples of chance and dominant available cue |
| `results/cue_reliance_multiarch_raw.csv` | Stage 2 per-run test accuracy (%) under COLOR, GRAY, BLUR, SHUFFLE, PHASE; 36 runs, all converged |
| `results/cue_reliance_multiarch_summary.csv` | Stage 2 mean COLOR accuracy and retention ratios (condition / COLOR) per backbone and dataset |
| `results/nitik_leakage_test.csv` | Nitik COLOR accuracy under a random split (LEAKY) vs. a rotation-grouped split (GROUP): 100.0% in both |
| `figures/fig_cue_reliance_multiarch.png` | Retention ratios plotted per dataset and backbone |
| `data/duplicate_audit/ALL_summary.csv` | Duplicate audit summary per dataset |
| `data/duplicate_audit/ALL_leak.csv` | Mean test-set leakage under a random 70/30 split (Nitik 82.1%, DION 10.7%, Jambi 1.6%) |
| `data/duplicate_audit/*_leak_under_paper_split.csv` | Per-seed leakage under a random split |
| `data/duplicate_audit/*_dup_edges.csv`, `dion_crossclass_pairs.csv` | Detected duplicate pairs and cross-label collisions |
| `data/splits/*_grouped_splits.csv` | Grouped splits in which all members of a duplicate/rotation group fall on the same side |
| `validation/fake_*` | Synthetic validation: all planted duplicates recovered, no false positives |

## Stage 2 protocol

One classifier per backbone (ConvNeXt-T, ResNet-50, ViT-B/16, Swin-T) is
fine-tuned on intact colour images and then evaluated, without retraining, on
the same test images under five conditions: COLOR (intact), GRAY (colour
removed), BLUR (fine detail removed), SHUFFLE (patch permutation, global layout
removed) and PHASE (Fourier phase scrambled, spatial structure removed while
the amplitude spectrum is kept). Retention is the condition accuracy divided by
the COLOR accuracy of the same model and split.

Training uses AdamW (lr 1e-4), 10 epochs, 224x224 inputs, ImageNet-pretrained
torchvision weights, and a stratified 70/30 split per seed (seeds 0, 1, 2).
Run `code/basa_cue_reliance_multiarch.ipynb` on Kaggle with GPU after adjusting
the dataset paths in cell M1; it writes the two `cue_reliance_multiarch_*` CSVs
and the figure.

For Nitik, the grouped split keeps all rotations of one fabric crop together.
This removes rotation leakage but not acquisition-level dependence, since all 60
classes come from a single photographic session.

## Running the audit

Dependencies: `numpy`, `pandas`, `pillow`, `scipy`.

Edit the dataset roots in `DATASETS` at the top of `code/leakage_audit.py`, then:

```bash
python code/leakage_audit.py --dataset nitik --stage all
python code/leakage_audit.py --dataset dion  --stage all
python code/leakage_audit.py --dataset jambi --stage all
```

To reproduce the synthetic validation, set `ROOT` in
`code/validate_leakage_audit.py` to a writable folder, run it to generate the
benchmark and `planted.json`, then run the audit on that folder.

## Contact

Correspondence: `adeherti@unama.ac.id`.
