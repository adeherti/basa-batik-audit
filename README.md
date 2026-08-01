# BASA — Batik Acquisition-Shortcut Audit

Reproducibility repository for the paper:

> **BASA: What Do Networks Learn When Accuracy Saturates? Auditing Acquisition
> Shortcuts and Contamination in Batik Image Datasets**
> Herti Yani, Hindriyanto Dwi Purnomo, Irwan Sembiring, Yessica Nataliani.

This repository releases the **audit code** and all **derived index, split, and
result files** that back the tables and figures in the paper. Raw batik images are
**not** redistributed: the public datasets are linked below and should be
downloaded from their original sources; the private Jambi set is not shared.

---

## Datasets (download from original sources)

| Name in paper | Type | Source |
|---|---|---|
| Batik Nitik 960 | public, single-source clean control | Mendeley Data — Minarno et al., *Data* 8(4):63, 2023, doi:10.3390/data8040063 |
| DION (Indonesian Batik Motifs) | public, assembled | Kaggle: `dionisiusdh/indonesian-batik-motifs` |
| Corak App | public, assembled (replication) | Kaggle: `alfanme/indonesian-batik-motifs-corak-app` |
| Batik Jambi | private, assembled | not redistributed (third-party copyright) |

The CSV index/split files in this repo reference images by filename and MD5, so
you can align them with your own local download of each dataset.

---

## Repository structure

```
basa-audit/
├── code/                     # audit pipeline
│   ├── leakage_audit.py          # duplicate + leakage audit (Sections 2.3, 3.2)
│   ├── validate_leakage_audit.py # validation on synthetic planted duplicates
│   └── basa_leakage_audit.ipynb  # notebook walkthrough
├── data/
│   ├── splits/               # leakage-free grouped train/test splits (released per paper)
│   ├── index/                # per-image index: filename, label, dims, bytes, MD5, dup_group
│   └── duplicate_audit/      # duplicate edges, cross-label pairs, per-split leakage
├── results/                  # aggregated + raw numbers behind the paper tables
└── validation/              # synthetic 91-image benchmark with planted duplicates
```

Note: `results/corak_probeA.csv` and `results/corak_probeB.csv` are single-row
summaries (no per-seed breakdown), unlike the mean/std format used for the other
three datasets (`table2_probe_a.csv`, `table3_raw.csv`, etc.). Per-seed raw values
for Corak App, if available, should be added here for full consistency.

---

## How to run the leakage/duplicate audit

Dependencies: `numpy`, `pandas`, `pillow`, `scipy`.

```bash
python code/leakage_audit.py --dataset dion   --stage all
python code/leakage_audit.py --dataset corak  --stage all
python code/leakage_audit.py --dataset jambi  --stage all
python code/leakage_audit.py --dataset nitik  --stage all   # rotation re-check
```

Validate the duplicate detector against planted ground truth:

```bash
python code/validate_leakage_audit.py
```

---

## Mapping CSV files to paper tables

| File | Backs |
|---|---|
| `results/table2_probe_a.csv`, `results/probe_a_raw.csv` | Table 4 — Probe A (acquisition statistics) |
| `results/table_new_A_cue_availability.csv` | Table 7 — cue availability |
| `results/table3_raw.csv`, `results/table3_aggregated.csv` | Table 5 — Probe B (motif-destruction battery) |
| `results/table_new_B_cue_reliance.csv` | Table 8 — cue reliance |
| `results/table4_raw.csv`, `results/table4_aggregated.csv` | Table 6 — multi-architecture retention ratios |
| `results/corak_probeA.csv` | Section 3.8 — Probe A replication on Corak App (resolution 1.1× chance, colour 95.7%) |
| `results/corak_probeB.csv` | Section 3.8 — Probe B replication on Corak App (COLOR 97.3%, blur/colour ratio 0.98) |
| `data/duplicate_audit/ALL_summary.csv`, `ALL_leak.csv` | Table 3 — duplicate audit summary |
| `data/duplicate_audit/*_dup_edges.csv`, `*_crossclass_pairs.csv` | Section 3.2 — duplicate pairs, cross-label collisions |
| `data/duplicate_audit/*_leak_under_paper_split.csv` | Section 3.2 — per-seed leakage under the paper split |
| `data/splits/*_grouped_splits.csv` | Leakage-free grouped splits (released) |
| `validation/fake_*` | Section 2.3 — synthetic validation (all planted pairs recovered, no false positives) |

---

## Scope of this release / notes

- The code provided here reproduces the **duplicate and near-duplicate leakage
  audit**. Scripts for **Probe A** (Random-Forest acquisition-statistics classifier)
  and **Probe B** (ConvNeXt / ResNet / ViT / Swin fine-tuning) are available from the
  corresponding author on request; the numerical outputs of both probes are included
  in `results/` for all four datasets, including the Corak App replication
  (`corak_probeA.csv`, `corak_probeB.csv`).
- **Corak App leakage/duplicate audit** (index, grouped index, and grouped splits)
  is not yet included in this snapshot. Unlike Nitik, DION, and Jambi, a formal
  near-duplicate/leakage audit has not been run on Corak App; the paper does not
  report a duplicate count or leakage percentage for this dataset. `leakage_audit.py`
  supports it (`--dataset corak --stage all`) and the files will be added once
  generated.
- Jambi index/split CSVs contain filenames and MD5 hashes only, never the images.

## Citation

If you use this audit, please cite the paper above. Correspondence:
`adeherti@unama.ac.id`.
