#!/usr/bin/env python3
# =============================================================================
# BASA — Near-Duplicate Leakage Audit  (menjawab catatan promotor #4)
#
# Nitik rotation leakage SUDAH diuji (LEAKY-GROUP = 0). Skrip ini menutup lubang
# yang BELUM diukur: near-duplicate pada dataset rakitan web (DION, Corak, Jambi).
#
# Dijalankan bertahap. Tahap 1-3 CPU-only, ±2-10 menit. Tahap 4 hanya emit split.
#
#   python leakage_audit.py --dataset dion   --stage all
#   python leakage_audit.py --dataset corak  --stage all
#   python leakage_audit.py --dataset jambi  --stage all
#   python leakage_audit.py --dataset nitik  --stage all   # verifikasi ulang rotasi
#
# Output: CSV di ./leakage_out/ — siap dijadikan Tabel & dilampirkan ke paper.
# Dependensi: numpy, pandas, pillow, scipy  (semua sudah ada di Kaggle)
# =============================================================================

import argparse, hashlib, os, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct

Image.MAX_IMAGE_PIXELS = None

# ----------------------------------------------------------------------------- 
# KONFIGURASI DATASET — sesuaikan path bila berbeda
# -----------------------------------------------------------------------------
DATASETS = {
    "nitik": dict(
        root="/kaggle/input/datasets/hertiyani/batik-nitik-960-official-mendeley/"
             "Batik Nitik 960/Batik Nitik 960 Images",
        label_from="parent",      # kelas = nama folder induk
        instance_parser="nitik",  # grouping bawaan: <classnum> <Motif> <inst>[_rotate_XXX]
        expect_n=960,
    ),
    "dion": dict(
        root="/kaggle/input/datasets/dionisiusdh/indonesian-batik-motifs",
        label_from="parent",
        instance_parser=None,
        expect_n=983,
    ),
    "corak": dict(
        root="/kaggle/input/datasets/alfanme/indonesian-batik-motifs-corak-app",
        label_from="parent",
        instance_parser=None,
        expect_n=4950,
    ),
    "fake": dict(
        root="/home/claude/fake_ds",
        label_from="parent",
        instance_parser=None,
        expect_n=None,
    ),
    "jambi": dict(
        root="/kaggle/input/datasets/hertiyani/batik-jambi/batik jambi",
        label_from="parent",
        instance_parser=None,
        expect_n=161,
    ),
}

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OUT = Path("./leakage_out"); OUT.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# TAHAP 1 — indeks citra + hash
# -----------------------------------------------------------------------------
def phash(img, hash_size=8, highfreq_factor=4):
    """Perceptual hash 64-bit (DCT). Tahan terhadap resize/JPEG re-encode/perubahan
    kualitas ringan — persis kasus citra yang di-scrape ulang dari web."""
    n = hash_size * highfreq_factor
    im = img.convert("L").resize((n, n), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])          # buang DC agar tak bias kecerahan
    return (low > med).flatten()


def dhash(img, hash_size=8):
    """Difference hash — cue berbeda dari pHash. Dipakai sebagai konfirmasi silang
    agar tidak bergantung pada satu algoritma saja."""
    im = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def parse_nitik(stem):
    """<classnum> <MotifName> <instancenum>[_rotate_XXX]
    contoh: '1 Sekar Kemuning 1_rotate_180'; tangani spasi-ganda '31 Krawitan  2'."""
    rot = 0
    m = re.search(r"_rotate_(\d+)", stem)
    if m:
        rot = int(m.group(1))
        stem = re.sub(r"_rotate_\d+", "", stem)
    toks = re.sub(r"\s+", " ", stem).strip().split(" ")
    return dict(classnum=toks[0], instnum=toks[-1],
                motif=" ".join(toks[1:-1]), rotation=rot)


def stage1_index(cfg, name):
    rows = []
    root = Path(cfg["root"])
    if not root.exists():
        sys.exit(f"[FATAL] root tidak ditemukan: {root}\n"
                 f"        Perbaiki DATASETS['{name}']['root'].")

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    print(f"[1] {name}: {len(files)} file ditemukan (ekspektasi {cfg['expect_n']})")
    if cfg["expect_n"] and len(files) != cfg["expect_n"]:
        print(f"    [!] jumlah TIDAK cocok dengan paper — catat & jelaskan selisihnya.")

    bad = 0
    for p in files:
        try:
            raw = p.read_bytes()
            with Image.open(p) as im:
                im.load()
                w, h = im.size
                ph, dh = phash(im), dhash(im)
        except Exception as e:
            bad += 1
            print(f"    [corrupt] {p.name}: {e}")
            continue

        r = dict(
            path=str(p), fname=p.name,
            label=p.parent.name if cfg["label_from"] == "parent" else "?",
            width=w, height=h, bytes=len(raw),
            md5=hashlib.md5(raw).hexdigest(),
            phash="".join("1" if b else "0" for b in ph),
            dhash="".join("1" if b else "0" for b in dh),
        )
        if cfg["instance_parser"] == "nitik":
            r.update(parse_nitik(p.stem))
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}_index.csv", index=False)
    print(f"    -> {len(df)} terindeks, {bad} corrupt. leakage_out/{name}_index.csv")
    return df


# -----------------------------------------------------------------------------
# TAHAP 2 — klaster near-duplicate
# -----------------------------------------------------------------------------
DIHEDRAL = 8  # identitas, 3 rotasi, + cerminan masing-masing


class UF:
    def __init__(s, n): s.p = list(range(n))
    def find(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b):
        ra, rb = s.find(a), s.find(b)
        if ra != rb: s.p[rb] = ra


def _thumbs(df, size=64):
    """Thumbnail 64x64 grayscale, ternormalisasi (mean 0, std 1) -> hasil kali
    titik antar dua thumbnail = korelasi Pearson."""
    X = np.zeros((len(df), size * size), dtype=np.float32)
    for k, p in enumerate(df["path"]):
        with Image.open(p) as im:
            a = np.asarray(im.convert("L").resize((size, size),
                           Image.Resampling.LANCZOS), dtype=np.float32)
        a = a - a.mean()
        s = a.std()
        X[k] = (a / s if s > 1e-6 else a).ravel()
    return X


def _variants(X, size=64):
    """8 varian dihedral. Menangkap duplikat yang diputar/dicerminkan -- kasus
    nyata pada dataset ber-augmentasi tersimpan (Nitik: _rotate_90/180/270)."""
    A = X.reshape(-1, size, size)
    out = []
    for f in (False, True):
        B = A[:, :, ::-1] if f else A
        for r in range(4):
            out.append(np.rot90(B, r, axes=(1, 2)).reshape(len(A), -1).copy())
    return out


def stage2_cluster(df, name, thr=None, pix_corr=0.90, dihedral=True):
    """Klaster near-duplicate berbasis KORELASI PIKSEL, bukan perceptual hash.

    MENGAPA BUKAN pHash (diuji offline, jangan diubah tanpa uji ulang):
      Batik adalah tekstur PERIODIK. Pada citra periodik, ambang median DCT
      di pHash tidak stabil: koefisien menumpuk di sekitar median sehingga
      perubahan kecil membalik banyak bit. Pada uji sintetis bermotif periodik:
        - FALSE POSITIVE : 6 dari 9 pasangan lolos pHash<=5 padahal korelasi ~0.0
        - FALSE NEGATIVE : pasangan dgn korelasi 1.000 (duplikat sempurna)
                           punya jarak pHash 9, 10, bahkan 18 -> TERLEWAT
      Korelasi piksel memisahkan bersih (1.000 utk dup, ~0.0 utk non-dup) dan
      untuk n<=5000 dapat dihitung EKSAK via satu matmul (~1 detik, ~100 MB).
      Jadi tidak ada blocking, tidak ada false negative dari hashing.
    """
    n = len(df)
    print(f"\n[2] {name}: near-duplicate via KORELASI PIKSEL (>= {pix_corr})"
          f"{' + dihedral' if dihedral else ''}")
    X = _thumbs(df)
    Vs = _variants(X) if dihedral else [X]

    # korelasi maksimum lintas 8 varian, dihitung per-blok agar hemat RAM
    best = np.zeros((n, n), dtype=np.float32)
    d = X.shape[1]
    B = 1024
    for V in Vs:
        for a in range(0, n, B):
            blk = (X[a:a+B] @ V.T) / d
            np.maximum(best[a:a+B], blk, out=best[a:a+B])
    best = np.maximum(best, best.T)          # simetriskan (dihedral tidak simetris)
    np.fill_diagonal(best, -1.0)

    uf = UF(n)
    ii, jj = np.where(np.triu(best >= pix_corr, k=1))
    for i, j in zip(ii, jj):
        uf.union(int(i), int(j))
    linked = [(int(i), int(j), round(float(best[i, j]), 4)) for i, j in zip(ii, jj)]

    df = df.copy()
    df["dup_group"] = [uf.find(i) for i in range(n)]
    ed = pd.DataFrame(linked, columns=["i", "j", "pix_corr"])

    # -------------------------------------------------------------------------
    # Dua pemakaian, dua definisi -- JANGAN dicampur:
    #   edge langsung (ed)  -> SEMUA angka yang diklaim di paper.
    #   dup_group transitif -> HANYA untuk membuat split (over-group = konservatif).
    # Klaster berantai (A~B, B~C, tapi A!~C) tak boleh dilaporkan sebagai duplikat:
    # reviewer yang mengecek pasangan A-C akan menemukan dua citra tak mirip.
    # -------------------------------------------------------------------------
    dup_direct = set(ed["i"]) | set(ed["j"])
    sizes = df["dup_group"].value_counts()
    n_exact = int(df["md5"].duplicated().sum())
    n_trans = int((df["dup_group"].map(sizes) > 1).sum())

    chained = 0
    for g, sub in df[df["dup_group"].map(sizes) > 1].groupby("dup_group"):
        idx = sub.index.to_numpy()
        if len(idx) < 3:
            continue                      # klaster berpasangan tak mungkin berantai
        sub_m = best[np.ix_(idx, idx)]
        off = sub_m[~np.eye(len(idx), dtype=bool)]   # diagonal = -1, HARUS dibuang
        if float(off.min()) < pix_corr:
            chained += 1

    print(f"    citra                       : {n}")
    print(f"    md5 identik (exact dup)     : {n_exact}")
    print(f"    citra dgn kembaran LANGSUNG : {len(dup_direct)}  "
          f"({100*len(dup_direct)/n:.1f}%)   <- ANGKA UNTUK PAPER")
    print(f"    citra dlm klaster transitif : {n_trans}  ({100*n_trans/n:.1f}%)"
          f"   <- untuk split saja")
    print(f"    unique instances (grup)     : {len(sizes)}")
    print(f"    klaster terbesar            : {int(sizes.max())} citra")
    if chained:
        print(f"    [!] {chained} klaster BERANTAI -- dikecualikan dari angka laporan,")
        print(f"        tetap dipakai utk split (konservatif).")

    # --- konflik label: HANYA dari edge langsung ---
    lab = df["label"].to_numpy()
    xe = ed[lab[ed["i"]] != lab[ed["j"]]].copy()
    if len(xe):
        for c, s in (("label", lab), ("fname", df["fname"].to_numpy()),
                     ("path", df["path"].to_numpy())):
            xe[f"{c}_i"], xe[f"{c}_j"] = s[xe["i"]], s[xe["j"]]
    n_xi = len(set(xe["i"]) | set(xe["j"])) if len(xe) else 0
    print(f"    PASANGAN lintas-kelas       : {len(xe)}  ({n_xi} citra terlibat)")
    if len(xe):
        print(f"    [!!] citra ~sama diberi label BERBEDA -> ini KONTAMINASI, bukan")
        print(f"         sekadar kebocoran. Hubungkan ke Sec. 3.1 (angka 31,8%).")
        print(f"         WAJIB verifikasi mata sebelum diklaim di paper.")
        xe[["fname_i", "label_i", "fname_j", "label_j", "pix_corr",
            "path_i", "path_j"]].sort_values("pix_corr", ascending=False)\
          .to_csv(OUT / f"{name}_crossclass_pairs.csv", index=False)

    ed.to_csv(OUT / f"{name}_dup_edges.csv", index=False)
    df.to_csv(OUT / f"{name}_index_grouped.csv", index=False)
    return df, dict(dataset=name, n=n, exact_dup=n_exact,
                    dup_imgs_direct=len(dup_direct),
                    dup_rate=round(100*len(dup_direct)/n, 1),
                    dup_imgs_transitive=n_trans, n_groups=len(sizes),
                    largest=int(sizes.max()), chained_clusters=chained,
                    xclass_pairs=len(xe), xclass_images=n_xi)


# -----------------------------------------------------------------------------
# TAHAP 3 — berapa banyak test set yang bocor di bawah split paper (70/30 strat)
# -----------------------------------------------------------------------------
def stratified_split(df, seed, test_size=0.30):
    """Replika split paper: 70/30 stratified per kelas, TANPA grouping."""
    rng = np.random.default_rng(seed)
    te = []
    for _, sub in df.groupby("label"):
        idx = sub.index.to_numpy().copy()
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_size)))
        te.extend(idx[:k].tolist())
    te = set(te)
    return np.array([i not in te for i in df.index]), np.array([i in te for i in df.index])


def stage3_leak(df, name, seeds=(0, 1, 2, 3, 4)):
    print(f"\n[3] {name}: kebocoran near-dup di bawah split 70/30 stratified paper")
    sizes = df["dup_group"].value_counts()
    rows = []
    for s in seeds:
        tr_m, te_m = stratified_split(df, s)
        tr_groups = set(df.loc[tr_m, "dup_group"])
        te_idx = df.index[te_m]
        leaked = int(df.loc[te_idx, "dup_group"].isin(tr_groups).sum() -
                     (df.loc[te_idx, "dup_group"].map(sizes) == 1).sum() * 0)
        # hanya hitung yang grupnya memang >1 anggota
        te_multi = df.loc[te_idx][df.loc[te_idx, "dup_group"].map(sizes) > 1]
        leaked = int(te_multi["dup_group"].isin(tr_groups).sum())
        rows.append(dict(seed=s, n_test=int(te_m.sum()), leaked_test=leaked,
                         leak_pct=round(100*leaked/max(1, int(te_m.sum())), 1)))
        print(f"    seed {s}: {leaked}/{int(te_m.sum())} citra uji punya kembaran "
              f"di train ({rows[-1]['leak_pct']}%)")
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"{name}_leak_under_paper_split.csv", index=False)
    print(f"    RERATA kebocoran: {r['leak_pct'].mean():.1f}% ± {r['leak_pct'].std():.1f}")
    print(f"    -> Inilah angka yang HARUS masuk paper menggantikan caveat kualitatif §4.4.")
    return r


# -----------------------------------------------------------------------------
# TAHAP 4 — emit split bebas-bocor (grup dup utuh di satu sisi)
# -----------------------------------------------------------------------------
def stage4_emit(df, name, seeds=(0, 1, 2, 3, 4), test_size=0.30):
    print(f"\n[4] {name}: menulis split GROUPED bebas-bocor")
    out = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        # stratifikasi per kelas TAPI unit alokasi = dup_group
        gl = df.groupby("dup_group").agg(label=("label", lambda x: x.mode()[0]),
                                         n=("label", "size"))
        te_groups = set()
        for lab, sub in gl.groupby("label"):
            g = sub.index.to_numpy().copy(); rng.shuffle(g)
            target = sub["n"].sum() * test_size
            acc = 0
            for gid in g:
                if acc >= target: break
                te_groups.add(gid); acc += int(gl.loc[gid, "n"])
        split = np.where(df["dup_group"].isin(te_groups), "test", "train")
        t = df[["path", "fname", "label", "dup_group"]].copy()
        t["seed"], t["split"] = s, split
        out.append(t)
        ntr, nte = int((split == "train").sum()), int((split == "test").sum())
        overlap = len(set(df.loc[split == "train", "dup_group"]) &
                      set(df.loc[split == "test", "dup_group"]))
        print(f"    seed {s}: train {ntr} / test {nte} | overlap grup = {overlap} "
              f"{'OK' if overlap == 0 else '<< BUG'}")
    pd.concat(out).to_csv(OUT / f"{name}_grouped_splits.csv", index=False)
    print(f"    -> leakage_out/{name}_grouped_splits.csv")
    print(f"    Pakai file ini sebagai split di kode training yang sudah ada,")
    print(f"    lalu bandingkan akurasi COLOR: paper-split vs grouped-split.")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--stage", default="all", choices=["1", "2", "3", "4", "all"])
    ap.add_argument("--no_dihedral", action="store_true",
                help="matikan pencocokan rotasi/cermin (lebih cepat)")
    ap.add_argument("--pix_corr", type=float, default=0.90,
                help="ambang korelasi piksel utk verifikasi (anti false-positive)")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(","))
    cfg = DATASETS[a.dataset]

    idx_cached = OUT / f"{a.dataset}_index.csv"
    if a.stage in ("1", "all") or not idx_cached.exists():
        df = stage1_index(cfg, a.dataset)
    else:
        df = pd.read_csv(idx_cached, dtype={"phash": str, "dhash": str})
    if a.stage == "1": return

    df, summ = stage2_cluster(df, a.dataset, pix_corr=a.pix_corr,
                              dihedral=not a.no_dihedral)
    pd.DataFrame([summ]).to_csv(OUT / f"{a.dataset}_summary.csv", index=False)
    if a.stage == "2": return

    stage3_leak(df, a.dataset, seeds)
    if a.stage == "3": return

    stage4_emit(df, a.dataset, seeds)
    print("\nSELESAI. Semua CSV di ./leakage_out/")


if __name__ == "__main__":
    main()
