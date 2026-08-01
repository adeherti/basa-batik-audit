"""Generator v2: motif periodik TAPI dijamin berbeda antar-citra
(fase+frekuensi+tekstur dasar unik per citra), agar ground truth bersih."""
import numpy as np, shutil
from pathlib import Path
from PIL import Image
ROOT = Path("/home/claude/fake_ds"); shutil.rmtree(ROOT, ignore_errors=True)

def motif(seed, size=256):
    r = np.random.default_rng(seed * 7919)
    base = r.normal(0, 1, (8, 8))                      # cap unik per citra
    tile = np.kron(base, np.ones((size//8, size//8)))  # periodik (mirip batik)
    x, y = np.meshgrid(np.linspace(0, r.uniform(4,20)*np.pi, size),
                       np.linspace(0, r.uniform(4,20)*np.pi, size))
    g = np.sin(x + r.uniform(0,6)) * np.cos(y + r.uniform(0,6))
    a = 0.6*tile + 0.4*g
    a = (a - a.min())/(np.ptp(a)+1e-9)
    img = np.stack([a*r.uniform(120,255), a*r.uniform(120,255), a*r.uniform(120,255)], -1)
    return Image.fromarray(np.clip(img + r.normal(0,4,img.shape),0,255).astype(np.uint8))

planted = {"exact":[], "near":[], "xclass":[]}
classes = [f"kelas_{i}" for i in range(4)]
uid = 0
for ci, c in enumerate(classes):
    (ROOT/c).mkdir(parents=True, exist_ok=True)
for ci, c in enumerate(classes):
    for k in range(20):
        im = motif(ci*100+k); f0 = f"img_{uid:03d}.jpg"
        im.save(ROOT/c/f0, quality=92); uid += 1
        if k == 0:
            f1 = f"img_{uid:03d}.jpg"; shutil.copy(ROOT/c/f0, ROOT/c/f1); uid += 1
            planted["exact"].append(tuple(sorted((f0,f1))))
        if k == 1:
            f1 = f"img_{uid:03d}.jpg"
            im.resize((203,203), Image.Resampling.LANCZOS).save(ROOT/c/f1, quality=55); uid += 1
            planted["near"].append(tuple(sorted((f0,f1))))
        if k == 2 and ci < 3:
            f1 = f"img_{uid:03d}.jpg"
            im.resize((240,240), Image.Resampling.LANCZOS).save(ROOT/classes[ci+1]/f1, quality=80); uid += 1
            planted["xclass"].append(tuple(sorted((f0,f1))))
import json; json.dump({k:[list(t) for t in v] for k,v in planted.items()}, open("planted.json","w"))
print(f"total {uid} citra | exact {len(planted['exact'])} | near {len(planted['near'])} | xclass {len(planted['xclass'])}")
