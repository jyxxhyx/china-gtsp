#!/usr/bin/env python3
"""抓取344城非对称路网距离矩阵(OSRM table, 保留原始D[i][j]≠D[j][i])"""
import json, os, time, urllib.request
import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
BASE = "http://router.project-osrm.org/table/v1/driving/"
HDR = {'User-Agent': 'china-gtsp-asym/1.0'}

def table(coords):
    cs = ';'.join(f"{lo:.6f},{la:.6f}" for lo, la in coords)
    url = f"{BASE}{cs}?annotations=distance"
    for att in range(5):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=120) as r:
                return np.array(json.load(r)['distances']) / 1000.0
        except Exception as e:
            time.sleep(3*(att+1))
    raise RuntimeError

df = pd.read_csv(os.path.join(_root, 'data', 'gtsp', 'cities.csv'))
n = len(df)
coords_all = list(zip(df['lng'].values, df['lat'].values))
B = 100
blocks = [list(range(s, min(s+B, n))) for s in range(0, n, B)]
nb = len(blocks)
D = np.zeros((n, n))

t0 = time.time()
for bi in range(nb):
    for bj in range(bi, nb):
        if bi == bj:
            idx = blocks[bi]
            sub = table([coords_all[i] for i in idx])
            for a, ia in enumerate(idx):
                for b2, ib in enumerate(idx):
                    D[ia][ib] = sub[a][b2]      # 块内: 原始双向值保留!
        else:
            ia_list, ib_list = blocks[bi], blocks[bj]
            for ha in range(0, len(ia_list), 50):
                for hb in range(0, len(ib_list), 50):
                    pa, pb = ia_list[ha:ha+50], ib_list[hb:hb+50]
                    sub = table([coords_all[i] for i in pa] + [coords_all[i] for i in pb])
                    la = len(pa)
                    for a, ia in enumerate(pa):
                        for b2, ib in enumerate(pb):
                            D[ia][ib] = sub[a][la+b2]   # A→B
                            D[ib][ia] = sub[la+b2][a]   # B→A (原始值, 不平均!)
        print(f"  块({bi},{bj}) ✓ [{time.time()-t0:.0f}s]")
        time.sleep(1.0)

zeros = [(i,j) for i in range(n) for j in range(n) if i!=j and D[i][j]==0]
print(f"零值: {len(zeros)}")
np.save(os.path.join(_root, 'output', 'gtsp', 'road_matrix_asym.npy'), D)
print(f"saved, 不对称max={np.abs(D-D.T).max():.1f} km")
