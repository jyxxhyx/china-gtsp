#!/usr/bin/env python3
"""
fetch_road_matrix.py — 抓取344城两两路网距离矩阵 (OSRM table API)

策略: 100城/批的分块请求, 344城=4批 (100+100+100+44)
     对角块: 100×100; 非对角: 需要批i和批j的并集请求
     实际做法: 4批k-means无关划分, 块i∪块j做table(≤100约束下只能对角)
     → 改用: 每批100城请求自身table, 再单独补非对角块(每次50+50)
总计: 4对角 + 12个非对角块(每块拆成≤100坐标的子请求) ≈ 4 + 48 = 52请求

输出: output/road_matrix_344.json (344×344, 单位km)
"""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(_root)

BASE = "http://router.project-osrm.org/table/v1/driving/"
HDR = {'User-Agent': 'china-gtsp-roadmatrix/1.0 (research; contact: local)'}


def table(coords):
    """请求OSRM table, 带重试。coords: [(lng,lat)]"""
    cs = ';'.join(f"{lo:.6f},{la:.6f}" for lo, la in coords)
    url = f"{BASE}{cs}?annotations=distance"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            return np.array(data['distances']) / 1000.0   # → km
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"    重试{attempt+1} ({str(e)[:50]}), 等{wait}s")
            time.sleep(wait)
    raise RuntimeError(f"table请求失败: {len(coords)}坐标")


def main():
    df = pd.read_csv('data/cities.csv')
    n = len(df)
    coords_all = list(zip(df['lng'].values, df['lat'].values))
    print(f"{n}城市, 目标 {n*(n-1)//2} 对距离")

    # 分块: 100城/批
    B = 100
    blocks = [list(range(s, min(s + B, n))) for s in range(0, n, B)]
    nb = len(blocks)
    print(f"{nb}批: {[len(b) for b in blocks]}")

    D = np.zeros((n, n))

    # 缓存文件(断点续传)
    cache = os.path.join(_root, 'output', 'road_matrix_partial.npz')
    done = set()
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        D = z['D']
        done = set(z['done'].tolist())
        print(f"恢复缓存: {len(done)}块完成")

    t0 = time.time()
    n_req = 0
    for bi in range(nb):
        for bj in range(bi, nb):
            key = bi * 10 + bj
            if key in done:
                continue
            # 对角块直接请求
            if bi == bj:
                idx = blocks[bi]
                sub = table([coords_all[i] for i in idx])
                for a, ia in enumerate(idx):
                    for b2, ib in enumerate(idx):
                        D[ia][ib] = sub[a][b2]
                n_req += 1
                print(f"  块({bi},{bj}) {len(idx)}×{len(idx)} ✓ [{time.time()-t0:.0f}s, 累计{n_req}请求]")
            else:
                # 非对角: 拆成两半各≤50, 请求50+50=100坐标
                ia_list = blocks[bi]
                ib_list = blocks[bj]
                for ha in range(0, len(ia_list), 50):
                    for hb in range(0, len(ib_list), 50):
                        pa = ia_list[ha:ha+50]
                        pb = ib_list[hb:hb+50]
                        sub = table([coords_all[i] for i in pa] + [coords_all[i] for i in pb])
                        la, lb = len(pa), len(pb)
                        for a, ia in enumerate(pa):
                            for b2, ib in enumerate(pb):
                                D[ia][ib] = sub[a][la + b2]
                                D[ib][ia] = sub[la + b2][a]
                        n_req += 1
                print(f"  块({bi},{bj}) {len(ia_list)}×{len(ib_list)} ✓ [{time.time()-t0:.0f}s, 累计{n_req}请求]")

            done.add(key)
            np.savez(cache, D=D, done=np.array(sorted(done)))
            time.sleep(1.2)

    # 校验
    assert (D > 0).sum() == n * (n - 1), f"有零值: {(D==0).sum() - n}个(除对角)"
    asym = np.abs(D - D.T).max()
    print(f"\n矩阵完成: {n}×{n}, 最大不对称偏差 {asym:.4f} km (OSRM双向路线差异)")

    json.dump({'n': n, 'cities': df['name'].tolist(), 'matrix_km': D.round(3).tolist()},
              open(os.path.join(_root, 'output', 'road_matrix_344.json'), 'w'), ensure_ascii=False)
    np.save(os.path.join(_root, 'output', 'road_matrix_344.npy'), D)
    print("saved: output/road_matrix_344.json / .npy")
    print(f"总耗时 {time.time()-t0:.0f}s, 总请求 {n_req}")


if __name__ == '__main__':
    main()
