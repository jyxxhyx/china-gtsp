#!/usr/bin/env python3
"""
cetsp_refine_heur.py — 启发式解 (7,577.58) 的端点+驻点连续精化
每槽位候选 = 端点 + 段内驻点 (f(p)=d(prev,p)+d(p,next) 黄金分割), DP 迭代收敛。
用法: python3 cetsp_refine_heur.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from cnmaps import get_adm_maps
from cetsp_merge import short
from gtsp_core import haversine, haversine_vec
import cetsp_cnmaps as cm
import cetsp_exact_solve as es
import cetsp_34slot as m34

FIXED_NAMES = ['西藏', '青海', '新疆']


def main():
    t0 = time.time()
    polys = cm.load_polys()
    short2full = {short(nm): nm for nm in polys}
    names = sorted(short2full)
    fulls = [short2full[sn] for sn in names]
    name2idx = {sn: i for i, sn in enumerate(names)}
    fixed_idx = {name2idx[s] for s in FIXED_NAMES}
    TRIPLE = cm.find_triple(polys)

    # 读启发式解
    sol = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      'output', 'cetsp', 'cetsp_heur_s7.json')))
    order = [name2idx[p] for p in sol['slot_order']]
    print(f"启发式顺序: {' → '.join(names[i] for i in order)}")

    # 边界线段 (原始顶点, 不抽稀 — 端点+驻点用真边界)
    segs_list = []
    for i in range(34):
        segs_list.append(m34.boundary_segments(polys[fulls[i]]))
    sizes = [len(s) for s in segs_list]
    print(f"线段数: {min(sizes)}~{max(sizes)}")

    # 初始位置 (启发式解)
    pos = {order[t]: (v['lon'], v['lat']) for t, v in enumerate(sol['visits'])}
    for i in fixed_idx:
        pos[i] = TRIPLE

    # 端点+驻点迭代 DP — 局部精化版: 每槽位只在"当前点附近 ±R km"的线段上
    # 做驻点黄金分割 (启发式解已接近最优, 无需全段枚举)
    import math

    def local_slot_candidates(segs, cur_pos, prev_pos, next_pos, radius_km=8.0):
        """当前点附近的线段 (距离 ≤ radius_km) → 端点 + 段内驻点。"""
        lon, lat = cur_pos
        # 找附近线段
        near = []
        for (a, b) in segs:
            # 线段到当前点的平面距离 (近似)
            ax, ay = a
            bx, by = b
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = ((lon - ax) * dx + (lat - ay) * dy) / L2 if L2 > 0 else 0
            t = max(0.0, min(1.0, t))
            px, py = ax + t * dx, ay + t * dy
            d = ((px - lon) ** 2 + (py - lat) ** 2) ** 0.5
            if d < radius_km / 111.0:
                near.append((a, b))
        if not near:
            # 退化: 用全段但只取端点附近 (保守: 返回当前点)
            return [cur_pos]
        cands = []
        for (a, b) in near:
            cands.append(a)
            # 段内驻点 (黄金分割, 仅此段)
            def cost(t):
                ax, ay = a
                bx, by = b
                p = (ax + t * (bx - ax), ay + t * (by - ay))
                c = 0.0
                if prev_pos:
                    c += haversine(prev_pos, p)
                if next_pos:
                    c += haversine(p, next_pos)
                return c

            lo, hi = 0.0, 1.0
            for _ in range(40):
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                if cost(m1) < cost(m2):
                    hi = m2
                else:
                    lo = m1
            t_best = (lo + hi) / 2
            p_best = (a[0] + t_best * (b[0] - a[0]), a[1] + t_best * (b[1] - a[1]))
            cands.append(p_best)
        # 去重
        seen, uniq = set(), []
        for p in cands:
            k = (round(p[0], 6), round(p[1], 6))
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        return uniq

    for it in range(6):
        layers = []
        for t in range(34):
            i = order[t]
            if i in fixed_idx:
                layers.append([TRIPLE])
            else:
                prev_p = pos[order[t - 1]] if t > 0 else None
                next_p = pos[order[t + 1]] if t + 1 < 34 else None
                layers.append(local_slot_candidates(segs_list[i], pos[i],
                                                    prev_p, next_p))
        rad = [np.radians(np.array(l)) for l in layers]
        dp = np.zeros(len(rad[0]))
        back = []
        for t in range(1, 34):
            prev = rad[t - 1]
            cur = rad[t]
            D = haversine_vec(cur[:, 0][:, None], cur[:, 1][:, None],
                              prev[:, 0][None, :], prev[:, 1][None, :])
            V = D + dp[None, :]
            best = np.argmin(V, axis=1)
            dp = V[np.arange(len(cur)), best]
            back.append(best)
        last = int(np.argmin(dp))
        idx = [last]
        for bk in reversed(back):
            idx.append(bk[idx[-1]])
        idx.reverse()
        for t in range(34):
            pos[order[t]] = layers[t][idx[t]]
        km = sum(haversine(tuple(pos[order[t]]), tuple(pos[order[t + 1]]))
                 for t in range(33))
        print(f"  精化{it+1}: {km:.2f} km")
        if it > 0 and abs(km - prev_km) < 1e-3:
            break
        prev_km = km

    # 严格覆盖校验 (口径: 点距省边界 ≤ 1cm — covers 对线段插值点不可靠)
    from shapely.geometry import Point
    EPS = 1e-7
    cover = {}
    for t in range(34):
        lon, lat = pos[order[t]]
        p = Point(lon, lat)
        cov = [short(nm) for nm, g in polys.items()
               if g.covers(p) or g.boundary.distance(p) <= EPS]
        for nm in cov:
            cover.setdefault(nm, [])
    missing = [short(nm) for nm in sorted(polys) if short(nm) not in cover]
    print(f"\n严格 covers: {len(cover)}/34, 未覆盖: {missing if missing else '无 ✓'}")
    print(f"最终: {km:.2f} km (启发式 7577.58, 改进 {7577.58 - km:+.2f})")

    out = {
        'distance_open': round(km, 2),
        'baseline_heuristic': 7577.58,
        'diff': round(km - 7577.58, 2),
        'slot_order': [names[i] for i in order],
        'visits': [{'prov': names[order[t]], 'lon': float(pos[order[t]][0]),
                    'lat': float(pos[order[t]][1])} for t in range(34)],
        'method': 'cnmaps 34-slot heuristic + endpoint/stationary refine',
        'time_s': round(time.time() - t0, 1),
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'output', 'cetsp', 'cetsp_refined_result.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已保存: {out_path}")


if __name__ == '__main__':
    main()
