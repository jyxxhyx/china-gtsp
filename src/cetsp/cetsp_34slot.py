#!/usr/bin/env python3
"""
cetsp_34slot.py — 34 槽位模型: 每省一槽位, 共享点由优化决策
每槽位候选 = 该省全部边界线 (端点+驻点/粗采样); 相邻槽位选同一点 = 共享 (d=0)。
固定: 藏/青/新 三槽位 → 三界点 (89.711414, 36.093272) (不参与顺序搜索)。
阶段1: 粗候选 (每省均匀 ~60 点) + 2-opt/or-opt 顺序搜索 (固定槽位不动)
阶段2: 端点+驻点迭代精化 (最优顺序)
输出: 34 槽位各点 + 距离 + 共享检测。
用法: python3 cetsp_34slot.py [rounds]
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from cetsp_province import load_province_polygons
from cetsp_merge import short
from gtsp_core import haversine, haversine_vec

TRIPLE = (89.711414, 36.093272)           # 藏青新三界点 (固定)
FIXED_SLOTS = {0: TRIPLE, 1: TRIPLE, 2: TRIPLE}   # 藏, 青, 新 → 三界点
JING_KM = 7579.76

# 19 组合展开 → 34 省顺序 (初始)
EXPAND = [['西藏', '青海', '新疆'], ['甘肃', '宁夏'], ['陕西', '四川'], ['湖北', '重庆'],
          ['湖南', '贵州'], ['广西', '云南'], ['海南'], ['广东', '澳门'], ['香港'],
          ['台湾'], ['福建'], ['浙江', '江西'], ['上海'], ['安徽', '江苏'],
          ['河南', '山东'], ['山西'], ['北京', '天津', '河北'], ['辽宁', '内蒙古'],
          ['吉林', '黑龙江']]


def boundary_segments(poly):
    """省边界 → 线段列表 [(a,b)] (不 simplify, 原始顶点)。"""
    segs = []
    for sub in poly.geoms:
        ring = list(sub.exterior.coords)
        for i in range(len(ring) - 1):
            segs.append((ring[i], ring[i + 1]))
        for inter in sub.interiors:
            ring = list(inter.coords)
            for i in range(len(ring) - 1):
                segs.append((ring[i], ring[i + 1]))
    return segs


def uniform_cands(segs, n, fixed=None):
    """沿边界弧长均匀 n 个候选 (含端点)。fixed: 固定点则返回 [fixed]。"""
    if fixed:
        return [fixed]
    # 累积弧长
    cum, tot = [0.0], 0.0
    for (a, b) in segs:
        tot += haversine(tuple(a), tuple(b))
        cum.append(tot)
    if tot <= 0:
        return segs[0] if segs else [(0, 0)]
    pts, seen = [], set()
    for r in range(n):
        target = tot * r / n
        for j in range(len(segs)):
            if target <= cum[j + 1]:
                (a, b) = segs[j]
                frac = (target - cum[j]) / (cum[j + 1] - cum[j] + 1e-12)
                p = (a[0] + frac * (b[0] - a[0]), a[1] + frac * (b[1] - a[1]))
                k = (round(p[0], 6), round(p[1], 6))
                if k not in seen:
                    seen.add(k)
                    pts.append(p)
                break
    return pts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('rounds', type=int, nargs='?', default=3)
    ap.add_argument('--extra', type=int, default=0, help='每段额外均匀采样点数 (加密验证)')
    args = ap.parse_args()
    rounds = args.rounds
    EXTRA = args.extra
    t0 = time.time()
    polys = load_province_polygons('datav')
    polys_s = {short(nm): g for nm, g in polys.items()}
    names = sorted(polys_s)                      # 34 省 (短名, 排序)
    # 固定槽位: 按省名映射 (藏/青/新 → 三界点) — 不依赖排序索引
    name2idx = {nm: i for i, nm in enumerate(names)}
    FIXED_SLOTS = {name2idx[s]: TRIPLE for s in ['西藏', '青海', '新疆']}
    # 初始顺序: 19 组合展开
    init_order = [name2idx[s] for comb in EXPAND for s in comb if s in name2idx]
    missing = [i for i in range(34) if i not in init_order]
    init_order += missing
    print(f"34 槽位: {names}")
    print(f"初始顺序: {' → '.join(names[i] for i in init_order)}")

    segs_list = [boundary_segments(polys_s[nm]) for nm in names]
    # 粗候选 (每省 60 均匀点; 固定槽位 = 三界点)
    layers = [uniform_cands(segs_list[i], 60, fixed=FIXED_SLOTS.get(i)) for i in range(34)]
    sizes = [len(l) for l in layers]
    print(f"粗候选: 每槽位 {min(sizes)}~{max(sizes)} (总 {sum(sizes)})")

    rad_arr = [np.radians(np.array(l)) for l in layers]

    def dp_eval(order):
        dp = np.zeros(len(rad_arr[order[0]]))
        back = []
        for t in range(1, 34):
            prev = rad_arr[order[t - 1]]
            cur = rad_arr[order[t]]
            D = haversine_vec(cur[:, 0][:, None], cur[:, 1][:, None],
                              prev[:, 0][None, :], prev[:, 1][None, :])
            V = D + dp[None, :]
            best = np.argmin(V, axis=1)
            dp = V[np.arange(len(cur)), best]
            back.append(best)
        last = int(np.argmin(dp))
        path_idx = [last]
        for bk in reversed(back):
            path_idx.append(bk[path_idx[-1]])
        path_idx.reverse()
        path = [layers[order[t]][path_idx[t]] for t in range(34)]
        km = sum(haversine(tuple(path[i]), tuple(path[i + 1])) for i in range(33))
        return path, km

    # 阶段1: 顺序搜索 (固定槽位 0,1,2 保持在前 3 位)
    order, path, km = init_order, None, None
    path, km = dp_eval(order)
    print(f"初始顺序 DP: {km:.2f} km")
    best_order, best_km = list(order), km
    for rnd in range(rounds):
        improved = False
        for i in range(2, 33):
            for j in range(i + 2, 34):
                o2 = best_order[:i + 1] + best_order[i + 1:j + 1][::-1] + best_order[j + 1:]
                if o2[:3] != best_order[:3]:
                    continue
                _, k2 = dp_eval(o2)
                if k2 < best_km - 1e-3:
                    best_order, best_km = o2, k2
                    improved = True
                    print(f"  2-opt: {best_km:.2f} km (反转 {i+1}-{j+1})")
        for i in range(3, 34):
            for j in range(3, 34):
                if i == j:
                    continue
                o2 = list(best_order)
                v = o2.pop(i)
                o2.insert(j, v)
                _, k2 = dp_eval(o2)
                if k2 < best_km - 1e-3:
                    best_order, best_km = o2, k2
                    improved = True
                    print(f"  or-opt: {best_km:.2f} km (移 {i+1}→{j+1})")
        print(f"轮 {rnd+1}: {best_km:.2f} km {'改进' if improved else '无改进'}")
        if not improved:
            break
    print(f"阶段1 最优: {best_km:.2f} km ({time.time()-t0:.0f}s)")

    # 阶段2: 端点+驻点精化 (34 槽位, 最优顺序)
    print("\n阶段2: 端点+驻点迭代精化...")
    # 端点+驻点候选 (cetsp_exact_solve 的 slot_candidates 逻辑)
    import cetsp_exact_solve as es
    pos = {}
    for t in range(34):
        i = best_order[t]
        if i in FIXED_SLOTS:
            pos[i] = FIXED_SLOTS[i]
        elif segs_list[i]:
            pos[i] = uniform_cands(segs_list[i], 1)[0]
        else:
            pos[i] = (0.0, 0.0)
    for it in range(4):
        layers2 = []
        for t in range(34):
            i = best_order[t]
            if i in FIXED_SLOTS:
                layers2.append([FIXED_SLOTS[i]])
            else:
                prev_p = pos[best_order[t - 1]] if t > 0 else None
                next_p = pos[best_order[t + 1]] if t + 1 < 34 else None
                layers2.append(es.slot_candidates(segs_list[i], prev_p, next_p, EXTRA))
        rad2 = [np.radians(np.array(l)) for l in layers2]
        dp = np.zeros(len(rad2[0]))
        back = []
        for t in range(1, 34):
            prev = rad2[t - 1]
            cur = rad2[t]
            D = haversine_vec(cur[:, 0][:, None], cur[:, 1][:, None],
                              prev[:, 0][None, :], prev[:, 1][None, :])
            V = D + dp[None, :]
            best = np.argmin(V, axis=1)
            dp = V[np.arange(len(cur)), best]
            back.append(best)
        last = int(np.argmin(dp))
        path_idx = [last]
        for bk in reversed(back):
            path_idx.append(bk[path_idx[-1]])
        path_idx.reverse()
        for t in range(34):
            pos[best_order[t]] = layers2[t][path_idx[t]]
        km2 = sum(haversine(tuple(pos[best_order[t]]), tuple(pos[best_order[t + 1]]))
                  for t in range(33))
        print(f"  精化{it+1}: {km2:.2f} km")
        if it > 0 and abs(prev_k - km2) < 1e-3:
            break
        prev_k = km2

    # 共享检测
    pts34 = [pos[i] for i in best_order]
    shared = []
    for t in range(33):
        if haversine(pts34[t], pts34[t + 1]) < 1e-6:
            shared.append((names[best_order[t]], names[best_order[t + 1]]))
    print(f"\n最终: {km2:.2f} km (19槽位基准 7579.76, 差值 {km2-7579.76:+.2f})")
    print(f"相邻共享点: {shared if shared else '无'}")
    print("访问点 (顺序):")
    for t in range(34):
        lon, lat = pos[best_order[t]]
        print(f"  {names[best_order[t]]:<8} ({lon:.5f}, {lat:.5f})")

    out = {
        'distance_open': round(km2, 2),
        'baseline_19slot': JING_KM,
        'diff': round(km2 - JING_KM, 2),
        'slot_order': [names[i] for i in best_order],
        'visits': [{'prov': names[best_order[t]], 'lon': float(pos[best_order[t]][0]),
                    'lat': float(pos[best_order[t]][1])} for t in range(34)],
        'shared_pairs': shared,
        'method': '34-slot DP + order search + endpoint/stationary refine (DataV)',
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'output', 'cetsp', 'cetsp_34slot_result.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已保存: {out_path}")


if __name__ == '__main__':
    main()
