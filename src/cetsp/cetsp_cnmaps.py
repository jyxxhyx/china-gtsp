#!/usr/bin/env python3
"""
cetsp_cnmaps.py — cnmaps 数据 34 槽位模型重新分析 (修正三界点错误)
每省一槽位, 候选 = 真边界点 (原始环抽稀 + 投影回真边界, 保证严格 covers)。
固定: 藏/青/新 → cnmaps 真三界顶点 (89.709290, 36.093677) (三省环共享顶点, 严格覆盖3省)。
DP 精确求解 + 严格校验 (covers) + 共享检测。
用法: python3 cetsp_cnmaps.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from shapely.geometry import shape, Point
from shapely.ops import nearest_points
from cnmaps import get_adm_maps
from cetsp_merge import short
from gtsp_core import haversine, haversine_vec

TRIPLE_CN = (89.709290, 36.093677)    # 初始值; main 里动态重提取 (手打常数有浮点偏差, covers 会失败)


def find_triple(polys):
    """动态提取: 西藏环顶点中距青海/新疆边界最近的顶点 (真三界点, 原始 float)。"""
    best, best_d = None, float('inf')
    for sub in polys['西藏自治区'].geoms:
        for ring in (sub.exterior,) + tuple(sub.interiors):
            for x, y in ring.coords:
                p = Point(x, y)
                d = max(polys['青海省'].boundary.distance(p),
                        polys['新疆维吾尔自治区'].boundary.distance(p))
                if d < best_d:
                    best_d, best = d, (float(x), float(y))
    return best
FIXED_NAMES = ['西藏', '青海', '新疆']
EXPAND = [['西藏', '青海', '新疆'], ['甘肃', '宁夏'], ['陕西', '四川'], ['湖北', '重庆'],
          ['湖南', '贵州'], ['广西', '云南'], ['海南'], ['广东', '澳门'], ['香港'],
          ['台湾'], ['福建'], ['浙江', '江西'], ['上海'], ['安徽', '江苏'],
          ['河南', '山东'], ['山西'], ['北京', '天津', '河北'], ['辽宁', '内蒙古'],
          ['吉林', '黑龙江']]


def load_polys():
    out = {}
    for rec in get_adm_maps(level='省'):
        nm = rec.get('province')
        if nm:
            out[nm] = shape(rec['geometry'])
    return out


def decimate_ring(ring, step):
    """环顶点抽稀 (每 step 个取 1)。返回顶点列表。"""
    return [ring[i] for i in range(0, len(ring), step)]


def build_candidates(poly, n_cap=2000):
    """省边界候选: 抽稀顶点 (原环顶点的子集 — 本身就在真边界上, 无需投影)。"""
    raw = []
    for sub in poly.geoms:
        ring = list(sub.exterior.coords)
        raw.append(decimate_ring(ring, max(1, len(ring) // 800)))
        for inter in sub.interiors:
            ring = list(inter.coords)
            raw.append(decimate_ring(ring, max(1, len(ring) // 800)))
    all_pts = [p for ring in raw for p in ring[:-1]]
    if not all_pts:
        return []
    n = min(n_cap, len(all_pts))
    idx = np.linspace(0, len(all_pts) - 1, n).astype(int)
    out = []
    seen = set()
    for i in idx:
        p = all_pts[i]
        k = (round(p[0], 6), round(p[1], 6))
        if k not in seen:
            seen.add(k)
            out.append((float(p[0]), float(p[1])))
    return out


def main():
    t0 = time.time()
    polys = load_polys()
    short2full = {short(nm): nm for nm in polys}
    names = sorted(short2full)
    fulls = [short2full[sn] for sn in names]
    name2idx = {sn: i for i, sn in enumerate(names)}
    fixed_idx = {name2idx[s] for s in FIXED_NAMES}
    TRIPLE_CN = find_triple(polys)
    p0 = Point(TRIPLE_CN)
    print(f"cnmaps 34 省加载完成, 真三界顶点 ({TRIPLE_CN[0]:.6f},{TRIPLE_CN[1]:.6f}) "
          f"严格covers: {[short(nm) for nm, g in polys.items() if g.covers(p0)]}")

    # 候选 (真边界顶点抽稀)
    layers = []
    for i, sn in enumerate(names):
        if i in fixed_idx:
            layers.append([TRIPLE_CN])
        else:
            layers.append(build_candidates(polys[fulls[i]], n_cap=2500))
    sizes = [len(l) for l in layers]
    print(f"候选: 每槽位 {min(sizes)}~{max(sizes)}, 总 {sum(sizes):,} 点")

    # 初始顺序: 19 展开
    init_order = [name2idx[s] for comb in EXPAND for s in comb if s in name2idx]
    init_order += [i for i in range(34) if i not in init_order]

    # 粗候选 (每槽位 300 均匀) — 顺序搜索用
    coarse = [layers[i][::max(1, len(layers[i]) // 300)] for i in range(34)]
    rad_c = [np.radians(np.array(l)) for l in coarse]

    def dp_eval(order, rad, lyr):
        dp = np.zeros(len(rad[order[0]]))
        back = []
        for t in range(1, 34):
            prev = rad[order[t - 1]]
            cur = rad[order[t]]
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
        path = [lyr[order[t]][idx[t]] for t in range(34)]
        km = sum(haversine(tuple(path[i]), tuple(path[i + 1])) for i in range(33))
        return path, km

    # 顺序搜索 (粗候选)
    order = list(init_order)
    _, km_c = dp_eval(order, rad_c, coarse)
    print(f"粗候选初始: {km_c:.2f} km")
    best_order, best_km = list(order), km_c
    for rnd in range(2):
        improved = False
        for i in range(2, 33):
            for j in range(i + 2, 34):
                o2 = best_order[:i + 1] + best_order[i + 1:j + 1][::-1] + best_order[j + 1:]
                if o2[:3] != best_order[:3]:
                    continue
                _, k2 = dp_eval(o2, rad_c, coarse)
                if k2 < best_km - 1e-3:
                    best_order, best_km = o2, k2
                    improved = True
        for i in range(3, 34):
            for j in range(3, 34):
                if i == j:
                    continue
                o2 = list(best_order)
                v = o2.pop(i)
                o2.insert(j, v)
                _, k2 = dp_eval(o2, rad_c, coarse)
                if k2 < best_km - 1e-3:
                    best_order, best_km = o2, k2
                    improved = True
        print(f"轮 {rnd+1}: {best_km:.2f} km {'改进' if improved else '无改进'}")
        if not improved:
            break

    # 最终 DP (细候选, 最优顺序)
    rad = [np.radians(np.array(l)) for l in layers]
    path, km = dp_eval(best_order, rad, layers)
    print(f"\ncnmaps 34 槽位最优: {km:.2f} km (DataV 基准 7579.76)")

    # 严格覆盖校验 (covers) + 共享检测
    cover = {}
    shared = []
    for t in range(34):
        lon, lat = path[t]
        p = Point(lon, lat)
        cov = [short(nm) for nm, g in polys.items() if g.covers(p)]
        for nm in cov:
            cover.setdefault(nm, []).append(t)
        if t > 0 and haversine(tuple(path[t - 1]), tuple(path[t])) < 1e-6:
            shared.append((names[best_order[t - 1]], names[best_order[t]]))
    missing = [short(nm) for nm in sorted(polys) if short(nm) not in cover]
    print(f"严格 covers 覆盖: {len(cover)}/34")
    print(f"未覆盖: {missing if missing else '无 ✓'}")
    print(f"相邻共享: {shared}")
    print("访问点:")
    for t in range(34):
        lon, lat = path[t]
        p = Point(lon, lat)
        cov = [short(nm) for nm, g in polys.items() if g.covers(p)]
        print(f"  {names[best_order[t]]:<6} ({lon:.5f}, {lat:.5f}) {'/'.join(cov)}")
    print(f"耗时 {time.time()-t0:.0f}s")

    out = {
        'distance_open': round(km, 2),
        'baseline_datav': 7579.76,
        'slot_order': [names[i] for i in best_order],
        'visits': [{'prov': names[best_order[t]], 'lon': float(path[t][0]), 'lat': float(path[t][1])}
                   for t in range(34)],
        'strict_cover': {'n': len(cover), 'missing': missing},
        'shared_pairs': shared,
        'method': 'cnmaps 34-slot DP + order search (真边界投影候选)',
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'output', 'cetsp', 'cetsp_cnmaps_34slot_result.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已保存: {out_path}")


if __name__ == '__main__':
    main()
