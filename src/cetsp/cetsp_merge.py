#!/usr/bin/env python3
"""
cetsp_merge.py — 覆盖合并搜索: 从剪枝解出发, 用覆盖更多省的点合并相邻点对。
例: 新疆/青海 + 西藏/青海 → 青/藏/新三界点 (少1点, 路径基本不变)。
迭代尝试所有点对 + 候选三界点, 接受"替换后路径不增"的合并。
用法: python3 cetsp_merge.py [trials]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from cetsp_province import sample_boundaries, load_province_polygons, border_provinces
from gtsp_core import haversine


def short(nm):
    for suf in ('特别行政区', '壮族自治区', '回族自治区', '维吾尔自治区', '省', '市', '自治区'):
        nm = nm.replace(suf, '')
    return nm


def open_len(seq, pts):
    return sum(haversine(tuple(pts[seq[i]]), tuple(pts[seq[i + 1]])) for i in range(len(seq) - 1))


def best_insert(q, seq, pts):
    """q 插入 seq 的最优位置 (开放路径), 返回 (新seq, 增量)。"""
    best_seq, best_inc = None, float('inf')
    for pos in range(len(seq) + 1):
        if pos == 0:
            inc = haversine(tuple(pts[q]), tuple(pts[seq[0]]))
        elif pos == len(seq):
            inc = haversine(tuple(pts[seq[-1]]), tuple(pts[q]))
        else:
            a, b = seq[pos - 1], seq[pos]
            inc = haversine(tuple(pts[a]), tuple(pts[q])) + haversine(tuple(pts[q]), tuple(pts[b])) \
                  - haversine(tuple(pts[a]), tuple(pts[b]))
        if inc < best_inc:
            best_inc, best_seq = inc, seq[:pos] + [q] + seq[pos:]
    return best_seq, best_inc


def merge_search(tour, pts, polys, comb_to_pts, tol_km=8.0, verbose=True):
    """迭代合并: 对每对点 (i,j), 找覆盖 Si∪Sj 的候选点 q, 替换后路径不增则接受。"""
    tour = list(tour)
    covers = [set(border_provinces(tuple(pts[v]), polys, tol_km=tol_km)) for v in tour]
    n_merge = 0
    improved = True
    while improved:
        improved = False
        for i in range(len(tour)):
            for j in range(i + 1, len(tour)):
                S = covers[i] | covers[j]
                # 候选: 覆盖 ⊇ S 的组合的代表点 (限三界/双界组合)
                cands = []
                for comb, cpts in comb_to_pts.items():
                    if S <= set(comb):
                        cands.extend(cpts)
                if not cands:
                    continue
                # 去重 + 选代表 (几何中心最近)
                seen = set()
                reps = []
                for q in cands:
                    if q in seen:
                        continue
                    seen.add(q)
                    reps.append(q)
                # 替换: 删 i,j 插 q
                rest = [tour[k] for k in range(len(tour)) if k != i and k != j]
                rest_cov = [covers[k] for k in range(len(tour)) if k != i and k != j]
                old_km = open_len(tour, pts)
                for q in reps:
                    new_seq, inc = best_insert(q, rest, pts)
                    # 覆盖校验: q 必须补上 S 里 rest 没覆盖的
                    q_cov = set(border_provinces(tuple(pts[q]), polys, tol_km=tol_km))
                    if not (S - q_cov) <= set().union(*rest_cov) if rest_cov else (S - q_cov) <= set():
                        # 用 union 检查
                        rest_union = set()
                        for c in rest_cov:
                            rest_union |= c
                        if not (S - q_cov) <= rest_union:
                            continue
                    new_km = open_len(new_seq, pts)
                    if new_km <= old_km + 1e-6:
                        # 接受
                        tags_i = '/'.join(sorted(short(p) for p in covers[i]))
                        tags_j = '/'.join(sorted(short(p) for p in covers[j]))
                        tags_q = '/'.join(sorted(short(p) for p in q_cov))
                        tour = new_seq
                        covers = [set(border_provinces(tuple(pts[v]), polys, tol_km=tol_km)) for v in tour]
                        if verbose:
                            print(f"  合并: [{tags_i}] + [{tags_j}] -> [{tags_q}]  ({len(tour)+1}点 -> {len(tour)}点, {old_km:.1f} -> {new_km:.1f} km)")
                        n_merge += 1
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return tour, covers


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tol = 8.0
    # 细采样候选 (含三界点)
    pts, meta, clusters, names = sample_boundaries(step_km=5.0, max_per_prov=300)
    polys = load_province_polygons()

    # 覆盖组合表: comb -> 候选点 (排除南海诸岛: 含海南的组合只留 lat>=17.5 的点)
    comb_to_pts = {}
    for i in range(len(pts)):
        lon, lat = pts[i]
        if lat < 17.5 and '海南' in str(meta[i]):
            continue  # 南海诸岛: 不参与合并
        provs = frozenset(border_provinces((float(lon), float(lat)), polys, tol_km=tol))
        comb_to_pts.setdefault(provs, []).append(i)

    sol = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      'output', 'cetsp', 'cetsp_pruned_result.json')))
    tour = sol['tour']
    coords = np.array(sol['points'])
    print(f"输入: {len(tour)}点 {sol['distance_open']:.1f} km")

    # 注意: pruned_result 的 points 是全局 fine 索引? 检查: tour 是 fine 点全局索引
    # cetsp_pruned_result: tour 来自 cetsp_result(fine) — 全局索引与 pts 一致 ✓
    # 但 coords 是 sol['points'] (fine 全局) — 与 pts 相同数组? 是同一 fine 采样 ✓
    # 校验 pts 是否一致: 直接用 sol 的 points 重建
    pts = np.array(sol['points'])
    tour2, covers = merge_search(tour, pts, polys, comb_to_pts, tol_km=tol)

    # 覆盖校验
    covered = set()
    for v in tour2:
        covered |= set(border_provinces(tuple(pts[v]), polys, tol_km=tol))
    print(f"合并后: {len(tour2)}点 {open_len(tour2, pts):.1f} km, 覆盖 {len(covered)}/34")
    print("访问点:")
    for v in tour2:
        lon, lat = pts[v]
        provs = sorted(border_provinces((float(lon), float(lat)), polys, tol_km=tol), key=lambda p: -len(p))
        print(f"  ({lon:.3f}, {lat:.3f}) {'/'.join(short(p) for p in provs)}")

    out = {
        'distance_open': round(open_len(tour2, pts), 1),
        'n_points': len(tour2),
        'tour': [int(v) for v in tour2],
        'points': pts.tolist(),
        'visits': [{'lon': float(pts[v][0]), 'lat': float(pts[v][1]),
                    'provs': sorted(border_provinces(tuple(pts[v]), polys, tol_km=tol), key=lambda p: -len(p)),
                    'tag': '/'.join(short(p) for p in border_provinces(tuple(pts[v]), polys, tol_km=tol))}
                   for v in tour2],
        'comparison': {'city_gtsp_km': 9665.1,
                       'saving_vs_city_pct': round((1 - open_len(tour2, pts) / 9665.1) * 100, 1)},
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'output', 'cetsp', 'cetsp_merged_result.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已保存: {out_path}")


if __name__ == '__main__':
    main()
