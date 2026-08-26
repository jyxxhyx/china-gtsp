#!/usr/bin/env python3
"""
cetsp_exact_solve.py — 米级口径精确求解 (候选 = 线段端点 + 段内驻点)
19 槽位固定顺序; 每槽位可行线 = 组合省边界交集 (双界共享线/单省全界);
槽位1(藏青新)/17(京津冀) 为固定三界点 (交集为空, 实测 0m 三界点)。
候选集: 每段端点 + 段内 f(p)=d(prev,p)+d(p,next) 的驻点 (黄金分割)。
迭代: DP(候选内全局最优) → 用解更新相邻位置 → 重算驻点 → DP, 3 轮收敛。
输出: 规模 + 距离 + 时间。
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from shapely.geometry import Point
from cetsp_province import load_province_polygons
from cetsp_merge import short
from gtsp_core import haversine, haversine_vec

COMBS = [
    ['西藏', '青海', '新疆'], ['甘肃', '宁夏'], ['陕西', '四川'], ['湖北', '重庆'],
    ['湖南', '贵州'], ['广西', '云南'], ['海南'], ['广东', '澳门'], ['香港'],
    ['台湾'], ['福建'], ['浙江', '江西'], ['上海'], ['安徽', '江苏'],
    ['河南', '山东'], ['山西'], ['北京', '天津', '河北'], ['辽宁', '内蒙古'],
    ['吉林', '黑龙江'],
]
FIXED = {0: (89.711414, 36.093272), 16: (117.389879, 40.227958)}   # 三界点 (实测 0m)
JING_KM = 7579.76   # 井韶子坐标重算 (连续最优)


def feasible_segments(polys_s, comb):
    """组合省边界交集 → 线段列表 [(a, b)]"""
    inter = None
    for sn in comb:
        b = polys_s[sn].boundary
        inter = b if inter is None else inter.intersection(b)
    segs = []
    if inter is None or inter.is_empty:
        return segs
    geoms = [inter] if inter.geom_type == 'LineString' else \
            (list(inter.geoms) if inter.geom_type == 'MultiLineString' else [])
    if inter.geom_type == 'GeometryCollection':
        geoms = [g for g in inter.geoms if g.geom_type == 'LineString']
    for g in geoms:
        if g.geom_type != 'LineString':
            continue
        coords = list(g.coords)
        for i in range(len(coords) - 1):
            segs.append((coords[i], coords[i + 1]))
    return segs


def point_on_seg(seg, t):
    """线段内线性插值 t ∈ [0,1]"""
    (a, b) = seg
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def slot_candidates(segs, prev_pos, next_pos, extra_per_seg=0):
    """候选 = 每段端点 + 段内 f 驻点 + 每段均匀 extra 个加密点。返回坐标列表。"""
    cands = []
    for (a, b) in segs:
        cands.append(a)
        # 加密点 (均匀)
        if extra_per_seg > 0:
            for k in range(1, extra_per_seg + 1):
                cands.append(point_on_seg((a, b), k / (extra_per_seg + 1)))
        # 段内驻点: min f(t) = d(prev, p(t)) + d(p(t), next), t∈[0,1]
        if prev_pos is None and next_pos is None:
            cands.append(b)
            continue
        def cost(t):
            p = point_on_seg((a, b), t)
            c = 0.0
            if prev_pos is not None:
                c += haversine(tuple(prev_pos), p)
            if next_pos is not None:
                c += haversine(p, tuple(next_pos))
            return c
        lo, hi = 0.0, 1.0
        gr = (np.sqrt(5) - 1) / 2
        x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
        c1, c2 = cost(x1), cost(x2)
        for _ in range(40):
            if c1 < c2:
                hi, x2, c2 = x2, x1, c1
                x1 = hi - gr * (hi - lo)
                c1 = cost(x1)
            else:
                lo, x1, c1 = x1, x2, c2
                x2 = lo + gr * (hi - lo)
                c2 = cost(x2)
        t_best = (lo + hi) / 2
        # 端点或驻点 (若驻点不在端点附近, 加入)
        pb = point_on_seg((a, b), t_best)
        if haversine(tuple(a), pb) > 1e-6 and haversine(tuple(b), pb) > 1e-6:
            cands.append(pb)
        cands.append(b)
    # 去重 (同坐标)
    out = []
    seen = set()
    for p in cands:
        k = (round(p[0], 6), round(p[1], 6))
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def main():
    t0 = time.time()
    polys = load_province_polygons('datav')
    polys_s = {short(nm): g for nm, g in polys.items()}
    print("构建可行线段...")
    segs_list = []
    for comb in COMBS:
        segs_list.append(feasible_segments(polys_s, comb))
        L = sum(haversine(tuple(s[0]), tuple(s[1])) for s in segs_list[-1])
        print(f"  槽位{len(segs_list):2d} {'/'.join(comb):<14} 线段数 {len(segs_list[-1]):>6}"
              f" 总长 {L*1000:>12,.0f} m" + ("  [固定三界点]" if len(segs_list)-1 in FIXED else ""))

    # 初始位置: 每槽位用段中点 (或固定点)
    pos = {}
    for i in range(19):
        if i in FIXED:
            pos[i] = FIXED[i]
        elif segs_list[i]:
            mid = point_on_seg(segs_list[i][0], 0.5)
            pos[i] = mid
        else:
            pos[i] = (0.0, 0.0)

    order = list(range(19))

    def total_km():
        return sum(haversine(tuple(pos[order[i]]), tuple(pos[order[i + 1]]))
                   for i in range(len(order) - 1))

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--extra', type=int, default=0, help='每段额外均匀采样点数')
    args = ap.parse_args()
    EXTRA = args.extra

    best_km = None
    for it in range(4):
        t1 = time.time()
        # 候选构建 (用当前相邻位置)
        layers = []
        for i in range(19):
            if i in FIXED:
                layers.append([FIXED[i]])
            else:
                prev_p = pos[i - 1] if i > 0 else None
                next_p = pos[i + 1] if i + 1 < 19 else None
                layers.append(slot_candidates(segs_list[i], prev_p, next_p, EXTRA))
        sizes = [len(l) for l in layers]
        n_pair = sum(sizes[i] * sizes[i + 1] for i in range(18))
        print(f"\n迭代{it+1}: 每槽位候选 {min(sizes)}~{max(sizes)} (总 {sum(sizes)}), 边 {n_pair:,.0e}")

        # DP (候选内全局最优)
        rad = np.radians(np.array([p for l in layers for p in l]))
        # 槽位层索引偏移
        offset = [0]
        for l in layers:
            offset.append(offset[-1] + len(l))
        dp = {j: 0.0 for j in range(len(layers[0]))}
        back = []
        for t in range(1, 19):
            prev_keys = list(dp.keys())
            prev_vals = np.array(list(dp.values()))
            cur = {}
            bk = {}
            for j in range(len(layers[t])):
                p = layers[t][j]
                d = haversine_vec(np.radians(p[0]), np.radians(p[1]),
                                  np.radians([layers[t-1][k][0] for k in prev_keys]),
                                  np.radians([layers[t-1][k][1] for k in prev_keys]))
                m = int(np.argmin(d + prev_vals))
                cur[j] = d[m] + prev_vals[m]
                bk[j] = prev_keys[m]
            dp = cur
            back.append(bk)
        last = min(dp, key=dp.get)
        path_idx = [last]
        for bk in reversed(back):
            path_idx.append(bk[path_idx[-1]])
        path_idx.reverse()
        for t in range(19):
            pos[t] = layers[t][path_idx[t]]
        km = total_km()
        print(f"  DP 结果: {km:.2f} km (井韶子坐标 7579.76), 耗时 {time.time()-t1:.1f}s")
        if best_km is not None and best_km - km < 1e-3:
            break
        best_km = km

    print(f"\n最终: {best_km:.2f} km, 总耗时 {time.time()-t0:.1f}s")
    print("访问点:")
    for i in range(19):
        lon, lat = pos[i]
        print(f"  ({lon:.6f}, {lat:.6f})")

    out = {
        'distance_open': round(best_km, 2),
        'target': JING_KM,
        'diff': round(best_km - JING_KM, 2),
        'method': 'segment-endpoint + stationary-point candidates, DP, 19-slot (DataV)',
        'visits': [{'lon': float(pos[i][0]), 'lat': float(pos[i][1])} for i in range(19)],
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'output', 'cetsp', 'cetsp_exact_solve_result.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已保存: {out_path}")


if __name__ == '__main__':
    main()
