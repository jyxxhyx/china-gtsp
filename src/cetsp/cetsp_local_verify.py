#!/usr/bin/env python3
"""
cetsp_local_verify.py — 米级局部最优性验证
对 34 槽位解的每个点: 沿其省边界线, 在点前后 ±5km 弧段内做 100m 步长扫描,
检查 f(p)=d(prev,p)+d(p,next) 是否可改进。全部无改进 → 米级局部最优。
用法: python3 cetsp_local_verify.py [result_json]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from cetsp_province import load_province_polygons
from cetsp_merge import short
from gtsp_core import haversine
import cetsp_34slot as m34

STEP_M = 0.1     # 100m
RADIUS_KM = 5.0  # ±5km


def main():
    result_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'output', 'cetsp', 'cetsp_34slot_result.json')
    sol = json.load(open(result_path))
    order = [m34.name2idx if False else None][0]
    polys = load_province_polygons('datav')
    polys_s = {short(nm): g for nm, g in polys.items()}
    names = sorted(polys_s)
    name2idx = {nm: i for i, nm in enumerate(names)}
    segs_list = [m34.boundary_segments(polys_s[nm]) for nm in names]

    # 解: 每槽位 (按顺序) 的点
    visits = sol['visits']          # 按顺序
    slot_order = [name2idx[v['prov']] for v in visits]
    pts = {slot_order[t]: (v['lon'], v['lat']) for t, v in enumerate(visits)}

    total_improve = 0.0
    n_improved = 0
    print(f"{'省':<6}{'原位置':<24}{'最优位置(局部扫描)':<26}{'改进'}")
    for t in range(34):
        i = slot_order[t]
        lon, lat = pts[i]
        prev_p = pts[slot_order[t - 1]] if t > 0 else None
        next_p = pts[slot_order[t + 1]] if t + 1 < 34 else None
        segs = segs_list[i]
        # 找点所在的线段 (最近线段)
        best_seg, best_t = None, 0.0
        best_d = float('inf')
        for (a, b) in segs:
            # 点在线段上的投影参数
            ax, ay = a; bx, by = b
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            tproj = ((lon - ax) * dx + (lat - ay) * dy) / L2 if L2 > 0 else 0
            tproj = max(0.0, min(1.0, tproj))
            px, py = ax + tproj * dx, ay + tproj * dy
            d = haversine((lon, lat), (px, py))
            if d < best_d:
                best_d, best_seg, best_t = d, (a, b), tproj
        if best_seg is None:
            continue
        # 沿该线段 ±5km 扫描 (线段可能不够长 → 扩展到相邻线段)
        # 简化: 在所在线段上扫描; 若线段短于 10km 则扩大到整线段
        (a, b) = best_seg
        seg_len = haversine(tuple(a), tuple(b))
        f0 = 0.0
        if prev_p: f0 += haversine(tuple(prev_p), (lon, lat))
        if next_p: f0 += haversine((lon, lat), tuple(next_p))
        # 扫描范围: 该线段 + 前后邻段 (弧长 ±5km)
        # 构建局部弧段: 从该线段往前/后扩展至 5km
        segs_ext = []
        idx = segs.index(best_seg)
        # 往前扩展
        cum = 0.0
        for j in range(idx, -1, -1):
            L = haversine(tuple(segs[j][0]), tuple(segs[j][1]))
            if cum + L > RADIUS_KM and j != idx:
                break
            segs_ext.append(segs[j])
            cum += L
        segs_ext.reverse()
        cum = 0.0
        for j in range(idx + 1, len(segs)):
            L = haversine(tuple(segs[j][0]), tuple(segs[j][1]))
            if cum + L > RADIUS_KM:
                break
            segs_ext.append(segs[j])
            cum += L
        # 扫描
        best_p, best_f = (lon, lat), f0
        for (sx, sy) in segs_ext:
            L = haversine(tuple(sx), tuple(sy))
            n_steps = max(1, int(L / STEP_M))
            for k in range(n_steps + 1):
                tfrac = k / n_steps
                p = (sx[0] + tfrac * (sy[0] - sx[0]), sx[1] + tfrac * (sy[1] - sx[1]))
                f = 0.0
                if prev_p: f += haversine(tuple(prev_p), p)
                if next_p: f += haversine(p, tuple(next_p))
                if f < best_f - 1e-9:
                    best_f, best_p = f, p
        imp = f0 - best_f
        total_improve += max(0, imp)
        if imp > 1e-6:
            n_improved += 1
            print(f"{names[i]:<6}({lon:.5f},{lat:.5f})  ({best_p[0]:.5f},{best_p[1]:.5f})  {imp*1000:>8.1f} m")
    print(f"\n结果: 改进点数 {n_improved}/34, 总改进 {total_improve*1000:.1f} m")
    if n_improved == 0:
        print("✓ 全部槽位米级局部最优 (±5km 弧段 100m 扫描无改进)")
    else:
        print(f"⚠ 存在改进空间: 建议用改进点更新解后重跑")


if __name__ == '__main__':
    main()
