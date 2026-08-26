#!/usr/bin/env python3
"""
cetsp_province.py — CETSP 省界版构建器

问题: 344 个非凸多边形邻域(省级行政区边界), 求开放路径使每省至少访问一个边界点。
核心性质: 最优访问点必在边界上 (Gulczynski et al. 2006) → 只需采样边界。

模块:
  load_province_polygons()    cnmaps 34 省界 → {省名: shapely MultiPolygon}
  sample_boundaries(step_km)  边界球面步长采样 → 顶点/候选点/簇/元数据
  border_provinces(pt, tol)   多省交界检测 (画图 label 用)
  refine_visit_points(...)    局部加密精化 (粗采样解 → 细采样重选访问点)
"""
import math
import numpy as np
from cnmaps import get_adm_maps
from shapely.geometry import Point, MultiPolygon, Polygon

from gtsp_core import haversine


# ---------- 数据 ----------
DATAV_PATH = '/tmp/datav_100000_full.json'   # 阿里云 DataV GeoAtlas 省界 (井韶子同源)


def _flatten_polys(geom):
    """递归提取所有 Polygon (处理 GeometryCollection/MultiPolygon 嵌套)。"""
    if geom.geom_type == 'Polygon':
        return [geom]
    if geom.geom_type == 'MultiPolygon':
        return list(geom.geoms)
    if geom.geom_type == 'GeometryCollection':
        out = []
        for p in geom.geoms:
            out.extend(_flatten_polys(p))
        return out
    return []


def load_province_polygons(source='cnmaps'):
    """省界多边形 → {省名: MultiPolygon}。
    source='cnmaps': 默认 (cnmaps 标准地图数据);
    source='datav': 阿里云 DataV GeoAtlas GeoJSON (复现井韶子 7577.91 用)。"""
    from shapely.geometry import shape as _shape
    polys = {}
    if source == 'datav':
        import json as _json
        from shapely.validation import make_valid
        d = _json.load(open(DATAV_PATH))
        for f in d.get('features', []):
            nm = f.get('properties', {}).get('name')
            if not nm:
                continue
            g = _shape(f['geometry'])
            if not g.is_valid:
                g = make_valid(g)
            parts = _flatten_polys(g)
            if not parts:
                continue
            polys[nm] = MultiPolygon(parts) if len(parts) > 1 else MultiPolygon(parts)
        return polys
    for rec in get_adm_maps(level='省'):
        nm = rec.get('province')
        if not nm:
            continue
        g = rec['geometry']
        if isinstance(g, MultiPolygon):
            polys[nm] = g
        elif isinstance(g, Polygon):
            polys[nm] = MultiPolygon([g])
        else:
            # MapPolygon 等: 直接包装
            polys[nm] = MultiPolygon([g]) if g.geom_type == 'Polygon' else g
    return polys


# ---------- 边界采样 ----------
def _sample_ring(ring, step_km, meta_prov, meta_ring, simplify_tol=0.1):
    """沿环累积球面距离等距采样 (先 Douglas-Peucker 抽稀, 再按步长取点)。
    返回 (pts, meta, edges): pts 候选点, meta 每点 (省, 环, 边idx, t), edges 记录。"""
    simp = ring.simplify(simplify_tol, preserve_topology=True)
    coords = list(simp.coords)
    pts, meta, edges = [], [], []
    if len(coords) < 2:
        return pts, meta, edges
    # 起点
    pts.append((coords[0][0], coords[0][1]))
    meta.append((meta_prov, meta_ring, 0, 0.0))
    acc = 0.0
    edge_idx = 0
    for i in range(len(coords) - 1):
        (x1, y1), (x2, y2) = coords[i], coords[i + 1]
        seg_len = haversine((x1, y1), (x2, y2))
        if seg_len <= 1e-9:
            continue
        start = len(pts)
        acc += seg_len
        # 在边上等距取点
        t = (acc - seg_len) / seg_len
        while acc >= step_km - 1e-9:
            over = acc - step_km
            tt = min(1.0, 1.0 - over / seg_len)
            px = x1 + tt * (x2 - x1)
            py = y1 + tt * (y2 - y1)
            pts.append((float(px), float(py)))
            meta.append((meta_prov, meta_ring, edge_idx, tt))
            acc -= step_km
            if acc < step_km:
                break
        edges.append((start, len(pts), edge_idx))
        edge_idx += 1
    return pts, meta, edges


def sample_boundaries(step_km=25.0, max_per_prov=150, source='cnmaps'):
    """所有省界按步长采样。
    返回 (points, meta, clusters, prov_names):
      points:  (n,2) 经纬度; meta: 每点 (省名, 环类型, 边idx, t);
      clusters: 每省点索引列表; prov_names: 省名列表 (簇顺序)。
    max_per_prov: 每省候选点上限 (超限均匀抽稀, 控制 ILS 规模)。"""
    polys = load_province_polygons(source)
    points, meta, clusters, prov_names = [], [], [], []
    for nm in sorted(polys):
        poly = polys[nm]
        start_idx = len(points)
        prov_pts, prov_meta = [], []
        for poly_i, sub in enumerate(poly.geoms):
            # 外环
            pts, m, _ = _sample_ring(sub.exterior, step_km, nm, f'outer{poly_i}')
            prov_pts.extend(pts)
            prov_meta.extend(m)
            # 内环 (洞边界同样可访问)
            for ring_i, interior in enumerate(sub.interiors):
                pts, m, _ = _sample_ring(interior, step_km, nm, f'inner{poly_i}_{ring_i}')
                prov_pts.extend(pts)
                prov_meta.extend(m)
        # 每省点数上限: 均匀抽稀
        if len(prov_pts) > max_per_prov:
            keep = sorted(set(np.linspace(0, len(prov_pts) - 1, max_per_prov).astype(int)))
            prov_pts = [prov_pts[i] for i in keep]
            prov_meta = [prov_meta[i] for i in keep]
        points.extend(prov_pts)
        meta.extend(prov_meta)
        clusters.append(list(range(start_idx, len(points))))
        prov_names.append(nm)
    return (np.array(points), meta, clusters, prov_names)


# ---------- 点-多边形工具 (shapely) ----------
def point_in_province(lonlat, polys, tol_deg=1e-6):
    """点在哪些省的边界/内部 (含容差)。返回省名列表。"""
    pt = Point(lonlat[0], lonlat[1])
    hits = []
    for nm, poly in polys.items():
        # covers: 点在多边形内或边界上
        if poly.covers(pt) or poly.distance(pt) <= tol_deg:
            hits.append(nm)
    return hits


def border_provinces(lonlat, polys, tol_km=8.0):
    """多省交界检测: 距点 tol_km 内的所有省。lat 自适应度-公里换算。"""
    lat = abs(lonlat[1])
    km_per_deg = 111.32 * math.cos(math.radians(lat))
    tol_deg = tol_km / max(km_per_deg, 1.0)
    return point_in_province(lonlat, polys, tol_deg=tol_deg)


# ---------- 局部加密精化 ----------
def refine_visit_points(tour, coarse_coords, coarse_clusters, ils,
                        fine_pts=None, fine_clusters=None, max_rounds=10):
    """两级精化: 把粗解访问点映射到细采样候选(同簇最近), 再交替优化。
    tour: 粗顶点序列 (纯访问点)。返回 (new_tour, total_km) (顶点为 fine 索引)。"""
    from gtsp_core import haversine, haversine_vec
    n = len(tour)
    if fine_pts is None or fine_clusters is None:
        fine_pts, fine_clusters = coarse_coords, coarse_clusters
    rad_c = np.radians(np.asarray(coarse_coords, dtype=float))
    rad_f = np.radians(np.asarray(fine_pts, dtype=float))
    # 起点映射: 每簇粗访问点 → 细候选最近点
    new_tour = []
    for i, v in enumerate(tour):
        k = ils.vc[v]
        fc = np.asarray(fine_clusters[k], dtype=int)
        d = haversine_vec(rad_c[v, 0], rad_c[v, 1], rad_f[fc, 0], rad_f[fc, 1])
        new_tour.append(fc[int(np.argmin(d))])
    def open_total(t):
        return sum(haversine(fine_pts[t[i]], fine_pts[t[i + 1]]) for i in range(n - 1))
    best_total = open_total(new_tour)
    for rnd in range(max_rounds):
        changed = False
        for i in range(n):
            k = ils.vc[tour[i]]          # 簇 id 固定 (粗解)
            cands = np.asarray(fine_clusters[k], dtype=int)
            if i == 0:
                nxt = new_tour[i + 1]
                d2 = haversine_vec(rad_f[nxt, 0], rad_f[nxt, 1], rad_f[cands, 0], rad_f[cands, 1])
                best_j = int(np.argmin(d2))
            elif i == n - 1:
                prev = new_tour[i - 1]
                d1 = haversine_vec(rad_f[prev, 0], rad_f[prev, 1], rad_f[cands, 0], rad_f[cands, 1])
                best_j = int(np.argmin(d1))
            else:
                prev = new_tour[i - 1]
                nxt = new_tour[i + 1]
                d1 = haversine_vec(rad_f[prev, 0], rad_f[prev, 1], rad_f[cands, 0], rad_f[cands, 1])
                d2 = haversine_vec(rad_f[nxt, 0], rad_f[nxt, 1], rad_f[cands, 0], rad_f[cands, 1])
                best_j = int(np.argmin(d1 + d2))
            if cands[best_j] != new_tour[i]:
                new_tour[i] = cands[best_j]
                changed = True
        total = open_total(new_tour)
        if not changed:
            break
        best_total = min(best_total, total)
    return new_tour, best_total


# ---------- 顶层求解流程 ----------
def solve_cetsp(step_km=25.0, time_limit=300, fine_step_km=5.0, seed=42, verbose=True):
    """粗采样 ILS → 局部加密精化。返回解字典。"""
    from gtsp_core import solve_gtsp_open

    points, meta, clusters, prov_names = sample_boundaries(step_km)
    n = len(points)
    if verbose:
        sizes = [len(c) for c in clusters]
        print(f"省界采样: {len(prov_names)}省 {n}点 (每省 {min(sizes)}~{max(sizes)}, 平均 {n//len(clusters)})")

    tour, km, ils = solve_gtsp_open(clusters, coords=points, time_limit=time_limit,
                                    verbose=verbose, seed=seed)
    if verbose:
        print(f"粗采样开放路径: {km:.1f} km")

    # 局部精化: 细采样(5km)候选 → 映射 + 交替优化
    fine_pts, fine_meta, fine_clusters, _ = sample_boundaries(step_km=5.0, max_per_prov=300)
    new_tour, refined_km = refine_visit_points(tour, points, clusters, ils,
                                               fine_pts, fine_clusters)
    if verbose:
        print(f"细采样精化 ({len(fine_pts)}点): {refined_km:.1f} km")

    # 输出
    best_km = min(km, refined_km)
    best_tour = new_tour if refined_km < km else tour
    best_pts = fine_pts if refined_km < km else points
    sol = {
        'distance_open': round(best_km, 1),
        'step_km': step_km,
        'n_points': len(points),
        'n_fine_points': len(fine_pts),
        'tour': [int(v) for v in best_tour],
        'points': best_pts.tolist(),
        'prov_names': prov_names,
        'meta': fine_meta if refined_km < km else meta,
    }
    return sol


if __name__ == '__main__':
    import json, sys
    tlim = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    sol = solve_cetsp(step_km=25.0, time_limit=tlim)
    print(f"\n结果: {sol['distance_open']:.1f} km")
    # 打印访问点 + 省
    polys = load_province_polygons()
    for v in sol['tour']:
        lon, lat = sol['points'][v]
        provs = border_provinces((lon, lat), polys, tol_km=8)
        tag = '·'.join(provs) if len(provs) > 1 else provs[0]
        print(f"  ({lon:.3f}, {lat:.3f}) {tag}")
