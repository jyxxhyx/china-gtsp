#!/usr/bin/env python3
"""
LKH最优开放路径GTSP可视化地图 v2
- Songti SC字体
- 无标题、无左下角说明框
- 标签碰撞检测自动错开
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from cnmaps import get_adm_maps, draw_map

plt.rcParams['font.sans-serif'] = ['Songti SC', 'Hiragino Sans GB', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROJ = ccrs.PlateCarree()


def resolve_overlaps(positions, width=60, height=22, max_iter=200):
    """简单的力导向标签去重叠。positions: [(x_px, y_px), ...] 就地修改。"""
    n = len(positions)
    if n == 0:
        return positions
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                ox = width - abs(dx)
                oy = height - abs(dy)
                if ox > 0 and oy > 0:
                    # 沿重叠较小的方向推开
                    if ox < oy:
                        push = ox / 2 + 1
                        sgn = 1 if dx >= 0 else -1
                        positions[i][0] += sgn * push
                        positions[j][0] -= sgn * push
                    else:
                        push = oy / 2 + 1
                        sgn = 1 if dy >= 0 else -1
                        positions[i][1] += sgn * push
                        positions[j][1] -= sgn * push
                    moved = True
        if not moved:
            break
    return positions


def main():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cities.csv'))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'road_gtsp_asym_result.json')) as f:
        sol = json.load(f)
    path = sol['path']
    dist_open = sol['distance_open']
    print(f"路线: {dist_open:.0f} km, {len(path)}城")

    lons = df['lng'].values
    lats = df['lat'].values
    names = df['name'].values

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#ffffff')

    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_facecolor('#f5f7fa')
    ax.set_extent([73, 136, 15, 54], crs=PROJ)

    # 合规底图
    try:
        china_land = get_adm_maps(level='国', country='中华人民共和国', kind='陆地')
        if china_land:
            draw_map(china_land[0]['geometry'], color='#e8ecf0', edgecolor='#8899aa', linewidth=0.8)
    except Exception as e:
        print(f"国界警告: {e}")
    try:
        for prov in get_adm_maps(level='省'):
            try:
                draw_map(prov['geometry'], color='none', edgecolor='#c5cdd6', linewidth=0.4)
            except Exception:
                pass
    except Exception:
        pass

    # 候选城市
    ax.scatter(lons, lats, s=8, c='#b8c2cc', alpha=0.5, transform=PROJ, zorder=2)

    # 路线: OSRM真实路网polyline
    import json as _json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'road_segments_asym.json')) as f:
        road_segs = _json.load(f)
    for seg in road_segs:
        if not seg.get('coords'):
            continue
        seg_lons = [c[0] for c in seg['coords']]
        seg_lats = [c[1] for c in seg['coords']]
        ax.plot(seg_lons, seg_lats, '-', color='#d63031', linewidth=1.7,
                transform=PROJ, zorder=4)
    route_lons = [float(lons[c]) for c in path]
    route_lats = [float(lats[c]) for c in path]
    ax.scatter(route_lons, route_lats, s=55, c='#d63031', edgecolors='white',
               linewidths=1.2, transform=PROJ, zorder=5)

    # ===== 标签布局：8方位候选 + 避让路线线段 =====
    base_offset = {
        '哈尔滨市': (10, 8), '长春市': (10, 6), '阜新市': (8, -4), '赤峰市': (10, 10),
        '北京市': (-4, 14), '廊坊市': (12, 6), '天津市': (-18, 8), '滨州市': (10, -2),
        '连云港市': (10, 6), '上海市': (10, 8), '杭州市': (-4, 12), '黄山市': (12, 4),
        '上饶市': (-8, -16), '高雄市': (10, 6), '漳州市': (12, -2), '潮州市': (12, -2),
        '香港特别行政区': (12, -14), '澳门特别行政区': (-6, -16), '海口市': (4, -16),
        '百色市': (-8, -12), '六盘水市': (-12, -6), '昭通市': (-10, 8), '泸州市': (-12, 10),
        '重庆市': (-16, 10), '张家界市': (-16, 4), '十堰市': (-6, 12), '三门峡市': (8, 10),
        '运城市': (6, 10), '铜川市': (-8, 12), '固原市': (4, 12), '兰州市': (-4, 12),
        '西宁市': (-6, 12), '哈密市': (10, 8), '那曲市': (10, 6),
    }

    fig.canvas.draw()
    trans = PROJ.transform_points(ax.projection,
                                  np.array(route_lons), np.array(route_lats))
    disp = ax.transData.transform(trans[:, :2])

    # 路线线段的采样点（像素坐标，用于避让）
    route_pts = []
    for i in range(len(route_lons)):
        t = PROJ.transform_points(ax.projection,
                                  np.array([route_lons[i]]), np.array([route_lats[i]]))
        route_pts.append(ax.transData.transform(t[0, :2]))
    # 线段插值采样
    seg_samples = []
    for i in range(len(route_pts)):
        a = route_pts[i]
        b = route_pts[(i + 1) % len(route_pts)]
        for s in np.linspace(0, 1, 12, endpoint=False):
            seg_samples.append((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))
    seg_samples = np.array(seg_samples)

    # 城市点像素位置（所有点，不仅是路线上的）
    all_pts = []
    t = PROJ.transform_points(ax.projection, np.array(lons), np.array(lats))
    all_disp = ax.transData.transform(t[:, :2])

    LW, LH = 62, 22   # 标签宽高(px)
    DIST = 30         # 引导线长度(px)

    label_texts = []
    label_pos = []
    anchors = []

    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    for i, c in enumerate(path):
        nm = str(names[c])
        px, py = disp[i]

        # 8个候选位置（沿方位外推DIST）
        cands = []
        for ddx, ddy in directions:
            norm = max(abs(ddx), abs(ddy))
            ux, uy = ddx / norm, ddy / norm
            lx = px + ux * DIST - (LW / 2 if ux > 0 else -LW / 2 if ux < 0 else 0)
            ly = py + uy * DIST - (LH / 2 if uy > 0 else -LH / 2 if uy < 0 else 0)
            # 若纯水平/垂直，标签中心在方位上；否则左上角偏移对齐
            cx = px + ux * (DIST + LW / 2 * (abs(ux)))
            cy = py + uy * (DIST + LH / 2 * (abs(uy)))
            cands.append((cx, cy))

        # 打分：离路线采样点的最小距离（越大越好）+ 离其他候选城市点的距离
        best = None
        best_score = -1e18
        for (cx, cy) in cands:
            # 标签中心到路线的最小距离
            d_route = np.min((seg_samples[:, 0] - cx) ** 2 + (seg_samples[:, 1] - cy) ** 2)
            # 标签到其他所有城市的最小距离
            d_cities = np.min((all_disp[:, 0] - cx) ** 2 + (all_disp[:, 1] - cy) ** 2)
            # 标签不要出界（粗略）
            in_bounds = 0 < cx < fig.get_size_inches()[0] * 150 and 0 < cy < fig.get_size_inches()[1] * 150
            score = min(d_route, 40000) + min(d_cities, 2500) + (0 if in_bounds else -1e9)
            if score > best_score:
                best_score = score
                best = (cx, cy)

        label_pos.append(list(best))
        anchors.append((px, py))
        label_texts.append(nm)

    # 标签间碰撞微调
    resolve_overlaps(label_pos, width=LW + 6, height=LH + 6, max_iter=300)

    # 画引导线 + 标签（标签无白底框，只用细描边文字）
    inv = ax.transData.inverted()
    for i, (txt, (px, py), (ax_px, ay_py)) in enumerate(zip(label_texts, label_pos, anchors)):
        # 引导线：从锚点向标签方向画（到标签边缘）
        dx_l, dy_l = px - ax_px, py - ay_py
        ll = np.hypot(dx_l, dy_l)
        if ll > 1:
            ex = ax_px + dx_l * (1 - 14 / ll)
            ey = ay_py + dy_l * (1 - 14 / ll)
            sx = ax_px + dx_l * (8 / ll)
            sy = ay_py + dy_l * (8 / ll)
            s_d = inv.transform((sx, sy))
            e_d = inv.transform((ex, ey))
            ax.plot([s_d[0], e_d[0]], [s_d[1], e_d[1]], color='#7f8c8d',
                    linewidth=0.7, alpha=0.65, zorder=6, transform=ax.projection)

        dxy = inv.transform((px, py))
        ax.text(dxy[0], dxy[1], txt, fontsize=14.5, fontweight='normal', color='#1e272e',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                          edgecolor='none', alpha=0.55, linewidth=0),
                zorder=7, transform=ax.projection)

    # ===== 南海附图 =====
    ax_inset = fig.add_axes([0.82, 0.22, 0.16, 0.24], projection=PROJ)
    ax_inset.set_facecolor('#f5f7fa')
    ax_inset.set_extent([105, 123, 2, 24], crs=PROJ)
    try:
        if china_land:
            draw_map(china_land[0]['geometry'], color='#e8ecf0',
                     edgecolor='#8899aa', linewidth=0.6)
    except Exception:
        pass
    ax_inset.plot(route_lons, route_lats, '-', color='#d63031', linewidth=1.8,
                  transform=PROJ, zorder=4)
    ax_inset.scatter(route_lons, route_lats, s=30, c='#d63031',
                     edgecolors='white', linewidths=0.8, transform=PROJ, zorder=5)
    ax_inset.set_title('南海诸岛', fontsize=15, fontweight='bold')

    # 图例保留（仅路线信息，简短）
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='#d63031', lw=2.2,
               label=f'实际路网路线（非对称路网GTSP最优）总里程 {sum(s["road_km"] for s in road_segs if s["road_km"]):,.0f} km'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d63031',
               markersize=9, label='选中城市（34省各1）'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#b8c2cc',
               markersize=7, label=f'候选城市（{len(df)}个）'),
    ]
    ax.legend(handles=legend_elems, loc='lower left', fontsize=15,
              framealpha=0.95, edgecolor='#aab4be', borderpad=0.8)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'route_roadnet_asym.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"已保存: {os.path.abspath(out)}")


if __name__ == '__main__':
    main()
