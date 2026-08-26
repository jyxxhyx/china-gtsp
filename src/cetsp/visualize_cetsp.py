#!/usr/bin/env python3
"""
visualize_cetsp.py — CETSP 省界访问路线可视化
- 底图: cnmaps 合规省界 (暗色主题)
- 路径: 访问点间球面直线
- 访问点: 红色圆点; label 显示归属省, 多省交界处标注所有省 (如 陕/鄂/渝)
- 标注布局: 复用 visualize_roadnet_fixed 的贪心放置 (贴点优先, 防重叠)
用法: python3 visualize_cetsp.py [result_json]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from cnmaps import get_adm_maps, draw_map

plt.rcParams['font.sans-serif'] = ['Songti SC', 'Hiragino Sans GB', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROJ = ccrs.PlateCarree()
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FS = 13.0
RED = '#d63031'


def short(nm):
    for suf in ('特别行政区', '壮族自治区', '回族自治区', '维吾尔自治区', '省', '市', '自治区'):
        nm = nm.replace(suf, '')
    return nm


def place_labels(anchors, texts, sizes, fig_w, fig_h, red_pts,
                 radii=(6, 14, 24, 38, 56, 78, 104), pad=14):
    """贪心标签放置 (从 visualize_roadnet_fixed 提取)。"""
    n = len(anchors)
    dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    placed = []
    results = [None] * n

    def mkbox(cx, cy, w, h):
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def overlaps(b):
        for p in placed:
            if not (b[2] <= p[0] - pad or b[0] >= p[2] + pad or b[3] <= p[1] - pad or b[1] >= p[3] + pad):
                return True
        return False

    def covers_red(b):
        for (x, y) in red_pts:
            if b[0] - 3 <= x <= b[2] + 3 and b[1] - 3 <= y <= b[3] + 3:
                return True
        return False

    density = []
    for i in range(n):
        cnt = sum(1 for j in range(n) if j != i and
                  abs(anchors[i][0] - anchors[j][0]) < 220 and abs(anchors[i][1] - anchors[j][1]) < 130)
        density.append(cnt)
    order = sorted(range(n), key=lambda i: (-density[i], i))

    for i in order:
        ax_p, ay_p = anchors[i]
        w, h = sizes[i]
        best = None
        for r in radii:
            for dx, dy in dirs:
                norm = (dx * dx + dy * dy) ** 0.5
                ux, uy = dx / norm, dy / norm
                cx = ax_p + ux * (r + (w / 2) * abs(ux) + (h / 2) * abs(uy))
                cy = ay_p + uy * (r + (w / 2) * abs(ux) + (h / 2) * abs(uy))
                b = mkbox(cx, cy, w, h)
                if b[0] < 0 or b[1] < 0 or b[2] > fig_w or b[3] > fig_h:
                    continue
                if overlaps(b) or covers_red(b):
                    continue
                if best is None or r < best[0]:
                    best = (r, cx, cy)
        if best is None:
            best = (6, ax_p, ay_p)
        r, cx, cy = best
        placed.append(mkbox(cx, cy, w, h))
        results[i] = (cx, cy)
    return results


def main():
    result_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'output', 'cetsp', 'cetsp_result.json')
    sol = json.load(open(result_path))
    visits = sol['visits']
    dist = sol['distance_open']

    lons = np.array([v['lon'] for v in visits])
    lats = np.array([v['lat'] for v in visits])
    # label 用完整覆盖省集 (8km 容差标注多省交界; 严格校验见文档/verify 脚本)
    from cetsp_province import load_province_polygons, border_provinces
    _polys = load_province_polygons()
    tags = []
    for v in visits:
        provs = border_provinces((float(v['lon']), float(v['lat'])), _polys, tol_km=8)
        tags.append('/'.join(short(p) for p in sorted(provs, key=lambda p: -len(p))))

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#ffffff')
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_facecolor('#f5f7fa')
    ax.set_extent([73, 136, 15, 54], crs=PROJ)

    # 合规底图: 省界
    try:
        for prov in get_adm_maps(level='省'):
            try:
                draw_map(prov['geometry'], color='#e8ecf0', edgecolor='#8899aa', linewidth=0.8)
            except Exception:
                pass
    except Exception:
        pass

    # 路径 (访问点顺序连线, 首尾为开放端点)
    ax.plot(lons, lats, '-', color=RED, linewidth=2.0, transform=PROJ, zorder=4)
    ax.scatter(lons, lats, s=70, c=RED, edgecolors='white', linewidths=1.4,
               transform=PROJ, zorder=5)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    t = PROJ.transform_points(ax.projection, lons, lats)
    disp = ax.transData.transform(t[:, :2])
    anchors = [(float(disp[i][0]), float(disp[i][1])) for i in range(len(lons))]

    # label 文本: 多省交界用 / 分隔列出全部省
    texts = []
    for v in visits:
        provs = v.get('provs') or [v.get('prov')] or ['?']
        if len(provs) > 1:
            texts.append('/'.join(short(p) for p in provs))
        else:
            texts.append(short(provs[0]))

    sizes = []
    for txt in texts:
        tmp = ax.text(0, 0, txt, fontsize=FS, ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                                edgecolor='none', alpha=0.6, linewidth=0))
        bb = tmp.get_window_extent(renderer)
        sizes.append((bb.width, bb.height))
        tmp.remove()

    fig_w, fig_h = fig.canvas.get_width_height()
    red_pts = anchors
    label_pos = place_labels(anchors, texts, sizes, fig_w, fig_h, red_pts)

    inv = ax.transData.inverted()
    for i, (txt, (px, py), (ax_p, ay_p)) in enumerate(zip(texts, label_pos, anchors)):
        dx_l, dy_l = px - ax_p, py - ay_p
        ll = np.hypot(dx_l, dy_l)
        if ll > 30:
            ux, uy = dx_l / ll, dy_l / ll
            w, h = sizes[i]
            support = (w / 2) * abs(ux) + (h / 2) * abs(uy)
            ex = px - ux * support
            ey = py - uy * support
            sx = ax_p + ux * 6
            sy = ay_p + uy * 6
            s_d = inv.transform((sx, sy))
            e_d = inv.transform((ex, ey))
            ax.plot([s_d[0], e_d[0]], [s_d[1], e_d[1]], color='#57606a',
                    linewidth=1.1, alpha=0.9, zorder=6, transform=ax.projection)
        dxy = inv.transform((px, py))
        ax.text(dxy[0], dxy[1], txt, fontsize=FS, color='#1e272e',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                          edgecolor='none', alpha=0.6, linewidth=0),
                zorder=7, transform=ax.projection)

    # 南海附图
    ax_inset = fig.add_axes([0.82, 0.22, 0.16, 0.24], projection=PROJ)
    ax_inset.set_facecolor('#f5f7fa')
    ax_inset.set_extent([105, 123, 2, 24], crs=PROJ)
    try:
        for prov in get_adm_maps(level='省'):
            g = prov['geometry']
            if g.bounds[1] < 24:
                draw_map(g, color='#e8ecf0', edgecolor='#8899aa', linewidth=0.6)
    except Exception:
        pass
    ax_inset.plot(lons, lats, '-', color=RED, linewidth=1.8, transform=PROJ, zorder=4)
    ax_inset.scatter(lons, lats, s=30, c=RED, edgecolors='white', linewidths=0.8,
                     transform=PROJ, zorder=5)
    ax_inset.set_title('南海诸岛', fontsize=15, fontweight='bold')

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=RED, lw=2.4, label=f'省界访问路线 {dist:,.0f} km (球面)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RED,
               markersize=9, label='访问点（每省边界各1，可落多省交界）'),
    ]
    ax.legend(handles=legend_elems, loc='lower left', fontsize=14,
              framealpha=0.95, edgecolor='#aab4be', borderpad=0.8)

    out = os.path.join(BASE, 'output', 'cetsp', 'cetsp_route.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"已保存: {os.path.abspath(out)}")


if __name__ == '__main__':
    main()
