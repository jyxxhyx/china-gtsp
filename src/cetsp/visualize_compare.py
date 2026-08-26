#!/usr/bin/env python3
"""
visualize_compare.py — 两方案对比: 旧(7,578.01 DP) vs 新(7,575.22 启发式+精化)
主图: 两条路径叠加 (旧=灰虚线, 新=红实线); 4 个差异区域放大子图。
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
OLD_C = '#95a5a6'   # 旧方案灰
NEW_C = '#d63031'   # 新方案红


def load(rel):
    sol = json.load(open(os.path.join(BASE, rel)))
    lons = [v['lon'] for v in sol['visits']]
    lats = [v['lat'] for v in sol['visits']]
    provs = [v['prov'] for v in sol['visits']]
    return lons, lats, provs, sol['distance_open']


def draw_base(ax):
    ax.set_facecolor('#f5f7fa')
    try:
        for prov in get_adm_maps(level='省'):
            try:
                draw_map(prov['geometry'], color='#e8ecf0', edgecolor='#8899aa', linewidth=0.7)
            except Exception:
                pass
    except Exception:
        pass


def plot_path(ax, lons, lats, color, lw, ls, ms, z=4, label=None):
    ax.plot(lons, lats, '-', color=color, linewidth=lw, linestyle=ls,
            transform=PROJ, zorder=z, label=label)
    ax.scatter(lons, lats, s=ms, c=color, edgecolors='white', linewidths=1.0,
               transform=PROJ, zorder=z + 1)


def main():
    lons_o, lats_o, provs_o, d_o = load('output/cetsp_cnmaps_34slot_result.json')
    lons_n, lats_n, provs_n, d_n = load('output/cetsp_refined_result.json')

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#ffffff')
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_extent([73, 136, 15, 54], crs=PROJ)
    draw_base(ax)
    plot_path(ax, lons_o, lats_o, OLD_C, 2.0, '--', 45, z=4, label=f'旧方案 (DP) {d_o:,.2f} km')
    plot_path(ax, lons_n, lats_n, NEW_C, 2.4, '-', 55, z=6, label=f'新方案 (启发式+精化) {d_n:,.2f} km')
    ax.legend(loc='lower left', fontsize=13, framealpha=0.95, edgecolor='#aab4be')

    # 差异区域放大: 渝黔 / 苏皖 / 鲁豫 / 京津冀
    zooms = [
        ('渝黔段', [107.9, 109.2, 27.9, 29.3], [0.045, 0.42, 0.24, 0.26]),
        ('苏皖段', [116.6, 120.1, 31.7, 33.3], [0.045, 0.10, 0.24, 0.26]),
        ('鲁豫段', [114.7, 117.9, 34.2, 36.6], [0.72, 0.42, 0.25, 0.26]),
        ('京津冀段', [115.9, 117.9, 39.3, 40.6], [0.72, 0.10, 0.25, 0.26]),
    ]
    for (name, extent, rect) in zooms:
        a2 = fig.add_axes(rect, projection=PROJ)
        a2.set_extent(extent, crs=PROJ)
        draw_base(a2)
        plot_path(a2, lons_o, lats_o, OLD_C, 2.2, '--', 60, z=4)
        plot_path(a2, lons_n, lats_n, NEW_C, 2.6, '-', 75, z=6)
        a2.set_title(name, fontsize=13, fontweight='bold')
        # 标注省名
        for i, p in enumerate(provs_n):
            a2.annotate(p, (lons_n[i], lats_n[i]), textcoords='offset points',
                        xytext=(6, -10), fontsize=8.5, color=NEW_C, fontweight='bold')
        for i, p in enumerate(provs_o):
            if p not in provs_n or True:
                a2.annotate(p, (lons_o[i], lats_o[i]), textcoords='offset points',
                            xytext=(6, 8), fontsize=8.5, color=OLD_C, alpha=0.9)
        a2.grid(alpha=0.25, linestyle='--')

    # 南海附图
    ax_inset = fig.add_axes([0.82, 0.80, 0.14, 0.16], projection=PROJ)
    ax_inset.set_facecolor('#f5f7fa')
    ax_inset.set_extent([105, 123, 2, 24], crs=PROJ)
    try:
        for prov in get_adm_maps(level='省'):
            g = prov['geometry']
            if g.bounds[1] < 24:
                draw_map(g, color='#e8ecf0', edgecolor='#8899aa', linewidth=0.6)
    except Exception:
        pass
    plot_path(ax_inset, lons_n, lats_n, NEW_C, 1.6, '-', 25, z=5)
    ax_inset.plot(lons_o, lats_o, '-', color=OLD_C, linewidth=1.2, linestyle='--',
                  transform=PROJ, zorder=4)

    out = os.path.join(BASE, 'output', 'cetsp', 'cetsp_compare_7578_vs_7575.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"已保存: {out}")


if __name__ == '__main__':
    main()
