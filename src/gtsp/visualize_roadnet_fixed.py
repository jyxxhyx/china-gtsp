#!/usr/bin/env python3
"""
路线图 v3: 智能标签布局 (线上发布版 12,552.3 km, 台湾轮渡修复口径)
- 标签真实尺寸测量 (renderer)
- 多半径 × 8 方向候选, 按"标签边缘到锚点间隙"升序贪心放置
- 硬约束: 不出界 / 不压选中城市红点 / 不与已放置标签重叠
- 软罚: 压候选灰点数 / 压路线距离
- 台湾渡轮段 (ferry=True) 画红色虚线
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
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 行政区划代码前两位 → 省份简称
PROV = {11: '北京', 12: '天津', 13: '河北', 14: '山西', 15: '内蒙古', 21: '辽宁', 22: '吉林',
        23: '黑龙江', 31: '上海', 32: '江苏', 33: '浙江', 34: '安徽', 35: '福建', 36: '江西',
        37: '山东', 41: '河南', 42: '湖北', 43: '湖南', 44: '广东', 45: '广西', 46: '海南',
        50: '重庆', 51: '四川', 52: '贵州', 53: '云南', 54: '西藏', 61: '陕西', 62: '甘肃',
        63: '青海', 64: '宁夏', 65: '新疆', 71: '台湾', 81: '香港', 82: '澳门'}
MUNI = {'北京市', '天津市', '上海市', '重庆市', '香港特别行政区', '澳门特别行政区'}

FS = 13.0          # 标签字号
RED = '#d63031'
GRAY = '#b8c2cc'


def place_labels(anchors, texts, sizes, fig_w, fig_h, route_samples, red_pts, gray_pts,
                 radii=(6, 14, 24, 38, 56, 78, 104), pad=14, hints=None):
    """贪心标签放置。anchors: [(x,y)像素] 城市点; sizes: [(w,h)像素] 标签真实尺寸。
    hints: {index: (dx,dy)} 指定标签优先方向(像素坐标系), 该方向无合格位置时回退全方向。
    返回 [(cx,cy), ...] 标签中心, 以及每个标签的统计信息。"""
    n = len(anchors)
    hints = hints or {}
    dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    placed = []          # 已放置标签 bbox (x0,y0,x1,y1)
    results = [None] * n
    stats = [None] * n

    def mkbox(cx, cy, w, h):
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def overlaps(b):
        for p in placed:
            if not (b[2] <= p[0] - pad or b[0] >= p[2] + pad or b[3] <= p[1] - pad or b[1] >= p[3] + pad):
                return True
        return False

    def count_inside(b, pts):
        c = 0
        for (x, y) in pts:
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                c += 1
        return c

    def covers_red(b):
        for (x, y) in red_pts:
            if b[0] - 3 <= x <= b[2] + 3 and b[1] - 3 <= y <= b[3] + 3:
                return True
        return False

    def crowd_penalty(b, thresh=60):
        """与已放置标签投影接近(间隙<thresh)则罚 1, 用于避免'上下排列像重叠'。"""
        for p in placed:
            gx = max(0, max(b[0], p[0]) - min(b[2], p[2]))
            gy = max(0, max(b[1], p[1]) - min(b[3], p[3]))
            if gx < thresh and gy < thresh:
                return 1
        return 0

    # 放置顺序: 锚点密集区先放 (它们需要更多选择空间, 后放的避开先放的框)
    density = []
    for i in range(n):
        cnt = 0
        for j in range(n):
            if j != i and abs(anchors[i][0] - anchors[j][0]) < 220 and abs(anchors[i][1] - anchors[j][1]) < 130:
                cnt += 1
        density.append(cnt)
    order = sorted(range(n), key=lambda i: (-density[i], i))
    # hint 城市强制最先放, 按路线顺序 (滁州在南京前, 避免南京先占位)
    if hints:
        order = sorted([i for i in order if i in hints]) + [i for i in order if i not in hints]

    def select(cands_all):
        """优先 grade<=1 中 r 最小; 全部明显压线才选离路线最远。"""
        for target_grade in (0, 1):
            pool = [c for c in cands_all if c[5] <= target_grade]
            if pool:
                return min(pool, key=lambda c: (c[0], c[3]))
        if cands_all:
            return min(cands_all, key=lambda c: (-c[4], c[0]))
        return None

    for i in order:
        ax, ay = anchors[i]
        w, h = sizes[i]
        hint = hints.get(i)
        if hint is not None:
            dx, dy = hint[0], hint[1]
            rmin = hint[2] if len(hint) > 2 else 0
            hdirs = [(dx, dy)]
        else:
            hdirs = None
            rmin = 0

        def collect(dset):
            out = []
            for r in radii:
                if r < rmin:
                    continue
                for dx, dy in dset:
                    norm = (dx * dx + dy * dy) ** 0.5
                    ux, uy = dx / norm, dy / norm
                    # 标签中心: 锚点沿方向外推 r + 矩形支撑半宽
                    cx = ax + ux * (r + (w / 2) * abs(ux) + (h / 2) * abs(uy))
                    cy = ay + uy * (r + (w / 2) * abs(ux) + (h / 2) * abs(uy))
                    b = mkbox(cx, cy, w, h)
                    if b[0] < 0 or b[1] < 0 or b[2] > fig_w or b[3] > fig_h:
                        continue
                    if overlaps(b) or covers_red(b):
                        continue
                    g_cnt = count_inside(b, gray_pts)
                    d_route = np.min((route_samples[:, 0] - cx) ** 2 + (route_samples[:, 1] - cy) ** 2)
                    half = max(w, h) / 2
                    grade = 0 if d_route >= (half - 6) ** 2 else (1 if d_route >= (half - 20) ** 2 else 2)
                    out.append((r, cx, cy, g_cnt, d_route, grade))
            return out

        if hdirs:
            ch = collect(hdirs)
            pool = [c for c in ch if c[5] <= 1]
            best = min(pool, key=lambda c: (c[0], c[3])) if pool else None
            if best is None:
                best = select(collect(dirs))   # hint 方向无合格位置, 回退全方向
                if best is not None:
                    print(f"WARN: hint 回退 i={i} hdirs={hdirs} -> r={best[0]} d_route={best[4]**0.5:.0f}")
        else:
            best = select(collect(dirs))
        if best is not None:
            r, cx, cy, g_cnt, d_route, _g = best
            placed.append(mkbox(cx, cy, w, h))
            results[i] = (cx, cy)
            stats[i] = (r, np.hypot(cx - ax, cy - ay), g_cnt, d_route ** 0.5)
        else:
            # 兜底: 放宽压红点, 尽量靠近锚点
            for r in radii:
                for dx, dy in dirs:
                    norm = (dx * dx + dy * dy) ** 0.5
                    ux, uy = dx / norm, dy / norm
                    cx = ax + ux * (r + (w / 2) * abs(ux) + (h / 2) * abs(uy))
                    cy = ay + uy * (r + (w / 2) * abs(ux) + (h / 2) * abs(uy))
                    b = mkbox(cx, cy, w, h)
                    if b[0] < 0 or b[1] < 0 or b[2] > fig_w or b[3] > fig_h:
                        continue
                    if overlaps(b):
                        continue
                    d_route = np.min((route_samples[:, 0] - cx) ** 2 + (route_samples[:, 1] - cy) ** 2)
                    if best is None or (r, -d_route) < (best[0], -best[4]):
                        best = (r, cx, cy, 0, d_route, 0)
            if best is not None:
                r, cx, cy, _, d_route, _g = best
                placed.append(mkbox(cx, cy, w, h))
                results[i] = (cx, cy)
                stats[i] = (r, np.hypot(cx - ax, cy - ay), 0, d_route ** 0.5)
            else:
                results[i] = (ax, ay)
                stats[i] = (0, 0, -1, -1)
    return results, stats


def main():
    df = pd.read_csv(os.path.join(BASE, 'data', 'gtsp', 'cities.csv'))
    with open(os.path.join(BASE, 'output', 'gtsp', 'road_gtsp_fixed_result.json')) as f:
        sol = json.load(f)
    with open(os.path.join(BASE, 'output', 'gtsp', 'road_segments_fixed.json')) as f:
        road_segs = json.load(f)
    path = sol['path']
    dist_open = sol['distance_open']
    print(f"路线: {dist_open:.1f} km, {len(path)}城")

    lons = df['lng'].values
    lats = df['lat'].values
    names = df['name'].values
    codes = df['code'].values

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#ffffff')
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_facecolor('#f5f7fa')
    ax.set_extent([73, 136, 15, 54], crs=PROJ)

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

    # 候选城市 (灰点)
    ax.scatter(lons, lats, s=8, c=GRAY, alpha=0.5, transform=PROJ, zorder=2)

    # 路线 polyline (渡轮段虚线)
    seg_samples_all = []
    for seg in road_segs:
        if not seg.get('coords'):
            continue
        coords = seg['coords']
        ferry = seg.get('ferry', False)
        ls = '--' if ferry else '-'
        seg_lons = [c[0] for c in coords]
        seg_lats = [c[1] for c in coords]
        ax.plot(seg_lons, seg_lats, ls, color=RED, linewidth=1.7, transform=PROJ, zorder=4)
        # 降采样存路线采样点 (像素)
        step = max(1, len(coords) // 150)
        for c in coords[::step]:
            seg_samples_all.append(c)
    route_lons = [float(lons[c]) for c in path]
    route_lats = [float(lats[c]) for c in path]
    ax.scatter(route_lons, route_lats, s=55, c=RED, edgecolors='white',
               linewidths=1.2, transform=PROJ, zorder=5)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # 城市点像素坐标
    t = PROJ.transform_points(ax.projection, np.array(lons), np.array(lats))
    all_disp = ax.transData.transform(t[:, :2])
    t2 = PROJ.transform_points(ax.projection, np.array(route_lons), np.array(route_lats))
    disp = ax.transData.transform(t2[:, :2])

    # 路线采样点 → 像素
    t3 = PROJ.transform_points(ax.projection,
                               np.array([c[0] for c in seg_samples_all]),
                               np.array([c[1] for c in seg_samples_all]))
    route_samples = ax.transData.transform(t3[:, :2])

    fig_w, fig_h = fig.canvas.get_width_height()

    # 标签文字 + 真实尺寸
    texts = []
    for c in path:
        nm = str(names[c])
        code = str(codes[c])
        prov = PROV.get(int(code[:2]), '')
        if nm in MUNI:
            texts.append(nm)
        else:
            texts.append(f'{nm}·{prov}')
    sizes = []
    for txt in texts:
        tmp = ax.text(0, 0, txt, fontsize=FS, ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                                edgecolor='none', alpha=0.55, linewidth=0))
        bb = tmp.get_window_extent(renderer)
        sizes.append((bb.width, bb.height))
        tmp.remove()

    # 标签放置 (密集区手动指定方向; 方向为像素坐标, dy=+1 下, dy=-1 上)
    MANUAL = {
        '滁州市': (-1, 0),     # 左
        '南京市': (-1, -1),    # 左上 (回退自动即原位置)
        '上海市': (1, 0),      # 右
        '廊坊市': (1, 1),      # 右下 (北京/天津红点夹击, 远放但方向避开)
        '运城市': (0, 1),      # 下
        '三门峡市': (0, -1),   # 上
        '阳江市': (-1, 1),     # 左下 (海面空旷)
        '三明市': (1, -1),     # 右上 (更靠上)
        '赣州市': (1, 1),      # 右下
        '十堰市': (1, 1),      # 右下 (江汉平原空旷)
        '安康市': (0, -1),     # 上 (秦岭方向空旷)
        '兰州市': (0, 1),      # 下 (避开西宁红点)
        '固原市': (0, -1),     # 上 (给运城左侧腾位置)
    }
    hints = {k: MANUAL[str(names[path[k]])] for k in range(len(path)) if str(names[path[k]]) in MANUAL}
    anchors = [(float(disp[i][0]), float(disp[i][1])) for i in range(len(path))]
    red_pts = [(float(disp[i][0]), float(disp[i][1])) for i in range(len(path))]
    gray_pts = [(float(all_disp[i][0]), float(all_disp[i][1])) for i in range(len(lons))]
    label_pos, stats = place_labels(anchors, texts, sizes, fig_w, fig_h,
                                    route_samples, red_pts, gray_pts, hints=hints)

    print(f"{'城市':<8}{'间隙r':>5}{'中心距':>7}{'压灰点':>6}{'路线距':>7}")
    bboxes = []
    for i, c in enumerate(path):
        nm = str(names[c])
        r, cd, g, rd = stats[i]
        w, h = sizes[i]
        cx, cy = label_pos[i]
        bboxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        dx = cx - anchors[i][0]
        dy = cy - anchors[i][1]
        dstr = f"({'右' if dx > 40 else '左' if dx < -40 else '·'}{'下' if dy > 25 else '上' if dy < -25 else '·'})"
        print(f"{nm:<8}{r:>5}{cd:>7.1f}{g:>6}{rd:>7.1f} {dstr}")
    # 标签间最小间距检查
    min_gap = 1e9; gap_pair = None
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            a, b = bboxes[i], bboxes[j]
            gx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
            gy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
            if gx < 40 and gy < 40:   # x/y 投影都接近才比较
                gap = (gx * gx + gy * gy) ** 0.5
                if gap < min_gap:
                    min_gap = gap
                    gap_pair = (str(names[path[i]]), str(names[path[j]]), gx, gy)
    print(f"最近标签对: {gap_pair} (间隙 {min_gap:.1f}px)" if gap_pair else "标签无近距离对")
    # 压路线检查
    import numpy as _np
    over_route = []
    for i, c in enumerate(path):
        cx, cy = label_pos[i]
        w, h = sizes[i]
        d = _np.min((route_samples[:, 0] - cx) ** 2 + (route_samples[:, 1] - cy) ** 2) ** 0.5
        half = max(w, h) / 2
        if d < half:
            over_route.append((str(names[c]), round(d, 1), round(half, 1)))
    print("压路线标签:", over_route if over_route else "无")

    # 引导线 + 标签
    inv = ax.transData.inverted()
    for i, (txt, (px, py), (ax_px, ay_py)) in enumerate(zip(texts, label_pos, anchors)):
        r = stats[i][0]
        dx_l, dy_l = px - ax_px, py - ay_py
        ll = np.hypot(dx_l, dy_l)
        if ll > 22:   # 标签离点足够远才画引导线
            ux, uy = dx_l / ll, dy_l / ll
            w, h = sizes[i]
            support = (w / 2) * abs(ux) + (h / 2) * abs(uy)
            ex = px - ux * support          # 标签框边缘
            ey = py - uy * support
            sx = ax_px + ux * 4             # 城市点边缘
            sy = ay_py + uy * 4
            s_d = inv.transform((sx, sy))
            e_d = inv.transform((ex, ey))
            ax.plot([s_d[0], e_d[0]], [s_d[1], e_d[1]], color='#57606a',
                    linewidth=1.3, alpha=0.9, zorder=6, transform=ax.projection)
            # 标签端锚点圆点 (让短线也清晰可见)
            ax.plot([e_d[0]], [e_d[1]], marker='o', markersize=4,
                    color='#57606a', alpha=0.9, zorder=6, transform=ax.projection)
        dxy = inv.transform((px, py))
        ax.text(dxy[0], dxy[1], txt, fontsize=FS, fontweight='normal', color='#1e272e',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                          edgecolor='none', alpha=0.55, linewidth=0),
                zorder=7, transform=ax.projection)

    # 南海附图
    ax_inset = fig.add_axes([0.82, 0.22, 0.16, 0.24], projection=PROJ)
    ax_inset.set_facecolor('#f5f7fa')
    ax_inset.set_extent([105, 123, 2, 24], crs=PROJ)
    try:
        if china_land:
            draw_map(china_land[0]['geometry'], color='#e8ecf0',
                     edgecolor='#8899aa', linewidth=0.6)
    except Exception:
        pass
    for seg in road_segs:
        if not seg.get('coords'):
            continue
        ls = '--' if seg.get('ferry', False) else '-'
        ax_inset.plot([c[0] for c in seg['coords']], [c[1] for c in seg['coords']],
                      ls, color=RED, linewidth=1.8, transform=PROJ, zorder=4)
    ax_inset.scatter(route_lons, route_lats, s=30, c=RED,
                     edgecolors='white', linewidths=0.8, transform=PROJ, zorder=5)
    ax_inset.set_title('南海诸岛', fontsize=15, fontweight='bold')

    # 图例
    from matplotlib.lines import Line2D
    road_total = sum(s['road_km'] for s in road_segs if s.get('road_km'))
    legend_elems = [
        Line2D([0], [0], color=RED, lw=2.2, label=f'路线 {dist_open:,.0f} km'),
        Line2D([0], [0], color=RED, lw=2.2, ls='--', label='渡轮航线（台湾海峡）'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RED,
               markersize=9, label='选中城市（34省各1）'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GRAY,
               markersize=7, label=f'候选城市（{len(df)}个）'),
    ]
    ax.legend(handles=legend_elems, loc='lower left', fontsize=14,
              framealpha=0.95, edgecolor='#aab4be', borderpad=0.8)

    out = os.path.join(BASE, 'output', 'gtsp', 'gtsp_route_roadnet_fixed.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"已保存: {os.path.abspath(out)}")
    print(f"路网总里程(road_km合计): {road_total:,.1f} km")


if __name__ == '__main__':
    main()
