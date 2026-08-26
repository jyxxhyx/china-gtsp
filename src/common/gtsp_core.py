#!/usr/bin/env python3
"""
gtsp_core.py — 通用 GTSP 求解核心 (Schmidt & Irnich 2022 ILS 复现)

从 schmidt_ils.py 改造:
1. 距离从"矩阵索引 dist[a][b]"统一为"惰性函数 dist_fn(a, b)" —
   城市版(344城, 矩阵) 与 CETSP(省界采样, 1-2万顶点, 矩阵不可行) 共用一套求解器。
2. 坐标数组 coords 可选: 提供后 dist_fn 默认 = haversine(coords[a], coords[b])。
3. 开放路径入口 solve_gtsp_open(): 虚拟单点簇 + 零距离, 内部闭合求解后还原开放口径。

核心组件 (Schmidt & Irnich 2022, EURO J Comput Optim 10:100029):
  初始解 Random Insertion / VND(2-opt,3-opt,Relocation+,Swap+,SR,CO,Gutin)
  扰动 double-bridge / record-to-record 接受 (eps=3%, 几何冷却)
"""
import numpy as np
import random
import time
import math
from typing import Callable, List, Optional


def haversine(a_lonlat, b_lonlat):
    """球面距离 (km)。输入 (lon, lat) 度。"""
    lo1, la1 = map(math.radians, a_lonlat)
    lo2, la2 = map(math.radians, b_lonlat)
    dlat, dlon = la2 - la1, lo2 - lo1
    s = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(s))


def haversine_vec(lon1, lat1, lon2, lat2):
    """批量球面距离 (km)。输入弧度数组/标量, 输出 (n,) 数组。"""
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def make_dist_fn(dist_or_fn, coords=None):
    """统一距离接口: 接受 n×n 矩阵 或 (a,b)->float 函数。
    无函数时用 coords 的 haversine。返回 callable(a, b)。"""
    if callable(dist_or_fn):
        return dist_or_fn
    M = dist_or_fn
    if coords is not None and M is None:
        return lambda a, b: haversine(coords[a], coords[b])
    return lambda a, b: M[a][b]


class GTSP_ILS:
    def __init__(self, dist, clusters, coords=None):
        """
        dist:    n×n 矩阵 或 callable(a,b)->float 或 None(coords模式)
        clusters: 每簇顶点索引列表 [[...], ...]  (簇=省, 顶点=候选点)
        coords:   (n,2) 经纬度数组, 可选
        """
        self.dist = make_dist_fn(dist, coords)
        self.clusters = clusters
        self.n = sum(len(c) for c in clusters)
        self.N = len(clusters)
        # 顶点全局编号: clusters 已是全局索引 (0..n-1)
        self.vc = [0] * self.n
        for k, mem in enumerate(clusters):
            for v in mem:
                self.vc[v] = k
        self.mde = {}
        self.min_cluster = min(range(self.N), key=lambda k: len(clusters[k]))
        # 大候选集模式下 SR 的 MDE 构建是 O(n·N·m), 默认关闭
        self.use_sr = self.n <= 2000
        # 向量化: coords 提供后, min-over-cluster 内层用 numpy 批量
        self.coords = np.asarray(coords, dtype=float) if coords is not None else None
        self._rad = None
        if self.coords is not None:
            self._rad = np.radians(self.coords)

    def _cand_dist(self, a, cands_idx):
        """a 到候选点列表的批量距离 (km)。cands_idx: 顶点索引 list。"""
        if self._rad is not None:
            la1 = self._rad[a, 1]
            lo1 = self._rad[a, 0]
            return haversine_vec(lo1, la1, self._rad[cands_idx, 0], self._rad[cands_idx, 1])
        return np.array([self.dist(a, v) for v in cands_idx])

    # ---------- 距离工具 ----------
    def tour_cost(self, tour):
        n = len(tour)
        d = self.dist
        return sum(d(tour[i], tour[(i + 1) % n]) for i in range(n))

    # ---------- 初始解: Random Insertion ----------
    def initial_solution(self):
        v0 = random.choice(self.clusters[random.randrange(self.N)])
        tour = [v0]
        visited = {self.vc[v0]}
        remaining = [k for k in range(self.N) if k not in visited]
        d = self.dist
        while remaining:
            k = random.choice(remaining)
            best_cost, best_v, best_pos = float('inf'), None, None
            for v in self.clusters[k]:
                for pos in range(len(tour)):
                    a = tour[pos - 1] if pos > 0 else tour[-1]
                    b = tour[pos] if pos < len(tour) else tour[0]
                    delta = d(a, v) + d(v, b) - d(a, b)
                    if delta < best_cost:
                        best_cost, best_v, best_pos = delta, v, pos
            tour.insert(best_pos, best_v)
            visited.add(k)
            remaining.remove(k)
        return tour

    # ---------- TSP邻域 ----------
    def two_opt(self, tour):
        n = len(tour)
        d = self.dist
        for i in range(n - 1):
            a, b = tour[i], tour[(i + 1) % n]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c, dd = tour[j], tour[(j + 1) % n]
                if d(a, b) + d(c, dd) > d(a, c) + d(b, dd) + 1e-9:
                    new = tour[:]
                    new[i + 1:j + 1] = new[i + 1:j + 1][::-1]
                    return new
        return False

    def three_opt(self, tour):
        t = tour[:]
        changed = self.or_opt_segments(t, max_len=3)
        return t if changed else False

    def or_opt_segments(self, tour, max_len=3):
        n = len(tour)
        d = self.dist
        for L in range(1, max_len + 1):
            for i in range(n):
                if i + L > n:
                    continue
                seg = tour[i:i + L]
                a = tour[i - 1]
                b = tour[(i + L) % n]
                rest = tour[:i] + tour[i + L:]
                cur_gain = d(a, seg[0]) + d(seg[-1], b) - d(a, b)
                if cur_gain <= 1e-9:
                    continue
                m = len(rest)
                for j in range(m):
                    p, q = rest[j], rest[(j + 1) % m]
                    if p == a and q == b:
                        continue
                    new_cost = d(p, seg[0]) + d(seg[-1], q) - d(p, q)
                    if new_cost < cur_gain - 1e-9:
                        new_tour = rest[:j + 1] + seg + rest[j + 1:]
                        tour[:] = new_tour
                        return True
        return False

    def double_bridge(self, tour):
        n = len(tour)
        if n < 8:
            return tour
        a, b, c = sorted(random.sample(range(1, n), 3))
        return tour[:a] + tour[b:c] + tour[a:b] + tour[c:]

    # ---------- 多项式GTSP邻域 ----------
    def relocation_plus(self, tour):
        n = len(tour)
        d = self.dist
        cur = self.tour_cost(tour)
        for i in range(n):
            k = self.vc[tour[i]]
            rest = tour[:i] + tour[i + 1:]
            m = len(rest)
            for j in range(m):
                a, b = rest[j], rest[(j + 1) % m]
                cands = self.clusters[k]
                adds = self._cand_dist(a, cands) + self._cand_dist(b, cands) - d(a, b)
                best_j = int(np.argmin(adds))
                best_v, best_add = cands[best_j], adds[best_j]
                cand = rest[:j + 1] + [best_v] + rest[j + 1:]
                if self.tour_cost(cand) < cur - 1e-9:
                    return cand
        return False

    def swap_plus(self, tour):
        n = len(tour)
        d = self.dist
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, dd = tour[j], tour[(j + 1) % n]
                ki, kj = self.vc[b], self.vc[dd]
                if ki == kj:
                    continue
                dn = tour[(j + 1) % n]
                old4 = d(a, b) + d(b, tour[(i + 1) % n]) + d(c, dd) + d(dd, dn)
                candsi = self.clusters[ki]
                candsj = self.clusters[kj]
                d_vj_a = self._cand_dist(a, candsj)
                d_vj_nxt = self._cand_dist(tour[(i + 1) % n], candsj)
                d_vi_dd = self._cand_dist(dd, candsi)
                d_vi_dn = self._cand_dist(dn, candsi)
                new4 = d_vj_a[:, None] + d_vj_nxt[:, None] + d_vi_dd[None, :] + d_vi_dn[None, :]
                best_flat = int(np.argmin(new4))
                best_vj = candsj[best_flat // len(candsi)]
                best_vi = candsi[best_flat % len(candsi)]
                best_cost = new4.flat[best_flat]
                if best_cost < old4 - 1e-9:
                    trial = tour[:]
                    trial[i], trial[j] = best_vj, best_vi
                    if self.tour_cost(trial) < self.tour_cost(tour) - 1e-9:
                        return trial
        return False

    # ---------- 指数邻域 ----------
    def cluster_optimization(self, tour):
        n = len(tour)
        d = self.dist
        start = min(range(n), key=lambda p: len(self.clusters[self.vc[tour[p]]]))
        rot = tour[start:] + tour[:start]
        N = len(rot)
        cluster_seq = [self.vc[v] for v in rot]
        INF = float('inf')
        dp = [dict() for _ in range(N)]
        for v in self.clusters[cluster_seq[0]]:
            dp[0][v] = 0.0
        for t in range(1, N):
            prev, cur = dp[t - 1], dp[t]
            for v, dv in prev.items():
                for w in self.clusters[cluster_seq[t]]:
                    nd = dv + d(v, w)
                    if nd < cur.get(w, INF):
                        cur[w] = nd
        best_v, best_c = None, INF
        for v, dv in dp[N - 1].items():
            c = dv + d(v, rot[0])
            if c < best_c:
                best_c, best_v = c, v
        if best_c < self.tour_cost(tour) - 1e-9:
            path = [0] * N
            path[N - 1] = best_v
            for t in range(N - 1, 0, -1):
                w = path[t]
                target = dp[t - 1]
                for v in self.clusters[cluster_seq[t - 1]]:
                    if abs(target.get(v, INF) + d(v, w) - dp[t][w]) < 1e-6:
                        path[t - 1] = v
                        break
            if len(set(self.vc[v] for v in path)) == N:
                return path
        return False

    def gutin_neighborhood(self, tour):
        n = len(tour)
        d = self.dist
        Z = []
        for i in range(n):
            if (i - 1) % n in [z % n for z in Z]:
                continue
            if random.random() < 0.5:
                Z.append(i)
        if len(Z) < 2:
            return False
        Zset = set(Z)
        pulled = [(i, tour[i]) for i in sorted(Zset)]
        skeleton = [tour[i] for i in range(n) if i not in Zset]
        m = len(skeleton)
        if m < 2:
            return False
        unp = list(range(len(pulled)))
        used_j = set()
        placed = {}
        new_tour = None
        best_global = self.tour_cost(tour)
        # 贪心指派 (每次挑最小插入代价)
        for _ in range(len(pulled)):
            best_key, best_val = None, float('inf')
            for pi in unp:
                x = pulled[pi][1]
                k = self.vc[x]
                for j in range(m):
                    if j in used_j:
                        continue
                    a_, b_ = skeleton[j], skeleton[(j + 1) % m]
                    cands = self.clusters[k]
                    val = float(np.min(self._cand_dist(a_, cands) + self._cand_dist(b_, cands)))
                    if val < best_val:
                        best_val, best_key = val, (pi, j)
            if best_key is None:
                break
            pi, j = best_key
            placed[pi] = j
            used_j.add(j)
            unp.remove(pi)
        inserts = {}
        for pi, j in placed.items():
            inserts.setdefault(j, []).append(pulled[pi][1])
        new_tour = []
        for j in range(m):
            new_tour.append(skeleton[j])
            if j in inserts:
                new_tour.extend(inserts[j])
        if len(new_tour) == n and self.tour_cost(new_tour) < best_global - 1e-9:
            return new_tour
        return False

    def string_relocation_plus(self, tour, L=4):
        if not self.use_sr:
            return False
        if not self.mde:
            self._build_mde()
        n = len(tour)
        d = self.dist
        cur = self.tour_cost(tour)
        for i in range(n):
            k0 = self.vc[tour[i]]
            for x0 in self.clusters[k0]:
                string = [x0]
                for l in range(0, L):
                    if l > 0:
                        nxt_pos = (i + l) % n
                        nk = self.vc[tour[nxt_pos]]
                        string.append(self.mde[(string[-1], nk)])
                    rem_len = l + 1
                    if rem_len >= n:
                        break
                    aa = tour[(i - 1) % n]
                    bb = tour[(i + rem_len) % n]
                    rem_gain = d(aa, string[0]) + d(string[-1], bb) - d(aa, bb)
                    if rem_gain <= 1e-9:
                        continue
                    rest = tour[:i] + tour[i + rem_len:]
                    m = len(rest)
                    if m < 2:
                        continue
                    for j in range(m):
                        p, q = rest[j], rest[(j + 1) % m]
                        add = d(p, string[0]) + d(string[-1], q) - d(p, q)
                        if add < rem_gain - 1e-9:
                            cand = rest[:j + 1] + string + rest[j + 1:]
                            if self.tour_cost(cand) < cur - 1e-9:
                                return cand
        return False

    def _build_mde(self):
        """MDE查找表: (w, k) -> 簇k中离w最近的顶点"""
        d = self.dist
        for w in range(self.n):
            for k in range(self.N):
                best_v, best_dd = None, float('inf')
                for v in self.clusters[k]:
                    if v == w:
                        continue
                    dd = d(w, v)
                    if dd < best_dd:
                        best_dd, best_v = dd, v
                if best_v is None:
                    best_v = self.clusters[k][0]
                self.mde[(w, k)] = best_v

    # ---------- VND ----------
    def local_search(self, tour):
        cur_cost = self.tour_cost(tour)
        neighborhoods = [
            ('2opt', lambda t: self.two_opt(t)),
            ('3opt', lambda t: self.three_opt(t)),
            ('reloc+', lambda t: self.relocation_plus(t)),
            ('swap+', lambda t: self.swap_plus(t)),
            ('SR', lambda t: self.string_relocation_plus(t)),
            ('CO', lambda t: self.cluster_optimization(t)),
            ('gutin', lambda t: self.gutin_neighborhood(t)),
        ]
        order = list(range(len(neighborhoods)))
        random.shuffle(order)

        def valid(t):
            return len(t) == self.N and len(set(self.vc[v] for v in t)) == self.N

        improved_any = True
        while improved_any:
            improved_any = False
            for idx in order:
                fn = neighborhoods[idx][1]
                res = fn(list(tour))
                if isinstance(res, list):
                    if not valid(res):
                        continue
                    c = self.tour_cost(res)
                    if c < cur_cost - 1e-9:
                        tour = res
                        cur_cost = c
                        improved_any = True
        return tour

    # ---------- 主ILS ----------
    def solve(self, time_limit=60, verbose=True, warm_start=None, seed=None):
        if seed is not None:
            random.seed(seed)
        t0 = time.time()
        eps, h = 0.03, 0.8
        iters = 0
        no_improve = 0
        same_local_cnt = 0
        last_local_cost = None

        x = list(warm_start) if warm_start else self.initial_solution()
        x = self.local_search(x)
        x_best = list(x)
        c_best = self.tour_cost(x)
        if verbose:
            print(f"  初始局部最优: {c_best:.0f}")

        while time.time() - t0 < time_limit:
            iters += 1
            if same_local_cnt >= 3:
                x = self.initial_solution()
                same_local_cnt = 0
            else:
                x = self.double_bridge(x)
            x_new = self.local_search(x)
            c_new = self.tour_cost(x_new)
            c_cur = self.tour_cost(x)
            if c_new < c_cur or c_new <= (1 + eps) * c_best:
                x = x_new
                if c_new < c_best - 1e-9:
                    x_best = list(x_new)
                    c_best = c_new
                    no_improve = 0
                    if verbose:
                        print(f"  [{time.time()-t0:.1f}s] 新纪录: {c_best:.1f} (iter {iters})")
                else:
                    no_improve += 1
            else:
                no_improve += 1
            if last_local_cost is not None and abs(c_new - last_local_cost) < 1e-9:
                same_local_cnt += 1
            else:
                same_local_cnt = 0
            last_local_cost = c_new
            if no_improve >= 50:
                x = list(x_best)
                no_improve = 0
            if iters % self.N == 0:
                eps *= h
        if verbose:
            print(f"  ILS结束: {c_best:.1f} ({iters}次迭代, {time.time()-t0:.0f}s)")
        return x_best, c_best


# ========== 开放路径入口 (虚拟单点簇) ==========
def solve_gtsp_open(clusters, coords=None, dist=None, time_limit=120,
                    verbose=True, warm=None, seed=None, use_sr=None):
    """GTSP 开放路径求解。
    clusters: 每簇顶点索引列表; coords: (n,2) 经纬度 (dist=None 时必填);
    dist: 可选 (a,b)->km 函数或矩阵 (默认 haversine(coords));
    返回 (order, points, open_km):
      order  = 簇访问顺序 (含虚拟簇位置, 用 -1 标记两端), points = 每簇访问点索引。
    """
    n = sum(len(c) for c in clusters)
    N = len(clusters)
    # 虚拟点: 全局编号 n, 距所有点 0
    DUM = n
    clusters2 = clusters + [[DUM]]
    coords2 = None
    if coords is not None:
        coords2 = np.vstack([coords, [[0.0, 0.0]]])
    dist_fn = make_dist_fn(dist, coords2)

    def dist_open(a, b):
        if a == DUM or b == DUM:
            return 0.0
        return dist_fn(a, b)

    ils = GTSP_ILS(dist_open, clusters2, coords2)
    if use_sr is not None:
        ils.use_sr = use_sr
    warm2 = None
    if warm is not None:
        # warm: 开放序列 (不含虚拟点) → 自动补 DUM 成闭合回路
        warm2 = list(warm)
        if DUM not in warm2:
            warm2.append(DUM)
    tour, cost = ils.solve(time_limit=time_limit, verbose=verbose,
                           warm_start=warm2, seed=seed)
    # 还原开放路径: 虚拟点处断开
    d_pos = tour.index(DUM)
    open_tour = tour[d_pos + 1:] + tour[:d_pos]
    open_km = sum(dist_fn(open_tour[i], open_tour[i + 1]) for i in range(len(open_tour) - 1))
    order = [None] * N
    points = []
    for v in open_tour:
        k = ils.vc[v]
        order[k] = v
        points.append(v)
    return open_tour, open_km, ils
