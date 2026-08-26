#!/usr/bin/env python3
"""
Schmidt & Irnich (2022) ILS for GTSP — 复现
EURO J Comput Optim 10:100029

核心组件（按论文§3-4）:
1. 初始解: Random Insertion（随机选簇，最优顶点插入）
2. 局部搜索: VND with 邻域集
   - 2-Opt, 3-Opt (TSP邻域, first-improvement)
   - Double-Bridge (special 4-opt, perturbation也用)
   - Relocation+ (重定位+可换顶点)
   - Swap+ (交换+可换顶点)
   - CO (Cluster Optimization, 层状图最短路, DP)
   - Gutin邻域 (随机Z集 + 指派问题, 用贪心替代指派求解)
   - SR (String Relocation+, L=4, MDE启发式)
3. 扰动: 随机double-bridge
4. 接受准则: record-to-record travel (epsilon=3%, 几何冷却 h=0.8,
   每N次迭代更新)
5. 改进: 3次回落同一局部最优→重新构造；50次无改进→reset到最优
"""

import os
import numpy as np
import random
import time
import json
import math
from typing import List, Tuple


class GTSP_ILS:
    def __init__(self, dist, clusters):
        self.dist = dist
        self.clusters = clusters
        self.n = len(dist)
        self.N = len(clusters)
        # 顶点->簇
        self.vc = [0] * self.n
        for k, mem in enumerate(clusters):
            for v in mem:
                self.vc[v] = k
        # MDE查找表: MDE[w][i] = 簇i中距w最近的顶点 (§3.3 SR用)
        self.mde = {}
        # 最小簇（CO旋转用, Karapetyan-Gutin技巧）
        self.min_cluster = min(range(self.N), key=lambda k: len(clusters[k]))

    # ---------- 距离工具 ----------
    def tour_cost(self, tour):
        n = len(tour)
        return sum(self.dist[tour[i]][tour[(i + 1) % n]] for i in range(n))

    # ---------- §4.1 初始解: Random Insertion ----------
    def initial_solution(self):
        # 随机选起始顶点
        v0 = random.choice(self.clusters[random.randrange(self.N)])
        tour = [v0]
        visited = {self.vc[v0]}
        remaining = [k for k in range(self.N) if k not in visited]
        while remaining:
            k = random.choice(remaining)
            # 最优插入顶点和位置
            best_cost, best_v, best_pos = float('inf'), None, None
            for v in self.clusters[k]:
                for pos in range(len(tour)):
                    a = tour[pos - 1] if pos > 0 else tour[-1]
                    b = tour[pos] if pos < len(tour) else tour[0]
                    delta = self.dist[a][v] + self.dist[v][b] - self.dist[a][b]
                    if delta < best_cost:
                        best_cost, best_v, best_pos = delta, v, pos
            tour.insert(best_pos, best_v)
            visited.add(k)
            remaining.remove(k)
        return tour

    # ---------- §3.1 TSP邻域 ----------
    def two_opt(self, tour):
        """First-improvement 2-opt. 返回新list或False。"""
        n = len(tour)
        for i in range(n - 1):
            a, b = tour[i], tour[(i + 1) % n]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c, d = tour[j], tour[(j + 1) % n]
                if self.dist[a][b] + self.dist[c][d] > self.dist[a][c] + self.dist[b][d] + 1e-9:
                    new = tour[:]
                    new[i + 1:j + 1] = new[i + 1:j + 1][::-1]
                    return new
        return False

    def three_opt(self, tour):
        """简化3-opt: or-opt segments. 返回新list或False。"""
        t = tour[:]
        changed = self.or_opt_segments(t, max_len=3)
        return t if changed else False

    def or_opt_segments(self, tour, max_len=3):
        """Segment relocation (TSP版, 不换顶点)。原地修改tour。"""
        n = len(tour)
        for L in range(1, max_len + 1):
            for i in range(n):
                if i + L > n:
                    continue
                seg = tour[i:i + L]
                a = tour[i - 1]
                b = tour[(i + L) % n]
                rest = tour[:i] + tour[i + L:]
                cur_gain = self.dist[a][seg[0]] + self.dist[seg[-1]][b] - self.dist[a][b]
                if cur_gain <= 1e-9:
                    continue
                m = len(rest)
                for j in range(m):
                    p, q = rest[j], rest[(j + 1) % m]
                    if p == a and q == b:
                        continue
                    new_cost = self.dist[p][seg[0]] + self.dist[seg[-1]][q] - self.dist[p][q]
                    if new_cost < cur_gain - 1e-9:
                        new_tour = rest[:j + 1] + seg + rest[j + 1:]
                        tour[:] = new_tour
                        return True
        return False

    def double_bridge(self, tour):
        """随机double-bridge扰动 (§4.1 Perturbation)"""
        n = len(tour)
        if n < 8:
            return tour
        a, b, c = sorted(random.sample(range(1, n), 3))
        return tour[:a] + tour[b:c] + tour[a:b] + tour[c:]

    # ---------- §3.2 多项式GTSP邻域 ----------
    def relocation_plus(self, tour):
        """Relocation+: 移动x_i到别处，可换同簇顶点。 O(nN)"""
        n = len(tour)
        cur = self.tour_cost(tour)
        for i in range(n):
            k = self.vc[tour[i]]
            rest = tour[:i] + tour[i + 1:]
            m = len(rest)
            for j in range(m):
                a, b = rest[j], rest[(j + 1) % m]
                best_v, best_add = None, float('inf')
                for v in self.clusters[k]:
                    add = self.dist[a][v] + self.dist[v][b] - self.dist[a][b]
                    if add < best_add:
                        best_add, best_v = add, v
                cand = rest[:j + 1] + [best_v] + rest[j + 1:]
                if self.tour_cost(cand) < cur - 1e-9:
                    return cand
        return False

    def swap_plus(self, tour):
        """Swap+: 交换两个位置顶点，各自可换同簇顶点。O(n^2)"""
        n = len(tour)
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                old = self.dist[a][b] + self.dist[c][d]
                ki, kj = self.vc[b], self.vc[d]
                if ki == kj:
                    continue
                dn = tour[(j + 1) % n]
                old4 = self.dist[a][b] + self.dist[b][tour[(i + 1) % n]] + \
                       self.dist[c][d] + self.dist[d][dn]
                best_cost, best_vi, best_vj = float('inf'), None, None
                for vi in self.clusters[ki]:
                    for vj in self.clusters[kj]:
                        new4 = self.dist[a][vj] + self.dist[vj][tour[(i + 1) % n]] + \
                               self.dist[vi][d] + self.dist[vi][dn]
                        if new4 < best_cost:
                            best_cost, best_vi, best_vj = new4, vi, vj
                if best_cost < old4 - 1e-9:
                    trial = tour[:]
                    trial[i], trial[j] = best_vj, best_vi
                    if self.tour_cost(trial) < self.tour_cost(tour) - 1e-9:
                        return trial
        return False

    # ---------- §3.3 指数邻域 ----------
    def cluster_optimization(self, tour):
        """CO: 簇序固定, DP层状图选每簇最优顶点。
        Karapetyan-Gutin: 旋转到最小簇开头。O(n * m_max * m_min)"""
        n = len(tour)
        # 旋转使最小簇在首位
        start = min(range(n), key=lambda p: len(self.clusters[self.vc[tour[p]]]))
        rot = tour[start:] + tour[:start]
        N = len(rot)
        cluster_seq = [self.vc[v] for v in rot]

        m0 = len(self.clusters[cluster_seq[0]])
        INF = float('inf')
        # DP: dp[t][j] = 到第t层用簇cluster_seq[t]的顶点j的最短长度
        # 第0层与最后一层同簇（回路）
        dp = [dict() for _ in range(N)]
        for v in self.clusters[cluster_seq[0]]:
            dp[0][v] = 0.0
        for t in range(1, N):
            prev, cur = dp[t - 1], dp[t]
            for v, dv in prev.items():
                for w in self.clusters[cluster_seq[t]]:
                    nd = dv + self.dist[v][w]
                    if nd < cur.get(w, INF):
                        cur[w] = nd
        # 闭合: 最后顶点回到首顶点
        best_v, best_c = None, INF
        for v, dv in dp[N - 1].items():
            c = dv + self.dist[v][rot[0]]
            if c < best_c:
                best_c, best_v = c, v
        # 回溯
        if best_c < self.tour_cost(tour) - 1e-9:
            path = [0] * N
            path[N - 1] = best_v
            for t in range(N - 1, 0, -1):
                w = path[t]
                target = dp[t - 1]
                for v in self.clusters[cluster_seq[t - 1]]:
                    if abs(target.get(v, INF) + self.dist[v][w] - dp[t][w]) < 1e-6:
                        path[t - 1] = v
                        break
            # 检查每簇一个
            if len(set(self.vc[v] for v in path)) == N:
                return path
        return False

    def gutin_neighborhood(self, tour):
        """Gutin邻域适配: 随机Z集(期望N/3), 抽出Z中顶点,
        对每个空位用"最好簇代表"贪心重插（指派问题的贪心近似）。"""
        n = len(tour)
        # 随机Z: 相邻不共边, 逐点p=0.5
        Z = []
        for i in range(n):
            if (i - 1) % n in [z % n for z in Z]:
                continue
            if random.random() < 0.5:
                Z.append(i)
        if len(Z) < 2:
            return False
        Zset = set(Z)
        # 抽出的顶点及其簇
        pulled = [(i, tour[i]) for i in sorted(Zset)]
        # 剩余骨架
        skeleton = [tour[i] for i in range(n) if i not in Zset]
        m = len(skeleton)
        if m < 2:
            return False
        # 插入代价 a[i][j] = min_{x'∈V[xi]} (c_{xj-1,x'} + c_{x',xj+1})
        # 贪心指派: 每次挑(顶点,位置)最小代价
        unp = list(range(len(pulled)))  # pulled索引
        placed = {}
        cost_matrix = {}
        for pi in range(len(pulled)):
            x = pulled[pi][1]
            k = self.vc[x]
            for j in range(m):
                a_, b_ = skeleton[j], skeleton[(j + 1) % m]
                best = min(self.dist[a_][v] + self.dist[v][b_] for v in self.clusters[k])
                cost_matrix[(pi, j)] = best
        # 贪心
        used_j = set()
        for _ in range(len(unp)):
            best_key, best_val = None, float('inf')
            for pi in unp:
                for j in range(m):
                    if j in used_j:
                        continue
                    if cost_matrix[(pi, j)] < best_val:
                        best_val, best_key = cost_matrix[(pi, j)], (pi, j)
            if best_key is None:
                break
            pi, j = best_key
            placed[pi] = j
            used_j.add(j)
            unp.remove(pi)
        # 重建tour
        new_tour = []
        inserts = {}
        for pi, j in placed.items():
            inserts.setdefault(j, []).append(pulled[pi][1])
        for j in range(m):
            new_tour.append(skeleton[j])
            if j in inserts:
                new_tour.extend(inserts[j])
        if len(new_tour) == n and self.tour_cost(new_tour) < self.tour_cost(tour) - 1e-9:
            return new_tour
        return False

    def string_relocation_plus(self, tour, L=4):
        """SR: 移长≤L的字串到别处，串内顶点用MDE链式替换。O(NLn)"""
        if not self.mde:
            self._build_mde()
        n = len(tour)
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
                    rem_gain = self.dist[aa][string[0]] + self.dist[string[-1]][bb] - self.dist[aa][bb]
                    if rem_gain <= 1e-9:
                        continue
                    rest = tour[:i] + tour[i + rem_len:]
                    if rest:
                        rest = rest[:]
                        # 旋转rest使插入逻辑一致
                        pass
                    m = len(rest)
                    if m < 2:
                        continue
                    for j in range(m):
                        p, q = rest[j], rest[(j + 1) % m]
                        add = self.dist[p][string[0]] + self.dist[string[-1]][q] - self.dist[p][q]
                        if add < rem_gain - 1e-9:
                            cand = rest[:j + 1] + string + rest[j + 1:]
                            if self.tour_cost(cand) < cur - 1e-9:
                                return cand
        return False

    def _build_mde(self):
        """MDE查找表: 对每(顶点w, 簇i)存簇i中最近顶点"""
        for w in range(self.n):
            for k in range(self.N):
                best_v, best_d = None, float('inf')
                for v in self.clusters[k]:
                    if v == w:
                        continue
                    if self.dist[w][v] < best_d:
                        best_d, best_v = self.dist[w][v], v
                if best_v is None:
                    best_v = self.clusters[k][0]
                self.mde[(w, k)] = best_v

    # ---------- VND ----------
    def local_search(self, tour):
        """Random VND (§4.1): 每次调用随机排序邻域优先级。"""
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
            return len(t) == self.N and \
                   len(set(self.vc[v] for v in t)) == self.N

        improved_any = True
        while improved_any:
            improved_any = False
            for idx in order:
                name, fn = neighborhoods[idx]
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

    # ---------- 主ILS (Algorithm 2) ----------
    def solve(self, time_limit=60, verbose=True, warm_start=None):
        t0 = time.time()
        eps = 0.03
        h = 0.8
        iters = 0
        no_improve = 0
        same_local_cnt = 0
        last_local_cost = None

        if warm_start:
            x = list(warm_start)
        else:
            x = self.initial_solution()
        x = self.local_search(x)
        x_best = list(x)
        c_best = self.tour_cost(x)

        if verbose:
            print(f"  初始局部最优: {c_best:.0f}")

        while time.time() - t0 < time_limit:
            iters += 1
            # 扰动
            if same_local_cnt >= 3:
                x = self.initial_solution()
                same_local_cnt = 0
            else:
                x = self.double_bridge(x)
            # 局部搜索
            x_new = self.local_search(x)
            c_new = self.tour_cost(x_new)
            c_cur = self.tour_cost(x)

            # 接受准则: record-to-record
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

            # 回落同一局部最优检测
            if last_local_cost is not None and abs(c_new - last_local_cost) < 1e-9:
                same_local_cnt += 1
            else:
                same_local_cnt = 0
            last_local_cost = c_new

            # reset (§5.1): 50次无改进回到最优
            if no_improve >= 50:
                x = list(x_best)
                no_improve = 0

            # 冷却: 每N次迭代 eps *= h
            if iters % self.N == 0:
                eps *= h

        elapsed = time.time() - t0
        if verbose:
            print(f"  ILS结束: {c_best:.1f} ({iters}次迭代, {elapsed:.0f}s)")
        return x_best, c_best


# ========== China GTSP入口 ==========
def load_china():
    import pandas as pd
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(_root, 'data', 'gtsp', 'cities.csv'))
    pm = {'11':'北京','12':'天津','13':'河北','14':'山西','15':'内蒙古','21':'辽宁','22':'吉林','23':'黑龙江',
          '31':'上海','32':'江苏','33':'浙江','34':'安徽','35':'福建','36':'江西','37':'山东','41':'河南',
          '42':'湖北','43':'湖南','44':'广东','45':'广西','46':'海南','50':'重庆','51':'四川','52':'贵州',
          '53':'云南','54':'西藏','61':'陕西','62':'甘肃','63':'青海','64':'宁夏','65':'新疆','71':'台湾',
          '81':'香港','82':'澳门'}
    df['prov'] = df['code'].astype(str).str[:2].map(pm)
    cd = {}
    for idx, row in df.iterrows():
        if not isinstance(row['prov'], str):
            continue
        cd.setdefault(row['prov'], []).append(idx)
    names = sorted(cd)
    clusters = [cd[nm] for nm in names]

    n = len(df)
    dist = np.zeros((n, n))
    lons, lats = df['lng'].values, df['lat'].values
    for i in range(n):
        for j in range(i + 1, n):
            lo1, la1, lo2, la2 = map(math.radians, [lons[i], lats[i], lons[j], lats[j]])
            a = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
            d = 6371 * 2 * math.asin(math.sqrt(a))
            dist[i][j] = dist[j][i] = d
    return df, clusters, names, dist


if __name__ == '__main__':
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df, clusters, names, dist = load_china()
    print(f"China GTSP: {len(df)}城 {len(clusters)}簇")

    ils = GTSP_ILS(dist, clusters)

    # 用LKH 9,700km解做warm start对比（开放路径→ILS按闭合处理需要虚拟节点）
    # 这里先做闭合版实验: 加虚拟节点
    n = len(dist)
    N = n + 1
    dist2 = np.zeros((N, N))
    dist2[:n, :n] = dist
    clusters2 = clusters + [[n]]
    ils = GTSP_ILS(dist2, clusters2)

    # warm start: LKH开放解哈密→哈密闭合(9700+1137=10837) vs 从零开始
    import sys
    tlim = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    warm = None
    try:
        with open(os.path.join(_root, 'output', 'gtsp', 'schmidt_ils_result.json')) as f:
            sol = json.load(f)
        warm = sol['path'] + [n]  # 34城+虚拟D = 35
        print(f"warm start(开放9700km闭合版): {ils.tour_cost(warm):.0f}")
    except Exception as e:
        print("no warm start:", e)

    tour, cost = ils.solve(time_limit=tlim, warm_start=warm)
    # 去虚拟节点求开放距离
    d_pos = tour.index(n)
    open_tour = tour[d_pos+1:] + tour[:d_pos]
    open_cost = sum(dist[open_tour[i]][open_tour[i+1]] for i in range(len(open_tour)-1))
    print(f"\n开放路径距离: {open_cost:.1f} km (LKH记录: 9700)")
    print("路线:", ' → '.join(df.iloc[c]['name'] for c in open_tour))

    json.dump({'distance_open': round(open_cost, 1), 'path': open_tour,
               'cities': [df.iloc[c]['name'] for c in open_tour]},
              open(os.path.join(_root, 'output', 'gtsp', 'schmidt_ils_result.json'), 'w'),
              ensure_ascii=False, indent=2)
    print("saved: output/schmidt_ils_result.json")
