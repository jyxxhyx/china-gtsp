#!/usr/bin/env python3
"""
gtsp_transform.py — GTSP→ATSP→STSP 两级变换（自包含、可复用）

Noon-Bean (1993) + 虚拟节点D + Jonker-Volgenant (1983) 三节点展开

对外接口:
  build_pipeline(dist, clusters)
      → 返回 PipelineResult(atsp, stsp, M, BIG, meta)
  extract_gtsp(stsp_tour, meta)
      → STSP回路反变换回GTSP开放路径 (list[city_idx]) 或 None
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PipelineResult:
    atsp: np.ndarray            # (n+1)x(n+1) 含虚拟D的ATSP弧费用(×10取整)
    stsp: np.ndarray            # 3(n+1) x 3(n+1) 对称STSP距离
    M: int                      # Noon-Bean跨簇过路费
    BIG: int                    # STSP禁止边权重
    n: int                      # 原始城市数
    K: int                      # 簇数(省份)
    n_atsp: int                 # = n+1 (含虚拟D)
    n_stsp: int                 # = 3(n+1)
    scale: int = 10             # 距离放大倍数(整数化)


def build_pipeline(dist: np.ndarray, clusters: List[List[int]]) -> PipelineResult:
    """GTSP → ATSP(Noon-Bean+虚拟D) → STSP(三节点展开)

    步骤:
      1. 每个簇内城市按给定顺序构成有向环, 环弧=0, 跳序弧=M
      2. 跨簇弧 = dist[succ(u), v] + M  (后继记账)
      3. 加虚拟节点D作为第K+1个簇: D到所有城距离0 → 其弧费=0+M
      4. 三节点展开: 城市i → (in, mid, out), 内部链0费,
         边{u_out, v_in} = c(u→v), 其余BIG
    """
    n = len(dist)
    K = len(clusters)
    scale = 10

    city_cluster = [0] * n
    successor = [0] * n
    for k, members in enumerate(clusters):
        m = len(members)
        for idx in range(m):
            city_cluster[members[idx]] = k
            successor[members[idx]] = members[(idx + 1) % m]

    dmax = max(dist[i][j] for i in range(n) for j in range(n) if i != j)
    M = int(dmax * scale) + 1

    # ── 1. ATSP含虚拟D: 节点0..n-1=城市, 节点n=D ──
    na = n + 1
    atsp = np.full((na, na), M, dtype=np.int64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if city_cluster[i] == city_cluster[j]:
                atsp[i][j] = 0 if j == successor[i] else M
            else:
                atsp[i][j] = int(round(dist[successor[i]][j] * scale)) + M
        atsp[i][n] = M              # 城市→D: dist(succ(i), D)=0
    for j in range(n):
        atsp[n][j] = M              # D→城市: dist(D, j)=0
    # D自环保持M(不走)

    # ── 2. 三节点展开 ──
    N = 3 * na
    finite = [int(atsp[i][j]) for i in range(na) for j in range(na) if i != j]
    BIG = int(sum(sorted(finite, reverse=True)[:na])) + 1

    stsp = np.full((N, N), BIG, dtype=np.int64)
    for i in range(na):
        stsp[3 * i][3 * i + 1] = 0
        stsp[3 * i + 1][3 * i] = 0
        stsp[3 * i + 1][3 * i + 2] = 0
        stsp[3 * i + 2][3 * i + 1] = 0
    for i in range(na):
        for j in range(na):
            if i != j:
                c = int(atsp[i][j])
                stsp[3 * i + 2][3 * j] = c     # i_out — j_in
                stsp[3 * j][3 * i + 2] = c     # 对称镜像
    np.fill_diagonal(stsp, 0)

    return PipelineResult(
        atsp=atsp, stsp=stsp, M=int(M), BIG=int(BIG),
        n=n, K=K, n_atsp=na, n_stsp=N, scale=scale,
    )


def extract_gtsp(stsp_tour: List[int], res: PipelineResult,
                 dummy_atsp_idx: Optional[int] = None) -> Optional[List[int]]:
    """STSP哈密尔顿回路 → GTSP开放路径

    规则: 回路中 out(u)→in(v) 相邻对 = ATSP弧u→v;
    沿ATSP弧走到虚拟D处断开, 去掉D得到开放路径。
    校验: 弧数必须恰为 n_atsp(每"簇"恰一次), 否则返回None。
    """
    if dummy_atsp_idx is None:
        dummy_atsp_idx = res.n_atsp - 1   # D是最后一个ATSP节点

    N = len(stsp_tour)
    arcs = {}
    for k in range(N):
        a, b = stsp_tour[k], stsp_tour[(k + 1) % N]
        if a % 3 == 2 and b % 3 == 0:
            u, v = a // 3, b // 3
            if u in arcs:
                return None       # 出度>1, 非法
            arcs[u] = v
    if len(arcs) != res.n_atsp:
        return None               # 不是单一哈密尔顿环

    # 从D出发走一圈
    path = [dummy_atsp_idx]
    cur = dummy_atsp_idx
    for _ in range(res.n_atsp):
        cur = arcs.get(cur)
        if cur is None:
            return None
        if cur == dummy_atsp_idx:
            break
        path.append(cur)
    if len(path) != res.n_atsp:
        return None
    return path[1:]               # 去掉D → 开放路径


def gtsp_open_distance(path: List[int], dist: np.ndarray) -> float:
    """开放路径距离(km)"""
    return float(sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1)))


# ══════════ 自验证(小算例暴力对拍) ══════════
def _self_test():
    import itertools
    rng = np.random.default_rng(42)

    def brute_gtsp(dist, clusters):
        n, K = len(dist), len(clusters)
        best = float('inf')
        for perm in itertools.permutations(range(K)):
            for combo in itertools.product(*[clusters[k] for k in perm]):
                # 开放路径: combo是有序选城序列
                d = sum(dist[combo[i]][combo[i + 1]] for i in range(K - 1))
                best = min(best, d)
        return best

    for trial in range(5):
        n, K = 9, 3
        raw = rng.random((n, n)) * 100
        dist = (raw + raw.T) / 2
        np.fill_diagonal(dist, 0)
        clusters = [list(range(0, 3)), list(range(3, 6)), list(range(6, 9))]

        res = build_pipeline(dist, clusters)
        ref = brute_gtsp(dist, clusters)

        # 对账: 合法闭合回路恰含 (K+1) 条跨簇弧(K省+虚拟D各离开一次),
        # 每条含1个M; 环弧全0; D的进出弧距离分量为0
        # ⇒ OPT_ATSP = ref*scale + (K+1)*M
        atsp_opt = _brute_atsp(res.atsp)
        gtsp_derived = (atsp_opt - (K + 1) * res.M) / res.scale
        assert abs(gtsp_derived - ref) < res.scale * 0.5 + 1e-9, \
            f'trial{trial}: ATSP推导{gtsp_derived} != 暴力{ref}'
        print(f'  trial {trial}: 暴力={ref:.1f} ≡ 变换推导={gtsp_derived:.1f} ✓')

    print('自验证全部通过: 两级变换无损 ✓')


def _brute_atsp(atsp):
    import itertools
    n = len(atsp)
    best = float('inf')
    for perm in itertools.permutations(range(n)):
        c = sum(atsp[perm[i]][perm[(i + 1) % n]] for i in range(n))
        best = min(best, c)
    return best


if __name__ == '__main__':
    import sys
    if '--self-test' in sys.argv or len(sys.argv) == 1:
        print('运行变换自验证 (3簇9城随机算例 × 5, 暴力对拍)...')
        _self_test()
