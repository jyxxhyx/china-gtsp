# China GTSP & CETSP：走遍全国 34 省的两级挑战

同一张中国地图上的两个递进问题，一个代码仓库：

| 问题 | 一句话 | 最优结果 |
|---|---|---|
| **GTSP 路网版**（`src/gtsp/`） | 每省选 1 城市，真实公路最短开放路径 | **12,552 km**（哈尔滨→那曲，Concorde 下界 ≥12,267，gap ≈ 2%） |
| **CETSP 省界版**（`src/cetsp/） | 每省只需"擦到"边界，最短环游 | **7,575.22 km**（严格 34/34，逐点距省界 ≤1cm） |

回答文本（知乎）：
- GTSP 主篇：`../gtsp-materials/zhihu-gtsp-answer.md`
- CETSP 续篇：`../gtsp-materials/cetsp-知乎续篇.md`

## GTSP 路网版

344 候选城市分 34 簇（省），每簇选 1，真实路网距离（OSRM），开放路径。

```
GTSP ──Noon-Bean变换──→ ATSP ──三节点展开──→ STSP ──→ Concorde (精确)
        (簇内零费环+M惩罚)      (in/mid/out)         + Schmidt ILS (启发式)
```

主要产出（`output/gtsp/`）：
- `schmidt_ils_result.json` — ILS 启发式解
- `road_gtsp_fixed_result.json` + `route_roadnet.png` — 路网版最终路线
- `road_matrix_344.npy` — 344×344 路网距离矩阵（OSRM）

## CETSP 省界版

每省一个槽位，候选 = 省边界任意点，相邻槽可共享停靠点（三界点一点三省）。
cnmaps 边界（WGS84，GS(2024)0650），DP + 连续精化 + 米级局部最优验证。

主要产出（`output/cetsp/`）：
- `cetsp_refined_result.json` — 最终解（7,575.22 km，含逐点坐标）
- `cetsp_route_refined_7575.png` — 路线图
- `cetsp_compare_7578_vs_7575.png` — 与基线对比（差异段放大）

## 仓库结构

```
src/
  common/    共享组件: gtsp_core.py (Haversine 距离工具)
  gtsp/      路网版: 距离抓取(OSRM) / 变换 / Concorde / Schmidt ILS / 可视化
  cetsp/     省界版: 34槽位DP / 连续精化 / 局部最优验证 / 可视化
data/
  gtsp/      cities.csv (344城坐标)
  cetsp/     (边界数据运行时从 cnmaps 加载)
output/
  gtsp/      路网版结果
  cetsp/     省界版结果
```

## 复现

```bash
pip install -r requirements.txt   # numpy/pandas/matplotlib/cnmaps/cartopy

# GTSP (路网矩阵已缓存, 跳过抓取)
python3 src/gtsp/schmidt_ils.py 300        # ILS 启发式 (5min)
python3 src/gtsp/visualize_roadnet_fixed.py

# CETSP
python3 src/cetsp/cetsp_cnmaps.py          # 34槽位 DP (约10min)
python3 src/cetsp/cetsp_refine_heur.py     # 连续精化 → 7,575.22 (约30min)
python3 src/cetsp/cetsp_local_verify.py    # 局部最优验证
python3 src/cetsp/visualize_cetsp.py output/cetsp/cetsp_refined_result.json
```

## 引用

- Noon & Bean (1993). GTSP→ATSP transformation. *INFOR* 31(1).
- Jonker & Volgenant (1983). ATSP→STSP transformation. *OR Letters* 2(4).
- Schmidt & Irnich (2022). ILS for GTSP. *EURO J. Comput. Optim.* 10:100029.
- Applegate et al. (2006). *The TSP: A Computational Study*. (Concorde)

## 声明

- GTSP 距离为 OSRM 真实路网；台湾段为渡轮航线口径
- CETSP 坐标体系 WGS84；边界 cnmaps（GS(2024)0650 号审图，含九段线/南海诸岛）
- 仅供研究学习
