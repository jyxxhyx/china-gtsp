# China GTSP: 走遍全国34个省级行政区的最短开放路径

> 每个省级行政区选 1 个城市, 求一条**不回到起点**的最短路径 (open GTSP)。
> **当前最好解: 9,665 km** (哈尔滨 → 哈密), 下界证明 ≥ 9,44x km (Concorde branch-and-cut, 进行中)。

344 个候选城市 (地级市/直辖市/特别行政区), Haversine 大圆距离。

## 问题

Generalized Traveling Salesman Problem (GTSP):

- 344 个城市划分为 34 个簇 (省份);
- 每簇恰好访问 1 个城市;
- 最小化总距离;
- **开放路径** (不需要回到起点)。

## 方法链路

```
GTSP (344城/34簇)
  │  ① Noon-Bean 变换 (1993): 簇内零费环 + 后继记账 + M惩罚
  │     └─ 虚拟节点 D: 到所有城市距离 0, 单独成簇 → 开放路径
  ▼
ATSP (345×345, 不对称)
  │  ② Jonker-Volgenant 三节点展开 (1983): 每城拆 in/mid/out
  ▼
STSP (1035×1035, 对称)
  │  ③ Concorde branch-and-cut (精确求解)
  ▼
最优哈密尔顿回路 → 反变换 → GTSP 开放路径
```

两条变换均**无损**: STSP 最优值经对账公式 `(OPT − 35·M)/10` 精确还原 GTSP 距离 (随机小算例暴力对拍验证, 见 `gtsp_transform.py --self-test`)。

同时提供 **Schmidt-Irnich ILS** (2022) 的 Python 复现作为启发式上界:

- 7 邻域随机 VND: 2-opt / 3-opt / Double-Bridge / Relocation+ / Swap+ / CO / Gutin / SR+;
- record-to-record 接受准则 (ε=3%, 几何冷却);
- 300 秒内从 LKH 解 9,700 → **9,665 km**。

## 仓库结构

```
├── data/
│   └── cities.csv              # 344城 (code,name,lng,lat)
├── gtsp_transform.py           # 两级变换 + 自验证 (核心)
├── schmidt_ils.py              # Schmidt-Irnich ILS 复现 (启发式)
├── run_concorde.py             # Concorde 精确求解入口
├── visualize_route.py          # 合规中国地图路线可视化 (直线)
├── visualize_route_roadnet.py  # 路网版 (OSRM polyline)
├── fetch_road_matrix.py        # OSRM table 抓取对称化路网矩阵
├── fetch_road_matrix_asym.py   # OSRM table 抓取原始非对称矩阵
├── schmidt_ils_asym.py         # ILS 非对称版 (2-opt禁用, 其余邻域天然支持)
├── output/
│   ├── schmidt_ils_result.json # 当前最好解 9,665 km
│   └── route.png               # 路线图 (示例输出)
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 变换自验证 (3簇9城随机算例 × 5, 暴力对拍)
python3 gtsp_transform.py

# 2. 启发式求解 (5分钟)
python3 schmidt_ils.py 300

# 3. 精确求解 (需要 concorde, 见下节安装)
python3 run_concorde.py

# 4. 画路线图 (直线版 / 真实路网版)
python3 visualize_route.py            # 直线连接
python3 visualize_route_roadnet.py    # OSRM 实际路网 polyline

# 5. 路网版GTSP: 抓取344城路网距离矩阵 (OSRM table, ~1分钟)
python3 fetch_road_matrix.py
# 然后用schmidt_ils.py加载 road_matrix_344.npy 重解 (见README路网版说明)
```

路网版依赖 [OSRM](http://router.project-osrm.org) 公共服务返回实际行车路线
(`output/road_segments.json` 已含缓存的 33 段 polyline, 无需重新请求)。
路网总里程 12,753 km, 直线比 1.32。

## Concorde 安装

macOS (Apple Silicon 需 Rosetta) / Linux:

```bash
# 下载源码
wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
tar xzf co031219.tgz && cd concorde

# LP 后端: QSopt (免费)
#   Linux:   CFLAGS="-O2 -fcommon" ./configure --with-qsopt=/path/to/qsopt.a
#   macOS:   qsopt.a 只有 x86_64 版, Apple Silicon 用 Rosetta 编译:
curl -sL "https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt.a" -o qsopt.a
curl -sL "https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt.h" -o qsopt.h
CFLAGS="-O2 -fcommon -arch x86_64" ./configure --with-qsopt=$PWD/qsopt.a
make -j4
# 产物: TSP/concorde
```

注意: 1035 节点带 BIG 权重的实例, Concorde 收敛较慢 (数十小时级);
可先用 ILS 解换算 STSP 空间上界 `-u` 加速剪枝。

## 结果

## 两个版本的距离

| 版本 | 距离矩阵 | 最优解 | 说明 |
|---|---|---|---|
| 直线版 | Haversine 大圆 (对称) | **9,665 km** | ILS; Concorde 下界 ≥9,44x km (进行中) |
| 路网版 (对称化) | OSRM 双向均值 | 12,319 km | 直线解在路网上走需 12,753 km |
| **路网版 (非对称)** | OSRM 原始有向 D[i][j] | **12,300 km** | 路网本是有向图: 单行道/高速出入口使 A→B≠B→A |

非对称版是最严格的口径: 距离矩阵直接采用行进方向的 OSRM 值
(平均双向差 4.6 km, 最大 190 km), 解的里程与实际 polyline **严格相等**
(12,300.7 km 分毫不差)。方向效应本身参与优化——同一城市序列反向走
里程不同, 求解器选择了更优的行进方向。

路网版要点: 用 OSRM table API 抓取 344 城两两路网距离 (58,996 对),
在此矩阵上重解 GTSP —— 最优路线明显改变: 西段从"横断山直线"改为
兰州-西宁-哈密-那曲的真实公路走廊 (G109/G315)。

最优路线 (哈尔滨 → 哈密, 9,665 km):

哈尔滨 → 松原 → 通辽 → 朝阳 → 北京 → 廊坊 → 天津 → 滨州 → 盐城 → 上海 →
嘉兴 → 黄山 → 上饶 → 福州 → 台中 → 汕尾 → 香港 → 澳门 → 海口 → 北海 →
六盘水 → 昭通 → 泸州 → 重庆 → 张家界 → 十堰 → 三门峡 → 运城 → 铜川 →
固原 → 定西 → 昌都 → 玉树 → 哈密

## 引用

变换方法:

- Noon, C.E. & Bean, J.C. (1993). An efficient transformation of the generalized traveling salesman problem. *INFOR* 31(1):39-44.
- Jonker, R. & Volgenant, T. (1983). Transforming asymmetric into symmetric traveling salesman problems. *Operations Research Letters* 2(4):161-163.

求解算法:

- Schmidt, J. & Irnich, S. (2022). New neighborhoods and an iterated local search algorithm for the generalized traveling salesman problem. *EURO J. Comput. Optim.* 10:100029.
- Applegate, D., Bixby, R., Chvátal, V. & Cook, W. (2006). *The Traveling Salesman Problem: A Computational Study*. Princeton UP. (Concorde)

## 声明

- 距离为城市间 Haversine 大圆距离, 非实际路网;
- 地图使用 [cnmaps](https://github.com/Clarmy/cnmaps) 边界数据 (审图号 GS(2024)0650 号), 含九段线/南海诸岛;
- 仅供研究学习。
