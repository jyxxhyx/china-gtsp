#!/usr/bin/env python3
"""
run_concorde.py — 中国34省GTSP: Concorde精确求解入口

链路: cities.csv → Noon-Bean(+虚拟D) → 三节点展开 → concorde → 反变换 → JSON

用法:
  python3 run_concorde.py                       # 默认跑法
  python3 run_concorde.py -u 17147360           # 提供STSP空间上界加速剪枝
  python3 run_concorde.py --concorde ~/bin/concorde
"""
import argparse, json, math, os, shutil, subprocess, sys, tempfile, time
import numpy as np
import pandas as pd

from gtsp_transform import build_pipeline, extract_gtsp, gtsp_open_distance

PROV_MAP = {'11':'北京','12':'天津','13':'河北','14':'山西','15':'内蒙古','21':'辽宁','22':'吉林','23':'黑龙江',
            '31':'上海','32':'江苏','33':'浙江','34':'安徽','35':'福建','36':'江西','37':'山东','41':'河南',
            '42':'湖北','43':'湖南','44':'广东','45':'广西','46':'海南','50':'重庆','51':'四川','52':'贵州',
            '53':'云南','54':'西藏','61':'陕西','62':'甘肃','63':'青海','64':'宁夏','65':'新疆','71':'台湾',
            '81':'香港','82':'澳门'}


def load(csv_path):
    df = pd.read_csv(csv_path)
    df['prov'] = df['code'].astype(str).str[:2].map(PROV_MAP)
    cd = {}
    for idx, row in df.iterrows():
        if isinstance(row['prov'], str):
            cd.setdefault(row['prov'], []).append(idx)
    provs = sorted(cd)
    clusters = [cd[p] for p in provs]
    n = len(df)
    dist = np.zeros((n, n))
    lons, lats = df['lng'].values, df['lat'].values
    for i in range(n):
        for j in range(i + 1, n):
            lo1, la1, lo2, la2 = map(math.radians, [lons[i], lats[i], lons[j], lats[j]])
            a = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
            dist[i][j] = dist[j][i] = 6371 * 2 * math.asin(math.sqrt(a))
    return df, clusters, provs, dist


def write_tsplib(path, matrix, name='china_gtsp_stsp'):
    N = len(matrix)
    with open(path, 'w') as f:
        f.write(f'NAME: {name}\nTYPE: TSP\nDIMENSION: {N}\n')
        f.write('COMMENT: 3-node expansion of Noon-Bean ATSP with dummy node (open GTSP)\n')
        f.write('EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n')
        for i in range(N):
            f.write(' '.join(str(int(v)) for v in matrix[i]) + '\n')
        f.write('EOF\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--concorde', default=shutil.which('concorde') or os.path.expanduser('~/bin/concorde'))
    ap.add_argument('-u', '--upper-bound', type=int, default=None)
    ap.add_argument('--out', default='output/concorde_result.json')
    args = ap.parse_args()

    t0 = time.time()
    root = os.path.dirname(os.path.abspath(__file__))
    df, clusters, provs, dist = load(os.path.join(root, 'data', 'cities.csv'))
    print(f'[1/5] {len(df)}城 {len(clusters)}省簇')

    res = build_pipeline(dist, clusters)
    print(f'[2/5] Noon-Bean ATSP: {res.n_atsp}x{res.n_atsp}, M={res.M}')
    print(f'[3/5] 三节点STSP: {res.n_stsp}x{res.n_stsp}, BIG={res.BIG}')

    print('[4/5] Concorde branch-and-cut ...')
    with tempfile.TemporaryDirectory() as td:
        tsp_f, sol_f, log_f = (os.path.join(td, x) for x in ('i.tsp', 'i.sol', 'c.log'))
        write_tsplib(tsp_f, res.stsp)
        cmd = [args.concorde]
        if args.upper_bound:
            cmd += ['-u', str(args.upper_bound)]
        cmd += ['-o', sol_f, tsp_f]
        print(f'      $ {" ".join(cmd)}')
        with open(log_f, 'w') as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=td)
        if not os.path.exists(sol_f):
            print(f'未产生解, 日志尾部:')
            print(subprocess.run(['tail', '-15', log_f], capture_output=True, text=True).stdout)
            sys.exit(1)
        toks = open(sol_f).read().split()
        tour = [int(t) for t in toks[1:int(toks[0]) + 1]]
        shutil.copy(log_f, os.path.join(root, 'output', 'concorde_run.log'))

    print('[5/5] 反变换...')
    path = extract_gtsp(tour, res)
    if path is None:
        print('✗ 回路结构非法')
        sys.exit(1)
    d = gtsp_open_distance(path, dist)
    stsp_cost = sum(res.stsp[tour[k]][tour[(k+1) % len(tour)]] for k in range(len(tour)))
    derived = (stsp_cost - (res.K + 1) * res.M) / res.scale
    cities = [df.iloc[c]['name'] for c in path]

    print(f'\n════════ 结果 ════════')
    print(f'开放路径: {d:.1f} km  (STSP对账: {derived:.1f})')
    print(f'路线: {" → ".join(cities[:6])} → ... → {cities[-1]}')

    os.makedirs(os.path.join(root, 'output'), exist_ok=True)
    out = os.path.join(root, args.out) if not os.path.isabs(args.out) else args.out
    json.dump({'distance_open_km': round(d, 1), 'stsp_cost': int(stsp_cost),
               'derived_km': round(derived, 1), 'path': path, 'cities': cities,
               'provinces': provs, 'stats': {'n': res.n, 'K': res.K,
               'n_atsp': res.n_atsp, 'n_stsp': res.n_stsp, 'M': res.M, 'BIG': res.BIG,
               'elapsed_s': round(time.time()-t0, 1)}},
              open(out, 'w'), ensure_ascii=False, indent=2)
    print(f'已保存: {out}')


if __name__ == '__main__':
    main()
