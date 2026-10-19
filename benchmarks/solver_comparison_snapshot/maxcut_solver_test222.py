#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaxCut Solver Main Function
支持多种求解算法：ASB, BSB(GPU), DSB(GPU), NMFA, SimCIM, SA
所有算法参数集中配置，启用--getAllCut时指定汇总结果保存的txt文件路径
新增--checkpoint功能：在--getAllCut模式下记录不同迭代次数的结果
新增--parallel功能：在--getAllCut --checkpoint模式下支持并行运行
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool, cpu_count
from functools import partial

# 设置环境变量
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# 导入求解器
from ASB import ASB
from NMFA import NMFA
from SimCIM import SimCIM

# 尝试导入GPU版本的SB
try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        from tabu_SB_gpu import SB as SB_GPU
except ImportError:
    GPU_AVAILABLE = False
    torch = None

# 尝试导入SA (需要neal库)
try:
    from neal.simulated_annealing import simulated_annealing

    SA_AVAILABLE = True
except ImportError:
    SA_AVAILABLE = False
    print("Warning: neal library not found. SA solver will not be available.")

# ===================== 集中参数配置区域（所有算法参数都在此修改） =====================
ALL_ALGORITHMS_CONFIG = {
    'general': {
        'repeat_times': 100,
        'device': 'cpu'
    },
    'asb': {
        'n_iter': 10000,
        'batch_size': 1,
        'dt': 0.1,
        'M': 2
    },
    'bsb': {
        'n_iter': 10000,
        'batch_size': 1,
        'dt': 1.0
    },
    'dsb': {
        'n_iter': 10000,
        'batch_size': 1,
        'dt': 1.0
    },
    'nmfa': {
        'n_iter': 10000,
        'batch_size': 1,
        'alpha': 0.15,
        'sigma': 0.15
    },
    'simcim': {
        'n_iter': 10000,
        'batch_size': 1,
        'dt': 0.01,
        'momentum': 0.9,
        'sigma': 0.03,
        'pt': 6.5
    },
    'sa': {
        'n_sweeps': 10000,
        'num_samples': 1
    }
}


# ====================================================================================

def get_checkpoint_intervals(max_iter):
    """根据最大迭代次数确定checkpoint间隔，累积所有区间的checkpoint点"""
    checkpoints = []

    # 1-10: 每隔1次
    if max_iter >= 1:
        checkpoints.extend(range(1, min(max_iter, 10) + 1, 1))

    # 11-100: 每隔10次
    if max_iter > 10:
        checkpoints.extend(range(20, min(max_iter, 100) + 1, 10))

    # 101-1000: 每隔100次
    if max_iter > 100:
        checkpoints.extend(range(200, min(max_iter, 1000) + 1, 100))

    # 1001+: 每隔1000次
    if max_iter > 1000:
        checkpoints.extend(range(2000, max_iter + 1, 1000))

    # 确保max_iter在列表中
    if max_iter not in checkpoints:
        checkpoints.append(max_iter)

    # 去重并排序
    checkpoints = sorted(list(set(checkpoints)))

    return checkpoints


def set_random_seed():
    """生成并设置随机种子，确保每次运行结果不同"""
    seed = int((time.time() * 1000000 + os.getpid() + random.randint(0, 100000)) % (2 ** 31))

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return seed


def read_edge_list(filename):
    """读取边列表格式的文件"""
    print(f"Reading file: {filename}")
    df = pd.read_csv(filename, sep='\t', header=0, names=['Node1', 'Node2', 'Weight'])
    print(f"Total edges: {len(df)}")

    all_nodes = np.unique(np.concatenate([df['Node1'].values, df['Node2'].values]))
    n_nodes = len(all_nodes)
    print(f"Total nodes: {n_nodes}")

    node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}

    rows = df['Node1'].map(node_mapping).values
    cols = df['Node2'].map(node_mapping).values
    weights = df['Weight'].values

    J = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))
    J = J + J.T
    J = -J

    return J, node_mapping, reverse_mapping


def calc_cut_value(J, spins):
    """计算cut值"""
    if len(spins.shape) == 1:
        spins = spins.reshape(-1, 1)

    J_original = -J
    cut = 0.25 * (J_original.sum() - np.sum(J_original.dot(spins) * spins, axis=0))
    return cut


def solve_with_asb(J, params, device='cpu'):
    """使用ASB算法求解"""
    set_random_seed()

    backend = 'cpu-float32' if device == 'cpu' else 'gpu-float32'

    solver = ASB(J, n_iter=params['n_iter'], batch_size=params['batch_size'],
                 dt=params['dt'], M=params.get('M', 2), backend=backend)
    solver.update()

    if backend == 'gpu-float32':
        spins = torch.sign(solver.x).cpu().numpy()
    else:
        spins = np.sign(solver.x)

    spins[spins == 0] = 1
    cuts = calc_cut_value(J, spins)
    best_idx = np.argmax(cuts)
    best_cut = cuts[best_idx]

    return best_cut


def solve_with_bsb(J, params, device):
    """使用CPU/GPU版本的BSB算法求解"""
    set_random_seed()

    if not GPU_AVAILABLE and device == 'cuda':
        device = 'cpu'

    if torch is None:
        return None

    J_dense = J.toarray() if hasattr(J, 'toarray') else J
    J_tensor = torch.from_numpy(J_dense).float()
    xi = 1.0 / torch.abs(J_tensor.sum(axis=1)).max()
    J_tensor = J_tensor.to(device)
    J_sparse = J_tensor.to_sparse_csr()

    solver = SB_GPU(J_sparse, n_iter=params['n_iter'], xi=xi, dt=params['dt'],
                    batch_size=params['batch_size'], device=device)
    solver.update_b()

    spins = torch.sign(solver.x).cpu().numpy()
    spins[spins == 0] = 1
    cuts = calc_cut_value(J, spins)
    best_idx = np.argmax(cuts)
    best_cut = cuts[best_idx]

    return best_cut


def solve_with_dsb(J, params, device):
    """使用CPU/GPU版本的DSB算法求解"""
    set_random_seed()

    if not GPU_AVAILABLE and device == 'cuda':
        device = 'cpu'

    if torch is None:
        return None

    J_dense = J.toarray() if hasattr(J, 'toarray') else J
    J_tensor = torch.from_numpy(J_dense).float()
    xi = 1.0 / torch.abs(J_tensor.sum(axis=1)).max()
    J_tensor = J_tensor.to(device)
    J_sparse = J_tensor.to_sparse_csr()

    solver = SB_GPU(J_sparse, n_iter=params['n_iter'], xi=xi, dt=params['dt'],
                    batch_size=params['batch_size'], device=device)
    solver.update_d()

    spins = torch.sign(solver.x).cpu().numpy()
    spins[spins == 0] = 1
    cuts = calc_cut_value(J, spins)
    best_idx = np.argmax(cuts)
    best_cut = cuts[best_idx]

    return best_cut


def solve_with_nmfa(J, params, device='cpu'):
    """使用NMFA算法求解"""
    set_random_seed()

    backend = 'cpu-float32' if device == 'cpu' else 'gpu-float32'

    solver = NMFA(J, n_iter=params['n_iter'], batch_size=params['batch_size'],
                  alpha=params['alpha'], sigma=params['sigma'], backend=backend)
    solver.update()

    if backend == 'gpu-float32':
        spins = torch.sign(solver.x).cpu().numpy()
    else:
        spins = np.sign(solver.x)

    spins[spins == 0] = 1
    cuts = calc_cut_value(J, spins)
    best_idx = np.argmax(cuts)
    best_cut = cuts[best_idx]

    return best_cut


def solve_with_simcim(J, params, device='cpu'):
    """使用SimCIM算法求解"""
    set_random_seed()

    backend = 'cpu-float32' if device == 'cpu' else 'gpu-float32'

    solver = SimCIM(J, n_iter=params['n_iter'], batch_size=params['batch_size'],
                    dt=params['dt'], momentum=params['momentum'],
                    sigma=params['sigma'], pt=params['pt'], backend=backend)
    solver.update()

    if backend == 'gpu-float32':
        spins = torch.sign(solver.x).cpu().numpy()
    else:
        spins = np.sign(solver.x)

    spins[spins == 0] = 1
    cuts = calc_cut_value(J, spins)
    best_idx = np.argmax(cuts)
    best_cut = cuts[best_idx]

    return best_cut


def solve_with_sa(J, params):
    """使用SA算法求解"""
    seed = set_random_seed()

    if not SA_AVAILABLE:
        return None

    N = J.shape[0]
    h = [0.0] * N
    J_coo = coo_matrix(J)
    coupler_starts, coupler_ends, coupler_weights = [], [], []

    for i, j, w in zip(J_coo.row, J_coo.col, J_coo.data):
        if i < j:
            coupler_starts.append(i)
            coupler_ends.append(j)
            coupler_weights.append(-w)

    if len(coupler_starts) == 0:
        return None

    beta_schedule = np.linspace(0.01, 3, num=params['n_sweeps'])
    np_rand = np.random.RandomState(seed)
    initial_states = 2 * np_rand.randint(2, size=(params['num_samples'], N)).astype(np.int8) - 1

    samples, energies = simulated_annealing(
        params['num_samples'], h, coupler_starts, coupler_ends, coupler_weights,
        1, beta_schedule, seed, initial_states
    )

    cuts = []
    for i in range(params['num_samples']):
        spins = samples[i].reshape(-1, 1)
        cut = calc_cut_value(J, spins)[0]
        cuts.append(cut)

    cuts = np.array(cuts)
    best_idx = np.argmax(cuts)
    best_cut = cuts[best_idx]

    return best_cut


# ===================== 并行运行支持函数 =====================

def run_single_iteration(args_tuple):
    """
    单次迭代运行函数（用于并行）
    args_tuple: (run_idx, J, solver_name, solver_func, temp_params, actual_device, need_device)
    返回: (run_idx, cut, time_taken)
    """
    run_idx, J, solver_name, solver_func, temp_params, actual_device, need_device = args_tuple

    try:
        start_time = time.time()
        if need_device:
            cut = solver_func(J, temp_params, actual_device)
        else:
            cut = solver_func(J, temp_params)
        end_time = time.time()
        time_taken = end_time - start_time
        return (run_idx, cut, time_taken)
    except Exception as e:
        return (run_idx, None, None)


# ==============================================================

def save_results(filename, best_spins, reverse_mapping, best_cut):
    """保存结果到文件"""
    output_file = filename.replace('.txt', '_result.txt')

    with open(output_file, 'w') as f:
        f.write(f"# Best cut value: {best_cut:.4f}\n")
        f.write("# Node\tSpin\n")
        for idx, spin in enumerate(best_spins):
            node_id = reverse_mapping[idx]
            f.write(f"{node_id}\t{int(spin)}\n")

    print(f"\nResults saved to: {output_file}")


def run_all_solvers(J, save_path, enable_checkpoint=False, enable_parallel=False):
    """
    运行所有可用的求解器，使用集中配置的参数，并将汇总结果保存到指定路径
    enable_checkpoint: 是否启用checkpoint功能
    enable_parallel: 是否启用并行运行（仅在enable_checkpoint=True时生效）
    """
    repeat_times = ALL_ALGORITHMS_CONFIG['general']['repeat_times']
    device = ALL_ALGORITHMS_CONFIG['general']['device']
    actual_device = 'cuda' if device == 'cuda' and GPU_AVAILABLE else 'cpu'

    # 确定并行进程数
    if enable_parallel and enable_checkpoint:
        num_processes = min(cpu_count(), 10)
        print(f"并行模式已启用，将使用 {num_processes} 个进程")
    else:
        num_processes = 1

    solvers = [
        ('asb', solve_with_asb, True),
        ('bsb', solve_with_bsb, True),
        ('dsb', solve_with_dsb, True),
        ('nmfa', solve_with_nmfa, True),
        ('simcim', solve_with_simcim, True),
        ('sa', solve_with_sa, False)
    ]

    summary_results = {}

    print("\n" + "=" * 70)
    print(f"开始运行所有算法（集中配置参数）")
    print(f"通用配置：重复运行 {repeat_times} 次，设备 = {actual_device}")
    if enable_parallel and enable_checkpoint:
        print(f"并行运行：启用（进程数 = {num_processes}）")
    print(f"汇总结果将保存到：{save_path}")
    if enable_checkpoint:
        print(f"Checkpoint功能已启用")
    print("=" * 70)

    for solver_name, solver_func, need_device in solvers:
        if solver_name not in ALL_ALGORITHMS_CONFIG:
            print(f"\n{solver_name.upper()} 无配置参数，跳过")
            continue

        solver_params = ALL_ALGORITHMS_CONFIG[solver_name]

        # 获取该算法的最大迭代次数
        if solver_name == 'sa':
            max_iter = solver_params.get('n_sweeps', 10000)
        else:
            max_iter = solver_params.get('n_iter', 10000)

        print(f"\n{'=' * 20} 开始测试 {solver_name.upper()} 算法 {'=' * 20}")
        print(f"使用参数：{solver_params}")
        print(f"设备：{actual_device}")

        if solver_name == 'sa' and not SA_AVAILABLE:
            print(f"{solver_name.upper()} 算法不可用（缺少neal库），跳过")
            continue
        if (solver_name in ['bsb', 'dsb']) and torch is None:
            print(f"{solver_name.upper()} 算法不可用（缺少PyTorch），跳过")
            continue
        if (solver_name in ['asb', 'nmfa', 'simcim']) and actual_device == 'cuda' and not GPU_AVAILABLE:
            print(f"{solver_name.upper()} GPU模式不可用，将使用CPU")
            actual_device = 'cpu'

        # 如果启用checkpoint
        if enable_checkpoint:
            checkpoints = get_checkpoint_intervals(max_iter)
            print(f"Checkpoint迭代点: {checkpoints}")

            # 创建checkpoint文件并写入表头
            checkpoint_dir = os.path.dirname(save_path) or '.'
            checkpoint_file = os.path.join(checkpoint_dir, f"{solver_name}_checkpoint.txt")

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                f.write(f"{solver_name.upper()} Checkpoint 结果\n")
                f.write("=" * 120 + "\n")
                f.write(f"{'n_iter':<10} {'最大割值':<15} {'最小割值':<15} {'平均割值':<15} "
                        f"{'割值标准差':<15} {'最大时间(s)':<15} {'最小时间(s)':<15} {'平均时间(s)':<15}\n")
                f.write("-" * 120 + "\n")

            print(f"Checkpoint文件已创建: {checkpoint_file}")

            # 用于保存max_iter的结果
            final_avg_cut = None
            final_std_cut = None
            final_avg_time = None
            final_std_time = None

            for n_iter in checkpoints:
                print(f"\n  测试 n_iter={n_iter}...")

                # 修改参数中的n_iter
                temp_params = solver_params.copy()
                if solver_name == 'sa':
                    temp_params['n_sweeps'] = n_iter
                else:
                    temp_params['n_iter'] = n_iter

                # 并行或串行运行
                if enable_parallel:
                    # 并行运行
                    args_list = [
                        (i, J, solver_name, solver_func, temp_params, actual_device, need_device)
                        for i in range(repeat_times)
                    ]

                    with Pool(processes=num_processes) as pool:
                        results = pool.map(run_single_iteration, args_list)

                    cut_values = [r[1] for r in results if r[1] is not None]
                    time_values = [r[2] for r in results if r[2] is not None]
                else:
                    # 串行运行
                    cut_values = []
                    time_values = []

                    for run_idx in range(repeat_times):
                        try:
                            start_time = time.time()
                            if need_device:
                                cut = solver_func(J, temp_params, actual_device)
                            else:
                                cut = solver_func(J, temp_params)
                            end_time = time.time()
                            time_taken = end_time - start_time
                        except Exception as e:
                            if run_idx % 10 == 0:
                                print(f"    第 {run_idx + 1} 次运行出错：{e}，跳过该次运行")
                            continue

                        if cut is not None and time_taken is not None:
                            cut_values.append(cut)
                            time_values.append(time_taken)

                # 实时写入checkpoint结果到文件
                if cut_values and time_values:
                    max_cut = np.max(cut_values)
                    min_cut = np.min(cut_values)
                    avg_cut = np.mean(cut_values)
                    std_cut = np.std(cut_values)
                    max_time = np.max(time_values)
                    min_time = np.min(time_values)
                    avg_time = np.mean(time_values)

                    with open(checkpoint_file, 'a', encoding='utf-8') as f:
                        f.write(f"{n_iter:<10} {max_cut:<15.4f} {min_cut:<15.4f} "
                                f"{avg_cut:<15.4f} {std_cut:<15.4f} {max_time:<15.4f} {min_time:<15.4f} "
                                f"{avg_time:<15.4f}\n")

                    print(f"    n_iter={n_iter}: 平均割值={avg_cut:.4f} (±{std_cut:.4f}), "
                          f"平均时间={avg_time:.4f}s (已写入文件)")

                    # 如果这是max_iter，保存结果用于汇总
                    if n_iter == max_iter:
                        final_avg_cut = avg_cut
                        final_std_cut = std_cut
                        final_avg_time = avg_time
                        final_std_time = np.std(time_values)

                    # 清空列表释放内存
                    cut_values.clear()
                    time_values.clear()

            # 写入文件结束标记
            with open(checkpoint_file, 'a', encoding='utf-8') as f:
                f.write("-" * 120 + "\n")

            print(f"\n{solver_name.upper()} checkpoint结果已完成保存到: {checkpoint_file}")

            # 使用max_iter的结果作为汇总结果
            if final_avg_cut is not None:
                summary_results[solver_name] = {
                    'avg_cut': final_avg_cut,
                    'std_cut': final_std_cut,
                    'avg_time': final_avg_time,
                    'std_time': final_std_time,
                    'run_times': repeat_times
                }

                print(f"\n{solver_name.upper()} 汇总（使用checkpoint中n_iter={max_iter}的结果）：")
                print(f"  平均最大割值：{final_avg_cut:.4f} (±{final_std_cut:.4f})")
                print(f"  平均求解时间：{final_avg_time:.4f}s (±{final_std_time:.4f})")

        else:
            # 未启用checkpoint：常规模式
            cut_values = []
            time_values = []

            for run_idx in range(repeat_times):
                try:
                    start_time = time.time()
                    if need_device:
                        cut = solver_func(J, solver_params, actual_device)
                    else:
                        cut = solver_func(J, solver_params)
                    end_time = time.time()
                    time_taken = end_time - start_time
                except Exception as e:
                    if run_idx % 10 == 0:
                        print(f"第 {run_idx + 1} 次运行出错：{e}，跳过该次运行")
                    continue

                if cut is not None and time_taken is not None:
                    cut_values.append(cut)
                    time_values.append(time_taken)

            if cut_values and time_values:
                avg_cut = np.mean(cut_values)
                avg_time = np.mean(time_values)
                std_cut = np.std(cut_values)
                std_time = np.std(time_values)

                summary_results[solver_name] = {
                    'avg_cut': avg_cut,
                    'std_cut': std_cut,
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'run_times': len(cut_values)
                }

                print(f"\n{solver_name.upper()} 汇总（有效运行 {len(cut_values)} 次）：")
                print(f"  平均最大割值：{avg_cut:.4f} (±{std_cut:.4f})")
                print(f"  平均求解时间：{avg_time:.4f}s (±{std_time:.4f})")
            else:
                print(f"\n{solver_name.upper()} 无有效运行结果")

            cut_values.clear()
            time_values.clear()

    # 保存汇总结果
    print("\n" + "=" * 80)
    print("所有算法平均结果汇总（集中配置参数）")
    print("=" * 80)
    print(f"{'算法':<10} {'平均最大割值':<15} {'割值标准差':<15} {'平均求解时间(s)':<20} {'有效运行次数':<10}")
    print("-" * 80)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("MaxCut 所有算法求解结果汇总\n")
        f.write(f"通用配置：重复运行 {repeat_times} 次，设备 = {actual_device}\n")
        if enable_parallel and enable_checkpoint:
            f.write(f"并行运行：启用（进程数 = {num_processes}）\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'算法':<10} {'平均最大割值':<15} {'割值标准差':<15} {'平均求解时间(s)':<20} {'有效运行次数':<10}\n")
        f.write("-" * 80 + "\n")

        for solver_name, results in summary_results.items():
            print(f"{solver_name.upper():<10} {results['avg_cut']:<15.4f} {results['std_cut']:<15.4f} "
                  f"{results['avg_time']:<20.4f} {results['run_times']:<10}")
            f.write(f"{solver_name.upper():<10} {results['avg_cut']:<15.4f} {results['std_cut']:<15.4f} "
                    f"{results['avg_time']:<20.4f} {results['run_times']:<10}\n")

        f.write("-" * 80 + "\n")

    print("-" * 80)
    print(f"\n汇总结果已保存到文件：{save_path}")

    return summary_results


def solve_single_algorithm(J, solver_name, args):
    """运行单个指定算法（使用命令行参数）"""
    set_random_seed()

    spins = None
    cut = None
    time_taken = None

    default_params = ALL_ALGORITHMS_CONFIG.get(solver_name, {})
    default_dt = {
        'asb': 0.1,
        'bsb': 1.0,
        'dsb': 1.0,
        'simcim': 0.01
    }

    run_params = {
        'solver': solver_name.upper(),
        'n_iter': args.n_iter,
        'batch_size': args.batch_size,
        'dt': args.dt if args.dt is not None else default_dt.get(solver_name, default_params.get('dt', 1.0)),
        'device': args.device,
        'actual_device': 'cuda' if args.device == 'cuda' and GPU_AVAILABLE else 'cpu',
        'backend': 'cpu-float32' if args.device == 'cpu' else 'gpu-float32'
    }

    if solver_name == 'asb':
        run_params['M'] = default_params.get('M', 2)
    elif solver_name == 'nmfa':
        run_params['alpha'] = default_params.get('alpha', 0.15)
        run_params['sigma'] = default_params.get('sigma', 0.15)
    elif solver_name == 'simcim':
        run_params['momentum'] = default_params.get('momentum', 0.9)
        run_params['sigma'] = default_params.get('sigma', 0.03)
        run_params['pt'] = default_params.get('pt', 6.5)
    elif solver_name == 'sa':
        run_params['n_sweeps'] = args.n_iter
        run_params['num_samples'] = args.batch_size

    current_dt = run_params['dt']
    backend = run_params['backend']

    print("\n" + "=" * 50)
    print(f"Solving with {solver_name.upper()} (命令行参数模式)")
    print("=" * 50)

    print("运行参数：")
    for param_name, param_value in run_params.items():
        print(f"  {param_name}: {param_value}")
    print("-" * 50)

    start_time = time.time()
    if solver_name == 'asb':
        solver = ASB(J, n_iter=args.n_iter, batch_size=args.batch_size, dt=current_dt,
                     M=run_params['M'], backend=backend)
        solver.update()
        if backend == 'gpu-float32':
            spins = torch.sign(solver.x).cpu().numpy()
        else:
            spins = np.sign(solver.x)
        spins[spins == 0] = 1
        cuts = calc_cut_value(J, spins)
        best_idx = np.argmax(cuts)
        cut = cuts[best_idx]
        spins = spins[:, best_idx]
    elif solver_name == 'bsb':
        if torch is not None:
            J_dense = J.toarray()
            J_tensor = torch.from_numpy(J_dense).float()
            xi = 1.0 / torch.abs(J_tensor.sum(axis=1)).max()
            run_params['xi'] = xi
            print(f"  计算得到xi值: {xi:.6f}")
            J_tensor = J_tensor.to(args.device)
            J_sparse = J_tensor.to_sparse_csr()
            solver = SB_GPU(J_sparse, n_iter=args.n_iter, xi=xi, dt=current_dt,
                            batch_size=args.batch_size, device=args.device)
            solver.update_b()
            spins = torch.sign(solver.x).cpu().numpy()
            spins[spins == 0] = 1
            cuts = calc_cut_value(J, spins)
            best_idx = np.argmax(cuts)
            cut = cuts[best_idx]
            spins = spins[:, best_idx]
    elif solver_name == 'dsb':
        if torch is not None:
            J_dense = J.toarray()
            J_tensor = torch.from_numpy(J_dense).float()
            xi = 1.0 / torch.abs(J_tensor.sum(axis=1)).max()
            run_params['xi'] = xi
            print(f"  计算得到xi值: {xi:.6f}")
            J_tensor = J_tensor.to(args.device)
            J_sparse = J_tensor.to_sparse_csr()
            solver = SB_GPU(J_sparse, n_iter=args.n_iter, xi=xi, dt=current_dt,
                            batch_size=args.batch_size, device=args.device)
            solver.update_d()
            spins = torch.sign(solver.x).cpu().numpy()
            spins[spins == 0] = 1
            cuts = calc_cut_value(J, spins)
            best_idx = np.argmax(cuts)
            cut = cuts[best_idx]
            spins = spins[:, best_idx]
    elif solver_name == 'nmfa':
        solver = NMFA(J, n_iter=args.n_iter, batch_size=args.batch_size,
                      alpha=run_params['alpha'], sigma=run_params['sigma'], backend=backend)
        solver.update()
        if backend == 'gpu-float32':
            spins = torch.sign(solver.x).cpu().numpy()
        else:
            spins = np.sign(solver.x)
        spins[spins == 0] = 1
        cuts = calc_cut_value(J, spins)
        best_idx = np.argmax(cuts)
        cut = cuts[best_idx]
        spins = spins[:, best_idx]
    elif solver_name == 'simcim':
        solver = SimCIM(J, n_iter=args.n_iter, batch_size=args.batch_size, dt=current_dt,
                        momentum=run_params['momentum'], sigma=run_params['sigma'],
                        pt=run_params['pt'], backend=backend)
        solver.update()
        if backend == 'gpu-float32':
            spins = torch.sign(solver.x).cpu().numpy()
        else:
            spins = np.sign(solver.x)
        spins[spins == 0] = 1
        cuts = calc_cut_value(J, spins)
        best_idx = np.argmax(cuts)
        cut = cuts[best_idx]
        spins = spins[:, best_idx]
    elif solver_name == 'sa':
        if SA_AVAILABLE:
            N = J.shape[0]
            h = [0.0] * N
            J_coo = coo_matrix(J)
            coupler_starts, coupler_ends, coupler_weights = [], [], []
            for i, j, w in zip(J_coo.row, J_coo.col, J_coo.data):
                if i < j:
                    coupler_starts.append(i)
                    coupler_ends.append(j)
                    coupler_weights.append(-w)
            if len(coupler_starts) > 0:
                beta_schedule = np.linspace(0.01, 3, num=run_params['n_sweeps'])
                seed = set_random_seed()
                run_params['seed'] = seed
                print(f"  随机种子: {seed}")
                np_rand = np.random.RandomState(seed)
                initial_states = 2 * np_rand.randint(2, size=(run_params['num_samples'], N)).astype(np.int8) - 1
                samples, energies = simulated_annealing(
                    run_params['num_samples'], h, coupler_starts, coupler_ends, coupler_weights,
                    1, beta_schedule, seed, initial_states
                )
                cuts = []
                for i in range(run_params['num_samples']):
                    spin = samples[i].reshape(-1, 1)
                    cuts.append(calc_cut_value(J, spin)[0])
                cuts = np.array(cuts)
                best_idx = np.argmax(cuts)
                cut = cuts[best_idx]
                spins = samples[best_idx]
    end_time = time.time()
    time_taken = end_time - start_time

    return spins, cut, time_taken


def run_single_solver_with_checkpoint(J, solver_name, enable_parallel=False):
    """
    运行单个指定算法并生成checkpoint文件
    solver_name: 要运行的算法名称
    enable_parallel: 是否启用并行运行
    """
    repeat_times = ALL_ALGORITHMS_CONFIG['general']['repeat_times']
    device = ALL_ALGORITHMS_CONFIG['general']['device']
    actual_device = 'cuda' if device == 'cuda' and GPU_AVAILABLE else 'cpu'

    # 确定并行进程数
    if enable_parallel:
        num_processes = min(cpu_count(), 10)
        print(f"并行模式已启用，将使用 {num_processes} 个进程")
    else:
        num_processes = 1

    # 获取solver配置
    if solver_name not in ALL_ALGORITHMS_CONFIG:
        print(f"\n{solver_name.upper()} 无配置参数")
        return None

    solver_params = ALL_ALGORITHMS_CONFIG[solver_name]

    # 获取solver函数
    solvers_map = {
        'asb': (solve_with_asb, True),
        'bsb': (solve_with_bsb, True),
        'dsb': (solve_with_dsb, True),
        'nmfa': (solve_with_nmfa, True),
        'simcim': (solve_with_simcim, True),
        'sa': (solve_with_sa, False)
    }

    if solver_name not in solvers_map:
        print(f"未知的算法：{solver_name}")
        return None

    solver_func, need_device = solvers_map[solver_name]

    # 检查算法可用性
    if solver_name == 'sa' and not SA_AVAILABLE:
        print(f"{solver_name.upper()} 算法不可用（缺少neal库）")
        return None
    if (solver_name in ['bsb', 'dsb']) and torch is None:
        print(f"{solver_name.upper()} 算法不可用（缺少PyTorch）")
        return None
    if (solver_name in ['asb', 'nmfa', 'simcim']) and actual_device == 'cuda' and not GPU_AVAILABLE:
        print(f"{solver_name.upper()} GPU模式不可用，将使用CPU")
        actual_device = 'cpu'

    # 获取最大迭代次数
    if solver_name == 'sa':
        max_iter = solver_params.get('n_sweeps', 10000)
    else:
        max_iter = solver_params.get('n_iter', 10000)

    print("\n" + "=" * 70)
    print(f"开始运行 {solver_name.upper()} 算法（集中配置参数 + Checkpoint模式）")
    print(f"通用配置：重复运行 {repeat_times} 次，设备 = {actual_device}")
    if enable_parallel:
        print(f"并行运行：启用（进程数 = {num_processes}）")
    print(f"使用参数：{solver_params}")
    print("=" * 70)

    # 获取checkpoint间隔
    checkpoints = get_checkpoint_intervals(max_iter)
    print(f"Checkpoint迭代点: {checkpoints}")

    # 创建checkpoint文件并写入表头
    checkpoint_file = f"{solver_name}_checkpoint.txt"

    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        f.write(f"{solver_name.upper()} Checkpoint 结果\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'n_iter':<10} {'最大割值':<15} {'最小割值':<15} {'平均割值':<15} "
                f"{'割值标准差':<15} {'最大时间(s)':<15} {'最小时间(s)':<15} {'平均时间(s)':<15}\n")
        f.write("-" * 120 + "\n")

    print(f"Checkpoint文件已创建: {checkpoint_file}")

    # 遍历所有checkpoint点
    for n_iter in checkpoints:
        print(f"\n  测试 n_iter={n_iter}...")

        # 修改参数中的n_iter
        temp_params = solver_params.copy()
        if solver_name == 'sa':
            temp_params['n_sweeps'] = n_iter
        else:
            temp_params['n_iter'] = n_iter

        # 并行或串行运行
        if enable_parallel:
            # 并行运行
            args_list = [
                (i, J, solver_name, solver_func, temp_params, actual_device, need_device)
                for i in range(repeat_times)
            ]

            with Pool(processes=num_processes) as pool:
                results = pool.map(run_single_iteration, args_list)

            cut_values = [r[1] for r in results if r[1] is not None]
            time_values = [r[2] for r in results if r[2] is not None]
        else:
            # 串行运行
            cut_values = []
            time_values = []

            for run_idx in range(repeat_times):
                try:
                    start_time = time.time()
                    if need_device:
                        cut = solver_func(J, temp_params, actual_device)
                    else:
                        cut = solver_func(J, temp_params)
                    end_time = time.time()
                    time_taken = end_time - start_time
                except Exception as e:
                    if run_idx % 10 == 0:
                        print(f"    第 {run_idx + 1} 次运行出错：{e}，跳过该次运行")
                    continue

                if cut is not None and time_taken is not None:
                    cut_values.append(cut)
                    time_values.append(time_taken)

        # 实时写入checkpoint结果到文件
        if cut_values and time_values:
            max_cut = np.max(cut_values)
            min_cut = np.min(cut_values)
            avg_cut = np.mean(cut_values)
            std_cut = np.std(cut_values)
            max_time = np.max(time_values)
            min_time = np.min(time_values)
            avg_time = np.mean(time_values)

            with open(checkpoint_file, 'a', encoding='utf-8') as f:
                f.write(f"{n_iter:<10} {max_cut:<15.4f} {min_cut:<15.4f} "
                        f"{avg_cut:<15.4f} {std_cut:<15.4f} {max_time:<15.4f} {min_time:<15.4f} "
                        f"{avg_time:<15.4f}\n")

            print(f"    n_iter={n_iter}: 平均割值={avg_cut:.4f} (±{std_cut:.4f}), "
                  f"平均时间={avg_time:.4f}s (已写入文件)")

            # 清空列表释放内存
            cut_values.clear()
            time_values.clear()

    # 写入文件结束标记
    with open(checkpoint_file, 'a', encoding='utf-8') as f:
        f.write("-" * 120 + "\n")

    print(f"\n{solver_name.upper()} checkpoint结果已完成保存到: {checkpoint_file}")

    return checkpoint_file


def main():
    parser = argparse.ArgumentParser(description='MaxCut Solver')
    parser.add_argument('--file', type=str,
                        default=r'E:\my_work\phasing\phasing_v11_GPU\phasing_v21_final1\maxcut_solver\data_snps_block\connected_components\block2.txt',
                        help='Input file path')
    parser.add_argument('--solver', type=str, default=None,
                        choices=['asb', 'bsb', 'dsb', 'nmfa', 'simcim', 'sa'],
                        help='Solver to use')
    parser.add_argument('--n_iter', type=int, default=1000, help='迭代次数 (仅非getAllCut模式生效)')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小 (仅非getAllCut模式生效)')
    parser.add_argument('--dt', type=float, help='时间步长 (仅非getAllCut模式生效)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='计算设备 (cpu/cuda)')
    parser.add_argument('--save', action='store_true', help='保存结果到文件')
    parser.add_argument('--getAllCut', type=str, default=None,
                        help='运行所有算法（如指定--solver则仅运行该算法），并将汇总结果保存到指定的txt文件路径（如：./summary_results.txt）。如果同时指定--solver，则不生成汇总文件')
    parser.add_argument('--checkpoint', action='store_true',
                        help='启用checkpoint功能（仅在--getAllCut模式下生效），记录不同迭代次数的结果')
    parser.add_argument('--parallel', action='store_true',
                        help='启用并行运行（仅在--getAllCut --checkpoint模式下生效），加速100次重复运行')

    args = parser.parse_args()

    # 读取数据
    J, node_mapping, reverse_mapping = read_edge_list(args.file)
    print(f"\nProblem size: {J.shape[0]} nodes")
    print(f"Number of edges: {J.nnz // 2}")

    # 启用getAllCut模式
    if args.getAllCut is not None:
        if args.parallel and not args.checkpoint:
            print("Warning: --parallel 选项仅在同时启用 --checkpoint 时生效")

        # 检查是否指定了单个solver
        if args.solver is not None:  # 如果明确指定了非默认solver
            # 单个solver + checkpoint模式
            if args.checkpoint:
                print(
                    f"\n运行模式：单个算法({args.solver.upper()}) + Checkpoint + {'并行' if args.parallel else '串行'}")
                run_single_solver_with_checkpoint(J, args.solver, enable_parallel=args.parallel)
            else:
                print(f"Warning: 指定单个solver时建议同时启用 --checkpoint")
                run_all_solvers(J, args.getAllCut, enable_checkpoint=False, enable_parallel=False)
        else:
            # 运行所有算法模式
            print(
                f"\n运行模式：所有算法 + {'Checkpoint + ' if args.checkpoint else ''}{'并行' if args.parallel and args.checkpoint else '串行'}")
            run_all_solvers(J, args.getAllCut, enable_checkpoint=args.checkpoint,
                            enable_parallel=args.parallel)
        return

    # 未启用getAllCut：运行单个指定算法（原有逻辑）
    if args.checkpoint:
        print("Warning: --checkpoint 选项仅在 --getAllCut 模式下生效")
    if args.parallel:
        print("Warning: --parallel 选项仅在 --getAllCut --checkpoint 模式下生效")

    spins, cut, time_taken = solve_single_algorithm(J, args.solver, args)

    # 打印汇总结果
    print("\n" + "=" * 60)
    print("SUMMARY (命令行参数模式)")
    print("=" * 60)
    print(f"{'Solver':<15} {'Best Cut':<15} {'Time (s)':<15}")
    print("-" * 45)
    if cut is not None:
        print(f"{args.solver.upper():<15} {cut:<15.4f} {time_taken:<15.4f}")
    else:
        print(f"{args.solver.upper():<15} {'N/A':<15} {'N/A':<15}")
    print("-" * 45)

    # 保存结果（如果指定）
    if args.save and spins is not None and cut is not None:
        save_results(args.file, spins, reverse_mapping, cut)

    return {'spins': spins, 'cut': cut, 'time': time_taken}


if __name__ == '__main__':
    main()
