#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaxCut Solver - 精确时间追踪版本
只记录核心算法（update_b/update_d/simulated_annealing）的执行时间
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

# 设置环境变量
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# 尝试导入GPU版本的SB
try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        from tabu_SB_gpu import SB as SB_GPU
except ImportError:
    GPU_AVAILABLE = False
    torch = None

# 尝试导入SA
try:
    from neal.simulated_annealing import simulated_annealing

    SA_AVAILABLE = True
except ImportError:
    SA_AVAILABLE = False
    print("Warning: neal library not found. SA solver will not be available.")

# ===================== 算法配置 =====================
ALGORITHMS_CONFIG = {
    'general': {
        'repeat_times': 1,
    },
    'bsb_cpu': {
        'n_iter': 5000,
        'batch_size': 200,
        'dt': 1.0,
        'device': 'cpu'
    },
    'dsb_cpu': {
        'n_iter': 5000,
        'batch_size': 200,
        'dt': 1.0,
        'device': 'cpu'
    },
    'bsb_gpu': {
        'n_iter': 5000,
        'batch_size': 200,
        'dt': 1.0,
        'device': 'cuda'
    },
    'dsb_gpu': {
        'n_iter': 5000,
        'batch_size': 200,
        'dt': 1.0,
        'device': 'cuda'
    },
    'sa': {
        'n_sweeps': 5000,
        'num_samples': 200
    }
}


def get_checkpoint_intervals(max_iter):
    """根据最大迭代次数确定checkpoint间隔"""
    checkpoints = []

    # # 1-10: 每隔1次
    # if max_iter >= 1:
    #     checkpoints.extend(range(1, min(max_iter, 10) + 1, 1))

    # 1-10: 仅在1、5、10处生成（仅保留不超过max_iter的点）
    if max_iter >= 1:
        points_1_10 = [1, 5, 10]
        valid_points = [p for p in points_1_10 if p <= max_iter]
        checkpoints.extend(valid_points)

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
    """生成并设置随机种子"""
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
    J = -J  # 转换为能量最小化问题

    return J, node_mapping, reverse_mapping


def calc_energy(J, spins):
    """计算能量值（Ising模型）"""
    if len(spins.shape) == 1:
        spins = spins.reshape(-1, 1)

    # 能量 E = -sum(J_ij * s_i * s_j) / 2
    # 由于J已经对称，我们计算 -0.5 * sum(J * s * s^T)
    energy = -0.5 * np.sum(J.dot(spins) * spins, axis=0)
    return energy


def calc_cut_value(J, spins):
    """计算cut值（用于验证）"""
    if len(spins.shape) == 1:
        spins = spins.reshape(-1, 1)

    J_original = -J
    cut = 0.25 * (J_original.sum() - np.sum(J_original.dot(spins) * spins, axis=0))
    return cut


def solve_with_sb(J, params, algo_type='bsb', device='cpu'):
    """
    使用SB算法求解（BSB或DSB）
    返回: (best_energy, energies, best_cut, cuts, algo_time)
    algo_time 只包含 update_b() 或 update_d() 的时间
    """
    set_random_seed()

    if torch is None:
        return None, None, None, None, None

    # 准备数据（不计时）
    J_dense = J.toarray() if hasattr(J, 'toarray') else J
    J_tensor = torch.from_numpy(J_dense).float()
    xi = 1.0 / torch.abs(J_tensor.sum(axis=1)).max()
    J_tensor = J_tensor.to(device)
    J_sparse = J_tensor.to_sparse_csr()

    solver = SB_GPU(J_sparse, n_iter=params['n_iter'], xi=xi, dt=params['dt'],
                    batch_size=params['batch_size'], device=device)

    # ========== 只计时核心算法 ==========
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()  # 确保GPU操作完成

    start_time = time.time()

    if algo_type == 'bsb':
        solver.update_b()
    else:  # dsb
        solver.update_d()

    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()  # 确保GPU操作完成

    algo_time = time.time() - start_time
    # ========== 计时结束 ==========

    # 后处理（不计时）
    spins = torch.sign(solver.x).cpu().numpy()
    spins[spins == 0] = 1

    # 计算能量和割值
    energies = calc_energy(J, spins)
    cuts = calc_cut_value(J, spins)

    best_energy_idx = np.argmin(energies)
    best_cut_idx = np.argmax(cuts)

    best_energy = energies[best_energy_idx]
    best_cut = cuts[best_cut_idx]

    return best_energy, energies, best_cut, cuts, algo_time


def solve_with_sa(J, params):
    """
    使用SA算法求解
    返回: (best_energy, energies, best_cut, cuts, algo_time)
    algo_time 只包含 simulated_annealing() 的时间
    """
    seed = set_random_seed()

    if not SA_AVAILABLE:
        return None, None, None, None, None

    # 准备数据（不计时）
    N = J.shape[0]
    h = [0.0] * N
    J_coo = coo_matrix(J)
    coupler_starts, coupler_ends, coupler_weights = [], [], []

    for i, j, w in zip(J_coo.row, J_coo.col, J_coo.data):
        if i < j:
            coupler_starts.append(i)
            coupler_ends.append(j)
            coupler_weights.append(-w)  # 注意这里的符号

    if len(coupler_starts) == 0:
        return None, None, None, None, None

    beta_schedule = np.linspace(0.01, 3, num=params['n_sweeps'])
    np_rand = np.random.RandomState(seed)
    initial_states = 2 * np_rand.randint(2, size=(params['num_samples'], N)).astype(np.int8) - 1

    # ========== 只计时核心算法 ==========
    start_time = time.time()

    samples, energies = simulated_annealing(
        params['num_samples'], h, coupler_starts, coupler_ends, coupler_weights,
        1, beta_schedule, seed, initial_states
    )

    algo_time = time.time() - start_time
    # ========== 计时结束 ==========

    # 后处理（不计时）
    actual_energies = []
    actual_cuts = []
    for i in range(params['num_samples']):
        spins = samples[i].reshape(-1, 1)
        energy = calc_energy(J, spins)[0]
        cut = calc_cut_value(J, spins)[0]
        actual_energies.append(energy)
        actual_cuts.append(cut)

    actual_energies = np.array(actual_energies)
    actual_cuts = np.array(actual_cuts)

    best_energy_idx = np.argmin(actual_energies)
    best_cut_idx = np.argmax(actual_cuts)

    best_energy = actual_energies[best_energy_idx]
    best_cut = actual_cuts[best_cut_idx]

    return best_energy, actual_energies, best_cut, actual_cuts, algo_time


def run_single_iteration_for_energy(args_tuple):
    """单次迭代运行函数（用于并行）"""
    run_idx, J, solver_name, params = args_tuple

    try:
        if solver_name == 'sa':
            best_energy, _, best_cut, _, algo_time = solve_with_sa(J, params)
        elif solver_name in ['bsb_cpu', 'dsb_cpu']:
            algo_type = 'bsb' if 'bsb' in solver_name else 'dsb'
            best_energy, _, best_cut, _, algo_time = solve_with_sb(J, params, algo_type=algo_type, device='cpu')
        elif solver_name in ['bsb_gpu', 'dsb_gpu']:
            algo_type = 'bsb' if 'bsb' in solver_name else 'dsb'
            best_energy, _, best_cut, _, algo_time = solve_with_sb(J, params, algo_type=algo_type, device='cuda')
        else:
            return (run_idx, None, None, None)

        return (run_idx, best_energy, best_cut, algo_time)
    except Exception as e:
        print(f"Run {run_idx} error: {e}")
        return (run_idx, None, None, None)


def run_energy_tracking_experiments(J, output_dir='.', enable_parallel=False):
    """
    运行能量追踪实验
    记录累积时间和能量统计
    """
    repeat_times = ALGORITHMS_CONFIG['general']['repeat_times']

    # 确定并行进程数
    if enable_parallel:
        num_processes = min(cpu_count(), 10)
        print(f"并行模式已启用，将使用 {num_processes} 个进程")
    else:
        num_processes = 1

    # 定义要运行的算法（按指定顺序）
    algorithms = []

    # 1. BSB CPU
    if torch is not None:
        algorithms.append(('bsb_cpu', 'BSB (CPU)'))

    # 2. BSB GPU
    if torch is not None and GPU_AVAILABLE:
        algorithms.append(('bsb_gpu', 'BSB (GPU)'))

    # 3. DSB CPU
    if torch is not None:
        algorithms.append(('dsb_cpu', 'DSB (CPU)'))

    # 4. DSB GPU
    if torch is not None and GPU_AVAILABLE:
        algorithms.append(('dsb_gpu', 'DSB (GPU)'))

    # 5. SA
    if SA_AVAILABLE:
        algorithms.append(('sa', 'SA'))

    print("\n" + "=" * 80)
    print(f"开始能量追踪实验（精确计时版本）")
    print(f"重复运行 {repeat_times} 次")
    print(f"可用算法: {[name for _, name in algorithms]}")
    print(f"注意：只记录核心算法的执行时间")
    if enable_parallel:
        print(f"并行运行：启用（进程数 = {num_processes}）")
    print("=" * 80)

    # 为每个算法创建输出文件
    for solver_name, display_name in algorithms:
        print(f"\n{'=' * 60}")
        print(f"运行算法: {display_name}")
        print(f"{'=' * 60}")

        params = ALGORITHMS_CONFIG[solver_name]

        # 获取最大迭代次数
        if solver_name == 'sa':
            max_iter = params['n_sweeps']
        else:
            max_iter = params['n_iter']

        checkpoints = get_checkpoint_intervals(max_iter)
        print(f"Checkpoint迭代点: {checkpoints}")
        print(f"配置参数: {params}")

        # 创建输出文件
        output_file = os.path.join(output_dir, f"{solver_name}_energy_tracking.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{display_name} 能量与割值追踪结果（精确计时）\n")
            f.write("=" * 200 + "\n")
            f.write(f"{'n_iter':<10} {'累积时间(s)':<15} "
                    f"{'最大割值':<15} {'最小割值':<15} {'平均割值':<15} {'割值标准差':<15} "
                    f"{'最低能量':<15} {'最高能量':<15} {'平均能量':<15} {'能量标准差':<15} "
                    f"{'最大时间(s)':<15} {'最小时间(s)':<15} {'平均时间(s)':<15}\n")
            f.write("-" * 200 + "\n")

        print(f"输出文件已创建: {output_file}")

        # 累积时间记录
        cumulative_time = 0.0

        # 遍历所有checkpoint点
        for n_iter in checkpoints:
            print(f"\n  测试 n_iter={n_iter}...")

            # 修改参数
            temp_params = params.copy()
            if solver_name == 'sa':
                temp_params['n_sweeps'] = n_iter
            else:
                temp_params['n_iter'] = n_iter

            # 运行实验
            if enable_parallel:
                # 并行运行
                args_list = [
                    (i, J, solver_name, temp_params)
                    for i in range(repeat_times)
                ]

                with Pool(processes=num_processes) as pool:
                    results = pool.map(run_single_iteration_for_energy, args_list)

                energy_values = [r[1] for r in results if r[1] is not None]
                cut_values = [r[2] for r in results if r[2] is not None]
                time_values = [r[3] for r in results if r[3] is not None]
            else:
                # 串行运行
                energy_values = []
                cut_values = []
                time_values = []

                for run_idx in range(repeat_times):
                    result = run_single_iteration_for_energy((run_idx, J, solver_name, temp_params))
                    if result[1] is not None and result[2] is not None and result[3] is not None:
                        energy_values.append(result[1])
                        cut_values.append(result[2])
                        time_values.append(result[3])

            # 更新累积时间
            if time_values:
                cumulative_time += np.sum(time_values)

            # 计算统计量
            if energy_values and cut_values and time_values:
                # 割值统计
                max_cut = np.max(cut_values)
                min_cut = np.min(cut_values)
                avg_cut = np.mean(cut_values)
                std_cut = np.std(cut_values)

                # 能量统计
                min_energy = np.min(energy_values)
                max_energy = np.max(energy_values)
                avg_energy = np.mean(energy_values)
                std_energy = np.std(energy_values)

                # 时间统计
                max_time = np.max(time_values)
                min_time = np.min(time_values)
                avg_time = np.mean(time_values)

                # 写入文件
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"{n_iter:<10} {cumulative_time:<15.4f} "
                            f"{max_cut:<15.4f} {min_cut:<15.4f} {avg_cut:<15.4f} {std_cut:<15.4f} "
                            f"{min_energy:<15.4f} {max_energy:<15.4f} {avg_energy:<15.4f} {std_energy:<15.4f} "
                            f"{max_time:<15.4f} {min_time:<15.4f} {avg_time:<15.4f}\n")

                print(f"    n_iter={n_iter}: 平均割值={avg_cut:.4f} (±{std_cut:.4f}), "
                      f"平均能量={avg_energy:.4f} (±{std_energy:.4f}), 累积时间={cumulative_time:.4f}s")

            # 清空列表释放内存
            energy_values.clear()
            cut_values.clear()
            time_values.clear()

        # 写入文件结束标记
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("-" * 200 + "\n")
            f.write(f"总累积时间（仅核心算法）: {cumulative_time:.4f}s\n")

        print(f"\n{display_name} 完成，结果已保存到: {output_file}")

    print("\n" + "=" * 80)
    print("所有实验完成！")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='MaxCut Solver - 精确时间追踪版本')
    parser.add_argument('--file', type=str, required=True,
                        help='输入文件路径')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='输出目录（默认为当前目录）')
    parser.add_argument('--parallel', action='store_true',
                        help='启用并行运行')

    args = parser.parse_args()

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"创建输出目录: {args.output_dir}")

    # 读取数据
    J, node_mapping, reverse_mapping = read_edge_list(args.file)
    print(f"\nProblem size: {J.shape[0]} nodes")
    print(f"Number of edges: {J.nnz // 2}")

    # 运行实验
    run_energy_tracking_experiments(J, output_dir=args.output_dir,
                                    enable_parallel=args.parallel)


if __name__ == '__main__':
    main()