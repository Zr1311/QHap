import os

def calculate_hamming_error(block_fundation, block_haplotype, matching_type, site_values, error_site_counts):
    # 初始化相关变量
    hamming_error = 0
    hamming_error_site_values = []  # 用于记录产生hamming error的位点值

    # 遍历每个位点
    for i in range(len(block_fundation[0])):
        base_fundation_2 = block_fundation[1][i]
        base_fundation_3 = block_fundation[2][i]
        base_haplotype_2 = block_haplotype[1][i]
        base_haplotype_3 = block_haplotype[2][i]

        # 方式1: haplotype 2 与 fundation 2，haplotype 3 与 fundation 3 匹配
        if matching_type == 1:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_2 or base_haplotype_3 != base_fundation_3:
                        hamming_error += 1
                        hamming_error_site_values.append(site_values[i])
                else:
                    print(
                        f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")
                    error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1
            else:
                print(
                    f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")
                error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1

        # 方式2: haplotype 2 与 fundation 3，haplotype 3 与 fundation 2 匹配
        elif matching_type == 2:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_3 or base_haplotype_3 != base_fundation_2:
                        hamming_error += 1
                        hamming_error_site_values.append(site_values[i])
            else:
                print(
                    f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")
                error_site_counts[site_values[i]] = error_site_counts.get(site_values[i], 0) + 1

    return hamming_error, len(block_fundation[0]), hamming_error_site_values

def process_blocks(fundation_dir, haplotype_dir):
    total_hamming_errors = 0
    all_site_values = set()  # 用于收集所有块的位点信息
    block_results = {}
    block_hamming_error_site_values = {}
    error_site_counts = {}  # 用于记录每个位点触发打印语句的次数

    # 遍历文件夹中的文件
    fundation_files = sorted(os.listdir(fundation_dir))
    haplotype_files = sorted(os.listdir(haplotype_dir))
    file_count = len(fundation_files)

    for fundation_file, haplotype_file in zip(fundation_files, haplotype_files):
        # 读取每个文件的内容
        fundation_file_path = os.path.join(fundation_dir, fundation_file)
        haplotype_file_path = os.path.join(haplotype_dir, haplotype_file)

        # 读取fundation文件
        with open(fundation_file_path, 'r') as f:
            lines = f.readlines()
            site_values = lines[0].strip().split()  # 记录第一行的位点值
            block_fundation = [line.strip().split() for line in lines]
            all_site_values.update(site_values)  # 收集当前块的位点信息

        # 读取haplotype文件
        with open(haplotype_file_path, 'r') as f:
            block_haplotype = [line.strip().split() for line in f.readlines()]

        # 计算方式1和方式2的hamming error
        hamming_error_1, total_sites, hamming_site_values_1 = calculate_hamming_error(
            block_fundation, block_haplotype, matching_type=1, site_values=site_values,
            error_site_counts=error_site_counts)
        hamming_error_2, _, hamming_site_values_2 = calculate_hamming_error(
            block_fundation, block_haplotype, matching_type=2, site_values=site_values,
            error_site_counts=error_site_counts)

        # 选择较小的hamming error
        if hamming_error_1 <= hamming_error_2:
            block_hamming_error = hamming_error_1
            block_hamming_error_site_values[fundation_file] = hamming_site_values_1
        else:
            block_hamming_error = hamming_error_2
            block_hamming_error_site_values[fundation_file] = hamming_site_values_2

        block_hamming_rate = block_hamming_error / total_sites if total_sites > 0 else 0

        # 更新总的hamming error
        total_hamming_errors += block_hamming_error
        block_results[fundation_file] = (block_hamming_error, block_hamming_rate)

    # 计算去重后的位点数量
    total_sites_count = len(all_site_values)
    print(f"total_sites_count = {total_sites_count}")

    # 找出在每个文件处理中都触发打印语句的位点
    always_error_sites = [site for site, count in error_site_counts.items() if count == file_count]
    print("在每个文件处理中都触发打印语句的位点:")
    print(always_error_sites)

    # 计算总的hamming error rate
    total_hamming_rate = total_hamming_errors / total_sites_count if total_sites_count > 0 else 0
    print(f"total_hamming_errors = {total_hamming_errors}")
    print(f"total_hamming_rate = {total_hamming_rate}")
    return block_results, total_hamming_errors, total_hamming_rate, total_sites_count, block_hamming_error_site_values

def save_block_results_to_file(block_results, block_hamming_error_site_values, block_output_file):
    # 保存每个块的hamming error和rate到文件
    with open(block_output_file, 'w', encoding='utf-8') as f:
        f.write("每个块的hamming error和rate：\n")
        for block, (hamming_error, hamming_rate) in block_results.items():
            site_values = block_hamming_error_site_values[block]
            f.write(
                f"{block}: Hamming Error = {hamming_error}, Hamming Error Rate = {hamming_rate:.4f}, Hamming Error Site Values = {site_values}\n")

def save_total_results_to_file(total_hamming_errors, total_hamming_rate, total_sites_count, total_output_file):
    # 保存总的hamming error和rate到文件
    with open(total_output_file, 'w', encoding='utf-8') as f:
        f.write(f"位点总数 = {total_sites_count}\n")
        f.write(f"总的Hamming Error = {total_hamming_errors}\n")
        f.write(f"总的Hamming Error Rate = {total_hamming_rate:.4f}\n")
