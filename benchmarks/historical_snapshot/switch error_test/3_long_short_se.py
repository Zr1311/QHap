import os

def calculate_switch_error(block_fundation, block_haplotype, matching_type, site_values, error_site_counts):
    # 初始化相关变量
    short_switch_error = 0
    long_switch_error = 0
    in_switch_error = False
    consecutive_error_count = 0
    total_sites = len(block_fundation[0])  # 假设位点信息行的长度等于数据行数
    short_switch_error_site_values = []  # 用于记录产生 short switch error 的位点值
    long_switch_error_site_values = []  # 用于记录产生 long switch error 的位点值

    # 遍历每个位点
    for i in range(total_sites):
        base_fundation_2 = block_fundation[1][i]
        base_fundation_3 = block_fundation[2][i]
        base_haplotype_2 = block_haplotype[1][i]
        base_haplotype_3 = block_haplotype[2][i]

        # 方式1: haplotype 2 与 fundation 2，haplotype 3 与 fundation 3 匹配
        if matching_type == 1:
            if base_haplotype_2 == base_fundation_2 or base_haplotype_2 == base_fundation_3:
                if base_haplotype_3 == base_fundation_2 or base_haplotype_3 == base_fundation_3:
                    if base_haplotype_2 != base_fundation_2 or base_haplotype_3 != base_fundation_3:
                        if not in_switch_error:
                            consecutive_error_count = 1
                            in_switch_error = True
                        else:
                            consecutive_error_count += 1
                    elif base_haplotype_2 == base_fundation_2 and base_haplotype_3 == base_fundation_3:
                        if in_switch_error:
                            if consecutive_error_count == 1:
                                short_switch_error += 1
                                short_switch_error_site_values.append(site_values[i - 1])
                            else:
                                long_switch_error += 1
                                long_switch_error_site_values.extend(site_values[i - consecutive_error_count:i])
                            consecutive_error_count = 0
                            in_switch_error = False
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
                        if not in_switch_error:
                            consecutive_error_count = 1
                            in_switch_error = True
                        else:
                            consecutive_error_count += 1
                    elif base_haplotype_2 == base_fundation_3 and base_haplotype_3 == base_fundation_2:
                        if in_switch_error:
                            if consecutive_error_count == 1:
                                short_switch_error += 1
                                short_switch_error_site_values.append(site_values[i - 1])
                            else:
                                long_switch_error += 1
                                long_switch_error_site_values.extend(site_values[i - consecutive_error_count:i])
                            consecutive_error_count = 0
                            in_switch_error = False
                # else:
                #     print(f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")
            # else:
            #     print(f"位点: {site_values[i]}, base_haplotype_2={base_haplotype_2},base_haplotype_3={base_haplotype_3},base_fundation_2={base_fundation_2},base_fundation_3={base_fundation_3}")

    # 处理最后一个连续错误
    if in_switch_error:
        if consecutive_error_count == 1:
            short_switch_error += 1
            short_switch_error_site_values.append(site_values[total_sites - 1])
        else:
            long_switch_error += 1
            long_switch_error_site_values.extend(site_values[total_sites - consecutive_error_count:total_sites])

    return short_switch_error, long_switch_error, total_sites, short_switch_error_site_values, long_switch_error_site_values

def process_blocks(fundation_dir, haplotype_dir):
    total_short_switch_errors = 0
    total_long_switch_errors = 0
    all_site_values = set()  # 用于收集所有块的位点信息
    block_results = {}
    block_short_switch_error_site_values = {}
    block_long_switch_error_site_values = {}
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

        # 计算方式1和方式2的switch error
        short_error_1, long_error_1, total_sites, short_site_values_1, long_site_values_1 = calculate_switch_error(
            block_fundation, block_haplotype, matching_type=1, site_values=site_values,
            error_site_counts=error_site_counts)
        short_error_2, long_error_2, _, short_site_values_2, long_site_values_2 = calculate_switch_error(
            block_fundation, block_haplotype, matching_type=2, site_values=site_values,
            error_site_counts=error_site_counts)

        # 选择较小的switch error
        if short_error_1 + long_error_1 <= short_error_2 + long_error_2:
            block_short_switch_error = short_error_1
            block_long_switch_error = long_error_1
            block_short_switch_error_site_values[fundation_file] = short_site_values_1
            block_long_switch_error_site_values[fundation_file] = long_site_values_1
        else:
            block_short_switch_error = short_error_2
            block_long_switch_error = long_error_2
            block_short_switch_error_site_values[fundation_file] = short_site_values_2
            block_long_switch_error_site_values[fundation_file] = long_site_values_2

        # 修改分母为 total_sites - 1
        adjusted_denominator = total_sites - 1 if total_sites > 1 else 1  # 避免除以0
        block_short_switch_rate = block_short_switch_error / adjusted_denominator
        block_long_switch_rate = block_long_switch_error / adjusted_denominator

        # 更新总的switch error
        total_short_switch_errors += block_short_switch_error
        total_long_switch_errors += block_long_switch_error
        block_results[fundation_file] = (block_short_switch_error, block_short_switch_rate,
                                         block_long_switch_error, block_long_switch_rate)

    # 计算去重后的位点数量
    total_sites_count = len(all_site_values)
    print(f"total_sites_count = {total_sites_count}")

    # 找出在每个文件处理中都触发打印语句的位点
    always_error_sites = [site for site, count in error_site_counts.items() if count == file_count]
    print("在每个文件处理中都触发打印语句的位点:")
    print(always_error_sites)

    # 修改分母为 total_sites_count - 1
    adjusted_total_denominator = total_sites_count - 1 if total_sites_count > 1 else 1  # 避免除以0
    total_short_switch_rate = total_short_switch_errors / adjusted_total_denominator
    total_long_switch_rate = total_long_switch_errors / adjusted_total_denominator
    total_switch_errors = total_short_switch_errors + total_long_switch_errors
    total_switch_rate = total_switch_errors / adjusted_total_denominator
    print(f"total_short_switch_rate = {total_short_switch_rate}")
    print(f"total_long_switch_rate = {total_long_switch_rate}")
    print(f"total_switch_rate = {total_switch_rate}")
    return block_results, total_short_switch_errors, total_long_switch_errors, total_short_switch_rate, total_long_switch_rate, total_switch_errors, total_switch_rate, block_short_switch_error_site_values, block_long_switch_error_site_values

def save_block_results_to_file(block_results, block_short_switch_error_site_values, block_long_switch_error_site_values,
                               block_output_file):
    # 保存每个块的switch error和rate到文件
    with open(block_output_file, 'w', encoding='utf-8') as f:
        f.write("每个块的short switch error和rate：\n")
        for block, (short_error, short_rate, long_error, long_rate) in block_results.items():
            short_site_values = block_short_switch_error_site_values[block]
            f.write(
                f"{block}: Short Switch Error = {short_error}, Short Switch Error Rate = {short_rate:.4f}, Short Switch Error Site Values = {short_site_values}\n")
        f.write("\n每个块的long switch error和rate：\n")
        for block, (short_error, short_rate, long_error, long_rate) in block_results.items():
            long_site_values = block_long_switch_error_site_values[block]
            f.write(
                f"{block}: Long Switch Error = {long_error}, Long Switch Error Rate = {long_rate:.4f}, Long Switch Error Site Values = {long_site_values}\n")

def save_total_results_to_file(total_short_switch_errors, total_long_switch_errors, total_short_switch_rate,
                               total_long_switch_rate, total_switch_errors, total_switch_rate, total_output_file):
    # 保存总的switch error和rate到文件
    with open(total_output_file, 'w', encoding='utf-8') as f:
        f.write(
            f"总的Short Switch Error = {total_short_switch_errors}, 总的Short Switch Error Rate = {total_short_switch_rate:.4f}\n")
        f.write(
            f"总的Long Switch Error = {total_long_switch_errors}, 总的Long Switch Error Rate = {total_long_switch_rate:.4f}\n")
        f.write(f"总的Switch Error = {total_switch_errors}, 总的Switch Error Rate = {total_switch_rate:.4f}\n")
