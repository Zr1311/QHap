import sys
from collections import Counter


def process_vcf(input_file, output_file):
    # 初始化计数器
    genotype_counts = Counter()

    # 用于存储符合条件的行
    valid_lines = []

    try:
        # 读取文件并处理
        with open(input_file, 'r') as f:
            for line in f:
                # 跳过注释行
                if line.startswith('#'):
                    continue

                # 分割行数据
                parts = line.strip().split('\t')

                # 确保有足够的字段
                if len(parts) < 10:
                    continue

                # 获取最后一列，提取第一个冒号前的基因型部分
                last_column = parts[9]
                # 分割获取第一个冒号前的内容作为基因型
                genotype = last_column.split(':')[0]

                # 统计基因型
                genotype_counts[genotype] += 1

                # 保留符合条件的行（基因型为'0|1'或'1|0'）
                if genotype in ('0|1', '1|0'):
                    valid_lines.append(line)

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

    # 输出统计结果
    print("基因型统计:")
    print(f"0|1 的数量: {genotype_counts['0|1']}")
    print(f"1|0 的数量: {genotype_counts['1|0']}")

    # 计算其他类型的总数
    other_types = {gt: count for gt, count in genotype_counts.items() if gt not in ('0|1', '1|0')}
    total_other = sum(other_types.values())

    print(f"其他类型的数量: {total_other}")

    # 输出其他类型的具体情况
    if other_types:
        print("\n其他类型的具体分布:")
        for gt, count in sorted(other_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {gt}: {count}")

    # 写入过滤后的文件
    try:
        with open(output_file, 'w') as f:
            f.writelines(valid_lines)
        print(f"已将过滤后的结果保存到 {output_file}")
        return True
    except Exception as e:
        print(f"写入文件时出错: {e}")
        return False


if __name__ == "__main__":
    # 默认输入输出文件路径
    input_file = r"E:\my_work\phasing\phasing_v11_GPU\fundation_file\pav\chm13_smvar\chr1\pav_HG002_chr1_filter_final.vcf"
    output_file = r"E:\my_work\phasing\phasing_v11_GPU\fundation_file\pav\chm13_smvar\chr1\pav_HG002_chr1_filter_final_01_10_only.vcf"

    # 处理文件
    process_vcf(input_file, output_file)
