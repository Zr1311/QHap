import os


def read_bed_ranges(bed_file):
    """读取BED文件，返回chr22的所有有效区间范围"""
    ranges = []
    with open(bed_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == 'chr1':
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                    ranges.append((start, end))
                except ValueError:
                    print(f"警告：BED文件中无效的数值 {parts[1]} 或 {parts[2]}")
    return ranges


def is_position_in_ranges(position, ranges):
    """检查位点是否在任何一个有效区间内"""
    for start, end in ranges:
        if start <= position <= end:
            return True
    return False


def filter_vcf(vcf_file, ranges, output_file):
    """筛选VCF文件，保留在位点范围内的行"""
    with open(vcf_file, 'r') as in_f, open(output_file, 'w') as out_f:
        for line in in_f:
            # 保留注释行
            if line.startswith('#'):
                out_f.write(line)
                continue

            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == 'chr1':
                try:
                    position = int(parts[1])
                    if is_position_in_ranges(position, ranges):
                        out_f.write(line)
                except ValueError:
                    print(f"警告：VCF文件中无效的位点数值 {parts[1]}")


def main():
    # 输入文件路径
    vcf_file = r"E:\my_work\phasing\phasing_v11_GPU\fundation_file\pav\chm13_smvar\chr1\pav_HG002_chr1_filter_final_01_10_only.vcf"
    bed_file = r"E:\my_work\phasing\phasing_v11_GPU\fundation_file\pav\chm13_smvar\CHM13v2.0_HG2-T2TQ100-V1.1_smvar.benchmark.bed"

    # 输出文件路径（在原文件名后添加_filtered）
    file_name, file_ext = os.path.splitext(vcf_file)
    output_file = f"{file_name}_filtered{file_ext}"

    # 读取BED文件中的有效区间
    print("读取BED文件中的有效区间...")
    ranges = read_bed_ranges(bed_file)
    print(f"找到 {len(ranges)} 个chr6的有效区间")

    # 筛选VCF文件
    print("筛选VCF文件...")
    filter_vcf(vcf_file, ranges, output_file)
    print(f"筛选完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
