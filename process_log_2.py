# process_log2.py
import os
import sys
from glob import glob
from typing import List, Any

import pandas as pd
from tqdm import tqdm

# 1. 从原脚本中 import 已有的函数
# 请把 your_script 改成你原来脚本的名字（不含 .py）
from process_log import generate_patterns, read_log, logdata_to_df

# 2. 主流程
def main() -> None:
    patterns = generate_patterns()
    log_root = "logs"
    log_files = glob(os.path.join(log_root, "*/*.txt"))

    # 3. 收集所有能提取出完整信息的日志
    log_data_list: List[List[Any]] = []
    valid_paths: List[str] = []

    for log_path in tqdm(log_files, desc="Scanning logs"):
        data = read_log(log_path, patterns)
        if data:                          # 只要 read_log 返回非空列表，就是有效日志
            log_data_list.append(data)
            valid_paths.append(log_path)

    if not log_data_list:
        print("未找到任何有效日志，请检查 logs 目录及正则表达式。")
        sys.exit(0)

    # 4. 生成 DataFrame
    df = logdata_to_df(log_data_list)

    # 5. 把日志路径作为第一列
    df.insert(0, "path", valid_paths)

    # 6. 打印到终端
    # 设置 pandas 打印参数，确保全部显示
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n========== 所有有效日志提取结果 ==========\n")
    print(df.to_string(index=False))

    # 7. 保存到 Excel
    output_excel = "all_valid_logs.xlsx"
    df.to_excel(output_excel, index=False)
    print(f"\n已写入 Excel：{output_excel}")

if __name__ == "__main__":
    main()