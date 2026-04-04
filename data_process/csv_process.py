import os
import glob
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_line(line):
    # 提取所有数字
    numbers = re.findall(r'\d+', line)
    # 第一列是 frame，后面都是关节值
    return ','.join(numbers)

def process_csv(file_path):
    dir_name = os.path.dirname(file_path)
    output_path = os.path.join(dir_name, "states_clean.csv")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 处理表头
    header = "frame,j1,j2,j3,j4,j5,j10\n"

    # 处理每一行
    cleaned = [header]
    for line in lines[1:]:  # 跳过原表头
        if line.strip():
            cleaned.append(clean_line(line) + '\n')

    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned)

    print(f"✅ 成功：{output_path}")

def batch_process():
    
    files = glob.glob(os.path.join(SCRIPT_DIR ,"./data/task_*/states.csv"))
    if not files:
        print("❌ 未找到文件")
        return

    print(f"找到 {len(files)} 个文件\n")
    for f in files:
        process_csv(f)
    print("\n🎉 全部处理完成！")

if __name__ == "__main__":
    batch_process()
