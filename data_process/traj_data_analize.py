import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

# 支持中文显示（Windows/mac/Linux通用）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

def analyze_and_visualize(csv_path):
    """读取轨迹CSV → 分析 → 可视化"""
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在：{csv_path}")
        return

    # 1. 读取数据
    df = pd.read_csv(csv_path)
    print(f"\n📂 加载文件：{csv_path}")
    print(f"📊 总帧数：{len(df)}")
    print(f"📈 数据列：{list(df.columns)}")

    joint_cols = [col for col in df.columns if col != "frame"]
    print(f"🦾 关节维度：{len(joint_cols)} 维")

    # 2. 基本统计
    data = df[joint_cols].values
    frame_diffs = []
    for i in range(1, len(data)):
        dist = math.sqrt(np.sum((data[i] - data[i-1])**2))
        frame_diffs.append(dist)

    if len(frame_diffs) > 0:
        print(f"📏 平均帧间距离：{np.mean(frame_diffs):.2f}")
        print(f"📏 最小帧间距离：{np.min(frame_diffs):.2f}")
        print(f"📏 最大帧间距离：{np.max(frame_diffs):.2f}")
        print(f"🔍 距离 <4 的帧数：{sum(1 for d in frame_diffs if d <4)}")
    else:
        print("📏 只有1帧，无距离信息")

    print("\n✅ 数据预览：")
    print(df.head())

    # 3. 绘图：6个关节轨迹 + 帧间距离
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"轨迹可视化：{os.path.basename(csv_path)}", fontsize=16)

    frames = df["frame"].values

    # 子图1：各关节轨迹
    ax1 = axes[0]
    for i, col in enumerate(joint_cols):
        ax1.plot(frames, df[col], label=f"{col}", linewidth=1.5)
    ax1.set_title("各关节角度轨迹")
    ax1.set_xlabel("帧")
    ax1.set_ylabel("角度值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：帧间欧氏距离
    ax2 = axes[1]
    if len(frame_diffs) > 0:
        ax2.plot(range(1, len(frame_diffs)+1), frame_diffs, color='red', label="帧间运动距离", linewidth=1.5)
        ax2.axhline(y=4, color='orange', linestyle='--', label="阈值 4.0")
    ax2.set_title("帧间运动幅度（6维欧氏距离）")
    ax2.set_xlabel("帧")
    ax2.set_ylabel("距离")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("📊 机械臂轨迹可视化 & 数据分析工具")
    print("📌 使用方法：输入 states_clean.csv 或 states_filtered.csv 路径")
    print("="*60)

    while True:
        path = input("\n请输入轨迹CSV路径（输入 q 退出）：").strip()
        if path.lower() in ["q", "quit", "exit"]:
            print("👋 退出可视化工具")
            break
        analyze_and_visualize(path)
