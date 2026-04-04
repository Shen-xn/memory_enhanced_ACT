import os
import shutil
import glob

def restore_single_task(task_dir):
    """
    从 task_copy 恢复原始数据
    """
    backup_dir = os.path.join(task_dir, "task_copy")
    
    # 检查是否存在备份
    if not os.path.exists(backup_dir):
        print(f"⏭️  {task_dir} 无备份，跳过")
        return

    print(f"\n🔄 开始恢复：{task_dir}")

    # 列出备份里的所有内容（排除自身）
    backup_contents = []
    for item in os.listdir(backup_dir):
        item_path = os.path.join(backup_dir, item)
        backup_contents.append((item, item_path))

    # 先清空当前任务目录下 非task_copy 的所有内容
    for item in os.listdir(task_dir):
        target_path = os.path.join(task_dir, item)
        if item == "task_copy":
            continue
        
        # 删除处理后生成的所有文件/文件夹
        try:
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)
            else:
                os.remove(target_path)
            print(f"🗑️  清理：{item}")
        except Exception as e:
            print(f"⚠️  清理失败 {item}：{e}")

    # 从备份复制回去
    for item_name, src_path in backup_contents:
        dst_path = os.path.join(task_dir, item_name)
        try:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            print(f"✅ 恢复：{item_name}")
        except Exception as e:
            print(f"❌ 恢复失败 {item_name}：{e}")

    print(f"🎉 恢复完成：{task_dir}")

def batch_restore_all():
    print("=" * 60)
    print("📦 任务文件夹一键恢复脚本（从 task_copy 还原）")
    print("⚠️  会删除所有处理后文件，恢复到预处理前状态")
    print("=" * 60)

    confirm = input("\n确定要恢复所有任务文件夹吗？(y/n)：")
    if confirm.lower() != "y":
        print("🚫 已取消恢复")
        return

    # 获取所有 task_* 目录
    task_dirs = glob.glob("./data/task_*")
    task_dirs = [d for d in task_dirs if os.path.isdir(d)]

    if not task_dirs:
        print("❌ 未找到 task_* 文件夹")
        return

    print(f"\n🚀 找到 {len(task_dirs)} 个任务目录，开始恢复...")

    for task_dir in task_dirs:
        restore_single_task(task_dir)

    print("\n🎉🎉🎉 所有任务已成功恢复到原始状态！")

if __name__ == "__main__":
    batch_restore_all()
