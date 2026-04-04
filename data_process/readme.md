### 使用说明
1. 数据储存：必须在 ./data/task_*_<数字编号>/下储存。原始得到的数据有两个文件夹：depth/ rgb/ 和一个states.csv文件。处理后必须包含four_channel文件夹和states_filtered.csv。
   
2. 脚本 csv_process.py把 data/ 目录下的所有格式化数据中的states.csv文件中奇怪的字符数据改方便的state_clean.csv 文件保留在原目录下。

3. 脚本 data_process_1.py 也对 data/ 目录下的所有格式化数据做处理。对没一条数据：先将深度图截断在0-800，然后归一化到0-255，以jpg格式保存到原目录的 depth_normalized 文件夹中。把上一步生成的 state_clean.csv 进行 Savitzky-Golay 滤波器平滑轨迹，然后计算累积欧几里德距离变化，删除变化平缓的帧，具体参数见脚本内。最后删除了轨迹的帧，同时也删除rbg和深度图，以对齐数据一致性。
注意：对于已经处理好的轨迹数据，会自动跳过该条。

1. 脚本 rollback.py 用于将 data 下的所有处理后的数据进行回滚，恢复到从 Jetson 采集后刚刚传过来的版本。

2. 脚本 traj_data_analize.py 可以进行轨迹可视化，检查滤波和过滤的效果。

3. 脚本 data_process_2.py 通过对 depth_normalized 中的深度图 padding 的方式，将其改变到640*480分辨率，与另外三通道（RGB）对齐。然后与RGB合并，以四通道图的形式存在同路径的 four_channel 路径下，保留编号顺序。

4. 脚本dataloader用于加载数据集，请先按上述步骤处理后使用get_data_loaders加载数据集。
示例：
``` python
train_loader, val_loader = get_data_loaders(
        data_root="./data",
        future_steps=10,
        batch_size=8,
        num_workers=0
        )
```
5. TODO：写一个从数据集合中抽取一半轨迹，直接处理four_channel中的图片对其使用遮挡模拟生成算法并在task_obst_*中储存的脚本，必须处理前备份并有备份回复功能。
6. TODO：写一个利用me_block来处理four_channel的程序，用来生成memory_image_four_channel图像，这对于ACT的训练重要。