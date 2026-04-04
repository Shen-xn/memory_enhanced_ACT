import torch

# 检查CUDA是否可用
print("CUDA是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    # 显示GPU设备数量
    print("GPU设备数量:", torch.cuda.device_count())
    
    # 显示当前使用的GPU
    print("当前使用的GPU索引:", torch.cuda.current_device())
    print("当前使用的GPU名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    # 创建一个张量并移动到GPU
    x = torch.tensor([1.0, 2.0]).cuda()
    print("张量所在设备:", x.device)
else:
    print("无法使用GPU，请检查配置")