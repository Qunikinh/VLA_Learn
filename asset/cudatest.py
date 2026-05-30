import torch
print(torch.cuda.is_available())  # 必须输出True
print(torch.cuda.current_device())  # 输出当前设备号


    