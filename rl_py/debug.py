import inspect
import torch

def shape_debug(tensor):
    # 获取上一层函数调用的帧
    frame = inspect.currentframe().f_back
    # 获取该帧中的局部变量字典
    variables = frame.f_locals

    # 查找变量名
    tensor_name = None
    for name, value in variables.items():
        if value is tensor:
            tensor_name = name
            break

    # 打印变量名和形状
    if tensor_name and torch.is_tensor(tensor):
        print(f"{tensor_name}.shape = {tensor.shape}")
    elif tensor_name and not torch.is_tensor(tensor):
        print(f"{tensor_name} = {tensor}")
    else:
        print("Variable not found in the local scope.")