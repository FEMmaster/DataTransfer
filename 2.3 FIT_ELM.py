import torch
import torch.nn as nn
import numpy as np
import time
import random

from utils_functions import VisualizationTools
from utils_functions import NumericalTools
from utils_functions import MeshTools
from utils_functions import setup_seed

# 自定义激活函数定义（仍然保留类）
class SinActivation(nn.Module):
    def forward(self, x): return torch.sin(torch.pi * x)

class CosActivation(nn.Module):
    def forward(self, x): return torch.cos(torch.pi * x)

def get_activation_fn(act_name):
    registry = {
        'tanh':    lambda: nn.Tanh(),
        'sigmoid': lambda: nn.Sigmoid(),
        'sin':     lambda: SinActivation(),
        'cos':     lambda: CosActivation()
    }
    if act_name not in registry:
        raise ValueError(f"Unsupported activation function: {act_name}")
    return registry[act_name]()  # 返回一个新的激活函数模块

def apply_weight_init(layer, method_str):
    # 拆分方法名和参数值
    parts = method_str.rsplit('_', 1)
    if len(parts) == 2:
        param = float(parts[1])           # 尝试将最后一段转为参数
        method = parts[0]                 # 只有成功转换时才拆分方法名
    else:
        param = None
        method = method_str               # 转换失败说明是完整方法名，无参数
        
    methods = {
        'normal': lambda x: nn.init.normal_(x.weight, mean=0.0, std=(param or 1.0)),
        'uniform': lambda x: nn.init.uniform_(x.weight, a=-(param or 1.0), b=(param or 1.0)),
        'xavier_uniform': lambda x: nn.init.xavier_uniform_(x.weight, gain=(param or 1.0)),
        'xavier_normal': lambda x: nn.init.xavier_normal_(x.weight, gain=(param or 1.0)),
        'kaiming_uniform': lambda x: nn.init.kaiming_uniform_(x.weight),
        'kaiming_normal': lambda x: nn.init.kaiming_normal_(x.weight),
        'orthogonal': lambda x: nn.init.orthogonal_(x.weight, gain=(param or 1.0))
    }
    if method not in methods:
        raise ValueError(f"Unsupported weight init method: {method}")
    methods[method](layer)

def apply_bias_init(layer, method):
    methods = {
        'normal': lambda x: nn.init.normal_(x.bias, mean=0.0, std=0.1),
        'uniform': lambda x: nn.init.uniform_(x.bias, a=-0.8, b=0.8),
    }
    if method not in methods:
        raise ValueError(f"Unsupported bias init method: {method}")
    methods[method](layer)
       
class ELM(nn.Module):
    def __init__(self, mlp_layers, act='tanh', w_init='xavier_normal'):
        super().__init__()
        self.model = nn.Sequential()
        for i in range(len(mlp_layers) - 1):
            linear = nn.Linear(mlp_layers[i], mlp_layers[i+1], bias=True)
            apply_weight_init(linear, w_init)
            apply_bias_init(linear, 'normal')
            self.model.add_module(f'fc{i+1}', linear)
            self.model.add_module(f'act{i+1}', get_activation_fn(act))

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    setup_seed(3407)
    dtype = torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = "/home/paper2/train_data/" 
    result_dir = "/home/paper2/test_result/"
 
    raw_data = np.load(data_dir + 'mesh_data.npz')
    T1_p, T1_c, T1_v = raw_data['T1_p'], raw_data['T1_c'], raw_data['T1_v'] 
    T2_p, T2_c, T2_v = raw_data['T2_p'], raw_data['T2_c'], raw_data['T2_v']
    T3_p, T3_c, T3_v = raw_data['T3_p'], raw_data['T3_c'], raw_data['T3_v']
    T4_p, T4_c, T4_v = raw_data['T4_p'], raw_data['T4_c'], raw_data['T4_v']
    T5_p, T5_c, T5_v = raw_data['T5_p'], raw_data['T5_c'], raw_data['T5_v']

    # t = 0.01
    # T1_v = np.exp(-50*((T1_p[:,0]-0.5*np.cos(2*np.pi*t))**2+(T1_p[:,1]-0.5*np.sin(2*np.pi*t))**2))
    # T2_v = np.exp(-50*((T2_p[:,0]-0.5*np.cos(2*np.pi*t))**2+(T2_p[:,1]-0.5*np.sin(2*np.pi*t))**2))
    # T3_v = np.exp(-50*((T3_p[:,0]-0.5*np.cos(2*np.pi*t))**2+(T3_p[:,1]-0.5*np.sin(2*np.pi*t))**2))
    # T4_v = np.exp(-50*((T4_p[:,0]-0.5*np.cos(2*np.pi*t))**2+(T4_p[:,1]-0.5*np.sin(2*np.pi*t))**2))
    # T5_v = np.exp(-50*((T5_p[:,0]-0.5*np.cos(2*np.pi*t))**2+(T5_p[:,1]-0.5*np.sin(2*np.pi*t))**2))
    
    X_train_T1, T1_cells, y_train_T1 = MeshTools.convert_mesh_group(T1_p, T1_c, T1_v, dtype, device)
    X_test_T2,  T2_cells,  y_test_T2 = MeshTools.convert_mesh_group(T2_p, T2_c, T2_v, dtype, device)
    X_test_T3,  T3_cells,  y_test_T3 = MeshTools.convert_mesh_group(T3_p, T3_c, T3_v, dtype, device)
    X_test_T4,  T4_cells,  y_test_T4 = MeshTools.convert_mesh_group(T4_p, T4_c, T4_v, dtype, device)
    X_test_T5,  T5_cells,  y_test_T5 = MeshTools.convert_mesh_group(T5_p, T5_c, T5_v, dtype, device)
    
    
    # 创建 ELM 实例
    # mlp_layers = [2, 1024, 4096]
    mlp_layers = [2, 256, 1024]
    act = 'sin'
    w_init = 'uniform_0.8'
    with torch.no_grad():
        # model = ELM([2, 512, 512], 'tanh', 'uniform_0.2').to(device, dtype) # t=0.00
        model = ELM([2, 256, 512], 'tanh', 'uniform_0.2').to(device, dtype) # t=0.01

        start_time = time.time()
        H = model(X_train_T1)
        coef_solution = torch.linalg.lstsq(H, y_train_T1).solution
    
        print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
        
        # 计算训练损失
        y_pred = H @ coef_solution
        loss = torch.mean((y_train_T1 - y_pred) ** 2)
        print(f"训练损失: {loss:.2e}")
        
        # ========== 测试阶段 ==========
        for X_test, y_test, cells, name in [
            (X_train_T1, y_train_T1, T1_cells, 'T1'),
            (X_test_T2, y_test_T2, T2_cells, 'T2'),
            (X_test_T3, y_test_T3, T3_cells, 'T3'),
            (X_test_T4, y_test_T4, T4_cells, 'T4'),
            (X_test_T5, y_test_T5, T5_cells, 'T5'),
        ]:
            # 真值
            u_true = y_test.detach().cpu().numpy().squeeze()

            # 预测
            H_test = model(X_test)
            u_pred = (H_test @ coef_solution).detach().cpu().numpy().squeeze()

            # 误差计算
            X_test = X_test.detach().cpu().numpy()
            cells = cells.cpu().numpy()
            VisualizationTools.plot_solution_comparison(X_test, cells, u_true, u_pred, save_path = f"{result_dir}ELM_{name}.pdf")
            rel_l2, abs_mean = NumericalTools.calculate_errors(u_true, u_pred)
            print(f'[Case {name}] Relative L2 Error: {rel_l2:.2e}, Absolute Mean Error: {abs_mean:.2e}')
