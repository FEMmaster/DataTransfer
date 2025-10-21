# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:55:28 2023

@author: Bear
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import matplotlib.pyplot as plt

import time

from utils_functions import VisualizationTools
from utils_functions import NumericalTools
from utils_functions import MeshTools
from utils_functions import setup_seed


#%% 一些可选参数
# 可选的初始化方法
init_methods = {
    'normal': lambda x: nn.init.normal_(x.weight, mean=0.0, std=0.1),
    'uniform': lambda x: nn.init.uniform_(x.weight, a=-0.1, b=0.1),
    'xavier_uniform': lambda x: nn.init.xavier_uniform_(x.weight),
    'xavier_normal': lambda x: nn.init.xavier_normal_(x.weight),
    'kaiming_uniform': lambda x: nn.init.kaiming_uniform_(x.weight, mode='fan_in', nonlinearity='relu'),
    'kaiming_normal': lambda x: nn.init.kaiming_normal_(x.weight, mode='fan_in', nonlinearity='relu'),
    'orthogonal': lambda x: nn.init.orthogonal_(x.weight, gain=1)
}

# 可选的激活函数
activation_funcs = {
    'tanh': lambda: nn.Tanh(),
    'relu': lambda: nn.ReLU(),
    'leaky_relu': lambda: nn.LeakyReLU(),
    'sigmoid': lambda: nn.Sigmoid(),
    'softplus': lambda: nn.Softplus(),
    'none': lambda: nn.Identity()  # 如果不需要激活函数，可以选择'none'
}

# 可选的学习率调度器
lr_schedulers = {
    'step_lr': lambda opt: lr_scheduler.StepLR(opt, step_size=20, gamma=0.1),
    'multi_step_lr': lambda opt: lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 90], gamma=0.1),
    'exponential_lr': lambda opt: lr_scheduler.ExponentialLR(opt, gamma=0.97),
    'cosine_annealing_lr': lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=100),
    'reduce_lr_on_plateau': lambda opt: lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6),
    'cyclic_lr': lambda opt: lr_scheduler.CyclicLR(opt, base_lr=1e-4, max_lr=1e-2, step_size_up=10, mode='triangular'),
    'one_cycle_lr': lambda opt: lr_scheduler.OneCycleLR(opt, max_lr=1e-2, total_steps=100),
    'none': lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: 1)  # 保持学习率不变
}


#%% 神经网络模型
class NN(nn.Module):
    def __init__(self, mlp_layers, activation_funcs, init_methods):
        super(NN, self).__init__()
        self.model = nn.Sequential()

        for i in range(len(mlp_layers) - 2):
            linear_layer = nn.Linear(mlp_layers[i], mlp_layers[i + 1], bias=True)
            init_methods(linear_layer)
            self.model.add_module(f'fc{i + 1}', linear_layer)
            self.model.add_module(f'act{i + 1}', activation_funcs())

        final_layer = nn.Linear(mlp_layers[-2], mlp_layers[-1], bias=False)
        init_methods(final_layer)
        self.model.add_module(f'fc{len(mlp_layers) - 1}', final_layer)
        
    def forward(self, x):
        return self.model(x)

def exact_solution(X):
    # Split input tensor into x and y components
    x, y = X[:, 0], X[:, 1]
    
    # Set time parameter (still 0.00 as in original)
    t = torch.tensor(0.00, device=X.device)
    return torch.exp(-50*((x-0.5*torch.cos(2*torch.pi*t))**2 + (y-0.5*torch.sin(2*torch.pi*t))**2))

    # return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    # return torch.sin(5 * torch.pi * x) * torch.sin(5 * torch.pi * y)
    # return (torch.sin(10 * torch.pi * x) * torch.sin(10 * torch.pi * y) + 
    #         0.5 * torch.exp(-50 * ((x - 0.3)**2 + (y - 0.7)**2)))
    
    # # Poisson 6 L-shaped region solution:
    # theta = torch.atan2(y, x)
    # theta = torch.where(theta >= 0, theta, theta + 2 * torch.pi)
    # r = torch.sqrt(x**2 + y**2)
    # return r**(2/3) * torch.sin(2/3 * theta)


#%% 超参数设置
if __name__ == '__main__':
    # setup_seed(3407)
    # ==================== 初始化配置 ====================
    domain = (-1, 1, -1, 1)
    mlp_layers = (2, 80, 80, 80, 1)
    dtype = torch.float64
    activation = 'tanh'
    init_mode = 'xavier_normal'
    lr_strategy = 'none'
    adam_epochs = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "/home/paper2/train_data/" 
    result_dir = "/home/paper2/test_result/"
    
    raw_data = np.load(data_dir + 'mesh_data.npz')
    T1_p, T1_c, T1_v = raw_data['T1_p'], raw_data['T1_c'], raw_data['T1_v']
    T2_p, T2_c, T2_v = raw_data['T2_p'], raw_data['T2_c'], raw_data['T2_v']
    T3_p, T3_c, T3_v = raw_data['T3_p'], raw_data['T3_c'], raw_data['T3_v']
    T4_p, T4_c, T4_v = raw_data['T4_p'], raw_data['T4_c'], raw_data['T4_v']
    T5_p, T5_c, T5_v = raw_data['T5_p'], raw_data['T5_c'], raw_data['T5_v']

    X_train_T1, T1_cells, y_train_T1 = MeshTools.convert_mesh_group(T1_p, T1_c, T1_v, dtype, device)
    X_test_T2,  T2_cells,  y_test_T2 = MeshTools.convert_mesh_group(T2_p, T2_c, T2_v, dtype, device)
    X_test_T3,  T3_cells,  y_test_T3 = MeshTools.convert_mesh_group(T3_p, T3_c, T3_v, dtype, device)
    X_test_T4,  T4_cells,  y_test_T4 = MeshTools.convert_mesh_group(T4_p, T4_c, T4_v, dtype, device)
    X_test_T5,  T5_cells,  y_test_T5 = MeshTools.convert_mesh_group(T5_p, T5_c, T5_v, dtype, device)

    # ==================== 模型初始化 ====================
    model = NN(mlp_layers, activation_funcs[activation], init_methods[init_mode]).to(device).double()


    #%% Adam训练
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_schedulers[lr_strategy](optimizer_adam)
    loss_log_adam = []
    start_time = time.time()
    for epoch in range(adam_epochs):
        model.zero_grad()
        output = model(X_train_T1)
        loss = torch.mean((output - y_train_T1[:,None]) ** 2)
        loss.backward()
        optimizer_adam.step()
        if lr_strategy == 'reduce_lr_on_plateau':
            scheduler.step(loss)
        else:
            scheduler.step()
        
        loss_log_adam.append(loss.item())
        current_lr = optimizer_adam.param_groups[0]['lr']
        if (epoch + 1) % 100 == 0:
            info = f'Epoch # {epoch+1:4d}/{adam_epochs}\ttime:{time.time()-start_time:.1f}\t' + \
                f'loss:{loss.item():.2e}, ' + \
                f'lr:{current_lr:.2e}'
            print(info)

    #%% LBFGS训练
    optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=10000, line_search_fn="strong_wolfe")
    loss_log_lbfgs = []
    it = [0]      # 使用列表来存储迭代计数
    
    def closure():
        model.zero_grad()
        output = model(X_train_T1)
        loss = torch.mean((output - y_train_T1[:,None]) ** 2)
        loss.backward()
        it[0] += 1
        
        loss_log_lbfgs.append(loss.item())
        if (it[0]) % 100 == 0:
            info = f'Iter # {it[0]:4d}\ttime:{time.time() - start_time:.1f}\t' + \
                f'loss:{loss.item():.2e}'
            print(info)
        return loss
    optimizer_lbfgs.step(closure)

    print(f"used time: {time.time() - start_time:.1f}")

    #%% loss曲线
    plt.rcParams.update({'font.size':18})
    plt.figure(figsize=(6, 5))
    plt.plot(loss_log_adam + loss_log_lbfgs, label='$loss$')
    # plt.xlabel('epochs')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(result_dir + 'NN_loss.pdf')
    plt.show()

    #%% Predict
    for X_test, y_test, cells, name in [
        (X_train_T1, y_train_T1, T1_cells, 'T1'),
        (X_test_T2,  y_test_T2,  T2_cells,  'T2'),
        (X_test_T3,  y_test_T3,  T3_cells,  'T3'),
        (X_test_T4,  y_test_T4,  T4_cells,  'T4'),
        (X_test_T5,  y_test_T5,  T5_cells,  'T5'),
    ]:
        # Convert data to NumPy
        u_true = y_test.detach().cpu().numpy().squeeze()
        X_test_np = X_test.detach().cpu().numpy()
        cells_np = cells.cpu().numpy()

        # Predict
        u_pred = model(X_test).detach().cpu().numpy().squeeze()

        # Optional: mask out invalid values if needed (e.g., L-shape domains)
        # mask = some_condition(X_test_np)  # boolean mask
        # u_pred[mask] = np.nan
        # u_true[mask] = np.nan

        # Plot and save results
        VisualizationTools.plot_solution_comparison(
            X_test_np, cells_np, u_true, u_pred, save_path=f"{result_dir}NN_{name}.pdf"
        )

        # Compute errors
        rel_l2, abs_mean = NumericalTools.calculate_errors(u_true, u_pred)
        print(f'[Case {name}] Relative L2 Error: {rel_l2:.2e}, Absolute Mean Error: {abs_mean:.2e}')
