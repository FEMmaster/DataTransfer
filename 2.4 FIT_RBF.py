import torch
import torch.nn as nn
import numpy as np
import time

from utils_functions import VisualizationTools
from utils_functions import NumericalTools
from utils_functions import MeshTools
from utils_functions import setup_seed

class RBFELM(nn.Module):
    def __init__(self, mlp_layers, X_train, gamma=None):
        super().__init__()
        mlp_layers[1] = min(mlp_layers[1], X_train.shape[0])
        self.layers = mlp_layers

        self.gamma = gamma
        # linear = nn.Linear(mlp_layers[1], 3000, bias=True)
        self.build_centers(X_train)

    def build_centers(self, X_train):
        # 从训练集中随机选择部分点作为 RBF 中心
        N = X_train.shape[0]
        idx = torch.randperm(N)[:self.layers[1]]
        self.centers = X_train[idx].detach().clone()  # [L, d]

        # 自动设置 gamma
        if self.gamma is None:
            from sklearn.neighbors import NearestNeighbors
            with torch.no_grad():
                X_cpu = X_train.cpu().numpy()
                nbrs = NearestNeighbors(n_neighbors=2).fit(X_cpu)
                distances, _ = nbrs.kneighbors(X_cpu)
                avg_spacing = distances[:, 1].mean()
                self.gamma = 1.0 / (avg_spacing ** 2)
            print(f"[RBFELM] Estimated gamma: {self.gamma:.4e}")

    def forward(self, X):
        x = X.unsqueeze(1)  # [N, 1, d]
        c = self.centers.unsqueeze(0)  # [1, L, d]
        dist_sq = torch.sum((x - c)**2, dim=-1)  # [N, L]
        return torch.exp(-self.gamma * dist_sq)


if __name__ == "__main__":

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
    
    # setup_seed(3407)
    
    # 创建 ELM 实例
    with torch.no_grad():
        # model = RBFELM([2, 350] , X_train_T1, gamma=40.0).to(device, dtype) # t=0.00
        model = RBFELM([2, 100] , X_train_T1, gamma=50.0).to(device, dtype) # t=0.01
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
            VisualizationTools.plot_solution_comparison(X_test, cells, u_true, u_pred, save_path = f"{result_dir}RBF_{name}.pdf")
            rel_l2, abs_mean = NumericalTools.calculate_errors(u_true, u_pred)
            print(f'[Case {name}] Relative L2 Error: {rel_l2:.2e}, Absolute Mean Error: {abs_mean:.2e}')
