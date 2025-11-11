import torch
import torch.nn as nn
import numpy as np
import os

from utils_functions import VisualizationTools
from utils_functions import NumericalTools
from utils_functions import MeshTools
from utils_functions import setup_seed


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(torch.pi * x)

class CosActivation(nn.Module):
    def forward(self, x):
        return torch.cos(torch.pi * x)
    
class GaussActivation(nn.Module):
    def forward(self, x):
        return torch.exp(- x**2)

ACTIVATION_REGISTRY = {
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'relu': nn.ReLU(),
    'sin': SinActivation(),
    'cos': CosActivation(),
    'gauss': GaussActivation()
}

def apply_weight_init(layer, method):
    methods = {
        'normal': lambda x: nn.init.normal_(x.weight, mean=0.0, std=0.5),
        'uniform': lambda x: nn.init.uniform_(x.weight, a=-0.7, b=0.7),
        'xavier_uniform': lambda x: nn.init.xavier_uniform_(x.weight, gain=6.0),
        'xavier_normal': lambda x: nn.init.xavier_normal_(x.weight, gain=1.0),
        'kaiming_uniform': lambda x: nn.init.kaiming_uniform_(x.weight),
        'kaiming_normal': lambda x: nn.init.kaiming_normal_(x.weight),
        'orthogonal': lambda x: nn.init.orthogonal_(x.weight, gain=1)
    }
    if method not in methods:
        raise ValueError(f"Unsupported weight init method: {method}")
    methods[method](layer)

def apply_bias_init(layer, method):
    methods = {
        'normal': lambda x: nn.init.normal_(x.bias, mean=0.0, std=0.5),
        'uniform': lambda x: nn.init.uniform_(x.bias, a=-0.5, b=0.5),
    }
    if method not in methods:
        raise ValueError(f"Unsupported bias init method: {method}")
    methods[method](layer)

class PIELM(nn.Module):
    def __init__(self, mlp_layers, act='tanh', w_init='xavier_normal', b_init='normal'):
        super(PIELM, self).__init__()
        self.layers = mlp_layers
        self.weight_init = w_init
        self.bias_init = b_init
        self.activation_fn = self.get_activation_fn(act)
        self.model = self.build_network()

    def get_activation_fn(self, act_name):
        if act_name not in ACTIVATION_REGISTRY:
            raise ValueError(f"Unsupported activation: {act_name}")
        return ACTIVATION_REGISTRY[act_name]

    def build_network(self):
        layers = []
        for i in range(len(self.layers) - 1):
            linear = nn.Linear(self.layers[i], self.layers[i + 1], bias=True)
            apply_weight_init(linear, self.weight_init)
            # apply_bias_init(linear, self.bias_init)
            layers.append(linear)
            layers.append(self.activation_fn)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ------------------------- 主程序 ------------------------- #
# setup_seed(3407)
mlp_layers = [2, 256, 2048]
act = 'sin'
w_init = 'uniform'
b_init = 'normal'
dtype = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建保存结果的目录
results_dir = "iteration_results"
os.makedirs(results_dir, exist_ok=True)

# 训练数据加载
T1_p_np, T1_c_np, T1_v_np = MeshTools.load_vtu_mesh("/home/fit_elm/train0.vtu")
T2_p_np, T2_c_np, T2_v_np = MeshTools.load_vtu_mesh("/home/fit_elm/square_uniform.vtu")
# t = 0.0
# T2_v_np = np.exp(-50*((T2_p_np[:,0]-0.5*np.cos(2*np.pi*t))**2+(T2_p_np[:,1]-0.5*np.sin(2*np.pi*t))**2))

# 初始化增量训练数据
X_all_gauss = np.empty((0, T1_p_np.shape[1]))
y_all_gauss = np.empty((0,))

max_iterations = 20
tolerance = 9e-5
pielm = PIELM(mlp_layers, act, w_init, b_init).to(device, dtype)
for order in range(max_iterations):
    print(f"\n🔄 开始第 {order + 1} 次迭代")
    
    combined_p = np.vstack([T1_p_np, X_all_gauss])
    combined_v = np.hstack([T1_v_np, y_all_gauss])

    # 转Tensor
    T1_p = torch.tensor(combined_p, dtype=dtype).to(device)
    T1_v = torch.tensor(combined_v, dtype=dtype).view(-1, 1).to(device)

    # 训练ELM
    with torch.no_grad():

        print("🔧 Training...")
        H = pielm(T1_p)
        coef_solution = torch.linalg.lstsq(H, T1_v).solution

        y_pred_train = H @ coef_solution
        loss = torch.mean((T1_v - y_pred_train) ** 2)
        print(f"📉 Training MSE Loss: {loss.item():.4e}")

        # 用 (order+1) 阶高斯点评估误差
        Lambda_eval, _ = MeshTools.get_quadrature_rule(2, order + 1)
        X_gauss, y_gauss = MeshTools.generate_oversample_gauss(T1_c_np, T1_p_np, T1_v_np, Lambda_eval)
        
        
        # y_gauss = np.exp(-50*((X_gauss[:,0]-0.5*np.cos(2*np.pi*t))**2+(X_gauss[:,1]-0.5*np.sin(2*np.pi*t))**2))

        X_eval_tensor = torch.tensor(X_gauss, dtype=dtype).to(device)
        y_true_tensor = torch.tensor(y_gauss, dtype=dtype).view(-1, 1).to(device)
        H_eval = pielm(X_eval_tensor)
        y_pred_eval = H_eval @ coef_solution
        abs_errors = torch.abs(y_true_tensor - y_pred_eval).detach().cpu().numpy().flatten()

        print(f'{np.mean(abs_errors):.4e}')
        
        # 判断是否收敛
        if np.mean(abs_errors) < tolerance:
            print(f"🎉 Training converged at order {order + 1}")
            
            # 保存收敛时的结果图
            iteration_str = f"iter{order+1:02d}_converged"
            VisualizationTools.mesh_with_gauss(T1_p_np, T1_c_np, 
                                             save_path=os.path.join(results_dir, f"{iteration_str}_mesh_raw.pdf"))
            VisualizationTools.mesh_with_gauss(T1_p_np, T1_c_np, gauss_points=X_all_gauss, 
                                             save_path=os.path.join(results_dir, f"{iteration_str}_mesh_with_gauss.pdf"))
            
            # 测试集评估
            print("🔮 Predicting on new mesh...")
            T2_p = torch.tensor(T2_p_np, dtype=dtype).to(device)
            T2_v = torch.tensor(T2_v_np, dtype=dtype).view(-1, 1).to(device)
            H_test = pielm(T2_p)
            y_pred_test = H_test @ coef_solution
            y_pred_test_np = y_pred_test.cpu().numpy().flatten()
            error = torch.mean(torch.abs(T2_v - y_pred_test)).item()
            print(f"📉 Test ABS Error: {error:.4e}")
            
            VisualizationTools.plot_solution_comparison(T2_p_np, T2_c_np, T2_v_np, y_pred_test_np, 
                                                      os.path.join(results_dir, f"{iteration_str}_solution_comparison.pdf"))
            break

        # 误差分析，筛选高误差区域
        mean_e = np.mean(abs_errors)
        std_e = np.std(abs_errors)
        k = 9.0  # 可调节
        selected_mask = abs_errors > (mean_e + k * std_e)

        X_new = X_gauss[selected_mask]
        y_new = y_gauss[selected_mask]

        # 添加并去重
        X_all_gauss = np.vstack([X_all_gauss, X_new])
        y_all_gauss = np.hstack([y_all_gauss, y_new])
        # Xy_all = np.hstack([X_all_gauss, y_all_gauss.reshape(-1, 1)])
        # Xy_all_unique = np.unique(Xy_all, axis=0)
        # X_all_gauss = Xy_all_unique[:, :-1]
        # y_all_gauss = Xy_all_unique[:, -1]

        # 测试集评估
        print("🔮 Predicting on new mesh...")
        # H_test = pielm(T2_p)
        # y_pred_test = H_test @ coef_solution
        # y_pred_test_np = y_pred_test.cpu().numpy().flatten()
        # error = torch.mean(torch.abs(T2_v - y_pred_test)).item()
        # print(f"📉 Test ABS Error: {error:.4e}")

    # 可视化 - 保存每次迭代的结果
    iteration_str = f"iter{order+1:02d}"
    
    # 保存带高斯点的网格图
    VisualizationTools.mesh_with_gauss(T1_p_np, T1_c_np, gauss_points=X_all_gauss, 
                                     save_path=os.path.join(results_dir, f"{iteration_str}_mesh_with_gauss.pdf"))
    
    # 保存解的比较图
    # VisualizationTools.plot_solution_comparison(T2_p_np, T2_c_np, T2_v_np, y_pred_test_np, 
    #                                           os.path.join(results_dir, f"{iteration_str}_solution_comparison.pdf"))

    print(f"✅ 第 {order + 1} 次迭代完成，结果图已保存至 {results_dir} 目录")

# 保存原始网格图
VisualizationTools.mesh_with_gauss(T1_p_np, T1_c_np, 
                                    save_path=os.path.join(results_dir, f"mesh_raw.pdf"))
print("🎉 所有迭代完成！")