import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

import time

from scipy.spatial import KDTree
import warnings

from utils_functions import VisualizationTools
from utils_functions import NumericalTools
from utils_functions import MeshTools
from utils_functions import setup_seed


class SinActivation(nn.Module):
    def forward(self, x): return torch.sin(torch.pi * x)

class CosActivation(nn.Module):
    def forward(self, x): return torch.cos(torch.pi * x)
    
class GaussActivation(nn.Module):
    def forward(self, x): return torch.exp(- x**2)

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

class PointLocator:
    def __init__(self, nodes, cells, cell_values=None):
        """
        初始化点定位器
        
        参数:
            nodes: 节点坐标数组，形状为 (NN, 2) 或 (NN, 3)
            cells: 单元-节点连接关系数组，形状为 (NC, 3) 三角形网格
            cell_values: 单元值数组，形状为 (NC, ...)，可选
        """
        self.nodes = nodes
        self.cells = cells
        self.cell_values = cell_values
        self.NN = nodes.shape[0]  # 节点数量
        self.NC = cells.shape[0]  # 单元数量
        self.ftype = nodes.dtype
        self.itype = cells.dtype
        
        # 构建单元邻接关系
        self.cell2cell = self.build_cell2cell()
        
        # 构建初始查询索引
        self.start = np.zeros(self.NN, dtype=self.itype)
        for i in range(3):  # 三角形有三个顶点
            self.start[cells[:, i]] = np.arange(self.NC)
    
    def build_edge2cell(self):
        """
        Generate the edge2cell data structure from cell and localEdge.
        """
        cell = self.cells
        NC = self.NC
        NEC = 3
        itype = np.int32
        localEdge = np.array([(1, 2), (2, 0), (0, 1)])

        totalEdge = cell[:, localEdge].reshape(-1, 2)
        
        _, i0, j = np.unique(np.sort(totalEdge, axis=-1), 
                            return_index=True, return_inverse=True, axis=0)
        
        NE = i0.shape[0]
        edge2cell = np.zeros((NE, 4), dtype=itype)
        i1 = np.zeros(NE, dtype=itype)
        i1[j] = np.arange(NEC * NC, dtype=itype)
        
        edge2cell[:, 0] = i0 // NEC
        edge2cell[:, 1] = i1 // NEC
        edge2cell[:, 2] = i0 % NEC
        edge2cell[:, 3] = i1 % NEC
        
        return edge2cell
    
    def build_cell2cell(self):
        """
        构建与参考结果完全一致的单元邻接矩阵
        返回格式：cell2cell[i,j] = k 表示单元i的第j条边邻接单元k（边界边指向自身）
        """
        edge2cell = self.build_edge2cell()
        NC = self.NC
        NEC = 3
        cell2cell = np.zeros((NC, NEC), dtype=self.itype)
        
        cell2cell[edge2cell[:, 0], edge2cell[:, 2]] = edge2cell[:, 1]
        cell2cell[edge2cell[:, 1], edge2cell[:, 3]] = edge2cell[:, 0]
        
        return cell2cell

    def find_point(self, points, verbose=False):
        """
        查找点在哪个单元内并计算重心坐标，并可选验证结果
        
        参数:
            points: 查询点坐标，形状为 (NP, 2)
            verbose: 是否显示验证信息 (默认False)
            
        返回:
            (cell_indices, barycentric_coords)
        """
        NP = points.shape[0]
        
        # 使用KDTree找到最近的节点作为初始猜测
        tree = KDTree(self.nodes)
        _, nearest_nodes = tree.query(points)
        start = self.start[nearest_nodes]  # 初始单元索引
        
        isNotOK = np.ones(NP, dtype=bool)
        a = np.zeros((NP, 3), dtype=self.ftype)
        
        while np.any(isNotOK):
            idx = start[isNotOK]
            pp = points[isNotOK]
            cell_nodes = self.nodes[self.cells[idx]]
            
            # 向量化计算所有面积坐标
            v = cell_nodes - pp[:, None, :]  # (N,3,2)
            a[isNotOK] = np.cross(v[:, [1,2,0]], v[:, [2,0,1]])
            
            min_a = a[isNotOK].min(axis=1)
            isOutCell = min_a < -1e-10
            
            # 更新需要继续处理的点
            idx0 = np.where(isNotOK)[0]
            start[idx0[isOutCell]] = self.cell2cell[
                idx[isOutCell], 
                a[isNotOK][isOutCell].argmin(axis=1)]
            isNotOK[idx0] = isOutCell

        # 归一化重心坐标
        barycentric = a / a.sum(axis=1, keepdims=True)
        
        # 验证结果
        if verbose:
            self._verify_points(points, start, barycentric)
        
        return start, barycentric

    def _verify_points(self, points, cell_indices, bary_coords):
        """
        内部验证方法
        
        参数:
            points: 查询点坐标数组，形状为 (NP, 2)
            cell_indices: 单元索引数组，形状为 (NP,)
            bary_coords: 重心坐标数组，形状为 (NP, 3)
        """
        ERROR_THRESHOLD = 1e-10
        
        for i, (point, cell_idx, coords) in enumerate(zip(points, cell_indices, bary_coords)):
            print(f"\n点 {i+1}: {point}")
            print(f"  所在单元: {cell_idx} (顶点: {self.cells[cell_idx]})")
            print(f"  重心坐标: {coords}")
            
            # 重建点坐标
            reconstructed = coords @ self.nodes[self.cells[cell_idx]]
            error = np.linalg.norm(reconstructed - point)
            print(f"  重建验证: {reconstructed} (误差: {error:.2e})")
            
            # 误差检查
            if error > ERROR_THRESHOLD:
                warnings.warn(f"⚠️ 警告: 点 {i+1} 的重建误差 ({error:.2e}) 超过阈值 ({ERROR_THRESHOLD:.1e})")
                print("  ⚠️ 可能原因:")
                print("  - 点不在单元内")
                print("  - 数值计算精度问题")
                print("  - 网格数据可能有误")
    
    def interpolate(self, points):
        # 1. 找到点所在的单元和重心坐标
        cell_indices, bary_coords = self.find_point(points)
        
        # 2. 获取单元对应的节点索引 (NP, 3)
        cell_nodes = self.cells[cell_indices]
        
        # 3. 获取节点值 (NP, 3, ...)
        node_vals = self.cell_values[cell_nodes]
        
        # 4. 计算插值：Σ(φ_i * u_i)
        return np.einsum('ni,n...i->n...', bary_coords, node_vals)
    
    def plot_mesh(self, points=None, title='Mesh Visualization', figsize=(8, 8)):
        """
        绘制网格和查询点
        
        参数:
            points: 可选，要绘制的查询点坐标数组
            title: 图表标题
            figsize: 图表大小
        """
        plt.figure(figsize=figsize)
        
        # 绘制三角形单元
        for cell in self.cells:
            triangle = self.nodes[cell]
            poly = Polygon(triangle, closed=True, fill=None, edgecolor='gray', alpha=0.5)
            plt.gca().add_patch(poly)
        
        # 绘制节点
        plt.scatter(self.nodes[:,0], self.nodes[:,1], c='blue', s=50, label='Nodes')
        for i, node in enumerate(self.nodes):
            plt.text(node[0], node[1], str(i), ha='right', va='bottom', 
                    fontsize=12, color='blue', weight='bold')
        
        # 绘制单元编号
        for i, cell in enumerate(self.cells):
            centroid = np.mean(self.nodes[cell], axis=0)
            plt.text(centroid[0], centroid[1], str(i), ha='center', va='center',
                   fontsize=10, color='red')
        
        # 绘制查询点（如果提供）
        if points is not None:
            plt.scatter(points[:,0], points[:,1], c='green', s=30, 
                       marker='x', label='Query Points')
        
        plt.title(title)
        plt.legend()
        plt.gca().set_aspect('equal')
        plt.show()

# ------------------------- 主程序 ------------------------- #
setup_seed(3407)
mlp_layers = [2, 128, 1024]
act = 'sin'
w_init = 'uniform'
b_init = 'normal'
dtype = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练数据
T1_p_np, T1_c_np, T1_v_np = MeshTools.load_vtu_mesh("/home/fit_elm/meshu00004.vtu")   # meshu00004
VisualizationTools.mesh_with_gauss(T1_p_np, T1_c_np, save_path="mesh_T1.pdf")  # 原始网格图

# 测试数据
# T2_p_np, T2_c_np, T2_v_np = MeshTools.load_vtu_mesh("/home/fit_elm/meshu00005.vtu")
T2_p_np, T2_c_np, T2_v_np = MeshTools.load_vtu_mesh("/home/fit_elm/square_uniform.vtu")
# VisualizationTools.mesh_with_gauss(T2_p_np, T2_c_np, save_path="mesh_T2.pdf")  # 原始网格图

# ############################### 在自适应网格T1较大单元基础上增加高斯点模块 #################################
# Lambda, _ = MeshTools.get_quadrature_rule(2, 8)
# X_gauss, y_gauss = MeshTools.generate_oversample_gauss(T1_c_np, T1_p_np, T1_v_np, Lambda, area_threshold=0.007)
# VisualizationTools.mesh_with_gauss(T1_p_np, T1_c_np, gauss_points=X_gauss, save_path="mesh_with_gauss.pdf")  # 加入高斯点

# T1_p_np = np.vstack([T1_p_np, X_gauss])
# T1_v_np = np.hstack([T1_v_np, y_gauss])
# #########################################################################################################

# ############################### 将自适应网格T1插值到均匀一致网格T3上模块 #################################
# domain_x = [-1, 1]
# domain_y = [-1, 1]
# num_boundary_points = 100
# num_interior_points = 10000
# T3_p_np, T3_c_np = MeshTools.make_uniform_mesh(domain_x, domain_y, num_boundary_points, num_interior_points)
# VisualizationTools.mesh_with_gauss(T3_p_np, T3_c_np, save_path="mesh_T3.pdf")
# locator = PointLocator(T1_p_np, T1_c_np, T1_v_np)
# T3_v_np = locator.interpolate(T3_p_np)

# T1_p_np = T3_p_np
# T1_v_np = T3_v_np
# #########################################################################################################

# 训练数据转换为PyTorch张量
T1_p = torch.tensor(T1_p_np, dtype=dtype).to(device)
T1_v = torch.tensor(T1_v_np, dtype=dtype).view(-1, 1).to(device) 

# 测试数据转换为PyTorch张量
T2_p = torch.tensor(T2_p_np, dtype=dtype).to(device)
T2_v = torch.tensor(T2_v_np, dtype=dtype).view(-1, 1).to(device)

# 创建并训练 ELM
with torch.no_grad():
    pielm = PIELM(mlp_layers, act, w_init, b_init).to(device, dtype)
    
    print("Training...")
    start_time = time.time()
    H = pielm(T1_p)
    
    coef_solution = torch.linalg.lstsq(H, T1_v).solution

    print(f"Training done in {time.time() - start_time:.4f} sec")

    y_pred_train = H @ coef_solution
    loss = torch.mean((T1_v - y_pred_train) ** 2)
    print(f"Training MSE Loss: {loss:.4e}")

    # 在 T2 上预测
    print("Predicting on new mesh...")
    H_test = pielm(T2_p)
    y_pred_test = H_test @ coef_solution
    y_pred_test_np = y_pred_test.detach().cpu().numpy().flatten()
    error = torch.mean(torch.abs(T2_v - y_pred_test))
    print(f"Training ABS Error: {error:.15e}")

VisualizationTools.plot_solution_comparison(T2_p_np, T2_c_np, T2_v_np, y_pred_test_np, "my_solution_comparison.pdf")

print("All done.")