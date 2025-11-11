import numpy as np
    
from utils_functions import MeshTools
from utils_functions import VisualizationTools

def exact(x, y):
    # return np.sin(np.pi * x) * np.sin(np.pi * y)
    # return np.sin(5*np.pi * x) * np.sin(5*np.pi * y)

    t = 0.00
    return np.exp(-50*((x-0.5*np.cos(2*np.pi*t))**2+(y-0.5*np.sin(2*np.pi*t))**2))

    # return (np.sin(10*np.pi*x) * np.sin(10*np.pi*y) + 0.5*np.exp(-50*((x-0.3)**2 + (y-0.7)**2)))
    
    # 泊松6 L型区域
    # import torch
    # x_torch = torch.from_numpy(x).double()
    # y_torch = torch.from_numpy(y).double()
    # theta = torch.atan2(y_torch, x_torch)
    # theta = torch.where(theta >= 0, theta, theta + 2 * torch.pi)
    # r = torch.sqrt(x_torch**2 + y_torch**2)
    # value = r**(2/3) * torch.sin(2/3 * theta)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    # return value.numpy()
    
data_dir = "/home/paper2/train_data/"  # 注意末尾加斜线

# 从VTU文件加载数据
T1_points, T1_cells, T1_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00004.vtu")
T2_points, T2_cells, T2_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00003.vtu")
T3_points, T3_cells, T3_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00005.vtu")
T4_points, T4_cells, T4_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00002.vtu")

# T5_points, T5_cells = MeshTools.make_mesh([-1, 1], [-1, 1], 100, 10000)
T5_points, T5_cells = MeshTools.make_uniform_mesh([-1, 1], [-1, 1], 100, 10000)
T5_valu = exact(T5_points[:, 0], T5_points[:, 1])

# 保存数据到字典
data = {
    'T1_p': T1_points, 'T1_c': T1_cells, 'T1_v': T1_valu,
    'T2_p': T2_points, 'T2_c': T2_cells, 'T2_v': T2_valu,
    'T3_p': T3_points, 'T3_c': T3_cells, 'T3_v': T3_valu,
    'T4_p': T4_points, 'T4_c': T4_cells, 'T4_v': T4_valu,
    'T5_p': T5_points, 'T5_c': T5_cells, 'T5_v': T5_valu,
}
np.savez(data_dir + 'mesh_data.npz', **data)

# zoom_config = (-0.2, -0.3, 0.2, 0.2)  # 统一缩放参数

VisualizationTools.mesh_with_gauss(T1_points, T1_cells, save_path=data_dir + "document_T1.pdf")
VisualizationTools.mesh_with_gauss(T2_points, T2_cells, save_path=data_dir + "document_T2.pdf")
VisualizationTools.mesh_with_gauss(T3_points, T3_cells, save_path=data_dir + "document_T3.pdf")
VisualizationTools.mesh_with_gauss(T4_points, T4_cells, save_path=data_dir + "document_T4.pdf")
VisualizationTools.mesh_with_gauss(T5_points, T5_cells, zoom_area = (-0.2, -0.3, 0.2, 0.2), save_path=data_dir + "document_T5.pdf")