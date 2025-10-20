import numpy as np
    
from utils_functions import MeshTools
from utils_functions import VisualizationTools


data_dir = "/home/paper2/train_data/"  # 注意末尾加斜线

# # 从VTU文件加载数据
# T1_points, T1_cells, T1_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00003.vtu")
# T2_points, T2_cells, T2_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00004.vtu")
# T3_points, T3_cells, T3_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00005.vtu")
# T4_points, T4_cells, T4_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00002.vtu")
# T5_points, T5_cells, T5_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00001.vtu")
# 从VTU文件加载数据
T1_points, T1_cells, T1_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00105.vtu")
T2_points, T2_cells, T2_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00104.vtu")
T3_points, T3_cells, T3_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00103.vtu")
T4_points, T4_cells, T4_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00102.vtu")
T5_points, T5_cells, T5_valu = MeshTools.load_vtu_mesh(data_dir + "meshu00101.vtu")

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
VisualizationTools.mesh_with_gauss(T5_points, T5_cells, save_path=data_dir + "document_T5.pdf")