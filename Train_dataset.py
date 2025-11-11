import numpy as np

from utils_functions import Dataset
from utils_functions import VisualizationTools


data_dir = "/home/paper2/train_data/"  # 注意末尾加斜线

domain = (-1, 1, -1, 1)  # xmin, xmax, ymin, ymax
num_boundary = 200
num_collocation = 1000

# 初始化数据集
dataset = Dataset(domain)

X_res, X_bcs = dataset.train_data_square(num_boundary, num_collocation)
VisualizationTools.plot_training_points(X_res, X_bcs, save_path=data_dir + "dataset_square.pdf")

X_res, X_bcs = dataset.train_data_Lshape(num_boundary, num_collocation)
VisualizationTools.plot_training_points(X_res, X_bcs, save_path=data_dir + "dataset_Lshape.pdf")

X_res, X_bcs = dataset.train_data_triangle(num_boundary, num_collocation)
VisualizationTools.plot_training_points(X_res, X_bcs, save_path=data_dir + "dataset_triangle.pdf")

# 保存数据到字典
data = {
    'T1_p': X_res, 'T1_v': X_bcs
}
np.savez(data_dir + 'mesh_data.npz', **data)