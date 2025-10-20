import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from scipy.spatial import Delaunay
from scipy.stats import qmc

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import vtk
from vtk.util.numpy_support import vtk_to_numpy


class VisualizationTools:
    """封装所有可视化相关函数"""
    @staticmethod
    def plot_2d_comparison(xx, yy, u_true, u_pred, save_path="Result2D.pdf"):
        fig = plt.figure(figsize=(18, 5))

        extent = [xx.min(), xx.max(), yy.min(), yy.max()]

        plt.subplot(1, 3, 1)
        plt.imshow(u_true, extent=extent, origin='lower', cmap='jet', aspect='equal')
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(r'Reference $u(x,y)$')

        plt.subplot(1, 3, 2)
        plt.imshow(u_pred, extent=extent, origin='lower', cmap='jet', aspect='equal')
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(r'Predicted $u(x,y)$')

        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(u_true - u_pred), extent=extent, origin='lower', cmap='jet', aspect='equal')
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Absolute error')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.close()

    @staticmethod
    def plot_3d_comparison(xx, yy, u_true, u_pred, save_path="Result3D.pdf"):
        fig = plt.figure(figsize=(18, 5))

        # 子图1: 真实解
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(xx, yy, u_true, cmap='jet', edgecolor='none')
        # ax1.view_init(elev=90, azim=-90)
        ax1.set_title('Exact')
        fig.colorbar(surf1, ax=ax1, shrink=0.5)

        # 子图2: 预测解
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(xx, yy, u_pred, cmap='jet', edgecolor='none')
        # ax2.view_init(elev=90, azim=-90)
        ax2.set_title('Prediction')
        fig.colorbar(surf2, ax=ax2, shrink=0.5)

        # 子图3: 误差
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(xx, yy, np.abs(u_true - u_pred), cmap='jet', edgecolor='none')
        # ax3.view_init(elev=90, azim=-90)
        ax3.set_title('Error')
        fig.colorbar(surf3, ax=ax3, shrink=0.5)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.close()

    @staticmethod
    def plot_solution_comparison(points, cells, true_solution, predicted_solution, save_path="comparison.pdf"):
        """
        绘制三角形网格上的真解、预测解和绝对误差对比图
        
        参数:
            points: 节点坐标数组，形状为 (n_nodes, 2)
            cells: 三角形单元连接关系，形状为 (n_cells, 3)
            true_solution: 真实解值，形状为 (n_nodes,)
            predicted_solution: 预测解值，形状为 (n_nodes,)
            save_path: 结果保存路径(默认"comparison.pdf")
        """
        triangulation = tri.Triangulation(points[:, 0], points[:, 1], cells)
        abs_error = np.abs(true_solution - predicted_solution)
        
        # 设置固定字体大小
        title_fontsize = 18
        tick_fontsize = 18
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 设置全局字体大小
        plt.rcParams.update({'font.size': tick_fontsize})
        
        # 真解子图
        t1 = ax1.tripcolor(triangulation, true_solution, shading='gouraud', cmap='jet')
        fig.colorbar(t1, ax=ax1)
        ax1.set_title('True Solution', fontsize=title_fontsize)
        ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # 预测解子图
        t2 = ax2.tripcolor(triangulation, predicted_solution, shading='gouraud', cmap='jet')
        fig.colorbar(t2, ax=ax2)
        ax2.set_title('Predicted Solution', fontsize=title_fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # 绝对误差子图
        t3 = ax3.tripcolor(triangulation, abs_error, shading='gouraud', cmap='jet')
        fig.colorbar(t3, ax=ax3)
        ax3.set_title('Absolute Error', fontsize=title_fontsize)
        ax3.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

    @staticmethod
    def plot_point_scalar_field(points, values, save_path="scatter.pdf"):
        """绘制二维标量场的散点图"""
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(points[:, 0], points[:, 1], c=values, cmap='jet', 
                        s=1, edgecolors='none')
        plt.colorbar(sc, label="Scalar Value")
        plt.title("Scatter Plot of Field")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
    @staticmethod    
    def visualize_solution(node, elem, u):
        """高质量可视化解决方案
        Args:
            node: 节点坐标数组 (N,2)
            elem: 单元连接数组 (M,3)
            u: 解向量 (N,)
            filename: 保存文件名（可选）
        """
        # 创建三角剖分
        triangulation = tri.Triangulation(node[:, 0], node[:, 1], elem)
        plt.figure(figsize=(10, 8))
        plt.gcf().set_facecolor('white')
        
        levels = np.linspace(u.min(), u.max(), 256)
        plt.tricontourf(triangulation, u, levels=levels, cmap='jet', extend='both')
        
        cbar = plt.colorbar(label='Solution Value', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
    @staticmethod
    def plot_training_points(X_res, X_bcs, save_path: str = 'train_points.pdf'):
        """
        可视化训练点分布
        
        参数:
            X_res: 残差点坐标 (N,2)
            X_bcs: 边界点坐标 (M,2) 
            save_path: PDF保存路径
        """
        plt.figure(figsize=(6, 5))
        plt.scatter(X_bcs[:, 0], X_bcs[:, 1], color='r', s=8, label='Boundary points')
        plt.scatter(X_res[:, 0], X_res[:, 1], color='b', s=8, label='Collocation points')
        plt.legend()
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.close()

    @staticmethod
    def mesh_with_gauss(points, cells, gauss_points=None, zoom_area=None, save_path="mesh.pdf"):
        """可视化网格，可选显示高斯点和局部放大图"""
        triangulation = tri.Triangulation(points[:, 0], points[:, 1], cells)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制主网格
        ax.set_facecolor('#80e673')
        ax.triplot(triangulation, 'k-', linewidth=0.8, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        
        # 处理L型区域
        VisualizationTools._add_l_shape_mask(ax, points, cells)
    
        # 绘制高斯点
        if gauss_points is not None:
            ax.scatter(gauss_points[:, 0], gauss_points[:, 1], 
                      color='red', s=5, zorder=2)
    
        # 添加局部放大图
        if zoom_area is not None:
            VisualizationTools._add_zoom_inset(ax, triangulation, gauss_points, zoom_area, points)
        
        # plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

    @staticmethod
    def _add_l_shape_mask(ax, points, cells):
        """为L型区域添加白色遮罩"""
        # 1. 判断是否为 L 型区域（右下角重心数量很少）
        tri_centers = np.mean(points[cells], axis=1)
        x_mid = 0.5 * (points[:, 0].min() + points[:, 0].max())
        y_mid = 0.5 * (points[:, 1].min() + points[:, 1].max())
        
        corner_mask = (tri_centers[:, 0] >= x_mid) & (tri_centers[:, 1] <= y_mid)
        if np.sum(corner_mask) < 10:  # 如果是L型区域
            delta = 3e-3    # 缩小遮罩范围，避免完全覆盖边线
            x0, x1 = points[:, 0].min(), points[:, 0].max()
            y0, y1 = points[:, 1].min(), points[:, 1].max()
            
            notch_coords = np.array([
                [x_mid + delta, y0],
                [x1, y0],
                [x1, y_mid - delta],
                [x_mid + delta, y_mid - delta]
            ])
            notch_patch = plt.Polygon(notch_coords, color='white', zorder=1)
            ax.add_patch(notch_patch)

    @staticmethod
    def _add_zoom_inset(ax, triangulation, gauss_points, zoom_area, points):
        """添加局部放大图"""
        ox, oy, sx, sy = zoom_area
        x0, x1 = points[:, 0].min(), points[:, 0].max()
        y0, y1 = points[:, 1].min(), points[:, 1].max()
        xlen, ylen = x1 - x0, y1 - y0
        xmid, ymid = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        
        xcenter = xmid + ox * xlen
        ycenter = ymid + oy * ylen
        xmin = xcenter - 0.5 * sx * xlen
        xmax = xcenter + 0.5 * sx * xlen
        ymin = ycenter - 0.5 * sy * ylen
        ymax = ycenter + 0.5 * sy * ylen
        
        ax_inset = inset_axes(ax, width="45%", height="45%", loc='upper right',
                             bbox_to_anchor=(0.0, 0.0, 0.97, 0.97),
                             bbox_transform=ax.transAxes)
        ax_inset.set_facecolor('#80e673')
        ax_inset.triplot(triangulation, 'k-', linewidth=1, alpha=1.0)
        
        if gauss_points is not None:
            ax_inset.scatter(gauss_points[:, 0], gauss_points[:, 1], 
                           color='red', s=10, zorder=2)
        
        # 隐藏子图坐标轴
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        
         # 设置放大范围
        ax_inset.set_xlim(xmin, xmax)
        ax_inset.set_ylim(ymin, ymax)
        ax_inset.grid(True, linestyle=':', alpha=0.9)
        
        for spine in ax_inset.spines.values():
            spine.set_linewidth(3)      # 边框线宽
            spine.set_color('White')    # 边框颜色
            spine.set_alpha(1.0)       # 不透明度
        
        # 添加连接线
        mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="White",
                  linestyle="-", alpha=1.0, linewidth=3)
       
        
class MeshTools:
    """封装所有网格处理相关函数"""

    @staticmethod
    def load_vtu_mesh(path):
        """
        读取 VTU 文件，并直接返回第一个变量的值（不是字典）
        节点坐标和节点函数值只返回前两列
        
        返回:
            tuple: (节点坐标数组(前两列), 单元连接性数组, 第一个变量的数组(前两列), 变量名列表)
        """
        # 读取文件
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(path)
        reader.Update()
        grid = reader.GetOutput()

        # 节点坐标 (Nx3)，只取前两列
        points = grid.GetPoints()
        node_coords = vtk_to_numpy(points.GetData())[:, :2]  # 只保留x,y坐标

        # 单元连接性 (Mxk)
        cells = grid.GetCells()
        connectivity = vtk_to_numpy(cells.GetConnectivityArray())
        offsets = vtk_to_numpy(cells.GetOffsetsArray())
        k = offsets[1] - offsets[0] if len(offsets) > 1 else len(connectivity)
        cell_connectivity = connectivity.reshape(-1, k)

        # 节点数据（直接返回第一个变量）
        point_data = grid.GetPointData()
        first_var_array = None
        
        if point_data.GetNumberOfArrays() > 0:
            # 获取第一个变量数组
            first_var_array = vtk_to_numpy(point_data.GetArray(0))
            # 如果数组是二维且列数>2，则只取前两列
            if first_var_array.ndim == 2 and first_var_array.shape[1] > 2:
                first_var_array = first_var_array[:, :2]
        
        return node_coords, cell_connectivity, first_var_array

    @staticmethod
    def trans_cartesian(node, elem, Lambda):
        """将重心坐标转换为直角坐标"""
        node1 = node[elem[:, 0], :]
        node2 = node[elem[:, 1], :]
        node3 = node[elem[:, 2], :]
        
        gaussPoints = np.einsum('ij,k->ikj', node1, Lambda[:, 0]) + \
                      np.einsum('ij,k->ikj', node2, Lambda[:, 1]) + \
                      np.einsum('ij,k->ikj', node3, Lambda[:, 2])
                      
        # 排列方式为[所有单元的第一个高斯点, 所有单元的第二个高斯点, 所有单元的第三个高斯点, ...]
        gaussPoints = gaussPoints.transpose(1, 0, 2).reshape(-1, 2)
        
        ve2 = node1 - node3
        ve3 = node2 - node1
        area = 0.5 * np.abs(-ve3[:, 0] * ve2[:, 1] + ve3[:, 1] * ve2[:, 0])
        
        return gaussPoints, area
    
    @staticmethod
    def convert_mesh_group(points, cells, values, dtype=torch.float64, device='cpu'):
        """
        将单组 mesh 数据 (p, c, v) 转为 torch tensor。

        参数:
            points: ndarray, 形如 [N, 2]，网格点坐标
            cells: ndarray, 形如 [M, K]，单元连接关系
            values: ndarray, 形如 [N, 1] 或 [N]，节点对应值
            dtype: torch.float, 点和数值的精度
            device: torch.device, 默认为 'cpu'

        返回:
            X: torch.tensor
            cells: torch.LongTensor
            y: torch.tensor
        """
        X = torch.tensor(points, dtype=dtype).to(device)
        C = torch.tensor(cells, dtype=torch.long).to(device)
        y = torch.tensor(values, dtype=dtype).to(device)
        return X, C, y
       
    @staticmethod
    def get_quadrature_rule(dimension, order):
        # 处理一维的情况
        if dimension == 1:
            if order > 10:
                order = 10
            if order == 1:
                A = np.array([[0, 2.0000000000000000000000000]])
            elif order == 2:
                A = np.array([[0.5773502691896257645091488, 1.0000000000000000000000000],
                            [-0.5773502691896257645091488, 1.0000000000000000000000000]])
            elif order == 3:
                A = np.array([[0, 0.8888888888888888888888889],
                            [0.7745966692414833770358531, 0.5555555555555555555555556],
                            [-0.7745966692414833770358531, 0.5555555555555555555555556]])
            elif order == 4:
                A = np.array([[0.3399810435848562648026658, 0.6521451548625461426269361],
                            [0.8611363115940525752239465, 0.3478548451374538573730639],
                            [-0.3399810435848562648026658, 0.6521451548625461426269361],
                            [-0.8611363115940525752239465, 0.3478548451374538573730639]])
            elif order == 5:
                A = np.array([[0, 0.5688888888888888888888889],
                            [0.5384693101056830910363144, 0.4786286704993664680412915],
                            [0.9061798459386639927976269, 0.2369268850561890875142640],
                            [-0.5384693101056830910363144, 0.4786286704993664680412915],
                            [-0.9061798459386639927976269, 0.2369268850561890875142640]])
            elif order == 6:
                A = np.array([[0.2386191860831969086305017, 0.4679139345726910473898703],
                            [0.6612093864662645136613996, 0.3607615730481386075698335],
                            [0.9324695142031520278123016, 0.1713244923791703450402961],
                            [-0.2386191860831969086305017, 0.4679139345726910473898703],
                            [-0.6612093864662645136613996, 0.3607615730481386075698335],
                            [-0.9324695142031520278123016, 0.1713244923791703450402961]])
            elif order == 7:
                A = np.array([[0, 0.4179591836734693877551020],
                            [0.4058451513773971669066064, 0.3818300505051189449503698],
                            [0.7415311855993944398638648, 0.2797053914892766679014678],
                            [0.9491079123427585245261897, 0.1294849661688696932706114],
                            [-0.4058451513773971669066064, 0.3818300505051189449503698],
                            [-0.7415311855993944398638648, 0.2797053914892766679014678],
                            [-0.9491079123427585245261897, 0.1294849661688696932706114]])
            elif order == 8:
                A = np.array([[0.1834346424956498049394761, 0.3626837833783619829651504],
                            [0.5255324099163289858177390, 0.3137066458778872873379622],
                            [0.7966664774136267395915539, 0.2223810344533744705443560],
                            [0.9602898564975362316835609, 0.1012285362903762591525314],
                            [-0.1834346424956498049394761, 0.3626837833783619829651504],
                            [-0.5255324099163289858177390, 0.3137066458778872873379622],
                            [-0.7966664774136267395915539, 0.2223810344533744705443560],
                            [-0.9602898564975362316835609, 0.1012285362903762591525314]])
            elif order == 9:
                A = np.array([[0, 0.3302393550012597631645251],
                            [0.3242534234038089290385380, 0.3123470770400028400686304],
                            [0.6133714327005903973087020, 0.2606106964029354623187429],
                            [0.8360311073266357942994298, 0.1806481606948574040584720],
                            [0.9681602395076260898355762, 0.0812743883615744119718922],
                            [-0.3242534234038089290385380, 0.3123470770400028400686304],
                            [-0.6133714327005903973087020, 0.2606106964029354623187429],
                            [-0.8360311073266357942994298, 0.1806481606948574040584720],
                            [-0.9681602395076260898355762, 0.0812743883615744119718922]])
            elif order == 10:
                A = np.array([[0.1488743389816312108848260, 0.2955242247147528701738930],
                            [0.4333953941292471907992659, 0.2692667193099963550912269],
                            [0.6794095682990244062343274, 0.2190863625159820439955349],
                            [0.8650633666889845107320967, 0.1494513491505805931457763],
                            [0.9739065285171717200779640, 0.0666713443086881375935688],
                            [-0.1488743389816312108848260, 0.2955242247147528701738930],
                            [-0.4333953941292471907992659, 0.2692667193099963550912269],
                            [-0.6794095682990244062343274, 0.2190863625159820439955349],
                            [-0.8650633666889845107320967, 0.1494513491505805931457763],
                            [-0.9739065285171717200779640, 0.0666713443086881375935688]])
            Lambda = np.array([(A[:, 0] + 1) / 2, 1 - (A[:, 0] + 1) / 2]).T
            Weight = A[:, 1] / 2
            
        elif dimension == 2:
            if order > 9:
                order = 9
            if order == 1:  # Order 1, nQuad 1
                Lambda = np.array([[1/3, 1/3, 1/3]]) 
                Weight = 1
            elif order == 2:  # Order 2, nQuad 3
                Lambda = np.array([[2/3, 1/6, 1/6],
                                    [1/6, 2/3, 1/6],
                                    [1/6, 1/6, 2/3]])
                Weight = np.array([1/3, 1/3, 1/3])
            elif order == 3:  # Order 3, nQuad 4
                Lambda = np.array([[1/3, 1/3, 1/3],
                                    [0.6, 0.2, 0.2],
                                    [0.2, 0.6, 0.2],
                                    [0.2, 0.2, 0.6]])
                Weight = np.array([-27/48, 25/48, 25/48, 25/48])
            elif order == 4:  # Order 4, nQuad 6
                Lambda = np.array([[0.108103018168070, 0.445948490915965, 0.445948490915965],
                                    [0.445948490915965, 0.108103018168070, 0.445948490915965],
                                    [0.445948490915965, 0.445948490915965, 0.108103018168070],
                                    [0.816847572980459, 0.091576213509771, 0.091576213509771],
                                    [0.091576213509771, 0.816847572980459, 0.091576213509771],
                                    [0.091576213509771, 0.091576213509771, 0.816847572980459]])
                Weight = np.array([0.223381589678011, 0.223381589678011, 0.223381589678011,
                                0.109951743655322, 0.109951743655322, 0.109951743655322])
            elif order == 5:  # Order 5, nQuad 7
                Lambda = np.array([[1/3, 1/3, 1/3],
                                    [0.059715871789770, 0.470142064105115, 0.470142064105115],
                                    [0.470142064105115, 0.059715871789770, 0.470142064105115],
                                    [0.470142064105115, 0.470142064105115, 0.059715871789770],
                                    [0.797426985353087, 0.101286507323456, 0.101286507323456],
                                    [0.101286507323456, 0.797426985353087, 0.101286507323456],
                                    [0.101286507323456, 0.101286507323456, 0.797426985353087]])
                Weight = np.array([0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506,
                                0.125939180544827, 0.125939180544827, 0.125939180544827])
            elif order == 6:  # Order 6, nQuad 12
                A = np.array([[0.249286745170910, 0.249286745170910, 0.116786275726379],
                            [0.249286745170910, 0.501426509658179, 0.116786275726379],
                            [0.501426509658179, 0.249286745170910, 0.116786275726379],
                            [0.063089014491502, 0.063089014491502, 0.050844906370207],
                            [0.063089014491502, 0.873821971016996, 0.050844906370207],
                            [0.873821971016996, 0.063089014491502, 0.050844906370207],
                            [0.310352451033784, 0.636502499121399, 0.082851075618374],
                            [0.636502499121399, 0.053145049844817, 0.082851075618374],
                            [0.053145049844817, 0.310352451033784, 0.082851075618374],
                            [0.636502499121399, 0.310352451033784, 0.082851075618374],
                            [0.310352451033784, 0.053145049844817, 0.082851075618374],
                            [0.053145049844817, 0.636502499121399, 0.082851075618374]])
                Lambda = np.column_stack((A[:, 0], A[:, 1], 1 - np.sum(A[:, :2], axis=1)))
                Weight = A[:, 2]
            elif order == 7:  # Order 7, nQuad 13
                A = np.array([[0.333333333333333, 0.333333333333333, -0.149570044467682],
                            [0.260345966079040, 0.260345966079040, 0.175615257433208],
                            [0.260345966079040, 0.479308067841920, 0.175615257433208],
                            [0.479308067841920, 0.260345966079040, 0.175615257433208],
                            [0.065130102902216, 0.065130102902216, 0.053347235608838],
                            [0.065130102902216, 0.869739794195568, 0.053347235608838],
                            [0.869739794195568, 0.065130102902216, 0.053347235608838],
                            [0.312865496004874, 0.638444188569810, 0.077113760890257],
                            [0.638444188569810, 0.048690315425316, 0.077113760890257],
                            [0.048690315425316, 0.312865496004874, 0.077113760890257],
                            [0.638444188569810, 0.312865496004874, 0.077113760890257],
                            [0.312865496004874, 0.048690315425316, 0.077113760890257],
                            [0.048690315425316, 0.638444188569810, 0.077113760890257]])
                Lambda = np.column_stack((A[:, 0], A[:, 1], 1 - np.sum(A[:, :2], axis=1)))
                Weight = A[:, 2]
            elif order == 8:  # Order 8, nQuad 16
                A = np.array([[0.333333333333333, 0.333333333333333, 0.144315607677787],
                            [0.081414823414554, 0.459292588292723, 0.095091634267285],
                            [0.459292588292723, 0.081414823414554, 0.095091634267285],
                            [0.459292588292723, 0.459292588292723, 0.095091634267285],
                            [0.658861384496480, 0.170569307751760, 0.103217370534718],
                            [0.170569307751760, 0.658861384496480, 0.103217370534718],
                            [0.170569307751760, 0.170569307751760, 0.103217370534718],
                            [0.898905543365938, 0.050547228317031, 0.032458497623198],
                            [0.050547228317031, 0.898905543365938, 0.032458497623198],
                            [0.050547228317031, 0.050547228317031, 0.032458497623198],
                            [0.008394777409958, 0.263112829634638, 0.027230314174435],
                            [0.008394777409958, 0.728492392955404, 0.027230314174435],
                            [0.263112829634638, 0.008394777409958, 0.027230314174435],
                            [0.728492392955404, 0.008394777409958, 0.027230314174435],
                            [0.263112829634638, 0.728492392955404, 0.027230314174435],
                            [0.728492392955404, 0.263112829634638, 0.027230314174435]])
                Lambda = np.column_stack((A[:, 0], A[:, 1], 1 - np.sum(A[:, :2], axis=1)))
                Weight = A[:, 2]
            elif order == 9:  # Order 9, nQuad 19
                A = np.array([[0.333333333333333, 0.333333333333333, 0.097135796282799],
                            [0.020634961602525, 0.489682519198738, 0.031334700227139],
                            [0.489682519198738, 0.020634961602525, 0.031334700227139],
                            [0.489682519198738, 0.489682519198738, 0.031334700227139],
                            [0.125820817014127, 0.437089591492937, 0.07782754100474],
                            [0.437089591492937, 0.125820817014127, 0.07782754100474],
                            [0.437089591492937, 0.437089591492937, 0.07782754100474],
                            [0.623592928761935, 0.188203535619033, 0.079647738927210],
                            [0.188203535619033, 0.623592928761935, 0.079647738927210],
                            [0.188203535619033, 0.188203535619033, 0.079647738927210],
                            [0.910540973211095, 0.044729513394453, 0.025577675658698],
                            [0.044729513394453, 0.910540973211095, 0.025577675658698],
                            [0.044729513394453, 0.044729513394453, 0.025577675658698],
                            [0.036838412054736, 0.221962989160766, 0.043283539377289],
                            [0.036838412054736, 0.741198598784498, 0.043283539377289],
                            [0.221962989160766, 0.036838412054736, 0.043283539377289],
                            [0.741198598784498, 0.036838412054736, 0.043283539377289],
                            [0.221962989160766, 0.741198598784498, 0.043283539377289],
                            [0.741198598784498, 0.221962989160766, 0.043283539377289]])
                Lambda = np.column_stack((A[:, 0], A[:, 1], 1 - np.sum(A[:, :2], axis=1)))
                Weight = A[:, 2]
        else:
            print("目前只有一维和二维的高斯点及其权重, 第一输入只能选择1或者2") 
        return Lambda, Weight

    @staticmethod
    def generate_oversample_gauss(triangles, coords, values, Lambda):
        new_X, new_y = [], []
        for tri in triangles:
            pts = coords[tri]  # triangle vertex coordinates: shape (3, 2)
            for lam in Lambda:
                l1, l2, l3 = lam
                xy = l1 * pts[0] + l2 * pts[1] + l3 * pts[2]
                val = l1 * values[tri[0]] + l2 * values[tri[1]] + l3 * values[tri[2]]
                new_X.append(xy)
                new_y.append(val)
        return np.array(new_X), np.array(new_y)
    
    @staticmethod
    def make_L_shaped_mesh(domain_x, domain_y, num_boundary_points, num_interior_points):
        import numpy as np
        from scipy.spatial import Delaunay

        x_min, x_max = domain_x
        y_min, y_max = domain_y
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        N = num_boundary_points
        N_half = N // 2

        # 六条边，按逆时针走（注意短边只采样 N_half 个点）
        left        = np.column_stack((np.full(N, x_min), np.linspace(y_min, y_max, N)))
        top         = np.column_stack((np.linspace(x_min, x_max, N), np.full(N, y_max)))
        right_upper = np.column_stack((np.linspace(x_mid, x_max, N_half), np.full(N_half, y_mid)))
        bottom      = np.column_stack((np.linspace(x_min, x_mid, N_half), np.full(N_half, y_min)))
        right_lower = np.column_stack((np.full(N_half, x_max), np.linspace(y_mid, y_max, N_half)))
        notch_left  = np.column_stack((np.full(N_half, x_mid), np.linspace(y_min, y_mid, N_half)))

        boundary_points = np.vstack([left, top, right_upper, bottom, right_lower, notch_left])
        boundary_points = np.unique(boundary_points, axis=0)

        # === 3. 内部点（剔除右下角）
        buffer = 0.003 * min(x_max - x_min, y_max - y_min)  # 可调节的边界缓冲距离
        interior_points = np.random.rand(num_interior_points, 2)
        interior_points[:, 0] = interior_points[:, 0] * (x_max - x_min) + x_min
        interior_points[:, 1] = interior_points[:, 1] * (y_max - y_min) + y_min
        mask = ~((interior_points[:, 0] > x_mid-buffer) & (interior_points[:, 1] < y_mid+buffer))
        interior_points = interior_points[mask]

        # === 4. 构造初始三角剖分 + 优化
        points = np.vstack([boundary_points, interior_points])
        tri = Delaunay(points)

        for k in range(4):
            for i in range(len(points)):
                if not np.any(np.all(points[i] == boundary_points, axis=1)):
                    neighbors = np.unique(tri.simplices[np.any(tri.simplices == i, axis=1)])
                    neighbors = neighbors[neighbors != i]
                    points[i] = np.mean(points[neighbors], axis=0)

        tri = Delaunay(points)

        # === 5. 删除右下角的单元（剔除重心落在右下角的）
        centers = np.mean(points[tri.simplices], axis=1)
        keep = ~((centers[:, 0] >= x_mid) & (centers[:, 1] <= y_mid))
        cells = tri.simplices[keep]
        
        return points, cells
    
    @staticmethod
    def make_uniform_mesh(domain_x, domain_y, num_boundary_points, num_interior_points=None):
        """
        生成均匀的三角形网格（保持与原函数相同的接口）
        
        参数:
            domain_x: x方向的边界 [x_min, x_max]
            domain_y: y方向的边界 [y_min, y_max]
            num_interior_points: 忽略此参数（仅为兼容性保留）
            num_boundary_points: 每条边上的点数(默认为100)
            
        返回:
            points: 节点坐标数组 (n, 2)
            cells: 三角形单元连接关系 (m, 3)
        """
        # 生成均匀的网格点（使用num_boundary_points作为每条边的点数）
        x = np.linspace(domain_x[0], domain_x[1], num_boundary_points)
        y = np.linspace(domain_y[0], domain_y[1], num_boundary_points)
        X, Y = np.meshgrid(x, y)
        
        # 将网格点展平为一维数组
        points = np.column_stack((X.flatten(), Y.flatten()))
        
        # 创建Delaunay三角剖分
        tri = Delaunay(points)
        
        # 提取点和单元信息
        points = tri.points
        elements = tri.simplices
        
        return points, elements

    @staticmethod
    def make_mesh(domain_x, domain_y, num_boundary_points, num_interior_points):
        # 生成边界点
        BX, BY = np.meshgrid([domain_y[0], domain_y[1]], np.linspace(domain_x[0], domain_x[1], num_boundary_points))
        LY, LX = np.meshgrid([domain_x[0], domain_x[1]], np.linspace(domain_y[0], domain_y[1], num_boundary_points))
        boundary_points = np.unique(np.vstack([np.column_stack((LX.flatten(), LY.flatten())),
                                            np.column_stack((BX.flatten(), BY.flatten()))]), axis=0)
    
        # 生成内部点
        interior_points = np.random.rand(num_interior_points, 2)
        interior_points[:, 0] = interior_points[:, 0] * (domain_x[1] - domain_x[0]) + domain_x[0]
        interior_points[:, 1] = interior_points[:, 1] * (domain_y[1] - domain_y[0]) + domain_y[0]

        # 组合边界点和内部点
        points = np.vstack([boundary_points, interior_points])

        # 优化网格内部节点
        tri = Delaunay(points)
        for k in range(4):  # 减少循环次数以提高效率
            for i in range(len(points)):
                if not np.any(np.all(points[i] == boundary_points, axis=1)):  # 如果不是边界点
                    neighbors = np.unique(tri.simplices[np.any(tri.simplices == i, axis=1)])
                    neighbors = neighbors[neighbors != i]  # 移除自身
                    points[i] = np.mean(points[neighbors], axis=0)  # 更新点为邻居点的平均值

        # 重新创建三角剖分
        tri = Delaunay(points)

        # 输出单元信息
        points = tri.points
        elements = tri.simplices

        return points, elements
        
    
class NumericalTools:
    """封装所有数值计算相关函数"""
    @staticmethod
    def calculate_errors(u_true, u_pred):
        """计算相对L2误差和绝对平均误差"""
        relative_l2_error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)        
        absolute_mean_error = np.mean(np.abs(u_true - u_pred))
        return relative_l2_error, absolute_mean_error

    @staticmethod
    def num_integral_nn(cart_gauss_point, nn_model, weight, area):
        """神经网络数值积分"""
        H = nn_model(cart_gauss_point)
        U = H @ coef_solution
        U = U.reshape(len(weight), -1)
        return area * (weight @ U)

    @staticmethod
    def num_integral_fem(bary_gauss_point, cell, value, weight, area):
        """有限元数值积分"""
        value = value[:, 0]
        cell2dof = value[cell]
        U = bary_gauss_point @ cell2dof.T
        return area * (weight @ U)

class Dataset:
    def __init__(self, domain, buffer=1e-2):
        self.domain = domain
        self.buffer = buffer
    
    def adjust_domain(self):
        xmin, xmax, ymin, ymax = self.domain
        buffer = self.buffer
        return xmin + buffer, xmax - buffer, ymin + buffer, ymax - buffer
    
    def train_data_square(self, num_boundary_point, num_collocation_point, verbose=None):        
        X_res = self.sample_uniform(num_collocation_point)
        # X_res = self.sample_random(num_collocation_point)
        # X_res = self.sample_sobel(num_collocation_point)
        # X_res = self.sample_LatinHypercube(num_collocation_point)
        xmin, xmax, ymin, ymax = self.domain      
        X_bcs = np.vstack([
            np.hstack((xmin * np.ones((num_boundary_point // 4, 1)), np.linspace(ymin, ymax, num_boundary_point // 4)[:, None])),
            np.hstack((xmax * np.ones((num_boundary_point // 4, 1)), np.linspace(ymin, ymax, num_boundary_point // 4)[:, None])),
            np.hstack((np.linspace(xmin, xmax, num_boundary_point // 4)[:, None], ymin * np.ones((num_boundary_point // 4, 1)))),
            np.hstack((np.linspace(xmin, xmax, num_boundary_point // 4)[:, None], ymax * np.ones((num_boundary_point // 4, 1))))
        ])
        return X_res, X_bcs
    
    def train_data_Lshape(self, num_boundary_point, num_collocation_point, verbose=None):       
        X_res = self.sample_uniform(num_collocation_point)
        # X_res = self.sample_random(num_collocation_point)
        # X_res = self.sample_sobel(num_collocation_point)
        # X_res = self.sample_LatinHypercube(num_collocation_point)
        xmin, xmax, ymin, ymax = self.domain
        indices = ~((X_res[:, 0] >= (xmin + xmax) / 2) & (X_res[:, 1] <= (ymin + ymax) / 2))
        X_res = X_res[indices]
        X_bcs = np.vstack([
            np.hstack((xmin * np.ones((num_boundary_point // 4, 1)), np.linspace(ymin, ymax, num_boundary_point // 4)[:, None])),
            np.hstack((xmax * np.ones((num_boundary_point // 8, 1)), np.linspace((ymin + ymax) / 2, ymax, num_boundary_point // 8)[:, None])),
            np.hstack(((xmin + xmax) / 2 * np.ones((num_boundary_point // 8, 1)), np.linspace(ymin, (ymin + ymax) / 2, num_boundary_point // 8)[:, None])),
            np.hstack((np.linspace(xmin, (xmin + xmax) / 2, num_boundary_point // 8)[:, None], ymin * np.ones((num_boundary_point // 8, 1)))),
            np.hstack((np.linspace((xmin + xmax) / 2, xmax, num_boundary_point // 8)[:, None], (ymin + ymax) / 2 * np.ones((num_boundary_point // 8, 1)))),
            np.hstack((np.linspace(xmin, xmax, num_boundary_point // 4)[:, None], ymax * np.ones((num_boundary_point // 4, 1))))
        ])
        return X_res, X_bcs
    
    # 这个有点问题, 三角形斜边上的点滤不干净
    def train_data_triangle(self, num_boundary_point, num_collocation_point, verbose=None):       
        X_res = self.sample_random(num_collocation_point)
        # X_res = self.sample_random(num_collocation_point)
        # X_res = self.sample_sobel(num_collocation_point)
        # X_res = self.sample_LatinHypercube(num_collocation_point)
        xmin, xmax, ymin, ymax = self.domain
        mask = (X_res[:, 1] < np.round((ymin - ymax) / (xmax - xmin) * (X_res[:, 0] - xmin) + ymax, 5))
        X_res = X_res[mask]
        X_bcs = np.vstack([
            np.linspace(np.array([xmin, ymin]), np.array([xmin, ymax]), num_boundary_point // 3),
            np.linspace(np.array([xmin, ymin]), np.array([xmax, ymin]), num_boundary_point // 3),
            np.linspace(np.array([xmin, ymax]), np.array([xmax, ymin]), num_boundary_point // 3)
        ])        
        return X_res, X_bcs

    def eval_data(self, num_test_point=40000):
        X = self.sample_uniform(num_test_point)
        num_points = X.shape[0]
        grid_size = int(np.sqrt(num_points))
        xx = X[:, 0].reshape((grid_size, grid_size))
        yy = X[:, 1].reshape((grid_size, grid_size))
        return X, xx, yy
    
    def sample_uniform(self, num_collocation_point):
        xmin, xmax, ymin, ymax = self.adjust_domain()
        x = np.linspace(xmin, xmax, int(np.sqrt(num_collocation_point)) + 3)
        y = np.linspace(ymin, ymax, int(np.sqrt(num_collocation_point)) + 3)
        xx, yy = np.meshgrid(x, y)
        X = np.concatenate([xx.reshape((-1, 1)), yy.reshape((-1, 1))], axis=1)
        return X

    def sample_random(self, num_collocation_point):
        xmin, xmax, ymin, ymax = self.adjust_domain()
        xx = np.random.uniform(xmin, xmax, num_collocation_point)
        yy = np.random.uniform(ymin, ymax, num_collocation_point)
        X = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)
        return X

    def sample_sobel(self, num_collocation_point):
        xmin, xmax, ymin, ymax = self.adjust_domain()
        sampler = qmc.Sobol(d=2, scramble=False)
        X = sampler.random(n=num_collocation_point)
        X = np.array([xmin, ymin]) + (np.array([xmax - xmin, ymax - ymin]) * X)
        return X
    
    def sample_LatinHypercube(self, num_collocation_point):
        xmin, xmax, ymin, ymax = self.adjust_domain()
        sampler = qmc.LatinHypercube(d=2)
        points = sampler.random(n=num_collocation_point)
        X = np.array([xmin, ymin]) + (np.array([xmax - xmin, ymax - ymin]) * points)
        return X

class DataEnhance:
    def __init__(self, domain):
        self.domain = domain

    def nearby_point(self, Num_points=50, origin_radius=0.1):
        xmin, xmax, ymin, ymax = self.domain
        origin_point = [(xmin + xmax) / 2, (ymin + ymax) / 2]  # 区域中心点
        theta = np.random.uniform(low=0, high=2*np.pi, size=Num_points)  # 随机角度
        r = np.sqrt(np.random.uniform(low=0, high=origin_radius**2, size=Num_points))  # 随机半径
        X = np.array([r * np.cos(theta) + origin_point[0], r * np.sin(theta) + origin_point[1]]).T
        return X

    def nearby_axis(self, Num_points=1000):
        xmin, xmax, ymin, ymax = self.domain
        range = 0.1  # 定义中垂线附近的范围
        x_coords_vertical = np.random.uniform((xmin + xmax) / 2 - range, (xmin + xmax) / 2 + range, Num_points // 2)
        y_coords_vertical = np.random.uniform(ymin, ymax, Num_points // 2)
        x_coords_horizontal = np.random.uniform(xmin, xmax, Num_points // 2)
        y_coords_horizontal = np.random.uniform((ymin + ymax) / 2 - range, (ymin + ymax) / 2 + range, Num_points // 2)
        points_vertical = np.column_stack((x_coords_vertical, y_coords_vertical))
        points_horizontal = np.column_stack((x_coords_horizontal, y_coords_horizontal))
        X = np.vstack((points_vertical, points_horizontal)) 
        return X

    def nearby_boundary(self, Num_points=4000):
        xmin, xmax, ymin, ymax = self.domain
        buffer = 0.1  # 在边界内0.1范围内生成点
        quarter_points = Num_points // 4  # 分为四部分，每个边界相同数量的点

        # 生成顶部和底部边界点
        x_top_bottom = np.random.uniform(low=xmin, high=xmax, size=quarter_points * 2)
        y_top = np.random.uniform(ymax - buffer, ymax, quarter_points)
        y_bottom = np.random.uniform(ymin, ymin + buffer, quarter_points)

        # 生成左侧和右侧边界点
        y_left_right = np.random.uniform(ymin, ymax, quarter_points * 2)
        x_left = np.random.uniform(xmin, xmin + buffer, quarter_points)
        x_right = np.random.uniform(xmax - buffer, xmax, quarter_points)

        # 将坐标组合成点数组
        top_bottom_points = np.column_stack((x_top_bottom, np.concatenate([y_top, y_bottom])))
        left_right_points = np.column_stack((np.concatenate([x_left, x_right]), y_left_right))

        # 合并所有点
        X = np.vstack((top_bottom_points, left_right_points))
        return X   
    
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    # import os
    # torch.set_num_threads(1)  # 限制为单线程，确保 determinism
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
