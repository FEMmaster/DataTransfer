import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation
import os
import pyvista as pv
from typing import Tuple, List, Dict, Any, Callable, Optional
from scipy.spatial import Delaunay

import matplotlib.tri as tri
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ===================== 小工具函数 =====================
def generate_2d_meshes(x_domain, y_domain, N):
    """生成两套2D网格"""
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    
    # 生成规则网格
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x, y)
    points_A = np.column_stack([X.ravel(), Y.ravel()])
    tri_A = Delaunay(points_A)
    
    # 生成网格B：边界点 + 三角形中点
    boundary_mask = ((points_A[:, 0] == x_min) | (points_A[:, 0] == x_max) | 
                    (points_A[:, 1] == y_min) | (points_A[:, 1] == y_max))
    boundary_points = points_A[boundary_mask]
    
    triangles = points_A[tri_A.simplices]
    centroids = np.mean(triangles, axis=1)
    points_B = np.vstack([boundary_points, centroids])
    tri_B = Delaunay(points_B)
    
    return (
        {'nodes': points_A, 'triangles': tri_A.simplices},
        {'nodes': points_B, 'triangles': tri_B.simplices}
    )

def interpolate_tri(x_new, x_nodes, y_nodes, tri=None):
    """三角线性插值"""
    if tri is None:
        delaunay = Delaunay(x_nodes)
        tri = Triangulation(x_nodes[:, 0], x_nodes[:, 1], delaunay.simplices)
    
    lin = LinearTriInterpolator(tri, y_nodes)
    z = lin(x_new[:, 0], x_new[:, 1])
    return np.asarray(z)

def interpolate_rbf(x_new, x_nodes, y_nodes, L=1000, gamma=10.0):
    """RBF径向基函数插值"""
    x_nodes = np.atleast_2d(x_nodes)
    x_new = np.atleast_2d(x_new)
    n, d = x_nodes.shape
    L = min(L, n)
    
    # 选择中心点
    idx = np.random.choice(n, size=L, replace=False)
    C = x_nodes[idx]
    
    # 优化距离计算
    def compute_rbf_matrix(X, centers):
        dist_sq = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2)
        return np.exp(-gamma * dist_sq)
    
    H = compute_rbf_matrix(x_nodes, C)
    K = compute_rbf_matrix(x_new, C)
    
    w, *_ = np.linalg.lstsq(H, y_nodes, rcond=None)
    return K @ w

def plot_true_solution(func, save_dir, title_offset=0.9):
    """绘制真解图"""
    print("正在绘制真解图...")

    # 真值场栅格
    RES = 200
    xg = np.linspace(-1, 1, RES)
    yg = np.linspace(-1, 1, RES)
    Xg, Yg = np.meshgrid(xg, yg, indexing="xy")
    Ftrue = func(Xg, Yg)

    # 字体风格参数
    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 14
    cbar_fontsize = 16

    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'font.serif': ['Times New Roman'],
        'font.weight': 'normal',   # 不再全局加粗
    })

    # 创建真解图
    fig_true = plt.figure(figsize=(10, 10))
    ax_true = fig_true.add_subplot(111, projection="3d")
    ax_true.view_init(elev=20, azim=225)

    # 绘制真解表面
    surf_true = ax_true.plot_surface(
        Xg, Yg, Ftrue, cmap="viridis", alpha=0.9,
        antialiased=True, linewidth=0
    )

    # ===== 手动设置 colorbar 范围和刻度 =====
    cbar = fig_true.colorbar(
        surf_true, ax=ax_true, shrink=0.6, aspect=20, pad=0.05
    )
    # 强制颜色范围（例如最小值固定为 -2）
    cbar.mappable.set_clim(vmin=-2, vmax=np.nanmax(Ftrue))
    cbar.locator = LinearLocator(5)
    cbar.formatter = FormatStrFormatter('%.2f')
    cbar.update_ticks()

    # colorbar 样式
    cbar.ax.tick_params(
        labelsize=cbar_fontsize,
        width=1.0, length=5, direction='in'
    )
    
    # ===== 标题 =====
    ax_true.text2D(
        0.5, title_offset, "Reference Solution",
        transform=ax_true.transAxes,
        ha='center', va='bottom',
        fontsize=title_fontsize,
        fontweight='bold',
        family='serif'
    )

    # ===== 坐标轴刻度数量与样式 =====
    num_ticks = 5
    ax_true.set_xticks(np.linspace(xg.min(), xg.max(), num_ticks))
    ax_true.set_yticks(np.linspace(yg.min(), yg.max(), num_ticks))
    ax_true.set_zticks(np.linspace(Ftrue.min(), Ftrue.max(), num_ticks))
    ax_true.tick_params(axis='both', which='major',
                        direction='in', width=1.2, length=8,
                        labelsize=tick_fontsize, pad=5)

    # 设置平行投影
    set_parallel_projection(ax_true)

    # 保存真解图
    true_solution_path = os.path.join(save_dir, "true_solution.pdf")
    plt.savefig(true_solution_path, dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig_true)
    print(f"真解图已保存: {true_solution_path}")

def plot_error_fields(A_nodes, B_nodes, tri_A, tri_B, error_A, error_B, filename, method_name, iteration):
    """
    绘制二维平面误差图（双子图）
    - 左图: B→A Absolute Error (Mesh A)
    - 右图: A→B Absolute Error (Mesh B)
    """

    print(f"正在绘制二维误差图（{method_name}, iter={iteration}）...")

    # 全局字体控制
    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 14

    plt.rcParams.update({
        'font.size': tick_fontsize,
        'font.family': 'serif',
        'mathtext.fontset': 'stix'
    })

    # 统一坐标范围
    xmin = min(A_nodes[:, 0].min(), B_nodes[:, 0].min())
    xmax = max(A_nodes[:, 0].max(), B_nodes[:, 0].max())
    ymin = min(A_nodes[:, 1].min(), B_nodes[:, 1].min())
    ymax = max(A_nodes[:, 1].max(), B_nodes[:, 1].max())
    xticks = np.linspace(xmin, xmax, 5)
    yticks = np.linspace(ymin, ymax, 5)

    def _setup_subplot(ax, nodes, cells, data, title):
        """单个子图设置"""
        triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], cells)

        # 填色
        t = ax.tripcolor(triangulation, data, shading='gouraud', cmap='jet')

        # ✅ 绘制网格线（单元边界）
        ax.triplot(triangulation, color='k', linewidth=0.4, alpha=0.4)

        # colorbar 紧贴右侧，无 label
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(t, cax=cax)
        cbar.formatter = FuncFormatter(lambda x, _: f'{x:.2e}')
        cbar.locator = LinearLocator(numticks=5)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=tick_fontsize, direction='in', width=0.6, length=4)
        cbar.set_label("")  # ✅ 不显示标签文字

        # 坐标轴风格
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks[1:])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlabel('$x$', fontsize=label_fontsize, fontweight='bold', family='serif')
        ax.set_ylabel('$y$', fontsize=label_fontsize, fontweight='bold', family='serif')
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize,
                       direction='in', width=1.0, length=5)
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', family='serif', pad=10)
        ax.set_aspect('equal')

        return t

    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plt.subplots_adjust(wspace=0.20)

    # 左右子图标题包含 method_name 和 iteration
    title1 = f"{method_name} - Iter {iteration+1} - B→A"
    title2 = f"{method_name} - Iter {iteration+1} - A→B"

    # 绘制两个误差场
    _setup_subplot(ax1, A_nodes, tri_A, error_A, title1)
    _setup_subplot(ax2, B_nodes, tri_B, error_B, title2)

    # ✅ 保存路径格式化
    base_dir = os.path.dirname(filename)
    save_name = f"{method_name}_iter{iteration:02d}.pdf"
    save_path = os.path.join(base_dir, save_name)

    # 保存并关闭
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=300)
    plt.close(fig)

    print(f"✅ 保存误差图: {save_name}")
    
def plot_mesh(points, cells, zoom_area=None, save_dir=None, filename="mesh.pdf"):
    """
    绘制有限元网格，可选添加局部放大区域。
    
    参数：
        points : ndarray, shape (N, 2)
            节点坐标 (x, y)
        cells : ndarray, shape (M, 3)
            单元连接关系（节点索引）
        zoom_area : tuple (ox, oy, sx, sy), 可选
            放大区域参数：
                ox, oy 表示放大区域中心相对于整体网格中心的偏移比例；
                sx, sy 表示放大窗口在 x, y 方向上的尺寸比例。
        save_path : str
            保存路径（例如 "mesh.pdf"）
    """
    
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # 字体设置：Times 常规体
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.weight': 'normal'
    })
    
    # 生成三角剖分
    triangulation = tri.Triangulation(points[:, 0], points[:, 1], cells)
    
    # 主图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#80e673')
    ax.triplot(triangulation, color='k', linewidth=0.8, alpha=0.9)
    
    # 坐标与边界控制
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # ===== 添加局部放大图 =====
    if zoom_area is not None:
        ox, oy, sx, sy = zoom_area
        x0, x1 = points[:, 0].min(), points[:, 0].max()
        y0, y1 = points[:, 1].min(), points[:, 1].max()
        xlen, ylen = x1 - x0, y1 - y0
        xmid, ymid = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        
        # 计算放大窗口范围
        xcenter = xmid + ox * xlen
        ycenter = ymid + oy * ylen
        xmin = xcenter - 0.5 * sx * xlen
        xmax = xcenter + 0.5 * sx * xlen
        ymin = ycenter - 0.5 * sy * ylen
        ymax = ycenter + 0.5 * sy * ylen

        # 创建 inset
        ax_inset = inset_axes(
            ax, width="40%", height="40%", loc='upper right',
            bbox_to_anchor=(0.02, 0.02, 0.96, 0.96),
            bbox_transform=ax.transAxes
        )

        ax_inset.set_facecolor('#80e673')
        ax_inset.triplot(triangulation, 'k-', linewidth=1, alpha=1.0)
        ax_inset.set_xlim(xmin, xmax)
        ax_inset.set_ylim(ymin, ymax)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_aspect('equal')

        # 调整边框
        for spine in ax_inset.spines.values():
            spine.set_linewidth(3)
            spine.set_color('White')
            spine.set_alpha(1.0)

        # 连接线
        mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="White",
                   linestyle="-", linewidth=1.2, alpha=0.9)

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig)
    print(f"Mesh 图已保存: {save_path}")

def set_parallel_projection(ax):
    """设置平行投影（轴测投影）"""
    # 方法1：新版本matplotlib
    try:
        ax.set_proj_type('ortho')  # 正交投影（平行投影）
    except:
        pass
    
    # 方法2：通过修改视图角度来模拟平行投影
    ax.view_init(elev=20, azim=225)

    
# ===================== 测试函数族（2D） =====================
def f2_smooth(x, y):
    return np.sin(2*np.pi*x) + 0.3*x + 0.1*np.cos(5*np.pi*y)

def f2_spike(x, y):
    g1 = np.exp(-200*((x-0.35)**2 + (y-0.35)**2))
    g2 = -0.6*np.exp(-120*((x-0.75)**2 + (y-0.70)**2))
    return g1 + g2

def f2_multi_peaks(x, y):
    """三角函数产生多个峰谷，边界为0"""
    # 边界衰减函数 - 在边界处为0，中心为1
    boundary = np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # 原始三角函数
    z = (np.sin(6*np.pi*x) * np.cos(4*np.pi*y) +
         0.6 * np.sin(8*np.pi*(x-0.3)) * np.cos(6*np.pi*(y+0.2)) -
         0.4 * np.cos(10*np.pi*x) * np.sin(8*np.pi*y))
    
    # 应用边界衰减
    z *= boundary
    
    return z

def tiaohe(x, y):
    return -x**2 - y**2

# ===================== PyVista 绘图类 =====================
class AnisotropicPlotter:
    """各向异性缩放绘图器"""
    
    def __init__(
        self,
        window_size: int,
        background_color: str,
        fixed_z_scale: float,
        fixed_bounds: Tuple[float, float, float, float, float, float]
    ):
        self.window_size = window_size
        self.background_color = background_color
        self.fixed_z_scale = fixed_z_scale  # 固定Z轴缩放比例
        self.fixed_bounds = fixed_bounds  # 固定Z轴刻度范围
        self.plotter = None
        
    def scale_about_center(self, mesh: pv.PolyData, sxyz: Tuple[float, float, float], 
                          center: np.ndarray) -> pv.PolyData:
        """以中心点为基准进行缩放"""
        m = mesh.copy()
        m.translate(-center, inplace=True)
        m.scale(sxyz, inplace=True)
        m.translate(center, inplace=True)
        return m
    
    def setup_camera(self, center: np.ndarray, elev_deg: float, azim_deg: float, 
                    scale_factors: Tuple[float, float, float]):
        """设置正交相机 - 固定相机距离"""
        az = np.deg2rad(azim_deg)
        el = np.deg2rad(elev_deg)
        
        # 固定相机距离，不再依赖几何尺寸
        r_cam = 2.5  # 固定距离
        
        # 计算相机位置向量
        cam_vec_world = np.array([
            r_cam * np.cos(el) * np.cos(az),
            r_cam * np.cos(el) * np.sin(az),
            r_cam * np.sin(el)
        ])
        
        # 对齐几何缩放
        cam_vec_scaled = cam_vec_world * np.array(scale_factors)
        
        self.plotter.camera.parallel_projection = True
        self.plotter.camera.position = (center + cam_vec_scaled).tolist()
        self.plotter.camera.focal_point = center.tolist()
        self.plotter.camera.up = (
            -np.sin(el) * np.cos(az),
            -np.sin(el) * np.sin(az),
            np.cos(el)
        )
        
        # 设置固定的正交投影缩放
        self.plotter.camera.parallel_projection = True
        self.plotter.camera.parallel_scale = 1.8  # 固定投影缩放
    
    def add_axis_labels(self, orig_bounds: Tuple[float, float, float, float, float, float],
                       scaled_bounds: Tuple[float, float, float, float, float, float],
                       center: np.ndarray, scale_factors: Tuple[float, float, float],
                       L_xy: float, font_size: int = 100):
        """添加坐标轴刻度标签"""
        sx, sy, sz = scale_factors
        
        # 使用固定的坐标轴范围来生成刻度
        x_min, x_max, y_min, y_max, z_min, z_max = self.fixed_bounds
        
        # 刻度设置
        x_ticks = np.linspace(x_min, x_max, 5)
        y_ticks = np.linspace(y_min, y_max, 5)  
        z_ticks = np.linspace(z_min, z_max, 5)
        
        # X轴标签（底部前缘）
        for x_val in x_ticks:
            x_scaled = center[0] + (x_val - center[0]) * sx
            label_pos = [x_scaled - 0.10 * L_xy, scaled_bounds[2] - 0.08 * L_xy, scaled_bounds[4]]
            self._add_single_label(label_pos, f"{x_val:.1f}", font_size)
        
        # Y轴标签（右侧前缘）
        for y_val in y_ticks:
            y_scaled = center[1] + (y_val - center[1]) * sy
            label_pos = [scaled_bounds[0] - 0.15 * L_xy, y_scaled - 0.08 * L_xy, scaled_bounds[4]]
            self._add_single_label(label_pos, f"{y_val:.1f}", font_size)
            
        # Z轴标签（左侧后角）
        for z_val in z_ticks:
            z_scaled = center[2] + (z_val - center[2]) * sz
            label_pos = [scaled_bounds[0] - 0.18 * L_xy, scaled_bounds[3], z_scaled + 0.06 * L_xy]
            self._add_single_label(label_pos, f"{z_val:.1f}", font_size)
    
    def _add_single_label(self, position: List[float], text: str, font_size: int):
        """添加单个标签"""
        self.plotter.add_point_labels(
            [position],
            [text],
            font_size=font_size,
            text_color='black',
            shape_color='white',
            shape_opacity=0,
            margin=0,
            show_points=False,
            always_visible=True,
            font_family='times',
            bold=True,
            italic=True
        )
    
    def add_title(self, title_text: str, font_size: int = 35):
        """使用 VTK 直接控制文本属性"""
        from vtkmodules.vtkRenderingCore import vtkTextActor

        text_actor = vtkTextActor()
        text_actor.SetInput(title_text)

        prop = text_actor.GetTextProperty()
        prop.SetFontFamilyToTimes()
        prop.BoldOn()
        
        # 方法1: 使用非常大的字号
        prop.SetFontSize(150)  # 尝试非常大的值
        
        prop.SetColor(0, 0, 0)
        prop.SetJustificationToCentered()
        prop.SetVerticalJustificationToTop()
        prop.ShadowOn()
        prop.SetShadowOffset(3, -3)

        # 设置位置
        text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor.SetPosition(0.5, 0.999)  # 降低位置
        
        self.plotter.renderer.AddActor(text_actor)

    
    def plot_mesh(self, nodes: np.ndarray, cells: np.ndarray, node_values: np.ndarray,
                elev_deg: float = 45, azim_deg: float = 220,
                surface_color: Tuple[float, float, float] = (0.31, 0.90, 0.24),
                node_color: Tuple[float, float, float] = (0.90, 0.49, 0.13),
                title: str = "",  # 添加标题参数
                output_file: str = "output.jpg"):
        """
        绘制网格和节点
        
        Parameters:
        -----------
        nodes : np.ndarray, shape (n_nodes, 3)
            节点坐标数组，每行是一个节点的(x, y, z)坐标
        cells : np.ndarray, shape (n_cells, n_vertices)
            网格单元关系数组，每行是一个单元的顶点索引
        node_values : np.ndarray, shape (n_nodes,)
            节点处的函数值，用于确定曲面高度
        """
        # 创建曲面几何 (x, y, node_values)
        x, y = nodes[:, 0], nodes[:, 1]
        z_values = node_values
        
        # 创建PyVista网格
        pts = np.column_stack([x, y, z_values])
        
        # 构建面数组
        faces = []
        for cell in cells:
            faces.append(len(cell))  # 顶点数量
            faces.extend(cell)       # 顶点索引
        faces = np.array(faces, dtype=np.int64)
        
        surf = pv.PolyData(pts, faces)
        orig_bounds = surf.bounds
        
        # 使用固定的缩放系数，不再自动计算
        sx, sy = 1.0, 1.0  # X,Y轴不缩放
        sz = self.fixed_z_scale  # Z轴固定缩放比例
        
        # 计算中心点 - 固定为中心 (0.5, 0.5, 0) 以确保一致性
        center = np.array([0.5, 0.5, -1.5])
        
        # 计算XY方向的范围用于标签定位
        rx, ry = (orig_bounds[1] - orig_bounds[0], 
                  orig_bounds[3] - orig_bounds[2])
        L_xy = max(rx, ry, 1e-12)
        
        # 创建节点球体（使用固定缩放）
        SPHERE_RADIUS_RATIO = 0.006
        r_vis = SPHERE_RADIUS_RATIO * 1.0  # 使用固定尺寸，不再依赖L
        
        # 创建球体时考虑各向异性缩放
        ellipsoid = pv.ParametricEllipsoid(r_vis/sx, r_vis/sy, r_vis/sz, u_res=20, v_res=20)
        
        nodes_pv = pv.PolyData(pts)
        glyphs = nodes_pv.glyph(geom=ellipsoid, scale=False, orient=False)
        
        # 缩放几何
        surf_scaled = self.scale_about_center(surf, [sx, sy, sz], center)
        glyphs_scaled = self.scale_about_center(glyphs, [sx, sy, sz], center)

        # 初始化绘图器
        high_res_factor = 4  # 4倍超采样
        self.plotter = pv.Plotter(
            off_screen=True, 
            window_size=(self.window_size * high_res_factor, 
                        self.window_size * high_res_factor)
        )
        self.plotter.set_background(self.background_color)
        
        # 添加网格
        self.plotter.add_mesh(
            surf_scaled,
            color=surface_color,
            show_edges=True, edge_color="black", line_width=1.0,
            smooth_shading=True, lighting=True, opacity=1.0
        )
        self.plotter.add_mesh(
            glyphs_scaled,
            color=node_color,
            smooth_shading=True, lighting=True, opacity=1.0
        )
        
        # 设置相机 - 不再传递L参数
        self.setup_camera(center, elev_deg, azim_deg, (sx, sy, sz))
        
        if title:
            self.add_title(title, font_size=35)
        
        # 添加坐标轴标签
        x0, x1, y0, y1, z0, z1 = self.fixed_bounds
        scaled_bounds = (x0, x1, y0, y1, z0 * sz, z1 * sz)
        self.add_axis_labels(orig_bounds, scaled_bounds, center, (sx, sy, sz), L_xy)
        
        # 添加坐标轴（隐藏默认标签，使用固定范围）
        actor = self.plotter.show_bounds(
            bounds=scaled_bounds,
            grid='back', location='outer',
            show_xaxis=True, show_yaxis=True, show_zaxis=True,
            xtitle='', ytitle='', ztitle='',
            color='black', font_size=12,
            n_xlabels=5, n_ylabels=5, n_zlabels=5,
            fmt="%.1f",
            font_family='arial',
        )
        
        # 隐藏默认标签
        actor.SetXAxisLabelVisibility(0)
        actor.SetYAxisLabelVisibility(0)
        actor.SetZAxisLabelVisibility(0)
        
        # 保存结果
        self.plotter.screenshot(output_file, transparent_background=False)
        self.plotter.close()
        print(f"✅ Saved: {output_file}")

# ===================== 修改后的主程序封装 =====================
def run_interpolation_experiment(func, A, B, K, interpolate_func, method_name, save_dir, tri_A, tri_B):
    """
    运行插值实验的主函数 - 使用PyVista绘图

    参数:
    func: 测试函数
    nA, nB: A/B 节点数量
    K: 迭代步数
    interpolate_func: 插值函数 (tri或rbf)
    method_name: 方法名称，用于保存文件
    save_dir: 保存目录

    返回:
    errors_A: A点误差记录列表
    errors_B: B点误差记录列表
    """
    errors_A = []  # A点的误差记录 (B→A阶段)
    errors_B = []  # B点的误差记录 (A→B阶段)

    # 创建方法特定的保存目录
    method_save_dir = os.path.join(save_dir, method_name)
    os.makedirs(method_save_dir, exist_ok=True)

    # 初始化数据
    yA_pred = func(A[:, 0], A[:, 1]).copy()
    yB_pred = np.full(B.shape[0], np.nan)

    # ===================== 使用PyVista绘制每一帧 =====================
    print(f"正在为 {method_name} 创建PyVista动画帧...")

    # 创建PyVista绘图器
    # 在run_interpolation_experiment函数中找到创建plotter的地方，修改为：
    plotter = AnisotropicPlotter(
        window_size=800, 
        background_color="white",
        fixed_z_scale=1.0,      # Z轴缩放为原来的一半
        fixed_bounds = (-1, 1, -1, 1, -2, 0)   # Z轴刻度固定为-1到1
    )

    for iteration in range(K):  # 包括第0次迭代
        for phase in [0, 1]:  # 0: A→B, 1: B→A
            
            phase_name_title = 'A->B' if phase == 0 else 'B->A'
            phase_name_save = 'A_to_B' if phase == 0 else 'B_to_A'  # 文件名使用安全字符
            
            # 统一标题格式，不区分 Initial
            plot_title = f"{method_name} - Iter {iteration+1} - {phase_name_title}"
            
            if iteration == 0 and phase == 0:
                # 初始状态：A→B
                yB_pred = interpolate_func(B, A, yA_pred)
                yB_true = func(B[:, 0], B[:, 1])
                yB_error = np.abs(yB_pred - yB_true)
                nodes = B
                cells = tri_B
                node_values = yB_pred
                surface_color = (0.31, 0.90, 0.24)  # 绿色表面
                node_color = (0.45, 0.62, 0.81)     # 橙色节点
                error = np.mean(yB_error)
                errors_B.append(error)
            
            elif iteration == 0 and phase == 1:
                # 初始迭代的 B→A 阶段 - 添加这个分支！
                yA_pred = interpolate_func(A, B, yB_pred)
                yA_true = func(A[:, 0], A[:, 1])
                yA_error = np.abs(yA_pred - yA_true)
                nodes = A
                cells = tri_A
                node_values = yA_pred
                surface_color = (0.31, 0.90, 0.24)  # 绿色表面
                node_color = (0.90, 0.49, 0.13)     # 蓝色节点 - 使用不同的颜色！
                error = np.mean(yA_error)
                errors_A.append(error)
                print(f"{method_name} - Iter {iteration+1}, B→A: A点绝对平均误差 = {error:.6f}")
               
            elif iteration > 0:
                if phase == 0:  # A→B 阶段
                    yB_pred = interpolate_func(B, A, yA_pred)
                    yB_true = func(B[:, 0], B[:, 1])
                    yB_error = np.abs(yB_pred - yB_true)
                    nodes = B
                    cells = tri_B
                    node_values = yB_pred
                    surface_color = (0.31, 0.90, 0.24)  # 绿色表面
                    node_color = (0.45, 0.62, 0.81)     # 橙色节点
                    error = np.mean(yB_error)
                    errors_B.append(error)
                    print(f"{method_name} - Iter {iteration+1}, A→B: B点绝对平均误差 = {error:.6f}")
                    
                else:  # B→A 阶段
                    yA_pred = interpolate_func(A, B, yB_pred)
                    yA_true = func(A[:, 0], A[:, 1])
                    yA_error = np.abs(yA_pred - yA_true)
                    nodes = A
                    cells = tri_A
                    node_values = yA_pred
                    surface_color = (0.31, 0.90, 0.24)  # 绿色表面
                    node_color = (0.90, 0.49, 0.13)     # 蓝色节点
                    error = np.mean(yA_error)
                    errors_A.append(error)
                    print(f"{method_name} - Iter {iteration+1}, B→A: A点绝对平均误差 = {error:.6f}")

            # 为当前帧创建节点坐标数组
            nodes_3d = np.column_stack([nodes[:, 0], nodes[:, 1], np.zeros_like(nodes[:, 0])])

            # 使用PyVista绘制当前帧 - 只保存一次！
            frame_filename = os.path.join(method_save_dir, f"iter{iteration+1:02d}_phase{phase}_{phase_name_save}.jpg")
            plotter.plot_mesh(
                nodes=nodes_3d,
                cells=cells,
                node_values=node_values,
                elev_deg=35,
                azim_deg=226,
                surface_color=surface_color,
                node_color=node_color,
                title=plot_title,
                output_file=frame_filename
            )

        error_plot_filename = os.path.join(method_save_dir, f"error_iter{iteration+1:02d}.jpg")
        plot_error_fields(A, B, tri_A, tri_B, yA_error, yB_error, error_plot_filename, method_name, iteration)
    
    print(f"\n{method_name} PyVista动画帧完成！")
    print(f"保存位置: {method_save_dir}")

    return errors_A, errors_B

# ===================== 主执行部分 =====================
if __name__ == "__main__":
    # ===================== 可调参数 =====================
    func = tiaohe      # 选择其中一个二维测试函数
    N = 100    # A 节点数量
    K = 200               # 迭代步（一次 A→B + 一次 B→A 算一轮）- 减少次数用于测试

    mesh_A, mesh_B = generate_2d_meshes([-1,1], [-1,1], N)

    A = mesh_A['nodes']
    B = mesh_B['nodes']

    tri_A = mesh_A['triangles']
    tri_B = mesh_B['triangles']
 
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, "2d_interpolation_animation_smooth")
    os.makedirs(save_dir, exist_ok=True)

    # ===================== 绘制真解图 =====================
    plot_mesh(A, tri_A, zoom_area=(-0.2, -0.3, 0.2, 0.2), save_dir=save_dir, filename="mesh_A.pdf")
    plot_mesh(B, tri_B, zoom_area=(-0.2, -0.3, 0.2, 0.2), save_dir=save_dir, filename="mesh_B.pdf")
    plot_true_solution(func, save_dir)
    
    # ===================== 运行不同插值方法的实验 =====================
    results = {}

    # # 运行三角线性插值
    # errors_A_tri, errors_B_tri = run_interpolation_experiment(
    #     func, A, B, K, interpolate_tri, "TriLinear", save_dir, tri_A=tri_A, tri_B=tri_B
    # )
    # results['TriLinear'] = {'errors_A': errors_A_tri, 'errors_B': errors_B_tri}

    # 运行RBF插值
    errors_A_rbf, errors_B_rbf = run_interpolation_experiment(
        func, A, B, K, interpolate_rbf, "RBFELM", save_dir, tri_A=tri_A, tri_B=tri_B
    )
    results['RBFELM'] = {'errors_A': errors_A_rbf, 'errors_B': errors_B_rbf}

    # ===================== 统一绘制误差比较图 =====================
    print("\n正在绘制统一的误差比较图...")
    
    fig, ax = plt.subplots(figsize=(10, 5))
        
    styles = {
        'TriLinear_A': {'color': '#E67E22', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.2},
        'TriLinear_B': {'color': '#2E86C1', 'marker': 's', 'linestyle': '--', 'linewidth': 2.2},
        'RBF_A': {'color': '#E67E22', 'marker': '^', 'linestyle': '-', 'linewidth': 2.2},
        'RBF_B': {'color': '#2E86C1', 'marker': 'D', 'linestyle': '--', 'linewidth': 2.2}
    }

    # 绘制四条线 - 添加 markevery 参数避免标记过于密集
    ax.semilogy(range(1, len(results['TriLinear']['errors_B']) + 1), 
                results['TriLinear']['errors_B'], 
                **styles['TriLinear_B'], markersize=5, markevery=10, 
                label='TriLinear A→B')

    ax.semilogy(range(1, len(results['TriLinear']['errors_A']) + 1), 
                results['TriLinear']['errors_A'], 
                **styles['TriLinear_A'], markersize=5, markevery=10,
                label='TriLinear B→A')

    ax.semilogy(range(1, len(results['RBFELM']['errors_B']) + 1), 
                results['RBFELM']['errors_B'], 
                **styles['RBF_B'], markersize=5, markevery=10,
                label='RBFELM A→B')

    ax.semilogy(range(1, len(results['RBFELM']['errors_A']) + 1), 
                results['RBFELM']['errors_A'], 
                **styles['RBF_A'], markersize=5, markevery=10,
                label='RBFELM B→A')


    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold', family='serif', labelpad=6)
    ax.set_ylabel('Mean Absolute Error (log scale)', fontsize=12, fontweight='bold', family='serif', labelpad=6)
    ax.set_title('Error Convergence Comparison', fontsize=14, fontweight='bold', family='serif', pad=16)

    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, framealpha=0.9, loc='center right')
    plt.tight_layout()

    # 保存误差比较图
    error_comparison_path = os.path.join(save_dir, "error_comparison.pdf")
    plt.savefig(error_comparison_path, dpi=200, bbox_inches='tight')
    print(f"误差比较图: {error_comparison_path}")

    # 显示最终误差统计
    print(f"\nFinal error statistics:")
    for method_name in ['TriLinear', 'RBFELM']:
        result = results[method_name]
        print(f"{method_name} - Final error A (B→A): {result['errors_A'][-1]:.6f}")
        print(f"{method_name} - Final error B (A→B): {result['errors_B'][-1]:.6f}")