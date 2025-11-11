import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os

# ===================== 小工具函数 =====================
def generate_1d_meshes(domain, N):
    """
    生成1D的两套网格
    
    参数:
    domain: 求解域 [x_start, x_end]
    N: 单元数量
    
    返回:
    mesh_A, mesh_B: 两套网格的数据
    """
    x_start, x_end = domain
    
    # 网格A: 均匀剖分
    nodes_A = np.linspace(x_start, x_end, N + 1)
    elements_A = [(i, i+1) for i in range(N)]
    
    # 网格B: 端点 + 所有区间中点
    midpoints = [(nodes_A[i] + nodes_A[i+1]) / 2 for i in range(N)]
    nodes_B = np.sort(np.concatenate([[x_start, x_end], midpoints]))
    elements_B = [(i, i+1) for i in range(len(nodes_B)-1)]
    
    return {'nodes': nodes_A, 'elements': elements_A}, {'nodes': nodes_B, 'elements': elements_B}

def interpolate(x_new, x_nodes, y_nodes):
    """
    通用插值接口。
    默认实现是 np.interp（线性插值），
    之后你可以在这里替换成其他插值方法。
    """
    return np.interp(x_new, x_nodes, y_nodes)

def interpolate_rbf(x_new, x_nodes, y_nodes, L=80, gamma=100.0):
    """1D RBF径向基函数插值方法"""
    n = len(x_nodes)
    L = min(L, n)
    
    # 1) 从训练点随机选择 L 个中心
    idx = np.random.choice(n, size=L, replace=False)
    C = x_nodes[idx]  # 形状 [L]
    
    # 2) 构造高斯 RBF 设计矩阵
    # 对于训练数据: 计算 x_nodes 和 C 之间的距离矩阵
    dist_sq_H = (x_nodes[:, np.newaxis] - C[np.newaxis, :])**2  # [n, L]
    H = np.exp(-gamma * dist_sq_H)  # [n, L]
    
    # 对于预测数据: 计算 x_new 和 C 之间的距离矩阵
    dist_sq_K = (x_new[:, np.newaxis] - C[np.newaxis, :])**2  # [m, L]
    K = np.exp(-gamma * dist_sq_K)  # [m, L]
    
    # 3) 最小二乘解权重 w，并前传预测
    w, *_ = np.linalg.lstsq(H, y_nodes, rcond=None)
    y_pred = K @ w
    
    return y_pred

def add_single_inset(ax, ROI, inset_box,
                     Xdense, f_dense, A, yA, B, yB,
                     colors, widths, sizes, draw_connector=True,
                     loc1=1, loc2=3):
    """
    创建单个放大窗
    ROI = (x1, x2, y1, y2)，inset_box = (x0, y0, w, h)
    """
    x1, x2, y1, y2 = ROI
    axins = ax.inset_axes(inset_box, transform=ax.transAxes)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # 曲线
    lt, = axins.plot(Xdense, f_dense, color=colors["true"], linewidth=widths["true"], zorder=1)
    la, = axins.plot(A, yA, color=colors["A"], linewidth=widths["thick"], zorder=2)
    lb, = axins.plot(B, yB, color=colors["B"], linewidth=widths["thin"], zorder=3)

    # 节点
    sa = axins.scatter(A, yA, s=sizes["A"], c=colors["A"], marker="o",
                       linewidths=widths["scatter_edge"], zorder=2)
    sb = axins.scatter(B, yB, s=sizes["B"], c=colors["B"], marker="o",
                       linewidths=widths["scatter_edge"], zorder=3)

    axins.set_xticks([])
    axins.set_yticks([])

    if draw_connector:
        con = mark_inset(ax, axins, loc1=loc1, loc2=loc2,
                        fc="none", ec="0.5", linestyle="--", alpha=0.6)
    else:
        con = None

    return {
        "axins": axins,
        "lt": lt, "la": la, "lb": lb,
        "sa": sa, "sb": sb,
        "connector": con,
    }

def update_inset(phase, A, B, yA, yB, inset):
    """更新放大窗：曲线与散点"""
    if inset is None:
        return
    
    # 更新曲线数据 - 这会自动更新散点位置
    inset["la"].set_data(A, yA)
    inset["lb"].set_data(B, yB)
    
    # 更新线宽（根据相位）
    inset["la"].set_linewidth(2.8 if phase == 0 else 1.6)
    inset["lb"].set_linewidth(1.6 if phase == 0 else 2.8)
    
    # 更新透明度
    inset["la"].set_alpha(1.0 if phase == 0 else 0.6)
    inset["lb"].set_alpha(0.6 if phase == 0 else 1.0)
    
    # 动态调整zorder：确保粗线在细线之上
    if phase == 0:
        inset["la"].set_zorder(2)  # 粗线A在上面
        inset["lb"].set_zorder(3)  # 细线B在下面
    else:
        inset["lb"].set_zorder(2)  # 粗线B在上面
        inset["la"].set_zorder(3)  # 细线A在下面

    # 更新散点样式（不需要更新位置，因为set_data已经处理了）
    if phase == 0:
        inset["sa"].set_sizes([20] * len(A))  # A点大
        inset["sb"].set_sizes([10] * len(B))  # B点小
        inset["sa"].set_alpha(1.0)   # A点不透明
        inset["sb"].set_alpha(0.6)   # B点半透明
        inset["sa"].set_zorder(2)    # 与粗线同层
        inset["sb"].set_zorder(3)    # 与细线同层
    else:
        inset["sa"].set_sizes([10] * len(A))  # A点小
        inset["sb"].set_sizes([20] * len(B))  # B点大
        inset["sa"].set_alpha(0.6)   # A点半透明
        inset["sb"].set_alpha(1.0)   # B点不透明
        inset["sa"].set_zorder(3)    # 与细线同层
        inset["sb"].set_zorder(2)    # 与粗线同层

    # 重新绘制散点
    inset["sa"].set_offsets(np.c_[A, yA])
    inset["sb"].set_offsets(np.c_[B, yB])
    
# ===================== 测试函数族 =====================
def f_smooth(x):
    return np.sin(np.pi*x)

def f_spike(x):
    return np.exp(-400*(x-0.35)**2) - 0.5*np.exp(-200*(x-0.75)**2)

def f_multi_peaks(x, n_peaks=3):
    """
    确定性的多峰函数（没有随机数）
    n_peaks 控制大致的波峰/波谷数量
    """
    base = np.sin(2*np.pi*n_peaks*x)              # 主频，决定大致峰谷数量
    harmonics = sum(
        (1/(k+1)) * np.sin(2*np.pi*(n_peaks+k+1)*x + k*np.pi/4)
        for k in range(1, 5)                      # 加几个固定谐波，幅度逐渐减小
    )
    return base + 0.4 * harmonics


def run_interpolation_experiment(func, A, B, K, interpolate_func, method_name, save_dir,
                                 show_inset=True, ROI_list=None, INSET_BOX_list=None, LOC_list=None):
    """1D 插值交替实验 —— 视觉风格固定，输出插值结果图与误差图"""
    method_save_dir = os.path.join(save_dir, method_name)
    os.makedirs(method_save_dir, exist_ok=True)

    errors_A, errors_B = [], []
    yA, yB = func(A).copy(), np.full_like(B, np.nan)
    Xdense = np.linspace(0, 1, 1000)
    f_dense = func(Xdense)

    # 主图
    fig, ax = plt.subplots(figsize=(10, 8))
    line_true, = ax.plot(Xdense, f_dense, color="k", linewidth=8.0, label="True f(x)", zorder=1)
    line_A, = ax.plot(A, yA, color="#E67E22", linewidth=6.0, label="A grid", zorder=2)
    line_B, = ax.plot(B, yB, color="#2E86C1", linewidth=4.0, alpha=0.5, label="B grid", zorder=3)
    scat_A = ax.scatter(A, yA, s=100, c="#E67E22", zorder=2)
    scat_B = ax.scatter(B, yB, s=100, c="#2E86C1", zorder=3)

    # 坐标轴样式与标签
    label_fontsize = 30
    tick_fontsize = 28
    ax.set_xlim(A.min(), A.max())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel('$x$', fontsize=label_fontsize, fontweight='bold', family='serif')
    ax.set_ylabel('$y$', fontsize=label_fontsize, fontweight='bold', family='serif')
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, direction='in', width=1.0)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper left', fontsize=19, framealpha=0.9, edgecolor='k')

    # 刻度参数
    ax.tick_params(axis='both', which='major', labelsize=23, direction='in', width=1.2, length=5)

    # 放大框
    if show_inset:
        colors = {"true": "k", "A": "#E67E22", "B": "#2E86C1"}
        widths = {"true": 8.0, "thick": 6.0, "thin": 4.0, "scatter_edge": 1.0}
        sizes = {"A": 100, "B": 100}

        insets = []
        for i in range(len(ROI_list)):
            ins = add_single_inset(
                ax=ax,
                ROI=ROI_list[i],
                inset_box=INSET_BOX_list[i],
                Xdense=Xdense,
                f_dense=f_dense,
                A=A,
                yA=yA,
                B=B,
                yB=yB,
                colors=colors,
                widths=widths,
                sizes=sizes,
                draw_connector=True,
                loc1=LOC_list[i][0],
                loc2=LOC_list[i][1]
            )
            insets.append(ins)
            
    # ===================== 主循环 =====================
    for iteration in range(K):
        for phase in [0, 1]:
            if phase == 0:
                yB = interpolate_func(B, A, yA)
                yB_true = func(B)
                error_B = np.mean(np.abs(yB - yB_true))
                errors_B.append(error_B)
                print(f"Iter {iteration}, A→B: B点误差 = {error_B:.10f}")

                line_A.set_linewidth(6.0); line_A.set_alpha(1.0)
                line_B.set_linewidth(4.0); line_B.set_alpha(0.5)
                
                scat_A.set_sizes([100] * len(scat_A.get_offsets()))  # 固定 A 点的大小
                scat_B.set_sizes([60] * len(scat_B.get_offsets()))  # 固定 B 点的大小
                scat_A.set_alpha(1.0)  # A 点完全不透明
                scat_B.set_alpha(0.5)  # B 点稍微透明
                
                line_A.set_zorder(2); line_B.set_zorder(3)
                scat_A.set_zorder(2); scat_B.set_zorder(3)
            else:
                yA = interpolate_func(A, B, yB)
                yA_true = func(A)
                error_A = np.mean(np.abs(yA - yA_true))
                errors_A.append(error_A)
                print(f"Iter {iteration}, B→A: A点误差 = {error_A:.10f}")

                line_B.set_linewidth(6.0); line_B.set_alpha(1.0)
                line_A.set_linewidth(4.0); line_A.set_alpha(0.5)

                scat_A.set_sizes([60] * len(scat_A.get_offsets()))  # 固定 A 点的大小
                scat_B.set_sizes([100] * len(scat_B.get_offsets()))  # 固定 B 点的大小
                scat_A.set_alpha(0.5)  # A 点完全不透明
                scat_B.set_alpha(1.0)  # B 点稍微透明
            
                line_A.set_zorder(3); line_B.set_zorder(2)
                scat_A.set_zorder(3); scat_B.set_zorder(2)
                
            # 更新曲线与散点
            line_A.set_data(A, yA)
            line_B.set_data(B, yB)
            scat_A.set_offsets(np.c_[A, yA])
            scat_B.set_offsets(np.c_[B, yB])

            if ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.legend(loc='upper left', fontsize=19, framealpha=0.9, edgecolor='k')
            
            # 放大区域参数
            if show_inset and iteration <= 10:
                for idx, ins in enumerate(insets):
                    axins = ins["axins"]
                    axins.set_visible(True)
                    
                    # 设置放大区域的 x 和 y 范围
                    axins.set_xlim(ROI_list[idx][0], ROI_list[idx][1])
                    axins.set_ylim(ROI_list[idx][2], ROI_list[idx][3])
                    
                    # 显示连接线
                    if ins["connector"] is not None:
                        ins["connector"][0].set_visible(True)
                        ins["connector"][1].set_visible(True)
                    
                    # 更新放大图 - 所有样式设置都在 update_inset 中完成
                    update_inset(phase, A, B, yA, yB, ins)

            else:
                # 隐藏放大图
                for ins in insets:
                    ins["axins"].set_visible(False)
                    if ins["connector"] is not None:
                        ins["connector"][0].set_visible(False)
                        ins["connector"][1].set_visible(False)

            phase_name = 'A→B' if phase == 0 else 'B→A'
            ax.set_title(f"{method_name} - Iter {iteration+1}, {phase_name}", 
                    fontsize=32, fontweight='bold', family='serif', pad=20)
            frame_filename = os.path.join(method_save_dir, f"{method_name}_iter{iteration+1:02d}_phase{phase}_{phase_name}.pdf")
            plt.tight_layout(pad=2.0)
            plt.savefig(frame_filename, dpi=200, bbox_inches='tight')
            print(f"保存: {os.path.basename(frame_filename)}")

        # 误差图
        # 设置专业字体
        fig_err, ax_err = plt.subplots(figsize=(12, 9), constrained_layout=True)  # 增大图形尺寸
        yA_true, yB_true = func(A), func(B)
        err_A, err_B = np.abs(yA - yA_true), np.abs(yB - yB_true)

        # 绘制误差曲线和散点
        ax_err.plot(A, err_A, color="#E67E22", linewidth=6.0, marker='o', markersize=12, 
                label='A (B→A)', zorder=3, alpha=0.8, markevery=10)
        ax_err.plot(B, err_B, color="#2E86C1", linewidth=6.0, marker='o', markersize=12, 
                label='B (A→B)', zorder=3, alpha=0.8, markevery=10)
        
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)         # ✅ 开启科学计数法
        formatter.set_powerlimits((0, 0))      # ✅ 强制所有数都用科学计数法
        ax_err.yaxis.set_major_formatter(formatter)
        ax_err.yaxis.get_offset_text().set_fontsize(28)
        ax_err.yaxis.get_offset_text().set_fontweight('bold')
        ax_err.yaxis.get_offset_text().set_family('serif')

        # 坐标轴设置
        ax_err.set_xlim(0, 1)
        ax_err.set_xlabel('$x$', fontsize=30, fontweight='bold', family='serif')
        ax_err.set_ylabel('Absolute Error', fontsize=30, fontweight='bold', family='serif')

        # 标题和图例
        ax_err.set_title(f"{method_name} - Iter {iteration+1}", 
                        fontsize=32, fontweight='bold', family='serif', pad=20)
        ax_err.legend(loc='best', framealpha=0.9, edgecolor='k', fontsize=20)

        # 网格和刻度设置
        ax_err.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax_err.tick_params(axis='both', which='major', labelsize=28, 
                        direction='in', width=1.2, length=5)

        # 调整布局并保存
        err_path = os.path.join(method_save_dir, f"error_iter{iteration+1:02d}.pdf")
        plt.savefig(err_path, dpi=300, bbox_inches='tight', pad_inches=0.1, 
                    facecolor='white', edgecolor='none')
        plt.close(fig_err)
        print(f"保存误差图: {os.path.basename(err_path)}")

    plt.close(fig)
    print(f"\n{method_name} 实验完成！结果保存至 {method_save_dir}\n")
    return errors_A, errors_B

    
if __name__ == "__main__":
    # ===================== 可调参数 =====================
    func = f_smooth
    N = 100
    K = 200   # 动画迭代次数

    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, "1d_interpolation_animation_smooth")
    os.makedirs(save_dir, exist_ok=True)

    SHOW_INSET = True

    # 第一个放大图参数
    ROI1 = (0.28, 0.42, 0.6, 1.05)
    INSET_BOX1 = (0.60, 0.50, 0.35, 0.35)
    LOC1_1, LOC1_2 = 1, 3

    # ===================== 数据准备 =====================
    mesh_A, mesh_B = generate_1d_meshes([0, 1], N)
    A, B = mesh_A['nodes'], mesh_B['nodes']

    # ===================== 插值实验 1: Linear =====================
    print("\n==============================")
    print("运行方法: Linear_interp")
    print("==============================")

    errors_A_linear, errors_B_linear = run_interpolation_experiment(
        func=func,
        A=A,
        B=B, 
        K=K,
        interpolate_func=interpolate,
        method_name="Linear_interp",
        save_dir=save_dir,
        show_inset=SHOW_INSET,
        ROI_list=[ROI1],
        INSET_BOX_list=[INSET_BOX1],
        LOC_list=[(LOC1_1, LOC1_2)]
    )

    # ===================== 插值实验 2: RBF =====================
    print("\n==============================")
    print("运行方法: RBF_interp")
    print("==============================")

    errors_A_rbf, errors_B_rbf = run_interpolation_experiment(
        func=func,
        A=A,
        B=B,
        K=K,
        interpolate_func=interpolate_rbf,
        method_name="RBF_interp",
        save_dir=save_dir,
        show_inset=SHOW_INSET,
        ROI_list=[ROI1],
        INSET_BOX_list=[INSET_BOX1],
        LOC_list=[(LOC1_1, LOC1_2)]
    )

  # ===================== 统一绘制误差收敛曲线 =====================
    # 设置专业字体
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'mathtext.fontset': 'stix'
    })

    fig, ax = plt.subplots(figsize=(10, 4.3))  # 增大图形尺寸

    # 定义绘图样式（颜色、线型、标记）
    styles = {
        'Linear_A': {'color': '#E67E22', 'marker': 'o', 'linestyle': '-',  'linewidth': 2.7},
        'Linear_B': {'color': '#2E86C1', 'marker': 's', 'linestyle': '--', 'linewidth': 2.7},
        'RBF_A':    {'color': '#E67E22', 'marker': '^', 'linestyle': '-',  'linewidth': 2.7},
        'RBF_B':    {'color': '#2E86C1', 'marker': 'D', 'linestyle': '--', 'linewidth': 2.7},
    }

    # 绘制四条误差曲线（使用 log y 轴）
    plt.semilogy(range(1, len(errors_B_linear) + 1),
                errors_B_linear,
                **styles['Linear_B'], markersize=6, markevery=10, label='Linear A→B', zorder=3)

    plt.semilogy(range(1, len(errors_A_linear) + 1),
                errors_A_linear,
                **styles['Linear_A'], markersize=6, markevery=10, label='Linear B→A', zorder=3)

    plt.semilogy(range(1, len(errors_B_rbf) + 1),
                errors_B_rbf,
                **styles['RBF_B'], markersize=6, markevery=10, label='RBF A→B', zorder=3)

    plt.semilogy(range(1, len(errors_A_rbf) + 1),
                errors_A_rbf,
                **styles['RBF_A'], markersize=6, markevery=10, label='RBF B→A', zorder=3)

    # 坐标轴与标题设置
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Absolute Error', fontsize=14, fontweight='bold')
    plt.title('Convergence History of Interpolation Methods', fontsize=16, fontweight='bold', pad=20)

    # 网格和图例设置
    plt.grid(True, which='both', alpha=0.4, linestyle='--', linewidth=0.8)
    plt.legend(fontsize=12, framealpha=0.9, edgecolor='k', loc='center right')

    # 刻度设置
    ax.tick_params(axis='both', which='major', labelsize=12, 
                direction='in', width=1.2, length=5)

    # 调整布局
    plt.tight_layout(pad=2.0)

    # 保存图像
    converge_path = os.path.join(save_dir, "convergence_comparison.pdf")
    plt.savefig(converge_path, dpi=300, bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"\n收敛曲线已保存至: {converge_path}\n")