import numpy as np
import scipy.interpolate
import os
import json

def BD_RATE(R1, D1, R2, D2, piecewise=1, higher_better=False):
    """
    计算 Bjontegaard Delta Rate (BD-Rate)

    参数:
    R1, D1: Anchor (Baseline) 的 Rate 和 Distortion
    R2, D2: Test (Ours) 的 Rate 和 Distortion
    piecewise: 1 (推荐) 使用 PCHIP 插值防止震荡; 0 使用多项式拟合(仅适合平滑且密集的曲线)
    higher_better: True 表示质量指标(如 Acc/PSNR)，直接以该指标为积分轴，不做取反
    """

    # --- 内部函数：数据清洗 ---
    def clean_data(R, D, higher_better=False):
        # 允许 None / null，并统一转为 float
        R = np.array([np.nan if v is None else v for v in R], dtype=float)
        D = np.array([np.nan if v is None else v for v in D], dtype=float)

        # 1. 移除无效值 (NaN/Inf) + 过滤非正 rate (log 需要)
        valid_mask = np.isfinite(R) & np.isfinite(D) & (R > 0)
        R = R[valid_mask]
        D = D[valid_mask]

        if len(D) < 1: return np.array([]), np.array([])

        # 2. 先按 rate 排序，构造单调 RD 曲线（去除测量噪声造成的回摆）
        sorted_by_r = np.argsort(R)
        R = R[sorted_by_r]
        D = D[sorted_by_r]

        # 3. 约束质量/失真随 rate 单调变化
        if higher_better:
            # 质量指标：rate 越大，质量不应下降
            D = np.maximum.accumulate(D)
        else:
            # 失真指标：rate 越大，失真不应上升
            D = np.minimum.accumulate(D)

        # 4. 按 Distortion/Quality (X轴) 排序
        # BD-Rate 计算积分时，X 轴必须单调
        sorted_indices = np.argsort(D)
        R = R[sorted_indices]
        D = D[sorted_indices]

        # 5. 去重 (处理重复的 X 值)
        # 如果存在多个相同的 D 值，只保留 R 最小的那个 (帕累托最优原则)
        unique_D, unique_indices = np.unique(D, return_index=True)

        # np.unique 返回的是排序后的唯一 D，我们需要找到对应的最优 R
        final_R = []
        final_D = []

        # 使用简单的遍历来确保对应正确的 R
        current_d = None
        min_r = float('inf')

        for r_val, d_val in zip(R, D):
            if d_val != current_d:
                if current_d is not None:
                    final_R.append(min_r)
                    final_D.append(current_d)
                current_d = d_val
                min_r = r_val
            else:
                if r_val < min_r:
                    min_r = r_val
        # 添加最后一个点
        if current_d is not None:
            final_R.append(min_r)
            final_D.append(current_d)

        return np.array(final_R), np.array(final_D)

    # --- 1. 预处理数据 ---
    R1, D1 = clean_data(R1, D1, higher_better=higher_better)
    R2, D2 = clean_data(R2, D2, higher_better=higher_better)

    # 检查点数，至少需要2个点才能插值
    if len(D1) < 2 or len(D2) < 2:
        return np.nan

    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # --- 2. 确定积分区间 ---
    # 必须取交集，严禁外推
    min_int = max(np.min(D1), np.min(D2))
    max_int = min(np.max(D1), np.max(D2))

    # 检查是否有有效重叠
    # 加上一个小量 epsilon 防止浮点误差
    if max_int <= min_int + 1e-6:
        return np.nan

    # --- 3. 拟合与积分 ---
    if piecewise == 0:
        # 多项式拟合 (不推荐用于稀疏数据)
        try:
            order1 = min(3, len(D1) - 1)
            order2 = min(3, len(D2) - 1)
            p1 = np.polyfit(D1, lR1, order1)
            p2 = np.polyfit(D2, lR2, order2)

            p_int1 = np.polyint(p1)
            p_int2 = np.polyint(p2)

            int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
            int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
        except np.linalg.LinAlgError:
            return np.nan
    else:
        # PCHIP 插值 (推荐)
        # 在重叠区间内生成采样点
        samples = np.linspace(min_int, max_int, num=100)

        try:
            v1 = scipy.interpolate.pchip_interpolate(D1, lR1, samples)
            v2 = scipy.interpolate.pchip_interpolate(D2, lR2, samples)
        except ValueError:
            # 通常由 x 不严格单调引起，clean_data 应该已经解决此问题
            return np.nan

        # 梯形法则积分
        int1 = np.trapz(v1, x=samples)
        int2 = np.trapz(v2, x=samples)

    # --- 4. 计算最终 BD-Rate ---
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100

    return avg_diff


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
    root_dir = os.path.join(_PROJECT_ROOT, "results", "result_json", "split")
    # 'Hyper Packing 8bitkmeans' 'Hyper Packing 10bitkmeans' 'Hyper Packing 12bitkmeans'
    # "chen2019", "VTM", "Hyper padding", "Hyper packing", "VQ"
    ours = "Hyper padding"
    baseline_list = ["VTM",]  
    # 配置:
    # 如果计算 Acc/PSNR (质量), distortion_metric = "Acc@1"
    # 如果计算 MSE (误差), distortion_metric = "MSE"
    bpp_metric = "BPFP"
    distortion_metric = "Acc@1"
    higher_better = distortion_metric.lower().startswith("acc")
    print(f"Calculating BD-Rate with distortion metric: {distortion_metric}, bpp metric: {bpp_metric}")
    print(f"Ours: {ours}")
    print(f"{'Model':<10} | {'Layer':<8} | {'Baseline':<15} | {'BD-Rate (%)':<15}")
    print("-" * 60)

    for model in ["dinov2"]:
        json_path = os.path.join(root_dir, f"{model}_results.json")
        with open(json_path, 'r') as f:
            data = json.load(f)

        for layer in ["blk05", "blk11", "blk17", "blk23"]:
            if layer not in data: continue

            # 获取 Ours 的数据
            if ours not in data[layer]: continue

            R_ours = data[layer][ours][bpp_metric]
            D_ours = data[layer][ours][distortion_metric]

            for codec in baseline_list:
                if codec not in data[layer]: continue

                # 获取 Baseline 的数据
                R_base = data[layer][codec][bpp_metric]
                D_base = data[layer][codec][distortion_metric]

                # 计算 BD-Rate (默认开启 piecewise 防止数值爆炸)
                val = BD_RATE(
                    R_base,
                    D_base,
                    R_ours,
                    D_ours,
                    piecewise=1,
                    higher_better=higher_better,
                )

                # 格式化输出
                if np.isnan(val):
                    val_str = "N/A (No Overlap)"
                else:
                    val_str = f"{val:+.2f}"

                print(f"{model:<10} | {layer:<8} | {codec:<15} | {val_str:<15}")
