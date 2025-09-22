# Created by MacBook Pro at 21.09.25
from math import pi
import matplotlib.pyplot as plt

import config


def plot_pentagon_radar(
        models_to_values,
        properties=("size", "color", "shape", "group_num", "group_size"),
        title="GRM – Property-wise Accuracy (Radar)",
        out_pdf_path="radar_plot.pdf",
        value_range=(0, 100),
        show_markers=True,
):
    """
    生成 5 轴雷达图，比较多模型在五个属性上的表现，并保存为 PDF。
    Args:
        models_to_values (dict[str, list[float]]): 模型到 5 个数值的映射（顺序与 properties 一致）
        properties (Iterable[str]): 5 个属性名（顺时针）
        title (str): 图标题
        out_pdf_path (str): PDF 输出路径
        value_range (tuple[float, float]): 径向坐标 (min_val, max_val)
        show_markers (bool): 是否在顶点绘制标记点
    Returns:
        str: 保存的 PDF 路径
    """
    props = list(properties)
    assert len(props) == 5, "该函数固定为 5 个属性（五边形）。"

    # 顶点角度（首尾闭合）
    angles = [n / float(len(props)) * 2 * pi for n in range(len(props))]
    angles += angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)

    # 径向范围与方向设置
    min_val, max_val = value_range
    ax.set_ylim(min_val, max_val)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # 轴与刻度
    ax.set_xticks([n / float(len(props)) * 2 * pi for n in range(len(props))])
    ax.set_xticklabels(props)

    grid_steps = 5
    ring_vals = [min_val + i * (max_val - min_val) / grid_steps for i in range(1, grid_steps + 1)]
    ax.set_yticks(ring_vals)
    ax.set_yticklabels([f"{int(v)}" for v in ring_vals])
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    # 逐模型绘制：使用默认颜色循环 + 半透明填充
    for model_name, vals in models_to_values.items():
        assert len(vals) == 5, f"{model_name} 必须提供 5 个数值。"
        data = list(vals) + vals[:1]
        line, = ax.plot(angles, data, linewidth=2, label=model_name)  # 线色由默认循环决定
        ax.fill(angles, data, alpha=0.20)  # 半透明填充
        # if show_markers:
        #     ax.scatter(angles[:-1], vals, s=25)  # 顶点标记

    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    fig.tight_layout()
    fig.savefig(out_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_pdf_path


# ---- 示例调用（可删除）----
if __name__ == "__main__":
    demo_models = {
        "ViT": [54.7, 55.7, 55.6, 54.8, 54.4],
        "LLaVA": [53.6, 52.6, 52.7, 52.4, 52.3],
        "InternVL3": [62.6, 65.7, 64.4, 62.8, 62.6],
        "GPT-5": [65.5, 69.5, 75.7, 68.5, 68.9],
        "GRM": [73.0, 73.0, 75.9, 72.4, 72.4],
    }
    output_file = config.get_proj_output_path() / "radar_plot_demo.pdf"
    plot_pentagon_radar(
        models_to_values=demo_models,
        properties=("size", "color", "shape", "group_num", "group_size"),
        title="Property-wise Task Accuracy",
        out_pdf_path=output_file,
        value_range=(40, 80),
        show_markers=True,
    )
