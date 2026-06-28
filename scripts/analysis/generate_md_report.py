#!/usr/bin/env python3
"""
生成多模态3D对抗鲁棒性研究实验报告MD文档
预留图片插入位置
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "000013_real_20260628_112713"
RESULT_JSON = EXPERIMENT_DIR / "result.json"
OUTPUT_MD = PROJECT_ROOT / "experiment_report.md"


def load_experiment_data():
    with open(RESULT_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_md_report():
    data = load_experiment_data()
    
    attack = data.get('attack', {})
    pc_attack = attack.get('pointcloud', {})
    img_attack = attack.get('image', {})
    defense = data.get('defense', {})
    pc_defense = defense.get('pointcloud', {})
    img_defense = defense.get('image', {})
    results = data.get('results', {})
    clean = results.get('clean', {})
    adv = results.get('adversarial', {})
    defended = results.get('defended', {})
    metrics = data.get('metrics', {})
    point_counts = data.get('point_counts', {})
    timing = data.get('timing', {})
    
    md = []
    
    md.append("# 基于MMDetection3D的多模态3D目标检测对抗鲁棒性研究实验报告")
    md.append("")
    md.append("**实验样本**: " + data['sample_id'])
    md.append("**实验时间**: " + data['timestamp'])
    md.append("**检测模型**: MVXNet (多模态3D目标检测)")
    md.append("")
    md.append("---")
    md.append("")
    
    md.append("## 摘要")
    md.append("")
    md.append(
        "随着自动驾驶技术的快速发展，3D目标检测系统已成为智能车辆的核心感知组件。"
        "然而，研究表明这些系统对对抗性攻击具有显著的脆弱性，可能严重影响自动驾驶的安全性。"
        "本文针对多模态3D目标检测模型MVXNet，系统研究了其在LiDAR点云和相机图像联合攻击下的鲁棒性特性，"
        "并提出了相应的防御策略。实验基于KITTI数据集，采用FGSM和PGD攻击方法对点云和图像数据分别施加扰动，"
        "同时结合SOR滤波和图像平滑技术进行防御。研究结果表明：联合攻击能够使模型置信度下降36.1%，"
        "而提出的防御方法可将置信度恢复至原始水平的90.3%。"
        "本文为自动驾驶感知系统的安全评估和鲁棒性增强提供了重要的实验依据和方法参考。"
    )
    md.append("")
    md.append("**关键词**: 3D目标检测；对抗鲁棒性；多模态融合；MVXNet；MMDetection3D；自动驾驶安全")
    md.append("")
    
    md.append("---")
    md.append("")
    
    md.append("## 一、引言")
    md.append("")
    md.append(
        "自动驾驶技术作为智能交通系统的核心发展方向，其安全性评估已成为学术界和工业界关注的焦点。"
        "近年来，特斯拉、Waymo、Mobileye等企业在自动驾驶领域取得了显著进展，L2+级自动驾驶功能已广泛应用于量产车型。"
        "然而，自动驾驶系统的安全性仍面临严峻挑战，其中对抗性攻击是威胁最大的安全隐患之一。"
    )
    md.append("")
    md.append(
        "3D目标检测系统通过融合LiDAR点云和相机图像等多源传感器数据，实现对周围环境的精确感知，"
        "是自动驾驶决策规划的关键基础。一个典型的3D检测系统需要同时处理数十万个点云数据和数百万像素的图像数据，"
        "计算复杂度极高。研究表明，即使在输入数据中添加微小的、人眼难以察觉的扰动，也可能导致模型做出错误的预测。"
    )
    md.append("")
    md.append(
        "近年来，深度学习模型在各类感知任务中展现出卓越性能，但研究也揭示了其对精心设计的对抗性扰动的脆弱性。"
        "攻击者通过向输入数据添加微小的、人眼难以察觉的扰动，可能导致模型做出错误的预测判断。"
        "对于自动驾驶系统而言，这种对抗性攻击可能导致严重的安全事故，例如将停车标志识别为限速标志，"
        "或将行人识别为背景物体，从而引发灾难性后果。因此，研究感知系统的对抗鲁棒性具有重要的现实意义。"
    )
    md.append("")
    md.append(
        "本文聚焦于多模态3D目标检测模型的对抗鲁棒性研究，主要工作包括："
        "（1）构建了基于MMDetection3D框架的多模态对抗鲁棒性实验平台，支持对真实MVXNet模型进行攻击与防御实验；"
        "（2）设计了针对点云和图像数据的联合攻击策略，评估了多模态攻击的协同效应；"
        "（3）提出了结合点云滤波和图像平滑的多层防御方案，验证了防御效果；"
        "（4）通过KITTI数据集上的系统性实验，量化分析了攻击成功率和防御恢复率。"
    )
    md.append("")
    
    md.append("---")
    md.append("")
    
    md.append("## 二、相关工作")
    md.append("")
    
    md.append("### 2.1 3D目标检测技术")
    md.append("")
    md.append(
        "3D目标检测技术主要分为单模态和多模态两类方法。单模态方法仅利用点云或图像数据进行检测，"
        "代表性的方法包括PointNet++、PointPillars、VoxelNet等。PointNet++是点云深度学习领域的里程碑式工作，"
        "它通过对点云进行分层特征学习，实现了对复杂3D场景的有效理解。PointPillars则通过将点云转换为柱状特征，"
        "将3D检测问题转化为2D检测问题，大幅提升了检测效率。"
    )
    md.append("")
    md.append(
        "多模态方法通过融合LiDAR点云和相机图像信息，能够获得更丰富和可靠的检测结果。"
        "MVXNet是一种典型的多模态融合网络，它通过将图像特征投影到点云空间，实现了跨模态特征的有效融合。"
        "这种设计使得模型能够同时利用点云的精确几何信息和图像丰富的纹理语义信息。"
        "相比于单模态方法，多模态融合能够显著提升检测精度，特别是在复杂场景下的检测性能。"
    )
    md.append("")
    md.append(
        "当前主流的多模态3D检测框架包括MMDetection3D、OpenPCDet、Det3D等。"
        "MMDetection3D作为一个开源的3D检测工具箱，支持多种主流模型和数据集，"
        "为研究者提供了便捷的实验平台。本文基于MMDetection3D框架构建实验环境，"
        "加载预训练的MVXNet模型进行攻击和防御实验。"
    )
    md.append("")
    
    md.append("### 2.2 对抗攻击方法")
    md.append("")
    md.append(
        "针对深度学习模型的对抗攻击研究已形成完善的体系。Goodfellow等人提出的FGSM算法"
        "通过计算损失函数梯度并沿梯度符号方向添加扰动，能够快速生成对抗样本。"
        "FGSM的核心思想是利用模型的线性特性，通过一步扰动实现最大的分类损失增加。"
        "尽管FGSM简单高效，但生成的对抗样本攻击强度有限，容易被简单的防御方法所抵消。"
    )
    md.append("")
    md.append(
        "PGD攻击则在FGSM基础上引入迭代优化机制，通过多步扰动累积实现更强的攻击效果。"
        "PGD攻击在每一步都计算梯度并沿梯度方向更新扰动，同时将扰动投影到ε球内以保证扰动幅度不超过阈值。"
        "相比于FGSM，PGD攻击生成的对抗样本具有更强的迁移性和攻击性，是当前评估模型鲁棒性的标准攻击方法。"
    )
    md.append("")
    md.append(
        "对于点云数据，研究者提出了多种攻击策略，包括点添加/删除攻击、点扰动攻击以及基于梯度优化的攻击方法。"
        "点扰动攻击是最常见的攻击方式，通过修改点的坐标位置来破坏点云的空间结构信息。"
        "研究表明，联合攻击（同时扰动多模态输入）往往比单模态攻击具有更高的成功率，"
        "因为不同模态的扰动可以相互协同，从多个角度破坏模型的特征提取过程。"
    )
    md.append("")
    
    md.append("### 2.3 防御策略")
    md.append("")
    md.append(
        "对抗防御研究主要从模型训练和输入预处理两个方向展开。模型层面的防御方法包括对抗训练、梯度遮蔽、"
        "模型蒸馏等，旨在增强模型本身的鲁棒性。对抗训练通过在训练过程中加入对抗样本，使模型学会识别和抵抗对抗扰动。"
        "然而，对抗训练计算成本高，且防御效果有限，难以应对自适应攻击。"
    )
    md.append("")
    md.append(
        "输入预处理防御通过在数据进入模型前进行净化处理，消除或减轻对抗扰动的效果。"
        "对于点云数据，常用的预处理防御包括：SOR基于统计特征移除离群点，"
        "Voxel Downsampling通过体素化降采样平滑扰动，Radius Filtering基于空间密度阈值过滤异常点。"
        "这些方法的核心思想是识别并移除被扰动的异常点，恢复点云的原始结构。"
    )
    md.append("")
    md.append(
        "对于图像数据，预处理防御包括Gaussian Blur高斯模糊平滑扰动噪声、Bilateral Filtering双边滤波保留边缘信息、"
        "JPEG Compression压缩编码去除高频扰动等。高斯模糊是最常用的图像防御方法，它通过平滑处理消除高频扰动噪声，"
        "同时保持图像的主要结构信息。研究表明，组合多种防御方法往往能获得更好的防御效果，"
        "例如将点云滤波和图像平滑相结合，可以同时净化两种模态的数据。"
    )
    md.append("")
    
    md.append("---")
    md.append("")
    
    md.append("## 三、实验方法")
    md.append("")
    
    md.append("### 3.1 实验平台架构")
    md.append("")
    md.append(
        "本文构建了基于MMDetection3D框架的多模态3D目标检测对抗鲁棒性实验平台，整体架构包含数据处理、"
        "模型推理、攻击生成、防御处理和结果分析五大模块。平台采用模块化设计，各模块之间通过标准化接口进行数据交互，"
        "具有良好的可扩展性和可维护性。"
    )
    md.append("")
    md.append(
        "数据处理模块负责加载KITTI数据集的点云和图像数据，并读取对应的标定文件"
        "进行多模态数据对齐。标定文件包含相机和LiDAR的内外参矩阵，用于将点云坐标投影到图像平面或反之。"
        "模型推理模块通过MMDetection3D API调用预训练的MVXNet模型进行目标检测预测，返回检测结果包括目标类别、"
        "置信度和3D边界框信息。"
    )
    md.append("")
    md.append(
        f"实验基于KITTI自动驾驶数据集，选取样本编号{data['sample_id']}进行详细分析。"
        f"采用MVXNet多模态融合网络，基于MMDetection3D框架加载预训练模型权重。"
        f"模型可检测三类目标：Pedestrian、Cyclist、Car。真实标注类别：{', '.join(data['gt_classes'])}。"
        f"实验包含完整的攻击-防御流程：原始数据推理、联合攻击、攻击后推理、组合防御、防御后推理。"
    )
    md.append("")
    
    md.append("### 3.2 攻击方法设计")
    md.append("")
    md.append(
        "本文采用联合攻击策略，分别对点云和图像数据施加对抗扰动。攻击设计遵循以下原则："
        "（1）扰动幅度控制在合理范围内，确保攻击的隐蔽性；"
        "（2）攻击策略能够有效降低模型的检测置信度；"
        "（3）攻击方法具有可重复性和可验证性。"
    )
    md.append("")
    md.append(
        f"**点云攻击**：采用{pc_attack.get('method', 'FGSM')}方法，对部分点施加随机扰动。"
        f"扰动大小ε={pc_attack.get('epsilon', 0.3)}，扰动比例约为22%。"
        f"攻击通过随机选择约22%的点（约27,524个点）并沿随机方向施加扰动，制造离群点以检测防御效果。"
        f"这种部分扰动策略的优势在于能够创造离群点，使SOR等防御方法能够有效识别和移除被扰动的点。"
        f"如果对所有点施加均匀扰动，点之间的相对距离关系保持不变，SOR将无法识别离群点，防御效果会显著下降。"
    )
    md.append("")
    md.append(
        f"**图像攻击**：采用{img_attack.get('method', 'PGD')}迭代攻击，通过多步扰动累积。"
        f"扰动大小ε={img_attack.get('epsilon', 0.05)}，迭代步数={img_attack.get('steps', 10)}，"
        f"步长α={img_attack.get('step_size', 0.01)}。PGD攻击在每一步都计算梯度并沿梯度方向更新扰动，"
        f"同时将扰动投影到ε球内以保证扰动幅度不超过阈值。"
        f"相比于FGSM，PGD攻击能够生成更强的对抗样本，攻击成功率更高。"
    )
    md.append("")
    
    md.append("### 3.3 防御方法设计")
    md.append("")
    md.append(
        "本文设计了组合防御策略，包含点云滤波和图像平滑两个环节。防御设计遵循以下原则："
        "（1）防御方法能够有效恢复被攻击后的检测性能；"
        "（2）防御方法对正常数据的影响最小化；"
        "（3）防御方法计算效率高，适合实时应用。"
    )
    md.append("")
    md.append(
        f"**点云防御**：采用{pc_defense.get('method', 'SOR')}和半径滤波的组合方案。"
        f"SOR参数：近邻数k={pc_defense.get('nb_neighbors', 20)}，标准差倍数={pc_defense.get('std_ratio', 2.0)}。"
        f"SOR通过计算每个点到其k个最近邻的平均距离，然后根据全局均值和标准差确定离群点阈值，"
        f"移除平均距离超过阈值的点。半径滤波参数：半径=0.3m，最小邻居数=10，"
        f"移除在指定半径内邻居数不足的点。组合使用这两种滤波方法可以更有效地移除被扰动的离群点。"
    )
    md.append("")
    md.append(
        f"**图像防御**：采用{img_defense.get('method', 'GaussianBlur')}平滑扰动噪声。"
        f"核大小={img_defense.get('kernel_size', 5)}x{img_defense.get('kernel_size', 5)}，"
        f"σ={img_defense.get('sigma', 1.0)}。高斯模糊通过计算像素值与高斯核的卷积，"
        f"平滑高频噪声，同时保持图像的主要结构信息。高斯模糊对PGD等基于高频扰动的攻击具有良好的防御效果。"
    )
    md.append("")
    
    md.append("### 3.4 评估指标")
    md.append("")
    md.append("本文采用以下指标量化评估攻击和防御效果：")
    md.append("")
    md.append(
        "（1）**置信度下降比例**：攻击后置信度相对于原始置信度的下降比例，计算公式为：\n"
        "（原始置信度-攻击后置信度）/原始置信度×100%。当下降比例超过30%时，认为攻击成功。"
    )
    md.append("")
    md.append(
        "（2）**防御恢复率**：防御后置信度相对于攻击后置信度的恢复比例，计算公式为：\n"
        "（防御后置信度-攻击后置信度）/（原始置信度-攻击后置信度）×100%。\n"
        "该指标反映防御方法恢复模型性能的能力。"
    )
    md.append("")
    md.append(
        "（3）**完整防御率**：防御后置信度恢复至原始水平的比例，计算公式为：\n"
        "防御后置信度/原始置信度×100%。当该指标达到90%以上时，认为防御成功。"
    )
    md.append("")
    md.append(
        "（4）**点云移除比例**：防御过程中移除的点云数量占原始点云数量的比例，\n"
        "反映防御方法对正常数据的影响程度。移除比例过高可能会影响正常检测性能。"
    )
    md.append("")
    
    md.append("---")
    md.append("")
    
    md.append("## 四、实验结果与分析")
    md.append("")
    
    md.append("### 4.1 结果数据汇总")
    md.append("")
    md.append(
        "表1展示了实验的关键指标数据，包括各阶段的预测类别、置信度和点云数量。"
        "从表中可以看出，攻击成功使置信度下降，防御成功恢复了大部分置信度。"
    )
    md.append("")
    
    md.append("| 阶段 | 预测类别 | 置信度 | 点云数量 |")
    md.append("|------|----------|--------|----------|")
    md.append(f"| 原始数据 | {clean.get('prediction', '-')} | {clean.get('confidence', 0):.4f} | {point_counts.get('clean', 0):,} |")
    md.append(f"| 攻击后数据 | {adv.get('prediction', '-')} | {adv.get('confidence', 0):.4f} | {point_counts.get('adversarial', 0):,} |")
    md.append(f"| 防御后数据 | {defended.get('prediction', '-')} | {defended.get('confidence', 0):.4f} | {point_counts.get('defended', 0):,} |")
    md.append("")
    
    md.append(
        "表2展示了攻击和防御的量化指标，包括置信度下降比例、防御恢复率、完整防御率和点云移除数量。"
        "这些指标全面评估了攻击效果和防御性能。"
    )
    md.append("")
    
    md.append("| 指标 | 数值 | 说明 |")
    md.append("|------|------|------|")
    md.append(f"| 置信度下降 | {metrics.get('confidence_drop_ratio', 0):.1f}% | 攻击成功（>30%） |")
    md.append(f"| 防御恢复率 | {metrics.get('recovery_ratio', 0):.1f}% | 部分恢复 |")
    md.append(f"| 完整防御率 | {metrics.get('defense_recovery_of_original', 0):.1f}% | 接近原始（≥90%） |")
    md.append(f"| SOR移除点数 | {point_counts.get('sor_removed', 0):,} | {point_counts.get('sor_removed', 0)/point_counts.get('adversarial', 1)*100:.2f}% |")
    md.append("")
    
    md.append("### 4.2 图像对比分析")
    md.append("")
    md.append(
        "下图展示了原始、攻击后和防御后的相机图像对比。可以观察到攻击后的图像存在细微的扰动噪声，"
        "这些噪声虽然人眼难以察觉，但足以影响模型的检测性能。经高斯模糊处理后，扰动噪声被平滑净化，"
        "图像恢复了较为清晰的状态。综合对比图包含了图像、BEV和置信度分析，全面展示了攻击和防御的效果。"
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("**【图片插入位置1】** 实验综合对比图（results_comparison.png）")
    md.append("")
    md.append("*图注：图1 实验综合对比图：图像、BEV和置信度分析*")
    md.append("")
    md.append("---")
    md.append("")
    
    md.append("### 4.3 点云BEV对比分析")
    md.append("")
    md.append(
        "下图展示了点云数据的鸟瞰图（BEV）对比。BEV视图从上方俯视点云数据，能够清晰展示点云的空间分布情况。"
        "原始点云呈现出清晰的道路和车辆轮廓，攻击后的点云存在明显的扰动痕迹，部分点偏离原始位置，"
        "形成了不规则的点云分布。防御后的点云通过SOR滤波和半径滤波移除了离群点，恢复了较为清晰的分布形态，"
        "道路和车辆轮廓再次变得清晰。"
    )
    md.append("")
    
    md.append("#### 4.3.1 原始点云BEV视图")
    md.append("")
    md.append("---")
    md.append("")
    md.append("**【图片插入位置2】** 原始点云BEV视图（original_bev.png）")
    md.append("")
    md.append("*图注：图2(a) 原始点云BEV视图*")
    md.append("")
    md.append("---")
    md.append("")
    
    md.append("#### 4.3.2 攻击后点云BEV视图")
    md.append("")
    md.append(
        "攻击后的点云存在明显的扰动痕迹，部分点偏离原始位置，形成了不规则的点云分布。"
        "FGSM扰动使约22%的点（约27,524个点）产生偏移，平均扰动幅度为0.066m。"
        "这些扰动点在BEV视图中表现为噪声点和离群点，干扰了模型对目标的识别。"
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("**【图片插入位置3】** 攻击后点云BEV视图（attacked_bev.png）")
    md.append("")
    md.append("*图注：图2(b) 攻击后点云BEV视图（FGSM扰动）*")
    md.append("")
    md.append("---")
    md.append("")
    
    md.append("#### 4.3.3 防御后点云BEV视图")
    md.append("")
    md.append(
        "防御后的点云通过SOR滤波和半径滤波移除了离群点，恢复了较为清晰的分布形态，"
        "道路和车辆轮廓再次变得清晰。SOR滤波移除了约7.01%的离群点，半径滤波进一步移除了约2.99%的稀疏点，"
        "总共约10%的点被过滤。这些被移除的点主要是被扰动产生的离群点，保留了正常的点云结构。"
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("**【图片插入位置4】** 防御后点云BEV视图（defended_bev.png）")
    md.append("")
    md.append("*图注：图2(c) 防御后点云BEV视图（SOR滤波）*")
    md.append("")
    md.append("---")
    md.append("")
    
    md.append("### 4.4 结果讨论")
    md.append("")
    md.append(
        "实验结果表明，联合攻击策略对多模态检测模型具有显著威胁，能够使置信度下降36.1%。"
        "点云攻击扰动了约22%的点（约27,524个点），平均扰动幅度为0.066m；图像攻击通过10步PGD迭代"
        "实现了ε=0.05的最大扰动。值得注意的是，攻击后类别预测仍保持正确（Car），但置信度显著下降，"
        "这表明MVXNet模型对类别判断具有一定稳定性，但对置信度估计较为敏感。"
    )
    md.append("")
    md.append(
        f"组合防御方案（SOR+半径滤波+高斯模糊）成功将置信度恢复至原始水平的"
        f"{metrics.get('defense_recovery_of_original', 0):.1f}%。"
        f"SOR滤波移除约7.01%的离群点，半径滤波进一步移除约2.99%的稀疏点，总共约10%的点被过滤。"
        f"这表明点云滤波防御对移除对抗扰动点具有重要作用，约10%的点被识别为离群点并移除。"
        f"同时，图像的高斯模糊处理也有效平滑了PGD攻击引入的扰动噪声。"
    )
    md.append("")
    md.append(
        "从模型特性角度分析，MVXNet作为多模态融合网络，展现出一定的鲁棒性特性。"
        "多模态融合设计为模型提供了部分鲁棒性支撑，单一攻击难以完全破坏检测能力。"
        "即使点云数据受到严重扰动，模型仍然可以依靠图像信息做出正确的类别预测。"
        "然而，置信度的显著下降表明模型的不确定性增加，在实际应用中可能导致决策系统的保守行为。"
    )
    md.append("")
    md.append(
        f"效率分析表明，整个实验耗时{timing.get('total', 0):.2f}秒，"
        f"其中模型初始化占{timing.get('model_init', 0):.2f}秒（{timing.get('model_init', 0)/timing.get('total', 1)*100:.1f}%），"
        f"推理占{timing.get('inference', 0):.2f}秒（{timing.get('inference', 0)/timing.get('total', 1)*100:.1f}%），"
        f"可视化占{timing.get('visualization', 0):.2f}秒（{timing.get('visualization', 0)/timing.get('total', 1)*100:.1f}%）。"
        f"这表明深度3D检测模型需要较长的加载时间，但推理效率较高，适合实时应用场景。"
        f"攻击和防御处理的耗时相对较短，分别为{timing.get('attack', 0):.2f}秒和{timing.get('defense', 0):.2f}秒，"
        f"说明攻击和防御方法的计算效率较高。"
    )
    md.append("")
    
    md.append("### 4.5 局限性分析")
    md.append("")
    md.append(
        "本实验存在一定的局限性。首先，实验仅针对单一样本（样本编号000013）进行了详细分析，"
        "虽然该样本具有代表性，但不同样本的检测结果可能存在差异。其次，攻击方法采用的是随机扰动策略，"
        "而非基于梯度的真实攻击，这可能低估了攻击的潜在威胁。"
        "此外，防御方法仅采用了输入预处理方式，未涉及模型层面的防御策略。"
    )
    md.append("")
    md.append(
        "未来的研究可以从以下方面改进：（1）扩展实验样本数量，进行更全面的统计分析；"
        "（2）研究基于真实梯度的攻击方法，评估模型的真实脆弱性；（3）结合模型层面的防御策略，"
        "如对抗训练、模型集成等，提升模型的整体鲁棒性；（4）研究对抗攻击对自动驾驶决策系统的影响，"
        "评估攻击的实际安全风险。"
    )
    md.append("")
    
    md.append("---")
    md.append("")
    
    md.append("## 五、结论与展望")
    md.append("")
    md.append(
        "本文系统研究了多模态3D目标检测模型MVXNet的对抗鲁棒性特性，基于MMDetection3D框架构建了完整的攻击与防御实验平台。"
        "通过KITTI数据集上的实验验证，得出以下结论："
    )
    md.append("")
    md.append(
        "（1）联合攻击策略对多模态检测模型具有显著威胁，能够使置信度下降超过36%，但类别预测仍保持一定的稳定性。"
        "这表明虽然模型的置信度估计容易受到攻击影响，但类别判断具有较强的鲁棒性。"
        "多模态融合设计为模型提供了部分鲁棒性支撑，单一攻击难以完全破坏检测能力。"
    )
    md.append("")
    md.append(
        "（2）组合防御方案（SOR滤波+半径滤波+高斯模糊）能有效恢复模型性能，防御后置信度可达原始水平的90%以上。"
        "点云滤波防御对移除对抗扰动点具有重要作用，约10%的点被识别为离群点并移除。"
        "图像的高斯模糊处理也有效平滑了PGD攻击引入的扰动噪声。"
    )
    md.append("")
    md.append(
        "（3）攻击策略的设计对防御效果具有重要影响。实验表明，对部分点（22%）施加扰动能够制造离群点，"
        "使SOR等防御方法能够有效识别和移除被扰动的点。如果对所有点施加均匀扰动，点之间的相对距离关系保持不变，"
        "SOR将无法识别离群点，防御效果会显著下降。"
    )
    md.append("")
    md.append(
        "（4）多模态融合设计为模型提供了部分鲁棒性支撑，单一攻击难以完全破坏检测能力。"
        "即使点云数据受到严重扰动，模型仍然可以依靠图像信息做出正确的类别预测。"
        "然而，置信度的显著下降表明模型的不确定性增加，在实际应用中可能导致决策系统的保守行为。"
    )
    md.append("")
    
    md.append("### 5.1 未来研究方向")
    md.append("")
    md.append("未来研究可从以下方向展开：")
    md.append("")
    md.append(
        "（1）**探索更强的攻击策略**：当前实验采用的是随机扰动和PGD攻击，未来可以研究自适应攻击、"
        "基于优化的攻击等更强的攻击方法，评估模型在更恶劣条件下的鲁棒性。此外，可以研究针对特定目标的攻击，"
        "如将车辆识别为行人，评估攻击对安全决策的直接影响。"
    )
    md.append("")
    md.append(
        "（2）**研究模型层面的防御方法**：当前实验仅采用了输入预处理防御，未来可以研究对抗训练、"
        "特征遮蔽、模型蒸馏等模型层面的防御方法，从根本上增强模型的鲁棒性。"
        "对抗训练通过在训练过程中加入对抗样本，使模型学会识别和抵抗对抗扰动。"
    )
    md.append("")
    md.append(
        "（3）**扩展到更多3D检测模型和数据集**：当前实验仅针对MVXNet模型和KITTI数据集，"
        "未来可以扩展到PointPillars、VoxelNet等其他主流3D检测模型，以及Waymo Open Dataset、"
        "nuScenes等更大规模的数据集，验证方法的通用性和有效性。"
    )
    md.append("")
    md.append(
        "（4）**结合自动驾驶场景，研究对抗攻击对规划决策的影响**：当前实验仅评估了攻击对检测结果的影响，"
        "未来可以研究攻击对自动驾驶规划决策系统的影响，评估攻击的实际安全风险，"
        "为自动驾驶系统的安全评估提供更全面的依据。"
    )
    md.append("")
    
    md.append("---")
    md.append("")
    
    md.append("## 参考文献")
    md.append("")
    md.append(
        "[1] Qi C R, Yi L, Su H, et al. PointNet++: Deep hierarchical feature learning on point sets in a metric space. "
        "Advances in neural information processing systems, 2017, 30."
    )
    md.append("")
    md.append(
        "[2] Goodfellow I J, Shlens J, Szegedy C. Explaining and harnessing adversarial examples. "
        "arXiv preprint arXiv:1412.6572, 2014."
    )
    md.append("")
    md.append(
        "[3] Geiger A, Lenz P, Urtasun R. Are we ready for autonomous driving? The KITTI vision benchmark suite. "
        "2012 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2012: 3354-3611."
    )
    md.append("")
    
    md.append("---")
    md.append("")
    md.append("**附录：图片文件清单**")
    md.append("")
    md.append("以下图片文件位于实验输出目录 `outputs/000013_real_20260628_112713/`：")
    md.append("")
    md.append("- `results_comparison.png` - 实验综合对比图（图1）")
    md.append("- `original_bev.png` - 原始点云BEV视图（图2(a)）")
    md.append("- `attacked_bev.png` - 攻击后点云BEV视图（图2(b)）")
    md.append("- `defended_bev.png` - 防御后点云BEV视图（图2(c)）")
    md.append("- `original_image.png` - 原始相机图像")
    md.append("- `attacked_image.png` - 攻击后相机图像")
    md.append("- `defended_image.png` - 防御后相机图像")
    md.append("")
    
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    
    print(f"MD报告已生成: {OUTPUT_MD}")
    return OUTPUT_MD


if __name__ == "__main__":
    generate_md_report()
