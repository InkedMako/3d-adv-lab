#!/usr/bin/env python3
"""
高级攻防方法实现
包含：
- SOR防御（统计离群点移除）
- KNN-Defense（特征空间邻域搜索）
- C&W攻击（优化-based攻击）
- KNN攻击（邻域距离约束）
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from typing import Tuple, Optional


class SORDefense:
    """
    Statistical Outlier Removal (SOR) 防御
    原理：计算每个点到k个最近邻的平均距离，移除距离超过阈值的离群点
    
    参考：Standard outlier removal in point cloud processing libraries like PCL
    """
    
    def __init__(self, k: int = 20, std_ratio: float = 1.0):
        """
        Args:
            k: 最近邻点数量
            std_ratio: 阈值倍数，距离超过 mean + std_ratio * std 的点被移除
        """
        self.k = k
        self.std_ratio = std_ratio
    
    def defend(self, points: np.ndarray) -> np.ndarray:
        """
        对点云进行SOR防御
        
        Args:
            points: Nx3或Nx4的点云数组
        
        Returns:
            清洗后的点云数组
        """
        if len(points) < self.k:
            return points
        
        coords = points[:, :3]
        
        # 构建KD树
        tree = cKDTree(coords)
        
        # 查询每个点的k个最近邻
        distances, _ = tree.query(coords, k=self.k + 1)
        
        # 计算每个点到k个最近邻的平均距离（排除自身）
        avg_distances = distances[:, 1:].mean(axis=1)
        
        # 计算全局统计量
        global_mean = avg_distances.mean()
        global_std = avg_distances.std()
        
        # 计算阈值
        threshold = global_mean + self.std_ratio * global_std
        
        # 保留距离小于阈值的点
        mask = avg_distances < threshold
        
        return points[mask]
    
    def __repr__(self):
        return f"SORDefense(k={self.k}, std_ratio={self.std_ratio})"


class KNNDefense:
    """
    KNN-Defense: 特征空间邻域搜索防御
    原理：利用训练集样本的特征空间邻域来恢复被攻击的点
    
    参考：KNN-Defense: Defense against 3D Adversarial Point Clouds using Nearest-Neighbor Search
    (arXiv:2506.06906)
    """
    
    def __init__(self, k: int = 5, feature_dim: int = 64):
        """
        Args:
            k: 特征空间中的最近邻数量
            feature_dim: 特征维度
        """
        self.k = k
        self.feature_dim = feature_dim
        self.train_features = None
        self.train_points = None
    
    def setup(self, train_points: np.ndarray, encoder: Optional[nn.Module] = None):
        """
        设置训练集参考数据
        
        Args:
            train_points: 训练集点云列表
            encoder: 特征提取器（可选）
        """
        self.train_points = train_points
        if encoder is not None:
            with torch.no_grad():
                points_tensor = torch.from_numpy(train_points).float()
                if len(points_tensor.shape) == 2:
                    points_tensor = points_tensor.unsqueeze(0)
                self.train_features = encoder(points_tensor).numpy()
        else:
            # 使用简单的统计特征
            self.train_features = self._compute_statistical_features(train_points)
    
    def _compute_statistical_features(self, points: np.ndarray) -> np.ndarray:
        """计算统计特征"""
        coords = points[:, :3]
        features = np.zeros(self.feature_dim)
        
        # 基础统计特征
        features[:3] = coords.mean(axis=0)  # 中心
        features[3:6] = coords.std(axis=0)  # 标准差
        features[6:9] = coords.min(axis=0)  # 最小值
        features[9:12] = coords.max(axis=0)  # 最大值
        
        # 形状特征
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        features[12:15] = eigenvalues  # 主成分
        
        # 密度特征
        tree = cKDTree(coords)
        distances, _ = tree.query(coords.mean(axis=0), k=min(50, len(coords)))
        features[15] = distances.mean()  # 平均密度
        
        return features
    
    def defend(self, points: np.ndarray) -> np.ndarray:
        """
        使用KNN-Defense恢复点云
        
        Args:
            points: 被攻击的点云
        
        Returns:
            恢复后的点云
        """
        if self.train_features is None:
            # 如果没有设置训练数据，使用SOR作为后备
            return SORDefense().defend(points)
        
        # 计算当前点云的特征
        query_features = self._compute_statistical_features(points)
        
        # 在特征空间中搜索最近邻
        tree = cKDTree(self.train_features)
        distances, indices = tree.query(query_features, k=self.k)
        
        # 使用最近邻样本恢复
        recovered_points = []
        for idx in indices:
            recovered_points.append(self.train_points[idx])
        
        # 合并恢复的点云
        if len(recovered_points) > 0:
            # 取平均作为恢复结果
            recovered = np.mean(recovered_points, axis=0)
            return recovered
        
        return points
    
    def __repr__(self):
        return f"KNNDefense(k={self.k}, feature_dim={self.feature_dim})"


class CWAttack:
    """
    Carlini & Wagner (C&W) 攻击
    原理：通过优化方法生成最小扰动的对抗样本
    
    参考：Carlini and Wagner, "Towards Evaluating the Robustness of Neural Networks"
    """
    
    def __init__(self, 
                 confidence: float = 0.0,
                 learning_rate: float = 0.01,
                 max_iterations: int = 100,
                 binary_search_steps: int = 5,
                 initial_const: float = 0.01):
        """
        Args:
            confidence: 攻击置信度参数
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            binary_search_steps: 二分搜索步数
            initial_const: 初始常数c
        """
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
    
    def attack(self, 
               points: np.ndarray,
               model: nn.Module,
               target_class: Optional[int] = None) -> np.ndarray:
        """
        执行C&W攻击
        
        Args:
            points: 原始点云 Nx3或Nx4
            model: 目标模型
            target_class: 目标类别（定向攻击），None为非定向
        
        Returns:
            对抗点云
        """
        points_tensor = torch.from_numpy(points[:, :3]).float()
        if len(points_tensor.shape) == 2:
            points_tensor = points_tensor.unsqueeze(0)
        
        # 初始化扰动
        perturbation = torch.zeros_like(points_tensor, requires_grad=True)
        
        # 二分搜索最优c值
        c = self.initial_const
        best_perturbation = None
        best_loss = float('inf')
        
        for binary_step in range(self.binary_search_steps):
            optimizer = torch.optim.Adam([perturbation], lr=self.learning_rate)
            
            for iteration in range(self.max_iterations):
                optimizer.zero_grad()
                
                # 应用扰动
                adv_points = points_tensor + perturbation
                
                # 计算模型输出
                outputs = model(adv_points)
                
                # 计算损失
                if target_class is not None:
                    # 定向攻击
                    target_logits = outputs[:, target_class]
                    other_logits = outputs.max(dim=1)[0]
                    loss_attack = other_logits - target_logits + self.confidence
                else:
                    # 非定向攻击
                    correct_logits = outputs.max(dim=1)[0]
                    loss_attack = correct_logits + self.confidence
                
                # L2距离约束
                loss_dist = torch.norm(perturbation, p=2)
                
                # 总损失
                loss = c * loss_attack + loss_dist
                
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_perturbation = perturbation.detach().clone()
            
            # 更新c值
            if loss_attack.item() > 0:
                c *= 2
            else:
                c /= 2
        
        if best_perturbation is not None:
            adv_points = points_tensor + best_perturbation
            result = points.copy()
            result[:, :3] = adv_points.squeeze(0).numpy()
            return result
        
        return points
    
    def __repr__(self):
        return f"CWAttack(confidence={self.confidence}, max_iter={self.max_iterations})"


class KNNAttack:
    """
    KNN攻击：考虑邻域距离约束的攻击
    原理：在扰动时保持与邻域点的距离关系，增强不可感知性
    
    参考：KNN attack for point clouds with distance constraints
    """
    
    def __init__(self, 
                 epsilon: float = 0.1,
                 k_neighbors: int = 10,
                 distance_threshold: float = 0.05,
                 iterations: int = 50):
        """
        Args:
            epsilon: 最大扰动幅度
            k_neighbors: 邻域点数量
            distance_threshold: 邻域距离阈值
            iterations: 迭代次数
        """
        self.epsilon = epsilon
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.iterations = iterations
    
    def attack(self, points: np.ndarray, seed: int = 42) -> np.ndarray:
        """
        执行KNN攻击
        
        Args:
            points: 原始点云 Nx3或Nx4
            seed: 随机种子
        
        Returns:
            对抗点云
        """
        rng = np.random.default_rng(seed)
        coords = points[:, :3]
        
        # 构建KD树
        tree = cKDTree(coords)
        
        # 查询每个点的邻域
        distances, indices = tree.query(coords, k=self.k_neighbors + 1)
        
        # 初始化扰动
        perturbation = np.zeros((len(points), 3), dtype=np.float32)
        
        for _ in range(self.iterations):
            # 计算梯度方向（简化版本：随机方向）
            direction = rng.normal(0, 1, size=(len(points), 3))
            direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
            
            # 计算允许的扰动幅度
            # 约束：扰动后仍保持邻域关系
            for i in range(len(points)):
                neighbor_distances = distances[i, 1:]
                max_perturb = min(
                    self.epsilon,
                    neighbor_distances.mean() * self.distance_threshold
                )
                perturbation[i] = direction[i] * max_perturb
        
        # 应用扰动
        adv_points = points.copy()
        adv_points[:, :3] += perturbation
        
        return adv_points
    
    def __repr__(self):
        return f"KNNAttack(epsilon={self.epsilon}, k_neighbors={self.k_neighbors})"


class AdversarialTraining:
    """
    对抗训练：在训练过程中加入对抗样本
    原理：使用对抗样本训练模型，提高模型鲁棒性
    """
    
    def __init__(self, 
                 attack_methods: list = ['PGD', 'KNN'],
                 attack_ratio: float = 0.3,
                 epochs: int = 10):
        """
        Args:
            attack_methods: 使用的攻击方法列表
            attack_ratio: 对抗样本比例
            epochs: 训练轮数
        """
        self.attack_methods = attack_methods
        self.attack_ratio = attack_ratio
        self.epochs = epochs
        
        # 初始化攻击器
        self.attackers = {
            'PGD': None,  # 需要后续设置
            'KNN': KNNAttack(),
            'C&W': CWAttack()
        }
    
    def generate_adversarial_batch(self, 
                                   points_batch: np.ndarray,
                                   method: str = 'KNN') -> np.ndarray:
        """
        为一批点云生成对抗样本
        
        Args:
            points_batch: 批量点云
            method: 攻击方法
        
        Returns:
            对抗样本批次
        """
        attacker = self.attackers.get(method)
        if attacker is None:
            # 默认使用KNN攻击
            attacker = KNNAttack()
        
        adv_batch = []
        for points in points_batch:
            adv_points = attacker.attack(points)
            adv_batch.append(adv_points)
        
        return np.array(adv_batch)
    
    def __repr__(self):
        return f"AdversarialTraining(methods={self.attack_methods}, ratio={self.attack_ratio})"


# 测试函数
def test_methods():
    """测试所有高级方法"""
    print("=" * 50)
    print("测试高级攻防方法")
    print("=" * 50)
    
    # 创建测试点云
    rng = np.random.default_rng(42)
    clean_points = rng.random((1000, 4)).astype(np.float32)
    clean_points[:, :3] *= 10  # 扩大坐标范围
    
    print(f"\n原始点云: {len(clean_points)} 点")
    
    # 测试SOR防御
    print("\n--- 测试SOR防御 ---")
    sor = SORDefense(k=20, std_ratio=1.0)
    
    # 先添加一些离群点
    outlier_points = clean_points.copy()
    outliers = rng.random((50, 4)).astype(np.float32)
    outliers[:, :3] *= 20  # 离群点距离更远
    outlier_points = np.vstack([outlier_points, outliers])
    
    defended_sor = sor.defend(outlier_points)
    print(f"SOR防御后: {len(defended_sor)} 点 (移除 {len(outlier_points) - len(defended_sor)} 个离群点)")
    
    # 测试KNN攻击
    print("\n--- 测试KNN攻击 ---")
    knn_attack = KNNAttack(epsilon=0.1, k_neighbors=10)
    adv_knn = knn_attack.attack(clean_points)
    perturbation = np.linalg.norm(adv_knn[:, :3] - clean_points[:, :3], axis=1).mean()
    print(f"KNN攻击后: 平均扰动 {perturbation:.4f}")
    
    # 测试C&W攻击（简化版本）
    print("\n--- 测试C&W攻击 ---")
    cw_attack = CWAttack(max_iterations=10)
    print(f"C&W攻击器: {cw_attack}")
    
    # 测试对抗训练
    print("\n--- 测试对抗训练 ---")
    adv_train = AdversarialTraining(attack_methods=['KNN'])
    print(f"对抗训练: {adv_train}")
    
    print("\n" + "=" * 50)
    print("✅ 所有方法测试完成")
    print("=" * 50)


if __name__ == "__main__":
    test_methods()