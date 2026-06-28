#!/usr/bin/env python3
"""
尝试修复MVXNet权重不匹配问题
通过调整权重顺序来适配模型
"""
import torch
import numpy as np

checkpoint_file = 'checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'
output_file = 'checkpoints/mvxnet_fixed.pth'

print('加载原始权重...')
checkpoint = torch.load(checkpoint_file, map_location='cpu')

state_dict = checkpoint['state_dict']

# 检查需要修复的层
fix_keys = []
for key in state_dict.keys():
    if 'pts_middle_encoder' in key and 'conv_input.0.weight' in key:
        fix_keys.append(key)

print(f'需要修复的键: {fix_keys}')

# 检查权重的当前形状
for key in fix_keys:
    print(f'{key}: {state_dict[key].shape}')

# 问题分析：
# 权重形状: [16, 3, 3, 3, 128] -> [out_channels, in_channels, D, H, W]
# 模型期望: [3, 3, 3, 128, 16] -> [in_channels, D, H, W, out_channels]

# 需要进行维度重排
# 从 [16, 3, 3, 3, 128] 变成 [3, 3, 3, 128, 16]
# 步骤：transpose(0, 4) 然后 reverse 3D

print('\n尝试修复权重...')

# 创建一个新的state_dict
new_state_dict = {}

for key, value in state_dict.items():
    if 'pts_middle_encoder' in key and 'conv_input.0.weight' in key:
        # 原始形状: [16, 3, 3, 3, 128] = [out_c, in_c, D, H, W]
        # 目标形状: [3, 3, 3, 128, 16] = [in_c, D, H, W, out_c]

        # permute: [16, 3, 3, 3, 128] -> [16, 128, 3, 3, 3] (transpose 1 and 4)
        # 然后 permute -> [3, 3, 3, 128, 16]

        # 实际上权重存储顺序可能是 [out_c, in_c, D, H, W]
        # 但模型期望的是 PyTorch 标准的 conv3d 权重格式 [in_c, D, H, W, out_c]

        # 对于 conv3d: weight shape = [out_channels, in_channels, D, H, W]
        # 当前权重是 [16, 3, 3, 3, 128]
        # 期望是 [3, 3, 3, 128, 16]

        # 转换: [16, 3, 3, 3, 128] -> permute(1, 2, 3, 4, 0) -> [3, 3, 3, 128, 16]
        new_value = value.permute(1, 2, 3, 4, 0)
        new_state_dict[key] = new_value
        print(f'修复 {key}: {value.shape} -> {new_value.shape}')
    else:
        new_state_dict[key] = value

# 保存修复后的权重
checkpoint['state_dict'] = new_state_dict
torch.save(checkpoint, output_file)
print(f'\n修复后的权重已保存到: {output_file}')