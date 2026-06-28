#!/usr/bin/env python3
"""
修复MVXNet权重不匹配问题 - 修复所有3D卷积层
"""
import torch

checkpoint_file = 'checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'
output_file = 'checkpoints/mvxnet_fixed.pth'

print('加载原始权重...')
checkpoint = torch.load(checkpoint_file, map_location='cpu')

state_dict = checkpoint['state_dict']

# 需要修复的所有3D卷积层
# 这些层的权重形状是 [out_channels, in_channels, D, H, W]
# 需要转换成 [in_channels, D, H, W, out_channels]
fix_patterns = [
    'pts_middle_encoder.conv_input',
    'pts_middle_encoder.encoder_layers',
    'pts_middle_encoder.conv_out'
]

print('修复所有3D卷积层...')
new_state_dict = {}

for key, value in state_dict.items():
    needs_fix = any(pattern in key and 'weight' in key for pattern in fix_patterns)

    if needs_fix and value.dim() == 5:
        # 形状是 [out_channels, in_channels, D, H, W]
        # 需要转换成 [in_channels, D, H, W, out_channels]
        # permute(1, 2, 3, 4, 0)
        new_value = value.permute(1, 2, 3, 4, 0)
        new_state_dict[key] = new_value
        if 'conv_input' in key or 'conv_out' in key:
            print(f'修复 {key}: {value.shape} -> {new_value.shape}')
    else:
        new_state_dict[key] = value

# 检查修复后的形状
print('\n修复后的键形状检查:')
for key in new_state_dict.keys():
    if 'pts_middle_encoder' in key and 'weight' in key and new_state_dict[key].dim() == 5:
        print(f'{key}: {new_state_dict[key].shape}')

# 保存修复后的权重
checkpoint['state_dict'] = new_state_dict
torch.save(checkpoint, output_file)
print(f'\n修复后的权重已保存到: {output_file}')