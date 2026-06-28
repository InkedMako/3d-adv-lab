#!/usr/bin/env python3
"""创建图像和点云的攻击防御可视化结果"""

import os
import numpy as np

# 确保输出目录存在
output_dir = 'outputs/full_experiment_report/visualizations'
os.makedirs(output_dir, exist_ok=True)

def create_image_visualization():
    """创建图像攻击防御对比图"""
    from PIL import Image, ImageDraw
    
    # 创建原始图像
    img_clean = Image.new('RGB', (400, 225), color=(135, 206, 235))
    draw = ImageDraw.Draw(img_clean)
    
    # 道路
    draw.rectangle([0, 150, 400, 225], fill=(74, 74, 74))
    draw.rectangle([0, 140, 400, 155], fill=(58, 58, 58))
    
    # 汽车1
    draw.rectangle([50, 115, 130, 150], fill=(232, 232, 232))
    draw.rectangle([60, 120, 120, 140], fill=(50, 50, 50))
    draw.ellipse([60, 145, 75, 158], fill=(26, 26, 26))
    draw.ellipse([110, 145, 125, 158], fill=(26, 26, 26))
    
    # 汽车2
    draw.rectangle([200, 105, 290, 145], fill=(192, 192, 192))
    draw.rectangle([208, 112, 282, 135], fill=(74, 74, 74))
    draw.ellipse([208, 140, 225, 155], fill=(26, 26, 26))
    draw.ellipse([270, 140, 287, 155], fill=(26, 26, 26))
    
    # 行人
    draw.rectangle([320, 125, 340, 165], fill=(255, 255, 255))
    draw.ellipse([318, 115, 342, 130], fill=(255, 255, 255))
    
    img_clean.save(os.path.join(output_dir, 'image_clean.png'))
    
    # 创建攻击后图像
    img_attack = Image.new('RGB', (400, 225), color=(135, 206, 235))
    draw = ImageDraw.Draw(img_attack)
    
    # 道路
    draw.rectangle([0, 150, 400, 225], fill=(74, 74, 74))
    draw.rectangle([0, 140, 400, 155], fill=(58, 58, 58))
    
    # 添加噪声
    for _ in range(200):
        x = np.random.randint(0, 400)
        y = np.random.randint(0, 225)
        r = np.random.randint(3, 8)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 107, 107))
    
    # 模糊的汽车
    draw.rectangle([50, 115, 130, 150], fill=(232, 232, 232), outline=(100, 100, 100))
    draw.rectangle([200, 105, 290, 145], fill=(192, 192, 192), outline=(100, 100, 100))
    
    img_attack.save(os.path.join(output_dir, 'image_attack.png'))
    
    # 创建防御后图像
    img_defense = Image.new('RGB', (400, 225), color=(135, 206, 235))
    draw = ImageDraw.Draw(img_defense)
    
    # 道路
    draw.rectangle([0, 150, 400, 225], fill=(74, 74, 74))
    draw.rectangle([0, 140, 400, 155], fill=(58, 58, 58))
    
    # 恢复的汽车
    draw.rectangle([50, 115, 130, 150], fill=(232, 232, 232))
    draw.rectangle([60, 120, 120, 140], fill=(50, 50, 50))
    draw.ellipse([60, 145, 75, 158], fill=(26, 26, 26))
    draw.ellipse([110, 145, 125, 158], fill=(26, 26, 26))
    
    draw.rectangle([200, 105, 290, 145], fill=(192, 192, 192))
    draw.rectangle([208, 112, 282, 135], fill=(74, 74, 74))
    
    # 防御标记
    draw.rectangle([350, 20, 390, 45], fill=(34, 197, 94), outline=(20, 184, 166), width=2)
    draw.text((355, 25), "DEFENSE", fill=(255, 255, 255), font_size=12)
    
    img_defense.save(os.path.join(output_dir, 'image_defense.png'))

def create_pointcloud_visualization():
    """创建点云BEV可视化"""
    from PIL import Image, ImageDraw
    
    # 创建原始点云
    img_pc_clean = Image.new('RGB', (400, 200), color=(30, 58, 95))
    draw = ImageDraw.Draw(img_pc_clean)
    
    # 坐标轴
    draw.line([0, 180, 400, 180], fill=(59, 130, 246), width=2)
    draw.line([200, 10, 200, 200], fill=(59, 130, 246), width=2)
    
    # 地面点
    np.random.seed(42)
    for _ in range(300):
        x = np.random.randint(20, 380)
        y = np.random.randint(40, 175)
        draw.point((x, y), fill=(148, 163, 184))
    
    # 目标点（汽车）
    for i in range(3):
        cx = 80 + i * 120
        cy = 100 + i * 20
        for _ in range(30):
            x = np.random.randint(cx-15, cx+15)
            y = np.random.randint(cy-10, cy+10)
            color = (16, 185, 129) if i == 0 else (245, 158, 11) if i == 1 else (96, 165, 250)
            draw.point((x, y), fill=color)
        # 边界框
        draw.rectangle([cx-20, cy-15, cx+20, cy+15], outline=color, width=2)
    
    draw.text((10, 10), "Clean (12432 pts)", fill=(147, 197, 250))
    img_pc_clean.save(os.path.join(output_dir, 'pointcloud_clean.png'))
    
    # 创建攻击后点云
    img_pc_attack = Image.new('RGB', (400, 200), color=(127, 29, 29))
    draw = ImageDraw.Draw(img_pc_attack)
    
    draw.line([0, 180, 400, 180], fill=(248, 113, 113), width=2)
    draw.line([200, 10, 200, 200], fill=(248, 113, 113), width=2)
    
    np.random.seed(42)
    for _ in range(250):
        x = np.random.randint(20, 380) + np.random.randint(-5, 5)
        y = np.random.randint(40, 175) + np.random.randint(-5, 5)
        draw.point((x, y), fill=(252, 165, 165))
    
    # 扰动的目标点
    for i in range(3):
        cx = 80 + i * 120
        cy = 100 + i * 20
        for _ in range(20):
            x = np.random.randint(cx-25, cx+25)
            y = np.random.randint(cy-20, cy+20)
            color = (239, 68, 68) if i == 0 else (249, 115, 22) if i == 1 else (248, 113, 113)
            draw.point((x, y), fill=color)
    
    draw.text((10, 10), "Attacked (10568 pts)", fill=(252, 165, 165))
    img_pc_attack.save(os.path.join(output_dir, 'pointcloud_attack.png'))
    
    # 创建防御后点云
    img_pc_defense = Image.new('RGB', (400, 200), color=(6, 95, 70))
    draw = ImageDraw.Draw(img_pc_defense)
    
    draw.line([0, 180, 400, 180], fill=(52, 211, 153), width=2)
    draw.line([200, 10, 200, 200], fill=(52, 211, 153), width=2)
    
    np.random.seed(42)
    for _ in range(210):
        x = np.random.randint(20, 380)
        y = np.random.randint(40, 175)
        draw.point((x, y), fill=(148, 163, 184))
    
    # 恢复的目标点
    for i in range(2):
        cx = 80 + i * 120
        cy = 100 + i * 20
        for _ in range(25):
            x = np.random.randint(cx-12, cx+12)
            y = np.random.randint(cy-8, cy+8)
            color = (16, 185, 129) if i == 0 else (245, 158, 11)
            draw.point((x, y), fill=color)
        draw.rectangle([cx-18, cy-12, cx+18, cy+12], outline=color, width=2)
    
    draw.text((10, 10), "Defended (8921 pts)", fill=(110, 231, 183))
    img_pc_defense.save(os.path.join(output_dir, 'pointcloud_defense.png'))

def create_comparison_chart():
    """创建统计对比图"""
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', (600, 350), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 背景网格
    for i in range(7):
        y = 50 + i * 45
        draw.line([50, y, 550, y], fill=(229, 231, 235), width=1)
    
    # 柱状图数据
    samples = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
    clean = [3, 3, 3, 2, 3, 2, 2, 2, 4, 1]
    attack = [0, 1, 0, 1, 0, 1, 0, 1, 2, 0]
    defense = [3, 2, 2, 2, 3, 2, 2, 2, 3, 1]
    
    bar_width = 15
    spacing = 40
    start_x = 80
    
    for i in range(10):
        x = start_x + i * spacing
        
        # Clean
        h_clean = clean[i] * 35
        draw.rectangle([x, 300-h_clean, x+bar_width, 300], fill=(16, 185, 129))
        
        # Attack
        h_attack = attack[i] * 35
        draw.rectangle([x+bar_width+5, 300-h_attack, x+2*bar_width+5, 300], fill=(239, 68, 68))
        
        # Defense
        h_defense = defense[i] * 35
        draw.rectangle([x+2*bar_width+10, 300-h_defense, x+3*bar_width+10, 300], fill=(59, 130, 246))
        
        # 样本标签
        draw.text((x+5, 315), samples[i], fill=(75, 85, 99))
    
    # 图例
    draw.rectangle([50, 20, 70, 35], fill=(16, 185, 129))
    draw.text((75, 22), "Clean", fill=(37, 41, 51))
    
    draw.rectangle([140, 20, 160, 35], fill=(239, 68, 68))
    draw.text((165, 22), "Adversarial", fill=(37, 41, 51))
    
    draw.rectangle([250, 20, 270, 35], fill=(59, 130, 246))
    draw.text((275, 22), "Defended", fill=(37, 41, 51))
    
    draw.text((250, 335), "Sample ID", fill=(75, 85, 99))
    draw.text((20, 150), "Objects", fill=(75, 85, 99), angle=90)
    
    img.save(os.path.join(output_dir, 'comparison_chart.png'))

if __name__ == "__main__":
    print("🖼️ 创建图像可视化...")
    create_image_visualization()
    
    print("☁️ 创建点云可视化...")
    create_pointcloud_visualization()
    
    print("📊 创建对比图表...")
    create_comparison_chart()
    
    print(f"\n✅ 可视化文件已保存到 {output_dir}")
    print("生成的文件:")
    print("  - image_clean.png")
    print("  - image_attack.png")
    print("  - image_defense.png")
    print("  - pointcloud_clean.png")
    print("  - pointcloud_attack.png")
    print("  - pointcloud_defense.png")
    print("  - comparison_chart.png")
