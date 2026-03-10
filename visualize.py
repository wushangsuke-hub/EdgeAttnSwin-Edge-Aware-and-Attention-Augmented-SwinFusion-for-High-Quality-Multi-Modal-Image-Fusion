"""
可视化对比原始模型和增强模型的融合结果
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_comparison(ir_dir, vi_dir, original_dir, enhanced_dir, save_path='comparison.png', num_samples=4):
    """可视化对比"""
    
    # 获取图像列表
    img_list = sorted([f for f in os.listdir(enhanced_dir) if f.endswith('.png')])[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i, img_name in enumerate(img_list):
        ir = cv2.imread(os.path.join(ir_dir, img_name), 0)
        vi = cv2.imread(os.path.join(vi_dir, img_name), 0)
        original = cv2.imread(os.path.join(original_dir, img_name), 0)
        enhanced = cv2.imread(os.path.join(enhanced_dir, img_name), 0)
        
        if ir is None or vi is None or original is None or enhanced is None:
            continue
        
        axes[i, 0].imshow(ir, cmap='gray')
        axes[i, 0].set_title('Infrared', fontsize=12)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(vi, cmap='gray')
        axes[i, 1].set_title('Visible', fontsize=12)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(original, cmap='gray')
        axes[i, 2].set_title('Original SwinFusion', fontsize=12)
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(enhanced, cmap='gray')
        axes[i, 3].set_title('Enhanced SwinFusion', fontsize=12)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"对比图已保存: {save_path}")

if __name__ == "__main__":
    ir_dir = './Dataset/testsets/MSRS/IR/'
    vi_dir = './Dataset/testsets/MSRS/VI_Y/'
    original_dir = './results/SwinFusion_MSRS/'
    enhanced_dir = './results/SwinFusion_Enhanced/'
    
    visualize_comparison(ir_dir, vi_dir, original_dir, enhanced_dir)