"""
测试增强版 SwinFusion
"""

import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import time
import argparse

from models.network_swinfusion_enhanced import SwinFusionEnhanced


def test(args):
    # 设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    model = SwinFusionEnhanced(
        img_size=128,
        in_chans=1,
        embed_dim=60,
        Ex_depths=[4],
        Fusion_depths=[2, 2],
        Re_depths=[4],
        Ex_num_heads=[6],
        Fusion_num_heads=[6, 6],
        Re_num_heads=[6],
        window_size=8,
        mlp_ratio=2.,
        upscale=1,
        img_range=1.,
        use_cbam=True,
        use_edge_aware=False,
        use_se=True,
        use_multi_scale=True
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载成功! (Epoch: {checkpoint.get('epoch', 'N/A')})")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 获取测试图像
    ir_list = sorted([f for f in os.listdir(args.ir_dir) 
                      if f.endswith(('.png', '.jpg', '.bmp'))])
    
    print(f"测试 {len(ir_list)} 张图像...")
    total_time = 0
    
    for img_name in tqdm(ir_list):
        # 读取图像
        ir_img = cv2.imread(os.path.join(args.ir_dir, img_name), 0)
        vi_img = cv2.imread(os.path.join(args.vi_dir, img_name), 0)
        
        if ir_img is None or vi_img is None:
            print(f"跳过: {img_name}")
            continue
        
        # 转换为tensor
        ir_tensor = torch.from_numpy(ir_img.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
        vi_tensor = torch.from_numpy(vi_img.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
        
        # 融合
        start_time = time.time()
        with torch.no_grad():
            ir_tensor = ir_tensor.to(device)
            vi_tensor = vi_tensor.to(device)
            fused = model(ir_tensor, vi_tensor)
        total_time += time.time() - start_time
        
        # 保存结果
        fused_img = (fused.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.save_dir, img_name), fused_img)
    
    print(f"\n测试完成!")
    print(f"平均处理时间: {total_time / len(ir_list):.2f}秒/张")
    print(f"结果保存在: {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Enhanced SwinFusion')
    
    parser.add_argument('--model_path', type=str, 
                        default='D:\\tkn\\SwinFusion-Enhanced\\experiments\\SwinFusion_Enhanced_EAM\\models\\best_model.pth')
    parser.add_argument('--ir_dir', type=str, 
                        default='D:\\tkn\\SwinFusion-Enhanced\\dataset\\CT-MRI\\test\\CT')
    parser.add_argument('--vi_dir', type=str, 
                        default='D:\\tkn\\SwinFusion-Enhanced\\dataset\\CT-MRI\\test\\MRI')
    parser.add_argument('--save_dir', type=str, 
                        default='./results/SwinFusion_EAM/')
    
    args = parser.parse_args()
    test(args)