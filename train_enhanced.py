"""
SwinFusion Enhanced 训练脚本
用于训练优化后的图像融合模型
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from tqdm import tqdm
import math

# 导入增强版模型
from models.network_swinfusion_enhanced import SwinFusionEnhanced


# ======================= 损失函数 =======================
class SSIMLoss(nn.Module):
    """SSIM 损失函数"""
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, img1, img2):
        channel = img1.size(1)
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window.to(img1.device)
        else:
            window = self._create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel)


class GradientLoss(nn.Module):
    """梯度损失 - 保持边缘细节"""
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Sobel算子
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, img1, img2):
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        # 计算梯度
        pred_grad_x = torch.nn.functional.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = torch.nn.functional.conv2d(pred, sobel_y, padding=1)
        pred_grad = torch.abs(pred_grad_x) + torch.abs(pred_grad_y)

        img1_grad_x = torch.nn.functional.conv2d(img1, sobel_x, padding=1)
        img1_grad_y = torch.nn.functional.conv2d(img1, sobel_y, padding=1)
        img1_grad = torch.abs(img1_grad_x) + torch.abs(img1_grad_y)

        img2_grad_x = torch.nn.functional.conv2d(img2, sobel_x, padding=1)
        img2_grad_y = torch.nn.functional.conv2d(img2, sobel_y, padding=1)
        img2_grad = torch.abs(img2_grad_x) + torch.abs(img2_grad_y)

        # 取最大梯度作为目标
        target_grad = torch.max(img1_grad, img2_grad)

        return torch.nn.functional.l1_loss(pred_grad, target_grad)


class IntensityLoss(nn.Module):
    """强度损失"""
    def __init__(self, mode='max'):
        super(IntensityLoss, self).__init__()
        self.mode = mode

    def forward(self, pred, img1, img2):
        if self.mode == 'max':
            target = torch.max(img1, img2)
        else:
            target = (img1 + img2) / 2
        return torch.nn.functional.l1_loss(pred, target)


class PSNRLoss(nn.Module):
    """PSNR 损失函数"""
    def __init__(self):
        super(PSNRLoss, self).__init__()

class PSNRLoss(nn.Module):
    """PSNR 损失函数"""
    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        # 避免 mse 为 0 或 nan
        mse = torch.clamp(mse, min=1e-10, max=1.0)
        # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        # 为了最小化损失，我们使用 -PSNR (所以越小越好)
        max_i = torch.tensor(1.0, device=mse.device, dtype=mse.dtype)
        psnr = 20 * torch.log10(max_i) - 10 * torch.log10(mse)
        return -psnr  # 负值，因为我们想最大化 PSNR


class QabfLoss(nn.Module):
    """Qabf 损失函数 (基于边缘保持)"""
    def __init__(self):
        super(QabfLoss, self).__init__()
        # Sobel算子
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, img1, img2):
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        def get_edge_info(img):
            gx = torch.nn.functional.conv2d(img, sobel_x, padding=1)
            gy = torch.nn.functional.conv2d(img, sobel_y, padding=1)
            mag = torch.sqrt(gx**2 + gy**2)
            angle = torch.atan2(gy, gx)
            return mag, angle

        gA, aA = get_edge_info(img1)
        gB, aB = get_edge_info(img2)
        gF, aF = get_edge_info(pred)

        # Qabf 参数 (简化的版本)
        Tg, kg, Dg = 0.9994, -15, 0.5
        Ta, ka, Da = 0.9879, -22, 0.8

        def get_Q(g_src, a_src, g_f, a_f):
            Qg = Tg / (1 + torch.exp(kg * (g_src - g_f) + Dg))
            Qa = Ta / (1 + torch.exp(ka * (torch.abs(a_src - a_f) - Da)))
            return Qg * Qa

        QA = get_Q(gA, aA, gF, aF)
        QB = get_Q(gB, aB, gF, aF)

        # 加权平均
        total_weight = torch.sum(gA + gB) + 1e-8
        weight = (gA + gB) / total_weight
        qabf = torch.sum((QA * gA + QB * gB) * weight) / total_weight
        
        # 确保 qabf 在合理范围内
        qabf = torch.clamp(qabf, min=0.0, max=1.0)
        
        return 1 - qabf  # 转换为损失 (1 - Qabf)


class FusionLoss(nn.Module):
    """融合损失函数 = SSIM + 梯度 + 强度 + PSNR + Qabf"""
    def __init__(self, lambda_ssim=15, lambda_grad=20, lambda_int=20, lambda_psnr=1, lambda_qabf=2):
        super(FusionLoss, self).__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.lambda_int = lambda_int
        self.lambda_psnr = lambda_psnr
        self.lambda_qabf = lambda_qabf
        
        self.ssim_loss = SSIMLoss()
        self.grad_loss = GradientLoss()
        self.int_loss = IntensityLoss(mode='max')
        self.psnr_loss = PSNRLoss()
        self.qabf_loss = QabfLoss()

class FusionLoss(nn.Module):
    """融合损失函数 = SSIM + 梯度 + 强度 + PSNR + Qabf"""
    def __init__(self, lambda_ssim=15, lambda_grad=20, lambda_int=20, lambda_psnr=1, lambda_qabf=2):
        super(FusionLoss, self).__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.lambda_int = lambda_int
        self.lambda_psnr = lambda_psnr
        self.lambda_qabf = lambda_qabf
        
        self.ssim_loss = SSIMLoss()
        self.grad_loss = GradientLoss()
        self.int_loss = IntensityLoss(mode='max')
        # 暂时注释掉新的损失函数
        # self.psnr_loss = PSNRLoss()
        # self.qabf_loss = QabfLoss()

    def forward(self, pred, img1, img2):
        # SSIM损失
        ssim_loss1 = self.ssim_loss(pred, img1)
        ssim_loss2 = self.ssim_loss(pred, img2)
        ssim_loss = 0.5 * (ssim_loss1 + ssim_loss2)
        
        # 梯度损失
        grad_loss = self.grad_loss(pred, img1, img2)
        
        # 强度损失
        int_loss = self.int_loss(pred, img1, img2)
        
        # 暂时注释掉新的损失
        # PSNR损失 (与两个源图像比较)
        # psnr_loss1 = self.psnr_loss(pred, img1)
        # psnr_loss2 = self.psnr_loss(pred, img2)
        # psnr_loss = 0.5 * (psnr_loss1 + psnr_loss2)
        
        # Qabf损失
        # qabf_loss = self.qabf_loss(pred, img1, img2)
        
        # 检查是否有 nan
        if torch.isnan(ssim_loss) or torch.isnan(grad_loss) or torch.isnan(int_loss):
            print(f"NaN detected: SSIM={ssim_loss.item():.4f}, Grad={grad_loss.item():.4f}, Int={int_loss.item():.4f}")
        
        # 总损失
        total_loss = self.lambda_ssim * ssim_loss + \
                     self.lambda_grad * grad_loss + \
                     self.lambda_int * int_loss
                     # + self.lambda_psnr * psnr_loss + \
                     # self.lambda_qabf * qabf_loss
        
        return total_loss, {
            'ssim': ssim_loss.item(),
            'grad': grad_loss.item(),
            'int': int_loss.item(),
            'psnr': 0.0,  # 暂时设为0
            'qabf': 0.0,  # 暂时设为0
            'total': total_loss.item()
        }


# ======================= 数据集 =======================
class FusionDataset(Dataset):
    """VIF数据集"""
    def __init__(self, ir_dir, vi_dir, patch_size=128, augment=True):
        self.ir_dir = ir_dir
        self.vi_dir = vi_dir
        self.patch_size = patch_size
        self.augment = augment
        
        self.ir_list = sorted([f for f in os.listdir(ir_dir) 
                               if f.endswith(('.png', '.jpg', '.bmp'))])
        print(f"数据集大小: {len(self.ir_list)}")
        
    def __len__(self):
        return len(self.ir_list)
    
    def __getitem__(self, idx):
        img_name = self.ir_list[idx]
        
        # 读取图像
        ir_path = os.path.join(self.ir_dir, img_name)
        vi_path = os.path.join(self.vi_dir, img_name)
        
        ir_img = cv2.imread(ir_path, 0)
        vi_img = cv2.imread(vi_path, 0)
        
        if ir_img is None or vi_img is None:
            # 如果读取失败，返回随机噪声
            ir_img = np.random.randint(0, 255, (self.patch_size, self.patch_size), dtype=np.uint8)
            vi_img = np.random.randint(0, 255, (self.patch_size, self.patch_size), dtype=np.uint8)
        
        h, w = ir_img.shape
        
        # 随机裁剪
        if h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            ir_img = ir_img[top:top+self.patch_size, left:left+self.patch_size]
            vi_img = vi_img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            ir_img = cv2.resize(ir_img, (self.patch_size, self.patch_size))
            vi_img = cv2.resize(vi_img, (self.patch_size, self.patch_size))
        
        # 数据增强
        if self.augment:
            # 随机水平翻转
            if np.random.rand() > 0.5:
                ir_img = np.fliplr(ir_img).copy()
                vi_img = np.fliplr(vi_img).copy()
            # 随机垂直翻转
            if np.random.rand() > 0.5:
                ir_img = np.flipud(ir_img).copy()
                vi_img = np.flipud(vi_img).copy()
            # 随机旋转90度
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                ir_img = np.rot90(ir_img, k).copy()
                vi_img = np.rot90(vi_img, k).copy()
        
        # 转换为tensor
        ir_tensor = torch.from_numpy(ir_img.astype(np.float32)).unsqueeze(0) / 255.0
        vi_tensor = torch.from_numpy(vi_img.astype(np.float32)).unsqueeze(0) / 255.0
        
        return ir_tensor, vi_tensor


# ======================= 训练函数 =======================
def validate(model, val_loader, criterion, device):
    """验证函数"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for ir, vi in val_loader:
            ir = ir.to(device)
            vi = vi.to(device)
            fused = model(ir, vi)
            loss, _ = criterion(fused, ir, vi)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def train(args):
    # 设备设置
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用 Apple MPS 加速")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用 CPU (训练会较慢)")
    
    # 创建保存目录
    save_dir = f'./experiments/{args.exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/models', exist_ok=True)
    os.makedirs(f'{save_dir}/logs', exist_ok=True)
    
    # 保存配置
    with open(f'{save_dir}/config.txt', 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
    
    # 加载数据
    print("\n加载训练数据...")
    train_dataset = FusionDataset(
        ir_dir=args.ir_dir,
        vi_dir=args.vi_dir,
        patch_size=args.patch_size,
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Mac上设为0
        pin_memory=True,
        drop_last=True
    )
    
    # 加载验证数据
    val_ir_dir = args.ir_dir.replace('train', 'test')
    val_vi_dir = args.vi_dir.replace('train', 'test')
    print(f"\n加载验证数据... (IR: {val_ir_dir}, VI: {val_vi_dir})")
    val_dataset = FusionDataset(
        ir_dir=val_ir_dir,
        vi_dir=val_vi_dir,
        patch_size=args.patch_size,
        augment=False  # 验证时不进行数据增强
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    # 创建模型
    print("\n创建模型...")
    model = SwinFusionEnhanced(
        img_size=args.patch_size,
        in_chans=1,
        embed_dim=args.embed_dim,
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
        # 优化模块开关
        use_cbam=args.use_cbam,
        use_edge_aware=args.use_edge_aware,
        use_se=args.use_se,
        use_multi_scale=args.use_multi_scale
    ).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    # 损失函数
    criterion = FusionLoss(
        lambda_ssim=args.lambda_ssim,
        lambda_grad=args.lambda_grad,
        lambda_int=args.lambda_int,
        lambda_psnr=args.lambda_psnr,
        lambda_qabf=args.lambda_qabf
    )
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 训练日志
    log_file = open(f'{save_dir}/logs/train_log.txt', 'w')
    log_file.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Config: {vars(args)}\n\n")
    
    # 开始训练
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    best_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_ssim = 0
        epoch_grad = 0
        epoch_int = 0
        # epoch_psnr = 0
        # epoch_qabf = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        
        for batch_idx, (ir, vi) in enumerate(pbar):
            ir = ir.to(device)
            vi = vi.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            fused = model(ir, vi)
            
            # 检查输入和输出
            if torch.isnan(ir).any() or torch.isnan(vi).any():
                print(f"输入包含 NaN: IR={torch.isnan(ir).any().item()}, VI={torch.isnan(vi).any().item()}")
            if torch.isnan(fused).any():
                print(f"模型输出包含 NaN")
                print(f"输入范围: IR=[{ir.min().item():.4f}, {ir.max().item():.4f}], VI=[{vi.min().item():.4f}, {vi.max().item():.4f}]")
            
            # 计算损失
            loss, loss_dict = criterion(fused, ir, vi)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            epoch_loss += loss_dict['total']
            epoch_ssim += loss_dict['ssim']
            epoch_grad += loss_dict['grad']
            epoch_int += loss_dict['int']
            # epoch_psnr += loss_dict['psnr']
            # epoch_qabf += loss_dict['qabf']
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'ssim': f"{loss_dict['ssim']:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_ssim = epoch_ssim / num_batches
        avg_grad = epoch_grad / num_batches
        avg_int = epoch_int / num_batches
        # avg_psnr = epoch_psnr / num_batches
        # avg_qabf = epoch_qabf / num_batches
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 记录日志
        elapsed = time.time() - start_time
        log_msg = f"Epoch [{epoch}/{args.epochs}] " \
                  f"Train Loss: {avg_loss:.4f} " \
                  f"Val Loss: {val_loss:.4f} " \
                  f"SSIM: {avg_ssim:.4f} " \
                  f"LR: {scheduler.get_last_lr()[0]:.6f} " \
                  f"Time: {elapsed/60:.1f}min"
        
        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型（基于验证损失）
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f'{save_dir}/models/best_model.pth')
            print(f"  >> 保存最佳模型 (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  >> 验证损失未改善，耐心计数: {patience_counter}/{args.patience}")
        
        if patience_counter >= args.patience:
            print(f"\n早停触发！连续 {args.patience} 轮验证损失无改善")
            break
        
        # 保存检查点
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f'{save_dir}/models/epoch_{epoch}.pth')
            print(f"  >> 保存检查点: epoch_{epoch}.pth")
    
    # 训练完成
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"训练完成! 总用时: {total_time/3600:.2f}小时")
    print(f"训练了 {epoch} 轮")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {save_dir}/models/")
    print("="*60)
    
    log_file.write(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Total time: {total_time/3600:.2f} hours\n")
    log_file.write(f"Trained for {epoch} epochs\n")
    log_file.write(f"Best validation loss: {best_val_loss:.4f}\n")
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Enhanced SwinFusion')
    
    # 数据路径
    parser.add_argument('--ir_dir', type=str, 
                        default='D:\\tkn\\SwinFusion-Enhanced\\dataset\\CT-MRI\\train\\CT',
                        help='红外图像目录')
    parser.add_argument('--vi_dir', type=str, 
                        default='D:\\tkn\\SwinFusion-Enhanced\\dataset\\CT-MRI\\train\\MRI',
                        help='可见光图像目录')
    
    # 实验名称
    parser.add_argument('--exp_name', type=str, 
                        default='SwinFusion_Enhanced_EAM',
                        help='实验名称')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=5000, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--patch_size', type=int, default=128, help='图像块大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--embed_dim', type=int, default=60, help='嵌入维度')
    
    # 损失函数权重
    parser.add_argument('--lambda_ssim', type=float, default=15, help='SSIM损失权重')
    parser.add_argument('--lambda_grad', type=float, default=20, help='梯度损失权重')
    parser.add_argument('--lambda_int', type=float, default=20, help='强度损失权重')
    parser.add_argument('--lambda_psnr', type=float, default=1, help='PSNR损失权重')
    parser.add_argument('--lambda_qabf', type=float, default=2, help='Qabf损失权重')
    
    # 优化模块开关
    parser.add_argument('--use_cbam', action='store_true', default=True, help='使用CBAM')
    parser.add_argument('--use_edge_aware', action='store_true', default=False, help='使用边缘感知')
    parser.add_argument('--use_se', action='store_true', default=True, help='使用SE注意力')
    parser.add_argument('--use_multi_scale', action='store_true', default=True, help='使用多尺度融合')
    
    # 其他
    parser.add_argument('--save_freq', type=int, default=10, help='保存频率')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值（连续多少轮验证损失无改善则停止）')
    
    args = parser.parse_args()
    
    train(args)