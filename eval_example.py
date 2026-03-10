import os
import sys
import csv
import argparse
from tqdm import tqdm
import numpy as np
import cv2


# Ensure project root is on sys.path so `from utils.Evaluator` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from Evaluator import Evaluator, image_read_cv2

# ====== 可在此处直接填写数据集路径 ======
# 如果填写了下面任一变量，脚本将使用这些路径而不是要求命令行参数。
# 示例：
# DEFAULT_FUSED_DIR = r'D:\dataset\fused'
# DEFAULT_SRCA_DIR = r'D:\dataset\vis'
# DEFAULT_SRCB_DIR = r'D:\dataset\ir'
# 或者使用 mapping 文件： DEFAULT_MAPPING_FILE = r'D:\dataset\mapping.txt'
DEFAULT_FUSED_DIR = 'results\\SwinFusion_EAM'
DEFAULT_SRCA_DIR = 'dataset\\CT-MRI\\test\\CT'
DEFAULT_SRCB_DIR = 'dataset\\CT-MRI\\test\\MRI'
DEFAULT_MAPPING_FILE = ''
# =============================================


def to_gray_if_needed(img, mode):
	# image_read_cv2 returns float32 arrays.
	if mode == 'YCrCb':
		# YCrCb returns 3-channel, Y is channel 0
		if img.ndim == 3:
			return img[:, :, 0]
		return img
	elif mode == 'RGB':
		if img.ndim == 3:
			return np.mean(img, axis=2)
		return img
	else:  # GRAY
		if img.ndim == 3:
			return np.mean(img, axis=2)
		return img


def main():
	parser = argparse.ArgumentParser(description='Batch evaluate fused images against two sources')
	parser.add_argument('--fused_dir', required=False)
	parser.add_argument('--srcA_dir', required=False)
	parser.add_argument('--srcB_dir', required=False)
	parser.add_argument('--mapping_file', required=False, help='Optional: text/csv file with rows: fused_path,srcA_path,srcB_path (absolute or relative)')
	parser.add_argument('--out_csv', default='eval_results.csv')
	parser.add_argument('--mode', choices=['GRAY', 'YCrCb', 'RGB'], default='GRAY', help='How to read images; if YCrCb, take Y channel')
	args = parser.parse_args()

	# allow hardcoded defaults at top of file to override CLI
	fused_dir = DEFAULT_FUSED_DIR or args.fused_dir
	srcA_dir = DEFAULT_SRCA_DIR or args.srcA_dir
	srcB_dir = DEFAULT_SRCB_DIR or args.srcB_dir
	mapping_file = DEFAULT_MAPPING_FILE or args.mapping_file
	out_csv = args.out_csv
	mode = args.mode

	if not mapping_file and not (fused_dir and srcA_dir and srcB_dir):
		print('Error: must provide either --mapping_file or all of --fused_dir, --srcA_dir, --srcB_dir (or set defaults at top of file).')
		print('Run with -h for help.')
		return

	rows = []
	sums = {'MI':0.0, 'SCD':0.0, 'VIF':0.0, 'Qabf':0.0, 'SSIM':0.0}
	count = 0

	# 支持 mapping 文件或按目录匹配两种模式
	if mapping_file:
		if not os.path.exists(mapping_file):
			print('Mapping file not found:', mapping_file)
			return
		with open(mapping_file, 'r') as mf:
			for line in mf:
				line = line.strip()
				if not line or line.startswith('#'):
					continue
				parts = [p.strip() for p in line.split(',')] if ',' in line else line.split()
				if len(parts) < 3:
					print('Invalid mapping line (need 3 paths):', line)
					continue
				fused_path, a_path, b_path = parts[0], parts[1], parts[2]
				base_dir = os.path.dirname(os.path.abspath(mapping_file))
				if not os.path.isabs(fused_path):
					fused_path = os.path.join(base_dir, fused_path)
				if not os.path.isabs(a_path):
					a_path = os.path.join(base_dir, a_path)
				if not os.path.isabs(b_path):
					b_path = os.path.join(base_dir, b_path)

				if not (os.path.exists(fused_path) and os.path.exists(a_path) and os.path.exists(b_path)):
					print(f'Warning: files missing for mapping line, skipping:\n  {fused_path}\n  {a_path}\n  {b_path}')
					continue

				F = image_read_cv2(fused_path, mode=mode)
				A = image_read_cv2(a_path, mode=mode)
				B = image_read_cv2(b_path, mode=mode)

				F = to_gray_if_needed(F, mode)
				A = to_gray_if_needed(A, mode)
				B = to_gray_if_needed(B, mode)

				res = Evaluator.evaluate(F, A, B)
				row = {'file': os.path.basename(fused_path)}
				row.update(res)
				rows.append(row)
				for k in sums.keys():
					val = res.get(k, np.nan)
					if not np.isnan(val):
						sums[k] += val
				count += 1
	else:
		fused_files = sorted([f for f in os.listdir(fused_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])
		if len(fused_files) == 0:
			print('No image files found in', fused_dir)
			return

		for fname in tqdm(fused_files, desc='Evaluating'):
			fused_path = os.path.join(fused_dir, fname)
			a_path = os.path.join(srcA_dir, fname)
			b_path = os.path.join(srcB_dir, fname)
			if not (os.path.exists(a_path) and os.path.exists(b_path)):
				print(f'Warning: matching sources not found for {fname}, skipping')
				continue

			F = image_read_cv2(fused_path, mode=mode)
			A = image_read_cv2(a_path, mode=mode)
			B = image_read_cv2(b_path, mode=mode)

			F = to_gray_if_needed(F, mode)
			A = to_gray_if_needed(A, mode)
			B = to_gray_if_needed(B, mode)

			res = Evaluator.evaluate(F, A, B)
			row = {'file': fname}
			row.update(res)
			rows.append(row)
			for k in sums.keys():
				val = res.get(k, np.nan)
				if not np.isnan(val):
					sums[k] += val
			count += 1

	# write CSV
	if count > 0:
		with open(out_csv, 'w', newline='') as csvfile:
			fieldnames = ['file', 'MI', 'SCD', 'VIF', 'Qabf', 'SSIM']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()
			for r in rows:
				writer.writerow(r)
			avg_row = {'file':'AVERAGE'}
			for k in sums:
				avg_row[k] = sums[k] / count
			writer.writerow(avg_row)

		print(f'Evaluation done: {count} images. Results saved to {out_csv}')
		print('Averages:', {k: sums[k]/count for k in sums})
	else:
		print('No image pairs evaluated.')


if __name__ == '__main__':
	main()