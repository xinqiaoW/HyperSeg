# HyperSeg_lp 测试与结果保存说明

本项目提供两种与光谱信息相关的推理路径：
- 使用 HyperSeg（SAM + HyperFree 融合）的测试脚本
- 直接使用 HyperFree 或仅 SAM 生成并保存结果

本文档说明如何运行测试、如何获得与保存结果，以及输出文件的格式规范。

---

## 1. 环境与检查点
- **SAM 权重**：如 `sam_vit_h_*.pth` 或 `sam_vit_l_*.pth`
- **HyperFree 权重**：如 `HyperFree-h.pth`
- 设备：默认 `cuda`，可通过参数切换

建议将输出目录统一到：
- `outputs/hyperspectral_classification`：分类/评估与推理结果
- `outputs/out_images`：训练过程中的中间可视化（若使用训练脚本）

> 若数据路径与脚本内默认路径不同，请按需修改脚本或将数据放到相同相对路径。

---

## 2. 使用脚本测试 HyperSeg（融合 SAM + HyperFree）
脚本：`scripts/test.py`

典型命令（以 LongKou 数据集为例）：

```bash
python scripts/test.py \
  --dataset LongKou \
  --device cuda:0 \
  --sam_checkpoint /path/to/sam_vit_*.pth \
  --hyperfree_checkpoint /path/to/HyperFree-h.pth \
  --channel_proj_spectral
```

- **数据路径**：`scripts/test.py` 中 LongKou 的默认路径为：
  - 影像：`./Data/LongKou/WHU-Hi-LongKou.tif`
  - 标注：`./Data/LongKou/WHU-Hi-LongKou_gt.tif`
  若路径不同，请修改脚本顶部相应变量。
- **可选消融**：
  - 仅用 SAM（关闭光谱相关模块）：追加 `--ignore_hsi_module --ignore_spectral_query`，且可不传 `--hyperfree_checkpoint`。
  - 仅用 HyperFree：建议使用第 3 节的 HyperFree 基线推理方式。
- **保存位置**：脚本会创建并使用 `outputs/hyperspectral_classification` 作为输出目录。

运行后，会在 `outputs/hyperspectral_classification` 下保存：
- `mask_{clsIdx}_{compIdx}.png`：每个类别、每个连通域的二值掩膜
- `labelled.png`：整幅彩色标签图（随机调色）
- 以上图像尺寸与输入影像一致

---

## 3. 直接使用 HyperFree 基线生成与保存
当需单独评测 HyperFree 时，可直接调用 `hyperfree` 包。示例代码：

```python
# save as examples/run_hyperfree.py (or run in notebook)
import os, cv2, torch, numpy as np
from skimage import io
from hyperfree import build_HyperFree_vit_h
from hyperfree import SamAutomaticMaskGenerator  # hyperfree 版本的自动掩膜生成器

# 1) 读取 HSI：形状 [C, H, W]
img = io.imread('/path/to/HSI.tif')
img_norm = (img - img.min()) / (img.max() - img.min())
img_u8 = (img_norm * 255).astype(np.uint8)            # 自动生成器期望 HWC uint8
image_hwc = img_u8.transpose(1, 2, 0)                 # [H, W, C]

# 2) 构建 HyperFree 模型
model = build_HyperFree_vit_h(checkpoint='/path/to/HyperFree-h.pth')

# 3) 创建生成器（阈值可按需调整）
mask_generator = SamAutomaticMaskGenerator(
    model,
    points_per_side=32,
    pred_iou_thresh=0.0,
    stability_score_thresh=0.0,
)

# 可选：波段中心与地面分辨率（若有）
wavelengths = None                  # 或者传入 [float, ...]
GSD = torch.tensor([1.0])

# 4) 生成掩膜
anns = mask_generator.generate(image_hwc, spectral_lengths=wavelengths, GSD=GSD)

# 5) 保存结果（逐掩膜二值图）
os.makedirs('outputs/hyperspectral_classification', exist_ok=True)
for i, ann in enumerate(anns):
    m = ann['segmentation'].astype(np.uint8) * 255   # [H, W], 0/255
    cv2.imwrite(f'outputs/hyperspectral_classification/hf_mask_{i:04d}.png', m)
```

说明：
- `anns` 为列表，每个元素包含：`segmentation`(HxW 二值)、`bbox`(XYWH)、`area`、`predicted_iou`、`point_coords`、`stability_score`、`crop_box` 等。
- 如需将多掩膜合成为标签图，可自行为每个 `ann` 赋 `label` 并叠加；或按面积排序后覆盖写入。

---

## 4. 仅使用 SAM 生成与保存
若希望只用 SAM（不使用光谱模块与 HyperFree），可沿用第 2 节脚本并禁用光谱分支：

```bash
python scripts/test.py \
  --dataset LongKou \
  --device cuda:0 \
  --sam_checkpoint /path/to/sam_vit_*.pth \
  --ignore_hsi_module --ignore_spectral_query
```

或直接使用 `hyperseg` 包的自动生成器（同第 3 节流程），区别在于模型的构建与调用：

```python
# 简要示例：使用 hyperseg 自动掩膜生成器
import os, cv2, numpy as np, torch
from skimage import io
from hyperseg import build_seg_vit_h
from hyperseg.automatic_mask_generator import SamAutomaticMaskGenerator

img = io.imread('/path/to/HSI.tif')                  # [C, H, W]
img_u8 = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
image_hwc = img_u8.transpose(1, 2, 0)

sam_only = build_seg_vit_h(
    sam_checkpoint='/path/to/sam_vit_*.pth',
    hyperfree_checkpoint=None,
    ignore_hsi_module=True,
    ignore_spectral_query=True,
)
mask_generator = SamAutomaticMaskGenerator(sam_only, points_per_side=32)
anns, _ = mask_generator.generate(image_hwc, wavelengths=None, GSD=torch.tensor([1.0]))

os.makedirs('outputs/hyperspectral_classification', exist_ok=True)
for i, ann in enumerate(anns):
    cv2.imwrite(f'outputs/hyperspectral_classification/sam_mask_{i:04d}.png', ann['segmentation'].astype(np.uint8) * 255)
```

---

## 5. 输出与保存格式说明
- **二值掩膜 PNG**：
  - 文件名：`mask_{clsIdx}_{compIdx}.png` 或 `sam_mask_{i:04d}.png` / `hf_mask_{i:04d}.png`
  - 尺寸：与输入影像一致（H×W）
  - 类型：8-bit 单通道，像素值 0/255（0 为背景，255 为前景）
- **彩色标签/可视化**：
  - 文件名：`labelled.png`（来自脚本的汇总随机上色）或自定义 `overlay.png`
  - 尺寸：H×W×3，8-bit，RGB 0–255
- **结果字典（内存对象）**：
  - 键：`segmentation`(HxW)、`area`、`bbox`(XYWH)、`predicted_iou`、`point_coords`、`stability_score`、`crop_box`
  - 可按需序列化为 JSON/Numpy：如将 `segmentation` 堆叠保存到 `mask_stack.npy`（形状 `[N, H, W]`，`uint8`）

---

## 6. 常见问题
- 路径不存在/读取失败：请检查数据与权重路径；若使用 `scripts/test.py`，可在脚本中修改 `data_path` 与 `gt_path`。
- 颜色随机：`labelled.png` 使用随机颜色，可根据需要改为固定调色板。
- 速度与显存：`points_per_side` 越大，生成点越密集，显存与耗时会增加，可根据 GPU 能力调整。