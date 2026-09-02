import os
import sys
# Windows 下 torch/numpy 等混合科学依赖会多次初始化 Intel OpenMP 运行时，
# 若不设置会在运行时报 "OMP: Error #15 ... libiomp5md.dll already initialized"，
# 导致推理中途崩溃。必须在 import torch 之前设置（与 agent.py 同规则）。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 预防 single OpenMP runtime 绑定导致的性能/崩溃问题。
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import time
import glob
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import datetime  # ✅ 新增：用于显示时间

# Windows 控制台默认 GBK 编码，无法编码 emoji（✔/❌/✅ 等），会导致
# print 抛 UnicodeEncodeError，即使推理已成功也会被误判为失败。
# 这里强制 stdout/stderr 使用 UTF-8，避免 GBK 控制台崩掉。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 假设模型文件名为 YYnet.py
from YYnet import CDNet
from agent_context_policy import safe_error_summary

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _resolve_device(device) -> str:
    """把调用方可能传入的 'auto'/''/None 规范化为有效设备字符串。

    torch.load(map_location=...) 与 model.to(...) 只接受 'cuda'/'cpu' 等有效值，
    若直接传字符串 'auto' 会抛 RuntimeError。统一在此解析为实际可用的设备。
    """
    try:
        if device and str(device).lower() not in ("auto", "none", "cpu", "cuda"):
            # 其它字符串（如 'cuda:0'）原样返回，交由 torch 校验
            return str(device)
    except Exception:
        pass
    import torch as _t
    return "cuda" if _t.cuda.is_available() else "cpu"


# =======================================================
#  1. 复刻脚本逻辑 + ✅ 生成缩略图掩码
# =======================================================
def compute_stats_and_mask(src):
    scale_factor = max(src.width, src.height) // 1024
    scale_factor = max(1, scale_factor)

    target_h = int(src.height / scale_factor)
    target_w = int(src.width / scale_factor)

    try:
        thumb = src.read([1, 2, 3], out_shape=(3, target_h, target_w)).transpose(1, 2, 0)
    except Exception:
        return None, None, None

    thumb = np.nan_to_num(thumb, nan=0.0, posinf=1.0, neginf=0.0)
    valid_mask = np.max(thumb, axis=2) > 0.001

    thumb = np.clip(thumb, 0, 1)
    thumb_uint8 = (thumb * 255).astype(np.uint8)

    if np.max(thumb_uint8) == 0: return "EMPTY", None, None

    stats = []
    clip_limit = 1.0
    for i in range(3):
        band = thumb_uint8[:, :, i]
        valid_pixels = band[band > 0]
        if valid_pixels.size == 0:
            stats.append((0, 255))
            continue
        min_val = np.percentile(valid_pixels, 0.5)
        max_val = np.percentile(valid_pixels, 99.5)
        min_val = max(0, min_val - clip_limit)
        max_val = min(255, max_val + clip_limit)
        stats.append((min_val, max_val))

    return stats, valid_mask, scale_factor


# =======================================================
#  2. GPU 预处理器
# =======================================================
class GPUPreProcessor(torch.nn.Module):
    def __init__(self, stats, device):
        super().__init__()
        self.device = device
        mins = torch.tensor([s[0] for s in stats], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        maxs = torch.tensor([s[1] for s in stats], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.min_vals = mins
        self.scales = 255.0 / (maxs - mins + 1e-6)
        self.mean = torch.tensor((0.3876, 0.4297, 0.4462), dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.2091, 0.1981, 0.1846), dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def forward(self, batch_tensor):
        x = torch.nan_to_num(batch_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        x = torch.clamp(x, 0, 1) * 255.0
        x = torch.floor(x)
        x = (x - self.min_vals) * self.scales
        x = torch.clamp(x, 0, 255)
        x = x / 255.0
        x = (x - self.mean) / self.std
        return x


# =======================================================
#  3. 余弦权重 (保持不变，确保函数存在)
# =======================================================
def get_cosine_weights(patch_size, overlap):
    half_overlap = overlap / 2

    def create_cosine_1d(length):
        weights = np.ones(length, dtype=np.float32)
        transition_len = int(half_overlap)
        for i in range(transition_len):
            angle = np.pi / 2 * (i / transition_len)
            weights[i] = np.sin(angle)
        for i in range(transition_len):
            angle = np.pi / 2 * (i / transition_len)
            weights[length - 1 - i] = np.sin(angle)
        return weights

    weights_x = create_cosine_1d(patch_size)
    weights_y = create_cosine_1d(patch_size)
    weights = np.outer(weights_y, weights_x)
    return weights.astype(np.float32)


# =======================================================
#  4. 智能坐标生成
# =======================================================
def get_smart_inference_coords(H, W, patch_size, stride, valid_mask, scale_factor):
    if H < patch_size or W < patch_size: return [(0, 0)]

    h_coords = list(range(0, H - patch_size, stride)) + [H - patch_size]
    h_coords = sorted(list(set(h_coords)))

    w_coords = list(range(0, W - patch_size, stride)) + [W - patch_size]
    w_coords = sorted(list(set(w_coords)))

    final_coords = []
    mask_h, mask_w = valid_mask.shape

    for top in h_coords:
        for left in w_coords:
            t_start = int(top / scale_factor)
            t_end = int((top + patch_size) / scale_factor)
            l_start = int(left / scale_factor)
            l_end = int((left + patch_size) / scale_factor)

            t_end = min(t_end, mask_h)
            l_end = min(l_end, mask_w)

            mask_patch = valid_mask[t_start:t_end, l_start:l_end]

            if np.any(mask_patch):
                final_coords.append((top, left))

    return final_coords


# =======================================================
#  5. Dataset
# =======================================================
class RawInferenceDataset(Dataset):
    def __init__(self, tiff_path, coords, patch_size):
        self.tiff_path = tiff_path
        self.coords = coords
        self.patch_size = patch_size

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        top, left = self.coords[idx]
        window = Window(left, top, self.patch_size, self.patch_size)
        try:
            with rasterio.open(self.tiff_path) as src:
                img = src.read([1, 2, 3], window=window).astype(np.float32)
        except Exception:
            img = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
        # 边缘窗口读到的尺寸可能小于 patch_size；补零到固定尺寸，便于 DataLoader 组 batch
        h, w = int(img.shape[1]), int(img.shape[2])
        if h < self.patch_size or w < self.patch_size:
            padded = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
            padded[:, :h, :w] = img
            img = padded
        return img, top, left


# =======================================================
#  6. 智能拼接器
# =======================================================
class SmartStitcher:
    # ... (保持原样不动) ...
    def __init__(self, H, W, device):
        self.H = H
        self.W = W
        self.device = device
        required_mb = (H * W * 4 * 2) / (1024 ** 2)
        self.mode = "cpu_ram"
        self.gpu_res = None
        self.gpu_wgt = None
        self.cpu_res = None
        self.cpu_wgt = None
        if required_mb < 3000:
            try:
                self.gpu_res = torch.zeros((H, W), dtype=torch.float32, device=device)
                self.gpu_wgt = torch.zeros((H, W), dtype=torch.float32, device=device)
                self.mode = "gpu"
            except Exception:
                pass
        if self.mode != "gpu":
            self.cpu_res = np.zeros((H, W), dtype=np.float32)
            self.cpu_wgt = np.zeros((H, W), dtype=np.float32)
            self.mode = "cpu_ram"

    def add_batch(self, preds, batch_weights, tops, lefts):
        patch_size = preds.shape[1]
        if self.mode == "gpu":
            weighted_preds = preds * batch_weights
            for i in range(len(tops)):
                t = int(tops[i].item() if torch.is_tensor(tops[i]) else tops[i])
                l = int(lefts[i].item() if torch.is_tensor(lefts[i]) else lefts[i])
                # 与影像真实范围对齐：窗口右下角可能越界，切片高度/宽度会小于 patch_size
                ah = min(patch_size, self.H - t)
                aw = min(patch_size, self.W - l)
                if ah <= 0 or aw <= 0:
                    continue
                self.gpu_res[t : t + ah, l : l + aw] += weighted_preds[i, :ah, :aw]
                self.gpu_wgt[t : t + ah, l : l + aw] += batch_weights[i, :ah, :aw]
        else:
            weighted_preds = (preds * batch_weights).cpu().numpy()
            tops_np = tops.numpy() if torch.is_tensor(tops) else np.asarray(tops)
            lefts_np = lefts.numpy() if torch.is_tensor(lefts) else np.asarray(lefts)
            bw_np = batch_weights.cpu().numpy()
            for i in range(len(tops)):
                t, l = int(tops_np[i]), int(lefts_np[i])
                ah = min(patch_size, self.H - t)
                aw = min(patch_size, self.W - l)
                if ah <= 0 or aw <= 0:
                    continue
                self.cpu_res[t : t + ah, l : l + aw] += weighted_preds[i, :ah, :aw]
                self.cpu_wgt[t : t + ah, l : l + aw] += bw_np[i, :ah, :aw]

    def finalize(self):
        if self.mode == "gpu":
            final = self.gpu_res / (self.gpu_wgt + 1e-6)
            binary = (final > 0.5).byte().cpu().numpy() * 255
            del self.gpu_res, self.gpu_wgt
            torch.cuda.empty_cache()
            return binary
        else:
            final = self.cpu_res / (self.cpu_wgt + 1e-6)
            binary = (final > 0.5).astype(np.uint8) * 255
            return binary


# =======================================================
#  7. ✅ 主处理流程 (装载物理刹车 stop_callback)
# =======================================================
def process_geotiff(model, tiff_path, output_path, device, current_idx=0, total_batch=0, stop_callback=None):
    device = _resolve_device(device)
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    progress_info = f"[{current_idx}/{total_batch}]" if total_batch > 0 else ""

    print(f"\n=== {time_str} {progress_info} 正在处理: {os.path.basename(tiff_path)} ===")

    patch_size = 1024
    overlap = 512
    stride = patch_size - overlap
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    cosine_map_np = get_cosine_weights(patch_size, overlap)
    cosine_map_cuda = torch.from_numpy(cosine_map_np).to(device)

    try:
        with rasterio.open(tiff_path) as src:
            stats, valid_mask, scale_factor = compute_stats_and_mask(src)

            if stats == "EMPTY": return True
            if stats is None: return True

            H, W = src.height, src.width
            crs, transform = src.crs, src.transform

            coords = get_smart_inference_coords(H, W, patch_size, stride, valid_mask, scale_factor)

            if len(coords) == 0:
                print("⚠️ 跳过: 影像有效区域不足一个切片")
                return True

            print(f"  > 有效切片数: {len(coords)} (已过滤无效背景)")

        stitcher = SmartStitcher(H, W, device)
        gpu_preprocessor = GPUPreProcessor(stats, device)
        dataset = RawInferenceDataset(tiff_path, coords, patch_size)

        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )

        # 🚨 最关键的急刹车逻辑：每次预测一批图片前，听一下外面是否喊停
        for raw_batch, tops, lefts in tqdm(loader, desc=f"Inference ({stitcher.mode})"):
            if stop_callback and stop_callback():
                print(f"\n🚨 收到中断信号！立刻终止 {os.path.basename(tiff_path)} 的 GPU 推理！")
                return False # 返回 False 代表被中途打断

            raw_batch = raw_batch.to(device, non_blocking=True)

            with autocast():
                with torch.no_grad():
                    input_tensor = gpu_preprocessor(raw_batch)
                    _, outputs, _ = model(input_tensor, (patch_size, patch_size))
                    outputs = F.interpolate(outputs, size=(patch_size, patch_size),
                                            mode="bilinear", align_corners=False)
                    preds = torch.sigmoid(outputs).squeeze(1)

            preds = preds.float()
            batch_weights = cosine_map_cuda.unsqueeze(0).expand(preds.shape[0], -1, -1)
            stitcher.add_batch(preds, batch_weights, tops, lefts)

        binary = stitcher.finalize()
        with rasterio.open(output_path, "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="uint8", crs=crs, transform=transform, compress='lzw') as dst:
            dst.write(binary, 1)

        print(f"✔ 完成: {output_path}")
        return True # 返回 True 代表顺利完成一张图

    except Exception as e:
        print(f"❌ 失败: {safe_error_summary(e)}")
        import traceback
        traceback.print_exc()
        return False




# =======================================================
#  主程序
# =======================================================
def load_model(model_path, device):
    print(f">>> 加载模型: {os.path.basename(model_path)}")
    device = _resolve_device(device)
    model = CDNet(backbone='resnet50', output_stride=16, img_size=1024,
                  n_class=1, img_chan=3, chan_num=64, fuzzy_num=16)
    # 安全加载：weights_only=True 拒绝 pickle 任意对象，防止投毒权重 RCE。
    # 本项目权重为纯 OrderedDict[str, Tensor]（597 键，全部 Tensor），已实测兼容。
    state = torch.load(model_path, map_location=device, weights_only=True)
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    input_folder = r"H:\我的云端硬盘\guangdong1"
    output_folder = r"E:\GEE_data\output_view_guangdong1"
    model_path = r"E:\Code\GEE\best_train_loss_model_resnet50.pth"

    os.makedirs(output_folder, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.name == 'nt': print("Windows Optimization Active.")

    model = load_model(model_path, device)

    # ✅ 调试统计变量
    processed = set()
    session_count = 0

    print("\n🚀 监听模式 (V17.3 - 信息增强版)...")
    print(f"📂 监控目录: {input_folder}")

    while True:
        # 1. 扫描所有 TIF
        all_tifs = glob.glob(os.path.join(input_folder, "*.tif"))

        # 2. 筛选出真正需要处理的文件
        pending_tifs = []
        for tif in all_tifs:
            if "_mask" in tif or not tif.endswith(('.tif', '.tiff', '.TIF', '.TIFF')): continue

            save_name = os.path.basename(tif).replace(".tif", "_mask.tif")
            save_path = os.path.join(output_folder, save_name)

            if tif not in processed and not os.path.exists(save_path):
                pending_tifs.append(tif)
            else:
                processed.add(tif)

        # 3. 批量处理
        if pending_tifs:
            print(f"\n📦 发现 {len(pending_tifs)} 个新任务，开始处理...")

            # 这里将 idx 和 total 传进去
            for idx, tif in enumerate(pending_tifs):
                save_name = os.path.basename(tif).replace(".tif", "_mask.tif")
                save_path = os.path.join(output_folder, save_name)

                try:
                    process_geotiff(model, tif, save_path, device, current_idx=idx + 1, total_batch=len(pending_tifs))
                    session_count += 1
                except KeyboardInterrupt:
                    print("\n🛑 用户手动停止")
                    return
                except Exception as e:
                    print(f"❌ 错误: {safe_error_summary(e)}")

                processed.add(tif)

        else:
            # ✅ 4. 心跳保活信息
            # 使用 \r 覆盖当前行，实现动态刷新效果
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\r⏳ [{current_time}] 暂无新文件... (本次运行累计成功: {session_count} 张) | 监听中...", end="",
                  flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
