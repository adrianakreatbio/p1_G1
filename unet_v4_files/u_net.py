import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
from typing import Tuple, List, Optional

# ---------------------------
# Model
# ---------------------------

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(cin, cout, 1)
        )
        self.conv = ConvBlock(cout*2, cout)

    def forward(self, x, skip):
        x = self.up(x)
        dh, dw = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class GelUNet(nn.Module):
    """
    Input: (B,1,H,W) in [0,1]
    Output: (B,1,H,W) logits
    """
    def __init__(self, in_channels=1, out_channels=1, feats=(32,64,128,256)):
        super().__init__()
        f1,f2,f3,f4 = feats
        self.enc1 = ConvBlock(in_channels, f1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.enc2 = ConvBlock(f1, f2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.enc3 = ConvBlock(f2, f3)
        self.pool3 = nn.MaxPool2d(2,2)
        self.enc4 = ConvBlock(f3, f4)
        self.pool4 = nn.MaxPool2d(2,2)
        self.bottleneck = ConvBlock(f4, f4*2)
        self.up4 = UpBlock(f4*2, f4)
        self.up3 = UpBlock(f4,   f3)
        self.up2 = UpBlock(f3,   f2)
        self.up1 = UpBlock(f2,   f1)
        self.head = nn.Conv2d(f1, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)

# ---------------------------
# Inference helpers
# ---------------------------

@torch.no_grad()
def predict_mask(model: GelUNet, img_tensor: torch.Tensor, thresh: float = 0.25) -> torch.Tensor:
    """
    img_tensor: (1,1,H,W) float32 in [0,1]
    returns binary mask (1,1,H,W)
    """
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    model.eval()
    logits = model(img_tensor)
    prob = torch.sigmoid(logits)
    return (prob >= thresh).float()


def preprocess_gel(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)

    # large-kernel background (rolling-ball approx)
    bg = cv2.GaussianBlur(img, (0,0), 35)
    img = cv2.subtract(img, bg)                     # remove glow/gradient
    img = np.clip(img, 0, None)

    # contrast-limited adaptive hist eq. on 8-bit
    img8 = np.clip(img / (np.percentile(img, 99.9) + 1e-6), 0, 1)
    img8 = (img8*255).astype(np.uint8)
    img8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img8)

    # back to [0,1] float with mild gamma <1 to lift lows
    img = (img8.astype(np.float32) / 255.0) ** 0.9
    return torch.from_numpy(img)[None,None,...]


# ---------------------------
# Lane detection (mask-based)
# ---------------------------

def lanes_from_mask_cc(mask_bin: np.ndarray, n_lanes: Optional[int] = None) -> List[Tuple[int,int]]:
    """
    mask_bin: uint8 (H,W) 0/1, cleaned band mask for lane grouping
    returns list of (x0,x1) lane intervals left->right
    """
    H, W = mask_bin.shape

    # vertical dilation to merge bands within the same lane without merging neighbors
    tall = cv2.getStructuringElement(cv2.MORPH_RECT, (7, max(H//4, 40)))
    col  = cv2.dilate(mask_bin, tall, iterations=1)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(col)
    boxes: List[Tuple[int,int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        # reject tiny or short blobs
        if w < 0.03 * W or h < 0.15 * H:
            continue
        boxes.append((int(x), int(x+w)))

    if not boxes:
        return []

    # sort by horizontal center
    boxes.sort(key=lambda ab: (ab[0] + ab[1]) // 2)

    # if n_lanes specified, keep widest n
    if n_lanes and len(boxes) > n_lanes:
        widths = np.array([b-a for a,b in boxes])
        keep = np.argsort(widths)[::-1][:n_lanes]
        boxes = [boxes[i] for i in sorted(keep)]
    return boxes


# ---------------------------
# Quantification
# ---------------------------

def integrated_density(image: np.ndarray, mask: np.ndarray, lane: Tuple[int,int]) -> float:
    x0, x1 = lane
    roi_img  = image[:, x0:x1]
    roi_mask = mask[:,  x0:x1]
    return float((roi_img * roi_mask).sum())


def concentrations_relative_to_lane1(img: np.ndarray, model: GelUNet, n_lanes: Optional[int] = None, lane1_conc: float = 1.0,):
    """Returns per-lane relative concentrations and absolute using lane 1 as base."""

    # preprocess
    t = preprocess_gel(img)                      # (1,1,H,W)
    im = t.squeeze().cpu().numpy().astype(np.float32)

    # predict soft mask
    device = next(model.parameters()).device
    t = t.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(t)
        prob = torch.sigmoid(logits)             # (1,1,H,W)

    # --- start robust binarization pipeline ---
    # 1) soft map from model
    m_soft = prob.squeeze().cpu().numpy().astype(np.float32)

    # 2) suppress background glow in the soft map
    bg = cv2.GaussianBlur(m_soft, (0,0), 15)          # smooth background
    m_supp = np.clip(m_soft - 0.6*bg, 0, 1)           # remove low, flat responses

    # 3) anisotropic smoothing to connect bands and kill speckle
    m_supp = cv2.GaussianBlur(m_supp, (21,3), sigmaX=5, sigmaY=0.8)
    m_supp = cv2.medianBlur((m_supp*255).astype(np.uint8), 3).astype(np.float32)/255.0

    # 4) adaptive hysteresis on the cleaned map
    p90 = float(np.percentile(m_supp, 90))
    hi  = max(0.28, min(0.45, p90))                  # tighter high threshold
    lo  = max(0.12, hi*0.5)

    strong = (m_supp >= hi).astype(np.uint8)
    weak   = ((m_supp >= lo) & (m_supp < hi)).astype(np.uint8)
    grown  = cv2.dilate(strong, np.ones((3,3), np.uint8), 1)
    m_hyst = np.where((weak==1) & (grown==1), 1, strong).astype(np.uint8)

    # 5) fill holes and remove small blobs
    from scipy.ndimage import binary_fill_holes
    m_fill = binary_fill_holes(m_hyst).astype(np.uint8)
    m_fill = cv2.morphologyEx(m_fill, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    H, W = m_fill.shape
    min_area = int(0.0015 * H * W)                   # tune 0.0008–0.003
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m_fill, connectivity=8)
    m_vis = np.zeros_like(m_fill, dtype=np.uint8)
    for i in range(1, num):
        _, _, w, h, area = stats[i]
        if area >= min_area:
            m_vis[lab == i] = 1

    # --- end robust binarization pipeline ---

    # lanes from cleaned binary mask
    lanes = lanes_from_mask_cc((m_vis > 0).astype(np.uint8), n_lanes=n_lanes)
    if len(lanes) == 0:
        return {"lanes": lanes, "relative": [], "absolute": [], "mask_vis": m_vis, "bands_abs": []}

    # lane totals from soft mask
    lane_totals = [integrated_density(im, m_soft, L) for L in lanes]

    # per-band components inside each lane (use binary mask for geometry, soft mask for density)
    bands_abs: List[List[float]] = []
    abs_conc: List[float] = []

    # absolute concentration for lane i = (lane_total_i / lane_total_1) * lane1_conc
    if lane_totals[0] <= 1e-6:
        rel = [np.nan] * len(lane_totals)
    else:
        rel = [np.nan if t <= 1e-6 else t / lane_totals[0] for t in lane_totals]
    abs_conc = [r * lane1_conc if np.isfinite(r) else np.nan for r in rel]

    H = m_vis.shape[0]
    for (x0, x1), lane_total_soft, lane_abs in zip(lanes, lane_totals, abs_conc):
        lane_bin = (m_vis[:, x0:x1] > 0).astype(np.uint8)

        # connected components = candidate bands
        num, lab, stats, _ = cv2.connectedComponentsWithStats(lane_bin, connectivity=8)

        # filter tiny blobs
        min_band_area = max(80, int(0.0008 * H * (x1 - x0)))
        band_list = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < min_band_area:
                continue
            mask_i = (lab == i).astype(np.uint8)
            # density of this band using soft map
            roi_soft = m_soft[:, x0:x1]
            d = float((roi_soft * mask_i).sum())
            band_list.append((y, d))  # keep y for top→bottom ordering


        # --- to ignore the low intensity bands begins
        pix = int(mask_i.sum())
        mean_prob = d / (pix + 1e-6)

        # thresholds: area, mean prob, and fraction of lane total
        min_band_area = max(120, int(0.0012 * H * (x1 - x0)))
        min_mean     = 0.12           # drop very dim blobs
        min_frac     = 0.001          # <0.1% of lane intensity is ignored

        if area < min_band_area or mean_prob < min_mean or d < min_frac * lane_total_soft:
            continue
        # --- to ignore the low intensity bands ends


        # sort by vertical position
        band_list.sort(key=lambda t: t[0])
        dens_list = [d for _, d in band_list]

        # convert each band density to absolute concentration by proportional split of lane_abs
        if lane_total_soft > 1e-6 and np.isfinite(lane_abs):
            scale = lane_abs / lane_total_soft
            bands_abs.append([d * scale for d in dens_list])
        else:
            bands_abs.append([])

    # ----- Enforce to follow standard begins
    # Make lane 1 total exactly the entered standard
    if len(abs_conc) > 0 and np.isfinite(abs_conc[0]):
        abs_conc[0] = float(lane1_conc)

    # For every lane, rescale its band list so sum(bands) == lane total
    for i in range(len(bands_abs)):
        target = float(abs_conc[i]) if i < len(abs_conc) and np.isfinite(abs_conc[i]) else 0.0
        s = float(sum(bands_abs[i])) if bands_abs[i] else 0.0
        if target > 0.0 and s > 1e-9:
            scale = target / s
            bands_abs[i] = [b * scale for b in bands_abs[i]]
    # ----- Enforce to follow standard ends


    return {"lanes": lanes, "relative": rel, "absolute": abs_conc, "mask_vis": m_vis, "bands_abs": bands_abs}

# ---------------------------
# Example usage (CLI test)
# ---------------------------
if __name__ == "__main__":
    import imageio.v3 as iio
    model = GelUNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load("trained_weights.pth", map_location="cpu"))
    img8 = iio.imread("test/gel_image.png")
    res = concentrations_relative_to_lane1(img8, model, n_lanes=None, lane1_conc=50.0)
    print(res)
