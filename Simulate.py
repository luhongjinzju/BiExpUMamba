#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Microtubule TIRF end-to-end simulation (merged)

Overview:
1) Structure & geometry: generate a 3D microtubule voxel volume (surface-hugging, tapering, min-gap / crossing control)
2) Optical forward model: TIRF evanescent decay + 3D PSF convolution + z-integration -> noise-free ideal TIRF image
3) Temporal & sensor noise: separate double-exponential bleaching for foreground/background + low-frequency background drift
   + camera noise pipeline -> T-frame 16-bit stack

Usage:
- mode=generate: start from random structure and generate ideal + multiple GT masks + noisy stack
    python 微管TIRF仿真_合并版.py --mode generate --out_dir "/path/to/out" --T 100 --seed 2025
- mode=from_image: start from an existing 2D image (mask/ideal) and generate only the temporal noisy stack
    python 微管TIRF仿真_合并版.py --mode from_image --image_path "/path/to/img.png" --out_dir "/path/to/out"
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import binary_closing, disk, remove_small_objects
from tifffile import imwrite


# ============================ Basic size / output ============================
H, W = 256, 256  # Output image size (pixels): H=height (rows), W=width (cols)
T = 100  # Number of frames (time dimension length)

OUT_DIR = "/Users/tirfm_merged"  # Default output directory (overridable via --out_dir)

# ============================ Optics / sampling ============================
lambda_exc_nm = 488.0  # Excitation wavelength (nm), used to compute TIRF penetration depth dp
lambda_em_nm = 520.0  # Emission wavelength (nm), used to estimate PSF scale (sigma)
n1, n2 = 1.518, 1.333  # Refractive indices: n1=glass/immersion side, n2=sample side (sets TIRF condition and dp)
NA = 1.45  # Objective numerical aperture (affects PSF scale)
px_size_nm = 108.0  # Effective sample-plane pixel size (nm/px), converts PSF nm scale to pixels
dz_nm = 50.0  # z step (nm/layer), defines 3D voxel spacing and axial PSF sigma (in layers)
Z_total_nm = 400.0  # Total z thickness (nm); TIRF mainly sees the first few hundred nm near the interface
target_theta_deg = 76.0  # Incidence angle (deg); larger angle -> smaller dp (shallower TIRF)

# ============================ Structure generation ============================
N_PATHS = 6  # Number of microtubules (smaller -> sparser)
MIN_GAP_PX = 8  # Minimum separation (pixels), prevents tubes from sticking together
MAX_CROSS_PER_PATH = 0  # Max crossings per tube (0 tries to avoid crossings)
CROSS_WINDOW_PX = 4  # Lateral window (pixels) within which crossings are allowed

SIGMA_MIN = 0.2  # Minimum tube width parameter at ends (pixels; used for rasterization)
SIGMA_MID_R = (0.8, 1.2)  # Random range of mid-section width parameter (pixels)
TAPER_PWR = 1.7  # Taper strength (larger -> sharper ends)
WAVINESS = 0.13  # Small width waviness amplitude (0~0.3 typical)

# ============================ GT / evaluation ============================
PSF_EVAL_SCALE = 1.8  # Effective PSF scale factor for evaluation (accounts for aberration/defocus broadening)
PSF_EVAL_THRESH = 0.4  # Threshold (0~1) used to binarize the PSF-matched GT
ALPHA_ISO = 0.60  # Iso-intensity ratio used by gt_mask_iso and gt_mask_tirf_alpha (0.55~0.65 typical)

# ============================ Camera noise (electron domain) ============================
fg_peak_e = 4500.0  # Foreground peak electrons: ideal=1 maps to ~this many e- at frame 0 (global brightness knob)
read_noise_e = 2.5  # Read noise std (electrons)
e_per_adu_mean = 0.9  # Mean conversion gain (electrons/ADU): ADU = electrons / gain
bias_adu = 1000.0  # Bias / black level (ADU)
bias_jitter_adu = 0.5  # Per-frame bias jitter (ADU)
gain_fpn_sigma = 0.005  # Pixel gain non-uniformity (multiplicative FPN; relative std)
col_bias_adu = 2.0  # Column fixed-pattern bias amplitude (ADU)
row_bias_adu = 1.5  # Row fixed-pattern bias amplitude (ADU)
hot_px_ratio = 0.0002  # Hot pixel probability per pixel
hot_px_e = 60.0  # Extra electrons per hot pixel per frame

# ============================ Background / drift (electron domain) ============================
bg_const_e = 35.0  # Constant background (e-/px/frame): autofluorescence/scatter/dark current baseline
bg_low_sigma_px = 22  # Low-frequency background smoothing scale (pixels): larger -> slower spatial variation
bg_low_amp_e = 12.0  # Low-frequency background amplitude (electrons)
bg_drift_ppm = 280  # Temporal drift magnitude for the low-frequency term (ppm, total span)

global_intensity_jitter = 0.02  # Per-frame global intensity jitter (relative std): laser power/focus fluctuation

fg_bleach_params = dict(  # Foreground double-exponential bleaching: a1/a2 weights, tau1/tau2 time constants (frames)
    a1=0.72,
    tau1=85.0,
    a2=0.28,
    tau2=550.0,
    jitter=0.02,  # Per-frame curve jitter (relative std)
)
bg_bleach_params = dict(  # Background double-exponential bleaching: often slower/weaker than foreground
    a1=0.40,
    tau1=180.0,
    a2=0.60,
    tau2=1200.0,
    jitter=0.01,
)

col_drift_rw_sigma_adu = 0.0  # Column bias random-walk drift (ADU/step); >0 enables time-varying drift
row_drift_rw_sigma_adu = 0.0  # Row bias random-walk drift (ADU/step); >0 enables time-varying drift

MAX_U16 = 65535  # Max uint16 value (for writing 16-bit TIFF)


def sigma_xy_px(lambda_em_nm=520.0, NA=1.45, px_nm=108.0):
    """Estimate the lateral PSF Gaussian sigma (in pixels)."""
    sig_nm = 0.21 * lambda_em_nm / max(NA, 1e-6)
    return max(sig_nm / px_nm, 0.5)


def sigma_z_layers(lambda_em_nm=520.0, NA=1.45, dz_nm=50.0):
    """Estimate the axial PSF Gaussian sigma (in z-layers)."""
    sig_nm = 0.66 * lambda_em_nm / max(NA**2, 1e-6)
    return max(sig_nm / dz_nm, 0.8)


def tirf_dp_nm(lambda_nm, n1, n2, theta_deg):
    """Compute TIRF evanescent penetration depth dp (nm)."""
    th = np.deg2rad(theta_deg)
    inner = n1**2 * (np.sin(th) ** 2) - n2**2
    inner = max(inner, 1e-12)
    return lambda_nm / (4 * np.pi * np.sqrt(inner))


def _stamp_disc_bool(canvas_bool, x, y, r, h, w):
    """Stamp a filled disk onto a 2D boolean canvas (union write)."""
    if r < 0.4:
        r = 0.4
    y0 = max(0, int(math.floor(y - r)))
    y1 = min(h - 1, int(math.ceil(y + r)))
    x0 = max(0, int(math.floor(x - r)))
    x1 = min(w - 1, int(math.ceil(x + r)))
    if y0 > y1 or x0 > x1:
        return
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    disk_mask = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    sub = canvas_bool[y0 : y1 + 1, x0 : x1 + 1]
    sub |= disk_mask
    canvas_bool[y0 : y1 + 1, x0 : x1 + 1] = sub


def worm_paths_with_info(
    h,
    w,
    n_paths,
    seed_rng,
    geom_rng,
    margin=16,
    persistence_px=120.0,
    step_px=2.0,
    Z_layers=8,
):
    """
    Generate 2D worm-like-chain centerlines and per-point metadata:
    - sigma_t: tube lateral width parameter varying along arc length (pixels)
    - z_t: surface-hugging height in z (in layers)
    """
    paths, infos = [], []
    for _ in range(n_paths):
        side = geom_rng.integers(0, 4)
        if side == 0:
            x0, y0 = margin, geom_rng.uniform(margin, h - margin)
            theta0 = geom_rng.uniform(-0.3, 0.3)
        elif side == 1:
            x0, y0 = w - margin, geom_rng.uniform(margin, h - margin)
            theta0 = geom_rng.uniform(np.pi - 0.3, np.pi + 0.3)
        elif side == 2:
            x0, y0 = geom_rng.uniform(margin, w - margin), margin
            theta0 = geom_rng.uniform(np.pi / 2 - 0.3, np.pi / 2 + 0.3)
        else:
            x0, y0 = geom_rng.uniform(margin, w - margin), h - margin
            theta0 = geom_rng.uniform(-np.pi / 2 - 0.3, -np.pi / 2 + 0.3)

        pts = [(x0, y0)]
        theta = theta0
        Lmax = geom_rng.uniform(0.9, 1.2) * max(h, w)
        Lacc = 0.0
        while Lacc < Lmax:
            dtheta = geom_rng.normal(0, math.sqrt(step_px / persistence_px))
            theta += dtheta
            x1 = pts[-1][0] + step_px * np.cos(theta)
            y1 = pts[-1][1] + step_px * np.sin(theta)
            if x1 < margin or x1 > w - margin or y1 < margin or y1 > h - margin:
                break
            pts.append((x1, y1))
            Lacc += step_px

        p = np.array(pts, dtype=np.float32)
        if len(p) < 8:
            continue
        for _ in range(2):
            p[1:-1] = 0.7 * p[1:-1] + 0.15 * (p[:-2] + p[2:])

        seg = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
        s = np.concatenate([[0], np.cumsum(seg)])
        L = s[-1]
        if L < 1e-6:
            continue
        t = s / L

        sigma_mid = seed_rng.uniform(*SIGMA_MID_R)
        sigma_t = SIGMA_MIN + (sigma_mid - SIGMA_MIN) * (np.sin(np.pi * t) ** TAPER_PWR)
        sigma_t *= 1.0 + WAVINESS * np.sin(2 * np.pi * (3 * t + seed_rng.uniform(0, 1)))

        z0 = seed_rng.uniform(0.05, 0.15) * (Z_layers - 1)
        amp = seed_rng.uniform(0.05, 0.30) * (Z_layers - 1)
        zt = z0 + amp * np.sin(2 * np.pi * (t + seed_rng.uniform(0, 1)))

        paths.append(p)
        infos.append({"pts": p, "sigma_t": sigma_t.astype(np.float32), "z_t": zt.astype(np.float32)})

    return paths, infos


def add_gaussian3d(vol, z, y, x, sz, sxy):
    """Add an anisotropic 3D Gaussian blob into the voxel volume (rasterize centerline into a tube)."""
    sz = max(sz, 0.25)
    sxy = max(sxy, 0.25)
    rz, rxy = int(3 * sz + 1), int(3 * sxy + 1)
    z0, z1 = max(0, int(z) - rz), min(vol.shape[0], int(z) + rz + 1)
    y0, y1 = max(0, int(y) - rxy), min(vol.shape[1], int(y) + rxy + 1)
    x0, x1 = max(0, int(x) - rxy), min(vol.shape[2], int(x) + rxy + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    g = np.exp(-(((zz - z) ** 2) / (2 * sz * sz) + ((yy - y) ** 2 + (xx - x) ** 2) / (2 * sxy * sxy)))
    vol[z0:z1, y0:y1, x0:x1] += g


def generate_volume(h, w, Z_layers, seed):
    """
    Generate a 3D microtubule voxel volume (normalized to 0..1), and return the path infos for GT generation.
    This includes min-gap and limited-crossing control.
    """
    seed_rng = np.random.default_rng(seed)
    geom_rng = np.random.default_rng(seed + 1)
    vol = np.zeros((Z_layers, h, w), dtype=np.float32)

    _, infos = worm_paths_with_info(
        h=h,
        w=w,
        n_paths=N_PATHS,
        seed_rng=seed_rng,
        geom_rng=geom_rng,
        Z_layers=Z_layers,
    )

    forbid = np.zeros((h, w), dtype=bool)
    cross_x_all = seed_rng.uniform(32, w - 32, size=len(infos))

    for i, info in enumerate(infos):
        p = info["pts"]
        sigma_t = info["sigma_t"]
        zt = info["z_t"]

        dist = distance_transform_edt(~forbid)
        cross_x = float(cross_x_all[i])
        crosses = 0
        step_px = 0.9

        for j in range(len(p) - 1):
            p0, p1 = p[j], p[j + 1]
            L = float(np.linalg.norm(p1 - p0))
            nstep = max(1, int(np.ceil(L / step_px)))
            for k in range(nstep):
                a = k / nstep
                xy = (1 - a) * p0 + a * p1
                x, y = float(xy[0]), float(xy[1])
                ix, iy = int(round(x)), int(round(y))
                if not (0 <= ix < w and 0 <= iy < h):
                    continue

                allow_cross = (abs(ix - cross_x) <= CROSS_WINDOW_PX) and (crosses < MAX_CROSS_PER_PATH)
                if not allow_cross and dist[iy, ix] < MIN_GAP_PX:
                    continue
                if allow_cross:
                    crosses += 1

                # Local lateral/axial widths; z_here is in "layers"
                sxy = float((1 - a) * sigma_t[j] + a * sigma_t[j + 1])
                sz = max(0.6 * sxy, 0.45)
                z_here = float((1 - a) * zt[j] + a * zt[j + 1])

                add_gaussian3d(vol, z_here, y, x, sz, sxy)
                forbid[max(0, iy - 1) : min(h, iy + 2), max(0, ix - 1) : min(w, ix + 2)] = True

    if vol.max() > 0:
        vol /= vol.max()
    return vol, infos


def tirf_project(vol, theta_deg, dz_nm, sig_xy, sig_z):
    """
    TIRF forward imaging:
    - Physical chain: vol * exp(-z/dp) -> 3D Gaussian PSF -> z integration -> 2D ideal image (normalized to 0..1)
    - Also returns struct_support: whether any structure contributed to that pixel through the physical chain
    """
    dp_nm = tirf_dp_nm(lambda_exc_nm, n1, n2, theta_deg)
    z_nm = np.arange(vol.shape[0], dtype=np.float32) * dz_nm
    atten = np.exp(-z_nm[:, None, None] / max(dp_nm, 1e-6)).astype(np.float32)

    E = vol * atten
    if sig_xy > 0 or sig_z > 0:
        E = gaussian_filter(E, sigma=(sig_z, sig_xy, sig_xy))
    img = E.sum(axis=0)
    img_norm = img / img.max() if img.max() > 0 else img

    vol_bin = (vol > 0).astype(np.float32)
    E_bin = vol_bin * atten
    if sig_xy > 0 or sig_z > 0:
        E_bin = gaussian_filter(E_bin, sigma=(sig_z, sig_xy, sig_xy))
    img_bin = E_bin.sum(axis=0)
    thr = float(img_bin.max()) * 1e-3 if img_bin.max() > 0 else 0.0
    mask_bw = (img_bin >= thr).astype(np.uint8)

    return img_norm.astype(np.float32), float(dp_nm), mask_bw


def rasterize_gold_masks(
    paths_info,
    h,
    w,
    dp_nm,
    sig_psf_xy,
    dz_nm,
    rng,
    core_scale=1.0,
    alpha_iso=ALPHA_ISO,
    min_gap_px=MIN_GAP_PX,
    max_cross=MAX_CROSS_PER_PATH,
    cross_win_px=CROSS_WINDOW_PX,
):
    """
    Generate two commonly used GT masks:
    - mask_core: geometric GT (depends only on centerline and local width; no PSF)
    - mask_iso: iso-intensity GT (accounts for PSF + TIRF decay; alpha_iso controls the iso-intensity ratio)
    """
    K_ISO = float(np.sqrt(-2.0 * np.log(max(1e-6, alpha_iso))))
    mask_core = np.zeros((h, w), dtype=bool)
    mask_iso = np.zeros((h, w), dtype=bool)
    forbid = np.zeros((h, w), dtype=bool)
    cross_x_all = rng.uniform(32, w - 32, size=len(paths_info))

    for i, info in enumerate(paths_info):
        p = info["pts"]
        sigma_t = info["sigma_t"]
        zt = info["z_t"]
        dist = distance_transform_edt(~forbid)
        cross_x = cross_x_all[i]
        crosses = 0
        step_px = 0.9

        for j in range(len(p) - 1):
            p0, p1 = p[j], p[j + 1]
            L = float(np.linalg.norm(p1 - p0))
            nstep = max(1, int(np.ceil(L / step_px)))
            for k in range(nstep + 1):
                a = k / nstep
                x = float((1 - a) * p0[0] + a * p1[0])
                y = float((1 - a) * p0[1] + a * p1[1])
                ix, iy = int(round(x)), int(round(y))
                if not (0 <= ix < w and 0 <= iy < h):
                    continue
                allow_cross = (abs(ix - cross_x) <= cross_win_px) and (crosses < max_cross)
                if not allow_cross and dist[iy, ix] < min_gap_px:
                    continue
                if allow_cross:
                    crosses += 1

                sxy = float((1 - a) * sigma_t[j] + a * sigma_t[j + 1])
                z_here_layers = float((1 - a) * zt[j] + a * zt[j + 1])
                z_nm_here = z_here_layers * dz_nm
                A = math.exp(-z_nm_here / max(dp_nm, 1e-9))

                r_core = max(0.8, core_scale * sxy)
                _stamp_disc_bool(mask_core, x, y, r_core, h, w)

                if A > alpha_iso:
                    sigma_eff = math.sqrt(sxy * sxy + sig_psf_xy * sig_psf_xy)
                    r_iso = max(0.8, K_ISO * sigma_eff)
                    _stamp_disc_bool(mask_iso, x, y, r_iso, h, w)

                forbid[max(0, iy - 1) : min(h, iy + 2), max(0, ix - 1) : min(w, ix + 2)] = True

    mask_core = binary_closing(mask_core, disk(1))
    mask_core = remove_small_objects(mask_core, min_size=40)
    mask_iso = binary_closing(mask_iso, disk(1))
    mask_iso = remove_small_objects(mask_iso, min_size=40)
    return mask_core.astype(np.uint8), mask_iso.astype(np.uint8)


def time_curve_doubleexp(T, a1, tau1, a2, tau2, jitter, rng):
    """Generate a double-exponential bleaching curve normalized to 1 at frame 0."""
    t = np.arange(T, dtype=np.float32)
    cur = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)
    cur = cur / (cur[0] + 1e-8)
    cur = cur * (1.0 + rng.normal(0.0, jitter, size=T).astype(np.float32))
    cur[0] = 1.0
    return np.clip(cur, 0, None).astype(np.float32)


def lowfreq_map(h, w, sigma_px, amp, rng):
    """Generate low-frequency background: white noise -> Gaussian smoothing -> normalize -> scale."""
    r = rng.standard_normal((h, w)).astype(np.float32)
    low = gaussian_filter(r, sigma_px)
    low = (low - low.min()) / (low.max() - low.min() + 1e-8)
    return amp * low


def simulate_sensor_stack(ideal_norm, out_dir, seed, T_frames):
    """
    Convert a noise-free ideal image into a T-frame camera observation stack (uint16):
    - Foreground: ideal_norm * fg_peak_e * fg_curve[t]
    - Background: bg_const_e * bg_curve[t] + low-frequency drift + hot pixels
    - Noise: Poisson (shot) + Gaussian read noise + FPN (gain) + row/col bias + bias level
    """
    rng = np.random.default_rng(seed)
    Hh, Ww = ideal_norm.shape

    fg_curve = time_curve_doubleexp(T_frames, rng=rng, **fg_bleach_params)
    bg_curve = time_curve_doubleexp(T_frames, rng=rng, **bg_bleach_params)

    # Fixed-pattern noise (FPN): per-pixel gain/bias non-uniformity
    gain_map = e_per_adu_mean * (1.0 + rng.normal(0.0, gain_fpn_sigma, size=(Hh, Ww)).astype(np.float32))
    col_pat = rng.normal(0.0, col_bias_adu, size=(1, Ww)).astype(np.float32)
    row_pat = rng.normal(0.0, row_bias_adu, size=(Hh, 1)).astype(np.float32)
    hot_mask = (rng.random((Hh, Ww)) < hot_px_ratio).astype(np.float32)
    lowbg_e = lowfreq_map(Hh, Ww, bg_low_sigma_px, bg_low_amp_e, rng=np.random.default_rng(seed + 123))

    # Slow temporal drift for the low-frequency term (ppm = parts per million)
    drift = 1.0 + np.linspace(0, bg_drift_ppm / 1e6 * T_frames, T_frames, dtype=np.float32)

    stack = np.empty((T_frames, Hh, Ww), dtype=np.uint16)
    global_jit = rng.normal(0.0, global_intensity_jitter, size=T_frames).astype(np.float32)

    col_rw = np.zeros((T_frames, 1, Ww), dtype=np.float32)
    row_rw = np.zeros((T_frames, Hh, 1), dtype=np.float32)
    if col_drift_rw_sigma_adu > 0:
        col_rw[:, 0, :] = np.cumsum(
            rng.normal(0.0, col_drift_rw_sigma_adu, size=(T_frames, Ww)).astype(np.float32), axis=0
        )
    if row_drift_rw_sigma_adu > 0:
        row_rw[:, :, 0] = np.cumsum(
            rng.normal(0.0, row_drift_rw_sigma_adu, size=(T_frames, Hh)).astype(np.float32), axis=0
        )

    for i in range(T_frames):
        # Build electron-domain "ideal electrons" (before noise)
        fg_e = fg_peak_e * ideal_norm * fg_curve[i]
        bg_e = (bg_const_e * bg_curve[i]) + lowbg_e * drift[i] + hot_mask * hot_px_e
        ideal_e = np.clip((fg_e + bg_e) * (1.0 + global_jit[i]), 0, None)

        # Electron-domain noise: Poisson shot noise + Gaussian read noise
        shot_e = rng.poisson(ideal_e).astype(np.float32)
        read_e = rng.normal(0.0, read_noise_e, size=(Hh, Ww)).astype(np.float32)
        electrons = shot_e + read_e

        # Electrons -> ADU (divide by gain) and add bias/row/col patterns
        adu = electrons / gain_map
        adu += bias_adu + (col_pat + col_rw[i]) + (row_pat + row_rw[i]) + rng.normal(0.0, bias_jitter_adu)

        # Quantize to 16-bit (uint16)
        frame_u16 = np.clip(adu, 0, MAX_U16).astype(np.uint16)
        stack[i] = frame_u16

    # Save 16-bit stack (T, H, W)
    imwrite(os.path.join(out_dir, "tirfm_100frames.tif"), stack)

    # Save 8-bit preview: percentile stretch (1%~99%)
    p1, p99 = np.percentile(stack, 1), np.percentile(stack, 99)
    prev = np.clip((stack.astype(np.float32) - p1) / max(p99 - p1, 1), 0, 1)
    imwrite(os.path.join(out_dir, "tirfm_100frames_preview8bit.tif"), (prev * 255).astype(np.uint8))

    return stack, fg_curve, bg_curve


def load_image_as_float01(path, hw):
    """Load an image, convert to float32 in [0,1], and resize to hw."""
    img = Image.open(path).convert("L").resize(hw[::-1], Image.NEAREST)
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def write_params(out_dir, params):
    """Write key simulation parameters to params.json for reproducibility."""
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def run_generate(out_dir, seed, T_frames):
    """End-to-end generation: 3D structure -> ideal TIRF -> multiple GT masks -> noisy temporal stack."""
    os.makedirs(out_dir, exist_ok=True)

    Z_layers = int(np.ceil(Z_total_nm / dz_nm))
    sig_xy = sigma_xy_px(lambda_em_nm, NA, px_size_nm)
    sig_z = sigma_z_layers(lambda_em_nm, NA, dz_nm)

    # 1) Structure (3D)
    vol, paths_info = generate_volume(H, W, Z_layers, seed=seed)
    imwrite(os.path.join(out_dir, "volume_uint16.tif"), (vol * 65535).astype(np.uint16))

    # 2) Optical forward model (TIRF)
    img_norm, dp_nm, mask_bw = tirf_project(vol, target_theta_deg, dz_nm, sig_xy, sig_z)
    Image.fromarray((img_norm * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"tirfm_ideal_theta_{target_theta_deg:.1f}_dp_{dp_nm:.0f}nm.png")
    )
    Image.fromarray((mask_bw * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"tirfm_struct_support_theta_{target_theta_deg:.1f}.png")
    )
    imwrite(
        os.path.join(out_dir, f"tirfm_struct_support_theta_{target_theta_deg:.1f}.tif"),
        (mask_bw.astype(np.uint16) * 65535),
    )

    # 3) GT masks (choose based on your evaluation needs)
    gt_volproj = (vol > 0).max(axis=0).astype(np.uint8)
    Image.fromarray(gt_volproj * 255).save(os.path.join(out_dir, f"gt_mask_volproj_theta_{target_theta_deg:.1f}.png"))
    imwrite(
        os.path.join(out_dir, f"gt_mask_volproj_theta_{target_theta_deg:.1f}.tif"),
        (gt_volproj.astype(np.uint16) * 65535),
    )

    alpha_eval = ALPHA_ISO
    mask_tirf = (img_norm >= alpha_eval).astype(np.uint8)
    Image.fromarray((mask_tirf * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"gt_mask_tirf_alpha{int(alpha_eval*100)}_theta_{target_theta_deg:.1f}.png")
    )
    imwrite(
        os.path.join(out_dir, f"gt_mask_tirf_alpha{int(alpha_eval*100)}_theta_{target_theta_deg:.1f}.tif"),
        (mask_tirf.astype(np.uint16) * 65535),
    )

    rng_masks = np.random.default_rng(seed + 7)
    mask_core, mask_iso = rasterize_gold_masks(
        paths_info,
        H,
        W,
        dp_nm,
        sig_xy,
        dz_nm,
        rng=rng_masks,
        core_scale=1.00,
        alpha_iso=ALPHA_ISO,
        min_gap_px=MIN_GAP_PX,
        max_cross=MAX_CROSS_PER_PATH,
        cross_win_px=CROSS_WINDOW_PX,
    )
    Image.fromarray((mask_core * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"gt_mask_core_theta_{target_theta_deg:.1f}.png")
    )
    imwrite(os.path.join(out_dir, f"gt_mask_core_theta_{target_theta_deg:.1f}.tif"), (mask_core.astype(np.uint16) * 65535))
    Image.fromarray((mask_iso * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"gt_mask_iso_alpha{int(ALPHA_ISO*100)}_theta_{target_theta_deg:.1f}.png")
    )
    imwrite(
        os.path.join(out_dir, f"gt_mask_iso_alpha{int(ALPHA_ISO*100)}_theta_{target_theta_deg:.1f}.tif"),
        (mask_iso.astype(np.uint16) * 65535),
    )

    A_core = int(mask_core.sum())
    flat = img_norm.astype(np.float32).ravel()
    if A_core <= 0:
        mask_tirf_areamatch = np.zeros_like(img_norm, dtype=np.uint8)
    else:
        order = np.argsort(flat)[::-1]
        k = min(A_core, flat.size)
        t = float(flat[order[k - 1]])
        mask_tirf_areamatch = (img_norm >= t).astype(np.uint8)

    Image.fromarray((mask_tirf_areamatch * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"gt_mask_tirf_areamatch_theta_{target_theta_deg:.1f}.png")
    )
    imwrite(
        os.path.join(out_dir, f"gt_mask_tirf_areamatch_theta_{target_theta_deg:.1f}.tif"),
        (mask_tirf_areamatch.astype(np.uint16) * 65535),
    )

    mask_core_f = mask_core.astype(np.float32)
    sigma_eval = PSF_EVAL_SCALE * sig_xy
    mask_blur = gaussian_filter(mask_core_f, sigma=sigma_eval)
    mask_psf_gt = (mask_blur >= PSF_EVAL_THRESH).astype(np.uint8)
    Image.fromarray((mask_psf_gt * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"gt_mask_psfmatched_theta_{target_theta_deg:.1f}.png")
    )
    imwrite(
        os.path.join(out_dir, f"gt_mask_psfmatched_theta_{target_theta_deg:.1f}.tif"),
        (mask_psf_gt.astype(np.uint16) * 65535),
    )

    # 4) Temporal + noise: generate T-frame noisy stack
    stack, fg_curve, bg_curve = simulate_sensor_stack(img_norm, out_dir, seed=seed + 1000, T_frames=T_frames)

    # 5) Parameter record (reproducibility)
    params = {
        "seed": seed,
        "H": H,
        "W": W,
        "T": int(T_frames),
        "Z_layers": int(Z_layers),
        "dz_nm": float(dz_nm),
        "lambda_exc_nm": float(lambda_exc_nm),
        "lambda_em_nm": float(lambda_em_nm),
        "NA": float(NA),
        "px_size_nm": float(px_size_nm),
        "n1": float(n1),
        "n2": float(n2),
        "theta_deg": float(target_theta_deg),
        "dp_nm": float(dp_nm),
        "psf_sigma_xy_px": float(sig_xy),
        "psf_sigma_z_layers": float(sig_z),
        "ALPHA_ISO": float(ALPHA_ISO),
        "fg_curve_first_last": [float(fg_curve[0]), float(fg_curve[-1])],
        "bg_curve_first_last": [float(bg_curve[0]), float(bg_curve[-1])],
    }
    write_params(out_dir, params)

    return stack


def run_from_image(image_path, out_dir, seed, T_frames):
    """
    Start from an existing 2D image (mask or ideal):
    - Apply a lateral PSF blur once to obtain an ideal image
    - Then run the bleaching + sensor noise pipeline to generate a temporal stack
    """
    os.makedirs(out_dir, exist_ok=True)

    img01 = load_image_as_float01(image_path, (H, W))
    sig_xy = sigma_xy_px(lambda_em_nm, NA, px_size_nm)
    ideal = gaussian_filter(img01, sig_xy)
    if ideal.max() > 0:
        ideal = ideal / ideal.max()

    imwrite(os.path.join(out_dir, "binary_input.tif"), (img01.astype(np.uint16) * MAX_U16))
    Image.fromarray((ideal * 255).astype(np.uint8)).save(os.path.join(out_dir, "ideal_from_image.png"))

    stack, fg_curve, bg_curve = simulate_sensor_stack(ideal, out_dir, seed=seed + 1000, T_frames=T_frames)

    params = {
        "seed": seed,
        "mode": "from_image",
        "input_path": str(image_path),
        "H": H,
        "W": W,
        "T": int(T_frames),
        "psf_sigma_xy_px": float(sig_xy),
        "fg_curve_first_last": [float(fg_curve[0]), float(fg_curve[-1])],
        "bg_curve_first_last": [float(bg_curve[0]), float(bg_curve[-1])],
    }
    write_params(out_dir, params)

    return stack


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["generate", "from_image"], default="generate")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--image_path", default="")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--T", type=int, default=T)
    return p.parse_args()


def main():
    """Main entry: choose the pipeline based on mode."""
    args = parse_args()
    if args.mode == "generate":
        run_generate(args.out_dir, seed=args.seed, T_frames=args.T)
    else:
        if not args.image_path:
            raise ValueError("mode=from_image requires --image_path")
        run_from_image(args.image_path, args.out_dir, seed=args.seed, T_frames=args.T)


if __name__ == "__main__":
    main()
