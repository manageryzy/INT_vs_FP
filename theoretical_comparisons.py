import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as _norm

# ----------------------------
# Standard normal PDF/CDF (array-safe)
# ----------------------------
def phi(z):
    z = np.asarray(z, dtype=np.float64)
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)

def Phi(z):
    z = np.asarray(z, dtype=np.float64)
    try:
        from scipy.special import ndtr  # preferred, stable
        return ndtr(z)
    except ImportError:
        import math
        erf_vec = np.vectorize(math.erf)  # fallback
        return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))

# ----------------------------
# Weights and probabilities in Z-domain
# ----------------------------
def p_sub(t0, t1):
    return 2.0 * (Phi(t1) - Phi(t0))

def w_zero(t0):
    return 2.0 * Phi(t0) - 1.0 - 2.0 * t0 * phi(t0)

def w_norm(t1):
    return 2.0 * (t1 * phi(t1) + (1.0 - Phi(t1)))

# ----------------------------
# QSNR models (baseline)
# ----------------------------
def qsnr_int(kappa, b, rho=1.5):
    kappa = np.asarray(kappa, dtype=np.float64)
    return 10.8 + 6.02 * (b - 1) - 20.0 * np.log10(rho) - 20.0 * np.log10(kappa)

def qsnr_fp(kappa, M, B, Qmax, rho=1.5):
    kappa = np.asarray(kappa, dtype=np.float64)

    # Coefficients from proof
    alpha_M = 1.0 / (24.0 * (2.0 ** (2 * M)))
    beta = (2.0 ** (2.0 * (1 - B - M))) / (12.0 * (Qmax ** 2))

    # Thresholds in Z-domain (T/σ)
    scale = (rho * kappa) / Qmax
    tau0 = scale * (2.0 ** (-B - M))  # T0/σ
    tau1 = scale * (2.0 ** (1 - B))   # TN/σ

    # Fractions
    psub  = p_sub(tau0, tau1)
    wzero = w_zero(tau0)
    wnorm = w_norm(tau1)

    # Relative MSE and QSNR
    err_power = alpha_M * wnorm + beta * (rho * kappa) ** 2 * psub + wzero
    return -10.0 * np.log10(err_power)

# ----------------------------
# QSNR models (NV corrections, k=16)
# ----------------------------
def qsnr_int_nv(kappa, b, k=16, rho=1.5):
    kappa = np.asarray(kappa, dtype=np.float64)
    # 原始相对 MSE（忽略 -1 与饱和边界影响）
    Qmax_approx = 2.0 ** (b - 1)  # 近似 (2^{b-1})
    R0 = (rho * kappa) ** 2 / (12.0 * (Qmax_approx ** 2))
    R_corr = R0 * (1.0 - 1.0 / k)
    return -10.0 * np.log10(R_corr)

def qsnr_fp_nv(kappa, M, B, Qmax, k=16, rho=1.5):
    kappa = np.asarray(kappa, dtype=np.float64)

    alpha_M = 1.0 / (24.0 * (2.0 ** (2 * M)))
    beta = (2.0 ** (2.0 * (1 - B - M))) / (12.0 * (Qmax ** 2))

    # Thresholds in Z-domain (T/σ)
    scale = (rho * kappa) / Qmax
    tau0 = scale * (2.0 ** (-B - M))  # T0/σ
    tau1 = scale * (2.0 ** (1 - B))   # TN/σ

    # Fractions
    psub  = p_sub(tau0, tau1)
    wzero = w_zero(tau0)
    wnorm = w_norm(tau1)

    # 正常区修正：wnorm_corr = max(0, wnorm - κ^2/k)
    wnorm_corr = np.maximum(0.0, wnorm - (kappa ** 2) / float(k))

    err_power = alpha_M * wnorm_corr + beta * (rho * kappa) ** 2 * psub + wzero
    return -10.0 * np.log10(err_power)

# ----------------------------
# QSNR models (MF: Mixed Format, outlier isolation)
# ----------------------------
def _mf_params(n=1, G=16):
    """Return (z_n, p_keep, sigma_kept_sq).

    z_n: outlier threshold in σ units (|x| > z_n·σ are outliers)
    p_keep: fraction of elements kept (quantized)
    sigma_kept_sq: variance of the truncated distribution (normalized to global σ²=1)
    """
    p_keep = 1.0 - n / G
    z_n = _norm.ppf(1.0 - n / (2.0 * G))
    # Variance of truncated |x| < z_n distribution, normalized to global σ²=1
    sigma_kept_sq = 1.0 - 2.0 * z_n * phi(z_n) / p_keep
    return z_n, p_keep, sigma_kept_sq

def qsnr_mf_int(kappa, b, n=1, G=16, rho=1.5):
    """MF-INT: outliers stored losslessly, rest quantized with INT-b.

    Quantizer range adapts: min(κ, z_n). When κ < z_n, behaves like MX
    (no real outliers). When κ > z_n, quantizer tightens to z_n.
    QSNR normalized to total signal power σ²=1.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    z_n, p_keep, sigma_kept_sq = _mf_params(n, G)
    kappa_eff = np.minimum(kappa, z_n)
    base = qsnr_int(kappa_eff, b, rho)
    correction = -10.0 * np.log10(p_keep * sigma_kept_sq)
    return base + correction

def qsnr_mf_fp(kappa, M, B, Qmax, n=1, G=16, rho=1.5):
    """MF-FP: outliers stored losslessly, rest quantized with FP."""
    kappa = np.asarray(kappa, dtype=np.float64)
    z_n, p_keep, sigma_kept_sq = _mf_params(n, G)
    kappa_eff = np.minimum(kappa, z_n)
    base = qsnr_fp(kappa_eff, M, B, Qmax, rho)
    correction = -10.0 * np.log10(p_keep * sigma_kept_sq)
    return base + correction

# ----------------------------
# QSNR models (SD: Sigma-Delta inspired)
# ----------------------------
def qsnr_sd(kappa, b, L=2, A_max=None, rho=1.5):
    """SD with overload penalty.

    Ideal SQNR = 10·log10( 3(2L+1)/π^(2L) · OSR^(2L+1) )

    Modulator has fixed dynamic range A_max (in σ units). Elements with
    |x| > A_max cause overload; their error ≈ κ² (clipping).
    If A_max is None, it is optimized for κ_target=3 (center of target range).

    QSNR_SD(κ) = -10·log10( (1-p_ol)·err_quant + p_ol·κ² )
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    err_quant = 1.0 / (3.0 * (2*L + 1) / (np.pi ** (2*L)) * (b ** (2*L + 1)))
    if A_max is None:
        A_max = _optimal_A_max(b, L)
    p_ol = 2.0 * (1.0 - Phi(A_max / kappa))
    err_total = (1.0 - p_ol) * err_quant + p_ol * kappa ** 2
    return -10.0 * np.log10(err_total)

def _optimal_A_max(b, L, kappa_lo=2.0, kappa_hi=4.0, n_pts=20):
    """Find A_max that maximizes mean SD QSNR over κ ∈ [kappa_lo, kappa_hi]."""
    from scipy.optimize import minimize_scalar
    kappa_range = np.linspace(kappa_lo, kappa_hi, n_pts)
    err_quant = 1.0 / (3.0 * (2*L + 1) / (np.pi ** (2*L)) * (b ** (2*L + 1)))
    def neg_mean_qsnr(A):
        p_ol = 2.0 * (1.0 - Phi(A / kappa_range))
        err = (1.0 - p_ol) * err_quant + p_ol * kappa_range ** 2
        return np.mean(10.0 * np.log10(err))
    res = minimize_scalar(neg_mean_qsnr, bounds=(0.1, 20.0), method='bounded')
    return res.x

def _optimal_L(b, L_range=range(1, 9)):
    """Find L that maximizes SD QSNR at κ=3 for given OSR=b.
    Stability constraint: OSR >= 2^L."""
    stable = [L for L in L_range if b >= 2 ** L]
    if not stable:
        return 1
    kappa_target = np.array([3.0])
    return max(stable, key=lambda L: float(qsnr_sd(kappa_target, b, L)[0]))

def qsnr_mfsd(kappa, b, L=2, n=1, G=16):
    """MFSD: MF outlier removal + SD quantization of remaining values.
    After removing outliers, SD sees fixed κ_eff = z_n/sqrt(σ_kept²).
    A_max is optimized for κ_eff (the actual peak of the truncated distribution).
    QSNR corrected back to global σ²=1.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    z_n, p_keep, sigma_kept_sq = _mf_params(n, G)
    kappa_eff = z_n / np.sqrt(sigma_kept_sq)
    # Optimize A_max for κ_eff (truncated distribution has fixed peak)
    A_max = _optimal_A_max(b, L, kappa_lo=kappa_eff, kappa_hi=kappa_eff, n_pts=1)
    sqnr_trunc = qsnr_sd(np.array([kappa_eff]), b, L, A_max=A_max)[0]
    correction = -10.0 * np.log10(p_keep * sigma_kept_sq)
    return np.full_like(kappa, sqnr_trunc + correction)

# ----------------------------
# Styling
# ----------------------------
bit_colors = {"8": "#4c72b0", "6": "#2ca02c", "4": "#d62728", "3": "#ff7f0e", "2": "#8c564b"}
line_styles = {"INT": "-", "FP": "--", "MF-INT": ":", "MF-FP": "-.", "SD": (0,(3,1,1,1)), "MFSD": (0,(5,1))}
markers     = {"INT": "o", "FP": "s",  "MF-INT": "^", "MF-FP": "D", "SD": "v", "MFSD": "P"}

E4M3_PURPLE = "#9467bd"

def get_curve_color(kind, bits, scale):
    if bits == 4 and scale == "E4M3":
        return E4M3_PURPLE
    return bit_colors[str(bits)]
def get_display_label(kind, bits, scale, params=None):
    if kind in ("MF-INT", "MF-FP"):
        suffix = "INT" if kind == "MF-INT" else "FP"
        return f"MF{suffix}{bits}"
    if kind == "SD":
        L = params["L"] if params else 2
        return f"SD{bits}(L={L})"
    if kind == "MFSD":
        L = params["L"] if params else 2
        return f"MFSD{bits}(L={L})"
    if scale == "UE8M0":
        return f"MXINT{bits}" if kind == "INT" else f"MXFP{bits}"
    elif scale == "E4M3":
        if kind == "INT" and bits == 4:
            return "NVINT4"
        elif kind == "FP" and bits == 4:
            return "NVFP4"
    return f"{kind}{bits} ({scale})"
# ----------------------------
# Formats and parameters
# ----------------------------
formats = [
    ("INT8",      "INT", 8, {"b": 8},                        1.5,  "UE8M0"),
    ("FP8 E4M3",  "FP",  8, {"M": 3, "B": 7, "Qmax": 448.0}, 1.5,  "UE8M0"),

    ("INT6",      "INT", 6, {"b": 6},                        1.5,  "UE8M0"),
    ("FP6 E2M3",  "FP",  6, {"M": 3, "B": 1, "Qmax": 7.5},   1.5,  "UE8M0"),

    ("INT4",      "INT", 4, {"b": 4},                        1.5,  "UE8M0"),
    ("FP4 E2M1",  "FP",  4, {"M": 1, "B": 1, "Qmax": 6.0},   1.5,  "UE8M0"),

    ("INT3",      "INT", 3, {"b": 3},                        1.5,  "UE8M0"),
    ("FP3 E1M1",  "FP",  3, {"M": 1, "B": 0, "Qmax": 3.0},   1.5,  "UE8M0"),

    ("INT2",      "INT", 2, {"b": 2},                        1.5,  "UE8M0"),
    ("FP2 E1M0",  "FP",  2, {"M": 0, "B": 0, "Qmax": 2.0},   1.5,  "UE8M0"),

    # NV 系列（E4M3 标签，rho 略小）
    ("INT4",      "INT", 4, {"b": 4},                        1.05, "E4M3"),
    ("FP4 E2M1",  "FP",  4, {"M": 1, "B": 1, "Qmax": 6.0},   1.05, "E4M3"),

    # MF 系列（G=16, n=1 outlier per group, lossless）
    ("MF-INT8",   "MF-INT", 8, {"b": 8,                         "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MF-FP8",    "MF-FP",  8, {"M": 3, "B": 7, "Qmax": 448.0, "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MF-INT6",   "MF-INT", 6, {"b": 6,                         "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MF-FP6",    "MF-FP",  6, {"M": 3, "B": 1, "Qmax": 7.5,   "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MF-INT4",   "MF-INT", 4, {"b": 4,                         "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MF-FP4",    "MF-FP",  4, {"M": 1, "B": 1, "Qmax": 6.0,   "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MF-INT3",   "MF-INT", 3, {"b": 3,                         "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MF-FP3",    "MF-FP",  3, {"M": 1, "B": 0, "Qmax": 3.0,   "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MF-INT2",   "MF-INT", 2, {"b": 2,                         "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MF-FP2",    "MF-FP",  2, {"M": 0, "B": 0, "Qmax": 2.0,   "n": 1, "G": 16}, 1.5, "UE8M0"),

    # SD 系列（auto-tuned L per bit-width）
    ("SD8",  "SD",   8, {"b": 8, "L": _optimal_L(8)},                    1.5, "UE8M0"),
    ("SD6",  "SD",   6, {"b": 6, "L": _optimal_L(6)},                    1.5, "UE8M0"),
    ("SD4",  "SD",   4, {"b": 4, "L": _optimal_L(4)},                    1.5, "UE8M0"),
    ("SD3",  "SD",   3, {"b": 3, "L": _optimal_L(3)},                    1.5, "UE8M0"),
    ("SD2",  "SD",   2, {"b": 2, "L": _optimal_L(2)},                    1.5, "UE8M0"),

    # MFSD 系列（MF + SD, auto-tuned L）
    ("MFSD8", "MFSD", 8, {"b": 8, "L": _optimal_L(8), "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MFSD6", "MFSD", 6, {"b": 6, "L": _optimal_L(6), "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MFSD4", "MFSD", 4, {"b": 4, "L": _optimal_L(4), "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MFSD3", "MFSD", 3, {"b": 3, "L": _optimal_L(3), "n": 1, "G": 16}, 1.5, "UE8M0"),
    ("MFSD2", "MFSD", 2, {"b": 2, "L": _optimal_L(2), "n": 1, "G": 16}, 1.5, "UE8M0"),
]

# ----------------------------
# Sweep κ and compute
# ----------------------------
kappa = np.linspace(1.0, 12, 600)

curves = {}  # label -> y
pairs_by_group = {}  # (bits(str), scale) -> {"INT": (label, y), "FP": (label, y)}

for name, kind, bits, params, rho, scale in formats:
    if kind == "MF-INT":
        y = qsnr_mf_int(kappa, b=params["b"], n=params["n"], G=params["G"], rho=rho)
    elif kind == "MF-FP":
        y = qsnr_mf_fp(kappa, M=params["M"], B=params["B"], Qmax=params["Qmax"], n=params["n"], G=params["G"], rho=rho)
    elif kind == "SD":
        y = qsnr_sd(kappa, b=params["b"], L=params["L"])
    elif kind == "MFSD":
        y = qsnr_mfsd(kappa, b=params["b"], L=params["L"], n=params["n"], G=params["G"])
    elif kind == "INT":
        if scale == "E4M3":
            y = qsnr_int_nv(kappa, b=params["b"], k=16, rho=rho)
        else:
            y = qsnr_int(kappa, b=params["b"], rho=rho)
    else:
        if scale == "E4M3":
            y = qsnr_fp_nv(kappa, M=params["M"], B=params["B"], Qmax=params["Qmax"], k=16, rho=rho)
        else:
            y = qsnr_fp(kappa, M=params["M"], B=params["B"], Qmax=params["Qmax"], rho=rho)

    disp_label = get_display_label(kind, bits, scale, params)
    curves[disp_label] = y

    gkey = (str(bits), scale)
    pairs_by_group.setdefault(gkey, {})
    pairs_by_group[gkey][kind] = (disp_label, y)

# ----------------------------
# Plot — one subplot per bit-width
# ----------------------------
def find_intersections(x, y_a, y_b):
    d = y_a - y_b
    idx = np.where(d[:-1] * d[1:] <= 0)[0]
    points = []
    for i in idx:
        x0, x1 = x[i], x[i+1]
        d0, d1 = d[i], d[i+1]
        if d1 == d0:
            continue
        xc = x0 - d0 * (x1 - x0) / (d1 - d0)
        yc = np.interp(xc, [x0, x1], [y_a[i], y_a[i+1]])
        points.append((xc, yc))
    return points

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

bit_groups = [8, 6, 4, 3, 2]
fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=140, sharey=False)

for ax, bits in zip(axes, bit_groups):
    b = str(bits)
    ax.set_title(f"{bits}-bit", fontsize=13)
    ax.set_xlabel("κ (crest factor)", fontsize=12)
    if bits == 8:
        ax.set_ylabel("QSNR (dB)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    plot_entries = [e for e in formats if e[2] == bits]
    local_ys = []

    for name, kind, ebits, params, rho, scale in plot_entries:
        disp_label = get_display_label(kind, ebits, scale, params)
        y = curves[disp_label]
        color = get_curve_color(kind, ebits, scale)
        ls = line_styles[kind]
        mk = markers[kind]
        ax.plot(kappa, y, label=disp_label, color=color, linestyle=ls,
                marker=mk, markevery=40, markersize=5, linewidth=2, alpha=0.95)
        local_ys.append(y)

    # INT vs FP intersection annotation
    grp_ue = pairs_by_group.get((b, 'UE8M0'), {})
    if "INT" in grp_ue and "FP" in grp_ue:
        name_int, y_int = grp_ue["INT"]
        name_fp,  y_fp  = grp_ue["FP"]
        pts = find_intersections(kappa, y_int, y_fp)
        color_ann = get_curve_color("FP", bits, "UE8M0")
        for xc, yc in pts:
            ax.scatter(xc, yc, color=color_ann, edgecolors='black', s=60, marker='X', zorder=6)
            ax.annotate(f'κ={xc:.1f}\n{yc:.1f}dB', xy=(xc, yc),
                        xytext=(8, 8), textcoords='offset points', fontsize=9,
                        color=color_ann,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color_ann, alpha=0.85),
                        arrowprops=dict(arrowstyle='->', color=color_ann, lw=1.0))

    ymin = min(np.min(y) for y in local_ys)
    ymax = max(np.max(y) for y in local_ys)
    ax.set_ylim(ymin - 2, ymax + 2)
    ax.legend(loc='upper right', frameon=True)

fig.suptitle("QSNR vs κ: MX / NV / MF formats", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('./qsnr_vs_kappa.png', bbox_inches='tight', dpi=300)
print("QSNR vs κ plot saved as qsnr_vs_kappa.png")
plt.show()
