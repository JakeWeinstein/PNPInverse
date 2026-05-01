"""Plot clean (noise-free) target I-V curves from cached .npz files."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

# Load the most recent cache file
cache_dir = os.path.join(_ROOT, "StudyResults", "target_cache")
npz_files = sorted(
    [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".npz")],
    key=os.path.getmtime,
    reverse=True,
)

print(f"Found {len(npz_files)} cached target files:")
for f in npz_files:
    data = np.load(f)
    print(f"  {os.path.basename(f)}: {len(data['phi_applied'])} pts, "
          f"eta range [{data['phi_applied'].min():.1f}, {data['phi_applied'].max():.1f}]")

# Use the one from the v16 run (22 pts)
for f in npz_files:
    data = np.load(f)
    if len(data["phi_applied"]) == 22:
        print(f"\nUsing: {os.path.basename(f)}")
        break

eta = data["phi_applied"]
cd = data["current_density"]
pc = data["peroxide_current"]

# Sort by eta for plotting
sort_idx = np.argsort(eta)
eta_sorted = eta[sort_idx]
cd_sorted = cd[sort_idx]
pc_sorted = pc[sort_idx]

out_dir = os.path.join(_ROOT, "StudyResults", "clean_target_plots")
os.makedirs(out_dir, exist_ok=True)

# --- Plot 1: Total current density ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(eta_sorted, cd_sorted, "o-", color="steelblue", linewidth=2, markersize=6)
ax.set_xlabel(r"Dimensionless overpotential $\hat{\eta}$", fontsize=13)
ax.set_ylabel("Total current density (scaled)", fontsize=13)
ax.set_title("Clean Target: Total Current Density vs Overpotential", fontsize=14)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)
fig.tight_layout()
path1 = os.path.join(out_dir, "clean_total_current.png")
fig.savefig(path1, dpi=150)
print(f"Saved: {path1}")
plt.close(fig)

# --- Plot 2: Peroxide current ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(eta_sorted, pc_sorted, "s-", color="darkorange", linewidth=2, markersize=6)
ax.set_xlabel(r"Dimensionless overpotential $\hat{\eta}$", fontsize=13)
ax.set_ylabel("Peroxide current (scaled)", fontsize=13)
ax.set_title("Clean Target: Peroxide Current vs Overpotential", fontsize=14)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)
fig.tight_layout()
path2 = os.path.join(out_dir, "clean_peroxide_current.png")
fig.savefig(path2, dpi=150)
print(f"Saved: {path2}")
plt.close(fig)

# --- Plot 3: Both on same figure (dual y-axis) ---
fig, ax1 = plt.subplots(figsize=(10, 6))
color1 = "steelblue"
color2 = "darkorange"

ln1 = ax1.plot(eta_sorted, cd_sorted, "o-", color=color1, linewidth=2,
               markersize=6, label="Total current density")
ax1.set_xlabel(r"Dimensionless overpotential $\hat{\eta}$", fontsize=13)
ax1.set_ylabel("Total current density (scaled)", fontsize=13, color=color1)
ax1.tick_params(axis="y", labelcolor=color1, labelsize=11)
ax1.tick_params(axis="x", labelsize=11)

ax2 = ax1.twinx()
ln2 = ax2.plot(eta_sorted, pc_sorted, "s-", color=color2, linewidth=2,
               markersize=6, label="Peroxide current")
ax2.set_ylabel("Peroxide current (scaled)", fontsize=13, color=color2)
ax2.tick_params(axis="y", labelcolor=color2, labelsize=11)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize=11, loc="lower left")

ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax1.axvline(0, color="gray", linewidth=0.5, linestyle="--")
ax1.grid(True, alpha=0.3)
ax1.set_title("Clean Target I-V Curves (True Parameters, No Noise)", fontsize=14)
fig.tight_layout()
path3 = os.path.join(out_dir, "clean_both_observables.png")
fig.savefig(path3, dpi=150)
print(f"Saved: {path3}")
plt.close(fig)

print(f"\nAll plots saved to: {out_dir}")

# Print data summary
print(f"\nData summary:")
print(f"  eta range: [{eta_sorted[0]:.1f}, {eta_sorted[-1]:.1f}] ({len(eta)} points)")
print(f"  CD range:  [{cd_sorted.min():.6f}, {cd_sorted.max():.6f}]")
print(f"  PC range:  [{pc_sorted.min():.6f}, {pc_sorted.max():.6f}]")
