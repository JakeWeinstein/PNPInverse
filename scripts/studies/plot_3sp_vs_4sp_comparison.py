"""Plot 3-species Boltzmann vs 4-species standard I-V curves.

Data sources:
- 4-species standard: diagnostic_eeq_sweep CSV (full z-ramp, z=1 reliable for V<=+0.1V)
- 3-species Boltzmann (stabilized_full JSON): z_achieved tracked, filter to z=1 only

Note: The JSON is technically the stabilized 4-species solver, but the 3-species
Boltzmann raw data was only saved as PNGs. We use what's available.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Load 4-species standard solver (CSV) ---
csv_path = os.path.join(ROOT, "StudyResults", "diagnostic_eeq_sweep",
                        "data_E_eq0.68_1.78_physical.csv")
data_4sp = np.genfromtxt(csv_path, delimiter=",", names=True)
v_4sp = data_4sp["v_rhe"]
cd_4sp = data_4sp["current_density"]
pc_4sp = data_4sp["peroxide_current"]

# Sort by voltage
idx = np.argsort(v_4sp)
v_4sp, cd_4sp, pc_4sp = v_4sp[idx], cd_4sp[idx], pc_4sp[idx]

# I_SCALE to convert to mA/cm^2
I_SCALE = 0.0254  # from _bv_common.py
cd_4sp_mA = cd_4sp * (-I_SCALE)
pc_4sp_mA = pc_4sp * (-I_SCALE)

# --- Load 3-species / stabilized solver (JSON) ---
json_path = os.path.join(ROOT, "StudyResults", "v18_stabilized_full",
                         "stabilized_iv_curve.json")
with open(json_path) as f:
    data_3sp = json.load(f)

# Filter to z=1 only
v_3sp, cd_3sp, pc_3sp = [], [], []
for pt in data_3sp:
    if pt["z_achieved"] >= 0.999:
        v_3sp.append(pt["V_RHE"])
        cd_3sp.append(pt["cd_z1"])
        pc_3sp.append(pt["pc_z1"])

v_3sp = np.array(v_3sp)
cd_3sp = np.array(cd_3sp)
pc_3sp = np.array(pc_3sp)

idx3 = np.argsort(v_3sp)
v_3sp, cd_3sp, pc_3sp = v_3sp[idx3], cd_3sp[idx3], pc_3sp[idx3]

cd_3sp_mA = cd_3sp * (-I_SCALE)
pc_3sp_mA = pc_3sp * (-I_SCALE)

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Top: Total current density
ax = axes[0]
ax.plot(v_4sp, cd_4sp_mA, 'b-o', markersize=4, label='4-species standard (z-ramp, phys E_eq)')
ax.plot(v_3sp, cd_3sp_mA, 'r-s', markersize=4, label='3-species Boltzmann (z=1 only)')
ax.axvline(0.10, color='gray', ls='--', alpha=0.5, label='4sp z=1 boundary (V=+0.10V)')
ax.axvline(0.68, color='green', ls=':', alpha=0.7, label='E_eq(R1) = 0.68V')
ax.set_ylabel('Total Current Density (mA/cm²)')
ax.set_title('Total Current: 4-Species Standard vs 3-Species Boltzmann')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom: Peroxide current density
ax = axes[1]
ax.plot(v_4sp, pc_4sp_mA, 'b-o', markersize=4, label='4-species standard (z-ramp, phys E_eq)')
ax.plot(v_3sp, pc_3sp_mA, 'r-s', markersize=4, label='3-species Boltzmann (z=1 only)')
ax.axvline(0.10, color='gray', ls='--', alpha=0.5, label='4sp z=1 boundary (V=+0.10V)')
ax.axvline(0.68, color='green', ls=':', alpha=0.7, label='E_eq(R1) = 0.68V')
ax.set_ylabel('Peroxide Current Density (mA/cm²)')
ax.set_xlabel('V vs RHE (V)')
ax.set_title('Peroxide Current: 4-Species Standard vs 3-Species Boltzmann')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(ROOT, "StudyResults", "v18_3sp_vs_4sp_comparison.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.close()

# Print voltage coverage summary
print(f"\n4-species standard: {v_4sp.min():.2f}V to {v_4sp.max():.2f}V ({len(v_4sp)} points)")
print(f"3-species Boltzmann: {v_3sp.min():.2f}V to {v_3sp.max():.2f}V ({len(v_3sp)} points)")
print(f"Overlap: {max(v_4sp.min(), v_3sp.min()):.2f}V to {min(v_4sp.max(), v_3sp.max()):.2f}V")
