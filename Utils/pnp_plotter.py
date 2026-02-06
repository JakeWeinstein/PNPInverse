import matplotlib.pyplot as plt
from firedrake import Function
from firedrake.pyplot import tripcolor, tricontour
import matplotlib.animation as animation
import numpy as np
import matplotlib.tri as tri
import imageio
import io
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# Save all render outputs under FireDrakeEnvCG/Inverse/Renders
RENDERS_DIR = Path(__file__).resolve().parent / "Renders"
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

def plot_solutions(U_prev, z_vals, mode, num_steps, dt, t,
                   save=True, show=True, output_png=None, return_fig=False,
                   c_lims=None, phi_lim=None):
    """
    Plot concentrations (any number of species) and potential for a given mixed state U_prev.

    If return_fig is True, the caller manages closing the figure.
    Limits can be fixed with tuples (vmin, vmax) per field.
    """
    n_species = len(z_vals)
    phi_idx = n_species

    if c_lims is None:
        c_lims = [None] * n_species
    if len(c_lims) != n_species:
        raise ValueError(f"c_lims must have length {n_species} to match z_vals")

    # Create figure with subplots for each field
    fig_width = 5 * (n_species + 1) / 2  # scale width with number of panels
    fig, axes = plt.subplots(1, n_species + 1, figsize=(fig_width, 4))
    axes = np.atleast_1d(axes)
    # Fixed spacing to prevent jitter between frames
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.12, wspace=0.28)

    cmap_cycle = ['viridis', 'plasma', 'magma', 'cividis', 'inferno']

    # Plot each species concentration
    for i in range(n_species):
        cmap = cmap_cycle[i % len(cmap_cycle)]
        tripcolor(U_prev.sub(i), axes=axes[i], cmap=cmap,
                  vmin=None if c_lims[i] is None else c_lims[i][0],
                  vmax=None if c_lims[i] is None else c_lims[i][1])
        axes[i].set_title(f'Concentration c{i} (z={z_vals[i]:+d}) at t={t:.3f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_aspect('equal')
        plt.colorbar(axes[i].collections[0], ax=axes[i], label=f'c{i}')

    # Plot electric potential
    tripcolor(U_prev.sub(phi_idx), axes=axes[-1], cmap='coolwarm',
              vmin=None if phi_lim is None else phi_lim[0],
              vmax=None if phi_lim is None else phi_lim[1])
    axes[-1].set_title(f'Electric Potential phi at t={t:.3f}')
    axes[-1].set_xlabel('x')
    axes[-1].set_ylabel('y')
    axes[-1].set_aspect('equal')
    plt.colorbar(axes[-1].collections[0], ax=axes[-1], label='phi')

    bv_status = "with BV" if mode == 1 else "with Robin" if mode == 2 else "Normal"
    fig.suptitle(f'PNP Solution ({bv_status}): {num_steps} time steps, dt={dt}', fontsize=14, y=1.02)
    if save:
        if output_png is None:
            output_png = f'pnp_solution_t{t:.3f}.png'
        # Force outputs into Renders directory while keeping the requested filename
        output_path = RENDERS_DIR / Path(output_png).name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    if show:
        plt.show()
    if return_fig:
        return fig, axes
    else:
        plt.close(fig)
        return None, None
    
def create_animations(snapshots, mode, mesh, W, z_vals, num_steps, dt,
                      fps=10, fmt="mp4", fix_limits=True):
    """
    Use plot_solutions to render each frame, then stitch into an animation.

    snapshots: dict with lists
        required: 't', 'phi'
        concentrations: either key 'c' (list of per-species lists) or per-species keys 'c0','c1',...
    W: mixed FunctionSpace matching those snapshots
    fix_limits: if True, use global vmin/vmax per field to keep color scales constant
    """
    n_species = len(z_vals)
    phi_idx = n_species

    times = list(snapshots.get("t", []))
    phi_list = list(snapshots.get("phi", []))

    if "c" in snapshots:
        c_lists = [list(arrs) for arrs in snapshots.get("c", [])]
    else:
        c_lists = [list(snapshots.get(f"c{i}", [])) for i in range(n_species)]

    if len(c_lists) != n_species:
        raise ValueError(f"Expected concentration lists for {n_species} species, got {len(c_lists)}")

    lengths = [len(times), len(phi_list)] + [len(lst) for lst in c_lists]
    usable_frames = min(lengths) if all(l > 0 for l in lengths) else 0
    if usable_frames == 0:
        print("  Warning: no usable frames (t/c/phi mismatch or empty). Skipping animations.")
        return

    times = times[:usable_frames]
    phi_list = phi_list[:usable_frames]
    c_lists = [lst[:usable_frames] for lst in c_lists]

    # Compute global limits to keep color scales steady (optional)
    if fix_limits:
        def finite_minmax(arrs, pad_frac=0.05):
            flat = np.concatenate([np.ravel(a) for a in arrs])
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                return (None, None)
            vmin, vmax = flat.min(), flat.max()
            if vmin == vmax:
                # avoid zero range
                delta = 1e-9 if vmin == 0 else abs(vmin)*1e-3
                vmin -= delta
                vmax += delta
            # pad range to stabilize colorbars
            span = vmax - vmin
            vmin -= pad_frac * span
            vmax += pad_frac * span
            return (vmin, vmax)
        c_lims = [finite_minmax(lst) for lst in c_lists]
        # Skip the first timestep for phi when deriving global limits to avoid transient spikes
        phi_limit_data = phi_list[1:] if len(phi_list) > 1 else phi_list
        phi_lim = finite_minmax(phi_limit_data)
    else:
        c_lims = [None] * n_species
        phi_lim = None

    images = []
    print(f"\nRendering {usable_frames} frames via plot_solutions...")

    for i in range(usable_frames):
        frame_func = Function(W)
        for s in range(n_species):
            frame_func.sub(s).dat.data[:] = c_lists[s][i]
        frame_func.sub(phi_idx).dat.data[:] = phi_list[i]

        fig, _ = plot_solutions(frame_func, z_vals, mode, num_steps, dt, times[i],
                                save=False, show=False, output_png=None, return_fig=True,
                                c_lims=c_lims, phi_lim=phi_lim)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)  # fixed size; avoid bbox_inches to keep frames same shape
        buf.seek(0)
        images.append(imageio.v2.imread(buf))
        plt.close(fig)

    fname = RENDERS_DIR / f"pnp_animation{mode}.{fmt}"
    try:
        # imageio mp4 requires imageio-ffmpeg; if missing, this will raise
        imageio.mimsave(fname, images, fps=fps)
        print(f"  Saved animation to {fname}")
        return
    except Exception as e:
        print(f"  Warning: failed to save {fmt} via imageio ({e}); trying matplotlib FFMpegWriter...")
        try:
            Writer = animation.FFMpegWriter
            writer = Writer(fps=fps)
            fig = plt.figure()
            im = plt.imshow(images[0])
            with writer.saving(fig, fname, dpi=150):
                for frame in images:
                    im.set_data(frame)
                    writer.grab_frame()
            plt.close(fig)
            print(f"  Saved animation to {fname} via matplotlib/ffmpeg")
            return
        except Exception as e2:
            print(f"  Warning: matplotlib ffmpeg save failed ({e2}); falling back to GIF.")
            alt = RENDERS_DIR / f"pnp_animation{mode}.gif"
            try:
                imageio.mimsave(alt, images, fps=fps)
                print(f"  Saved animation to {alt}")
            except Exception as ee:
                print(f"  Failed to save animation as gif as well: {ee}")

    print("Animation generation complete!")
