"""
kite_animate.py

Animated 3D flight visualization: the kite traces its path live, a
glowing marker sized and colored by instantaneous power output, a
comet-tail trail, and a live power-vs-time readout -- saved as a
looping GIF (renders fine inline in a GitHub README).

CLI usage:
    python kite_animate.py --arch groundgen
    python kite_animate.py --arch flygen
    python kite_animate.py --arch groundgen --optimize
    python kite_animate.py --help
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from kite_dynamics import KiteConfig, WindEnvironment
import kite_flygen
import kite_groundgen
import kite_compare
from kite_path_optimizer import maximize_energy
from kite_environments import LOCATION_PRESETS, build_environment


# ─────────────────────────────────────────────
#  Dark palette (validated categorical dark steps -- see the project's
#  dataviz reference palette; slot 1 = active/power color, everything
#  else is chart chrome from the same reference)
# ─────────────────────────────────────────────

BG_PAGE       = "#0d0d0d"
BG_SURFACE    = "#1a1a19"
INK_PRIMARY   = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED     = "#898781"
GRIDLINE      = "#2c2c2a"
BLUE          = "#3987e5"    # categorical slot 1 (dark)
ORANGE        = "#d95926"    # categorical slot 2 (dark)
ORANGE_HOT    = "#ff8a4c"
GRAY_INACTIVE = "#55534e"    # retraction / inactive phase
GRAY_INACTIVE_RGB = to_rgb(GRAY_INACTIVE)

# Fixed per-strategy identity colors for --compare mode (categorical --
# same role assignment as kite_compare.py's static plot, restepped for
# a dark surface): gray for the un-optimized baseline, blue and red are
# categorical dark slots 1 and 8.
CMP_COLORS = {
    "Standard figure-8": "#d8d6c5",
    "Physics-optimal":   BLUE,
    "Fly-Gen":           "#e66767",
}

POWER_CMAP = LinearSegmentedColormap.from_list(
    "power", ["#3a2418", ORANGE, ORANGE_HOT, "#ffe2b0"])


def power_color(frac):
    return POWER_CMAP(float(np.clip(frac, 0.0, 1.0)))


# ─────────────────────────────────────────────
#  Simulation
# ─────────────────────────────────────────────

def build_sim(arch, cfg, env, optimize, seed, maxiter, popsize, **kwargs):
    if arch == "flygen":
        mod = kite_flygen
        sim_args = (cfg, env, kwargs["duration"])
        params = mod.DEFAULT_PARAMS
    else:
        mod = kite_groundgen
        sim_args = (cfg, env, kwargs["l_min"], kwargs["l_max"],
                    kwargs["elev0"], kwargs["cycles"])
        params = mod.DEFAULT_PARAMS

    if optimize:
        params, sim, _ = maximize_energy(
            mod.simulate, mod.PARAM_BOUNDS, sim_args, seed=seed,
            maxiter=maxiter, popsize=popsize, seed_params=mod.DEFAULT_PARAMS,
            label=f"{arch} figure-8")
    else:
        sim = mod.simulate(params, *sim_args)
    return sim, params


# ─────────────────────────────────────────────
#  Styling helpers
# ─────────────────────────────────────────────

def _style_3d_axis(ax):
    ax.set_facecolor(BG_SURFACE)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(BG_SURFACE)
        pane.set_edgecolor(GRIDLINE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = GRIDLINE
        axis.label.set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_MUTED, labelsize=7)


def _ground_rings(ax, radius):
    theta = np.linspace(0, 2 * np.pi, 120)
    for frac in (0.5, 1.0):
        r = radius * frac
        ax.plot(r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta),
                color=GRIDLINE, linewidth=0.8, alpha=0.6)


def _cumulative_energy_wh(t, P):
    """
    Running total energy delivered so far (trapezoidal integral of P,
    in Wh) -- shown in the HUD instead of instantaneous power so
    retraction's cost shows up as the total simply dipping or
    flattening, the same way it does in the real world, rather than
    needing a sign flip plus a "(retracting)" label to explain a
    negative instantaneous number.
    """
    energy_J = np.concatenate([[0.0], np.cumsum(0.5 * (P[:-1] + P[1:]) * np.diff(t))])
    return energy_J / 3600.0


def _save_gif_optimized(fig, anim, path, fps, dpi):
    """
    matplotlib's PillowWriter writes an unoptimized (typically
    web-safe 216- or 256-color, no LZW dedup) GIF -- for a multi-path
    animation that's tens of MB. Re-encoding through PIL with an
    adaptive palette and optimize=True cuts that by ~3-4x with no
    visible quality loss (verified by inspection, not just assumed).
    """
    raw_path = path + ".raw.gif"
    anim.save(raw_path, writer=animation.PillowWriter(fps=fps), dpi=dpi,
              savefig_kwargs={"facecolor": BG_PAGE})
    plt.close(fig)

    from PIL import Image
    frames, durations = [], []
    with Image.open(raw_path) as im:
        for i in range(im.n_frames):
            im.seek(i)
            frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
            durations.append(im.info.get("duration", int(1000 / fps)))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    import os
    os.remove(raw_path)
    return path


# ─────────────────────────────────────────────
#  Animation
# ─────────────────────────────────────────────

TRAIL_WINDOW = 500   # most-recent real data points kept in the bright comet tail


def animate(sim, arch, path="kite_animation.gif", fps=24, n_frames=240, dpi=110):
    pos = sim["pos"]
    T, P, t = sim["T"], sim["P"], sim["t"]
    phase = sim.get("phase")
    n = len(t)

    peak_power = max(P.max(), 1.0)
    cum_wh = _cumulative_energy_wh(t, P)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    radius = np.max(np.hypot(x, y))

    fig = plt.figure(figsize=(13, 7.3), facecolor=BG_PAGE)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.5, 1.5, 1], hspace=0.5, wspace=0.25)

    ax3d = fig.add_subplot(gs[:, :2], projection="3d")
    ax3d.set_facecolor(BG_PAGE)
    _style_3d_axis(ax3d)
    _ground_rings(ax3d, radius)
    ax3d.scatter([0], [0], [0], color=INK_SECONDARY, s=40, marker="^", depthshade=False)
    ax3d.set_xlim(x.min() - 15, x.max() + 15)
    ax3d.set_ylim(y.min() - 15, y.max() + 15)
    ax3d.set_zlim(0, z.max() + 25)
    ax3d.set_xlabel("Downwind X (m)")
    ax3d.set_ylabel("Crosswind Y (m)")
    ax3d.set_zlabel("Altitude Z (m)")
    ax3d.view_init(elev=20, azim=-58)

    # Faint full-path guide, drawn once for context
    ax3d.plot(x, y, z, color=INK_MUTED, linewidth=0.6, alpha=0.25)

    tether_line, = ax3d.plot([0, x[0]], [0, y[0]], [0, z[0]],
                              color=INK_SECONDARY, linewidth=0.8, alpha=0.7)
    kite_halo = ax3d.scatter([x[0]], [y[0]], [z[0]], color=ORANGE, s=260,
                              alpha=0.25, depthshade=False)
    kite_dot = ax3d.scatter([x[0]], [y[0]], [z[0]], color=ORANGE, s=55,
                             depthshade=False, edgecolors="none")
    trail = Line3DCollection([], linewidths=2.2)
    ax3d.add_collection3d(trail, autolim=False)

    ax3d.set_title("Flight path -- wind blows in +X", color=INK_SECONDARY,
                    fontsize=10, pad=0)

    # Live power-vs-time strip
    axp = fig.add_subplot(gs[0:2, 2])
    axp.set_facecolor(BG_SURFACE)
    axp.set_xlim(t[0], t[-1])
    pmin = min(P.min(), 0.0)
    axp.set_ylim(pmin / 1000 - 1, peak_power / 1000 * 1.1)
    axp.axhline(0, color=GRIDLINE, linewidth=0.8)
    for spine in axp.spines.values():
        spine.set_color(GRIDLINE)
    axp.tick_params(colors=INK_MUTED, labelsize=7)
    axp.set_xlabel("Time (s)", color=INK_SECONDARY, fontsize=8)
    axp.set_ylabel("Power (kW)", color=INK_SECONDARY, fontsize=8)
    axp.grid(color=GRIDLINE, linewidth=0.5, alpha=0.5)
    power_line, = axp.plot([], [], color=ORANGE_HOT, linewidth=1.3)
    power_fill = [None]
    cursor = axp.axvline(t[0], color=INK_PRIMARY, linewidth=0.8, alpha=0.6)

    # HUD text
    hud = fig.text(0.015, 0.93, "", color=INK_PRIMARY, fontsize=11,
                    family="monospace", va="top",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_SURFACE,
                              edgecolor=GRIDLINE, alpha=0.88))

    title = "Fly-Gen" if arch == "flygen" else "Ground-Gen"
    fig.suptitle(f"{title} -- live flight", color=INK_PRIMARY,
                 fontsize=15, fontweight="bold", y=0.985)

    frame_idx = np.linspace(0, n - 1, n_frames).astype(int)

    def update(fi):
        i = frame_idx[fi]
        lo = max(0, i - TRAIL_WINDOW)
        xi, yi, zi = x[lo:i+1], y[lo:i+1], z[lo:i+1]

        tether_line.set_data_3d([0, x[i]], [0, y[i]], [0, z[i]])
        kite_halo._offsets3d = ([x[i]], [y[i]], [z[i]])
        kite_dot._offsets3d = ([x[i]], [y[i]], [z[i]])

        if len(xi) > 1:
            pts = np.array([xi, yi, zi]).T.reshape(-1, 1, 3)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            if phase is not None:
                colors = [GRAY_INACTIVE_RGB if phase[lo:i][j] == "in"
                          else power_color(P[lo:i][j] / peak_power)
                          for j in range(len(segs))]
            else:
                colors = [power_color(P[lo:i][j] / peak_power) for j in range(len(segs))]
            fade = np.linspace(0.15, 1.0, len(segs))
            colors = [(c[0], c[1], c[2], a) for c, a in zip(colors, fade)]
            trail.set_segments(segs)
            trail.set_color(colors)

        p_now = P[i]
        frac = max(p_now, 0.0) / peak_power
        glow_color = GRAY_INACTIVE if (phase is not None and phase[i] == "in") else power_color(frac)
        kite_dot.set_color([glow_color])
        kite_halo.set_color([glow_color])
        kite_halo.set_sizes([180 + 260 * frac])

        power_line.set_data(t[:i+1], P[:i+1] / 1000)
        if power_fill[0] is not None:
            power_fill[0].remove()
        power_fill[0] = axp.fill_between(t[:i+1], 0, P[:i+1] / 1000,
                                          color=ORANGE, alpha=0.25)
        cursor.set_xdata([t[i], t[i]])

        elev_deg = np.degrees(np.arcsin(np.clip(z[i] / max(np.linalg.norm(pos[i]), 1e-6), -1, 1)))
        lines = [
            f"t      {t[i]:6.1f} s",
            f"total  {cum_wh[i]:6.1f} Wh",
            f"tension{T[i]/1000:6.1f} kN",
            f"alt    {z[i]:6.0f} m  (elev {elev_deg:4.0f} deg)",
        ]
        hud.set_text("\n".join(lines))

        return tether_line, kite_dot, kite_halo, trail, power_line, cursor, hud

    ax3d.view_init(elev=20, azim=-58)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    return _save_gif_optimized(fig, anim, path, fps, dpi)


def animate_compare(results, path="kite_animation.gif", fps=24, n_frames=240, dpi=110):
    """
    All three kite_compare.py strategies animated together on one 3D
    plot -- each keeps a fixed identity color (see CMP_COLORS), each
    freezes at its final position once its own flight ends (they run
    different durations), and a live power-vs-time panel overlays all
    three.
    """
    strategies = [
        ("Standard figure-8", results["fixed"][1]),
        ("Physics-optimal",   results["optimal"][1]),
        ("Fly-Gen",           results["flygen"][1]),
    ]

    all_pos = np.concatenate([sim["pos"] for _, sim in strategies], axis=0)
    x_all, y_all, z_all = all_pos[:, 0], all_pos[:, 1], all_pos[:, 2]
    radius = np.max(np.hypot(x_all, y_all))
    global_peak_power = max(max(sim["P"].max() for _, sim in strategies), 1.0)
    p_lo = min(min(sim["P"].min() for _, sim in strategies), 0.0)
    t_max = max(sim["t"][-1] for _, sim in strategies)

    fig = plt.figure(figsize=(13, 7.3), facecolor=BG_PAGE)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.5, 1.5, 1], hspace=0.5, wspace=0.25)

    ax3d = fig.add_subplot(gs[:, :2], projection="3d")
    ax3d.set_facecolor(BG_PAGE)
    _style_3d_axis(ax3d)
    _ground_rings(ax3d, radius)
    ax3d.scatter([0], [0], [0], color=INK_SECONDARY, s=40, marker="^", depthshade=False)
    ax3d.set_xlim(x_all.min() - 15, x_all.max() + 15)
    ax3d.set_ylim(y_all.min() - 15, y_all.max() + 15)
    ax3d.set_zlim(0, z_all.max() + 25)
    ax3d.set_xlabel("Downwind X (m)")
    ax3d.set_ylabel("Crosswind Y (m)")
    ax3d.set_zlabel("Altitude Z (m)")
    ax3d.view_init(elev=20, azim=-58)
    ax3d.set_title("Flight paths -- wind blows in +X", color=INK_SECONDARY,
                    fontsize=10, pad=0)

    axp = fig.add_subplot(gs[0:2, 2])
    axp.set_facecolor(BG_SURFACE)
    axp.set_xlim(0, t_max)
    axp.set_ylim(p_lo / 1000 - 1, global_peak_power / 1000 * 1.1)
    axp.axhline(0, color=GRIDLINE, linewidth=0.8)
    for spine in axp.spines.values():
        spine.set_color(GRIDLINE)
    axp.tick_params(colors=INK_MUTED, labelsize=7)
    axp.set_xlabel("Time (s)", color=INK_SECONDARY, fontsize=8)
    axp.set_ylabel("Power (kW)", color=INK_SECONDARY, fontsize=8)
    axp.grid(color=GRIDLINE, linewidth=0.5, alpha=0.5)

    fig.suptitle("Standard vs. Physics-Optimal vs. Fly-Gen -- live",
                 color=INK_PRIMARY, fontsize=15, fontweight="bold", y=0.985)

    hud_time = fig.text(0.015, 0.93, "", color=INK_PRIMARY, fontsize=12,
                         family="monospace", va="top")

    paths = []
    for row, (name, sim) in enumerate(strategies):
        color = CMP_COLORS[name]
        pos = sim["pos"]
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        t, P, T = sim["t"], sim["P"], sim["T"]
        phase = sim.get("phase")
        peak = max(P.max(), 1.0)

        ax3d.plot(x, y, z, color=color, linewidth=0.9, alpha=0.45)
        tether_line, = ax3d.plot([0, x[0]], [0, y[0]], [0, z[0]],
                                  color=color, linewidth=0.7, alpha=0.5)
        halo = ax3d.scatter([x[0]], [y[0]], [z[0]], color=color, s=180,
                             alpha=0.28, depthshade=False)
        dot = ax3d.scatter([x[0]], [y[0]], [z[0]], color=color, s=45,
                            depthshade=False, edgecolors="none")
        trail = Line3DCollection([], linewidths=2.0)
        ax3d.add_collection3d(trail, autolim=False)

        power_line, = axp.plot([], [], color=color, linewidth=1.4, label=name)

        hud_row = fig.text(0.015, 0.885 - row * 0.045, "", color=color,
                            fontsize=10.5, family="monospace", va="top")

        paths.append(dict(name=name, color=to_rgb(color), x=x, y=y, z=z, t=t, P=P, T=T,
                           phase=phase, peak=peak, cum_wh=_cumulative_energy_wh(t, P),
                           tether_line=tether_line, halo=halo,
                           dot=dot, trail=trail, power_line=power_line, fill=None,
                           hud_row=hud_row))

    axp.legend(loc="lower right", fontsize=7, facecolor=BG_SURFACE,
               edgecolor=GRIDLINE, labelcolor=INK_SECONDARY)

    frame_times = np.linspace(0, t_max, n_frames)

    def update(fi):
        tg = frame_times[fi]
        artists = [hud_time]
        for pth in paths:
            t, x, y, z, P, T = pth["t"], pth["x"], pth["y"], pth["z"], pth["P"], pth["T"]
            i = min(int(np.searchsorted(t, tg, side="right")), len(t) - 1)
            lo = max(0, i - TRAIL_WINDOW)
            color = pth["color"]

            pth["tether_line"].set_data_3d([0, x[i]], [0, y[i]], [0, z[i]])
            pth["dot"]._offsets3d = ([x[i]], [y[i]], [z[i]])
            pth["halo"]._offsets3d = ([x[i]], [y[i]], [z[i]])

            xi, yi, zi = x[lo:i+1], y[lo:i+1], z[lo:i+1]
            if len(xi) > 1:
                pts = np.array([xi, yi, zi]).T.reshape(-1, 1, 3)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                fade = np.linspace(0.1, 0.95, len(segs))
                if pth["phase"] is not None:
                    dim = np.array([0.35 if pth["phase"][lo:i][j] == "in" else 1.0
                                     for j in range(len(segs))])
                    fade = fade * dim
                colors = [(color[0], color[1], color[2], a) for a in fade]
                pth["trail"].set_segments(segs)
                pth["trail"].set_color(colors)

            frac = max(P[i], 0.0) / global_peak_power
            retracting = pth["phase"] is not None and pth["phase"][i] == "in"
            alpha = 0.35 if retracting else 1.0
            pth["dot"].set_alpha(alpha)
            pth["halo"].set_sizes([120 + 260 * frac])
            pth["halo"].set_alpha(0.28 if not retracting else 0.1)

            j = min(i, len(t) - 1)
            pth["power_line"].set_data(t[:j+1], P[:j+1] / 1000)
            if pth["fill"] is not None:
                pth["fill"].remove()
            pth["fill"] = axp.fill_between(t[:j+1], 0, P[:j+1] / 1000,
                                            color=color, alpha=0.12)

            pth["hud_row"].set_text(f"{pth['name']:<18s}{pth['cum_wh'][i]:7.1f} Wh total")

            artists += [pth["tether_line"], pth["dot"], pth["halo"], pth["trail"],
                        pth["power_line"], pth["hud_row"]]

        hud_time.set_text(f"t = {tg:6.1f} s")
        return artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    return _save_gif_optimized(fig, anim, path, fps, dpi)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["flygen", "groundgen"], default="groundgen")
    p.add_argument("--optimize", action="store_true",
                    help="animate the physics-optimal flight instead of the default")
    p.add_argument("--compare", action="store_true",
                    help="animate all three kite_compare.py strategies together "
                         "on one plot instead of a single --arch flight")
    p.add_argument("--preset", type=str, default=None,
                    choices=list(LOCATION_PRESETS.keys()),
                    help="named location preset (--compare and single-arch both honor it)")
    p.add_argument("--wind-speed", type=float, default=12.0)
    p.add_argument("--shear", type=float, default=0.14)
    p.add_argument("--mass", type=float, default=50.0)
    p.add_argument("--area", type=float, default=20.0)
    p.add_argument("--tether-len", type=float, default=800.0,
                    help="Fly-Gen fixed tether length, m [800]")
    p.add_argument("--l-min", type=float, default=300.0)
    p.add_argument("--l-max", type=float, default=650.0)
    p.add_argument("--elev0", type=float, default=30.0)
    p.add_argument("--duration", type=float, default=90.0,
                    help="Fly-Gen flight duration, s [90]")
    p.add_argument("--cycles", type=int, default=1,
                    help="ground-gen number of pumping cycles [1]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=3)
    p.add_argument("--popsize", type=int, default=4)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--output", type=str, default="kite_animation.gif")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.preset is not None:
        env, site_elevation_msl = build_environment(args.preset)
    else:
        env = WindEnvironment(wind_speed_ref=args.wind_speed, shear_exp=args.shear)
        site_elevation_msl = 0.0
    print()

    if args.compare:
        cfg = KiteConfig(mass=args.mass, area=args.area, tether_len=args.l_max,
                          site_elevation_msl=site_elevation_msl)
        print("Running the full three-way search (this takes a few minutes -- "
              "each candidate is a full RK4 simulation)...")
        results = kite_compare.run_comparison(
            cfg, env, l_min=args.l_min, l_max=args.l_max, elev0_deg=args.elev0,
            n_cycles=args.cycles, seed=args.seed, maxiter=args.maxiter,
            popsize=args.popsize)
        print(f"\nRendering animation ({args.frames} frames @ {args.fps} fps)...")
        out = animate_compare(results, path=args.output, fps=args.fps,
                               n_frames=args.frames)
    else:
        cfg = KiteConfig(mass=args.mass, area=args.area,
                          tether_len=args.tether_len if args.arch == "flygen" else args.l_max,
                          site_elevation_msl=site_elevation_msl)
        sim, params = build_sim(
            args.arch, cfg, env, args.optimize, args.seed, args.maxiter, args.popsize,
            duration=args.duration, l_min=args.l_min, l_max=args.l_max,
            elev0=args.elev0, cycles=args.cycles)

        if sim.get("crashed") or sim.get("stalled"):
            print("** WARNING: this trajectory crashed or stalled -- animation "
                  "will show the failure.")

        print(f"Rendering animation ({args.frames} frames @ {args.fps} fps)...")
        out = animate(sim, args.arch, path=args.output, fps=args.fps,
                      n_frames=args.frames)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
