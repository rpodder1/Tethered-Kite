"""
kite_compare.py

Compares three airborne wind energy flight strategies for the same
kite, weather, and location:

  1. Standard ascending figure-8 - the simple, symmetric weave shape
     most commercial pumping-cycle systems standardize on. Only winch
     settings (reel-out/reel-in speed, depower) are tuned to the site.

  2. Fly-Gen (onboard generator) - fixed-length tether, no winch;
     onboard rotors extract power via added drag while the kite flies
     a continuous crosswind weave (historically Makani). Uses the
     full dynamic simulation from kite_dynamics.py, not the
     quasi-steady approximation the other two use.

  3. Physics-optimal pumping cycle - weave shape and winch settings
     are searched together to find whatever maximizes energy.

Usage (interactive - prompts you for weather/location/kite):
    python kite_compare.py

Usage (scripted):
    python kite_compare.py --wind-speed 12 --shear 0.14 --area 20 \\
        --site-elevation 500 --preset atacama_coast

Output: kite_compare.png (flight-path + power + energy comparison)
and a printed report.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D

from kite_dynamics import KiteConfig
from kite_optimizer import WindEnvironment, optimize_route
from kite_pumping import optimize_fixed_figure8, optimize_physics_optimal
from kite_environments import LOCATION_PRESETS, build_environment, prompt_for_scenario


COLOR_FIXED = "#9AA7B8"     # standard figure-8 -- slate/grey
COLOR_FLYGEN = "#E8664F"    # Fly-Gen -- coral/red
COLOR_OPT = "#2E86AB"       # physics-optimal -- blue
COLOR_REEL_IN = "#555555"   # generic "reel-in" phase shade


# ─────────────────────────────────────────────
#  Comparison driver
# ─────────────────────────────────────────────

def run_comparison(cfg, env, l_min=300.0, l_max=650.0, elev0_deg=28.0,
                    flygen_tether_len=800.0, flygen_elev0_deg=30.0,
                    report_cycles=3, seed=0, verbose=True):
    if verbose:
        print("=" * 60)
        print("STEP 1/3 - Standard ascending figure-8 (winch settings only)")
        print("=" * 60)
    fixed_params, fixed_sim, _ = optimize_fixed_figure8(
        cfg, env, l_min=l_min, l_max=l_max, elev0_deg=elev0_deg,
        report_cycles=report_cycles, seed=seed, verbose=verbose)

    if verbose:
        print()
        print("=" * 60)
        print("STEP 2/3 - Physics-optimal pumping cycle (free-form shape + winch)")
        print("=" * 60)
    optimal_params, optimal_sim, _ = optimize_physics_optimal(
        cfg, env, l_min=l_min, l_max=l_max, elev0_deg=elev0_deg,
        report_cycles=report_cycles, seed=seed, verbose=verbose)

    # Match Fly-Gen's simulated window to the pumping cycle's natural
    # duration so all three are compared over identical wall-clock time.
    match_duration = fixed_sim["t"][-1]

    if verbose:
        print()
        print("=" * 60)
        print(f"STEP 3/3 - Fly-Gen (onboard generator, fixed tether, full dynamics)")
        print(f"  (matched to the same {match_duration:.0f}s window)")
        print("=" * 60)
    flygen_cfg = KiteConfig(mass=cfg.mass, area=cfg.area, CL=cfg.CL, CD=cfg.CD,
                             tether_len=flygen_tether_len,
                             site_elevation_msl=cfg.site_elevation_msl)
    flygen_result = optimize_route(flygen_cfg, env, report_t_end=match_duration,
                                    elev0_deg=flygen_elev0_deg, seed=seed,
                                    verbose=verbose)
    flygen_sim = _normalize_flygen_sim(flygen_result.best)

    return dict(fixed=(fixed_params, fixed_sim),
                optimal=(optimal_params, optimal_sim),
                flygen=(flygen_result.best_params, flygen_sim),
                match_duration=match_duration)


def _normalize_flygen_sim(sim):
    """
    Fly-Gen's sim dict (from kite_optimizer.py) has slightly different
    keys than the pumping-cycle sim dicts (no 'phase' or 'l', since
    it's a continuous weave on a fixed tether, not a reel-out/reel-in
    cycle). Add the missing keys so downstream plotting/report code
    can treat all three uniformly.
    """
    sim = dict(sim)  # shallow copy, don't mutate the original
    sim["phase"] = np.array(["out"] * len(sim["t"]))   # continuously "generating"
    sim["l"] = np.linalg.norm(sim["pos"], axis=1)
    sim["shape_loss"] = 0.0   # not a meaningful concept for Fly-Gen
    return sim


# ─────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────

def print_report(cfg, env, results, scenario_label=""):
    fixed_params, fixed_sim = results["fixed"]
    optimal_params, optimal_sim = results["optimal"]
    flygen_params, flygen_sim = results["flygen"]
    match_duration = results["match_duration"]

    fixed_Wh = fixed_sim["energy_J"] / 3600.0
    optimal_Wh = optimal_sim["energy_J"] / 3600.0
    flygen_Wh = flygen_sim["energy_J"] / 3600.0

    print()
    print("=" * 72)
    print("THREE-WAY COMPARISON")
    print("=" * 72)
    if scenario_label:
        print(f"Location:    {scenario_label}")
    print(f"Environment: {env}")
    print(f"Kite:        {cfg}")
    print(f"Comparison window: {match_duration:.0f} s ({match_duration/60:.1f} min)")
    print()
    print(f"{'':26s}{'Std figure-8':>15s}{'Fly-Gen':>15s}{'Physics-optimal':>17s}")
    print(f"{'Total energy (Wh)':26s}{fixed_Wh:15.2f}{flygen_Wh:15.2f}{optimal_Wh:17.2f}")
    print(f"{'Average power (kW)':26s}{fixed_sim['mean_power']/1000:15.2f}"
          f"{flygen_sim['mean_power']/1000:15.2f}{optimal_sim['mean_power']/1000:17.2f}")
    print(f"{'Peak power (kW)':26s}{fixed_sim['peak_power']/1000:15.2f}"
          f"{flygen_sim['peak_power']/1000:15.2f}{optimal_sim['peak_power']/1000:17.2f}")
    print(f"{'Peak tension (kN)':26s}{fixed_sim['peak_tension']/1000:15.2f}"
          f"{flygen_sim['peak_tension']/1000:15.2f}{optimal_sim['peak_tension']/1000:17.2f}")
    print()

    ro, ri, dp, nw, az, el = fixed_params
    print(f"Standard figure-8:   {nw:.1f} weaves/stroke, az=±{az:.0f}°, elev=±{el:.0f}° "
          f"(fixed shape); reel-out {ro:.2f} m/s, reel-in {ri:.2f} m/s, depower {dp*100:.0f}%")
    ro2, ri2, dp2, nw2, az2, el2 = optimal_params
    print(f"Physics-optimal:     {nw2:.1f} weaves/stroke, az=±{az2:.0f}°, elev=±{el2:.0f}° "
          f"(found by search); reel-out {ro2:.2f} m/s, reel-in {ri2:.2f} m/s, depower {dp2*100:.0f}%")
    az_amp, f, phi, elev_amp, gen_load = flygen_params
    print(f"Fly-Gen:             figure-8 az=±{np.degrees(az_amp):.0f}°, "
          f"elev=±{np.degrees(elev_amp):.0f}° @ {f:.3f} Hz (pursuit-guidance "
          f"tracked); generator load {gen_load*100:.0f}% "
          f"(fixed tether {flygen_sim['l'][0]:.0f}m)")
    if flygen_sim.get("crashed") or flygen_sim.get("stalled"):
        print("  ** WARNING: Fly-Gen trajectory crashed or stalled during the "
              "simulated window -- its numbers above are not a sustained-flight result.")
    print()

    entries = [("Standard figure-8", fixed_Wh), ("Fly-Gen", flygen_Wh),
               ("Physics-optimal", optimal_Wh)]
    winner_name, winner_wh = max(entries, key=lambda e: e[1])
    runner_up_wh = sorted([e[1] for e in entries], reverse=True)[1]
    print(f"Winner for these conditions: {winner_name} "
          f"({winner_wh:.1f} Wh, {(winner_wh/max(runner_up_wh,1e-9)-1)*100:+.0f}% "
          f"vs. the next best)")
    print()
    print("Notes on comparability: the two pumping-cycle strategies (Standard")
    print("figure-8, Physics-optimal) use a quasi-steady approximation, the same")
    print("kind used in early-stage AWE feasibility studies. Fly-Gen uses the full")
    print("dynamic simulation instead (real steering physics, RK4 integration).")
    print("The comparison is fair in energy/power/tension terms, but keep in mind")
    print("the two modeling approaches have different fidelity trade-offs.")
    print("=" * 72)


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def plot_comparison(cfg, env, results, scenario_label="",
                     path="kite_compare.png", dpi=150):
    fixed_params, fixed_sim = results["fixed"]
    optimal_params, optimal_sim = results["optimal"]
    flygen_params, flygen_sim = results["flygen"]

    fixed_Wh = fixed_sim["energy_J"] / 3600.0
    optimal_Wh = optimal_sim["energy_J"] / 3600.0
    flygen_Wh = flygen_sim["energy_J"] / 3600.0

    strategies = [
        ("Standard figure-8", fixed_sim, COLOR_FIXED, fixed_Wh),
        ("Fly-Gen", flygen_sim, COLOR_FLYGEN, flygen_Wh),
        ("Physics-optimal", optimal_sim, COLOR_OPT, optimal_Wh),
    ]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1],
                           width_ratios=[1, 1, 1], hspace=0.42, wspace=0.32)
    title = "Standard Figure-8 vs. Fly-Gen vs. Physics-Optimal"
    if scenario_label:
        title += f"  -  {scenario_label}"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    def draw_path(ax, sim, title_txt, color):
        x, y, z = sim["pos"][:, 0], sim["pos"][:, 1], sim["pos"][:, 2]
        is_out = sim["phase"] == "out"
        pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        colors = [color if is_out[i] else COLOR_REEL_IN for i in range(len(segs))]
        lc = Line3DCollection(segs, colors=colors, linewidth=2.0)
        ax.add_collection3d(lc)
        ax.set_xlim(x.min()-20, x.max()+20)
        ax.set_ylim(y.min()-20, y.max()+20)
        ax.set_zlim(0, z.max()+30)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.set_title(title_txt, fontsize=10.5, fontweight="bold", color=color)
        ax.view_init(elev=18, azim=-55)
        ax.tick_params(labelsize=7)

    # ── Panels 1-3: 3D flight paths ────────────────────────────
    for i, (name, sim, color, wh) in enumerate(strategies):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        draw_path(ax, sim, f"{name}\n{wh:.1f} Wh", color)

    # ── Panel 4: Power vs time (spans full width, row 2) ──────
    ax4 = fig.add_subplot(gs[1, :])
    for name, sim, color, wh in strategies:
        ax4.plot(sim["t"], sim["P"]/1000, color=color, linewidth=1.5, label=name)
    ax4.axhline(0, color="black", linewidth=0.6)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Power (kW)")
    ax4.set_title("Instantaneous power", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # ── Panel 5: Head-to-head bars ──────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    metrics = ["Energy\n(Wh)", "Avg pwr\n(kW)", "Peak pwr\n(kW)"]
    xpos = np.arange(len(metrics))
    w = 0.26
    for i, (name, sim, color, wh) in enumerate(strategies):
        vals = [wh, sim["mean_power"]/1000, sim["peak_power"]/1000]
        ax5.bar(xpos + (i-1)*w, vals, w, color=color, label=name)
    ax5.set_xticks(xpos)
    ax5.set_xticklabels(metrics, fontsize=8)
    ax5.set_title("Head-to-head", fontsize=11, fontweight="bold")
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3, axis="y")

    # ── Panel 6: Cumulative energy ──────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1:])
    for name, sim, color, wh in strategies:
        cum = np.concatenate([[0], np.cumsum(
            0.5*(sim["P"][1:]+sim["P"][:-1])*np.diff(sim["t"]))]) / 3600.0
        ax6.fill_between(sim["t"], cum, color=color, alpha=0.15)
        ax6.plot(sim["t"], cum, color=color, linewidth=2, label=f"{name} - {cum[-1]:.2f} Wh")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Cumulative energy (Wh)")
    ax6.set_title("Cumulative energy (same time window)", fontsize=11, fontweight="bold")
    ax6.legend(fontsize=8, loc="upper left")
    ax6.grid(alpha=0.3)

    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", type=str, default=None,
                    choices=list(LOCATION_PRESETS.keys()),
                    help="named location preset (skips interactive prompts)")
    p.add_argument("--wind-speed", type=float, default=None,
                    help="wind speed (m/s) at 10m reference height")
    p.add_argument("--shear", type=float, default=None,
                    help="wind shear exponent")
    p.add_argument("--site-elevation", type=float, default=None,
                    help="site elevation above sea level (m)")
    p.add_argument("--lwc", type=float, default=None,
                    help="fog liquid water content (kg/m^3)")
    p.add_argument("--mass", type=float, default=50.0)
    p.add_argument("--area", type=float, default=20.0)
    p.add_argument("--cl", type=float, default=1.0)
    p.add_argument("--cd", type=float, default=0.15)
    p.add_argument("--l-min", type=float, default=300.0)
    p.add_argument("--l-max", type=float, default=650.0)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="kite_compare.png")
    p.add_argument("--non-interactive", action="store_true",
                    help="never prompt; use flags/defaults only")
    return p.parse_args()


def main():
    args = _parse_args()

    any_manual_flag = any(v is not None for v in
                           [args.wind_speed, args.shear, args.site_elevation,
                            args.lwc]) or args.preset is not None

    if not any_manual_flag and not args.non_interactive and sys.stdin.isatty():
        cfg, env, scenario_label = prompt_for_scenario()
    else:
        overrides = {}
        if args.wind_speed is not None:
            overrides["wind_speed_ref"] = args.wind_speed
        if args.shear is not None:
            overrides["shear_exp"] = args.shear
        if args.site_elevation is not None:
            overrides["site_elevation_msl"] = args.site_elevation
        if args.lwc is not None:
            overrides["lwc"] = args.lwc

        env, site_elevation_msl = build_environment(args.preset, **overrides)
        cfg = KiteConfig(mass=args.mass, area=args.area, CL=args.cl, CD=args.cd,
                          site_elevation_msl=site_elevation_msl)
        scenario_label = (LOCATION_PRESETS[args.preset]["label"]
                           if args.preset else "Custom (CLI flags)")

    results = run_comparison(cfg, env, l_min=args.l_min, l_max=args.l_max,
                              report_cycles=args.cycles, seed=args.seed)

    print_report(cfg, env, results, scenario_label)
    out_path = plot_comparison(cfg, env, results, scenario_label, path=args.output)
    print(f"\nComparison plot saved: {out_path}")


if __name__ == "__main__":
    main()
