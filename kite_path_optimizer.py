"""
kite_path_optimizer.py

Generic flight-path/control search: given ANY simulate function with
the shared signature `simulate(params, cfg, env, ...) -> result dict`
(both kite_flygen.simulate and kite_groundgen.simulate follow this
shape), searches for the parameter vector that maximizes average power
using a global optimizer (scipy's differential_evolution), subject to
a tether-tension safety cap.

One search routine for both power architectures, since the objective
-- "run the physics sim, read off mean_power, penalize anything that
stalls, crashes, or overloads the tether" -- doesn't care which
architecture produced the numbers.

CLI usage:
    python kite_path_optimizer.py --arch flygen --wind-speed 12
    python kite_path_optimizer.py --arch groundgen --wind-speed 12
    python kite_path_optimizer.py --help
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from kite_dynamics import KiteConfig, WindEnvironment


# ─────────────────────────────────────────────
#  Generic global search
# ─────────────────────────────────────────────

def maximize_energy(simulate_fn, param_bounds, sim_args, tension_cap=None,
                     seed=0, maxiter=30, popsize=15, seed_params=None,
                     verbose=True, label="flight pattern"):
    """
    Search `param_bounds` for whatever maximizes `simulate_fn`'s
    reported mean_power.

    simulate_fn : callable(params, *sim_args) -> dict with at least
                  mean_power, peak_tension, and (optionally)
                  stalled/crashed bool flags.
    sim_args    : positional args passed after params -- by
                  convention (cfg, env, ...), cfg first (used to size
                  the default tension cap if one isn't given).
    seed_params : an optional known-reasonable starting point (e.g. a
                  module's DEFAULT_PARAMS), planted as one member of
                  the initial population. With a small search budget
                  and wide bounds, a fully random population can land
                  on a worse local optimum than a hand-tuned default --
                  seeding guarantees the search never does worse than
                  a known-decent starting point instead of just hoping
                  random initialization finds its way there too.

    Returns (best_params, best_sim, opt_info).
    """
    cfg = sim_args[0]
    if tension_cap is None:
        # A fast crosswind kite normally runs tether tension well above
        # static weight -- lift/weight ratios of several dozen are
        # standard in AWE flight (that's the entire point of crosswind
        # flight). 50x keeps the cap a genuine safety backstop against
        # runaway candidates without rejecting the normal operating
        # range this model's successful flights actually run in.
        tension_cap = 50.0 * cfg.mass * 9.81

    def objective(x):
        r = simulate_fn(x, *sim_args)
        penalty = 0.0
        if r.get("stalled"):
            penalty += 1e5
        if r.get("crashed"):
            penalty += 1e5
        if r["peak_tension"] > tension_cap:
            penalty += (r["peak_tension"] - tension_cap) * 10.0
        return -r["mean_power"] + penalty

    if verbose:
        print(f"Searching for the optimal {label} "
              f"(tension cap {tension_cap/1000:.1f} kN)...")

    # differential_evolution is a global optimizer: it evolves a whole
    # population of candidate parameter vectors instead of climbing a
    # gradient from one starting guess, which suits an objective built
    # out of a physics simulation (no clean derivative to hand a
    # gradient-based method).
    # polish=False: the default gradient-based finite-difference polish
    # step at the end is cheap for a closed-form objective, but here the
    # objective is a full RK4 simulation -- polishing can multiply total
    # evaluation count several-fold for very little benefit over what
    # differential_evolution's population search already finds.
    init = "latinhypercube"
    if seed_params is not None:
        rng = np.random.default_rng(seed)
        lo = np.array([b[0] for b in param_bounds])
        hi = np.array([b[1] for b in param_bounds])
        pop = rng.uniform(lo, hi, size=(popsize * len(param_bounds), len(param_bounds)))
        pop[0] = np.clip(seed_params, lo, hi)
        init = pop

    res = differential_evolution(
        objective, param_bounds, seed=seed, maxiter=maxiter, popsize=popsize,
        init=init, mutation=(0.4, 1.2), recombination=0.7, tol=1e-6,
        polish=False,
    )
    best_params = res.x

    if verbose:
        print(f"Done. Best found: {-res.fun/1000.0:.2f} kW")

    best_sim = simulate_fn(best_params, *sim_args)
    return best_params, best_sim, res


# ─────────────────────────────────────────────
#  Shared reporting / plotting
# ─────────────────────────────────────────────

def print_summary(sim, title, params):
    energy_Wh = sim["energy_J"] / 3600.0
    lines = []
    lines.append("=" * 58)
    lines.append(title.upper())
    lines.append("=" * 58)
    lines.append(f"Simulated window:     {sim['duration']:.0f} s")
    if sim["crashed"] or sim["stalled"]:
        lines.append("  ** WARNING: this trajectory crashed or stalled during the "
                      "simulated window --")
        lines.append("     the numbers below include whatever happened after that "
                      "point and should")
        lines.append("     not be trusted as a sustained-flight result.")
    lines.append(f"  Total energy produced: {energy_Wh:.2f} Wh")
    lines.append(f"  Average power:         {sim['mean_power']/1000:.2f} kW")
    lines.append(f"  Peak power:            {sim['peak_power']/1000:.2f} kW")
    lines.append(f"  Peak tether tension:   {sim['peak_tension']/1000:.2f} kN")
    lines.append(f"  Params: {np.round(np.asarray(params, dtype=float), 4).tolist()}")
    lines.append("=" * 58)
    print("\n".join(lines))


def plot_flight(sim, title, path="kite_flight.png", color="#2E86AB", dpi=150):
    """Single-strategy report plot: 3D path, power, cumulative energy,
    tension. Works for any sim dict with the shared t/pos/T/P keys, so
    it's used by kite_flygen.py, kite_groundgen.py, and this file's
    own CLI."""
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.5])
    fig.suptitle(title, fontsize=14, fontweight="bold")

    x, y, z = sim["pos"][:, 0], sim["pos"][:, 1], sim["pos"][:, 2]
    ax1 = fig.add_subplot(gs[:, 0], projection="3d")
    ax1.plot(x, y, z, color=color, linewidth=1.6)
    ax1.set_xlim(x.min() - 20, x.max() + 20)
    ax1.set_ylim(y.min() - 20, y.max() + 20)
    ax1.set_zlim(0, z.max() + 30)
    ax1.set_xlabel("Downwind X (m)")
    ax1.set_ylabel("Crosswind Y (m)")
    ax1.set_zlabel("Altitude Z (m)")
    ax1.set_title("Flight path (3D), wind blows in +X")
    ax1.view_init(elev=18, azim=-55)
    ax1.scatter([0], [0], [0], color="black", s=25, marker="^")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sim["t"], sim["P"]/1000, color=color)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Power (kW)")
    ax2.set_title("Instantaneous power")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    cum = np.concatenate([[0], np.cumsum(
        0.5*(sim["P"][1:]+sim["P"][:-1])*np.diff(sim["t"]))]) / 3600.0
    ax3b = ax3.twinx()
    ax3.fill_between(sim["t"], cum, color=color, alpha=0.2)
    ax3.plot(sim["t"], cum, color=color, linewidth=2)
    ax3b.plot(sim["t"], sim["T"]/1000, color="#c0392b", linewidth=1.0, alpha=0.7)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Cumulative energy (Wh)", color=color)
    ax3b.set_ylabel("Tether tension (kN)", color="#c0392b")
    ax3.set_title(f"Cumulative energy: {cum[-1]:.2f} Wh (tension overlaid)")
    ax3.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["flygen", "groundgen"], default="flygen",
                    help="which power architecture to optimize [flygen]")
    p.add_argument("--wind-speed", type=float, default=12.0)
    p.add_argument("--shear", type=float, default=0.14)
    p.add_argument("--mass", type=float, default=50.0)
    p.add_argument("--area", type=float, default=20.0)
    p.add_argument("--tether-len", type=float, default=800.0,
                    help="Fly-Gen fixed tether length, m [800]")
    p.add_argument("--l-min", type=float, default=300.0,
                    help="ground-gen minimum tether length, m [300]")
    p.add_argument("--l-max", type=float, default=650.0,
                    help="ground-gen maximum tether length, m [650]")
    p.add_argument("--elev0", type=float, default=30.0)
    p.add_argument("--duration", type=float, default=30.0,
                    help="Fly-Gen report window, s [30]")
    p.add_argument("--cycles", type=int, default=1,
                    help="ground-gen number of pumping cycles to report [1]")
    p.add_argument("--seed", type=int, default=0)
    # Each candidate is a full RK4 sim, not a cheap closed-form eval --
    # keep the default population*generations budget small so a CLI run
    # finishes in roughly a minute. Raise these for a more thorough
    # (proportionally slower) search.
    p.add_argument("--maxiter", type=int, default=3)
    p.add_argument("--popsize", type=int, default=4)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    env = WindEnvironment(wind_speed_ref=args.wind_speed, shear_exp=args.shear)
    cfg = KiteConfig(mass=args.mass, area=args.area, tether_len=args.tether_len)
    print()

    if args.arch == "flygen":
        import kite_flygen as arch
        sim_args = (cfg, env, args.duration)
        best_params, best_sim, _ = maximize_energy(
            arch.simulate, arch.PARAM_BOUNDS, sim_args, seed=args.seed,
            maxiter=args.maxiter, popsize=args.popsize,
            seed_params=arch.DEFAULT_PARAMS, label="Fly-Gen figure-8")
        title = "Fly-Gen -- physics-optimal figure-8"
        out = args.output or "kite_flygen_optimal.png"
    else:
        import kite_groundgen as arch
        sim_args = (cfg, env, args.l_min, args.l_max, args.elev0, args.cycles)
        best_params, best_sim, _ = maximize_energy(
            arch.simulate, arch.PARAM_BOUNDS, sim_args, seed=args.seed,
            maxiter=args.maxiter, popsize=args.popsize,
            seed_params=arch.DEFAULT_PARAMS, label="ground-gen pumping cycle")
        title = "Ground-Gen -- physics-optimal pumping cycle"
        out = args.output or "kite_groundgen_optimal.png"

    print()
    print_summary(best_sim, title, best_params)
    out_path = plot_flight(best_sim, title, out,
                            color="#2E86AB" if args.arch == "flygen" else "#f0a020")
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
