"""
kite_optimizer.py

Given a wind (and optional fog) environment, searches for the kite
steering pattern that maximizes energy production, then reports and
visualizes the result. Builds on kite_dynamics.py.

CLI usage:
    python kite_optimizer.py --wind-speed 12 --shear 0.14 --tether-len 800
    python kite_optimizer.py --help

Library usage:
    from kite_optimizer import WindEnvironment, optimize_route
    from kite_dynamics import KiteConfig

    env    = WindEnvironment(wind_speed_ref=12.0, shear_exp=0.14, lwc=0.2e-3)
    cfg    = KiteConfig(mass=50, area=20, CL=1.0, CD=0.15, tether_len=800)
    result = optimize_route(cfg, env)
    print(result.summary())
    result.plot("my_run.png")
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.optimize import differential_evolution, minimize

from kite_dynamics import (
    KiteConfig, kite_forces, rk4_step, water_collection_rate, ETA_MESH,
)


# ─────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────

class WindEnvironment:
    """
    Wind (and optional fog) conditions the kite flies in.

    Wind follows a power-law shear profile and blows along +X. Ground
    stations for these systems are rotated to face the wind, so wind
    direction isn't a free parameter here -- the kite's flight plane
    is always defined relative to the wind.

    Parameters
    ----------
    wind_speed_ref : float - wind speed (m/s) at `ref_height`
    ref_height     : float - reference measurement height (m), typically 10m
    shear_exp      : float - power-law shear exponent (~0.10-0.20 typical
                              over open terrain/water; higher over rougher
                              terrain)
    lwc            : float - liquid water content (kg/m^3) if flying through
                              fog for water harvesting, 0.0 disables it
    """

    def __init__(self, wind_speed_ref=12.0, ref_height=10.0, shear_exp=0.14,
                 lwc=0.0):
        self.wind_speed_ref = wind_speed_ref
        self.ref_height = ref_height
        self.shear_exp = shear_exp
        self.lwc = lwc

    def wind_at(self, altitude):
        h = max(altitude, 1.0)
        speed = self.wind_speed_ref * (h / self.ref_height) ** self.shear_exp
        return np.array([speed, 0.0, 0.0])

    def __repr__(self):
        return (f"WindEnvironment(v_ref={self.wind_speed_ref:.1f} m/s @ "
                f"{self.ref_height:.0f}m, shear_exp={self.shear_exp:.2f}, "
                f"lwc={self.lwc*1e3:.3f} g/m³)")


# ─────────────────────────────────────────────
#  Steering policy
# ─────────────────────────────────────────────

V_NOMINAL = 15.0   # m/s -- rough reference crosswind speed for path pacing


def reference_point(s, params, elev0, l):
    """
    Moving target point on a parametric figure-8 in azimuth/elevation
    space -- the flight-path shape the guidance law below chases.

    `s` is a path-phase variable, not wall-clock time: the caller
    advances it by how far the kite has actually traveled (scaled by
    V_NOMINAL), not by dt directly. A target that races ahead on a
    fixed time schedule regardless of whether the kite can keep up is
    exactly what produced the old open-loop law's runaway drift, so
    the target's pace here is synchronized to real progress instead --
    it waits if the kite is flying slower than nominal, and won't ask
    for a turn tighter than the kite can actually fly.

    params = [az_amp, f, phi, elev_amp, gen_load] (only the first 4
    are used here; gen_load is read separately in simulate())
        az_amp   : azimuth half-width of the figure-8 (rad)
        f        : loop frequency (Hz, at nominal speed)
        phi      : phase (rad)
        elev_amp : elevation half-height of the figure-8 (rad),
                   oscillating at 2x the azimuth frequency (a figure-8
                   traces the elevation swing twice per azimuth swing)
    """
    az_amp, f, phi, elev_amp = params[:4]
    ang = 2 * np.pi * f * s + phi
    az_t = az_amp * np.sin(ang)
    el_t = elev0 + elev_amp * np.sin(2 * ang)
    return l * np.array([np.cos(el_t) * np.cos(az_t),
                          np.cos(el_t) * np.sin(az_t),
                          np.sin(el_t)])


PARAM_BOUNDS = [
    (0.0, 1.1),        # az_amp    -- 0 to ~63 deg
    (0.02, 0.4),        # f         -- Hz
    (0.0, 2 * np.pi),   # phi
    (-0.5, 0.5),        # elev_amp  -- ~+-29 deg
    (0.0, 1.0),         # gen_load  -- fraction of CD_gen_max engaged
]


# ─────────────────────────────────────────────
#  Guidance / control
# ─────────────────────────────────────────────

GUIDANCE_KP = 3.0   # rad bank command per rad heading error


def pursuit_bank_command(pos, vel, target_pos, Kp=GUIDANCE_KP):
    """
    Proportional-navigation-style guidance: banks toward whatever
    heading closes the angle to a moving target point on the reference
    figure-8, instead of a bank angle indexed blindly by wall-clock
    time.

    The old open-loop `bank(t)` law had no path back from "where the
    kite actually is" to "what bank to fly" -- any drift from the
    assumed trajectory (mismatched initial velocity, a gust, numerical
    drift) just compounded, since the steering law couldn't see it.
    That's what produced the slow-dive-to-crash failure mode a patched
    velocity nudge was band-aiding. This closes the loop on the kite's
    actual state instead: bank angle is a feedback response to heading
    error, so any drift is corrected the same way it was introduced.
    """
    l = np.linalg.norm(pos)
    r_hat = pos / l

    los = target_pos - pos
    los_tan = los - np.dot(los, r_hat) * r_hat
    los_norm = np.linalg.norm(los_tan)
    if los_norm < 1e-6:
        return 0.0
    desired_hat = los_tan / los_norm

    vel_tan = vel - np.dot(vel, r_hat) * r_hat
    vel_norm = np.linalg.norm(vel_tan)
    current_hat = vel_tan / vel_norm if vel_norm > 1e-3 else desired_hat

    cos_err = np.clip(np.dot(current_hat, desired_hat), -1.0, 1.0)
    sin_err = np.dot(np.cross(current_hat, desired_hat), r_hat)
    heading_err = np.arctan2(sin_err, cos_err)
    # kite_forces' coordinated turn uses turn_dir = cross(tether_hat, vel_tan_hat)
    # = -cross(r_hat, current_hat), so positive bank curves current_hat the
    # opposite rotational sense from positive heading_err -- flip sign here
    # to close the loop the right way instead of reinforcing the error.
    return -Kp * heading_err


# ─────────────────────────────────────────────
#  Simulation
# ─────────────────────────────────────────────


def simulate(cfg, env, params, t_end, dt=0.05, elev0_deg=30.0, warmup=5.0):
    """
    Fly the kite under a given steering policy and return time
    histories plus summary totals (energy, water, stall/crash flags).
    """
    elev0 = np.radians(elev0_deg)
    # Start on the reference path itself, heading along its tangent there,
    # rather than a fixed heading unrelated to the commanded flight-path
    # shape. Starting off the path (or worse, with a heading exactly
    # radial to the turn-force geometry) is a near-singular initial
    # condition the guidance loop then has to fight its way out of before
    # it can do anything useful.
    ds = 1e-4
    p0 = reference_point(0.0, params, elev0, cfg.tether_len)
    p1 = reference_point(ds, params, elev0, cfg.tether_len)
    r_hat0 = p0 / np.linalg.norm(p0)
    tan_dir = p1 - p0
    tan_dir -= np.dot(tan_dir, r_hat0) * r_hat0
    tan_norm = np.linalg.norm(tan_dir)
    tan_dir = tan_dir / tan_norm if tan_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
    pos = p0
    vel = tan_dir * V_NOMINAL
    gen_load = float(np.clip(params[4], 0.0, 1.0))

    steps = int(t_end / dt)
    t_hist = np.zeros(steps)
    pos_hist = np.zeros((steps, 3))
    T_hist = np.zeros(steps)
    P_hist = np.zeros(steps)
    vrel_hist = np.zeros(steps)
    mdot_hist = np.zeros(steps)

    s = 0.0   # path-phase progress, advanced by actual speed, not by t
    for i in range(steps):
        t = i * dt
        wind = env.wind_at(pos[2])
        target = reference_point(s, params, elev0, np.linalg.norm(pos))
        bank = np.clip(pursuit_bank_command(pos, vel, target), -1.3, 1.3)

        F, T, P, vr = kite_forces(pos, vel, wind, cfg, bank_angle=bank,
                                   gen_load=gen_load)
        v_rel_mag = np.linalg.norm(vr)

        t_hist[i] = t
        pos_hist[i] = pos
        T_hist[i] = T
        P_hist[i] = P
        vrel_hist[i] = v_rel_mag
        if env.lwc > 0:
            mdot_hist[i] = water_collection_rate(v_rel_mag, env.lwc, cfg)

        r_hat = pos / np.linalg.norm(pos)
        v_tan_mag = np.linalg.norm(vel - np.dot(vel, r_hat) * r_hat)
        s += dt * v_tan_mag / V_NOMINAL

        pos, vel, T, P, vr = rk4_step(pos, vel, wind, cfg, bank, dt,
                                       gen_load=gen_load)

    mask = t_hist >= warmup
    t_eval = t_hist[mask]
    duration = t_eval[-1] - t_eval[0] if len(t_eval) > 1 else 0.0
    energy_J = np.trapezoid(P_hist[mask], t_eval) if duration > 0 else 0.0
    mean_power = energy_J / duration if duration > 0 else 0.0
    water_kg = np.trapezoid(mdot_hist[mask], t_eval) if duration > 0 else 0.0

    stalled = bool(np.any(vrel_hist[mask] < 3.0))
    crashed = bool(np.any(pos_hist[mask, 2] < 5.0))

    return dict(
        t=t_hist, pos=pos_hist, T=T_hist, P=P_hist, vrel=vrel_hist,
        mdot=mdot_hist, energy_J=energy_J, mean_power=mean_power,
        peak_power=P_hist[mask].max() if duration > 0 else 0.0,
        peak_tension=T_hist[mask].max() if duration > 0 else 0.0,
        water_kg=water_kg, duration=duration, stalled=stalled,
        crashed=crashed,
    )


def _objective(x, cfg, env, t_end, dt, elev0_deg, warmup, tension_cap):
    r = simulate(cfg, env, x, t_end, dt, elev0_deg, warmup)
    penalty = 0.0
    if r["stalled"]:
        penalty += 1e5
    if r["crashed"]:
        penalty += 1e5
    if r["peak_tension"] > tension_cap:
        penalty += (r["peak_tension"] - tension_cap) * 10.0
    return -r["mean_power"] + penalty


# ─────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────

class OptimizationResult:
    def __init__(self, cfg, env, best_params, best_sim, baseline_sim,
                 report_t_end, opt_info):
        self.cfg = cfg
        self.env = env
        self.best_params = best_params
        self.best = best_sim
        self.baseline = baseline_sim
        self.report_t_end = report_t_end
        self.opt_info = opt_info

    @property
    def energy_kWh(self):
        return self.best["energy_J"] / 3.6e6

    @property
    def baseline_energy_kWh(self):
        return self.baseline["energy_J"] / 3.6e6

    def summary(self):
        az_amp, f, phi, elev_amp, gen_load = self.best_params
        CD_gen = gen_load * self.cfg.CD_gen_max
        lines = []
        lines.append("=" * 58)
        lines.append("OPTIMAL KITE ROUTE - RESULTS")
        lines.append("=" * 58)
        lines.append(f"Environment: {self.env}")
        lines.append(f"Kite:        {self.cfg}")
        lines.append("")
        lines.append("Optimal reference figure-8 (tracked via closed-loop pursuit")
        lines.append("guidance, not an open-loop bank schedule):")
        lines.append(f"  azimuth half-width:   {np.degrees(az_amp):.1f} deg")
        lines.append(f"  elevation half-height: {np.degrees(elev_amp):.1f} deg")
        lines.append(f"  loop frequency:       {f:.3f} Hz (period {1/f:.1f} s), phase {phi:.2f} rad")
        lines.append(f"  generator loading: {gen_load*100:.0f}% of max "
                      f"(CD_gen = {CD_gen:.3f}, on top of CD = {self.cfg.CD})")
        lines.append("")
        energy_Wh = self.best["energy_J"] / 3600.0
        base_Wh = self.baseline["energy_J"] / 3600.0
        lines.append(f"Simulated flight window: {self.report_t_end:.0f} s")
        if self.best["crashed"] or self.best["stalled"]:
            lines.append("  ** WARNING: this trajectory crashed or stalled during the "
                          "simulated window --")
            lines.append("     the numbers below include whatever happened after that "
                          "point and should")
            lines.append("     not be trusted as a sustained-flight result. Consider a "
                          "longer opt_t_end")
            lines.append("     so the search can see this failure mode before picking "
                          "a winner.")
        lines.append(f"  Total energy produced:   {energy_Wh:.2f} Wh "
                      f"({self.energy_kWh:.4f} kWh)")
        lines.append(f"  Average power:           {self.best['mean_power']:.1f} W")
        lines.append(f"  Peak power:              {self.best['peak_power']:.1f} W")
        lines.append(f"  Peak tether tension:     {self.best['peak_tension']:.0f} N")
        if self.env.lwc > 0:
            lines.append(f"  Water collected:         {self.best['water_kg']:.2f} kg "
                          f"({self.best['water_kg']:.2f} L)")
        lines.append("")
        improvement = (energy_Wh / base_Wh - 1.0) * 100 if base_Wh > 1e-9 else float("inf")
        lines.append("Contribution of active steering specifically (same generator")
        lines.append("loading as above, but reference held at a fixed point --")
        lines.append("no figure-8 weaving):")
        lines.append(f"  Fixed-heading energy:    {base_Wh:.2f} Wh")
        lines.append(f"  Steering improvement:    {improvement:+.0f}%")
        lines.append("=" * 58)
        return "\n".join(lines)

    def plot(self, path="kite_route.png", dpi=150):
        best, base = self.best, self.baseline

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.4, 1.4],
                               height_ratios=[1, 1])
        fig.suptitle("Optimal Kite Flight Route for Maximum Energy Production",
                     fontsize=14, fontweight="bold")

        # ── Flight path in 3D (X=downwind, Y=crosswind, Z=altitude) ──
        ax1 = fig.add_subplot(gs[:, 0], projection="3d")
        x, y, z = best["pos"][:, 0], best["pos"][:, 1], best["pos"][:, 2]
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        power_W = best["P"][:-1]
        lc = Line3DCollection(segments, cmap="plasma",
                               norm=plt.Normalize(power_W.min(), power_W.max()))
        lc.set_array(power_W)
        lc.set_linewidth(2.5)
        ax1.add_collection3d(lc)
        ax1.set_xlim(x.min() - 20, x.max() + 20)
        ax1.set_ylim(y.min() - 20, y.max() + 20)
        ax1.set_zlim(0, z.max() + 30)
        ax1.set_xlabel("Downwind X (m)")
        ax1.set_ylabel("Crosswind Y (m)")
        ax1.set_zlabel("Altitude Z (m)")
        ax1.set_title("Optimal flight path (3D)\nwind blows in +X")
        ax1.view_init(elev=18, azim=-55)
        # anchor point for reference
        ax1.scatter([0], [0], [0], color="black", s=25, marker="^")
        ax1.text(0, 0, 15, "anchor", fontsize=7)
        cb = fig.colorbar(lc, ax=ax1, fraction=0.046, pad=0.1,
                           orientation="horizontal", location="top")
        cb.set_label("Instantaneous power (W)")

        # ── Power vs time (optimal vs baseline) ───────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(best["t"], best["P"], color="#f0a020",
                 label="Optimized figure-8 (active steering)")
        ax2.plot(base["t"], base["P"], color="#888888",
                 linestyle="--", linewidth=1.2,
                 label="Fixed heading (no steering)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Power (W)")
        ax2.set_title("Instantaneous power")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # ── Cumulative energy ──────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        cum_best = np.concatenate([[0], np.cumsum(
            0.5 * (best["P"][1:] + best["P"][:-1]) * np.diff(best["t"]))]) / 3600.0
        cum_base = np.concatenate([[0], np.cumsum(
            0.5 * (base["P"][1:] + base["P"][:-1]) * np.diff(base["t"]))]) / 3600.0
        ax3.fill_between(best["t"], cum_best, color="#f0a020", alpha=0.3)
        ax3.plot(best["t"], cum_best, color="#f0a020",
                 label="Optimized figure-8")
        ax3.plot(base["t"], cum_base, color="#888888", linestyle="--",
                 linewidth=1.2, label="Fixed heading (no steering)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Cumulative energy (Wh)")
        ax3.set_title(f"Total energy: {cum_best[-1]:.2f} Wh "
                       f"({(cum_best[-1]/cum_base[-1]-1)*100:+.0f}% vs fixed heading)"
                       if cum_base[-1] > 1e-9 else "Total energy")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # ── Tether tension ─────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.plot(best["t"], best["T"], color="#c0392b",
                 label="Optimized figure-8")
        ax4.plot(base["t"], base["T"], color="#888888", linestyle="--",
                 linewidth=1.2, label="Fixed heading (no steering)")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Tension (N)")
        ax4.set_title("Tether tension")
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(path, dpi=dpi)
        plt.close(fig)
        return path


# ─────────────────────────────────────────────
#  Top-level optimizer
# ─────────────────────────────────────────────

def optimize_route(cfg, env, opt_t_end=50.0, report_t_end=90.0, opt_dt=0.15,
                    report_dt=0.05, elev0_deg=30.0, warmup=4.0,
                    tension_cap=None, seed=0, maxiter=6, popsize=6,
                    verbose=True):
    """
    Search for the steering pattern that maximizes average power for
    `cfg` flying in `env`, then re-simulate the winning pattern (and a
    fixed-heading baseline) over a longer `report_t_end` window for
    reporting.

    `opt_t_end` needs to be long enough for the search to actually see
    a candidate's crash/stall failure mode (the `_objective` penalty
    only fires within the window it evaluates) -- a candidate that
    looks fine for 20s can still fail by 200s+. Raising it trades
    search-time compute for confidence that the winner holds up over
    `report_t_end`; `maxiter`/`popsize` are lowered from a bare
    increase to keep total search cost roughly similar to a shorter
    horizon with a larger population.

    Returns an OptimizationResult.
    """
    if tension_cap is None:
        # Generic structural safety margin: ~8x static weight
        tension_cap = 8.0 * cfg.mass * 9.81

    if verbose:
        print(f"Environment: {env}")
        print(f"Kite:        {cfg}")
        print("Searching for the optimal steering pattern "
              f"(this simulates ~{opt_t_end:.0f}s of flight per candidate)...")

    # Stage 1: differential evolution for a global search. polish=False
    # so the gradient refinement below is an explicit, separately-
    # reported stage rather than a hidden step inside scipy's call.
    args = (cfg, env, opt_t_end, opt_dt, elev0_deg, warmup, tension_cap)
    res = differential_evolution(
        _objective, PARAM_BOUNDS, args=args,
        seed=seed, maxiter=maxiter, popsize=popsize,
        mutation=(0.4, 1.2), recombination=0.7, tol=1e-3, polish=False,
    )
    global_kw = -res.fun / 1000.0

    # Stage 2: local gradient-based refinement (finite-difference
    # gradient, L-BFGS-B) starting from the global search's winner.
    polish_res = minimize(_objective, res.x, args=args, method="L-BFGS-B",
                           bounds=PARAM_BOUNDS)
    if polish_res.fun < res.fun:
        best_params = polish_res.x
        polish_kw = -polish_res.fun / 1000.0
    else:
        best_params = res.x
        polish_kw = global_kw

    if verbose:
        print(f"Done. Global search: {global_kw:.2f} kW -> "
              f"gradient refine: {polish_kw:.2f} kW "
              f"(re-simulating over {report_t_end:.0f}s for the report)")

    best_sim = simulate(cfg, env, best_params, report_t_end, report_dt,
                         elev0_deg, warmup)
    # Baseline isolates the effect of active steering: same generator
    # loading as the optimized solution, but the reference held at a
    # fixed point (no figure-8 weaving).
    baseline_params = [0.0, 0.1, 0.0, 0.0, best_params[4]]
    baseline_sim = simulate(cfg, env, baseline_params, report_t_end,
                             report_dt, elev0_deg, warmup)

    return OptimizationResult(cfg, env, best_params, best_sim, baseline_sim,
                               report_t_end, res)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Environment
    p.add_argument("--wind-speed", type=float, default=12.0,
                    help="wind speed (m/s) at reference height [12.0]")
    p.add_argument("--ref-height", type=float, default=10.0,
                    help="reference height for wind speed (m) [10.0]")
    p.add_argument("--shear", type=float, default=0.14,
                    help="wind shear exponent, power law [0.14]")
    p.add_argument("--lwc", type=float, default=0.0,
                    help="fog liquid water content, kg/m^3 [0.0 = no fog]")
    # Kite
    p.add_argument("--mass", type=float, default=50.0, help="kite mass, kg [50]")
    p.add_argument("--area", type=float, default=20.0, help="wing area, m^2 [20]")
    p.add_argument("--cl", type=float, default=1.0, help="lift coefficient [1.0]")
    p.add_argument("--cd", type=float, default=0.15, help="drag coefficient [0.15]")
    p.add_argument("--tether-len", type=float, default=800.0,
                    help="tether length, m [800]")
    p.add_argument("--cd-gen-max", type=float, default=0.5,
                    help="max onboard generator drag coefficient [0.5]")
    p.add_argument("--elev0", type=float, default=30.0,
                    help="starting elevation angle, deg [30]")
    # Optimization / reporting
    p.add_argument("--opt-duration", type=float, default=20.0,
                    help="seconds simulated per candidate during search [20]")
    p.add_argument("--report-duration", type=float, default=90.0,
                    help="seconds simulated for the final report/plot [90]")
    p.add_argument("--seed", type=int, default=0, help="optimizer random seed")
    p.add_argument("--output", type=str, default="kite_route.png",
                    help="output plot filename [kite_route.png]")
    return p.parse_args()


def main():
    args = _parse_args()
    env = WindEnvironment(wind_speed_ref=args.wind_speed,
                           ref_height=args.ref_height,
                           shear_exp=args.shear, lwc=args.lwc)
    cfg = KiteConfig(mass=args.mass, area=args.area, CL=args.cl, CD=args.cd,
                      tether_len=args.tether_len, CD_gen_max=args.cd_gen_max)
    print()
    result = optimize_route(cfg, env, opt_t_end=args.opt_duration,
                             report_t_end=args.report_duration,
                             elev0_deg=args.elev0, seed=args.seed)
    print()
    print(result.summary())
    out_path = result.plot(args.output)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
