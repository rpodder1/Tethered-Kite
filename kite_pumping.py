"""
kite_pumping.py

Models the "pumping cycle" (reel-out / reel-in) power architecture
that most commercial airborne wind energy companies fly (Kitepower,
TwingTec, SkySails Power, Kitemill): the kite flies figure-8 loops
while the tether reels out under load (the power phase), then gets
pitched to a low-lift "depowered" state and reeled back in fast and
cheap (the retraction phase), and repeats.

Power model - quasi-steady crosswind kite theory (Loyd, 1980), the
same approximation used in early-stage AWE feasibility studies:

  Achieved crosswind airspeed:   v_a  ~ v_wind_eff * (CL/CD) * eta_xw
  Aerodynamic resultant force:   F  =  q * A * CL * sqrt(1 + (CD/CL)^2)
  Tether tension:                T ~ F
  Winch power:                   P  =  T * l_dot * eta_gen

Flight-path shape matters here for two reasons:

  1. Reel-speed dilution: reeling out subtracts from the wind speed
     available to power crosswind flight (part of the apparent wind
     goes into paying the tether out, not building airspeed).
  2. Cornering loss: the kite decelerates through each turn and
     reaccelerates down the straight, so tighter/more frequent
     weaving costs speed. Modeled with a simple turn_loss_factor.

Compares two ways to fly it:

  Fixed figure-8   - the simple, symmetric weave shape most companies
                      standardize on; only winch settings are tuned.
  Physics-optimal  - weave shape and winch settings are all searched
                      together to find whatever maximizes energy.
"""

import numpy as np
from scipy.optimize import differential_evolution

from kite_dynamics import KiteConfig, air_density, ETA_GEN, G
from kite_optimizer import WindEnvironment


# ─────────────────────────────────────────────
#  Quasi-steady crosswind kite power theory
# ─────────────────────────────────────────────

CROSSWIND_EFFICIENCY = 0.6   # fraction of Loyd's theoretical-max crosswind
                              # speed real systems achieve (imperfect
                              # steering, gusts, other losses)
V_REL_MAX = 55.0              # m/s - realistic structural/aero speed cap
BANK_MAX_STRUCTURAL = np.radians(60.0)   # realistic max bank a kite can hold


def turn_loss_factor(n_weaves, az_amp_rad, elev_amp_rad, v_ideal_straight, l_mid):
    """
    Fractional speed loss from cornering through a figure-8, derived
    from the same coordinated-turn relation the full RK4 dynamics model
    uses (`a_turn = g * tan(bank)` in kite_dynamics.kite_forces), so
    both fidelity levels agree on what a turn costs instead of each
    using its own free-fit curve.

    The total angular sweep of the figure-8 is divided among the
    n_weaves loops to get an effective turn radius; holding that radius
    at the (pre-loss) ideal crosswind speed requires a centripetal
    acceleration that may exceed what a structurally realistic bank
    angle can sustain -- the shortfall is the speed loss.
    """
    r_eff = max(l_mid * (az_amp_rad + elev_amp_rad) / max(n_weaves, 1e-6), 1.0)
    a_c = v_ideal_straight**2 / r_eff
    a_max = G * np.tan(BANK_MAX_STRUCTURAL)
    if a_c <= a_max:
        return 0.0
    return min(1.0 - np.sqrt(a_max / a_c), 0.85)


def crosswind_airspeed(v_wind_eff, CL, CD, shape_loss=0.0):
    """
    Achieved crosswind airspeed. Loyd's (1980) theory says a kite with
    glide ratio E = CL/CD flying perfectly crosswind could reach an
    airspeed ~E times the (effective) wind speed; real systems achieve
    a fraction of that, further reduced by cornering losses from the
    specific flight-path shape flown, and capped at a realistic
    structural/aerodynamic speed limit.
    """
    v_wind_eff = max(v_wind_eff, 0.0)
    v_ideal = v_wind_eff * (CL / CD) * CROSSWIND_EFFICIENCY * (1.0 - shape_loss)
    return min(v_ideal, V_REL_MAX)


def aero_resultant_force(rho, v_rel, area, CL, CD):
    """Magnitude of the combined lift+drag resultant (= tether tension
    for a taut, well-flown crosswind kite)."""
    q = 0.5 * rho * v_rel**2
    return q * area * CL * np.sqrt(1.0 + (CD / CL)**2)


# ─────────────────────────────────────────────
#  Pumping-cycle simulation (quasi-steady + parametric path)
# ─────────────────────────────────────────────

# Parameter vector everywhere in this module:
#   [reel_out_speed, reel_in_speed, depower_frac,
#    n_weaves, az_amp_deg, elev_amp_deg]

FIXED_SHAPE = dict(n_weaves=3.0, az_amp_deg=22.0, elev_amp_deg=9.0)

FULL_BOUNDS = [
    (0.5, 8.0),     # reel_out_speed  (m/s)
    (2.0, 20.0),    # reel_in_speed   (m/s, magnitude)
    (0.05, 0.6),    # depower_frac    (CL scale during retraction)
    (1.5, 10.0),    # n_weaves        (loops per reel-out stroke)
    (14.0, 45.0),   # az_amp_deg      (figure-8 azimuth half-width)
    (5.0, 25.0),    # elev_amp_deg    (figure-8 elevation half-height)
]

WINCH_ONLY_BOUNDS = FULL_BOUNDS[:3]   # for the fixed-figure-8 baseline


def simulate_pumping_cycle(cfg, env, params, l_min=300.0, l_max=650.0,
                            elev0_deg=28.0, n_cycles=3, report_dt=0.2):
    """
    Simulate `n_cycles` reel-out/reel-in pumping cycles using the
    quasi-steady power model, with a parametric expanding-figure-8
    flight path for visualization.

    Parameters
    ----------
    cfg, env       : KiteConfig, WindEnvironment
    params         : [reel_out_speed, reel_in_speed, depower_frac,
                       n_weaves, az_amp_deg, elev_amp_deg]
    l_min, l_max   : float - tether length range for the cycle (m)
    elev0_deg      : float - center elevation angle (deg)
    n_cycles       : int   - number of out+in cycles
    report_dt      : float - nominal spacing of reported path points

    Returns a dict of time histories plus summary totals.
    """
    (reel_out_speed, reel_in_speed, depower_frac,
     n_weaves, az_amp_deg, elev_amp_deg) = params

    elev0 = np.radians(elev0_deg)
    az_amp = np.radians(az_amp_deg)
    elev_amp = np.radians(elev_amp_deg)

    # Cycle-invariant: wind/air density only depend on the mid-tether
    # altitude, not on which cycle we're in.
    l_mid = 0.5 * (l_min + l_max)
    alt_mid = l_mid * np.sin(elev0)
    v_wind = np.linalg.norm(env.wind_at(alt_mid))
    rho = air_density(alt_mid, cfg.site_elevation_msl)

    v_ideal_straight = max(v_wind - reel_out_speed, 0.0) * (cfg.CL / cfg.CD) * CROSSWIND_EFFICIENCY
    shape_loss = turn_loss_factor(n_weaves, az_amp, elev_amp, v_ideal_straight, l_mid)

    t_hist, l_hist, pos_hist = [], [], []
    T_hist, P_hist, phase_hist = [], [], []

    t = 0.0
    for cyc in range(n_cycles):
        # ── Reel-out (power) phase ─────────────────────────────
        v_rel_out = crosswind_airspeed(v_wind - reel_out_speed, cfg.CL,
                                        cfg.CD, shape_loss)
        F_out = aero_resultant_force(rho, v_rel_out, cfg.area, cfg.CL, cfg.CD)
        T_out = F_out
        P_out = T_out * reel_out_speed * ETA_GEN

        duration_out = (l_max - l_min) / reel_out_speed
        n_steps_out = max(int(duration_out / report_dt), 8)
        for i in range(n_steps_out):
            frac = i / (n_steps_out - 1)
            l_i = l_min + frac * (l_max - l_min)
            phi = frac * 2 * np.pi * n_weaves
            az = az_amp * np.sin(phi)
            el = elev0 + elev_amp * np.sin(2 * phi)
            pos = np.array([l_i * np.cos(el) * np.cos(az),
                             l_i * np.cos(el) * np.sin(az),
                             l_i * np.sin(el)])
            t_hist.append(t + frac * duration_out)
            l_hist.append(l_i)
            pos_hist.append(pos)
            T_hist.append(T_out)
            P_hist.append(P_out)
            phase_hist.append("out")
        t += duration_out

        # ── Reel-in (retraction) phase ──────────────────────────
        CL_retract = cfg.CL * depower_frac
        v_rel_in = crosswind_airspeed(v_wind, CL_retract, cfg.CD, 0.0)
        F_in = aero_resultant_force(rho, v_rel_in, cfg.area, CL_retract, cfg.CD)
        T_in = F_in
        P_in = -T_in * reel_in_speed * ETA_GEN

        duration_in = (l_max - l_min) / reel_in_speed
        n_steps_in = max(int(duration_in / report_dt), 8)
        for i in range(n_steps_in):
            frac = i / (n_steps_in - 1)
            l_i = l_max - frac * (l_max - l_min)
            phi = frac * 2 * np.pi * 1.0
            az = 0.3 * az_amp * np.sin(phi)
            el = elev0 + 0.3 * elev_amp * np.sin(2 * phi)
            pos = np.array([l_i * np.cos(el) * np.cos(az),
                             l_i * np.cos(el) * np.sin(az),
                             l_i * np.sin(el)])
            t_hist.append(t + frac * duration_in)
            l_hist.append(l_i)
            pos_hist.append(pos)
            T_hist.append(T_in)
            P_hist.append(P_in)
            phase_hist.append("in")
        t += duration_in

    t_hist = np.array(t_hist)
    l_hist = np.array(l_hist)
    pos_hist = np.array(pos_hist)
    T_hist = np.array(T_hist)
    P_hist = np.array(P_hist)
    phase_hist = np.array(phase_hist)

    energy_J = np.trapezoid(P_hist, t_hist)
    duration = t_hist[-1] - t_hist[0]
    mean_power = energy_J / duration if duration > 0 else 0.0

    return dict(
        t=t_hist, l=l_hist, pos=pos_hist, T=T_hist, P=P_hist,
        phase=phase_hist, energy_J=energy_J, mean_power=mean_power,
        peak_power=P_hist.max(), min_power=P_hist.min(),
        peak_tension=T_hist.max(), duration=duration,
        shape_loss=shape_loss, params=list(params),
    )


def _objective(x, cfg, env, l_min, l_max, elev0_deg, tension_cap,
               fixed_shape):
    if fixed_shape is not None:
        full_x = list(x) + [fixed_shape["n_weaves"], fixed_shape["az_amp_deg"],
                             fixed_shape["elev_amp_deg"]]
    else:
        full_x = x
    r = simulate_pumping_cycle(cfg, env, full_x, l_min, l_max, elev0_deg,
                                n_cycles=1)
    penalty = 0.0
    if r["peak_tension"] > tension_cap:
        penalty += (r["peak_tension"] - tension_cap) * 5.0
    return -r["mean_power"] + penalty


def _run_search(cfg, env, bounds, fixed_shape, l_min, l_max, elev0_deg,
                 report_cycles, tension_cap, seed, maxiter, popsize, verbose,
                 label):
    if tension_cap is None:
        rho_ref = air_density(0.5*(l_min+l_max)*np.sin(np.radians(elev0_deg)),
                               cfg.site_elevation_msl)
        tension_cap = 3.0 * aero_resultant_force(rho_ref, V_REL_MAX,
                                                   cfg.area, cfg.CL, cfg.CD)
    if verbose:
        print(f"Searching for the {label} pattern "
              f"(l_min={l_min:.0f}m, l_max={l_max:.0f}m)...")

    # differential_evolution is a global optimizer: it evolves a whole
    # population of candidate parameter vectors instead of climbing a
    # gradient from one starting guess, which suits an objective built
    # out of a physics simulation (no clean derivative to hand a
    # gradient-based method).
    args = (cfg, env, l_min, l_max, elev0_deg, tension_cap, fixed_shape)
    res = differential_evolution(
        _objective, bounds, args=args,
        seed=seed, maxiter=maxiter, popsize=popsize,
        mutation=(0.4, 1.2), recombination=0.7, tol=1e-7,
    )
    best_x = res.x

    if fixed_shape is not None:
        best_params = list(best_x) + [fixed_shape["n_weaves"],
                                       fixed_shape["az_amp_deg"],
                                       fixed_shape["elev_amp_deg"]]
    else:
        best_params = list(best_x)

    if verbose:
        print(f"Done. Best found: {-res.fun/1000.0:.2f} kW")

    best_sim = simulate_pumping_cycle(cfg, env, best_params, l_min, l_max,
                                       elev0_deg, n_cycles=report_cycles)
    return best_params, best_sim, res


def optimize_fixed_figure8(cfg, env, l_min=300.0, l_max=650.0,
                            elev0_deg=28.0, report_cycles=3,
                            tension_cap=None, seed=0, maxiter=30,
                            popsize=15, verbose=True):
    """
    The "company standard" flight pattern: a simple, symmetric figure-8
    of fixed shape (FIXED_SHAPE). Only the winch settings (reel speeds,
    depower fraction) are tuned -- the flight-path SHAPE itself is not
    searched, matching how most commercial systems fly a standardized
    weave and mainly tune control-system/winch parameters per site.
    """
    return _run_search(cfg, env, WINCH_ONLY_BOUNDS, FIXED_SHAPE, l_min, l_max,
                        elev0_deg, report_cycles, tension_cap, seed, maxiter,
                        popsize, verbose, "standard figure-8 (winch-only)")


def optimize_physics_optimal(cfg, env, l_min=300.0, l_max=650.0,
                              elev0_deg=28.0, report_cycles=3,
                              tension_cap=None, seed=0, maxiter=40,
                              popsize=20, verbose=True):
    """
    The "pure physics and math" search: nothing about the flight-path
    shape is assumed. Weave count, azimuth amplitude, elevation
    amplitude, and all winch settings are searched together to find
    whatever combination truly maximizes net average power for this
    environment and kite.
    """
    return _run_search(cfg, env, FULL_BOUNDS, None, l_min, l_max, elev0_deg,
                        report_cycles, tension_cap, seed, maxiter, popsize,
                        verbose, "physics-optimal (fully free-form)")
