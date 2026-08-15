"""
kite_groundgen.py

Ground-Gen (pumping-cycle) power architecture: the kite flies a
crosswind figure-8 while a ground-station winch reels the tether out
under tension, spinning a generator (the power phase). Once fully
extended, the kite is pitched to a low-lift "depowered" state and
reeled back in fast and cheap (the retraction phase), and the cycle
repeats. This is the architecture most commercial AWE companies
(Kitepower, TwingTec, SkySails Power, Kitemill) fly.

The power (reel-out) phase drives the same full 6DOF rigid-body
dynamics as kite_flygen.py, with a varying tether length
(kite_dynamics.rk4_step_variable_tether) instead of a fixed one --
that's where the flight-path optimization value is, so it gets full
fidelity. Power there comes from the winch (T * reel_speed), not an
onboard generator, so gen_load is always 0.

The retraction (reel-in) phase uses the quasi-steady closed-form
crosswind kite theory (Loyd, 1980) instead: real systems fly this
phase deliberately gentle (depowered, minimal crosswind motion), and
actively steering a full 6DOF retraction toward a target just keeps
tangential speed -- and so centripetal tether tension -- as high as
during reel-out, defeating the point of depowering. See the comment
above the retraction block in `simulate()` for the full reasoning.

Running this file directly flies the DEFAULT_PARAMS pumping cycle (no
search) and plots it. For the flight-path/winch search that maximizes
energy, see kite_path_optimizer.py.
"""

import argparse
import numpy as np

from kite_dynamics import (
    KiteConfig, WindEnvironment, kite_forces_moments,
    rk4_step_variable_tether, initial_attitude,
    air_density as kd_air_density, ETA_GEN,
)
from kite_flygen import (
    reference_point, heading_error, steering_command, pitch_command,
    wind_shape_scale, V_NOMINAL,
)


# ── Quasi-steady crosswind kite theory (Loyd, 1980), retraction only ──
CROSSWIND_EFFICIENCY = 0.6   # fraction of Loyd's theoretical-max crosswind
                              # speed real systems achieve
V_REL_MAX = 55.0              # m/s -- realistic structural/aero speed cap


def _crosswind_airspeed(v_wind, CL, CD):
    v_ideal = max(v_wind, 0.0) * (CL / CD) * CROSSWIND_EFFICIENCY
    return min(v_ideal, V_REL_MAX)


def _aero_resultant_force(rho, v_rel, area, CL, CD):
    q = 0.5 * rho * v_rel**2
    return q * area * CL * np.sqrt(1.0 + (CD / CL)**2)


# params = [reel_out_speed, reel_in_speed, depower_frac, az_amp, f, elev_amp]
#
# Same steering-bandwidth reasoning as kite_flygen.PARAM_BOUNDS: the
# figure-8 shape has to stay inside what the roll control loop can
# actually track, or the kite loses the pattern and sags.
PARAM_BOUNDS = [
    (1.5, 5.0),           # reel_out_speed  (m/s) -- floor keeps worst-case
                           # reel-out duration (and so search cost) bounded
    (3.0, 15.0),          # reel_in_speed   (m/s, magnitude)
    (0.1, 0.6),            # depower_frac    (CL_scale during retraction)
    (0.05, 0.47),           # az_amp          (~3 to 27 deg)
    (0.005, 0.032),          # f               (Hz) -- slightly lower ceiling
                              # than kite_flygen's: a lengthening tether
                              # during reel-out makes the same weave shape
                              # marginally harder to track, empirically.
    (0.02, 0.19),             # elev_amp        (~1 to 11 deg)
]

# A reasonable, un-optimized pumping cycle -- moderate weave during
# reel-out, brisk reel-in, moderate depower, with margin below the
# trackable envelope above. What `python kite_groundgen.py` flies by
# default.
DEFAULT_PARAMS = [2.5, 8.0, 0.25, np.radians(25.0), 0.03, np.radians(10.0)]


def simulate(params, cfg, env, l_min=300.0, l_max=650.0, elev0_deg=30.0,
             n_cycles=2, dt=0.02, warmup_cycles=0):
    """
    Fly `n_cycles` reel-out/reel-in pumping cycles under the cascaded
    guidance/attitude controller (kite_flygen.heading_error /
    steering_command), full 6DOF dynamics, variable tether length.
    `params` first, matching kite_path_optimizer's generic
    `(params, *sim_args)` signature.
    """
    reel_out_speed, reel_in_speed, depower_frac, az_amp, f, elev_amp = params
    reel_out_speed = max(reel_out_speed, 0.1)
    reel_in_speed = max(reel_in_speed, 0.1)
    elev0 = np.radians(elev0_deg)
    # Shrink the figure-8's own aggressiveness at low wind -- see
    # kite_flygen.wind_shape_scale() -- and apply a flat stability
    # margin on top of that, wind-independent. Found by direct A/B
    # testing: once active pitch control (kite_flygen.pitch_command)
    # was wired in above, DEFAULT_PARAMS' full-amplitude shape started
    # crashing at 12-22 m/s -- reel-out from a growing/shorter tether
    # has less margin against the same pitch/roll excursion than
    # Fly-Gen's fixed-tether flight does with the identical shape and
    # gains (confirmed: the crash persisted with reeling switched off
    # and the starting tether length swept from 300-650m, so it's not
    # about tether length or reel rate specifically). A flat 15% cut to
    # az_amp/f/elev_amp cleared a full 12-22 m/s re-sweep with margin to
    # spare (also checked at +/-5%: 10% was still enough to crash,
    # 20% held); 5% was not enough. Not itself wind-dependent, so it
    # stacks multiplicatively with wind_shape_scale() rather than
    # replacing it.
    GROUNDGEN_SHAPE_MARGIN = 0.85
    wshape = wind_shape_scale(env.wind_speed_ref) * GROUNDGEN_SHAPE_MARGIN
    shape_params = [az_amp * wshape, f * wshape, 0.0, elev_amp * wshape]   # feeds reference_point()

    # Starting state at l_min, on the reference path, heading along its
    # tangent -- (re)established at the start of every cycle, not just
    # the first: a real system re-powers the kite into clean flight at
    # the start of each reel-out stroke, so any attitude transient left
    # over from the previous retraction isn't carried forward.
    ds = 1e-4

    def initial_state():
        p0 = reference_point(0.0, shape_params, elev0, l_min)
        p1 = reference_point(ds, shape_params, elev0, l_min)
        r_hat0 = p0 / np.linalg.norm(p0)
        tan_dir = p1 - p0
        tan_dir -= np.dot(tan_dir, r_hat0) * r_hat0
        tan_norm = np.linalg.norm(tan_dir)
        tan_dir = tan_dir / tan_norm if tan_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
        pos = p0
        vel = tan_dir * V_NOMINAL
        wind0 = env.wind_at(pos[2])
        v_rel_hat0 = (wind0 - vel) / np.linalg.norm(wind0 - vel)
        quat = initial_attitude(pos, v_rel_hat0, bank_angle=0.0)
        omega = np.zeros(3)
        return pos, vel, quat, omega

    pos, vel, quat, omega = initial_state()

    t_hist, l_hist, pos_hist = [], [], []
    T_hist, P_hist, phase_hist, vrel_hist = [], [], [], []

    t = 0.0
    s = 0.0
    for cyc in range(n_cycles):
        pos, vel, quat, omega = initial_state()
        # ── Reel-out (power) phase: fly the figure-8, winch pays out ──
        duration_out = (l_max - l_min) / reel_out_speed
        n_steps_out = max(int(duration_out / dt), 4)
        for i in range(n_steps_out):
            wind = env.wind_at(pos[2])
            l_now = np.linalg.norm(pos)

            target = reference_point(s, shape_params, elev0, l_now)
            he = heading_error(pos, vel, target)
            delta_steer = steering_command(he, omega[0])

            F, M, T, P_aero, Vr, alpha, lift_hat = kite_forces_moments(
                pos, vel, quat, omega, wind, cfg, delta_steer=0.0, gen_load=0.0)
            P = T * reel_out_speed   # winch power, not onboard-generator drag
            delta_pitch = pitch_command(alpha, omega[1], Vr)

            t_hist.append(t); l_hist.append(l_now); pos_hist.append(pos)
            T_hist.append(T); P_hist.append(P); phase_hist.append("out")
            vrel_hist.append(Vr)

            r_hat = pos / l_now
            v_tan_mag = np.linalg.norm(vel - np.dot(vel, r_hat) * r_hat)
            s += dt * v_tan_mag / V_NOMINAL

            pos, vel, quat, omega, T, P_aero, Vr, alpha, lift_hat = rk4_step_variable_tether(
                pos, vel, quat, omega, wind, cfg, delta_steer, dt, reel_out_speed,
                gen_load=0.0, CL_scale=1.0, delta_pitch=delta_pitch)
            t += dt

        # ── Reel-in (retraction) phase: quasi-steady, not full 6DOF ──
        # Real systems fly this phase deliberately gentle -- depowered,
        # a small kinematic wobble back toward center, nowhere near the
        # aggressive crosswind weave of the power phase. Actively
        # steering a full 6DOF retraction (chasing a target, feeding
        # back on heading error) keeps tangential speed -- and so
        # centripetal tether tension -- just as high as during reel-out,
        # defeating the entire point of depowering. Rather than fight
        # that with more control tuning, this phase uses the same
        # closed-form crosswind kite theory (Loyd, 1980) the project's
        # earlier quasi-steady model used: appropriate here since
        # retraction isn't where the flight-path optimization or the
        # interesting physics lives -- that's the power phase above.
        l_mid_in = 0.5 * (l_min + l_max)
        alt_mid_in = l_mid_in * np.sin(elev0)
        v_wind_in = np.linalg.norm(env.wind_at(alt_mid_in))
        rho_in = kd_air_density(alt_mid_in, cfg.site_elevation_msl)
        CL_retract = cfg.CL * depower_frac
        v_rel_in = _crosswind_airspeed(v_wind_in, CL_retract, cfg.CD)
        T_in = _aero_resultant_force(rho_in, v_rel_in, cfg.area, CL_retract, cfg.CD)
        P_in = -T_in * reel_in_speed * ETA_GEN

        duration_in = (l_max - l_min) / reel_in_speed
        n_steps_in = max(int(duration_in / dt), 4)
        az0 = np.arctan2(pos[1], pos[0])   # current azimuth, retraction wobbles around it
        for i in range(n_steps_in):
            frac = i / (n_steps_in - 1)
            l_i = l_max - frac * (l_max - l_min)
            phi = frac * 2 * np.pi
            az = az0 + 0.25 * shape_params[0] * np.sin(phi)
            el = elev0 + 0.25 * shape_params[3] * np.sin(2 * phi)
            pos = l_i * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

            t_hist.append(t); l_hist.append(l_i); pos_hist.append(pos)
            T_hist.append(T_in); P_hist.append(P_in); phase_hist.append("in")
            vrel_hist.append(v_rel_in)
            t += duration_in / n_steps_in

        # The next cycle's reel-out phase re-establishes clean flying
        # state via initial_state() at the top of the loop.
        s = 0.0

    t_hist = np.array(t_hist); l_hist = np.array(l_hist)
    pos_hist = np.array(pos_hist); T_hist = np.array(T_hist)
    P_hist = np.array(P_hist); phase_hist = np.array(phase_hist)
    vrel_hist = np.array(vrel_hist)

    finite = np.all(np.isfinite(pos_hist)) and np.all(np.isfinite(vrel_hist))
    duration = t_hist[-1] - t_hist[0] if len(t_hist) > 1 else 0.0
    energy_J = np.trapezoid(P_hist, t_hist) if duration > 0 else 0.0
    mean_power = energy_J / duration if duration > 0 else 0.0

    stalled = (not finite) or bool(np.any(vrel_hist < 3.0))
    crashed = (not finite) or bool(np.any(pos_hist[:, 2] < 5.0))

    return dict(
        t=t_hist, l=l_hist, pos=pos_hist, T=T_hist, P=P_hist,
        phase=phase_hist, vrel=vrel_hist, energy_J=energy_J,
        mean_power=mean_power,
        peak_power=P_hist.max() if len(P_hist) else 0.0,
        peak_tension=T_hist.max() if len(T_hist) else 0.0,
        duration=duration, stalled=stalled,
        crashed=crashed, params=list(params),
    )


# ─────────────────────────────────────────────
#  CLI -- run the default pumping cycle and plot it
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wind-speed", type=float, default=12.0)
    p.add_argument("--shear", type=float, default=0.14)
    p.add_argument("--mass", type=float, default=50.0)
    p.add_argument("--area", type=float, default=20.0)
    p.add_argument("--l-min", type=float, default=300.0)
    p.add_argument("--l-max", type=float, default=650.0)
    p.add_argument("--elev0", type=float, default=30.0)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--output", type=str, default="kite_groundgen.png")
    return p.parse_args()


def main():
    args = _parse_args()
    from kite_path_optimizer import plot_flight, print_summary

    env = WindEnvironment(wind_speed_ref=args.wind_speed, shear_exp=args.shear)
    cfg = KiteConfig(mass=args.mass, area=args.area, tether_len=args.l_max)

    print()
    print(f"Flying the default ground-gen pumping cycle ({args.cycles} cycles)...")
    sim = simulate(DEFAULT_PARAMS, cfg, env, l_min=args.l_min, l_max=args.l_max,
                    elev0_deg=args.elev0, n_cycles=args.cycles)

    print_summary(sim, "Ground-Gen (default pumping cycle)", DEFAULT_PARAMS)
    out = plot_flight(sim, "Ground-Gen -- default pumping cycle", args.output, color="#f0a020")
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
