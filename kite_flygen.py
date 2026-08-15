"""
kite_flygen.py

Fly-Gen power architecture: the kite carries small onboard turbines and
generates power directly from the apparent wind rushing past it, on a
fixed-length tether with no winch. This was the architecture Google's
Makani project flew before shutting down in 2020.

The kite flies a continuous crosswind figure-8, steered by a cascaded
control loop:

  1. Outer guidance loop (`pursuit_bank_command`): proportional-
     navigation-style steering toward a moving target point on a
     reference figure-8 -- the kite banks toward whatever heading
     closes the angle to the target, so it self-corrects from gusts
     or an off-nominal start instead of following a pre-planned
     schedule.
  2. Inner attitude loop (`steering_command`): the outer loop's output
     is a *desired* bank angle, not a value applied directly -- the
     kite's actual bank is a dynamic state now (kite_dynamics.py is
     full 6DOF), so a proportional-derivative controller converts
     bank error into a steering-line deflection, same as a real
     autopilot commanding a control surface rather than teleporting
     the airframe to an attitude.

Running this file directly flies the DEFAULT_PARAMS figure-8 (no
search) and plots it. For the flight-path/control search that
maximizes energy, see kite_path_optimizer.py.
"""

import argparse
import numpy as np

from kite_dynamics import (
    KiteConfig, WindEnvironment, kite_forces_moments, rk4_step,
    initial_attitude,
)


# ─────────────────────────────────────────────
#  Reference flight path (figure-8 in azimuth/elevation space)
# ─────────────────────────────────────────────

V_NOMINAL = 15.0   # m/s -- rough reference crosswind speed for path pacing

# params = [az_amp, f, phi, elev_amp, gen_load]
#
# Bounds are set by the steering loop's roll bandwidth, not just
# structural limits: the guidance/attitude cascade (heading error ->
# steering deflection -> roll dynamics -> realized turn) has real lag,
# unlike an idealized instant-bank kinematic model, so a figure-8
# demanding direction reversals faster than the kite can actually roll
# into just loses tracking and slowly sinks. kite_dynamics.CL_DELTA/
# CL_P set how much roll authority is available; these ranges were
# swept empirically against that authority to find what's flyable.
PARAM_BOUNDS = [
    (0.05, 0.47),         # az_amp    -- ~3 to 27 deg
    (0.005, 0.045),        # f         -- Hz
    (0.0, 2 * np.pi),     # phi
    (0.02, 0.19),           # elev_amp  -- ~1 to 11 deg
    (0.0, 0.5),            # gen_load  -- fraction of CD_gen_max engaged
]

# A reasonable, un-optimized figure-8 -- moderate weave, moderate
# generator loading, with margin below the trackable envelope above.
# This is what `python kite_flygen.py` flies by default;
# kite_path_optimizer.py searches from here for something better.
#
# Below ~10-11 m/s reference wind, this fixed-shape pattern used to
# lose the figure-8 and sink outright. Root cause (found by direct
# instrumentation, not guessed): sustained hard-over steering excites
# a pitch/angle-of-attack oscillation that passive pitch stiffness
# can't damp out fast enough at low wind, since that stiffness scales
# with wind speed^2 while the kite's pitch inertia doesn't. Gain-
# scheduling the roll loop and slowing the figure-8's turn rate alone
# both measured zero effect, because neither touches the pitch axis.
# Fixed with an active pitch/angle-of-attack control loop (see
# pitch_command below) plus scaling the figure-8's own aggressiveness
# down at low wind (WIND_SHAPE_FLOOR below) -- both needed together;
# pitch control alone stabilizes alpha but doesn't restore enough
# heading-tracking authority to hold the pattern by itself.
DEFAULT_PARAMS = [np.radians(25.0), 0.03, 0.0, np.radians(10.0), 0.35]

WIND_REF_SHAPE   = 12.0   # m/s -- condition DEFAULT_PARAMS' shape was tuned at
WIND_SHAPE_POWER = 6.0    # steep, empirically swept (see below)
WIND_SHAPE_FLOOR = 0.06   # never shrink the figure-8 below this fraction of
                          # nominal, however low the wind


def wind_shape_scale(wind_speed_ref):
    """
    Shrinks az_amp/f/elev_amp together at low wind, where available
    control authority (~ dynamic pressure ~ wind speed^2) can no
    longer execute the reference-condition figure-8 in the same real
    time. Steeply nonlinear (power 6, not 2): swept empirically against
    the active pitch controller above -- power=2 and power=3 both left
    a crash "gap" around 9-10 m/s that shifted but didn't close as the
    exponent was tuned (a real resonance-like dead zone, not a smooth
    function of scale alone), and only power=6 cleared the *entire*
    4-22 m/s range with no gaps in a direct sweep. Treat this as a
    fitted curve, not a derived one.
    """
    return float(np.clip((wind_speed_ref / WIND_REF_SHAPE) ** WIND_SHAPE_POWER,
                          WIND_SHAPE_FLOOR, 1.0))


def reference_point(s, params, elev0, l):
    """
    Moving target point on a parametric figure-8 in azimuth/elevation
    space -- the flight-path shape the guidance law below chases.

    `s` is a path-phase variable, not wall-clock time: it's advanced by
    how far the kite has actually traveled (scaled by V_NOMINAL), not
    by dt directly. That way the target waits for the kite if it's
    flying slower than nominal, instead of racing ahead on a fixed
    schedule and asking for a turn tighter than the kite can fly.

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


# ─────────────────────────────────────────────
#  Guidance / steering control
# ─────────────────────────────────────────────

K_HEADING = 4.0   # steering deflection per rad of heading error
K_DAMP    = 1.2   # steering deflection per rad/s of roll rate (anti-overshoot).
                   # Tuned together with kite_dynamics.CL_DELTA/CL_P: raising
                   # roll authority without raising damping just makes the
                   # loop reach saturation faster without actually tracking
                   # tighter figure-8s any better. Gain-scheduling this by
                   # apparent airspeed was tried and measured to have zero
                   # effect on the low-wind operating envelope (see
                   # DEFAULT_PARAMS' comment) -- delta_steer is already
                   # pinned at its clip limit at low wind, and you can't
                   # un-saturate an actuator already at its limit by
                   # raising the gain that feeds it.


def heading_error(pos, vel, target_pos):
    """
    Proportional-navigation-style guidance: how far the kite's current
    heading is from pointing at a moving target point on the reference
    figure-8, measured in the tangent plane of the tether sphere. Not
    a schedule indexed by wall-clock time -- if the kite ever drifts
    off the intended path (a gust, a slightly-off starting velocity),
    the error just shows up here and gets steered out, the same way a
    driver corrects for wind by watching the road instead of following
    a pre-planned steering sequence.
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
    return np.arctan2(sin_err, cos_err)


def steering_command(heading_err, roll_rate):
    """
    Steering deflection directly proportional to heading error, damped
    by roll rate. This is a single rate-command loop rather than a
    cascade through an explicit "desired bank angle" -- the kite's
    lift direction is tied to its actual body attitude now (see
    kite_dynamics.kite_forces_moments), so a big heading error just
    means "steer hard", and roll-rate damping keeps that from
    overshooting into a sustained roll once the heading starts
    closing, the same role rate feedback plays in a real turn
    coordinator.
    """
    return float(np.clip(K_HEADING * heading_err - K_DAMP * roll_rate, -1.0, 1.0))


# ─────────────────────────────────────────────
#  Active pitch / angle-of-attack control
# ─────────────────────────────────────────────
#
# Below a wind speed threshold, sustained hard-over steering excites a
# pitch oscillation that passive pitch stiffness (kite_dynamics.CM_ALPHA)
# can't damp out fast enough, because that stiffness scales with wind
# speed^2 while the kite's pitch inertia doesn't -- root-caused by
# direct instrumentation, not guessed (see project notes). Gain-
# scheduling the roll loop and slowing the figure-8's demanded turn
# rate were both tried first and both measured to have zero effect,
# because neither touches the pitch axis at all. This closes a real,
# independent feedback loop on alpha instead: the same heading-error ->
# steering-deflection idea above, but for pitch -> CM_PITCH_CTRL.

ALPHA_TRIM = np.radians(12.0)   # target angle of attack -- healthy lift,
                                 # comfortable margin below ALPHA_STALL (18 deg)
KP_ALPHA   = 0.5   # gains quoted at VR_REF_PITCH; see normalization below
KD_ALPHA   = 0.2
VR_REF_PITCH = 25.0   # m/s -- apparent airspeed KP_ALPHA/KD_ALPHA are tuned at

# The closed-loop pitch moment is q_dyn*S*c*CM_PITCH_CTRL*(gain*error),
# and q_dyn ~ Vr^2 -- so a *fixed* gain that's safely conservative at
# one airspeed becomes numerically too stiff for explicit RK4 at high
# Vr (confirmed: KP_ALPHA=3.0 was fine at 12 m/s's Vr but crashed 15+
# m/s outright) and too weak to matter at low Vr. Normalizing the gain
# by (VR_REF_PITCH/Vr)^2 keeps the *closed-loop stiffness* roughly
# constant across the flight envelope instead of the raw gain --
# standard dynamic-pressure gain scheduling, and unlike the roll axis
# (see steering_command's comment) this one isn't fighting a saturated
# actuator, so scheduling it actually changes the achieved response.
PITCH_NORM_MIN, PITCH_NORM_MAX = 0.15, 3.0


def pitch_command(alpha, pitch_rate, Vr):
    """Active pitch-trim control: drives alpha toward ALPHA_TRIM and
    damps pitch rate, the same P-D structure as steering_command but
    acting on kite_dynamics.CM_PITCH_CTRL instead of CL_DELTA, gain-
    scheduled by dynamic pressure (see module comment above)."""
    norm = np.clip((VR_REF_PITCH / max(Vr, 5.0))**2, PITCH_NORM_MIN, PITCH_NORM_MAX)
    err = ALPHA_TRIM - alpha
    return float(np.clip(norm * (KP_ALPHA * err - KD_ALPHA * pitch_rate), -1.0, 1.0))


# ─────────────────────────────────────────────
#  Simulation
# ─────────────────────────────────────────────

def simulate(params, cfg, env, t_end, dt=0.02, elev0_deg=30.0, warmup=5.0):
    """
    Fly the kite under the cascaded guidance/attitude controller and
    return time histories plus summary totals (energy, stall/crash
    flags). `params` first, matching the generic `(params, *sim_args)`
    signature kite_path_optimizer.py expects.
    """
    elev0 = np.radians(elev0_deg)
    # Shrink the figure-8's own aggressiveness at low wind -- see
    # wind_shape_scale() above.
    az_amp, f, phi, elev_amp, gen_load_raw = params
    wshape = wind_shape_scale(env.wind_speed_ref)
    flight_params = [az_amp * wshape, f * wshape, phi, elev_amp * wshape]

    # Start the kite on the reference path, heading along its tangent
    # there, so the guidance loop begins close to where it wants to be
    # instead of having to correct a large initial error first.
    ds = 1e-4
    p0 = reference_point(0.0, flight_params, elev0, cfg.tether_len)
    p1 = reference_point(ds, flight_params, elev0, cfg.tether_len)
    r_hat0 = p0 / np.linalg.norm(p0)
    tan_dir = p1 - p0
    tan_dir -= np.dot(tan_dir, r_hat0) * r_hat0
    tan_norm = np.linalg.norm(tan_dir)
    tan_dir = tan_dir / tan_norm if tan_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
    pos = p0
    vel = tan_dir * V_NOMINAL
    gen_load = float(np.clip(gen_load_raw, 0.0, 1.0))

    wind0 = env.wind_at(pos[2])
    v_rel_hat0 = (wind0 - vel) / np.linalg.norm(wind0 - vel)
    quat = initial_attitude(pos, v_rel_hat0, bank_angle=0.0)
    omega = np.zeros(3)

    steps = int(t_end / dt)
    t_hist = np.zeros(steps)
    pos_hist = np.zeros((steps, 3))
    T_hist = np.zeros(steps)
    P_hist = np.zeros(steps)
    vrel_hist = np.zeros(steps)
    alpha_hist = np.zeros(steps)

    s = 0.0   # path-phase progress, advanced by actual speed, not by t
    for i in range(steps):
        t = i * dt
        wind = env.wind_at(pos[2])
        target = reference_point(s, flight_params, elev0, np.linalg.norm(pos))
        he = heading_error(pos, vel, target)
        delta_steer = steering_command(he, omega[0])

        # Logging pass at the state used to derive delta_steer above
        # (T/P/alpha don't depend on delta_steer, only the moments do).
        F, M, T, P, Vr, alpha, lift_hat = kite_forces_moments(
            pos, vel, quat, omega, wind, cfg, delta_steer=0.0, gen_load=gen_load)
        delta_pitch = pitch_command(alpha, omega[1], Vr)

        t_hist[i] = t
        pos_hist[i] = pos
        T_hist[i] = T
        P_hist[i] = P
        vrel_hist[i] = Vr
        alpha_hist[i] = alpha

        r_hat = pos / np.linalg.norm(pos)
        v_tan_mag = np.linalg.norm(vel - np.dot(vel, r_hat) * r_hat)
        s += dt * v_tan_mag / V_NOMINAL

        pos, vel, quat, omega, T, P, Vr, alpha, lift_hat = rk4_step(
            pos, vel, quat, omega, wind, cfg, delta_steer, dt, gen_load=gen_load,
            delta_pitch=delta_pitch)

    finite = np.all(np.isfinite(pos_hist)) and np.all(np.isfinite(vrel_hist))
    mask = t_hist >= warmup
    t_eval = t_hist[mask]
    duration = t_eval[-1] - t_eval[0] if len(t_eval) > 1 else 0.0
    energy_J = np.trapezoid(P_hist[mask], t_eval) if duration > 0 else 0.0
    mean_power = energy_J / duration if duration > 0 else 0.0

    stalled = (not finite) or bool(np.any(vrel_hist[mask] < 3.0))
    crashed = (not finite) or bool(np.any(pos_hist[mask, 2] < 5.0))

    return dict(
        t=t_hist, pos=pos_hist, T=T_hist, P=P_hist, vrel=vrel_hist,
        alpha=alpha_hist, energy_J=energy_J,
        mean_power=mean_power,
        peak_power=P_hist[mask].max() if duration > 0 else 0.0,
        peak_tension=T_hist[mask].max() if duration > 0 else 0.0,
        duration=duration, stalled=stalled,
        crashed=crashed, phase=np.array(["out"] * steps),
        l=np.linalg.norm(pos_hist, axis=1), params=list(params),
    )


# ─────────────────────────────────────────────
#  CLI -- run the default figure-8 and plot it
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wind-speed", type=float, default=12.0)
    p.add_argument("--shear", type=float, default=0.14)
    p.add_argument("--mass", type=float, default=50.0)
    p.add_argument("--area", type=float, default=20.0)
    p.add_argument("--tether-len", type=float, default=800.0)
    p.add_argument("--elev0", type=float, default=30.0)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--output", type=str, default="kite_flygen.png")
    return p.parse_args()


def main():
    args = _parse_args()
    from kite_path_optimizer import plot_flight, print_summary

    env = WindEnvironment(wind_speed_ref=args.wind_speed, shear_exp=args.shear)
    cfg = KiteConfig(mass=args.mass, area=args.area, tether_len=args.tether_len)

    print()
    print(f"Flying the default Fly-Gen figure-8 for {args.duration:.0f}s...")
    sim = simulate(DEFAULT_PARAMS, cfg, env, args.duration, elev0_deg=args.elev0)

    print_summary(sim, "Fly-Gen (default figure-8)", DEFAULT_PARAMS)
    out = plot_flight(sim, "Fly-Gen -- default figure-8", args.output, color="#E8664F")
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
