"""
kite_dynamics.py

Physics engine for a tethered kite generating power from crosswind flight.

State: position (x, y, z) in meters from the ground anchor
(X=downwind, Y=crosswind, Z=up), velocity in m/s.

Forces: lift (perpendicular to apparent wind), drag (parasitic +
generator load), gravity, and tether tension.

Power comes from an onboard generator ("Fly-Gen" style, fixed tether
length, no winch): small rotors add drag to the airframe, and that
drag force times the apparent wind speed times efficiency is the
extracted power - same idea as a normal wind turbine, just facing a
much faster apparent wind because the kite is moving fast crosswind.

    F_gen = q * CD_gen * A
    P     = F_gen * v_rel * eta_gen

Loading the generator harder raises CD_gen and instantaneous power,
but also slows the kite down, which lowers v_rel - so there's a real
tradeoff, not a free lunch. gen_load (0-1) is the control input.
"""

import numpy as np


G         = 9.81
RHO0      = 1.225     # sea level air density, kg/m3
H_RHO     = 8500.0    # atmosphere scale height, m
ETA_GEN   = 0.85      # generator efficiency
ETA_MESH  = 0.25      # water collection mesh efficiency


class KiteConfig:
    """Fixed physical properties of the kite."""

    def __init__(
        self,
        mass       = 50.0,
        area       = 20.0,
        CL         = 1.0,
        CD         = 0.15,
        tether_len = 800.0,
        CD_gen_max = 0.5,
        site_elevation_msl = 0.0,
    ):
        self.mass       = mass
        self.area       = area
        self.CL         = CL
        self.CD         = CD
        self.tether_len = tether_len
        self.CD_gen_max = CD_gen_max
        self.site_elevation_msl = site_elevation_msl
        self.glide_ratio = CL / CD

        print(f"KiteConfig initialized:")
        print(f"  mass={mass}kg, area={area}m², CL={CL}, CD={CD}")
        print(f"  glide ratio CL/CD = {self.glide_ratio:.1f}")
        print(f"  tether length = {tether_len}m")
        print(f"  onboard generator: CD_gen_max = {CD_gen_max}")
        print(f"  site elevation = {site_elevation_msl:.0f}m MSL")
        print(f"  stall check: min v_rel for lift > weight = "
              f"{np.sqrt(2*mass*G / (RHO0*CL*area)):.1f} m/s")

    def __repr__(self):
        return (f"KiteConfig(mass={self.mass}kg, area={self.area}m², "
                f"CL={self.CL}, CD={self.CD}, l={self.tether_len}m, "
                f"CD_gen_max={self.CD_gen_max}, "
                f"site_elevation_msl={self.site_elevation_msl:.0f}m)")


def air_density(altitude_agl, site_elevation_msl=0.0):
    """Exponential atmosphere model, adjusted for site elevation above
    sea level (a high-altitude site has thinner air than a coastal one
    at the same height above the ground)."""
    total_altitude = altitude_agl + site_elevation_msl
    return RHO0 * np.exp(-total_altitude / H_RHO)


def compute_lift_hat(pos, v_rel_hat, bank_angle=0.0):
    """
    Lift unit vector, perpendicular to apparent wind.

    The kite steers by banking - rotating the lift vector around the
    tether axis. bank_angle=0 gives max upward lift; positive banks
    to the right.
    """
    tether_hat = -pos / np.linalg.norm(pos)

    span = np.cross(tether_hat, v_rel_hat)
    span_norm = np.linalg.norm(span)
    if span_norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    span_hat = span / span_norm

    lift_hat_0 = np.cross(v_rel_hat, span_hat)
    lift_hat_0 /= np.linalg.norm(lift_hat_0)
    if abs(bank_angle) < 1e-6:
        return lift_hat_0

    # Rotate around the tether axis by bank_angle (Rodrigues formula)
    K = -tether_hat
    cos_b, sin_b = np.cos(bank_angle), np.sin(bank_angle)
    lift_hat = (lift_hat_0 * cos_b
                + np.cross(K, lift_hat_0) * sin_b
                + K * np.dot(K, lift_hat_0) * (1 - cos_b))
    return lift_hat / np.linalg.norm(lift_hat)


def kite_forces(pos, vel, wind_vec, cfg, bank_angle=0.0, gen_load=0.0,
                 CL_scale=1.0):
    """
    Total force, tether tension, and instantaneous power at one instant.

    gen_load: onboard generator loading, 0-1.
    CL_scale: depower control, 0-1 (1 = full lift, lower = kite
    pitched to reduce lift and tension, used during retraction).

    Returns (F_total, T, P_inst, v_rel).
    """
    altitude = pos[2]
    rho      = air_density(altitude, cfg.site_elevation_msl)
    gen_load = min(max(gen_load, 0.0), 1.0)
    l_current = np.linalg.norm(pos)

    v_rel     = wind_vec - vel
    v_rel_mag = np.linalg.norm(v_rel)
    if v_rel_mag < 1e-3:
        F_grav = np.array([0.0, 0.0, -cfg.mass * G])
        return F_grav, 0.0, 0.0, v_rel
    v_rel_hat = v_rel / v_rel_mag

    q = 0.5 * rho * v_rel_mag**2

    # Drag: parasitic airframe drag + generator loading
    CD_gen  = gen_load * cfg.CD_gen_max
    F_drag_parasitic = q * cfg.CD * cfg.area * v_rel_hat
    F_drag_gen       = q * CD_gen * cfg.area * v_rel_hat
    F_drag           = F_drag_parasitic + F_drag_gen

    F_lift = q * cfg.CL * CL_scale * cfg.area * compute_lift_hat(pos, v_rel_hat, bank_angle)
    F_grav = np.array([0.0, 0.0, -cfg.mass * G])

    # Tether tension: centripetal term (needed to keep the kite moving
    # in a circle) plus the outward-radial component of everything else.
    tether_hat  = -pos / l_current
    outward_hat = pos / l_current
    F_aero_grav = F_lift + F_drag + F_grav
    T_aero      = np.dot(F_aero_grav, outward_hat)

    vel_tangential = vel - np.dot(vel, tether_hat) * tether_hat
    v_tan_mag      = np.linalg.norm(vel_tangential)
    T_centripetal  = cfg.mass * v_tan_mag**2 / l_current
    T              = T_centripetal + max(0.0, T_aero)
    F_tether       = T * tether_hat

    # Steering: bank angle sets turn curvature directly, same as a
    # coordinated turn in an aircraft (turn accel = g * tan(bank)).
    if v_tan_mag > 1e-3:
        vel_tan_hat = vel_tangential / v_tan_mag
        turn_dir    = np.cross(tether_hat, vel_tan_hat)
        turn_dir   /= np.linalg.norm(turn_dir)
        a_turn      = G * np.tan(bank_angle)
        F_steer     = cfg.mass * a_turn * turn_dir
    else:
        F_steer = np.zeros(3)

    F_total = F_lift + F_drag + F_grav + F_tether + F_steer

    F_gen_mag = CD_gen * q * cfg.area
    P_inst    = F_gen_mag * v_rel_mag * ETA_GEN

    return F_total, T, P_inst, v_rel


def water_collection_rate(v_rel_mag, LWC, cfg):
    """Water collected per second by a mesh sweeping through moist air."""
    return ETA_MESH * LWC * cfg.area * v_rel_mag


def rk4_step(pos, vel, wind_vec, cfg, bank_angle, dt, gen_load=0.0):
    """One RK4 integration step, fixed tether length."""

    def derivatives(p, v):
        F, T, P, vr = kite_forces(p, v, wind_vec, cfg, bank_angle, gen_load)
        return v, F / cfg.mass, T, P, vr

    dp1, da1, T, P, vr = derivatives(pos, vel)
    dp2, da2, _, _, _ = derivatives(pos + 0.5*dt*dp1, vel + 0.5*dt*da1)
    dp3, da3, _, _, _ = derivatives(pos + 0.5*dt*dp2, vel + 0.5*dt*da2)
    dp4, da4, _, _, _ = derivatives(pos + dt*dp3, vel + dt*da3)

    pos_new = pos + (dt/6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
    vel_new = vel + (dt/6.0) * (da1 + 2*da2 + 2*da3 + da4)

    # Snap back onto the tether sphere (RK4 drifts slightly off it)
    pos_new = pos_new * (cfg.tether_len / np.linalg.norm(pos_new))
    r_hat   = pos_new / np.linalg.norm(pos_new)
    vel_new = vel_new - np.dot(vel_new, r_hat) * r_hat

    return pos_new, vel_new, T, P, vr


def rk4_step_variable_tether(pos, vel, wind_vec, cfg, bank_angle, dt, l_dot,
                              CL_scale=1.0):
    """
    RK4 step with a changing tether length, for pumping-cycle mode:
    a ground winch reels the tether out (l_dot > 0, generating power
    under tension) or in (l_dot < 0, retracting, usually depowered).

    l_dot is treated as a directly-commanded winch speed.
    """

    def derivatives(p, v):
        F, T, P, vr = kite_forces(p, v, wind_vec, cfg, bank_angle,
                                   gen_load=0.0, CL_scale=CL_scale)
        return v, F / cfg.mass, T, P, vr

    l_old = np.linalg.norm(pos)

    dp1, da1, T, P, vr = derivatives(pos, vel)
    dp2, da2, _, _, _ = derivatives(pos + 0.5*dt*dp1, vel + 0.5*dt*da1)
    dp3, da3, _, _, _ = derivatives(pos + 0.5*dt*dp2, vel + 0.5*dt*da2)
    dp4, da4, _, _, _ = derivatives(pos + dt*dp3, vel + dt*da3)

    pos_new = pos + (dt/6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
    vel_new = vel + (dt/6.0) * (da1 + 2*da2 + 2*da3 + da4)

    # Force the tether to the commanded length, radial velocity = l_dot
    l_new   = max(l_old + l_dot*dt, 1.0)
    r_hat   = pos_new / np.linalg.norm(pos_new)
    pos_new = r_hat * l_new
    vel_tan = vel_new - np.dot(vel_new, r_hat) * r_hat
    vel_new = vel_tan + l_dot * r_hat

    return pos_new, vel_new, T, P, vr


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cfg = KiteConfig(mass=50.0, area=20.0, CL=1.0, CD=0.15, tether_len=800.0)
    print()

    print("=" * 55)
    print("Test 1: Static force balance at operating point")
    print("=" * 55)

    elev = np.radians(30)
    pos  = np.array([cfg.tether_len * np.cos(elev), 0.0, cfg.tether_len * np.sin(elev)])
    vel  = np.array([0.0, 20.0, 0.0])
    wind = np.array([13.3, 0.0, 0.0])

    F, T, P, vr = kite_forces(pos, vel, wind, cfg, bank_angle=0.0, gen_load=0.3)
    v_rel_mag = np.linalg.norm(vr)
    rho_alt   = air_density(pos[2])
    v_tan     = vel - np.dot(vel, pos/np.linalg.norm(pos)) * (pos/np.linalg.norm(pos))

    print(f"  Altitude:          {pos[2]:.0f} m")
    print(f"  Apparent wind:     {v_rel_mag:.1f} m/s")
    print(f"  Lift force:        {0.5*rho_alt*v_rel_mag**2*cfg.CL*cfg.area:.0f} N")
    print(f"  Tangential speed:  {np.linalg.norm(v_tan):.1f} m/s")
    print(f"  Tether tension:    {T:.1f} N")
    print(f"  Instant power:     {P/1000:.2f} kW  (gen_load=0.3)")
    print(f"  Weight:            {cfg.mass*G:.0f} N")
    print(f"  Lift/Weight ratio: {0.5*rho_alt*v_rel_mag**2*cfg.CL*cfg.area / (cfg.mass*G):.1f}")

    print()
    print("=" * 55)
    print("Test 2: Water collection - Atacama camanchaca fog")
    print("=" * 55)

    LWC      = 0.20e-3
    m_dot    = water_collection_rate(v_rel_mag, LWC, cfg)
    per_hour = m_dot * 3600
    per_day  = m_dot * 3600 * 10

    print(f"  LWC:               {LWC*1e3:.2f} g/m³")
    print(f"  Collection rate:   {m_dot*1000:.3f} g/s")
    print(f"  Per hour:          {per_hour:.2f} kg/hr")
    print(f"  Per fog day:       {per_day:.1f} liters/day")

    print()
    print("=" * 55)
    print("Test 3: Free flight - 30s, no controller")
    print("=" * 55)

    pos = np.array([cfg.tether_len * np.cos(np.radians(30)), 0.0,
                     cfg.tether_len * np.sin(np.radians(30))])
    vel = np.array([0.0, 15.0, 5.0])
    wind = np.array([13.3, 0.0, 0.0])
    dt, t_end = 0.05, 30.0
    steps = int(t_end / dt)

    t_hist   = np.zeros(steps)
    pos_hist = np.zeros((steps, 3))
    T_hist   = np.zeros(steps)
    P_hist   = np.zeros(steps)

    for i in range(steps):
        t_hist[i]   = i * dt
        pos_hist[i] = pos
        F, T, P, vr = kite_forces(pos, vel, wind, cfg, gen_load=0.3)
        T_hist[i]   = T
        P_hist[i]   = P / 1000
        pos, vel, T, P, vr = rk4_step(pos, vel, wind, cfg, 0.0, dt, gen_load=0.3)

    tether_lengths = np.linalg.norm(pos_hist, axis=1)
    print(f"  Tether length - min: {tether_lengths.min():.4f}m, "
          f"max: {tether_lengths.max():.4f}m  (should be {cfg.tether_len}m)")
    print(f"  Peak tension:    {T_hist.max():.0f} N")
    print(f"  Peak power:      {P_hist.max():.2f} kW")
    print(f"  Mean power:      {P_hist.mean():.2f} kW")

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(131)
    ax1.plot(pos_hist[:, 1], pos_hist[:, 2])
    ax1.set_xlabel('Y crosswind (m)')
    ax1.set_ylabel('Z altitude (m)')
    ax1.set_title('Kite path (front view)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(132)
    ax2.plot(t_hist, T_hist)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Tension (N)')
    ax2.set_title('Tether tension')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(133)
    ax3.plot(t_hist, P_hist, color='#f0a020')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Power (kW)')
    ax3.set_title('Instantaneous power')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_free_flight.png', dpi=150)
    plt.show()
    print("  Plot saved: test_free_flight.png")
