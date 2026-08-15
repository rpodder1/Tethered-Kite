"""
kite_dynamics.py

Full 6-degree-of-freedom rigid-body physics engine for a tethered kite
generating power from crosswind flight.

State (13 numbers, carried as four separate arrays rather than one
packed vector for readability): position `pos` (3, world frame,
X=downwind, Y=crosswind, Z=up, meters from the ground anchor),
velocity `vel` (3, world frame, m/s), attitude quaternion `quat`
(4, body->world, scalar-first [w,x,y,z]), and body-frame angular
velocity `omega` (3, rad/s).

Body axes follow the standard aircraft convention: x_b out the nose
(roughly the flight direction), y_b out the right wingtip, z_b down
through the belly (so lift acts along -z_b when wings are level).

Aerodynamics are built from angle of attack (alpha) and sideslip
(beta), computed from the kite's velocity relative to the air
expressed in body axes -- not assumed or commanded. Steering is a
single control input `delta_steer` (-1..1, differential brake/steering
line deflection) that produces a rolling moment (plus a little yaw
coupling); the kite's bank angle is then something that *emerges* from
rotational dynamics (inertia, damping, a rise time, possible overshoot)
rather than being kinematically snapped to a commanded value each
step. Pitch (angle of attack) is likewise a free dynamic state, with
its own restoring stiffness and damping, not clamped.

Power ("Fly-Gen" style, onboard generator, see kite_flygen.py): small
rotors add drag to the airframe, and that drag force times the
apparent wind speed times efficiency is the extracted power -- same
idea as a normal wind turbine, just facing a much faster apparent wind
because the kite is moving fast crosswind.

    F_gen = q * CD_gen * A
    P     = F_gen * v_rel * eta_gen

Loading the generator harder raises CD_gen and instantaneous power,
but also slows the kite down, which lowers v_rel -- so there's a real
tradeoff, not a free lunch. gen_load (0-1) is the control input.
"""

import numpy as np


G         = 9.81
RHO0      = 1.225     # sea level air density, kg/m3
H_RHO     = 8500.0    # atmosphere scale height, m
ETA_GEN   = 0.85      # generator efficiency
OMEGA_MAX = 15.0       # rad/s -- structural/aero rate limit per axis

# ── Aerodynamic stability & control derivatives ──────────────────
# Fixed physics constants (not per-kite tunables) for a low-aspect-
# ratio leading-edge-inflatable power kite. Values are representative
# textbook/flight-test magnitudes for this class of wing, not a fit to
# a specific real kite.
ASPECT_RATIO = 1.8               # b^2/S, typical for an LEI power kite
OSWALD_E     = 0.8                # span efficiency factor (induced drag)
ALPHA_STALL  = np.radians(18.0)   # critical angle of attack

CY_BETA   = -0.30   # side force per rad sideslip
CL_BETA   = -0.05   # roll moment per rad sideslip (dihedral effect, stabilizing)
CL_P      = -1.20   # roll damping (per unit p-hat). Scaled up with CL_DELTA
                     # below -- keeps the roll mode well-damped (not just
                     # fast) so raising steering authority doesn't turn
                     # into an oscillatory/overshooting roll response.
CL_DELTA  =  0.45   # roll moment per unit steering input. Raised from an
                     # earlier, more conservative value: a low-bandwidth
                     # roll response couldn't track figure-8s tight or
                     # fast enough to compete with commercial-scale
                     # weave rates, forcing very gentle default flight
                     # paths. A real kite's steering-line throw is a
                     # design choice, not a fixed constant -- this
                     # represents a kite built for snappier steering.
CM0       =  0.05   # pitching moment at alpha=0 (camber)
CM_ALPHA  = -0.30   # pitch stiffness (restoring toward trim, stabilizing)
CM_Q      = -2.0    # pitch damping (per unit q-hat). Kept small relative
                     # to typical rigid-aircraft values: combined with a
                     # kite's low pitch inertia, a larger value makes the
                     # pitch mode numerically stiff enough that explicit
                     # RK4 at the dt used elsewhere in this project can't
                     # resolve it (the fast damping root leaves RK4's
                     # stability region and the pitch axis diverges).
CN_BETA   =  0.06   # weathervane / yaw stability per rad sideslip
CN_R      = -0.30   # yaw damping (per unit r-hat). Raised alongside CL_P --
                     # faster roll response couples into faster yaw rates
                     # through the turn, and yaw damping needs to keep up.
CN_DELTA  = -0.02   # adverse yaw from steering input
CM_PITCH_CTRL = 1.2  # pitch moment per unit active pitch-control input
                     # (delta_pitch). A real bridled kite's depower line
                     # shifts pitch trim mechanically; this is that
                     # actuator, scaled by the same q_dyn as CM_ALPHA/CM_Q
                     # so its authority relative to the aerodynamic pitch
                     # terms stays consistent across the flight envelope
                     # (unlike scaling the *gain* on an already-saturated
                     # actuator, which doesn't help -- this is a distinct
                     # actuator on a distinct axis).


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
        aspect_ratio = ASPECT_RATIO,
        bridle_offset = 0.4,
    ):
        self.mass       = mass
        self.area       = area
        self.CL         = CL              # lift coefficient at alpha=0 (CL0)
        self.CD         = CD              # parasitic drag coefficient
        self.tether_len = tether_len
        self.CD_gen_max = CD_gen_max
        self.site_elevation_msl = site_elevation_msl
        self.glide_ratio = CL / CD

        # Planform: span/chord from area + aspect ratio (b^2/S).
        self.aspect_ratio = aspect_ratio
        self.span  = np.sqrt(area * aspect_ratio)
        self.chord = area / self.span

        # Finite-wing lift-curve slope (per rad) and induced-drag
        # factor, both standard closed-form functions of aspect ratio.
        self.CL_alpha  = 2 * np.pi * aspect_ratio / (aspect_ratio + 2)
        self.k_induced = 1.0 / (np.pi * aspect_ratio * OSWALD_E)

        # Moments of inertia: flat-plate estimate about the CG (most
        # of an inflatable kite's mass is in the wing skin/struts, not
        # concentrated at the center, so a uniform flat plate is a
        # reasonable first-order model). Pitch (chord-wise) uses a
        # larger radius-of-gyration factor than the uniform-plate 1/12
        # -- the leading-edge tube and strut battens concentrate mass
        # fore-and-aft more than a uniform sheet would.
        self.Ixx = mass * self.span**2  / 12.0
        self.Iyy = mass * self.chord**2 / 6.0
        self.Izz = mass * (self.span**2 + self.chord**2) / 12.0
        self.inertia = np.array([self.Ixx, self.Iyy, self.Izz])

        # Distance from CG to the bridle/tether attachment point,
        # along body -z (i.e. on the belly side) -- gives the tether
        # a moment arm, so a loaded tether pulls the nose toward the
        # tether line the way a real bridled kite weathervanes.
        self.bridle_offset = bridle_offset

        print(f"KiteConfig initialized:")
        print(f"  mass={mass}kg, area={area}m², CL0={CL}, CD0={CD}")
        print(f"  glide ratio CL/CD = {self.glide_ratio:.1f}")
        print(f"  span={self.span:.2f}m, chord={self.chord:.2f}m, "
              f"AR={aspect_ratio:.1f}")
        print(f"  inertia: Ixx={self.Ixx:.1f}, Iyy={self.Iyy:.1f}, "
              f"Izz={self.Izz:.1f} kg m²")
        print(f"  tether length = {tether_len}m, bridle offset = {bridle_offset}m")
        print(f"  onboard generator: CD_gen_max = {CD_gen_max}")
        print(f"  site elevation = {site_elevation_msl:.0f}m MSL")
        print(f"  stall check: min v_rel for lift > weight = "
              f"{np.sqrt(2*mass*G / (RHO0*CL*area)):.1f} m/s")

    def __repr__(self):
        return (f"KiteConfig(mass={self.mass}kg, area={self.area}m², "
                f"CL0={self.CL}, CD0={self.CD}, l={self.tether_len}m, "
                f"CD_gen_max={self.CD_gen_max}, "
                f"site_elevation_msl={self.site_elevation_msl:.0f}m)")


class WindEnvironment:
    """
    Wind conditions the kite flies in.

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
    """

    def __init__(self, wind_speed_ref=12.0, ref_height=10.0, shear_exp=0.14):
        self.wind_speed_ref = wind_speed_ref
        self.ref_height = ref_height
        self.shear_exp = shear_exp

    def wind_at(self, altitude):
        h = max(altitude, 1.0)
        speed = self.wind_speed_ref * (h / self.ref_height) ** self.shear_exp
        return np.array([speed, 0.0, 0.0])

    def __repr__(self):
        return (f"WindEnvironment(v_ref={self.wind_speed_ref:.1f} m/s @ "
                f"{self.ref_height:.0f}m, shear_exp={self.shear_exp:.2f})")


def air_density(altitude_agl, site_elevation_msl=0.0):
    """Exponential atmosphere model, adjusted for site elevation above
    sea level (a high-altitude site has thinner air than a coastal one
    at the same height above the ground)."""
    total_altitude = altitude_agl + site_elevation_msl
    return RHO0 * np.exp(-total_altitude / H_RHO)


# ─────────────────────────────────────────────
#  Quaternion utilities (scalar-first: [w, x, y, z])
# ─────────────────────────────────────────────

def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotmat(q):
    """Body->world rotation matrix from a unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


def rotmat_to_quat(R):
    """Standard trace-based rotation-matrix -> quaternion conversion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return quat_normalize(np.array([w, x, y, z]))


def compute_lift_hat(pos, v_rel_hat, bank_angle=0.0):
    """
    Reference ("nominal") lift-direction geometry: perpendicular to
    the apparent wind, tangent to the tether sphere. bank_angle=0
    gives max upward lift; positive banks to the right.

    This is not used to generate aerodynamic force any more (that now
    comes from the kite's actual attitude state -- see
    `kite_forces_moments`); it's kept as the reference geometry for
    two things that need a "what should zero/commanded bank look
    like" answer: setting an initial attitude (`initial_attitude`) and
    measuring how far the kite's actual bank has drifted from a
    commanded value (`bank_angle_from_lift`), for the control loops in
    kite_flygen.py / kite_groundgen.py.
    """
    tether_hat = -pos / np.linalg.norm(pos)

    span = np.cross(tether_hat, v_rel_hat)
    span_norm = np.linalg.norm(span)
    if span_norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    span_hat = span / span_norm

    lift_hat_0 = np.cross(span_hat, v_rel_hat)
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


def bank_angle_from_lift(pos, v_rel_hat, lift_hat_actual):
    """Inverse of compute_lift_hat: how far lift_hat_actual has been
    rotated (about the tether axis) from the zero-bank reference --
    i.e. the kite's current bank angle, for feedback control."""
    tether_hat = -pos / np.linalg.norm(pos)
    K = -tether_hat
    lift_hat_0 = compute_lift_hat(pos, v_rel_hat, 0.0)
    cos_b = np.clip(np.dot(lift_hat_0, lift_hat_actual), -1.0, 1.0)
    sin_b = np.dot(np.cross(lift_hat_0, lift_hat_actual), K)
    return np.arctan2(sin_b, cos_b)


def initial_attitude(pos, v_rel_hat, bank_angle=0.0):
    """
    Build a starting quaternion for a kite flying with a given
    apparent-wind direction and commanded bank, using the same
    zero-bank reference geometry as `compute_lift_hat`: nose along the
    kite's velocity-relative-to-air direction, belly (-z_b) toward the
    reference lift direction.
    """
    x_b = -v_rel_hat   # V_a_hat = (vel-wind)/|.| = -v_rel_hat
    lift_hat = compute_lift_hat(pos, v_rel_hat, bank_angle)
    z_b = -lift_hat
    z_b = z_b - np.dot(z_b, x_b) * x_b
    z_b /= np.linalg.norm(z_b)
    y_b = np.cross(z_b, x_b)
    R = np.column_stack([x_b, y_b, z_b])
    return rotmat_to_quat(R)


# ─────────────────────────────────────────────
#  Forces & moments
# ─────────────────────────────────────────────

def kite_forces_moments(pos, vel, quat, omega, wind_vec, cfg,
                         delta_steer=0.0, gen_load=0.0, CL_scale=1.0,
                         delta_pitch=0.0):
    """
    Total force (world frame), total moment (body frame), tether
    tension, instantaneous power, apparent airspeed, angle of attack,
    and the actual lift-direction unit vector, all at one instant.

    delta_steer : steering control, -1..1 (differential brake/line
                  deflection -- positive rolls right).
    gen_load    : onboard generator loading, 0-1.
    CL_scale    : depower control, 0-1 (1 = full lift; lower approximates
                  pulling the depower line to reduce effective lift,
                  used during pumping-cycle retraction).
    delta_pitch : active pitch-trim control, -1..1 (positive pitches
                  nose up / increases alpha). Distinct from CL_scale:
                  CL_scale directly scales lift force, delta_pitch
                  produces a pitching *moment* (via CM_PITCH_CTRL) that
                  the rotational dynamics have to respond to, same as
                  delta_steer does for roll.

    Returns (F_total, M_total_body, T, P_inst, v_rel_mag, alpha, lift_hat).
    """
    altitude = pos[2]
    rho      = air_density(altitude, cfg.site_elevation_msl)
    gen_load = min(max(gen_load, 0.0), 1.0)
    l_current = np.linalg.norm(pos)
    R = quat_to_rotmat(quat)

    V_a_w = vel - wind_vec           # kite's velocity relative to the air
    Vr = np.linalg.norm(V_a_w)
    if Vr < 1e-3:
        F_grav = np.array([0.0, 0.0, -cfg.mass * G])
        return F_grav, np.zeros(3), 0.0, 0.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0])

    V_a_b = R.T @ V_a_w
    u, v, w = V_a_b
    alpha = np.arctan2(w, u)
    beta  = np.arcsin(np.clip(v / Vr, -1.0, 1.0))

    # Lift/drag build-up from angle of attack, with a smooth
    # exponential stall falloff past the critical angle (same "simple
    # phenomenological curve" style as the exponential atmosphere
    # model above, not a hard clip).
    CL_lin = cfg.CL + cfg.CL_alpha * alpha
    over = max(abs(alpha) - ALPHA_STALL, 0.0)
    stall_factor = np.exp(-4.0 * over)
    CL = CL_scale * CL_lin * stall_factor
    CD_gen = gen_load * cfg.CD_gen_max
    CD = cfg.CD + cfg.k_induced * CL**2 + CD_gen
    CY = CY_BETA * beta

    q_dyn = 0.5 * rho * Vr**2
    S = cfg.area
    L_mag, D_mag, Y_mag = q_dyn*S*CL, q_dyn*S*CD, q_dyn*S*CY

    flow_hat  = V_a_w / Vr            # direction kite moves through the air
    v_rel_hat = -flow_hat             # apparent-wind direction (drag acts along this)
    x_b_w, y_b_w, z_b_w = R[:, 0], R[:, 1], R[:, 2]
    up_b_w = -z_b_w

    lift_hat = up_b_w - np.dot(up_b_w, flow_hat) * flow_hat
    ln = np.linalg.norm(lift_hat)
    lift_hat = lift_hat / ln if ln > 1e-6 else compute_lift_hat(pos, v_rel_hat, 0.0)

    side_hat = y_b_w - np.dot(y_b_w, flow_hat) * flow_hat
    sn = np.linalg.norm(side_hat)
    side_hat = side_hat / sn if sn > 1e-6 else np.zeros(3)

    F_lift = L_mag * lift_hat
    F_drag = D_mag * v_rel_hat
    F_side = Y_mag * side_hat
    F_grav = np.array([0.0, 0.0, -cfg.mass * G])

    # Tether tension: centripetal term (needed to keep the kite moving
    # in a circle) plus the outward-radial component of everything else.
    tether_hat  = -pos / l_current
    outward_hat = pos / l_current
    F_aero_grav = F_lift + F_drag + F_side + F_grav
    T_aero      = np.dot(F_aero_grav, outward_hat)

    vel_tangential = vel - np.dot(vel, tether_hat) * tether_hat
    v_tan_mag      = np.linalg.norm(vel_tangential)
    T_centripetal  = cfg.mass * v_tan_mag**2 / l_current
    T              = T_centripetal + max(0.0, T_aero)
    F_tether_w     = T * tether_hat

    F_total = F_lift + F_drag + F_side + F_grav + F_tether_w

    # Moments (body frame): aerodynamic stability/control derivatives,
    # plus the moment from the tether pulling on the bridle point
    # (offset below the CG) rather than through the CG itself -- this
    # is what makes a bridled kite weathervane its nose toward the
    # tether line under load.
    p, q_rate, r = omega
    b, c = cfg.span, cfg.chord
    # Floor Vr here (not in the force build-up above) so damping terms
    # don't blow up to a numerical singularity if the kite slows
    # toward a stall or momentary stop -- a real wing's rate damping
    # doesn't actually diverge at low airspeed, it just gets weak.
    Vr_damp = max(Vr, 3.0)
    p_hat, q_hat, r_hat = p*b/(2*Vr_damp), q_rate*c/(2*Vr_damp), r*b/(2*Vr_damp)

    L_roll  = q_dyn*S*b*(CL_BETA*beta + CL_P*p_hat + CL_DELTA*delta_steer)
    M_pitch = q_dyn*S*c*(CM0 + CM_ALPHA*alpha + CM_Q*q_hat + CM_PITCH_CTRL*delta_pitch)
    N_yaw   = q_dyn*S*b*(CN_BETA*beta + CN_R*r_hat + CN_DELTA*delta_steer)
    M_aero_b = np.array([L_roll, M_pitch, N_yaw])

    F_tether_b = R.T @ F_tether_w
    r_bridle_b = np.array([0.0, 0.0, cfg.bridle_offset])
    M_tether_b = np.cross(r_bridle_b, F_tether_b)

    M_total = M_aero_b + M_tether_b

    F_gen_mag = CD_gen * q_dyn * S
    P_inst    = F_gen_mag * Vr * ETA_GEN

    return F_total, M_total, T, P_inst, Vr, alpha, lift_hat


# ─────────────────────────────────────────────
#  6DOF RK4 integration
# ─────────────────────────────────────────────

def _state_derivatives(pos, vel, quat, omega, wind_vec, cfg,
                        delta_steer, gen_load, CL_scale, delta_pitch):
    F, M, T, P, Vr, alpha, lift_hat = kite_forces_moments(
        pos, vel, quat, omega, wind_vec, cfg, delta_steer, gen_load, CL_scale,
        delta_pitch)

    pos_dot = vel
    vel_dot = F / cfg.mass
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    quat_dot = 0.5 * quat_mult(quat, omega_quat)
    Iomega = cfg.inertia * omega
    omega_dot = (M - np.cross(omega, Iomega)) / cfg.inertia

    return pos_dot, vel_dot, quat_dot, omega_dot, T, P, Vr, alpha, lift_hat


def rk4_step(pos, vel, quat, omega, wind_vec, cfg, delta_steer, dt,
             gen_load=0.0, CL_scale=1.0, delta_pitch=0.0):
    """One RK4 step of the full 13-state 6DOF system, fixed tether
    length (Fly-Gen mode: tether payed out to `cfg.tether_len` and
    held there)."""

    def d(p, v, q, w):
        return _state_derivatives(p, v, q, w, wind_vec, cfg, delta_steer,
                                   gen_load, CL_scale, delta_pitch)

    dp1, dv1, dq1, dw1, T, P, Vr, alpha, lift_hat = d(pos, vel, quat, omega)
    dp2, dv2, dq2, dw2, *_ = d(pos + 0.5*dt*dp1, vel + 0.5*dt*dv1,
                               quat_normalize(quat + 0.5*dt*dq1), omega + 0.5*dt*dw1)
    dp3, dv3, dq3, dw3, *_ = d(pos + 0.5*dt*dp2, vel + 0.5*dt*dv2,
                               quat_normalize(quat + 0.5*dt*dq2), omega + 0.5*dt*dw2)
    dp4, dv4, dq4, dw4, *_ = d(pos + dt*dp3, vel + dt*dv3,
                               quat_normalize(quat + dt*dq3), omega + dt*dw3)

    pos_new   = pos + (dt/6.0)*(dp1 + 2*dp2 + 2*dp3 + dp4)
    vel_new   = vel + (dt/6.0)*(dv1 + 2*dv2 + 2*dv3 + dv4)
    quat_new  = quat_normalize(quat + (dt/6.0)*(dq1 + 2*dq2 + 2*dq3 + dq4))
    omega_new = np.clip(omega + (dt/6.0)*(dw1 + 2*dw2 + 2*dw3 + dw4),
                         -OMEGA_MAX, OMEGA_MAX)

    # Snap back onto the tether sphere (RK4 drifts slightly off it)
    l = np.linalg.norm(pos_new)
    pos_new = pos_new * (cfg.tether_len / l)
    r_hat   = pos_new / cfg.tether_len
    vel_new = vel_new - np.dot(vel_new, r_hat) * r_hat

    return pos_new, vel_new, quat_new, omega_new, T, P, Vr, alpha, lift_hat


def rk4_step_variable_tether(pos, vel, quat, omega, wind_vec, cfg,
                              delta_steer, dt, l_dot, gen_load=0.0,
                              CL_scale=1.0, delta_pitch=0.0):
    """
    RK4 step with a changing tether length, for pumping-cycle mode
    (kite_groundgen.py): a ground winch reels the tether out (l_dot >
    0, generating power under tension) or in (l_dot < 0, retracting,
    usually depowered).

    l_dot is treated as a directly-commanded winch speed.
    """

    def d(p, v, q, w):
        return _state_derivatives(p, v, q, w, wind_vec, cfg, delta_steer,
                                   gen_load, CL_scale, delta_pitch)

    l_old = np.linalg.norm(pos)

    dp1, dv1, dq1, dw1, T, P, Vr, alpha, lift_hat = d(pos, vel, quat, omega)
    dp2, dv2, dq2, dw2, *_ = d(pos + 0.5*dt*dp1, vel + 0.5*dt*dv1,
                               quat_normalize(quat + 0.5*dt*dq1), omega + 0.5*dt*dw1)
    dp3, dv3, dq3, dw3, *_ = d(pos + 0.5*dt*dp2, vel + 0.5*dt*dv2,
                               quat_normalize(quat + 0.5*dt*dq2), omega + 0.5*dt*dw2)
    dp4, dv4, dq4, dw4, *_ = d(pos + dt*dp3, vel + dt*dv3,
                               quat_normalize(quat + dt*dq3), omega + dt*dw3)

    pos_new   = pos + (dt/6.0)*(dp1 + 2*dp2 + 2*dp3 + dp4)
    vel_new   = vel + (dt/6.0)*(dv1 + 2*dv2 + 2*dv3 + dv4)
    quat_new  = quat_normalize(quat + (dt/6.0)*(dq1 + 2*dq2 + 2*dq3 + dq4))
    omega_new = np.clip(omega + (dt/6.0)*(dw1 + 2*dw2 + 2*dw3 + dw4),
                         -OMEGA_MAX, OMEGA_MAX)

    # Force the tether to the commanded length, radial velocity = l_dot
    l_new   = max(l_old + l_dot*dt, 1.0)
    r_hat   = pos_new / np.linalg.norm(pos_new)
    pos_new = r_hat * l_new
    vel_tan = vel_new - np.dot(vel_new, r_hat) * r_hat
    vel_new = vel_tan + l_dot * r_hat

    return pos_new, vel_new, quat_new, omega_new, T, P, Vr, alpha, lift_hat


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cfg = KiteConfig(mass=50.0, area=20.0, CL=1.0, CD=0.15, tether_len=800.0)
    print()

    print("=" * 55)
    print("Test 1: Static force/moment balance at operating point")
    print("=" * 55)

    elev = np.radians(30)
    pos  = np.array([cfg.tether_len * np.cos(elev), 0.0, cfg.tether_len * np.sin(elev)])
    vel  = np.array([0.0, 20.0, 0.0])
    wind = np.array([13.3, 0.0, 0.0])
    v_rel_hat0 = (wind - vel) / np.linalg.norm(wind - vel)
    quat = initial_attitude(pos, v_rel_hat0, bank_angle=0.0)
    omega = np.zeros(3)

    F, M, T, P, Vr, alpha, lift_hat = kite_forces_moments(
        pos, vel, quat, omega, wind, cfg, delta_steer=0.0, gen_load=0.3)
    rho_alt = air_density(pos[2])

    print(f"  Altitude:          {pos[2]:.0f} m")
    print(f"  Apparent airspeed: {Vr:.1f} m/s, alpha={np.degrees(alpha):.1f} deg")
    print(f"  Tether tension:    {T:.1f} N")
    print(f"  Instant power:     {P/1000:.2f} kW  (gen_load=0.3)")
    print(f"  Weight:            {cfg.mass*G:.0f} N")
    print(f"  Moment (roll,pitch,yaw): "
          f"[{M[0]:.1f}, {M[1]:.1f}, {M[2]:.1f}] N·m")

    print()
    print("=" * 55)
    print("Test 2: Free flight - 30s, delta_steer=0 (passive stability)")
    print("=" * 55)

    pos = np.array([cfg.tether_len * np.cos(np.radians(30)), 0.0,
                     cfg.tether_len * np.sin(np.radians(30))])
    vel = np.array([0.0, 15.0, 5.0])
    wind = np.array([13.3, 0.0, 0.0])
    v_rel_hat0 = (wind - vel) / np.linalg.norm(wind - vel)
    quat = initial_attitude(pos, v_rel_hat0, bank_angle=0.0)
    omega = np.zeros(3)
    dt, t_end = 0.01, 30.0
    steps = int(t_end / dt)

    t_hist   = np.zeros(steps)
    pos_hist = np.zeros((steps, 3))
    T_hist   = np.zeros(steps)
    P_hist   = np.zeros(steps)
    alpha_hist = np.zeros(steps)

    for i in range(steps):
        t_hist[i]   = i * dt
        pos_hist[i] = pos
        F, M, T, P, Vr, alpha, lift_hat = kite_forces_moments(
            pos, vel, quat, omega, wind, cfg, gen_load=0.3)
        T_hist[i] = T
        P_hist[i] = P / 1000
        alpha_hist[i] = np.degrees(alpha)
        pos, vel, quat, omega, T, P, Vr, alpha, lift_hat = rk4_step(
            pos, vel, quat, omega, wind, cfg, 0.0, dt, gen_load=0.3)

    tether_lengths = np.linalg.norm(pos_hist, axis=1)
    print(f"  Tether length - min: {tether_lengths.min():.4f}m, "
          f"max: {tether_lengths.max():.4f}m  (should be {cfg.tether_len}m)")
    print(f"  Peak tension:    {T_hist.max():.0f} N")
    print(f"  Peak power:      {P_hist.max():.2f} kW")
    print(f"  Mean power:      {P_hist.mean():.2f} kW")
    print(f"  Alpha range:     [{alpha_hist.min():.1f}, {alpha_hist.max():.1f}] deg")
    print(f"  quat norm drift: {abs(np.linalg.norm(quat) - 1.0):.2e}")

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(141)
    ax1.plot(pos_hist[:, 1], pos_hist[:, 2])
    ax1.set_xlabel('Y crosswind (m)')
    ax1.set_ylabel('Z altitude (m)')
    ax1.set_title('Kite path (front view)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(142)
    ax2.plot(t_hist, T_hist)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Tension (N)')
    ax2.set_title('Tether tension')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(143)
    ax3.plot(t_hist, P_hist, color='#f0a020')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Power (kW)')
    ax3.set_title('Instantaneous power')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(144)
    ax4.plot(t_hist, alpha_hist, color='#2E86AB')
    ax4.axhline(np.degrees(ALPHA_STALL), color='red', linestyle='--', linewidth=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angle of attack (deg)')
    ax4.set_title('Alpha (passive, no pitch control)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_free_flight.png', dpi=150)
    plt.show()
    print("  Plot saved: test_free_flight.png")
