"""
kite_environments.py

Turns weather/location inputs into the WindEnvironment and KiteConfig
objects the simulation needs. Provides a few named location presets
plus an interactive prompt for entering conditions by hand.
"""

from kite_dynamics import KiteConfig, WindEnvironment


# ─────────────────────────────────────────────
#  Location presets
# ─────────────────────────────────────────────
# Rough wind climates loosely modeled on real regions - not a live
# weather feed, just reasonable starting points you can override.
LOCATION_PRESETS = {
    "atacama_coast": dict(
        label="Atacama coast, Chile (steady trade winds)",
        wind_speed_ref=12.0, shear_exp=0.14, site_elevation_msl=500.0,
    ),
    "north_sea_offshore": dict(
        label="North Sea, offshore (strong, low-shear marine wind)",
        wind_speed_ref=10.5, shear_exp=0.10, site_elevation_msl=0.0,
    ),
    "patagonia_ridge": dict(
        label="Patagonia ridge, Argentina/Chile (very strong ridge wind)",
        wind_speed_ref=17.0, shear_exp=0.12, site_elevation_msl=250.0,
    ),
    "great_plains_us": dict(
        label="Great Plains, USA (moderate wind, rougher terrain)",
        wind_speed_ref=9.0, shear_exp=0.20, site_elevation_msl=700.0,
    ),
    "andes_highland": dict(
        label="High Andes plateau (thin air, strong daytime wind)",
        wind_speed_ref=11.0, shear_exp=0.16, site_elevation_msl=3800.0,
    ),
    "north_sea_calm": dict(
        label="North Sea, a calmer day (below-average wind)",
        wind_speed_ref=6.5, shear_exp=0.12, site_elevation_msl=0.0,
    ),
}


def build_environment(preset_key=None, **overrides):
    """
    Build a WindEnvironment (and return the site_elevation_msl
    separately, since that lives on KiteConfig) from a preset, with
    any field overridden by keyword.
    """
    if preset_key is not None:
        if preset_key not in LOCATION_PRESETS:
            raise KeyError(f"Unknown location preset '{preset_key}'. "
                            f"Options: {list(LOCATION_PRESETS.keys())}")
        params = dict(LOCATION_PRESETS[preset_key])
        params.pop("label", None)
    else:
        params = dict(wind_speed_ref=12.0, shear_exp=0.14,
                      site_elevation_msl=0.0)

    params.update(overrides)
    site_elevation_msl = params.pop("site_elevation_msl")
    env = WindEnvironment(wind_speed_ref=params["wind_speed_ref"],
                           shear_exp=params["shear_exp"])
    return env, site_elevation_msl


# ─────────────────────────────────────────────
#  Interactive prompt flow
# ─────────────────────────────────────────────

def _prompt_float(msg, default):
    raw = input(f"{msg} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Couldn't parse that, using default {default}.")
        return default


def prompt_for_scenario():
    """
    Interactive CLI flow: pick a location preset (or go fully custom),
    then optionally tweak the kite itself. Returns (cfg, env,
    scenario_label).
    """
    print("=" * 60)
    print("SITE / WEATHER SETUP")
    print("=" * 60)
    keys = list(LOCATION_PRESETS.keys())
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {LOCATION_PRESETS[k]['label']}")
    print(f"  {len(keys)+1}. Custom (enter your own numbers)")

    choice = input(f"Pick a location [1-{len(keys)+1}], "
                    f"or press Enter for #1: ").strip()
    if choice == "":
        idx = 0
    else:
        try:
            idx = int(choice) - 1
        except ValueError:
            idx = 0
        idx = max(0, min(idx, len(keys)))

    if idx == len(keys):
        print("\nCustom site - enter your own values (Enter to accept default):")
        wind_speed_ref = _prompt_float("Wind speed at 10m reference height (m/s)", 12.0)
        shear_exp = _prompt_float("Wind shear exponent (0.10 open/flat .. 0.25 rough terrain)", 0.14)
        site_elevation_msl = _prompt_float("Site elevation above sea level (m)", 0.0)
        label = "Custom site"
    else:
        key = keys[idx]
        preset = LOCATION_PRESETS[key]
        label = preset["label"]
        print(f"\nUsing preset: {label}")
        wind_speed_ref = preset["wind_speed_ref"]
        shear_exp = preset["shear_exp"]
        site_elevation_msl = preset["site_elevation_msl"]

    env = WindEnvironment(wind_speed_ref=wind_speed_ref, shear_exp=shear_exp)

    print()
    print("=" * 60)
    print("KITE SETUP")
    print("=" * 60)
    customize = input("Customize the kite? [y/N]: ").strip().lower()
    if customize == "y":
        mass = _prompt_float("Kite mass (kg)", 50.0)
        area = _prompt_float("Wing area (m^2)", 20.0)
        CL = _prompt_float("Lift coefficient", 1.0)
        CD = _prompt_float("Drag coefficient", 0.15)
    else:
        mass, area, CL, CD = 50.0, 20.0, 1.0, 0.15

    cfg = KiteConfig(mass=mass, area=area, CL=CL, CD=CD,
                      site_elevation_msl=site_elevation_msl)
    return cfg, env, label
