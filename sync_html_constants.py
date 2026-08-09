"""
sync_html_constants.py

kite_optimizer_app.html duplicates the physics constants from
kite_dynamics.py / kite_pumping.py in JavaScript (there's no shared
build step -- it's a standalone file), so a constant tweaked in one
place can silently drift from the other. This script doesn't unify
the full duplicated logic (that would need a real build pipeline);
it just keeps the small set of numeric constants most likely to
quietly diverge in sync, one direction, Python -> HTML.

Run it after changing any of the constants it lists below:
    python sync_html_constants.py
"""

import re

from kite_dynamics import G, RHO0, H_RHO, ETA_GEN
from kite_pumping import CROSSWIND_EFFICIENCY, V_REL_MAX, BANK_MAX_STRUCTURAL
import numpy as np

HTML_PATH = "kite_optimizer_app.html"

PLAIN_CONSTANTS = {
    "G": G,
    "RHO0": RHO0,
    "H_RHO": H_RHO,
    "ETA_GEN": ETA_GEN,
    "CROSSWIND_EFFICIENCY": CROSSWIND_EFFICIENCY,
    "V_REL_MAX": V_REL_MAX,
}


def _fmt(value):
    if float(value).is_integer():
        return f"{value:.1f}"
    return repr(float(value))


def sync():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    changed = []
    unchanged = []

    for name, value in PLAIN_CONSTANTS.items():
        pattern = re.compile(rf"(const {name} = )([^;]+)(;)")
        match = pattern.search(html)
        if not match:
            print(f"  ! {name}: not found in {HTML_PATH}, skipping")
            continue
        new_text = _fmt(value)
        if match.group(2).strip() == new_text:
            unchanged.append(name)
        else:
            html = pattern.sub(rf"\g<1>{new_text}\g<3>", html, count=1)
            changed.append((name, match.group(2).strip(), new_text))

    # BANK_MAX_STRUCTURAL is stored in JS as a degrees expression
    # (`const BANK_MAX_STRUCTURAL = <deg> * Math.PI / 180;`), not a bare
    # number, since that's the more readable form in JS -- sync the
    # degree literal instead of the raw radian value.
    deg_value = round(np.degrees(BANK_MAX_STRUCTURAL), 6)
    pattern = re.compile(r"(const BANK_MAX_STRUCTURAL = )([^*]+)(\* Math\.PI / 180;)")
    match = pattern.search(html)
    if match:
        new_text = _fmt(deg_value) + " "
        if match.group(2).strip() != _fmt(deg_value):
            html = pattern.sub(rf"\g<1>{new_text}\g<3>", html, count=1)
            changed.append(("BANK_MAX_STRUCTURAL (deg)", match.group(2).strip(), _fmt(deg_value)))
        else:
            unchanged.append("BANK_MAX_STRUCTURAL (deg)")
    else:
        print("  ! BANK_MAX_STRUCTURAL: not found in expected form, skipping")

    if changed:
        with open(HTML_PATH, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Patched {len(changed)} constant(s) in {HTML_PATH}:")
        for name, old, new in changed:
            print(f"  {name}: {old} -> {new}")
    else:
        print(f"All constants already in sync ({len(unchanged)} checked).")


if __name__ == "__main__":
    sync()
