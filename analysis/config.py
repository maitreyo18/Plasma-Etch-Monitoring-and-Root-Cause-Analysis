from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Analysis design choices (not facts from the data)
PHASE1_EXPS = [29, 31]           # normal wafers used to build the model
PHASE2_EXP = 33                  # normal wafers monitored as "live"
TRUNCATED_FRAC = 0.5             # wafer is excluded if a step/block has < this fraction of the median sample count
Endpt_WINDOW = (0.2, 0.8)        # central fraction of step 4 used for Endpt A plateau level and slope
OES_SKIP_FIRST = 2               # spectra skipped at the start of the main-etch window

# Univariate SPC variables (feature names from features.py)
KEY_VARS = [
    "clear_time", "step5_time",
    "TCP Top Pwr_mean_s4", "RF Btm Pwr_mean_s4", "BCl3 Flow_mean_s4",
    "Cl2 Flow_mean_s4", "Pressure_mean_s4", "Vat Valve_mean_s4",
    "He Press_mean_s4", "He Press_std_s4", "Endpt A_plateau",
]

# The data has no spec limits. Set (LSL, USL) in seconds to get Cp/Cpk; None reports process spread only.
CLEAR_TIME_SPEC = None

ALPHA = 0.99
KMAX = 10
CV_FOLDS = 7

# Palette (reference palette, light surface)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"

# Fault grouping by mechanism (a physics-based choice): what the controllers respond to
MECHANISM = {"Pr": "pressure", "Cl2": "gas flow", "BCl3": "gas flow", "TCP": "power", "RF": "power", "He": "He chuck"}

# Interpretation only (never used in any calculation): substring -> physical reading of a feature
HINTS = {
    "Vat Valve": "throttle valve moved: pressure loop compensating (flow or pressure fault)",
    "Pressure": "chamber pressure shift",
    "Endpt A": "etch-product emission level: etch rate / plasma density changed",
    "He Press": "backside helium: chuck cooling / regulation",
    "TCP Tuner": "TCP match retuned: coil coupling changed",
    "TCP Load": "TCP match retuned: coil coupling changed",
    "RF Load": "bias match retuned: plasma loading changed",
    "RF Tuner": "bias match retuned",
    "RF Pwr": "bias power reading",
    "Cl2 Flow": "chlorine supply",
    "BCl3 Flow": "BCl3 supply",
    "clear_time": "etch rate changed",
    "336.98": "N2 line: air leak / TiN",
    "395.8": "Al I: etch product",
    "394.4": "Al I: etch product",
    "725nm": "Cl I: etchant",
    "261.8": "AlCl: etch product",
    "272.2": "BCl: BCl3 dissociation",
    "324.8": "Cu I: film removal",
    "thd": "RF harmonic distortion: sheath / plasma state",
    "S1": "bias feed electrical response", "S2": "bias feed electrical response",
    "S3": "TCP coil feed electrical response", "S4": "TCP coil feed electrical response",
    "S34": "TCP coil feed electrical response",
}
