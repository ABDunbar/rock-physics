# ── Quartz (info_params.txt) ──────────────────────────────────────────────────
K_QTZ   = 36.8;  G_QTZ   = 44.0;  RHO_QTZ  = 2.65   # GPa, g/cc

# ── Clay / Shale (info_params.txt) ────────────────────────────────────────────
K_SH    = 15.0;  G_SH    =  5.0;  RHO_SH   = 2.72

# ── Brine (info_params.txt) ───────────────────────────────────────────────────
K_BR    = 2.80;  RHO_BR  = 1.09   # GPa, g/cc

# ── Dry gas at reservoir conditions ───────────────────────────────────────────
K_GAS   = 0.02;  RHO_GAS = 0.12   # GPa, g/cc

# ── Oil: density given directly; K_OIL computed via Batzle-Wang below ─────────
RHO_OIL = 0.78                     # g/cc

# ── Reservoir conditions ──────────────────────────────────────────────────────
RESERVOIR_TEMP   = 77.2    # °C
RESERVOIR_P      = 20.0    # MPa effective pressure
OIL_API          = 32
OIL_GOR          = 64      # sm³/sm³  (metric solution GOR)
OWC_DEPTH        = 2183    # m  (oil-water contact)
TOP_HEIMDAL      = 2153    # m  (top of Heimdal reservoir)
