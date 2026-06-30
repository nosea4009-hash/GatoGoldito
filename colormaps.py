"""
colormaps.py
============
Colormap personalizado para la Temperatura de Topes de Nube (Banda 13, IR ~10.3 um).

Esta es la paleta provista por el usuario (TRP), definida como pares
[posicion_0a255, [R, G, B, A]]. Se calibra a temperatura de forma LINEAL:

    - Posicion 15  (gris oscuro)  -> extremo CALIDO  = +40 C
    - Posicion 254 (purpura)      -> extremo FRIO    = -90 C

Para AJUSTAR el rango de temperatura, cambia POS/T_WARM y POS/T_COLD abajo.
Para cambiar colores, edita _RAW_PALETTE (formato identico al original).
"""

import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, Normalize

# --- Paleta original (tal cual fue provista): [posicion, [R, G, B, A]] --------
_RAW_PALETTE = [
    [15,  [24,  24,  24,  255]],
    [33,  [61,  61,  61,  255]],
    [52,  [128, 128, 128, 255]],
    [70,  [158, 158, 158, 255]],
    [125, [181, 181, 181, 255]],
    [135, [13,  192, 212, 255]],
    [144, [51,  128, 184, 255]],
    [150, [11,  11,  181, 255]],
    [159, [17,  199, 20,  255]],
    [168, [127, 224, 63,  255]],
    [177, [142, 227, 44,  255]],
    [186, [240, 236, 31,  255]],
    [195, [227, 164, 27,  255]],
    [205, [235, 2,   2,   255]],
    [214, [163, 10,  111, 255]],
    [236, [102, 4,   89,  255]],
    [254, [23,  0,   19,  255]],
]

# --- Calibracion posicion (0-255) -> temperatura (C), lineal ------------------
POS_WARM, T_WARM = 15,  40.0    # extremo calido (gris oscuro)
POS_COLD, T_COLD = 254, -90.0   # extremo frio (purpura)

# Rango de la barra de color (C)
VMIN = T_COLD   # -90
VMAX = T_WARM   # +40


def _pos_to_temp(pos: float) -> float:
    """Convierte una posicion de la paleta (0-255) a temperatura en C (lineal)."""
    frac = (pos - POS_WARM) / (POS_COLD - POS_WARM)
    return T_WARM + frac * (T_COLD - T_WARM)


def cloudtop_cmap():
    """Devuelve (cmap, norm) para la temperatura de topes de nube en C."""
    # (temperatura_C, color_rgb_0a1) para cada punto de la paleta
    pts = []
    for pos, (r, g, b, _a) in _RAW_PALETTE:
        temp = _pos_to_temp(pos)
        pts.append((temp, (r / 255.0, g / 255.0, b / 255.0)))

    # Ordena por temperatura ascendente y normaliza a [0, 1] sobre [VMIN, VMAX]
    pts.sort(key=lambda p: p[0])
    span = VMAX - VMIN
    norm_stops = [((t - VMIN) / span, c) for t, c in pts]

    # Asegura posiciones estrictamente crecientes
    eps = 1e-6
    for i in range(1, len(norm_stops)):
        if norm_stops[i][0] <= norm_stops[i - 1][0]:
            norm_stops[i] = (norm_stops[i - 1][0] + eps, norm_stops[i][1])
    # Fuerza extremos exactos 0.0 y 1.0
    norm_stops[0] = (0.0, norm_stops[0][1])
    norm_stops[-1] = (1.0, norm_stops[-1][1])

    cmap = LinearSegmentedColormap.from_list("cloudtop_trp", norm_stops, N=256)
    cmap.set_bad("black")  # pixeles sin dato
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    return cmap, norm


# Marcas (ticks) de la barra de color (C): -90, -80, ..., 40
COLORBAR_TICKS = np.arange(VMIN, VMAX + 1, 10)



# =============================================================================
#                  CANAL VISIBLE (Banda 2) - reflectancia
# =============================================================================
# El canal visible es REFLECTANCIA (0 a ~1), no temperatura. Se grafica en
# escala de grises: oscuro = poca reflectancia (superficie/oceano),
# claro = mucha reflectancia (nubes). Rango fijo 0 a 1.

VISIBLE_VMIN = 0.0
VISIBLE_VMAX = 1.0
VISIBLE_TICKS = np.arange(0.0, 1.01, 0.2)


def visible_cmap():
    """Devuelve (cmap, norm) en escala de grises para reflectancia (0-1)."""
    cmap = matplotlib.colormaps["gray"].copy()
    cmap.set_bad("black")
    norm = Normalize(vmin=VISIBLE_VMIN, vmax=VISIBLE_VMAX)
    return cmap, norm



# =============================================================================
#         RAINBOW IR (para el producto "sandwich": Visible + IR Banda 13)
# =============================================================================
# Reproduce el colorbar de referencia: -90 C = violeta, -20 C = rojo/rosa.
# (matplotlib 'rainbow' orientado con el frio en violeta y el calido en rojo.)
# set_bad transparente: los pixeles enmascarados (topes calidos) dejan ver el
# canal visible debajo en el sandwich.

RAINBOW_IR_VMIN = -90.0   # extremo frio (violeta)
RAINBOW_IR_VMAX = -20.0   # extremo calido coloreado (rojo)
RAINBOW_IR_TICKS = np.arange(-90.0, -19.0, 10.0)  # -90, -80, ..., -20


def rainbow_ir_cmap():
    """Devuelve (cmap, norm) rainbow para el IR del sandwich."""
    cmap = matplotlib.colormaps["rainbow"].copy()
    cmap.set_bad((0, 0, 0, 0))  # transparente (deja ver el visible debajo)
    norm = Normalize(vmin=RAINBOW_IR_VMIN, vmax=RAINBOW_IR_VMAX)
    return cmap, norm



# =============================================================================
#     NIGHTTIME MICROPHYSICS RGB (producto "noche": modo nocturno del GOES)
# =============================================================================
# Compuesto RGB de 3 bandas IR (funciona de dia y de noche, no depende del sol).
# Receta estandar EUMETSAT/CIRA (entradas en grados Celsius):
#
#   Rojo  = BT(12.3 um) - BT(10.3 um)   -> Banda 15 - Banda 13   [-4 ,  2] C
#   Verde = BT(10.3 um) - BT(3.9  um)   -> Banda 13 - Banda  7   [ 0 , 10] C
#   Azul  = BT(10.3 um)                 -> Banda 13              [-30.15, 19.85] C
#
# Interpretacion tipica:
#   - Niebla / nubes bajas (stratus): tonos verde-azulados claros.
#   - Nubes altas y profundas (convectivas): rojizas/marrones.
#   - Superficie despejada de noche: oscura / azul-violacea.
#
# Las diferencias de temperatura son identicas en C o K, y el canal azul usa el
# rango 243-293 K = -30.15 a 19.85 C, por eso se trabaja todo en Celsius.

NTMICRO_RED   = (-4.0, 2.0)       # T(12.3) - T(10.3)
NTMICRO_GREEN = (0.0, 10.0)       # T(10.3) - T(3.9)
NTMICRO_BLUE  = (-30.15, 19.85)   # T(10.3)  (243.0 - 293.0 K)


def _norm_channel(arr, vmin, vmax, gamma=1.0):
    """Normaliza un canal al rango [0, 1] con recorte y correccion gamma opcional."""
    out = (arr - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    if gamma and gamma != 1.0:
        out = np.power(out, 1.0 / gamma)
    return out


def nighttime_microphysics_rgb(t12_3, t10_3, t3_9):
    """Construye el RGB Nighttime Microphysics. Entradas en C, devuelve (H, W, 3)."""
    r = _norm_channel(t12_3 - t10_3, *NTMICRO_RED)
    g = _norm_channel(t10_3 - t3_9, *NTMICRO_GREEN)
    b = _norm_channel(t10_3,        *NTMICRO_BLUE)
    rgb = np.dstack([r, g, b])
    # Pixeles sin dato (NaN en cualquier banda) -> negro
    bad = ~(np.isfinite(t12_3) & np.isfinite(t10_3) & np.isfinite(t3_9))
    rgb[bad] = 0.0
    return rgb
