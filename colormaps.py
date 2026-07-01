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
#          PALETA "IR_custom" (segunda paleta personalizada del usuario)
# =============================================================================
# Misma calibracion posicion->temperatura que la paleta original (POS_WARM/
# T_WARM y POS_COLD/T_COLD, arriba): posicion 15 = +40 C, posicion 254 = -90 C.
# Solo cambian los COLORES.

IR_CUSTOM_RAW_PALETTE = [
    [15,  [0,   0,   0,   255]],
    [33,  [61,  61,  61,  255]],
    [52,  [145, 145, 145, 255]],
    [70,  [158, 158, 158, 255]],
    [125, [181, 181, 181, 255]],
    [135, [227, 179, 179, 255]],
    [144, [204, 61,  36,  255]],
    [150, [204, 83,  50,  255]],
    [159, [222, 134, 42,  255]],
    [168, [240, 217, 72,  255]],
    [177, [210, 227, 77,  255]],
    [186, [0,   201, 34,  255]],
    [195, [82,  227, 145, 255]],
    [205, [74,  235, 181, 255]],
    [214, [92,  186, 212, 255]],
    [236, [145, 45,  186, 255]],
    [254, [224, 168, 215, 255]],
]


# Escala de temperatura EXACTA (no lineal) provista por el usuario para
# "IR_custom": cada valor corresponde, EN EL MISMO ORDEN, al color de la
# misma posicion en IR_CUSTOM_RAW_PALETTE (17 temperaturas <-> 17 colores).
# OJO: los saltos NO son uniformes (30 C entre 10 y -20, pero solo 5 C entre
# -53 y -58); por eso no se puede derivar con la formula lineal posicion->temp
# que usa la paleta TRP original (esa si es lineal).
IR_CUSTOM_TEMPS = [
    40, 31, 20, 10, -20, -25, -30, -33, -38, -43,
    -48, -53, -58, -63, -68, -80, -90,
]


def ir_custom_cmap():
    """Devuelve (cmap, norm) de la paleta 'IR_custom', calibrada con la escala
    de temperatura EXACTA (no lineal) de IR_CUSTOM_TEMPS (mismo rango -90/+40 C
    que el producto 'ir', pero con puntos de quiebre distintos)."""
    pts = []
    for (_pos, (r, g, b, _a)), temp in zip(IR_CUSTOM_RAW_PALETTE, IR_CUSTOM_TEMPS):
        pts.append((temp, (r / 255.0, g / 255.0, b / 255.0)))

    pts.sort(key=lambda p: p[0])
    span = VMAX - VMIN
    norm_stops = [((t - VMIN) / span, c) for t, c in pts]
    eps = 1e-6
    for i in range(1, len(norm_stops)):
        if norm_stops[i][0] <= norm_stops[i - 1][0]:
            norm_stops[i] = (norm_stops[i - 1][0] + eps, norm_stops[i][1])
    norm_stops[0] = (0.0, norm_stops[0][1])
    norm_stops[-1] = (1.0, norm_stops[-1][1])

    cmap = LinearSegmentedColormap.from_list("IR_custom", norm_stops, N=256)
    cmap.set_bad("black")
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    return cmap, norm


# Nombres de colormaps personalizadas TRP, resolvibles via --cmap NOMBRE
# (ademas de los colormaps de matplotlib y las colortables de MetPy).
CUSTOM_CMAPS = {
    "IR_custom": ir_custom_cmap,
}



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



# =============================================================================
#     VAPOR DE AGUA NIVELES MEDIOS (producto "wv_medio": Banda 9, 6.9 um)
# =============================================================================
# El vapor de agua se mide como TEMPERATURA DE BRILLO (en C). El canal de 6.9 um
# "ve" la humedad de la media troposfera: aire seco (descendente) = mas calido,
# aire humedo / topes de nube = mas frio.
#
# Colortable: WVCIMSS (CIMSS, Univ. de Wisconsin) -> la paleta estandar de WV.
# Se obtiene desde MetPy (no esta registrada en matplotlib).
#
# Rango de brillo tipico del canal de WV (C): se realza ~ -75 (frio/humedo,
# topes) a +5 (calido/seco). Ajustable con WV_VMIN / WV_VMAX.

WV_VMIN = -75.0
WV_VMAX = 5.0
WV_TICKS = np.arange(-70.0, 6.0, 10.0)   # -70, -60, ..., 0


def midlevel_wv_cmap():
    """Devuelve (cmap, norm) para vapor de agua usando la colortable WVCIMSS."""
    from metpy.plots import ctables  # registrada en MetPy, no en matplotlib
    base = ctables.registry.get_colortable("WVCIMSS")
    # Copia para poder fijar el color de 'sin dato' sin alterar el original
    cmap = base.copy() if hasattr(base, "copy") else base
    cmap.set_bad("black")
    norm = Normalize(vmin=WV_VMIN, vmax=WV_VMAX)
    return cmap, norm



# =============================================================================
#     RESOLVEDOR DE COLORMAPS (matplotlib + colortables de MetPy)
# =============================================================================
# Permite usar tanto los colormaps de matplotlib (ej. "turbo", "rainbow") como
# las colortables propias de MetPy, que NO estan registradas en matplotlib.
# Las de MetPy se invocan con el prefijo "metpy_" para no chocar con nombres de
# matplotlib (ej. matplotlib tiene "rainbow" y MetPy tiene otra "rainbow"):
#
#     --cmap rainbow         -> rainbow de matplotlib
#     --cmap metpy_rainbow   -> rainbow de MetPy
#     --cmap metpy_ir_tpc    -> colortable IR del TPC (MetPy)
#
# Util tambien para --invert-cmap (agrega el reverso).

METPY_PREFIX = "metpy_"

# Colortables de MetPy mas utiles para satelite (sin el sufijo "_r").
SUGGESTED_METPY_CMAPS = [
    "rainbow", "ir_tpc", "ir_bd", "ir_rgbv", "ir_drgb", "ir_tv1",
    "WVCIMSS", "wv_tpc", "Carbone42",
]


def metpy_colortable_names():
    """Nombres de colortables de MetPy disponibles (excluye las variantes _r)."""
    from metpy.plots import ctables
    return sorted(n for n in ctables.registry if not n.endswith("_r"))


def resolve_cmap(name, invert=False):
    """Devuelve un Colormap de matplotlib por nombre, o None si no existe.

    Acepta colormaps de matplotlib, colortables de MetPy (prefijo 'metpy_')
    o paletas personalizadas TRP (ej. 'IR_custom').
    """
    cmap = None
    if name in CUSTOM_CMAPS:
        cmap, _norm = CUSTOM_CMAPS[name]()
    elif name.startswith(METPY_PREFIX):
        from metpy.plots import ctables
        key = name[len(METPY_PREFIX):]
        reg = ctables.registry
        if key in reg:
            cmap = reg.get_colortable(key)
    elif name in matplotlib.colormaps:
        cmap = matplotlib.colormaps[name]
    if cmap is None:
        return None
    cmap = cmap.copy()
    if invert:
        cmap = cmap.reversed()
    return cmap



# =============================================================================
#     DAY CLOUD PHASE DISTINCTION RGB (JMA / CIRA / EUMETSAT)
# =============================================================================
# RGB diurno para distinguir la fase de las nubes (liquido/hielo) y particulas.
# Receta estandar adaptada de JMA Himawari / CIRA GOES-R Quick Guides.
#
#   Rojo  = BT 10.3 um (Banda 13)        -> temperatura de brillo (C)
#   Verde = Reflectancia 1.6 um (Banda 5) -> % (0-100)
#   Azul  = Reflectancia 0.64 um (Banda 2) -> % (0-100)
#
# Interpretacion tipica:
#   - Nubes profundas convectivas (topes frios de hielo): rojo/naranja
#   - Nubes bajas/medias de agua liquida: verde/cyan
#   - Superficie despejada / nubes muy bajas: azul
#   - Nubes glaciadas: amarillo (mezcla rojo+verde)

DAY_CLOUD_PHASE_RED   = (-70.0, 60.0)   # BT 10.3 um (C)
DAY_CLOUD_PHASE_GREEN = (0.0, 60.0)     # Reflectancia 1.6 um (%)
DAY_CLOUD_PHASE_BLUE  = (0.0, 100.0)    # Reflectancia 0.64 um (%)


def day_cloud_phase_rgb(bt10_3, refl1_6, refl0_64):
    """Construye el RGB Day Cloud Phase Distinction.

    Entradas:
      - bt10_3: temperatura de brillo Banda 13 (C)
      - refl1_6: reflectancia Banda 5 (0-1) -> convertida internamente a %
      - refl0_64: reflectancia Banda 2 (0-1) -> convertida internamente a %

    Devuelve (H, W, 3) RGB normalizado [0, 1].
    """
    # Reflectancias de 0-1 a 0-100 (porcentaje)
    r5_pct = refl1_6 * 100.0
    r2_pct = refl0_64 * 100.0

    r = _norm_channel(bt10_3, *DAY_CLOUD_PHASE_RED)
    g = _norm_channel(r5_pct, *DAY_CLOUD_PHASE_GREEN)
    b = _norm_channel(r2_pct, *DAY_CLOUD_PHASE_BLUE)

    rgb = np.dstack([r, g, b])
    # Pixeles sin dato -> negro
    bad = ~(np.isfinite(bt10_3) & np.isfinite(refl1_6) & np.isfinite(refl0_64))
    rgb[bad] = 0.0
    return rgb



# =============================================================================
#   PALETA PERSONALIZADA TRP para el SANDWICH (alternativa a la rainbow)
# =============================================================================
# Reutiliza EXACTAMENTE la paleta original del usuario (cloudtop_cmap, la misma
# que usa el producto "ir"), pero con los pixeles enmascarados (topes calidos,
# por encima del umbral del sandwich) TRANSPARENTES en vez de negros, para que
# se siga viendo el canal visible debajo.

def custom_sandwich_cmap():
    """Paleta TRP original (la de cloudtop_cmap) para el overlay del sandwich."""
    cmap, norm = cloudtop_cmap()
    cmap = cmap.copy()
    cmap.set_bad((0, 0, 0, 0))  # transparente (deja ver el visible debajo)
    return cmap, norm


def ir_custom_sandwich_cmap():
    """Paleta 'IR_custom' (la segunda paleta personalizada) para el sandwich."""
    cmap, norm = ir_custom_cmap()
    cmap = cmap.copy()
    cmap.set_bad((0, 0, 0, 0))  # transparente (deja ver el visible debajo)
    return cmap, norm
