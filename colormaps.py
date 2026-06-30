"""
colormaps.py
============
Colormap personalizado para la Temperatura de Topes de Nube (Banda 13, IR ~10.3 um).

>>> ESTE ES EL ARCHIVO QUE DEBES EDITAR PARA PONER TU COLORMAP. <<<

El plot mapea la temperatura (en grados Celsius) usando:
    - VMIN / VMAX  : rango de temperatura cubierto por la barra de color.
    - cloudtop_cmap(): devuelve (cmap, norm) de matplotlib.

Por defecto se incluye un "enhancement" IR clasico:
    - Parte CALIDA  (de VMAX hasta TEMP_GRIS):  escala de grises (negro = calido).
    - Parte FRIA    (de TEMP_GRIS hasta VMIN):  realce de color para topes convectivos.

Cuando tengas tu colormap definitivo, reemplaza la lista `STOPS` (o toda la
funcion `cloudtop_cmap`) manteniendo la misma firma de retorno: (cmap, norm).
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# --- Rango de la barra de color (grados Celsius) -----------------------------
VMIN = -90.0   # tope mas frio representado
VMAX = 50.0    # superficie mas calida representada
TEMP_GRIS = -30.0  # a partir de aqui hacia lo calido va en escala de grises


def _t(temp_c: float) -> float:
    """Normaliza una temperatura (C) a la posicion [0, 1] dentro de [VMIN, VMAX]."""
    return (temp_c - VMIN) / (VMAX - VMIN)


# --- Puntos de control del colormap (temperatura_C, color) -------------------
# EDITA AQUI tu realce de color. Orden: de mas frio (VMIN) a mas calido (VMAX).
_STOPS_C = [
    (-90.0, "#ffffff"),  # topes extremadamente frios -> blanco
    (-80.0, "#c800c8"),  # magenta
    (-70.0, "#ff0000"),  # rojo
    (-60.0, "#ff9000"),  # naranja
    (-50.0, "#ffff00"),  # amarillo
    (-40.0, "#00c800"),  # verde
    (-30.0, "#ffffff"),  # transicion a grises (blanco)
    (50.0,  "#000000"),  # superficie calida -> negro
]


def cloudtop_cmap():
    """Devuelve (cmap, norm) para la temperatura de topes de nube en C."""
    stops = sorted(_STOPS_C, key=lambda s: s[0])
    positions = [_t(temp) for temp, _ in stops]
    colors = [c for _, c in stops]

    # Evita posiciones duplicadas exactas (matplotlib lo exige estrictamente creciente)
    eps = 1e-6
    for i in range(1, len(positions)):
        if positions[i] <= positions[i - 1]:
            positions[i] = positions[i - 1] + eps

    cmap = LinearSegmentedColormap.from_list(
        "cloudtop_trp", list(zip(positions, colors)), N=256
    )
    cmap.set_bad("black")  # pixeles sin dato
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    return cmap, norm


# Marcas (ticks) sugeridas para la barra de color (C)
COLORBAR_TICKS = np.arange(VMIN, VMAX + 1, 20)
