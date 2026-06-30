#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GOES-19 Nowcasting | TRP Meteorologia - Plots
=============================================
Plot de Temperatura de Topes de Nube (Banda 13, IR ~10.3 um) del satelite
GOES-19, con superposicion de rayos GLM (estrellas blancas), para nowcasting
sobre el Norte de Argentina / Region Pampeana / Misiones.

Caracteristicas:
  - Descarga ONLINE desde el bucket publico noaa-goes19 (AWS S3, anonimo).
  - Tiempo real (ultimo disponible) o fecha/hora especifica (UTC).
  - Salida PNG individual o GIF animado (varios tiempos) para nowcasting.
  - Estilo "recuadro blanco" clasico de MetPy.

Uso (ejemplos):
  # Imagen mas reciente, region completa
  python goes19_nowcasting.py

  # Una fecha/hora especifica (UTC) y solo Misiones
  python goes19_nowcasting.py --time "2026-01-15 18:30" --region misiones

  # Animacion GIF de los ultimos 8 cuadros
  python goes19_nowcasting.py --animate --frames 8 --region norte

Autor: TRP Meteorologia
"""

import os
import re
import sys
import glob
import argparse
import warnings
import datetime as dt

import numpy as np
import xarray as xr
import s3fs

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla (sandbox / servidor)
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import metpy  # noqa: F401  (habilita el accessor .metpy en xarray)

from colormaps import (cloudtop_cmap, COLORBAR_TICKS, VMIN, VMAX,
                       visible_cmap, VISIBLE_TICKS,
                       rainbow_ir_cmap, RAINBOW_IR_TICKS)
from estaciones import load_stations

warnings.filterwarnings("ignore")

# =============================================================================
#                              CONFIGURACION
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "output")
FONT_DIR = os.path.join(HERE, "fonts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

BUCKET = "noaa-goes19"
ABI_PRODUCT = "ABI-L2-CMIPF"   # Cloud & Moisture Imagery, Full Disk
GLM_PRODUCT = "GLM-L2-LCFA"    # Lightning Cluster-Filter Algorithm

# --- Marca / textos del plot ---
BRAND_TEXT = "TRP Meteorologia - Plots"
EXPERIMENTAL_TEXT = "PRODUCTO EXPERIMENTAL"

# --- Productos disponibles ----------------------------------------------------
# kind: "temp" (Kelvin -> Celsius) o "reflectance" (0-1, con correccion gamma).
PRODUCTS = {
    "ir": {
        "band": 13,
        "kind": "temp",
        "slug": "TopesNube",
        "title": "Temperatura de Topes de Nube (Banda 13)",
        "cbar_label": "Temperatura de Topes de Nube (\u00b0C)",
        "cmap_fn": cloudtop_cmap,
        "ticks": COLORBAR_TICKS,
        "extend": "both",
    },
    "visible": {
        "band": 2,
        "kind": "reflectance",
        "gamma": 2.0,            # aclara la imagen (reflectancia**(1/gamma))
        "slug": "Visible",
        "title": "Canal Visible (Banda 2, 0.64 \u00b5m)",
        "cbar_label": "Reflectancia (Canal Visible)",
        "cmap_fn": visible_cmap,
        "ticks": VISIBLE_TICKS,
        "extend": "neither",
    },
    "sandwich": {
        "band": 13,              # IR de overlay (el visible B2 es la base)
        "kind": "sandwich",
        "slug": "Sandwich",
        "title": "Sandwich (Visible + IR Banda 13)",
        "cbar_label": "Temperatura de Topes de Nube (\u00b0C)",
        "cmap_fn": rainbow_ir_cmap,
        "ticks": RAINBOW_IR_TICKS,
        "extend": "both",
        "ir_threshold": -20.0,   # solo colorea topes mas frios que esto
        "overlay_alpha": 0.85,   # transparencia del IR sobre el visible
    },
}
DEFAULT_PRODUCT = "ir"

# --- Colormaps sugeridas de matplotlib (cualquier nombre valido tambien sirve) ---
# Utiles para realces de satelite. Se pueden invertir con --invert-cmap.
SUGGESTED_CMAPS = [
    "rainbow", "jet", "turbo", "nipy_spectral", "gist_ncar", "gist_rainbow",
    "gist_stern", "Spectral", "coolwarm", "hsv", "cubehelix",
    "viridis", "inferno", "plasma", "magma", "cividis",
]

# --- Regiones predefinidas: extent [lon_min, lon_max, lat_min, lat_max] ---
REGIONS = {
    "completo":   [-68.0, -52.0, -40.0, -22.0],   # Pampeana + Norte + Misiones
    "pampeana":   [-66.0, -56.0, -40.0, -30.0],
    "norte":      [-67.0, -53.0, -30.0, -21.0],
    "misiones":   [-57.5, -53.2, -28.6, -25.2],
    "sur_brasil": [-58.0, -47.5, -34.0, -22.3],   # Parana + Santa Catarina + Rio Grande do Sul
    # Triple Frontera: N de Misiones + O de Santa Catarina + SO de Parana + E de Paraguay
    "triple_frontera": [-56.5, -51.8, -27.8, -24.0],
    "santa_catarina": [-54.2, -48.0, -29.6, -25.7],   # Estado de Santa Catarina (BR)
    "parana":         [-55.0, -47.8, -27.0, -22.2],   # Estado de Parana (BR)
}
DEFAULT_REGION = "completo"

# --- Ciudades (nombre, lon, lat). Se filtran a las que caen dentro del extent ---
CITIES = [
    # --- Argentina ---
    ("Buenos Aires", -58.38, -34.61),
    ("La Plata", -57.95, -34.92),
    ("Rosario", -60.64, -32.95),
    ("Cordoba", -64.18, -31.42),
    ("Santa Fe", -60.70, -31.63),
    ("Parana", -60.52, -31.73),
    ("Mar del Plata", -57.55, -38.00),
    ("Bahia Blanca", -62.27, -38.72),
    ("Santa Rosa", -64.29, -36.62),
    ("Mendoza", -68.84, -32.89),
    ("San Luis", -66.34, -33.30),
    ("Rio Cuarto", -64.35, -33.13),
    ("Venado Tuerto", -61.97, -33.75),
    ("San Nicolas", -60.22, -33.33),
    ("Rafaela", -61.49, -31.25),
    ("Concordia", -58.02, -31.39),
    ("Gualeguaychu", -58.51, -33.01),
    ("Reconquista", -59.65, -29.15),
    ("Goya", -59.26, -29.14),
    ("Paso de los Libres", -57.09, -29.71),
    ("Posadas", -55.90, -27.37),
    ("Obera", -55.12, -27.49),
    ("Eldorado", -54.62, -26.40),
    ("Puerto Iguazu", -54.57, -25.60),
    ("Bernardo de Irigoyen", -53.65, -26.25),
    ("San Vicente", -54.49, -26.99),
    ("Resistencia", -58.99, -27.46),
    ("Corrientes", -58.83, -27.47),
    ("Formosa", -58.17, -26.18),
    ("Sgo. del Estero", -64.26, -27.80),
    ("Tucuman", -65.22, -26.82),
    ("Salta", -65.41, -24.79),
    ("Catamarca", -65.78, -28.47),
    ("La Rioja", -66.86, -29.41),
    # --- Uruguay ---
    ("Montevideo (UY)", -56.16, -34.90),
    ("Salto (UY)", -57.96, -31.38),
    ("Paysandu (UY)", -58.08, -32.32),
    ("Rivera (UY)", -55.55, -30.90),
    # --- Paraguay ---
    ("Asuncion (PY)", -57.58, -25.30),
    ("Encarnacion (PY)", -55.87, -27.33),
    ("Ciudad del Este (PY)", -54.61, -25.51),
    ("Salto del Guaira (PY)", -54.31, -24.06),
    ("Hernandarias (PY)", -54.64, -25.40),
    # --- Brasil: Parana (PR) ---
    ("Curitiba (BR)", -49.27, -25.43),
    ("Foz do Iguacu (BR)", -54.59, -25.52),
    ("Cascavel (BR)", -53.46, -24.96),
    ("Francisco Beltrao (BR)", -53.06, -26.08),
    ("Pato Branco (BR)", -52.67, -26.23),
    ("Londrina (BR)", -51.16, -23.31),
    ("Maringa (BR)", -51.94, -23.42),
    ("Ponta Grossa (BR)", -50.16, -25.09),
    # --- Brasil: Santa Catarina (SC) ---
    ("Florianopolis (BR)", -48.55, -27.59),
    ("Joinville (BR)", -48.85, -26.30),
    ("Blumenau (BR)", -49.07, -26.92),
    ("Chapeco (BR)", -52.62, -27.10),
    ("Sao Miguel do Oeste (BR)", -53.52, -26.72),
    ("Dionisio Cerqueira (BR)", -53.63, -26.26),
    ("Lages (BR)", -50.33, -27.82),
    ("Criciuma (BR)", -49.37, -28.68),
    # --- Brasil: Rio Grande do Sul (RS) ---
    ("Porto Alegre (BR)", -51.23, -30.03),
    ("Caxias do Sul (BR)", -51.18, -29.17),
    ("Passo Fundo (BR)", -52.41, -28.26),
    ("Santa Maria (BR)", -53.81, -29.68),
    ("Pelotas (BR)", -52.34, -31.77),
    ("Rio Grande (BR)", -52.10, -32.03),
    ("Uruguaiana (BR)", -57.09, -29.76),
    ("Sant. do Livramento (BR)", -55.53, -30.89),
]

# --- Acumulacion de rayos GLM: minutos previos al tiempo de la imagen ---
GLM_MINUTES = 10

# --- Estaciones SMN ---
STATION_CATALOG = os.path.join(HERE, "estaciones_smn.txt")  # catalogo (lat/lon)


def _find_obs_file():
    """Devuelve el archivo de observaciones 'estado_tiempo*.txt' mas reciente."""
    files = sorted(glob.glob(os.path.join(HERE, "estado_tiempo*.txt")))
    return files[-1] if files else None

# =============================================================================
#                              FUENTES (Open Sans)
# =============================================================================
_FONT_REG_PATH = os.path.join(FONT_DIR, "OpenSans-Regular.ttf")
_FONT_BOLD_PATH = os.path.join(FONT_DIR, "OpenSans-Bold.ttf")
for _p in (_FONT_REG_PATH, _FONT_BOLD_PATH):
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)

# Se referencian por archivo (las TTF estaticas no traen bien el peso bold)
FONT_REG = fm.FontProperties(fname=_FONT_REG_PATH) if os.path.exists(_FONT_REG_PATH) else fm.FontProperties()
FONT_BOLD = fm.FontProperties(fname=_FONT_BOLD_PATH) if os.path.exists(_FONT_BOLD_PATH) else fm.FontProperties(weight="bold")

# =============================================================================
#                       ACCESO A DATOS (AWS S3 anonimo)
# =============================================================================
_FS = s3fs.S3FileSystem(anon=True)


def _parse_start_time(key: str) -> dt.datetime:
    """Extrae el tiempo de inicio de escaneo (campo _sYYYYJJJHHMMSSf_) del nombre."""
    m = re.search(r"_s(\d{14})", key)
    s = m.group(1)
    year, doy = int(s[0:4]), int(s[4:7])
    hh, mm, ss, tenth = int(s[7:9]), int(s[9:11]), int(s[11:13]), int(s[13])
    return (dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
            + dt.timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss,
                           milliseconds=100 * tenth))


def _list_hour(product: str, when: dt.datetime, band: int | None = None):
    """Lista las keys de un producto para una hora dada (UTC)."""
    doy = when.timetuple().tm_yday
    prefix = f"{BUCKET}/{product}/{when.year}/{doy:03d}/{when.hour:02d}/"
    try:
        keys = _FS.ls(prefix)
    except FileNotFoundError:
        return []
    if band is not None:
        keys = [k for k in keys if f"C{band:02d}_" in k]
    return keys


def _download(key: str) -> str:
    """Descarga una key de S3 a DATA_DIR (con cache). Devuelve la ruta local."""
    local = os.path.join(DATA_DIR, key.split("/")[-1])
    if not os.path.exists(local):
        _FS.get(key, local)
    return local


def find_abi(target: dt.datetime, band: int) -> tuple[str, dt.datetime]:
    """Encuentra el archivo ABI de la banda dada con inicio de escaneo mas cercano a target."""
    candidates = []
    for off in (-1, 0, 1):  # hora previa, actual y siguiente
        hour = target + dt.timedelta(hours=off)
        for k in _list_hour(ABI_PRODUCT, hour, band=band):
            candidates.append((k, _parse_start_time(k)))
    if not candidates:
        raise RuntimeError(f"No se hallaron archivos ABI C{band:02d} cerca de {target:%Y-%m-%d %H:%M} UTC")
    # Solo escaneos ya completados (<= target + 1 min de tolerancia)
    valid = [(k, t) for k, t in candidates if t <= target + dt.timedelta(minutes=1)] or candidates
    key, t_scan = min(valid, key=lambda kt: abs((kt[1] - target).total_seconds()))
    return _download(key), t_scan


def load_glm_flashes(t_scan: dt.datetime, minutes: int = GLM_MINUTES):
    """Acumula lat/lon de rayos GLM en una ventana [t_scan - minutes, t_scan]."""
    t0 = t_scan - dt.timedelta(minutes=minutes)
    lats, lons = [], []
    hours = {t0, t_scan}
    hours.add(t0 + dt.timedelta(hours=1))  # por si la ventana cruza una hora
    for hour in sorted(hours):
        for k in _list_hour(GLM_PRODUCT, hour):
            ts = _parse_start_time(k)
            if t0 <= ts <= t_scan:
                try:
                    g = xr.open_dataset(_download(k), engine="netcdf4")
                    if int(g.sizes.get("number_of_flashes", 0)) > 0:
                        lats.append(g["flash_lat"].values)
                        lons.append(g["flash_lon"].values)
                    g.close()
                except Exception:
                    continue
    if lats:
        return np.concatenate(lons), np.concatenate(lats)
    return np.array([]), np.array([])


# =============================================================================
#                          PROCESAMIENTO ABI
# =============================================================================
def load_abi_subset(path: str, extent, kind="temp", gamma=1.0, margin=1.5):
    """Abre el ABI, recorta a la region (lectura perezosa) y devuelve (data, extent_m, geos_crs, t_scan).

    kind="temp"        -> convierte Kelvin a Celsius.
    kind="reflectance" -> recorta a 0-1 y aplica correccion gamma.
    """
    ds = xr.open_dataset(path, engine="netcdf4")
    dat = ds.metpy.parse_cf("CMI")
    geos = dat.metpy.cartopy_crs

    lon0, lon1, lat0, lat1 = extent
    # Muestreo del borde de la region para hallar los limites en x/y (metros)
    n = 25
    edge_lon = np.concatenate([
        np.linspace(lon0, lon1, n), np.linspace(lon0, lon1, n),
        np.full(n, lon0), np.full(n, lon1)])
    edge_lat = np.concatenate([
        np.full(n, lat0), np.full(n, lat1),
        np.linspace(lat0, lat1, n), np.linspace(lat0, lat1, n)])
    pts = geos.transform_points(ccrs.PlateCarree(), edge_lon, edge_lat)
    xs, ys = pts[:, 0], pts[:, 1]
    xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
    mx = (xs.max() - xs.min()) * (margin - 1) / 2
    my = (ys.max() - ys.min()) * (margin - 1) / 2
    xmin, xmax = xs.min() - mx, xs.max() + mx
    ymin, ymax = ys.min() - my, ys.max() + my

    x = dat["x"].values
    y = dat["y"].values
    ix = np.where((x >= xmin) & (x <= xmax))[0]
    iy = np.where((y >= ymin) & (y <= ymax))[0]
    # isel + .values lee SOLO la porcion necesaria del disco (clave para Banda 2)
    sub = dat.isel(x=slice(int(ix.min()), int(ix.max()) + 1),
                   y=slice(int(iy.min()), int(iy.max()) + 1))

    raw = sub.values
    if kind == "temp":
        data = raw - 273.15  # Kelvin -> Celsius
    else:  # reflectance
        data = np.clip(raw, 0.0, 1.0)
        if gamma and gamma != 1.0:
            data = np.power(data, 1.0 / gamma)

    ext_m = [float(sub["x"].min()), float(sub["x"].max()),
             float(sub["y"].min()), float(sub["y"].max())]
    t_scan = _parse_start_time(os.path.basename(path))
    ds.close()
    return data, ext_m, geos, t_scan


# =============================================================================
#                              FEATURES DEL MAPA
# =============================================================================
def _provinces():
    return cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "10m",
        edgecolor="#666666", facecolor="none")


def _roads():
    return cfeature.NaturalEarthFeature(
        "cultural", "roads", "10m",
        edgecolor="#d98c00", facecolor="none")


# =============================================================================
#                   MODELO DE ESTACION (station plot)
# =============================================================================
def _circle_outline():
    """Marcador de circulo (contorno) de radio unitario."""
    return MplPath.unit_circle()


def _wedge_marker(oktas):
    """Marcador tipo 'pastel' relleno segun octas (0-8) para nubosidad WMO."""
    frac = max(0.0, min(oktas / 8.0, 1.0))
    if frac <= 0:
        return None
    # Cuna desde arriba (90 grados) en sentido horario
    return MplPath.wedge(90 - 360 * frac, 90)


def plot_stations(ax, stations, extent, colored=False):
    """Dibuja el modelo de estacion (nubosidad + barba de viento + T/Td/presion).

    Dibujo con matplotlib nativo (evita el StationPlot de MetPy, incompatible con
    matplotlib reciente). Textos en Open Sans Bold con contorno negro.
    colored=False -> todo blanco. colored=True -> T en rojo, Td en verde, P en blanco.
    """
    sel = [s for s in stations
           if extent[0] <= s["lon"] <= extent[1] and extent[2] <= s["lat"] <= extent[3]]
    if not sel:
        return

    # Colores de los valores
    if colored:
        c_temp, c_dew, c_pres = "#ff3b3b", "#36e23a", "#ffffff"
    else:
        c_temp = c_dew = c_pres = "#ffffff"

    stroke = [pe.withStroke(linewidth=1.6, foreground="black")]
    pc = ccrs.PlateCarree()
    lons = np.array([s["lon"] for s in sel])
    lats = np.array([s["lat"] for s in sel])

    # --- Circulo de nubosidad: contorno + relleno proporcional a las octas ---
    circ = ax.scatter(lons, lats, transform=pc, marker=_circle_outline(), s=150,
                      facecolors="none", edgecolors="white", linewidths=1.1, zorder=9)
    circ.set_path_effects(stroke)
    # Relleno por grupos de octas
    by_okta = {}
    for s in sel:
        ok = s["sky_oktas"]
        if ok and ok > 0:
            by_okta.setdefault(min(ok, 8), []).append((s["lon"], s["lat"]))
    for ok, pts in by_okta.items():
        wm = _wedge_marker(ok)
        if wm is None:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        fill = ax.scatter(xs, ys, transform=pc, marker=wm, s=150,
                          facecolors="white", edgecolors="none", zorder=9)
        fill.set_path_effects(stroke)

    # --- Barba de viento (en nudos) ---
    mask = np.array([s["has_wind"] for s in sel])
    if mask.any():
        u = np.array([s["u_kt"] for s in sel])
        v = np.array([s["v_kt"] for s in sel])
        b = ax.barbs(lons[mask], lats[mask], u[mask], v[mask], transform=pc,
                     length=6.5, linewidth=0.9, color="white", zorder=9)
        b.set_path_effects(stroke)

    # --- Valores: T arriba-izq, Td abajo-izq, presion arriba-der ---
    for s in sel:
        px, py = ax.projection.transform_point(s["lon"], s["lat"], pc)
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        common = dict(textcoords="offset points", fontproperties=FONT_BOLD,
                      fontsize=8.5, path_effects=stroke, zorder=10, clip_on=True)
        t, td, p = s["temp_c"], s["dewpoint_c"], s["pressure_hpa"]
        if np.isfinite(t):
            ax.annotate(f"{t:.0f}", (px, py), xytext=(-10, 7),
                        ha="right", va="bottom", color=c_temp, **common)
        if np.isfinite(td):
            ax.annotate(f"{td:.0f}", (px, py), xytext=(-10, -9),
                        ha="right", va="top", color=c_dew, **common)
        if np.isfinite(p):
            code = f"{int(round(p * 10)) % 1000:03d}"  # presion codificada (3 digitos)
            ax.annotate(code, (px, py), xytext=(10, 7),
                        ha="left", va="bottom", color=c_pres, **common)


# =============================================================================
#                              PLOTEO PRINCIPAL
# =============================================================================
def make_plot(data, ext_m, geos, t_scan, extent, glm_lon, glm_lat,
              out_path, product, show_cities=True, stations=None,
              station_color=False, base_data=None, base_ext=None,
              cmap_name=None, invert_cmap=False):
    cmap, norm = product["cmap_fn"]()
    # Override de colormap (--cmap) e inversion (--invert-cmap)
    if cmap_name:
        cmap = matplotlib.colormaps[cmap_name].copy()
    if invert_cmap:
        cmap = cmap.reversed()
    cmap.set_bad((0, 0, 0, 0) if product.get("kind") == "sandwich" else "black")

    fig = plt.figure(figsize=(10, 11.3), facecolor="white")

    # --- Recuadro blanco exterior (estilo MetPy) ---
    fig.add_artist(Rectangle((0.015, 0.015), 0.97, 0.97, transform=fig.transFigure,
                             fill=False, edgecolor="black", linewidth=1.6, zorder=10))

    # --- Eje del mapa ---
    # Mismo ancho (0.87) y borde izquierdo (0.065); panel mas alto hacia abajo
    # (borde inferior baja de 0.205 a 0.160, justo por encima del colorbar).
    ax = fig.add_axes([0.065, 0.160, 0.87, 0.745], projection=geos)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    # Capa base del sandwich: canal visible en escala de grises
    if base_data is not None and base_ext is not None:
        vcmap, vnorm = visible_cmap()
        ax.imshow(base_data, origin="upper", extent=base_ext, transform=geos,
                  cmap=vcmap, norm=vnorm, interpolation="nearest", zorder=0)

    # Dato de satelite (overlay). En sandwich, enmascara los topes calidos.
    plot_data = data
    thr = product.get("ir_threshold")
    if thr is not None:
        plot_data = np.ma.masked_greater(data, thr)
    ax.imshow(plot_data, origin="upper", extent=ext_m, transform=geos,
              cmap=cmap, norm=norm, interpolation="nearest",
              alpha=product.get("overlay_alpha", 1.0), zorder=1)

    # Geografia
    ax.add_feature(_roads(), linewidth=0.5, zorder=3, alpha=0.9)
    ax.add_feature(_provinces(), linewidth=0.6, zorder=4)
    ax.add_feature(cfeature.BORDERS, edgecolor="white", linewidth=0.9, zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor="white", linewidth=0.9, zorder=5)

    # Rayos GLM como estrellas blancas
    if glm_lon.size:
        m = ((glm_lon >= extent[0]) & (glm_lon <= extent[1])
             & (glm_lat >= extent[2]) & (glm_lat <= extent[3]))
        if m.any():
            ax.scatter(glm_lon[m], glm_lat[m], transform=ccrs.PlateCarree(),
                       marker="*", s=90, c="white", edgecolors="black",
                       linewidths=0.4, zorder=8)

    # Cuadricula lat/lon con numeros pequenos al costado
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color="white", alpha=0.35, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7, "color": "#222222"}
    gl.ylabel_style = {"size": 7, "color": "#222222"}

    # Ciudades + etiquetas en Open Sans Bold
    if show_cities:
        stroke = [pe.withStroke(linewidth=1.8, foreground="black")]
        for name, lon, lat in CITIES:
            if extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]:
                ax.plot(lon, lat, marker="o", markersize=3.2, color="white",
                        markeredgecolor="black", markeredgewidth=0.6,
                        transform=ccrs.PlateCarree(), zorder=7)
                ax.text(lon + 0.06, lat + 0.04, name, transform=ccrs.PlateCarree(),
                        fontproperties=FONT_BOLD, fontsize=7.5, color="white",
                        path_effects=stroke, zorder=7, va="bottom", ha="left")

    # Modelo de estacion SMN (nubosidad + viento + T/Td/presion)
    if stations:
        plot_stations(ax, stations, extent, colored=station_color)

    # --- Titulo ---
    fig.text(0.5, 0.945, f"GOES-19  \u00b7  {product['title']}",
             ha="center", va="center", fontproperties=FONT_BOLD, fontsize=15,
             color="black")
    fig.text(0.5, 0.918, f"{t_scan:%Y-%m-%d  %H:%M} UTC",
             ha="center", va="center", fontproperties=FONT_REG, fontsize=11,
             color="#333333")

    # --- Barra de color ---
    cax = fig.add_axes([0.16, 0.115, 0.68, 0.018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation="horizontal", extend=product["extend"])
    cb.set_ticks(product["ticks"])
    cb.ax.tick_params(labelsize=8, length=3)
    for lbl in cb.ax.get_xticklabels():
        lbl.set_fontproperties(FONT_REG)
    cb.outline.set_linewidth(0.8)

    # "Producto experimental" en rojo pequeno ARRIBA de la barra
    fig.text(0.16, 0.142, EXPERIMENTAL_TEXT, ha="left", va="center",
             fontproperties=FONT_BOLD, fontsize=8, color="red")

    # Etiqueta de la variable ABAJO de la barra
    fig.text(0.5, 0.082, product["cbar_label"], ha="center", va="center",
             fontproperties=FONT_BOLD, fontsize=11, color="black")

    # --- Marca abajo a la derecha ---
    fig.text(0.935, 0.038, BRAND_TEXT, ha="right", va="center",
             fontproperties=FONT_REG, fontsize=10, color="#444444")
    # GLM leyenda abajo izquierda
    fig.text(0.065, 0.038, "\u2605  Rayos GLM", ha="left", va="center",
             fontproperties=FONT_REG, fontsize=9, color="#444444")

    fig.savefig(out_path, dpi=130, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return out_path


# =============================================================================
#                              ORQUESTACION
# =============================================================================
def render_one(target: dt.datetime, region: str, glm_minutes: int,
               product: str = DEFAULT_PRODUCT, tag: str = None,
               stations=None, station_color=False,
               cmap_name=None, invert_cmap=False):
    prod = PRODUCTS[product]
    extent = REGIONS[region]
    base_data = base_ext = None
    if prod["kind"] == "sandwich":
        # Overlay IR (Banda 13) + base Visible (Banda 2)
        ir_path, t_scan = find_abi(target, 13)
        data, ext_m, geos, t_scan = load_abi_subset(ir_path, extent, kind="temp")
        vis_path, _ = find_abi(target, 2)
        base_data, base_ext, _g, _t = load_abi_subset(
            vis_path, extent, kind="reflectance", gamma=2.0)
    else:
        abi_path, t_scan = find_abi(target, prod["band"])
        data, ext_m, geos, t_scan = load_abi_subset(
            abi_path, extent, kind=prod["kind"], gamma=prod.get("gamma", 1.0))
    glm_lon, glm_lat = load_glm_flashes(t_scan, glm_minutes)
    tag = tag or f"{t_scan:%Y%m%d_%H%M}"
    out_path = os.path.join(OUT_DIR, f"GOES19_{prod['slug']}_{region}_{tag}.png")
    make_plot(data, ext_m, geos, t_scan, extent, glm_lon, glm_lat, out_path, prod,
              stations=stations, station_color=station_color,
              base_data=base_data, base_ext=base_ext,
              cmap_name=cmap_name, invert_cmap=invert_cmap)
    n_est = len([s for s in stations if extent[0] <= s["lon"] <= extent[1]
                 and extent[2] <= s["lat"] <= extent[3]]) if stations else 0
    print(f"  [OK] {out_path}  ({t_scan:%H:%M} UTC, {glm_lon.size} rayos, {n_est} estaciones)")
    return out_path


def render_animation(end: dt.datetime, region: str, glm_minutes: int,
                     frames: int, step_min: int, interval_ms: int,
                     product: str = DEFAULT_PRODUCT, stations=None,
                     station_color=False, cmap_name=None, invert_cmap=False):
    """Genera varios PNG y los combina en un GIF."""
    pngs = []
    print(f"Generando {frames} cuadros (cada {step_min} min)...")
    for i in range(frames - 1, -1, -1):
        target = end - dt.timedelta(minutes=step_min * i)
        try:
            pngs.append(render_one(target, region, glm_minutes, product,
                                   tag=f"frame_{frames - 1 - i:02d}",
                                   stations=stations, station_color=station_color,
                                   cmap_name=cmap_name, invert_cmap=invert_cmap))
        except Exception as e:
            print(f"  [skip] {target:%H:%M} UTC -> {e}")
    if len(pngs) < 2:
        print("No hay suficientes cuadros para animar.")
        return None
    try:
        from PIL import Image
    except ImportError:
        print("Pillow no instalado; se generaron PNGs pero no el GIF.")
        return None
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    slug = PRODUCTS[product]["slug"]
    gif_path = os.path.join(OUT_DIR, f"GOES19_{slug}_{region}_anim.gif")
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=interval_ms, loop=0)
    print(f"[GIF] {gif_path}")
    return gif_path


def print_listing():
    """Imprime el listado de variables, regiones y colormaps disponibles."""
    line = "=" * 64
    print(line)
    print("  GOES-19 Nowcasting (TRP Meteorologia) - Opciones disponibles")
    print(line)
    print("\nVARIABLES / PRODUCTOS  (--product NOMBRE):")
    for k, p in PRODUCTS.items():
        marca = "  (default)" if k == DEFAULT_PRODUCT else ""
        print(f"   {k:9s} -> {p['title']}{marca}")
    print("\nREGIONES  (--region NOMBRE):")
    for k, e in REGIONS.items():
        print(f"   {k:16s} lon[{e[0]:.1f}, {e[1]:.1f}]  lat[{e[2]:.1f}, {e[3]:.1f}]")
    print("\nCOLORMAPS sugeridas  (--cmap NOMBRE):")
    for i in range(0, len(SUGGESTED_CMAPS), 4):
        print("   " + "  ".join(f"{c:14s}" for c in SUGGESTED_CMAPS[i:i + 4]))
    print("   (Tambien sirve cualquier otro colormap valido de matplotlib.)")
    print("   --invert-cmap  -> invierte el colormap elegido (o el del producto).")
    print("\nOTRAS OPCIONES:")
    print("   --estaciones           Superpone el modelo de estaciones del SMN.")
    print("   --estaciones-color     Estaciones coloreadas (T roja, Td verde).")
    print("   --animate              Genera un GIF animado (--frames, --step).")
    print("   --time \"YYYY-MM-DD HH:MM\"  Fecha/hora UTC (default: lo mas reciente).")
    print("\nEJEMPLOS:")
    print("   python goes19_nowcasting.py --list")
    print("   python goes19_nowcasting.py --product ir --region norte --cmap turbo")
    print("   python goes19_nowcasting.py --product ir --cmap rainbow --invert-cmap")
    print("   python goes19_nowcasting.py --product sandwich --region triple_frontera \\")
    print("          --estaciones --estaciones-color")
    print(line)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GOES-19 Nowcasting (TRP Meteorologia)")
    p.add_argument("--time", default=None,
                   help='Fecha/hora UTC "YYYY-MM-DD HH:MM". Por defecto: ultimo disponible.')
    p.add_argument("--region", default=DEFAULT_REGION, choices=list(REGIONS),
                   help="Region a plotear.")
    p.add_argument("--product", default=DEFAULT_PRODUCT, choices=list(PRODUCTS),
                   help="Producto: ir (Topes de Nube B13), visible (B2) o sandwich.")
    p.add_argument("--cmap", default=None,
                   help="Colormap de matplotlib a usar (ver --list). Default: el del producto.")
    p.add_argument("--invert-cmap", action="store_true",
                   help="Invierte el colormap (equivale a agregar '_r').")
    p.add_argument("--list", action="store_true",
                   help="Lista variables, regiones y colormaps disponibles y sale.")
    p.add_argument("--glm-window", type=int, default=GLM_MINUTES,
                   help="Minutos de rayos GLM a acumular (default 10).")
    p.add_argument("--estaciones", action="store_true",
                   help="Superponer modelo de estaciones del SMN.")
    p.add_argument("--estaciones-color", action="store_true",
                   help="Estaciones coloreadas (T en rojo, Td en verde, P en blanco).")
    p.add_argument("--obs-file", default=None,
                   help="Archivo de observaciones SMN (default: el estado_tiempo*.txt mas reciente).")
    p.add_argument("--animate", action="store_true", help="Generar GIF animado.")
    p.add_argument("--frames", type=int, default=6, help="Cuadros del GIF.")
    p.add_argument("--step", type=int, default=10, help="Minutos entre cuadros (ABI=10).")
    p.add_argument("--interval", type=int, default=500, help="ms por cuadro del GIF.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Listado de opciones y salir
    if args.list:
        print_listing()
        return

    # Validar colormap
    if args.cmap and args.cmap not in matplotlib.colormaps:
        print(f"[error] El colormap '{args.cmap}' no existe en matplotlib.")
        print("        Usa --list para ver las colormaps sugeridas.")
        return

    if args.time:
        target = dt.datetime.strptime(args.time, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
    else:
        target = dt.datetime.now(dt.timezone.utc)

    print(f"GOES-19 | producto={args.product} | region={args.region} | "
          f"objetivo={target:%Y-%m-%d %H:%M} UTC")

    stations = None
    if args.estaciones:
        obs_file = args.obs_file or _find_obs_file()
        if obs_file and os.path.exists(STATION_CATALOG):
            stations = load_stations(STATION_CATALOG, obs_file)
            print(f"  Estaciones SMN: {len(stations)} cargadas desde {os.path.basename(obs_file)}")
        else:
            print("  [aviso] No se encontro el archivo de observaciones o el catalogo; "
                  "se omiten las estaciones.")

    if args.animate:
        render_animation(target, args.region, args.glm_window,
                         args.frames, args.step, args.interval, args.product,
                         stations=stations, station_color=args.estaciones_color,
                         cmap_name=args.cmap, invert_cmap=args.invert_cmap)
    else:
        render_one(target, args.region, args.glm_window, args.product,
                   stations=stations, station_color=args.estaciones_color,
                   cmap_name=args.cmap, invert_cmap=args.invert_cmap)


if __name__ == "__main__":
    main()
