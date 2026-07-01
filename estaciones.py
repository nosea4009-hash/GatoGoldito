"""
estaciones.py
=============
Lector de datos de estaciones del SMN (Servicio Meteorologico Nacional, Argentina)
para superponer modelos de estacion (station plots) sobre el satelite.

Combina dos archivos:
  1. Catalogo de estaciones (ancho fijo): nombre + lat/lon (grados y minutos).
  2. Observaciones "estado del tiempo" (separado por ';'): condiciones actuales.

Ambos archivos vienen en codificacion latin-1 (ISO-8859-1).

load_stations(catalogo, observaciones) -> lista de dicts con:
    name, lat, lon, temp_c, dewpoint_c, pressure_hpa,
    sky_oktas (0-8 o None), u_kt, v_kt (componentes de viento en nudos), has_wind
"""

import re
import os
import io
import math
import zipfile
import unicodedata
import urllib.request

ENCODING = "latin-1"
KMH_TO_KT = 0.539957  # km/h -> nudos

# --- Datos abiertos del SMN: "tiempo presente" (estado del tiempo actual) -----
# Devuelve un ZIP que contiene 'estado_tiempo{YYYYMMDD}.txt' con las
# observaciones mas recientes de las estaciones. Fuente: SMN (datos abiertos).
SMN_TIEPRE_URL = "https://ssl.smn.gob.ar/dpd/zipopendata.php?dato=tiepre"
_USER_AGENT = "Mozilla/5.0 (GOES19-Nowcasting; TRP Meteorologia)"


def download_latest_obs(dest_dir, timeout=30):
    """Descarga el 'tiempo presente' del SMN y extrae el estado_tiempo*.txt.

    Devuelve la ruta local del .txt extraido. Lanza excepcion si algo falla
    (el llamador decide como manejarlo, ej. usar un archivo local previo).
    """
    req = urllib.request.Request(SMN_TIEPRE_URL, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not txt_names:
            raise RuntimeError("El ZIP del SMN no contiene ningun .txt de observaciones.")
        name = txt_names[0]
        out_path = os.path.join(dest_dir, os.path.basename(name))
        with zf.open(name) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    return out_path

# Direcciones de viento (espanol) -> grados meteorologicos (desde donde sopla)
WIND_DIRS = {
    "NORTE": 0.0, "NORESTE": 45.0, "ESTE": 90.0, "SUDESTE": 135.0,
    "SUR": 180.0, "SUDOESTE": 225.0, "OESTE": 270.0, "NOROESTE": 315.0,
}

# Sufijos a quitar del nombre del catalogo para emparejar con las observaciones
_SUFIJOS = (" AERO", " AERODROMO", " AEROPUERTO", " OBSERVATORIO",
            " OBS.", " UNIVERSIDAD NACIONAL", " (EX JUBANY)")


def _norm(s: str) -> str:
    """Normaliza un nombre: sin acentos, mayusculas, sin sufijos ni simbolos."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.upper().strip()
    for suf in _SUFIJOS:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return re.sub(r"[^A-Z0-9]", "", s)


def _sky_to_oktas(estado: str):
    """Convierte el estado del cielo (texto) a octas de nubosidad (0-8)."""
    e = unicodedata.normalize("NFKD", estado).encode("ascii", "ignore").decode().upper().strip()
    if e.startswith("DESPEJADO"):
        return 0
    if e.startswith("ALGO NUBLADO"):
        return 2
    if e.startswith("PARCIALMENTE NUBLADO"):
        return 4
    if e.startswith("CUBIERTO"):
        return 8
    if e.startswith("NUBLADO"):
        return 6
    return None


def _to_float(s):
    try:
        return float(str(s).strip().replace(",", "."))
    except (ValueError, TypeError):
        return float("nan")


def load_catalog(path: str) -> dict:
    """Lee el catalogo (ancho fijo) -> {nombre_normalizado: (lat, lon)}."""
    cat = {}
    for line in open(path, encoding=ENCODING).read().splitlines()[2:]:
        if not line.strip():
            continue
        nombre = line[0:31].strip()
        toks = line[31:].split()
        # Primer entero con signo = grados de latitud; siguientes 3 = latmin, londeg, lonmin
        idx = next((i for i, t in enumerate(toks) if re.match(r"^-?\d+$", t)), None)
        if idx is None or idx + 3 >= len(toks):
            continue
        latd, latm = int(toks[idx]), int(toks[idx + 1])
        lond, lonm = int(toks[idx + 2]), int(toks[idx + 3])
        lat = (abs(latd) + latm / 60.0) * (-1 if latd < 0 else 1)
        lon = (abs(lond) + lonm / 60.0) * (-1 if lond < 0 else 1)
        cat[_norm(nombre)] = (lat, lon)
    return cat


def _parse_wind(campo: str):
    """ 'Norte  20' -> (dir_deg, speed_kmh). 'Calma'/'Variables' -> (None, speed)."""
    toks = campo.split()
    if not toks:
        return None, 0.0
    speed = 0.0
    dir_text = campo
    if re.match(r"^\d+(\.\d+)?$", toks[-1]):
        speed = float(toks[-1])
        dir_text = " ".join(toks[:-1])
    key = unicodedata.normalize("NFKD", dir_text).encode("ascii", "ignore").decode().upper().strip()
    return WIND_DIRS.get(key), speed


def load_stations(catalog_path: str, obs_path: str) -> list:
    """Combina catalogo + observaciones -> lista de estaciones con datos para plotear."""
    cat = load_catalog(catalog_path)
    stations = []
    for line in open(obs_path, encoding=ENCODING).read().splitlines():
        if not line.strip():
            continue
        f = line.split(";")
        if len(f) < 10:
            continue
        nombre = f[0].strip()
        coords = cat.get(_norm(nombre))
        if coords is None:
            continue  # sin lat/lon -> no se puede ubicar
        lat, lon = coords

        temp = _to_float(f[5])
        dew = _to_float(f[6])  # "No se calcula" -> NaN
        pres = _to_float(f[9].split("/")[0])
        sky = _sky_to_oktas(f[3])

        dir_deg, speed_kmh = _parse_wind(f[8])
        u = v = 0.0
        has_wind = False
        if dir_deg is not None and speed_kmh > 0:
            spd_kt = speed_kmh * KMH_TO_KT
            rad = math.radians(dir_deg)
            u = -spd_kt * math.sin(rad)   # componente este-oeste
            v = -spd_kt * math.cos(rad)   # componente norte-sur
            has_wind = True

        stations.append({
            "name": nombre, "lat": lat, "lon": lon,
            "temp_c": temp, "dewpoint_c": dew, "pressure_hpa": pres,
            "sky_oktas": sky, "u_kt": u, "v_kt": v, "has_wind": has_wind,
        })
    return stations
