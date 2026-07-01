"""
gfs_shear.py
============
Descarga viento del modelo GFS (NOAA NOMADS, 0.25 grados) y calcula la
CIZALLADURA (bulk shear) de capas para superponer como barbas de viento
sobre el plot de satelite GOES-19.

El satelite NO mide viento en niveles de la atmosfera (solo vientos derivados
de nubes, que no sirven para esto); por eso la cizalladura se calcula a partir
de un MODELO NUMERICO (GFS), va con ~4-6 horas de latencia de publicacion.

Definiciones (aproximadas con niveles de presion, uso operativo estandar):
  - Cizalladura profunda "0-6 km"  ~ viento en 500 hPa  MENOS viento en 10 m.
  - Cizalladura baja     "0-1 km"  ~ viento en 850 hPa  MENOS viento en 10 m.
(500 hPa y 850 hPa son las aproximaciones de presion mas usadas cuando no se
tienen niveles de modelo nativos; el propio SPC/NWS las usa de forma similar
cuando trabaja con datos en niveles de presion.)

Fuente: NOAA NOMADS - GFS 0.25 grados (filtro de subregion vía CGI, liviano).
"""

import os
import re
import datetime as dt
import urllib.request
import urllib.error
from urllib.parse import urlencode

import numpy as np
import xarray as xr

NOMADS_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
_USER_AGENT = "Mozilla/5.0 (GOES19-Nowcasting; TRP Meteorologia)"
MS_TO_KT = 1.943844  # m/s -> nudos

# Ciclos sinopticos del GFS (se publican 4 veces al dia, UTC)
_CYCLE_HOURS = (0, 6, 12, 18)


def _cycles_desc(target: dt.datetime, max_tries: int = 10):
    """Genera ciclos GFS (UTC) <= target, del mas reciente al mas viejo."""
    base = target.replace(minute=0, second=0, microsecond=0)
    base = base.replace(hour=(base.hour // 6) * 6)
    for i in range(max_tries):
        yield base - dt.timedelta(hours=6 * i)


def _build_url(cycle: dt.datetime, fhour: int, extent) -> str:
    lon0, lon1, lat0, lat1 = extent
    fname = f"gfs.t{cycle.hour:02d}z.pgrb2.0p25.f{fhour:03d}"
    dirpath = f"/gfs.{cycle:%Y%m%d}/{cycle.hour:02d}/atmos"
    params = {
        "file": fname,
        "lev_10_m_above_ground": "on",
        "lev_850_mb": "on",
        "lev_500_mb": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "subregion": "",
        "toplat": f"{lat1:.2f}",
        "bottomlat": f"{lat0:.2f}",
        "leftlon": f"{lon0:.2f}",
        "rightlon": f"{lon1:.2f}",
        "dir": dirpath,
    }
    return NOMADS_BASE + "?" + urlencode(params)


def _try_download(url: str, out_path: str, timeout=30) -> bool:
    """Descarga a out_path. Devuelve True si parece un GRIB2 valido."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False
    # Un GRIB2 real arranca con la marca 'GRIB'; una pagina de error HTML no.
    if len(data) < 200 or data[:4] != b"GRIB":
        return False
    with open(out_path, "wb") as f:
        f.write(data)
    return True


def fetch_gfs_grib(target: dt.datetime, extent, cache_dir: str, region_tag: str = "reg",
                   max_tries: int = 10):
    """Busca y descarga el archivo GFS (recortado a extent) mas cercano a target.

    Devuelve (ruta_local, cycle, fhour, valid_time).
    """
    os.makedirs(cache_dir, exist_ok=True)
    for cycle in _cycles_desc(target, max_tries=max_tries):
        fhour = int(round((target - cycle).total_seconds() / 3600.0))
        if fhour < 0 or fhour > 120:
            continue
        tag = f"gfs_{cycle:%Y%m%d%H}_f{fhour:03d}_{region_tag}.grib2"
        out_path = os.path.join(cache_dir, tag)
        if os.path.exists(out_path):
            valid_time = cycle + dt.timedelta(hours=fhour)
            return out_path, cycle, fhour, valid_time
        url = _build_url(cycle, fhour, extent)
        if _try_download(url, out_path):
            valid_time = cycle + dt.timedelta(hours=fhour)
            return out_path, cycle, fhour, valid_time
    raise RuntimeError("No se pudo descargar ningun ciclo/hora del GFS reciente "
                        "(revisa conexion a internet o intenta mas tarde).")


def _open_level(path, filter_keys):
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={"filter_by_keys": filter_keys,
                                        "indexpath": ""})
    return ds


def compute_shear(target: dt.datetime, extent, cache_dir: str, region_tag: str = "reg"):
    """Descarga el GFS mas cercano y calcula la cizalladura 0-6km y 0-1km.

    Devuelve un dict:
        lon, lat        -> arrays 1D (grados, lon en -180..180)
        u_deep, v_deep   -> componentes de cizalladura 0-6 km (nudos), 2D
        u_low,  v_low    -> componentes de cizalladura 0-1 km (nudos), 2D
        cycle, fhour, valid_time
    """
    path, cycle, fhour, valid_time = fetch_gfs_grib(target, extent, cache_dir, region_tag)

    sfc = _open_level(path, {"typeOfLevel": "heightAboveGround", "level": 10})
    iso = _open_level(path, {"typeOfLevel": "isobaricInhPa"})

    u10 = sfc["u10"].values * MS_TO_KT
    v10 = sfc["v10"].values * MS_TO_KT
    u_iso = iso["u"].values * MS_TO_KT   # (level, lat, lon) -> level orden: 850, 500
    v_iso = iso["v"].values * MS_TO_KT
    levels = iso["isobaricInhPa"].values
    i850 = int(np.argmin(np.abs(levels - 850)))
    i500 = int(np.argmin(np.abs(levels - 500)))

    lon = sfc["longitude"].values.copy()
    lon = np.where(lon > 180, lon - 360, lon)  # 0-360 -> -180..180
    lat = sfc["latitude"].values

    u_deep = u_iso[i500] - u10
    v_deep = v_iso[i500] - v10
    u_low = u_iso[i850] - u10
    v_low = v_iso[i850] - v10

    sfc.close()
    iso.close()

    return {
        "lon": lon, "lat": lat,
        "u_deep": u_deep, "v_deep": v_deep,
        "u_low": u_low, "v_low": v_low,
        "cycle": cycle, "fhour": fhour, "valid_time": valid_time,
    }
