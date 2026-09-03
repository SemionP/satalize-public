"""MEDI Platform — GEE Processing Functions"""

import math, json, tempfile, os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import streamlit as st
import ee

from config import BEACHES, HAIFA_BBOX_COORDS, ISRAEL_CLIP_COORDS

# Lazy-init GEE geometries
def _haifa_bbox(): return ee.Geometry.Rectangle(HAIFA_BBOX_COORDS)

# Mediterranean coast only — excludes Kinneret, Dead Sea, Red Sea
#
# FIXED (was): the Gaza-area segment used a flat longitude of 35.0 for both
# the Gaza/Egypt border point (31.2N) and the point labeled "Ashkelon"
# (31.55N, which is actually still Gaza-Strip latitude -- real Ashkelon is
# ~31.67N). Lon 35.0 is ~50km east of the real coast down there (it's the
# correct value further north, e.g. Rosh HaNikra/Lebanese border, but wrong
# here), so imagery clipped to this polygon under-covered the Gaza Strip's
# offshore water. Replaced with a proper trace of the Gaza + Ashkelon coast
# (coordinates checked against city-level geocoding, ~5-10km tolerance --
# this is a coarse clip boundary for satellite processing, not a legal
# maritime boundary claim) and pulled the SW corner south to 31.10 for a
# safety margin below the Rafah/Gaza-Egypt border.
_MED_COAST_COORDS = [
    [33.00, 31.10],  # SW — open Mediterranean (safety margin south of Gaza)
    [34.30, 31.10],  # S — south of Rafah / Gaza-Egypt border
    [34.30, 31.25],  # Rafah / Gaza-Egypt maritime border
    [34.55, 31.55],  # mid-Gaza coast
    [34.68, 31.62],  # Gaza/Israel border (Zikim area)
    [34.75, 31.68],  # Ashkelon
    [34.90, 31.80],  # Ashdod
    [34.85, 32.10],  # Tel Aviv
    [34.80, 32.35],  # Netanya
    [34.85, 32.55],  # Caesarea
    [34.90, 32.82],  # Haifa
    [35.00, 33.05],  # Rosh HaNikra
    [35.00, 33.15],  # Lebanese border
    [33.00, 33.15],  # NW — open sea
    [33.00, 31.10],  # close
]
def _israel_clip(): return ee.Geometry.Polygon([_MED_COAST_COORDS])

# Wide AOI for satellite collection filtering (includes full coast + offshore)
# Southern bound widened to 31.10 to match the _MED_COAST_COORDS fix above.
_MED_WIDE_BBOX = [33.0, 31.10, 35.1, 33.2]
def _med_wide(): return ee.Geometry.Rectangle(_MED_WIDE_BBOX)

def _to_ee_geom(raw):
    """Convert raw geometry dict from config to ee.Geometry."""
    if isinstance(raw, dict):
        t = raw.get("_type", "")
        c = raw.get("coords", [])
        if t == "rect": return ee.Geometry.Rectangle(c)
        if t == "poly": return ee.Geometry.Polygon([c] if c and not isinstance(c[0][0], list) else c)
        if t == "point": return ee.Geometry.Point(c)
    return raw  # already ee.Geometry or unknown

@st.cache_resource
def init_gee():
    creds=dict(st.secrets["gee_credentials"])
    with tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False) as f:
        f.write(json.dumps(creds)); tmp=f.name
    ee.Initialize(ee.ServiceAccountCredentials(creds["client_email"],tmp))
    os.unlink(tmp)
init_gee()

def _empty_atm():
    return {"wind_speed":None,"wind_dir_deg":None,"temp_c":None,"humidity":None,
            "precip_mm":None,"_error":None}

@st.cache_data(ttl=3600)
def get_atm(lat, lon):
    try:
        import requests as _req
        r=_req.get("https://api.open-meteo.com/v1/forecast",params={
            "latitude":round(lat,4),"longitude":round(lon,4),
            "current":"temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,weather_code",
            "wind_speed_unit":"ms","forecast_days":1},timeout=10)
        r.raise_for_status(); cur=r.json().get("current",{})
        ws,wd,tc=cur.get("wind_speed_10m"),cur.get("wind_direction_10m"),cur.get("temperature_2m")
        rh,pr=cur.get("relative_humidity_2m"),cur.get("precipitation")
        return {"wind_speed":round(ws,1) if ws else None,"wind_dir_deg":round(wd,1) if wd else None,
                "temp_c":round(tc,1) if tc else None,"humidity":round(rh,0) if rh else None,
                "precip_mm":round(pr,2) if pr else None,"_error":None}
    except Exception as e:
        return {**_empty_atm(),"_error":str(e)}

@st.cache_data(ttl=3600)
def get_sst(lat, lon):
    try:
        import requests as _req
        r=_req.get("https://marine-api.open-meteo.com/v1/marine",params={
            "latitude":lat,"longitude":lon,"current":"sea_surface_temperature","forecast_days":1},timeout=10)
        return r.json().get("current",{}).get("sea_surface_temperature")
    except: return None

@st.cache_data(ttl=14400)
def get_available_s3_dates(days_back=60):
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back)
    # Use wider bbox to catch all S3 passes over Israel
    wide_bbox = _med_wide()
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(wide_bbox)
            .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    dl = coll.aggregate_array("system:time_start").getInfo()
    dates = sorted(list(set([
        datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d") for d in dl
    ])), reverse=True)
    return dates

@st.cache_data(ttl=10800)
def get_modis_sst_anomaly(target_date_str):
    """
    MODIS MOD11A1 - Sea Surface Temperature anomaly.
    anomaly = today SST - 30-day mean SST
    Returns: ee.Image with band 'SST_anomaly' (degrees C) + scalar mean anomaly
    """
    wm  = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    t   = ee.Date(target_date_str)

    # Today SST (LST_Day_1km in Kelvin × 0.02 → Celsius)
    today_coll = (ee.ImageCollection("MODIS/061/MOD11A1")
                  .filterBounds(_haifa_bbox())
                  .filterDate(t.advance(-2,"day"), t.advance(1,"day"))
                  .select("LST_Day_1km"))
    if today_coll.size().getInfo() == 0:
        return None, None

    sst_today = today_coll.mean().multiply(0.02).subtract(273.15).updateMask(wm)

    # 30-day baseline
    baseline_coll = (ee.ImageCollection("MODIS/061/MOD11A1")
                     .filterBounds(_haifa_bbox())
                     .filterDate(t.advance(-31,"day"), t.advance(-1,"day"))
                     .select("LST_Day_1km"))
    sst_baseline  = baseline_coll.mean().multiply(0.02).subtract(273.15).updateMask(wm)

    anomaly_img = sst_today.subtract(sst_baseline).rename("SST_anomaly").clip(_haifa_bbox())

    # Scalar mean anomaly for MEDI engine
    try:
        val = anomaly_img.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = HAIFA_BBOX,
            scale     = 1000,
            bestEffort= True,
        ).getInfo()
        mean_anomaly = val.get("SST_anomaly")
        mean_anomaly = round(float(mean_anomaly), 2) if mean_anomaly is not None else None
    except Exception:
        mean_anomaly = None

    return anomaly_img, mean_anomaly


@st.cache_data(ttl=7200)
def process_modis_wqi(target_date_str):
    """
    MODIS MOD09GA - daily 250-500m WQI for Israel coast.
    Used as fallback when S3 not available, or as supplement.
    Returns: (wqi_layer, df_beaches, error, age_hours, source_label)
    """
    wm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    t  = ee.Date(target_date_str)
    # Merge Terra (MOD) + Aqua (MYD) for better daily coverage
    now_m = datetime.utcnow()
    end_m = ee.Date(now_m.strftime("%Y-%m-%d")).advance(1,"day")
    start_m = ee.Date((now_m - timedelta(days=3)).strftime("%Y-%m-%d"))
    terra = (ee.ImageCollection("MODIS/061/MOD09GA")
             .filterBounds(_haifa_bbox())
             .filterDate(start_m, end_m))
    aqua  = (ee.ImageCollection("MODIS/061/MYD09GA")
             .filterBounds(_haifa_bbox())
             .filterDate(start_m, end_m))
    coll  = terra.merge(aqua).sort("system:time_start", False)

    if coll.size().getInfo() == 0:
        return None, None, "No MODIS data for this date.", None, "MODIS Terra+Aqua"

    img_first   = coll.first()
    img_time_ms = img_first.get("system:time_start").getInfo()
    img_dt      = datetime.utcfromtimestamp(img_time_ms / 1000)
    age_hours   = (datetime.utcnow() - img_dt).total_seconds() / 3600

    # Cloud mask: bits 0-1 of state_1km == 0 (clear)
    qa    = img_first.select("state_1km")
    clear = qa.bitwiseAnd(0b11).eq(0)
    img   = img_first.updateMask(clear).updateMask(wm)

    b1 = img.select("sur_refl_b01")  # 645nm red
    b2 = img.select("sur_refl_b02")  # 859nm NIR
    b4 = img.select("sur_refl_b04")  # 545nm green

    ndwi_n = b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3, 0.3).clamp(0, 1)
    chl_n  = b4.divide(b1.add(1e-6)).unitScale(0.8, 2.5).clamp(0, 1)
    turb_n = ee.Image(1).subtract(b1.unitScale(0, 1500)).clamp(0, 1)

    raw = ndwi_n.add(chl_n).add(turb_n).divide(3).multiply(100).rename("WQI")
    wqi = raw.clip(_israel_clip()).updateMask(wm)

    def _pt(pt):
        try:
            v  = wqi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee.Geometry.Point([pt["lon"], pt["lat"]]).buffer(500),
                scale=500, bestEffort=True).getInfo()
            wv = v.get("WQI")
            return {**pt, "wqi": round(wv, 1) if wv else None}
        except:
            return {**pt, "wqi": None}

    with ThreadPoolExecutor(max_workers=4) as ex:
        pts = list(ex.map(_pt, BEACHES))

    return wqi, pd.DataFrame(pts), None, round(age_hours, 1), "MODIS"



@st.cache_data(ttl=21600)
def process_israel_s2(target_date_str):
    """
    Sentinel-2 MSI SR — 10m WQI for full Israeli Mediterranean coast.
    Uses mosaic() of all available tiles in the last 10 days to cover
    the entire coast (not just a single tile).
    """
    wm  = ee.Image(1)
    aoi = _israel_clip()
    now = datetime.utcnow()
    end = ee.Date(now.strftime("%Y-%m-%d")).advance(1, "day")
    start = ee.Date((now - timedelta(days=10)).strftime("%Y-%m-%d"))

    # filterBounds on full AOI — not just Haifa — to get all coastal tiles
    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .sort("system:time_start", False))

    if coll.size().getInfo() == 0:
        return None, None, "No Sentinel-2 data.", None, "Sentinel-2"

    # Get age from most recent image
    img_first   = coll.first()
    img_time_ms = img_first.get("system:time_start").getInfo()
    img_dt      = datetime.utcfromtimestamp(img_time_ms / 1000)
    age_hours   = (datetime.utcnow() - img_dt).total_seconds() / 3600

    # Mosaic all tiles — covers full coast
    # Use NDWI > 0 as water mask instead of SCL (SCL is per-tile and misses open sea)
    img_mosaic = coll.mosaic()
    b3  = img_mosaic.select("B3").divide(10000)
    b4  = img_mosaic.select("B4").divide(10000)
    b5  = img_mosaic.select("B5").divide(10000)
    b8  = img_mosaic.select("B8").divide(10000)
    b8a = img_mosaic.select("B8A").divide(10000)

    ndwi_raw = b3.subtract(b8).divide(b3.add(b8).add(1e-6))

    # Use same coverage as NDWI index layer — no additional mask
    ndwi_n = ndwi_raw.unitScale(-0.3, 0.5).clamp(0, 1)
    chl_n  = b5.divide(b4.add(1e-6)).unitScale(1.0, 3.5).clamp(0, 1)
    turb_n = ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0, 0.15)).clamp(0, 1)

    wqi = (ndwi_n.add(chl_n).add(turb_n)
           .divide(3).multiply(100)
           .clip(aoi)
           .rename("WQI")
           .updateMask(ndwi_raw.gt(-0.1)))

    def _pt(pt):
        try:
            v  = wqi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee.Geometry.Point([pt["lon"], pt["lat"]]).buffer(300),
                scale=10, bestEffort=True).getInfo()
            wv = v.get("WQI")
            return {**pt, "wqi": round(wv, 1) if wv else None}
        except:
            return {**pt, "wqi": None}

    with ThreadPoolExecutor(max_workers=4) as ex:
        pts = list(ex.map(_pt, BEACHES))

    return wqi, pd.DataFrame(pts), None, round(age_hours, 1), "Sentinel-2"









@st.cache_data(ttl=21600)
def get_available_dates_combined(days_back=7):
    """Returns list of dicts: {date, source} - S3 dates + daily MODIS fallback."""
    end      = datetime.utcnow()
    start    = end - timedelta(days=days_back)
    wide     = _med_wide()
    date_fmt = "%Y-%m-%d"

    # S3 dates only (one GEE call)
    s3_coll  = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                .filterBounds(wide)
                .filterDate(start.strftime(date_fmt), end.strftime(date_fmt)))
    s3_ts    = s3_coll.aggregate_array("system:time_start").getInfo()
    s3_dates = set(datetime.utcfromtimestamp(d/1000).strftime(date_fmt) for d in s3_ts)

    # MODIS: assume available every day (no extra GEE call)
    all_dates = [(end - timedelta(days=i)).strftime(date_fmt) for i in range(days_back)]

    result = []
    for d in all_dates:
        if d in s3_dates:
            result.append({"date": d, "source": "S3",    "label": f"🛰️ {d} · S3"})
        else:
            result.append({"date": d, "source": "MODIS", "label": f"📡 {d} · MODIS"})
    return result


@st.cache_data(ttl=7200)
def process_israel_wqi(target_date_str):
    wm=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    t=ee.Date(target_date_str)
    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(_haifa_bbox())
          .filterDate(t.advance(-2,'day'),t.advance(1,'day')))
    if coll.size().getInfo()==0: return None,None,"No Sentinel-3 data for this date.",None
    # Get actual image acquisition time
    img_first = coll.sort("system:time_start", False).first()
    img_time_ms = img_first.get("system:time_start").getInfo()
    img_dt = datetime.utcfromtimestamp(img_time_ms / 1000)
    age_hours = (datetime.utcnow() - img_dt).total_seconds() / 3600

    img=coll.median().clip(_israel_clip()).updateMask(wm)
    ndwi=img.normalizedDifference(['Oa06_radiance','Oa17_radiance'])
    b10,b11,b12=img.select('Oa10_radiance'),img.select('Oa11_radiance'),img.select('Oa12_radiance')
    mci=b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
    turb=img.select('Oa08_radiance')
    raw=ndwi.unitScale(-0.2,0.5).clamp(0,1).add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1)).add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1)).divide(3).multiply(100).rename('WQI')
    wqi=raw.reduceNeighborhood(reducer=ee.Reducer.mean(),kernel=ee.Kernel.square(radius=1,units='pixels')).rename('WQI').updateMask(wm)
    def _pt(pt):
        try:
            v=wqi.reduceRegion(reducer=ee.Reducer.mean(),geometry=ee.Geometry.Point([pt["lon"],pt["lat"]]).buffer(450),scale=300,bestEffort=True).getInfo()
            wv=v.get('WQI'); return {**pt,"wqi":round(wv,1) if wv else None}
        except: return {**pt,"wqi":None}
    with ThreadPoolExecutor(max_workers=4) as ex: pts=list(ex.map(_pt,BEACHES))
    return wqi, pd.DataFrame(pts), None, round(age_hours, 1)


@st.cache_data(ttl=14400)
def compute_beach_history_7d():
    """
    Compute WQI for each beach for each available date in last 14 days.
    Returns dict: {beach_name: [{date, wqi}, ...]}
    All dates computed in parallel per-beach via ThreadPoolExecutor.
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=15)
    wide     = _med_wide()
    wm_gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    # Ocean-only: exclude inland water (use SRTM elevation > 0 as land proxy)
    # Keep only pixels where distance to ocean shoreline is small
    # Simple: use GSW "transition" band - permanent sea water
    # Mediterranean-only mask — Kinneret/Dead Sea/Red Sea excluded by clip geometry
    wm = wm_gsw.clip(_israel_clip())

    # Get S3 dates
    s3_coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
               .filterBounds(wide)
               .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
               .sort("system:time_start", False))
    s3_ts_list = s3_coll.aggregate_array("system:time_start").getInfo()
    s3_dates = set()
    for ts in s3_ts_list:
        s3_dates.add(datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"))

    # S2 dates
    s2_coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(wide)
               .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
               .sort("system:time_start", False))
    s2_ts_list = s2_coll.aggregate_array("system:time_start").getInfo()
    s2_dates = set()
    for ts in s2_ts_list:
        s2_dates.add(datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"))

    # All dates = S3 + S2 + every day in range (MODIS fallback)
    days_back = 15
    all_day_dates = [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back)]
    seen_dates = set()
    date_ts = []
    for d in all_day_dates:
        if d not in seen_dates:
            seen_dates.add(d)
            src = "S3" if d in s3_dates else "S2" if d in s2_dates else "MODIS"
            date_ts.append((d, src))

    if not date_ts:
        return {}

    def _wqi_for_date(args):
        """Compute WQI image for one date - S3 preferred, MODIS fallback."""
        date_str, source = args
        try:
            t = ee.Date(date_str)
            if source == "S2":
                try:
                    s2c = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                           .filterBounds(_haifa_bbox())
                           .filterDate(t.advance(-5,"day"),t.advance(1,"day"))
                           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",30))
                           .sort("system:time_start",False))
                    if s2c.size().getInfo() == 0: return date_str, None
                    im2 = s2c.first().updateMask(ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30))
                    b3,b4,b5,b8,b8a=(im2.select("B3").divide(10000),im2.select("B4").divide(10000),
                                     im2.select("B5").divide(10000),im2.select("B8").divide(10000),im2.select("B8A").divide(10000))
                    ndwi_n=b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
                    chl_n=b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1)
                    turb_n=ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1)
                    wqi=ndwi_n.add(chl_n).add(turb_n).divide(3).multiply(100).rename("WQI").clip(_israel_clip())
                    return date_str, wqi
                except: return date_str, None
            if source == "S3":
                coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                        .filterBounds(_haifa_bbox())
                        .filterDate(t.advance(-1,"day"), t.advance(1,"day")))
                if coll.size().getInfo() == 0:
                    source = "MODIS"
                else:
                    img  = coll.median().clip(_israel_clip()).updateMask(wm)
                    ndwi = img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
                    b10,b11,b12 = img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
                    mci  = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
                    turb = img.select("Oa08_radiance")
                    raw  = (ndwi.unitScale(-0.2,0.5).clamp(0,1)
                            .add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1))
                            .add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1))
                            .divide(3).multiply(100).rename("WQI"))
                    wqi  = raw.reduceNeighborhood(
                        reducer=ee.Reducer.mean(),
                        kernel=ee.Kernel.square(radius=1,units="pixels")
                    ).rename("WQI").updateMask(wm)
                    return date_str, wqi
            if source == "MODIS":
                terra_h = ee.ImageCollection("MODIS/061/MOD09GA").filterBounds(_haifa_bbox()).filterDate(t.advance(-1,"day"),t.advance(1,"day"))
                aqua_h  = ee.ImageCollection("MODIS/061/MYD09GA").filterBounds(_haifa_bbox()).filterDate(t.advance(-1,"day"),t.advance(1,"day"))
                qa      = terra_h.merge(aqua_h).sort("system:time_start",False)
                if qa.size().getInfo() == 0:
                    return date_str, None
                img_m = qa.first()
                clear = img_m.select("state_1km").bitwiseAnd(0b11).eq(0)
                img_m = img_m.updateMask(clear).updateMask(wm)
                b1,b2,b4 = img_m.select("sur_refl_b01"),img_m.select("sur_refl_b02"),img_m.select("sur_refl_b04")
                ndwi_n = b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3,0.3).clamp(0,1)
                chl_n  = b4.divide(b1.add(1e-6)).unitScale(0.8,2.5).clamp(0,1)
                turb_n = ee.Image(1).subtract(b1.unitScale(0,1500)).clamp(0,1)
                wqi    = ndwi_n.add(chl_n).add(turb_n).divide(3).multiply(100).rename("WQI").clip(_israel_clip()).updateMask(wm)
                return date_str, wqi
        except:
            return date_str, None

    def _sample_beach_on_wqi(args):
        beach, date_str, wqi = args
        if wqi is None:
            return beach["name"], date_str, None
        try:
            v  = wqi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee.Geometry.Point([beach["lon"], beach["lat"]]).buffer(450),
                scale=300, bestEffort=True).getInfo()
            wv = v.get("WQI")
            return beach["name"], date_str, round(wv, 1) if wv else None
        except:
            return beach["name"], date_str, None

    # Compute WQI images for all dates (up to 4 in parallel)
    with ThreadPoolExecutor(max_workers=4) as ex:
        wqi_images = dict(ex.map(_wqi_for_date, date_ts))

    # Sample all beaches × all dates
    tasks = [(b, d, wqi_images.get(d)) for b in BEACHES for d, _ in date_ts]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_sample_beach_on_wqi, tasks))

    # Organize into dict
    history = {b["name"]: [] for b in BEACHES}
    for beach_name, date_str, wqi_val in results:
        history[beach_name].append({"date": date_str, "wqi": wqi_val})

    # Sort by date
    for name in history:
        history[name] = sorted(history[name], key=lambda x: x["date"])

    return history



@st.cache_data(ttl=7200)
def process_port_medi(port_key, target_date_str):
    """Compute WQI + SST anomaly for a specific port zone."""
    port = PORTS[port_key]
    bbox = port["bbox"]
    wm   = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    t    = ee.Date(target_date_str)

    # S3 WQI
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(bbox)
            .filterDate(t.advance(-2,"day"), t.advance(1,"day")))
    if coll.size().getInfo() == 0:
        wqi_val, age_h = None, 48.0
    else:
        img_first  = coll.sort("system:time_start", False).first()
        img_time   = img_first.get("system:time_start").getInfo()
        age_h      = (datetime.utcnow() - datetime.utcfromtimestamp(img_time/1000)).total_seconds() / 3600
        img        = coll.median().clip(bbox).updateMask(wm)
        ndwi       = img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
        b10,b11,b12= img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
        mci        = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
        turb       = img.select("Oa08_radiance")
        raw        = ndwi.unitScale(-0.2,0.5).clamp(0,1).add(
                        ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1)).add(
                        ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1)
                     ).divide(3).multiply(100).rename("WQI")
        try:
            val     = raw.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox, scale=300, bestEffort=True).getInfo()
            wqi_val = round(val.get("WQI"), 1) if val.get("WQI") else None
        except:
            wqi_val = None

    # MODIS SST anomaly
    try:
        today_sst  = (ee.ImageCollection("MODIS/061/MOD11A1")
                      .filterBounds(bbox)
                      .filterDate(t.advance(-2,"day"), t.advance(1,"day"))
                      .select("LST_Day_1km").mean().multiply(0.02).subtract(273.15).updateMask(wm))
        base_sst   = (ee.ImageCollection("MODIS/061/MOD11A1")
                      .filterBounds(bbox)
                      .filterDate(t.advance(-31,"day"), t.advance(-1,"day"))
                      .select("LST_Day_1km").mean().multiply(0.02).subtract(273.15).updateMask(wm))
        anom_val   = today_sst.subtract(base_sst).reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=bbox, scale=1000, bestEffort=True).getInfo()
        sst_anom   = round(float(anom_val.get("LST_Day_1km")), 2) if anom_val.get("LST_Day_1km") else None
    except:
        sst_anom = None

    return wqi_val, sst_anom, round(age_h, 1)


@st.cache_data(ttl=14400)
def get_global_wqi_layer(target_date_str, bbox_rect):
    lon_min,lat_min,lon_max,lat_max=bbox_rect
    bbox=ee.Geometry.Rectangle([lon_min,lat_min,lon_max,lat_max])
    wm=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    t=ee.Date(target_date_str)
    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(bbox)
          .filterDate(t.advance(-1,'day'),t.advance(1,'day')))
    if coll.size().getInfo()==0: return None,"No data for this area/date."
    img=coll.median().clip(bbox).updateMask(wm)
    ndwi=img.normalizedDifference(['Oa06_radiance','Oa17_radiance'])
    b10,b11,b12=img.select('Oa10_radiance'),img.select('Oa11_radiance'),img.select('Oa12_radiance')
    mci=b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
    turb=img.select('Oa08_radiance')
    raw=ndwi.unitScale(-0.2,0.5).clamp(0,1).add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1)).add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1)).divide(3).multiply(100).rename('WQI')
    return raw.reduceNeighborhood(reducer=ee.Reducer.mean(),kernel=ee.Kernel.square(radius=1,units='pixels')).rename('WQI').updateMask(wm), None

def get_bbox_from_map(map_data, zoom):
    if not map_data or not map_data.get("center"): return None
    lat=map_data["center"]["lat"]; lon=map_data["center"]["lng"]
    dp=360.0/(256*(2**zoom)); hw=dp*400; hh=dp*275
    return (max(-180,lon-hw),max(-85,lat-hh),min(180,lon+hw),min(85,lat+hh))

def haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0; p1,p2=math.radians(lat1),math.radians(lat2)
    a=math.sin(math.radians(lat2-lat1)/2)**2+math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

# =============================================================================
# UI
# =============================================================================
MODE_ISRAEL = "🏖️ Israel Coast"
MODE_GLOBAL = "🌍 Global"
mode = MODE_ISRAEL  # Default to Israel Coast

# Risk profile shown in MEDI tab only - initialized here for session state

# ── Israel Coast ──────────────────────────────────────────────────────────────



@st.cache_data(ttl=7200)
def compute_point_wqi(lat: float, lon: float, target_date_str: str, source: str = "S3") -> float | None:
    """
    WQI at nearest water pixel to (lat, lon).
    Uses a small buffer (500m) and takes only pixels where GSW >= 30%.
    Returns scalar WQI or None.
    """
    wm    = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    pt    = ee.Geometry.Point([lon, lat])
    buf   = pt.buffer(500)
    t     = ee.Date(target_date_str)

    try:
        if source == "S2":
            coll  = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                     .filterBounds(buf)
                     .filterDate(t.advance(-5,"day"), t.advance(1,"day"))
                     .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                     .sort("system:time_start", False))
            if coll.size().getInfo() == 0: return None
            img   = coll.first().updateMask(wm)
            b3,b4,b5,b8,b8a = (img.select("B3").divide(10000), img.select("B4").divide(10000),
                                img.select("B5").divide(10000), img.select("B8").divide(10000),
                                img.select("B8A").divide(10000))
            wqi   = (b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
                     .add(b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1))
                     .add(ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI").updateMask(wm))
        else:
            coll  = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                     .filterBounds(buf)
                     .filterDate(t.advance(-2,"day"), t.advance(1,"day")))
            if coll.size().getInfo() == 0: return None
            img   = coll.median().updateMask(wm)
            ndwi  = img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
            b10,b11,b12 = img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
            mci   = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
            turb  = img.select("Oa08_radiance")
            wqi   = (ndwi.unitScale(-0.2,0.5).clamp(0,1)
                     .add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1))
                     .add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI").updateMask(wm))

        val = wqi.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = buf,
            scale     = 300,
            bestEffort= True
        ).getInfo()
        wv = val.get("WQI")
        return round(float(wv), 1) if wv is not None else None
    except:
        return None


@st.cache_data(ttl=7200)
def compute_point_mci(lat: float, lon: float, target_date_str: str) -> float | None:
    """
    Raw MCI (chlorophyll index) value at the nearest water pixel to (lat, lon),
    S-3 OLCI only, on the same band-difference scale used by the
    "MCI (Chlorophyll)" tile in get_satellite_layers() (displayed there with
    min=-2, max=12 against CHL_PALETTE). This is NOT a calibrated
    chlorophyll-a concentration in mg/m3 -- see the note in
    get_satellite_layers(). Returns None if there's no water pixel or no
    S-3 coverage near this point/date.
    """
    wm  = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    pt  = ee.Geometry.Point([lon, lat])
    buf = pt.buffer(500)
    t   = ee.Date(target_date_str)

    try:
        coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                .filterBounds(buf)
                .filterDate(t.advance(-2, "day"), t.advance(1, "day")))
        if coll.size().getInfo() == 0:
            return None
        img = coll.median().updateMask(wm)
        b10, b11, b12 = img.select("Oa10_radiance"), img.select("Oa11_radiance"), img.select("Oa12_radiance")
        mci = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75 - 681.25) / (753.75 - 681.25)))).rename("MCI")

        val = mci.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = buf,
            scale     = 300,
            bestEffort= True,
        ).getInfo()
        mv = val.get("MCI")
        return round(float(mv), 2) if mv is not None else None
    except:
        return None


@st.cache_data(ttl=7200)
def compute_city_wqi(target_date_str, source="S3"):
    """
    Compute WQI for each city's maritime zone polygon.
    Returns dict: {city_name: wqi_value}
    """
    wm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    t  = ee.Date(target_date_str)

    def _get_wqi_image():
        if source == "S2":
            coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(_haifa_bbox())
                    .filterDate(t.advance(-5,"day"),t.advance(1,"day"))
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",30))
                    .sort("system:time_start",False))
            if coll.size().getInfo() == 0: return None
            img   = coll.first().updateMask(wm)
            b3,b4,b5,b8,b8a = (img.select("B3").divide(10000),img.select("B4").divide(10000),
                                img.select("B5").divide(10000),img.select("B8").divide(10000),
                                img.select("B8A").divide(10000))
            return (b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
                    .add(b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1))
                    .add(ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1))
                    .divide(3).multiply(100).rename("WQI").updateMask(wm))
        else:
            coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                    .filterBounds(_haifa_bbox())
                    .filterDate(t.advance(-2,"day"),t.advance(1,"day")))
            if coll.size().getInfo() == 0: return None
            img  = coll.median().clip(_israel_clip()).updateMask(wm)
            ndwi = img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
            b10,b11,b12 = img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
            mci  = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
            turb = img.select("Oa08_radiance")
            raw  = (ndwi.unitScale(-0.2,0.5).clamp(0,1)
                    .add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1))
                    .add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1))
                    .divide(3).multiply(100).rename("WQI"))
            return raw.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=ee.Kernel.square(radius=1,units="pixels")
            ).rename("WQI").updateMask(wm)

    wqi_img = _get_wqi_image()
    if wqi_img is None:
        return {city: None for city in MARITIME_ZONES}

    results = {}
    for city, polygon in MARITIME_ZONES.items():
        try:
            val = wqi_img.reduceRegion(
                reducer  = ee.Reducer.mean(),
                geometry = polygon,
                scale    = 300,
                bestEffort=True
            ).getInfo()
            wv = val.get("WQI")
            results[city] = round(float(wv), 1) if wv else None
        except:
            results[city] = None

    return results


@st.cache_data(ttl=86400)
def compute_beach_history_range(days_back: int):
    """Compute WQI history for N days. S3+S2+MODIS for all ranges."""
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back+1)
    wide  = ee.Geometry.Rectangle([34.0,29.0,36.0,33.5])
    wm    = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    fmt   = "%Y-%m-%d"

    s3_coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
               .filterBounds(wide)
               .filterDate(start.strftime(fmt), end.strftime(fmt))
               .sort("system:time_start",False))
    s3_ts  = s3_coll.aggregate_array("system:time_start").getInfo()
    s3_set = set(datetime.utcfromtimestamp(ts/1000).strftime(fmt) for ts in s3_ts)

    s2_coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(wide)
               .filterDate(start.strftime(fmt), end.strftime(fmt))
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
               .sort("system:time_start",False))
    s2_ts  = s2_coll.aggregate_array("system:time_start").getInfo()
    s2_set = set(datetime.utcfromtimestamp(ts/1000).strftime(fmt) for ts in s2_ts)

    # Every calendar day in window, best sensor priority S3>S2>MODIS
    all_days = [(end-timedelta(days=i)).strftime(fmt) for i in range(days_back+1)]
    seen=set(); date_ts=[]
    for d in all_days:
        if d not in seen:
            seen.add(d)
            if d in s3_set:   date_ts.append((d,"S3"))
            elif d in s2_set: date_ts.append((d,"S2"))
            else:             date_ts.append((d,"MODIS"))

    if not date_ts: return {}

    def _wqi_for_date(args):
        date_str,source=args
        try:
            t=ee.Date(date_str)
            if source=="S3":
                coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(wide)
                      .filterDate(t,t.advance(1,"day")))
                if coll.size().getInfo()==0:
                    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(wide)
                          .filterDate(t.advance(-1,"day"),t.advance(2,"day")))
                    if coll.size().getInfo()==0: return date_str,source,None
                img=coll.median().clip(_israel_clip()).updateMask(wm)
                ndwi=img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
                b10,b11,b12=img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
                mci=b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
                turb=img.select("Oa08_radiance")
                raw=(ndwi.unitScale(-0.2,0.5).clamp(0,1)
                     .add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1))
                     .add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI"))
                wqi=raw.reduceNeighborhood(reducer=ee.Reducer.mean(),
                    kernel=ee.Kernel.square(radius=1,units="pixels")).rename("WQI").updateMask(wm)
                return date_str,source,wqi
            elif source=="S2":
                coll=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(wide)
                      .filterDate(t,t.advance(1,"day"))
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",40))
                      .sort("system:time_start",False))
                if coll.size().getInfo()==0: return date_str,source,None
                img=coll.first().updateMask(wm)
                b3,b4,b5,b8,b8a=(img.select("B3").divide(10000),img.select("B4").divide(10000),
                                  img.select("B5").divide(10000),img.select("B8").divide(10000),
                                  img.select("B8A").divide(10000))
                wqi=(b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
                     .add(b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1))
                     .add(ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI").clip(_israel_clip()).updateMask(wm))
                return date_str,source,wqi
            else:
                t2=ee.ImageCollection("MODIS/061/MOD09GA").filterBounds(wide).filterDate(t,t.advance(1,"day"))
                a2=ee.ImageCollection("MODIS/061/MYD09GA").filterBounds(wide).filterDate(t,t.advance(1,"day"))
                qa=t2.merge(a2).sort("system:time_start",False)
                if qa.size().getInfo()==0: return date_str,source,None
                im=qa.first(); cl=im.select("state_1km").bitwiseAnd(0b11).eq(0)
                im=im.updateMask(cl).updateMask(wm)
                b1,b2,b4=im.select("sur_refl_b01"),im.select("sur_refl_b02"),im.select("sur_refl_b04")
                wqi=(b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3,0.3).clamp(0,1)
                     .add(b4.divide(b1.add(1e-6)).unitScale(0.8,2.5).clamp(0,1))
                     .add(ee.Image(1).subtract(b1.unitScale(0,1500)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI").clip(_israel_clip()).updateMask(wm))
                return date_str,source,wqi
        except: return date_str,source,None

    def _sample(args):
        beach,date_str,source,wqi=args
        if wqi is None: return beach["name"],date_str,source,None
        try:
            v=wqi.reduceRegion(reducer=ee.Reducer.mean(),
              geometry=ee.Geometry.Point([beach["lon"],beach["lat"]]).buffer(450),
              scale=300,bestEffort=True).getInfo()
            wv=v.get("WQI")
            return beach["name"],date_str,source,round(wv,1) if wv else None
        except: return beach["name"],date_str,source,None

    with ThreadPoolExecutor(max_workers=4) as ex:
        wqi_images=list(ex.map(_wqi_for_date,date_ts))  # list of (date,source,img)
    img_map={(d,s):img for d,s,img in wqi_images}
    tasks=[(b,d,s,img_map.get((d,s))) for b in BEACHES for d,s in date_ts]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results=list(ex.map(_sample,tasks))

    history={b["name"]:[] for b in BEACHES}
    for bn,ds,src,wv in results:
        if wv is not None:
            history[bn].append({"date":ds,"wqi":wv,"source":src})
    for n in history:
        history[n]=sorted(history[n],key=lambda x:x["date"])
    return history







@st.cache_data(ttl=3600)
def compute_zone_history_range(zones_json: str, days_back: int):
    """
    Compute WQI history for user-defined polygon zones over N days.
    Uses S3 + S2 + MODIS. Stores source (S3/S2/MODIS) per data point for tooltip.
    zones_json: JSON string of {name: {coords: [...]}} to make it cache-friendly.
    Returns dict: {zone_name: [{date, wqi, source}, ...]}
    """
    import json as _j
    zones = _j.loads(zones_json)
    if not zones:
        return {}

    end   = datetime.utcnow()
    start = end - timedelta(days=days_back+1)
    wide     = _med_wide()
    wm    = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    fmt   = "%Y-%m-%d"

    # ── Discover available dates per sensor ──────────────────────────────────
    s3_coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
               .filterBounds(wide)
               .filterDate(start.strftime(fmt), end.strftime(fmt))
               .sort("system:time_start", False))
    s3_ts  = s3_coll.aggregate_array("system:time_start").getInfo()
    # Keep individual timestamps so we can build one image per exact day
    s3_dates = {}  # date_str -> list of timestamps
    for ts in s3_ts:
        d = datetime.utcfromtimestamp(ts/1000).strftime(fmt)
        s3_dates.setdefault(d, []).append(ts)

    s2_coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(wide)
               .filterDate(start.strftime(fmt), end.strftime(fmt))
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
               .sort("system:time_start", False))
    s2_ts  = s2_coll.aggregate_array("system:time_start").getInfo()
    s2_dates = {}
    for ts in s2_ts:
        d = datetime.utcfromtimestamp(ts/1000).strftime(fmt)
        s2_dates.setdefault(d, []).append(ts)

    # Every calendar day in window: S3 > S2 > MODIS priority
    all_days = [(end - timedelta(days=i)).strftime(fmt) for i in range(days_back + 1)]
    seen = set(); date_ts = []
    for d in all_days:
        if d not in seen:
            seen.add(d)
            if d in s3_dates:
                date_ts.append((d, "S3"))
            elif d in s2_dates:
                date_ts.append((d, "S2"))
            else:
                date_ts.append((d, "MODIS"))

    if not date_ts:
        return {name: [] for name in zones}

    def _wqi_for_date(args):
        date_str, source = args
        try:
            t = ee.Date(date_str)
            if source == "S3":
                # Filter to the exact calendar day only (not ±2) to avoid merging adjacent days
                day_start = t
                day_end   = t.advance(1, "day")
                coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                        .filterBounds(wide)
                        .filterDate(day_start, day_end))
                if coll.size().getInfo() == 0:
                    # Fallback: try ±1 day in case of UTC boundary shift
                    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                            .filterBounds(wide)
                            .filterDate(t.advance(-1,"day"), t.advance(2,"day")))
                    if coll.size().getInfo() == 0:
                        return date_str, source, None
                img  = coll.median().clip(_israel_clip()).updateMask(wm)
                ndwi = img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
                b10,b11,b12 = img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
                mci  = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
                turb = img.select("Oa08_radiance")
                raw  = (ndwi.unitScale(-0.2,0.5).clamp(0,1)
                        .add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1))
                        .add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1))
                        .divide(3).multiply(100).rename("WQI"))
                wqi  = raw.reduceNeighborhood(
                    reducer=ee.Reducer.mean(),
                    kernel=ee.Kernel.square(radius=1, units="pixels")
                ).rename("WQI").updateMask(wm)
                return date_str, source, wqi
            elif source == "S2":
                coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(wide)
                        .filterDate(t, t.advance(1,"day"))
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                        .sort("system:time_start", False))
                if coll.size().getInfo() == 0:
                    return date_str, source, None
                img   = coll.first().updateMask(wm)
                b3,b4,b5,b8,b8a = (img.select("B3").divide(10000), img.select("B4").divide(10000),
                                    img.select("B5").divide(10000), img.select("B8").divide(10000),
                                    img.select("B8A").divide(10000))
                wqi = (b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
                       .add(b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1))
                       .add(ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1))
                       .divide(3).multiply(100).rename("WQI").clip(_israel_clip()).updateMask(wm))
                return date_str, source, wqi
            else:  # MODIS
                t2 = ee.ImageCollection("MODIS/061/MOD09GA").filterBounds(wide).filterDate(t, t.advance(1,"day"))
                a2 = ee.ImageCollection("MODIS/061/MYD09GA").filterBounds(wide).filterDate(t, t.advance(1,"day"))
                qa = t2.merge(a2).sort("system:time_start", False)
                if qa.size().getInfo() == 0:
                    return date_str, source, None
                im = qa.first(); cl = im.select("state_1km").bitwiseAnd(0b11).eq(0)
                im = im.updateMask(cl).updateMask(wm)
                b1,b2,b4 = im.select("sur_refl_b01"),im.select("sur_refl_b02"),im.select("sur_refl_b04")
                wqi = (b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3,0.3).clamp(0,1)
                       .add(b4.divide(b1.add(1e-6)).unitScale(0.8,2.5).clamp(0,1))
                       .add(ee.Image(1).subtract(b1.unitScale(0,1500)).clamp(0,1))
                       .divide(3).multiply(100).rename("WQI").clip(_israel_clip()).updateMask(wm))
                return date_str, source, wqi
        except:
            return date_str, source, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        raw_results = list(ex.map(_wqi_for_date, date_ts))

    # Sample each zone polygon on each date's WQI image
    history = {name: [] for name in zones}
    for date_str, source, wqi_img in raw_results:
        if wqi_img is None:
            continue
        for zname, zdata in zones.items():
            try:
                poly = ee.Geometry.Polygon([zdata["coords"]])
                val  = wqi_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=poly, scale=300, bestEffort=True
                ).getInfo()
                wv = val.get("WQI")
                if wv is not None:
                    history[zname].append({
                        "date": date_str,
                        "wqi": round(float(wv), 1),
                        "source": source   # ← stored for tooltip
                    })
            except:
                pass

    for name in history:
        history[name] = sorted(history[name], key=lambda x: x["date"])

    return history
@st.cache_data(ttl=7200)
def get_satellite_layers(source: str, target_date_str: str) -> dict:
    """
    Return GEE tile URLs for all relevant indices per satellite.

    S3 OLCI layers:
      WQI, NDWI, MCI (Chlorophyll), Turbidity, True Color

    S2 MSI layers:
      WQI, NDWI, Chlorophyll (B5/B4), Turbidity, True Color (RGB), False Color (NIR)

    MODIS layers:
      WQI, NDWI, True Color

    Returns dict: {layer_name: tile_url}
    """
    aoi = _israel_clip()
    wm  = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30).clip(aoi)
    t   = ee.Date(target_date_str)
    result = {}

    WQI_PALETTE   = ["#d73027","#f46d43","#fdae61","#fee08b","#d9ef8b","#a6d96a","#66bd63","#1a9850","#4575b4"]
    NDWI_PALETTE  = ["#8B4513","#DEB887","#F5DEB3","#ffffff","#add8e6","#1e90ff","#00008b"]
    TURB_PALETTE  = ["#1a9850","#fdae61","#d73027"]
    CHL_PALETTE   = ["#f7fcf0","#ccebc5","#7bccc4","#2b8cbe","#084081"]
    RGB_PALETTE   = None  # truecolor handled differently

    try:
        if source == "S3":
            coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                    .filterBounds(_med_wide())
                    .filterDate(t.advance(-2,"day"), t.advance(1,"day")))
            if coll.size().getInfo() == 0:
                return {"error": "No S3 data for this date"}
            img = coll.median().clip(aoi).updateMask(wm)

            ndwi = img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
            b10,b11,b12 = img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
            mci  = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
            turb = img.select("Oa08_radiance")
            raw  = (ndwi.unitScale(-0.2,0.5).clamp(0,1)
                    .add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1))
                    .add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1))
                    .divide(3).multiply(100).rename("WQI"))
            wqi = raw.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=ee.Kernel.square(radius=1, units="pixels")
            ).rename("WQI").updateMask(wm)

            result["🌊 WQI"] = wqi.getMapId(
                {"min":30,"max":90,"palette":WQI_PALETTE})["tile_fetcher"].url_format
            result["💧 NDWI"] = ndwi.getMapId(
                {"min":-0.2,"max":0.5,"palette":NDWI_PALETTE})["tile_fetcher"].url_format
            result["🌿 MCI (Chlorophyll)"] = mci.getMapId(
                {"min":-2,"max":12,"palette":CHL_PALETTE})["tile_fetcher"].url_format
            result["🟫 Turbidity"] = turb.getMapId(
                {"min":10,"max":80,"palette":TURB_PALETTE})["tile_fetcher"].url_format
            # True color: Oa08(665), Oa06(560), Oa04(490)
            tc = img.select(["Oa08_radiance","Oa06_radiance","Oa04_radiance"])
            result["🎨 True Color"] = tc.getMapId(
                {"bands":["Oa08_radiance","Oa06_radiance","Oa04_radiance"],
                 "min":0,"max":150,"gamma":1.4})["tile_fetcher"].url_format

        elif source == "S2":
            coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(_med_wide())
                    .filterDate(t.advance(-5,"day"), t.advance(1,"day"))
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                    .sort("system:time_start", False))
            if coll.size().getInfo() == 0:
                return {"error": "No S2 data for this date"}
            img = coll.first().updateMask(wm)
            b3  = img.select("B3").divide(10000)
            b4  = img.select("B4").divide(10000)
            b5  = img.select("B5").divide(10000)
            b8  = img.select("B8").divide(10000)
            b8a = img.select("B8A").divide(10000)
            b2  = img.select("B2").divide(10000)

            ndwi_n = b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
            chl_n  = b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1)
            turb_n = ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1)
            wqi    = (ndwi_n.add(chl_n).add(turb_n)
                      .divide(3).multiply(100)
                      .clip(aoi).updateMask(wm).rename("WQI"))
            ndwi_raw = b3.subtract(b8).divide(b3.add(b8))
            chl_raw  = b5.divide(b4.add(1e-6))
            turb_raw = b4.add(b8a).divide(2)

            result["🌊 WQI"] = wqi.getMapId(
                {"min":30,"max":90,"palette":WQI_PALETTE})["tile_fetcher"].url_format
            result["💧 NDWI"] = ndwi_raw.getMapId(
                {"min":-0.3,"max":0.5,"palette":NDWI_PALETTE})["tile_fetcher"].url_format
            result["🌿 Chlorophyll (B5/B4)"] = chl_raw.getMapId(
                {"min":1.0,"max":3.5,"palette":CHL_PALETTE})["tile_fetcher"].url_format
            result["🟫 Turbidity (B4+B8A)"] = turb_raw.getMapId(
                {"min":0,"max":0.15,"palette":TURB_PALETTE})["tile_fetcher"].url_format
            # True color RGB
            result["🎨 True Color (RGB)"] = img.getMapId(
                {"bands":["B4","B3","B2"],"min":0,"max":0.3,"gamma":1.4})["tile_fetcher"].url_format
            # False color NIR
            result["🔴 False Color (NIR)"] = img.getMapId(
                {"bands":["B8","B4","B3"],"min":0,"max":0.4,"gamma":1.4})["tile_fetcher"].url_format

        elif source == "MODIS":
            coll = (ee.ImageCollection("MODIS/061/MOD09GA")
                    .filterBounds(_med_wide())
                    .filterDate(t, t.advance(1,"day")))
            if coll.size().getInfo() == 0:
                return {"error": "No MODIS data for this date"}
            im = coll.first()
            cl = im.select("state_1km").bitwiseAnd(0b11).eq(0)
            im = im.updateMask(cl).updateMask(wm)
            b1 = im.select("sur_refl_b01")  # 645nm Red
            b2 = im.select("sur_refl_b02")  # 858nm NIR
            b3 = im.select("sur_refl_b03")  # 469nm Blue
            b4 = im.select("sur_refl_b04")  # 555nm Green

            ndwi_raw = b4.subtract(b2).divide(b4.add(b2))
            wqi = (ndwi_raw.unitScale(-0.3,0.3).clamp(0,1)
                   .add(b4.divide(b1.add(1e-6)).unitScale(0.8,2.5).clamp(0,1))
                   .add(ee.Image(1).subtract(b1.unitScale(0,1500)).clamp(0,1))
                   .divide(3).multiply(100)
                   .clip(aoi).updateMask(wm).rename("WQI"))

            result["🌊 WQI"] = wqi.getMapId(
                {"min":30,"max":90,"palette":WQI_PALETTE})["tile_fetcher"].url_format
            result["💧 NDWI"] = ndwi_raw.getMapId(
                {"min":-0.3,"max":0.3,"palette":NDWI_PALETTE})["tile_fetcher"].url_format
            result["🎨 True Color"] = im.getMapId(
                {"bands":["sur_refl_b01","sur_refl_b04","sur_refl_b03"],
                 "min":0,"max":3000,"gamma":1.4})["tile_fetcher"].url_format

    except Exception as e:
        result["error"] = str(e)

    return result


def sample_pixel_spectra(lat: float, lon: float, source: str, target_date_str: str) -> dict:
    pt = ee.Geometry.Point([lon, lat])
    t  = ee.Date(target_date_str)
    try:
        if source == "S2":
            coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(pt)
                    .filterDate(t.advance(-8,"day"), t.advance(1,"day"))
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                    .sort("system:time_start", False))
            if coll.size().getInfo() == 0: return {}
            img    = coll.first()
            bands  = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
            labels = {"B1":"443nm","B2":"490nm","B3":"560nm","B4":"665nm","B5":"705nm",
                      "B6":"740nm","B7":"783nm","B8":"842nm","B8A":"865nm","B11":"1610nm","B12":"2190nm"}
            vals   = img.select(bands).reduceRegion(ee.Reducer.mean(), pt.buffer(30), 10).getInfo()
            return {labels[b]: round((vals.get(b) or 0)/10000, 5) for b in bands}
        elif source == "S3":
            coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                    .filterBounds(pt)
                    .filterDate(t.advance(-3,"day"), t.advance(1,"day"))
                    .sort("system:time_start", False))
            if coll.size().getInfo() == 0: return {}
            img   = coll.first()
            bands = [f"Oa{str(i).zfill(2)}_radiance" for i in range(1,17)]
            wls   = ["400","412","443","490","510","560","620","665","674","681","709","754","760","764","767","779"]
            vals  = img.select(bands).reduceRegion(ee.Reducer.mean(), pt.buffer(300), 300).getInfo()
            return {f"{wls[i]}nm": round(vals.get(bands[i]) or 0, 3) for i in range(len(bands))}
        else:  # MODIS
            coll = (ee.ImageCollection("MODIS/061/MOD09GA")
                    .filterBounds(pt)
                    .filterDate(t.advance(-3,"day"), t.advance(1,"day"))
                    .sort("system:time_start", False))
            if coll.size().getInfo() == 0: return {}
            img    = coll.first()
            bands  = ["sur_refl_b01","sur_refl_b02","sur_refl_b03","sur_refl_b04","sur_refl_b05","sur_refl_b06","sur_refl_b07"]
            labels = {"sur_refl_b01":"645nm","sur_refl_b02":"858nm","sur_refl_b03":"469nm",
                      "sur_refl_b04":"555nm","sur_refl_b05":"1240nm","sur_refl_b06":"1640nm","sur_refl_b07":"2130nm"}
            vals   = img.select(bands).reduceRegion(ee.Reducer.mean(), pt.buffer(500), 500).getInfo()
            return {labels[b]: round((vals.get(b) or 0)/10000, 5) for b in bands}
    except Exception:
        return {}
