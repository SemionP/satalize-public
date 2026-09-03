"""MEDI Platform — Configuration & Constants"""

import math
from dataclasses import dataclass, field
from typing import Optional

# =============================================================================
# MEDI Risk Engine
# =============================================================================
_T = {
    "wqi":             {"low": 70, "mid": 50, "high": 35},
    "turbidity":       {"low": 0.3, "mid": 0.55, "high": 0.75},
    "chlorophyll":     {"low": 0.25, "mid": 0.5, "high": 0.7},
    "sst_anomaly":     {"low": 1.5,  "mid": 3.0, "high": 5.0},
}

PROFILES = {
    "Port Operations":       {"signals": ["wqi","turbidity"], "weights": [0.5,0.5], "description": "Vessel traffic, discharge risk, and water intake quality."},
    "Beach Safety":          {"signals": ["wqi","chlorophyll"], "weights": [0.6,0.4], "description": "Bathing water quality and algae/bloom risk."},
    "Aquaculture":           {"signals": ["wqi","chlorophyll","sst_anomaly"], "weights": [0.35,0.40,0.25], "description": "Bloom conditions, oxygen stress, feed disruption."},
    "ESG Compliance":        {"signals": ["wqi","turbidity","chlorophyll"], "weights": [0.4,0.3,0.3], "description": "Broad environmental footprint monitoring."},
    "Maritime Surveillance": {"signals": ["wqi","turbidity"], "weights": [0.4,0.6], "description": "Discharge events and water anomalies."},
}

@dataclass
class SignalReading:
    name: str; value: float
    raw_value: Optional[float] = None; unit: str = ""
    age_days: float = 0.0; confidence: float = 1.0

@dataclass
class MEDIResult:
    risk_score: float; risk_level: str; risk_color: str; trend: str
    trend_delta: Optional[float] = None; confidence: float = 0.0
    drivers: list = field(default_factory=list)
    profile: str = ""; explanation: str = ""; recommendation: str = ""; zone: str = ""

def _normalize_wqi(v): return max(0.0, min(1.0, 1.0 - v/100.0))

def _signal_risk(value, t):
    lo, mid, hi = t["low"], t["mid"], t["high"]
    if value <= lo:   return value/lo*0.33
    elif value <= mid: return 0.33+(value-lo)/(mid-lo)*0.34
    elif value <= hi:  return 0.67+(value-mid)/(hi-mid)*0.20
    else:
        excess=(value-hi)/(1.0-hi+1e-6)
        return 0.87+0.13*(1-math.exp(-3*excess))

def _confidence_from_signals(signals):
    if not signals: return 0.0
    return round(sum(s.confidence*math.exp(-0.2*s.age_days) for s in signals)/len(signals), 2)

def _detect_drivers(signal_risks, threshold=0.45):
    labels = {"wqi":"water quality degradation","turbidity":"turbidity anomaly",
              "chlorophyll":"algae/bloom signal","sst_anomaly":"SST anomaly"}
    drivers = [(labels.get(k,k),v) for k,v in signal_risks.items() if v>=threshold]
    drivers.sort(key=lambda x: x[1], reverse=True)
    return [d[0] for d in drivers]

def compute_medi(signals, profile_name, previous_score=None, zone=""):
    profile = PROFILES.get(profile_name, PROFILES["Beach Safety"])
    ps, pw  = profile["signals"], profile["weights"]
    signal_risks = {}; active = []
    for sn, w in zip(ps, pw):
        r = signals.get(sn)
        if r is None: continue
        val = _normalize_wqi(r.value) if sn=="wqi" else r.value
        val = max(0.0, min(1.0, val))
        signal_risks[sn] = _signal_risk(val, _T.get(sn, {"low":0.3,"mid":0.55,"high":0.75}))
        active.append(r)
    if not signal_risks:
        return MEDIResult(0,"UNKNOWN","#888888","STABLE",confidence=0.0,profile=profile_name,zone=zone)
    tw = sum(w for sn,w in zip(ps,pw) if sn in signal_risks)
    ws = sum(signal_risks[sn]*w for sn,w in zip(ps,pw) if sn in signal_risks)
    base = ws/tw if tw>0 else 0.0
    mx = max(signal_risks.values())
    if mx>0.85: base=base*0.6+mx*0.4
    score = round(base*100, 1)
    if score<25:   level,color="LOW","#1ecb7b"
    elif score<45: level,color="MODERATE","#7ecb1e"
    elif score<62: level,color="ELEVATED","#f0a500"
    elif score<78: level,color="HIGH","#e07b00"
    else:          level,color="CRITICAL","#e03c3c"
    if previous_score is None: trend,delta="STABLE",None
    else:
        delta=round(score-previous_score,1)
        trend="RISING" if delta>4 else "FALLING" if delta<-4 else "STABLE"
    return MEDIResult(score,level,color,trend,delta,
                      _confidence_from_signals(active),
                      _detect_drivers(signal_risks),profile_name,"","",zone)


# Geometries defined lazily — ee.Geometry created at runtime
HAIFA_BBOX_COORDS = [34.20, 31.20, 35.20, 33.20]
ISRAEL_CLIP_COORDS = [[34.95,33.10],[34.55,33.10],[34.15,32.50],[34.10,32.00],
    [34.15,31.50],[34.50,31.25],[34.75,31.25],[34.95,31.30],[35.02,31.60],[35.00,32.10],[35.05,32.60],[35.10,33.10],[34.95,33.10]]

BEACHES = [
    {"name":"Rosh HaNikra","lat":33.0765,"lon":35.0983},
    {"name":"Nahariya","lat":33.0048,"lon":35.0832},
    {"name":"Acre","lat":32.9280,"lon":35.0680},
    {"name":"Haifa North","lat":32.8380,"lon":34.9820},
    {"name":"Atlit","lat":32.6892,"lon":34.9368},
    {"name":"Caesarea","lat":32.4948,"lon":34.8912},
    {"name":"Netanya","lat":32.3318,"lon":34.8512},
    {"name":"Herzliya","lat":32.1648,"lon":34.7962},
    {"name":"Tel Aviv Center","lat":32.0798,"lon":34.7618},
    {"name":"Ashdod","lat":31.7848,"lon":34.6248},
    {"name":"Ashkelon","lat":31.6548,"lon":34.5448},
    {"name":"Zikim","lat":31.6098,"lon":34.5198},
]

# =============================================================================
# Port Zones - for MEDI Port Analysis
# =============================================================================
PORTS = {
    "🚢 Haifa Port": {
        "lat": 32.8230, "lon": 35.0020,
        "bbox": {"_type": "rect", "coords": [34.94, 32.78, 35.06, 32.87]},
        "radius_km": 5,
        "description": "Major Mediterranean cargo & passenger port",
        "atm_coords": (32.82, 35.00),
    },
    "⚓ Ashdod Port": {
        "lat": 31.8167, "lon": 34.6500,
        "bbox": {"_type": "rect", "coords": [34.60, 31.77, 34.70, 31.86]},
        "radius_km": 4,
        "description": "Israel's largest cargo port",
        "atm_coords": (31.82, 34.65),
    },
    "🐠 Eilat Port": {
        "lat": 29.5510, "lon": 34.9480,
        "bbox": {"_type": "rect", "coords": [34.91, 29.51, 34.99, 29.59]},
        "radius_km": 3,
        "description": "Red Sea port - coral reef proximity",
        "atm_coords": (29.55, 34.95),
    },
}


# =============================================================================
# Maritime Zone Polygons - Offshore areas per city (sea only)
# Each polygon: ~3-5km offshore, covers city coastal stretch
# =============================================================================
MARITIME_ZONES = {
    "Nahariya":  {"_type": "poly", "coords": [[
        [34.88, 33.00], [34.95, 33.00], [34.95, 33.05], [34.88, 33.05]
    ]]},
    "Acre":      {"_type": "poly", "coords": [[
        [34.90, 32.90], [34.97, 32.90], [34.97, 32.95], [34.90, 32.95]
    ]]},
    "Krayot":    {"_type": "poly", "coords": [[
        [34.92, 32.83], [34.98, 32.83], [34.98, 32.89], [34.92, 32.89]
    ]]},
    "Haifa":     {"_type": "poly", "coords": [[
        [34.88, 32.78], [34.97, 32.78], [34.97, 32.84], [34.88, 32.84]
    ]]},
    "Atlit":     {"_type": "poly", "coords": [[
        [34.88, 32.67], [34.95, 32.67], [34.95, 32.72], [34.88, 32.72]
    ]]},
    "Caesarea":  {"_type": "poly", "coords": [[
        [34.85, 32.47], [34.93, 32.47], [34.93, 32.53], [34.85, 32.53]
    ]]},
    "Hadera":    {"_type": "poly", "coords": [[
        [34.84, 32.42], [34.92, 32.42], [34.92, 32.47], [34.84, 32.47]
    ]]},
    "Netanya":   {"_type": "poly", "coords": [[
        [34.82, 32.28], [34.90, 32.28], [34.90, 32.35], [34.82, 32.35]
    ]]},
    "Herzliya":  {"_type": "poly", "coords": [[
        [34.77, 32.14], [34.85, 32.14], [34.85, 32.20], [34.77, 32.20]
    ]]},
    "Tel Aviv":  {"_type": "poly", "coords": [[
        [34.73, 32.04], [34.81, 32.04], [34.81, 32.12], [34.73, 32.12]
    ]]},
    "Palmahim":  {"_type": "poly", "coords": [[
        [34.68, 31.90], [34.76, 31.90], [34.76, 31.96], [34.68, 31.96]
    ]]},
    "Ashdod":    {"_type": "poly", "coords": [[
        [34.60, 31.77], [34.68, 31.77], [34.68, 31.84], [34.60, 31.84]
    ]]},
    "Ashkelon":  {"_type": "poly", "coords": [[
        [34.52, 31.63], [34.60, 31.63], [34.60, 31.69], [34.52, 31.69]
    ]]},
}

# Representative point for each city (for map marker)
CITY_POINTS = {
    "Nahariya": {"lat": 33.020, "lon": 34.915},
    "Acre":     {"lat": 32.924, "lon": 34.935},
    "Krayot":   {"lat": 32.860, "lon": 34.950},
    "Haifa":    {"lat": 32.810, "lon": 34.925},
    "Atlit":    {"lat": 32.690, "lon": 34.915},
    "Caesarea": {"lat": 32.500, "lon": 34.890},
    "Hadera":   {"lat": 32.445, "lon": 34.880},
    "Netanya":  {"lat": 32.315, "lon": 34.860},
    "Herzliya": {"lat": 32.170, "lon": 34.810},
    "Tel Aviv": {"lat": 32.080, "lon": 34.770},
    "Palmahim": {"lat": 31.930, "lon": 34.720},
    "Ashdod":   {"lat": 31.805, "lon": 34.640},
    "Ashkelon": {"lat": 31.660, "lon": 34.560},
}

# =============================================================================


PALETTE = ["#1D9E75","#378ADD","#7F77DD","#BA7517","#D4537E","#E24B4A","#639922","#D85A30"]
