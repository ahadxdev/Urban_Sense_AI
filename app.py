import utils.transport
import utils.session
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, os, requests, pandas as pd, warnings
from datetime import datetime
from uuid import uuid4
warnings.filterwarnings('ignore')

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading models...")
models = {}
for f in os.listdir("./models"):
    if f.endswith("_model.pkl"):
        state = f.replace("_model.pkl", "")
        pipeline = joblib.load(f"./models/{f}")
        models[state] = {
            "preprocessor": pipeline.named_steps['preprocessor'],
            "classifier":   pipeline.named_steps['classifier']
        }
        print(f"Loaded {state}")
print(f"Total: {len(models)} models")

print("Loading node features...")
node_features = pd.read_csv("node_features.csv", index_col="node_id")
print(f"Loaded {len(node_features)} nodes")

COLS = ['Start_Lat','Start_Lng','Temperature(F)','Wind_Chill(F)','Humidity(%)',
        'Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)',
        'Weather_Condition','Crossing','Junction','Railway','Stop','Traffic_Signal',
        'Sunrise_Sunset','hour','day','month','is_rush_hour','is_weekend','is_winter']

def get_weather(lat, lng, api_key):
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather",
            params={"lat":lat,"lon":lng,"appid":api_key,"units":"imperial"}, timeout=5)
        w = r.json()
        return {
            "Temperature(F)":    w["main"]["temp"],
            "Wind_Chill(F)":     w["main"].get("feels_like", w["main"]["temp"]),
            "Humidity(%)":       w["main"]["humidity"],
            "Pressure(in)":      w["main"]["pressure"] * 0.02953,
            "Visibility(mi)":    w.get("visibility", 10000) / 1609,
            "Wind_Speed(mph)":   w["wind"]["speed"],
            "Precipitation(in)": w.get("rain",{}).get("1h",0) * 0.0394,
            "Weather_Condition":  w["weather"][0]["main"],
        }
    except:
        return {"Temperature(F)":75.0,"Wind_Chill(F)":75.0,"Humidity(%)":60.0,
                "Pressure(in)":29.92,"Visibility(mi)":10.0,"Wind_Speed(mph)":5.0,
                "Precipitation(in)":0.0,"Weather_Condition":"Clear"}

def get_node_feats(node_id):
    try:
        if node_id in node_features.index:
            nf = node_features.loc[node_id]
            return {"Crossing":int(nf["crossing"]),"Junction":int(nf["junction"]),
                    "Railway":int(nf["railway"]),"Stop":int(nf["stop"]),
                    "Traffic_Signal":int(nf["traffic_signal"])}
    except: pass
    return {"Crossing":0,"Junction":0,"Railway":0,"Stop":0,"Traffic_Signal":0}

def predict_probability(state_key, features_dict):
    if state_key not in models: return 0.0
    df = pd.DataFrame([features_dict])[COLS]
    m = models[state_key]
    X = m["preprocessor"].transform(df)
    proba = m["classifier"].predict_proba(X)[0]
    return float(proba[1])  # probability of being dangerous

def get_risk(prob):
    if prob < 0.3:   return "#4ade80", "Safe"
    elif prob < 0.5: return "#facc15", "Moderate"
    elif prob < 0.7: return "#f97316", "Risky"
    else:            return "#f87171", "Dangerous"

class RoutePoint(BaseModel):
    node_id: int
    lat: float
    lng: float

class RouteCreateRequest(BaseModel):
    start_node: int
    end_node: int
    distance_meters: float
    path: list[RoutePoint]

class RoutePredictRequest(BaseModel):
    state: str = ""
    weather_api_key: str = ""
    
    class Config:
        extra = "ignore"

route_store: dict[str, dict] = {}

def _predict_route_from_points(points: list[RoutePoint], state: str, api_key: str):
    state_key = state.lower() if state else next(iter(models.keys()), "")
    if state_key not in models:
        raise HTTPException(status_code=400, detail=f"Unknown state model: {state}")

    api_key = api_key or os.getenv("WEATHER_API_KEY", "")
    if not points:
        return {"segments": [], "state": state_key}

    mid = points[len(points) // 2]
    weather = get_weather(mid.lat, mid.lng, api_key)
    now = datetime.now()
    time_feats = {
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "is_rush_hour": 1 if now.hour in [7, 8, 9, 16, 17, 18] else 0,
        "is_weekend": 1 if now.weekday() >= 5 else 0,
        "is_winter": 1 if now.month in [12, 1, 2] else 0,
        "Sunrise_Sunset": "Day" if 6 <= now.hour <= 18 else "Night"
    }

    results = []
    for point in points:
        nf = get_node_feats(point.node_id)
        features = {
            "Start_Lat": point.lat,
            "Start_Lng": point.lng,
            **weather,
            **nf,
            **time_feats
        }
        prob = predict_probability(state_key, features)
        color, risk = get_risk(prob)
        results.append({
            "node_id": point.node_id,
            "lat": point.lat,
            "lng": point.lng,
            # "probability": round(prob, 3),
            "risk": risk
            # "risk_color": color
        })

    return {"segments": results, "state": state_key}

class BatchRequest(BaseModel):
    points: list
    state: str
    weather_api_key: str = ""

@app.get("/")
def root():
    return {"status": "Urban Sense AI running", "models": list(models.keys())}

@app.post("/routes", status_code=201)
def create_route(route: RouteCreateRequest):
    route_id = str(uuid4())
    route_store[route_id] = {
        "id": route_id,
        "start_node": route.start_node,
        "end_node": route.end_node,
        "distance_meters": route.distance_meters,
        "path": [point.dict() for point in route.path],
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    return {"route_id": route_id}

@app.get("/routes/{route_id}")
def get_route(route_id: str):
    route = route_store.get(route_id)
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    return route

@app.post("/routes/{route_id}/predict")
def predict_route_by_id(route_id: str, payload: RoutePredictRequest = None):
    route = route_store.get(route_id)
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    points = [RoutePoint(**point) for point in route["path"]]
    state = payload.state if payload and payload.state else ""
    weather_api_key = payload.weather_api_key if payload and payload.weather_api_key else ""
    return _predict_route_from_points(points, state, weather_api_key)

@app.get("/routes/{route_id}/predict")
def get_route_predictions(route_id: str):
    route = route_store.get(route_id)
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    points = [RoutePoint(**point) for point in route["path"]]
    return _predict_route_from_points(points, "", "")

@app.post("/predict-route")
def predict_route(req: BatchRequest):
    api_key = req.weather_api_key or os.getenv("WEATHER_API_KEY", "")
    mid = req.points[len(req.points)//2]
    weather = get_weather(mid["lat"], mid["lng"], api_key)
    now = datetime.now()
    time_feats = {
        "hour": now.hour, "day": now.day, "month": now.month,
        "is_rush_hour": 1 if now.hour in [7,8,9,16,17,18] else 0,
        "is_weekend":   1 if now.weekday() >= 5 else 0,
        "is_winter":    1 if now.month in [12,1,2] else 0,
        "Sunrise_Sunset": "Day" if 6 <= now.hour <= 18 else "Night"
    }
    results = []
    for point in req.points:
        nf = get_node_feats(int(point.get("node_id", 0)))
        features = {
            "Start_Lat": point["lat"], "Start_Lng": point["lng"],
            **weather, **nf, **time_feats
        }
        prob = predict_probability(req.state.lower(), features)
        color, risk = get_risk(prob)
        results.append({
            "node_id":     point.get("node_id"),
            "lat":         point["lat"],
            "lng":         point["lng"],
            "probability": round(prob, 3)
        })
    return {"segments": results, "state": req.state}
