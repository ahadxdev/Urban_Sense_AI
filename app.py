from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, os, requests, pandas as pd, warnings
from datetime import datetime
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

class BatchRequest(BaseModel):
    points: list
    state: str
    weather_api_key: str = ""

@app.get("/")
def root():
    return {"status": "Urban Sense AI running", "models": list(models.keys())}

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
            "probability": round(prob, 3),
            "color":       color,
            "risk":        risk,
        })
    return {"segments": results, "state": req.state}
