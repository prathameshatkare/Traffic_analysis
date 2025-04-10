from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import joblib
import pandas as pd

app = FastAPI()

# Allow frontend to make requests (for JS fetch calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Load ML model
try:
    model = joblib.load("traffic_model.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# In-memory store
traffic_history = []

class TrafficData(BaseModel):
    CarCount: int
    BikeCount: int
    BusCount: int
    TruckCount: int

@app.post("/predict")
def predict_traffic(data: TrafficData):
    input_df = pd.DataFrame([data.dict()])
    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    traffic_history.append({
        "timestamp": datetime.now().strftime("%H:%M"),
        "CarCount": data.CarCount,
        "BikeCount": data.BikeCount,
        "BusCount": data.BusCount,
        "TruckCount": data.TruckCount,
        "prediction": prediction
    })

    return {"prediction": prediction}

@app.get("/history")
def get_prediction_history():
    return traffic_history

@app.get("/ui")
def read_root():
    return {"message": "🚦 Welcome to the Traffic Prediction API! Use /docs to test the model."}

@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

@app.get("/dashboard")
def serve_dashboard():
    return FileResponse("frontend/dashboard.html")
