"""
Vehicle-Price Estimator API
Çalıştır: uvicorn api:app --reload
"""
from typing import List, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import numpy as np

MODEL_PATH = "model_price.pkl"
SIMILAR_PATH = "model_similar.pkl"   # scikit-learn NearestNeighbors
DATASET_PATH = "dataset_clean.pkl"   # tüm dropdown verileri burada
STATIC_DIR = Path(__file__).parent / "static"

# Load model and data
try:
    model = joblib.load(MODEL_PATH)
    nn = joblib.load(SIMILAR_PATH)
    data = pd.read_pickle(DATASET_PATH)
except FileNotFoundError:
    raise RuntimeError("Eksik pickle. Önce train_model.py çalıştırın.")

app = FastAPI(title="Vehicle-Price Estimator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files and index.html
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def read_index():
    return FileResponse(STATIC_DIR / "index.html")

# Request/response schemas
class VehicleIn(BaseModel):
    Year: int
    Brand: str
    Model: str
    BodyType: str
    Transmission: str
    Kilometres: int
    Engine: Optional[float] = None
    FuelType: Optional[str] = None

    class Config:
        extra = "ignore"

class PredictionOut(BaseModel):
    predicted_price_aud: float

class SimilarVehicle(BaseModel):
    Brand: str
    Model: str
    Year: int
    Price: Optional[float] = None

    class Config:
        extra = "allow"

class SimilarOut(BaseModel):
    similar_vehicles: List[SimilarVehicle]

class BestOut(BaseModel):
    predicted_price_aud: float
    best_vehicle: SimilarVehicle

# Helper to convert input into DataFrame
def _to_frame(v: VehicleIn) -> pd.DataFrame:
    # Use column names matching the cleaned dataset exactly
    mapping = {
        "Brand": "Brand",
        "Model": "Model",
        "Year": "Year",
        "BodyType": "BodyType",
        "Transmission": "Transmission",
        "Kilometres": "Kilometres",
        "Engine": "Engine",
        "FuelType": "FuelType"
    }
    data_dict = v.dict()
    row = {mapping.get(k, k): data_dict[k] for k in data_dict}
    # Ensure all trained features are present
    feature_cols = [col for col in data.columns if col != "Price"]
    full_row = {col: row.get(col, None) for col in feature_cols}
    return pd.DataFrame([full_row])

# API endpoints
@app.get("/api/options/{field}", response_model=List[str])
def get_options(field: str):
    if field not in data.columns:
        raise HTTPException(400, f"Geçersiz alan: {field}")
    return sorted(data[field].dropna().astype(str).unique().tolist())

@app.get("/api/models", response_model=List[str])
def get_models(brand: str = Query(..., min_length=1)):
    col = "Brand" if "Brand" in data.columns else "Make"
    models = data[data[col] == brand]["Model"].dropna().unique().tolist()
    if not models:
        raise HTTPException(404, f"{brand} markasına ait model bulunamadı")
    return sorted(models)

@app.post("/api/predict", response_model=PredictionOut)
def predict(v: VehicleIn):
    try:
        y_hat = model.predict(_to_frame(v))[0]
    except Exception as e:
        raise HTTPException(500, f"Tahmin hatası: {e}")
    return {"predicted_price_aud": float(y_hat)}

@app.post("/api/similar", response_model=SimilarOut)
def similar(v: VehicleIn, k: int = Query(5, ge=1, le=20)):
    try:
        # Transform input for similarity search
        X_vec = model.named_steps['pre'].transform(_to_frame(v))
        if hasattr(X_vec, "toarray"):
            X_vec = X_vec.toarray()
        X_vec = np.nan_to_num(X_vec)
        _, idx = nn.kneighbors(X_vec, n_neighbors=k)
        rows = data.iloc[idx[0]]
    except Exception as e:
        raise HTTPException(500, f"Benzerlik hatası: {e}")
    sims = [
        SimilarVehicle(
            Brand=r.get("Brand", r.get("Make")),
            Model=r["Model"],
            Year=int(r["Year"]),
            Price=float(r["Price"]) if pd.notna(r["Price"]) else None
        ) for _, r in rows.iterrows()
    ]
    return {"similar_vehicles": sims}

@app.post("/api/best", response_model=BestOut)
def best(v: VehicleIn):
    try:
        y_hat = model.predict(_to_frame(v))[0]
        # Transform input for best match search
        X_vec = model.named_steps['pre'].transform(_to_frame(v))
        if hasattr(X_vec, "toarray"):
            X_vec = X_vec.toarray()
        X_vec = np.nan_to_num(X_vec)
        _, idx = nn.kneighbors(X_vec, n_neighbors=1)
        r = data.iloc[idx[0][0]]
        row_dict = r.to_dict()
        sanitized = {
            k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
            for k, v in row_dict.items()
        }
        best_vehicle = SimilarVehicle(**sanitized)
    except Exception as e:
        raise HTTPException(500, f"Best match hatası: {e}")
    return {"predicted_price_aud": float(y_hat), "best_vehicle": best_vehicle}