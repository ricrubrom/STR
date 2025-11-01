from typing import List, Optional
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from tensorflow.keras.models import load_model
import sys
FUENTES_DIR = "/lib"
sys.path.append(FUENTES_DIR)
from lib.preprocessing import preprocess_data
from sklearn.preprocessing import StandardScaler


class PredictRequest(BaseModel):
    # Opción A: enviar una lista de instancias ya preprocesadas/escaladas
    instances: Optional[List[List[float]]] = None

    # Opción B: enviar los campos individuales para un solo registro (todos opcionales)
    date: Optional[str] = Field(None, description="Timestamp ISO8601, e.g., '2025-03-10T03:00Z'")
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rain: Optional[float] = None
    snow: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    clouds: Optional[float] = None
    sunrise: Optional[int] = Field(None, description="Hora de amanecer en UNIX timestamp (UTC)")
    sunset: Optional[int] = Field(None, description="Hora de atardecer en UNIX timestamp (UTC)")
    working_day: Optional[bool] = None
    holiday: Optional[bool] = None
    
# Ejemplo de payload:
# {
#     "date": "2025-03-10T03:00Z",
#     "temperature": 16.05,
#     "humidity": 73,
#     "rain": 0.0,
#     "snow": 0.0,
#     "pressure": 1022.0,
#     "wind_speed": 4.92,
#     "wind_direction": 150,
#     "clouds": 9,
#     "sunrise": 1741599951,
#     "sunset": 1741645084,
#     "working_day": true,
#     "holiday": false
# }

app = FastAPI(title="Energy Consumption Inference")

def find_and_load_model(models_dir: str = "models"):
    # Buscar variantes de guardado comunes
    candidates = [
        os.path.join(models_dir, "energy_consumption_model.keras"),    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                model = load_model(c)
                return model
            except Exception as e:
                # Intentar la siguiente opción
                print(f"No se pudo cargar {c}: {e}")
    raise FileNotFoundError("No se encontró un modelo válido en 'models/'. Asegurate de guardar el modelo allí.")

# Cargar modelo al iniciar la app (mejor para rendimiento)
MODEL = None

@app.on_event("startup")
def startup_event():
    global MODEL
    try:
        MODEL = find_and_load_model()
        print("Modelo cargado correctamente.")
    except FileNotFoundError as e:
        MODEL = None
        print(str(e))


@app.get("/health")
def health():
    return {
        "model_loaded": MODEL is not None,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado en el servidor. Guarda el modelo en models/ y reiniciá el servicio.")
    # Comportamiento especial: si el cliente manda 'generation', devolverlo literal y terminar.
    
    try:
        if req.instances is not None:
            # Convertir a numpy array y validar forma
            X = np.array(req.instances, dtype=float)
            if X.ndim != 2:
                raise HTTPException(status_code=400, detail="'instances' debe ser una lista de vectores (2D). Ej: [[f1,f2,...],[...]]")

            # Asumimos que el cliente ya envió features en el orden correcto y ya preparados.
            X_np = X

        else:
            # Construir una instancia única a partir de campos individuales. El orden de fields
            # usado en el DataFrame no importa aquí porque la función `preprocess_data` maneja columnas por nombre.
            required_fields = [
                "date",
                "temperature",
                "humidity",
                "rain",
                "snow",
                "pressure",
                "wind_speed",
                "wind_direction",
                "clouds",
                "sunrise",
                "sunset",
                "working_day",
                "holiday",
            ]
            # Validar presencia de campos
            missing = [f for f in required_fields if getattr(req, f, None) is None]
            if missing:
                raise HTTPException(status_code=400, detail=f"Faltan campos para construir la instancia: {missing}. O enviá 'instances'.")

            # Construir DataFrame con una sola fila
            row = {
                "date": [req.date],
                "temperature": [req.temperature],
                "humidity": [req.humidity],
                "rain": [req.rain],
                "snow": [req.snow],
                "pressure": [req.pressure],
                "wind_speed": [req.wind_speed],
                "wind_direction": [req.wind_direction],
                "clouds": [req.clouds],
                "sunrise": [req.sunrise],
                "sunset": [req.sunset],
                "working_day": [req.working_day],
                "holiday": [req.holiday],
            }
            df = pd.DataFrame(row)

            # Aplicar preprocesamiento (devuelve DataFrame)
            df_proc = preprocess_data(df)

            # Convertir a numpy array X
            X_np = df_proc.values

        # Escalado y predicción (X_np es numpy.ndarray)
        x_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X_np)
        preds = MODEL.predict(X_scaled)

    except HTTPException:
        # Re-lanzar errores de validación HTTP tal cual
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

    # Aplanar y devolver
    preds_list = np.array(preds).reshape(-1).tolist()

    return {"predictions": preds_list, "n": len(preds_list)}


if __name__ == "__main__":
    # Ejecutar con: python inference_service.py
    import uvicorn
    uvicorn.run("inference_service:app", host="0.0.0.0", port=8000)
