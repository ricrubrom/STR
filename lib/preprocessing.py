import pandas as pd
import numpy as np
from typing import Tuple

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()

    # Convertir la primera columna a datetime UTC
    col0 = data.columns[0]

    if data[col0].isna().any():
        print("⚠️ Algunas fechas no se pudieron convertir correctamente.")

    data[col0] = pd.to_datetime(data[col0], utc=True, errors="coerce")

    print(data[col0][0:5])

    # --- Cálculo de "daylight" ---
    # sunrise y sunset están en epoch unix UTC, por lo tanto convertimos también:
    data["sunrise_dt"] = pd.to_datetime(data["sunrise"], unit="s", utc=True)
    data["sunset_dt"]  = pd.to_datetime(data["sunset"],  unit="s", utc=True)

    print(data["sunrise_dt"][0:5])
    print(data["sunset_dt"][0:5])

    # Determinar si hay luz solar (1 si entre sunrise y sunset)
    data["daylight"] = (
        (data[col0] >= data["sunrise_dt"]) &
        (data[col0] <= data["sunset_dt"])
    ).astype(int)

    # Crear columnas separadas de tiempo
    data["month"]  = data[col0].dt.month
    data["day"]    = data[col0].dt.day
    data["hour"]   = data[col0].dt.hour
    data["minute"] = data[col0].dt.minute

    # Eliminar columnas auxiliares
    data = data.drop(columns=[col0, "sunrise_dt", "sunset_dt"])

    # Convertir viento a componentes vectoriales
    data["wind_x"] = data["wind_speed"] * np.cos(np.deg2rad(data["wind_direction"]))
    data["wind_y"] = data["wind_speed"] * np.sin(np.deg2rad(data["wind_direction"]))
    data = data.drop(columns=["wind_direction", "wind_speed"])

    # Booleanos a enteros
    data["working_day"] = data["working_day"].astype(int)
    data["holiday"] = data["holiday"].astype(int)

    return data


