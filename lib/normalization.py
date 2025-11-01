from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import numpy as np

def normalize_data(X, T, test_size=0.2, random_state=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  # Normalización Z-score para X únicamente
  x_scaler = StandardScaler()
  X_scaled = x_scaler.fit_transform(X)

  # División train-test (X escalado, T en escala original)
  X_train, X_test, T_train, T_test = train_test_split(
      X_scaled, T, test_size=test_size, random_state=random_state
  )

  return X_train, X_test, T_train, T_test