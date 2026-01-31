import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class ModelArtifacts:
    """Container for persisted model artifacts."""

    model: RandomForestRegressor
    feature_columns: List[str]


class TemperaturePredictor:
    """Train and run a next-day maximum temperature predictor."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.data = self.load_and_preprocess_data(self.filepath)
        self.model: Optional[RandomForestRegressor] = None
        self.feature_columns: List[str] = []

    @staticmethod
    def _normalize_column(column: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_")

    def _rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize Open-Meteo column names into consistent snake_case."""
        column_map = {
            "time": "date",
            "weathercode_wmo_code": "weathercode",
            "temperature_2m_max_c": "temp_max",
            "temperature_2m_min_c": "temp_min",
            "precipitation_sum_mm": "precipitation",
            "windspeed_10m_max_km_h": "windspeed_max",
            "windgusts_10m_max_km_h": "windgusts_max",
            "shortwave_radiation_sum_mj_m": "radiation_sum",
        }
        normalized = {col: self._normalize_column(col) for col in data.columns}
        rename = {
            original: column_map[normalized_name]
            for original, normalized_name in normalized.items()
            if normalized_name in column_map
        }
        return data.rename(columns=rename)

    def load_and_preprocess_data(self, filepath: Path) -> pd.DataFrame:
        """Load the CSV file and engineer features for model training."""
        data = pd.read_csv(filepath)
        data = self._rename_columns(data)
        required_columns = [
            "date",
            "weathercode",
            "temp_max",
            "temp_min",
            "precipitation",
            "windspeed_max",
            "windgusts_max",
            "radiation_sum",
        ]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date").reset_index(drop=True)
        data["day_of_week"] = data["date"].dt.dayofweek
        data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
        data["month"] = data["date"].dt.month
        data["temp_range"] = data["temp_max"] - data["temp_min"]
        # Predict tomorrow's max temperature using today's features.
        data["target"] = data["temp_max"].shift(-1)

        lag_features = [
            "weathercode",
            "temp_max",
            "temp_min",
            "precipitation",
            "windspeed_max",
            "windgusts_max",
            "radiation_sum",
        ]
        for feature in lag_features:
            data[f"{feature}_lag1"] = data[feature].shift(1)

        data["rolling_mean_3d"] = data["temp_max"].rolling(window=3).mean().shift(1)
        data["rolling_mean_7d"] = data["temp_max"].rolling(window=7).mean().shift(1)
        data["rolling_std_7d"] = data["temp_max"].rolling(window=7).std().shift(1)
        data["day_to_day_change"] = data["temp_max"].diff()

        return data.dropna().reset_index(drop=True)

    def prepare_training_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Return model-ready features and labels."""
        features = self.data.drop(columns=["date", "target"])
        labels = self.data["target"].to_numpy()
        self.feature_columns = list(features.columns)
        return features, labels

    def train_model(self) -> None:
        """Train the Random Forest model and print basic metrics."""
        features, labels = self.prepare_training_data()
        train_x, test_x, train_y, test_y = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        self.model = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            max_depth=12,
            min_samples_leaf=2,
        )
        self.model.fit(train_x, train_y)
        prediction = self.model.predict(test_x)
        mse = mean_squared_error(test_y, prediction)
        mae = mean_absolute_error(test_y, prediction)
        print(f"Training complete. MSE={mse:.2f}, MAE={mae:.2f}")

    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on a held-out split and return error metrics."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        features, labels = self.prepare_training_data()
        train_x, test_x, train_y, test_y = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        prediction = self.model.predict(test_x)
        return {
            "mse": mean_squared_error(test_y, prediction),
            "mae": mean_absolute_error(test_y, prediction),
        }

    def predict_next_day(self) -> float:
        """Predict the next day's maximum temperature."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        features, _ = self.prepare_training_data()
        latest_features = features.iloc[[-1]]
        prediction = float(self.model.predict(latest_features)[0])
        return prediction

    def save_model(self, output_path: Path) -> None:
        """Serialize the trained model and feature metadata."""
        if self.model is None:
            raise ValueError("Model is not trained.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts = ModelArtifacts(model=self.model, feature_columns=self.feature_columns)
        joblib.dump(artifacts, output_path)

    def load_model(self, model_path: Path) -> None:
        """Load a serialized model and feature metadata."""
        artifacts: ModelArtifacts = joblib.load(model_path)
        self.model = artifacts.model
        self.feature_columns = artifacts.feature_columns
