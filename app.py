import os
import warnings
from pathlib import Path

import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


class CalibratedXGB:
    """Compatibility wrapper required to load the saved XGBoost artifacts."""

    def __init__(self, xgb_model, iso_reg):
        self.xgb_model = xgb_model
        self.iso_reg = iso_reg

    def predict_proba(self, X):
        raw = self.xgb_model.predict_proba(X)[:, 1]
        calibrated = np.clip(self.iso_reg.predict(raw), 0, 1)
        return np.column_stack([1 - calibrated, calibrated])


st.set_page_config(
    page_title="Crime Risk Predictor",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
LOCAL_MODEL_DIR = Path(os.getenv("IT5006_MODEL_DIR", PROJECT_ROOT / "Saved_Models"))
LOCAL_DATA_DIR = Path(os.getenv("IT5006_NIBRS_DIR", PROJECT_ROOT / "Testing Dataset NIBRS"))
ASSET_CACHE_DIR = BASE_DIR / ".asset_cache"
DEFAULT_GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1bmiKUoxqn7wkc7IPkTaC542zCFf_oFWK?usp=sharing"

ASSET_PATHS = {
    "crime_models": LOCAL_MODEL_DIR / "crime_models_v2.joblib",
    "data_scaler": LOCAL_MODEL_DIR / "data_scaler_v2.joblib",
    "feature_names": LOCAL_MODEL_DIR / "feature_names_v2.joblib",
    "composite_weights": LOCAL_MODEL_DIR / "composite_weights.joblib",
    "incident_data": LOCAL_DATA_DIR / "NIBRS_incident_Illinois.csv",
    "offense_data": LOCAL_DATA_DIR / "NIBRS_OFFENSE_Illinois.csv",
}

TARGET_COLS = ["y_tier1", "y_tier2", "y_tier3", "y_tier4"]
DEFAULT_THRESHOLDS = {"y_tier1": 0.10, "y_tier2": 0.20, "y_tier3": 0.35, "y_tier4": 0.35}
TARGET_TO_TIER = {
    "y_tier1": "Tier 1 - Lethal",
    "y_tier2": "Tier 2 - Personal Violence",
    "y_tier3": "Tier 3 - Property",
    "y_tier4": "Tier 4 - Public Order",
}
TARGET_TO_SHORT_TIER = {
    "y_tier1": "Tier 1",
    "y_tier2": "Tier 2",
    "y_tier3": "Tier 3",
    "y_tier4": "Tier 4",
}
TIERS = list(TARGET_TO_TIER.values())
NEEDS_SCALING = {
    "Logistic Regression": True,
    "Random Forest": False,
    "XGBoost": False,
    "Neural Network": True,
}

# Selected to keep the UI simple while using the best-performing model for each tier
# in the Illinois showcase notebook.
DEPLOYED_MODEL_BY_TARGET = {
    "y_tier1": "Logistic Regression",
    "y_tier2": "Neural Network",
    "y_tier3": "XGBoost",
    "y_tier4": "Neural Network",
}

NIBRS_SEVERITY_MAPPING = {
    "09A": "Tier 1 - Lethal",
    "09B": "Tier 1 - Lethal",
    "11A": "Tier 1 - Lethal",
    "100": "Tier 1 - Lethal",
    "64A": "Tier 1 - Lethal",
    "64B": "Tier 1 - Lethal",
    "120": "Tier 2 - Personal Violence",
    "13A": "Tier 2 - Personal Violence",
    "13B": "Tier 2 - Personal Violence",
    "11B": "Tier 2 - Personal Violence",
    "11C": "Tier 2 - Personal Violence",
    "11D": "Tier 2 - Personal Violence",
    "36A": "Tier 2 - Personal Violence",
    "36B": "Tier 2 - Personal Violence",
    "220": "Tier 3 - Property",
    "23A": "Tier 3 - Property",
    "23C": "Tier 3 - Property",
    "23H": "Tier 3 - Property",
    "240": "Tier 3 - Property",
    "200": "Tier 3 - Property",
    "290": "Tier 3 - Property",
    "13C": "Tier 3 - Property",
    "26A": "Tier 3 - Property",
    "90J": "Tier 3 - Property",
    "35A": "Tier 4 - Public Order",
    "35B": "Tier 4 - Public Order",
    "40A": "Tier 4 - Public Order",
    "520": "Tier 4 - Public Order",
    "39A": "Tier 4 - Public Order",
    "90C": "Tier 4 - Public Order",
    "90D": "Tier 4 - Public Order",
    "90G": "Tier 4 - Public Order",
}


def reset_threshold_state():
    st.session_state["ui_threshold_tier1"] = DEFAULT_THRESHOLDS["y_tier1"]
    st.session_state["ui_threshold_tier2"] = DEFAULT_THRESHOLDS["y_tier2"]
    st.session_state["ui_threshold_tier3"] = DEFAULT_THRESHOLDS["y_tier3"]
    st.session_state["ui_threshold_tier4"] = DEFAULT_THRESHOLDS["y_tier4"]


def get_google_drive_folder_url():
    if "gdrive_assets_folder_url" in st.secrets:
        return st.secrets["gdrive_assets_folder_url"]
    return os.getenv("GDRIVE_ASSETS_FOLDER_URL", DEFAULT_GDRIVE_FOLDER_URL)


@st.cache_data(show_spinner=False)
def download_google_drive_folder(folder_url):
    if not folder_url:
        return {}

    ASSET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_paths = gdown.download_folder(
        url=str(folder_url),
        output=str(ASSET_CACHE_DIR),
        quiet=False,
        use_cookies=False,
    )
    if not downloaded_paths:
        return {}
    return {Path(path).name: str(Path(path)) for path in downloaded_paths}


def resolve_asset_path(asset_name):
    local_path = ASSET_PATHS[asset_name]
    if local_path.exists():
        return str(local_path)

    downloaded_files = download_google_drive_folder(get_google_drive_folder_url())
    cached_path = ASSET_CACHE_DIR / local_path.name

    if local_path.name in downloaded_files:
        return downloaded_files[local_path.name]
    if cached_path.exists():
        return str(cached_path)

    raise FileNotFoundError(
        f"Missing required asset '{local_path.name}' in both local storage and the shared Google Drive folder."
    )


@st.cache_resource(show_spinner=False)
def load_artifacts():
    models = joblib.load(resolve_asset_path("crime_models"))
    scaler = joblib.load(resolve_asset_path("data_scaler"))
    feature_names = joblib.load(resolve_asset_path("feature_names"))
    return models, scaler, feature_names


def add_engineered_features(df_daily, composite_weights):
    df_daily = df_daily.sort_values(["agency_id", "Date"]).copy()

    for idx, tier in enumerate(TIERS, start=1):
        grouped = df_daily.groupby("agency_id")[tier]
        df_daily[f"tier{idx}_lag_7d_mean"] = grouped.transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        df_daily[f"tier{idx}_lag_30d_mean"] = grouped.transform(
            lambda x: x.shift(1).rolling(30, min_periods=1).mean()
        )
        df_daily[f"tier{idx}_lag_90d_mean"] = grouped.transform(
            lambda x: x.shift(1).rolling(90, min_periods=1).mean()
        )

    df_daily = df_daily.fillna(0.0)

    for idx in range(1, 5):
        df_daily[f"tier{idx}_surge_ratio"] = (
            df_daily[f"tier{idx}_lag_7d_mean"]
            / df_daily[f"tier{idx}_lag_30d_mean"].clip(lower=0.01)
        ).clip(upper=10.0)
        df_daily[f"tier{idx}_trend"] = (
            df_daily[f"tier{idx}_lag_7d_mean"] - df_daily[f"tier{idx}_lag_30d_mean"]
        )

    df_daily["violence_property_ratio"] = (
        df_daily["tier2_lag_7d_mean"] / df_daily["tier3_lag_7d_mean"].clip(lower=0.01)
    ).clip(upper=10.0)
    df_daily["disorder_to_violence"] = (
        df_daily["tier4_lag_7d_mean"] / df_daily["tier2_lag_7d_mean"].clip(lower=0.01)
    ).clip(upper=10.0)

    df_daily["month_sin"] = np.sin(2 * np.pi * df_daily["Date"].dt.month / 12)
    df_daily["month_cos"] = np.cos(2 * np.pi * df_daily["Date"].dt.month / 12)
    df_daily["dow_sin"] = np.sin(2 * np.pi * df_daily["Date"].dt.dayofweek / 7)
    df_daily["dow_cos"] = np.cos(2 * np.pi * df_daily["Date"].dt.dayofweek / 7)

    for target_col, weights in composite_weights.items():
        tier_number = target_col[-1]
        df_daily[f"composite_risk_tier{tier_number}"] = (
            df_daily["tier1_lag_7d_mean"] * weights[0]
            + df_daily["tier2_lag_7d_mean"] * weights[1]
            + df_daily["tier3_lag_7d_mean"] * weights[2]
            + df_daily["tier4_lag_7d_mean"] * weights[3]
        )

    for idx, tier in enumerate(TIERS, start=1):
        df_daily[f"y_tier{idx}"] = (df_daily[tier] > 0).astype(int)

    return df_daily


@st.cache_data(show_spinner=False)
def build_prediction_dataset():
    composite_weights = joblib.load(resolve_asset_path("composite_weights"))

    offenses = pd.read_csv(
        resolve_asset_path("offense_data"),
        usecols=["incident_id", "offense_code"],
        dtype={"incident_id": "int64", "offense_code": "string"},
    )
    incidents = pd.read_csv(
        resolve_asset_path("incident_data"),
        usecols=["incident_id", "incident_date", "agency_id"],
        dtype={"incident_id": "int64", "agency_id": "string"},
    )
    incidents["Date"] = pd.to_datetime(incidents["incident_date"], errors="coerce").dt.normalize()

    merged = offenses.merge(
        incidents[["incident_id", "Date", "agency_id"]],
        on="incident_id",
        how="inner",
    )
    merged["Severity_Tier"] = merged["offense_code"].map(NIBRS_SEVERITY_MAPPING)
    merged = merged.dropna(subset=["Date", "agency_id", "Severity_Tier"])

    daily_counts = (
        merged.groupby(["agency_id", "Date", "Severity_Tier"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TIERS, fill_value=0)
        .reset_index()
    )

    agencies = sorted(daily_counts["agency_id"].astype(str).unique().tolist())
    first_date = daily_counts["Date"].min()
    last_actual_date = daily_counts["Date"].max()
    all_dates = pd.date_range(first_date, last_actual_date + pd.Timedelta(days=1), freq="D")

    full_index = pd.MultiIndex.from_product(
        [agencies, all_dates],
        names=["agency_id", "Date"],
    )
    daily_counts = (
        daily_counts.set_index(["agency_id", "Date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    feature_frame = add_engineered_features(daily_counts, composite_weights)
    return feature_frame, agencies, first_date, last_actual_date


def build_prediction_output(models, scaler, feature_names, feature_row, thresholds):
    input_df = feature_row.loc[:, feature_names].astype(float)
    scaled_input = scaler.transform(input_df)
    raw_input = input_df.to_numpy()

    rows = []
    for target_index, target_col in enumerate(TARGET_COLS):
        model_name = DEPLOYED_MODEL_BY_TARGET[target_col]
        estimator = models[model_name][target_index]
        model_input = scaled_input if NEEDS_SCALING[model_name] else raw_input
        probability = float(estimator.predict_proba(model_input)[0, 1])
        rows.append(
            {
                "Tier": TARGET_TO_SHORT_TIER[target_col],
                "Chance next day": probability,
                "Threshold": thresholds[target_col],
                "Flagged at threshold": "Yes" if probability >= thresholds[target_col] else "No",
            }
        )

    return pd.DataFrame(rows)


try:
    models, scaler, feature_names = load_artifacts()
    feature_frame, agencies, first_date, last_actual_date = build_prediction_dataset()
except Exception as exc:
    st.error(f"Failed to load the app assets: {exc}")
    st.write("The shared Google Drive folder should contain these files:")
    for path in ASSET_PATHS.values():
        st.write(f"- {path.name}")
    st.stop()

missing_features = [feature for feature in feature_names if feature not in feature_frame.columns]
if missing_features:
    st.error("The generated Illinois feature set does not match the trained model inputs.")
    for feature in missing_features:
        st.write(f"- {feature}")
    st.stop()

st.title("Crime Risk Predictor")

if st.session_state.get("reset_thresholds_pending"):
    reset_threshold_state()
    st.session_state["reset_thresholds_pending"] = False

if "ui_threshold_tier1" not in st.session_state:
    reset_threshold_state()

control_col, info_col = st.columns([1.1, 0.9])

with control_col:
    selected_agency = st.selectbox("Agency ID", agencies, index=0)
    selected_date = st.date_input(
        "Date",
        value=first_date.date(),
        min_value=first_date.date(),
        max_value=last_actual_date.date(),
    )
    run_prediction = st.button("Predict next day", type="primary", use_container_width=True)

with info_col:
    deployed_models_df = pd.DataFrame(
        {
            "Tier": [TARGET_TO_TIER[target] for target in TARGET_COLS],
            "Model used": [DEPLOYED_MODEL_BY_TARGET[target] for target in TARGET_COLS],
        }
    )
    st.markdown("**Simple deployment setup**")
    st.dataframe(deployed_models_df, use_container_width=True, hide_index=True)
    st.markdown("**Prediction thresholds**")
    threshold_col1, threshold_col2 = st.columns(2)
    with threshold_col1:
        threshold_tier1 = st.number_input("Tier 1 threshold", min_value=0.0, max_value=1.0, key="ui_threshold_tier1", step=0.01, format="%.2f")
        threshold_tier2 = st.number_input("Tier 2 threshold", min_value=0.0, max_value=1.0, key="ui_threshold_tier2", step=0.01, format="%.2f")
    with threshold_col2:
        threshold_tier3 = st.number_input("Tier 3 threshold", min_value=0.0, max_value=1.0, key="ui_threshold_tier3", step=0.01, format="%.2f")
        threshold_tier4 = st.number_input("Tier 4 threshold", min_value=0.0, max_value=1.0, key="ui_threshold_tier4", step=0.01, format="%.2f")

current_thresholds = {
    "y_tier1": float(threshold_tier1),
    "y_tier2": float(threshold_tier2),
    "y_tier3": float(threshold_tier3),
    "y_tier4": float(threshold_tier4),
}

if run_prediction:
    as_of_timestamp = pd.Timestamp(selected_date)
    prediction_date = as_of_timestamp + pd.Timedelta(days=1)

    feature_row = feature_frame.loc[
        (feature_frame["agency_id"] == selected_agency) & (feature_frame["Date"] == prediction_date)
    ]

    if feature_row.empty:
        st.error("No prediction row is available for that agency and date.")
    else:
        prediction_df = build_prediction_output(
            models,
            scaler,
            feature_names,
            feature_row,
            current_thresholds,
        )
        display_df = prediction_df.copy()
        display_df["Chance next day"] = display_df["Chance next day"].map(lambda value: f"{value:.1%}")
        display_df["Threshold"] = display_df["Threshold"].map(lambda value: f"{value:.0%}")

        st.subheader(
            f"Predicted risk for agency {selected_agency} on {prediction_date.date().isoformat()}"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.session_state["reset_thresholds_pending"] = True
        st.rerun()
