"""Dataset loading. The Robyn demo data (MIT licence, Meta) is downloaded, not committed."""
from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ROBYN_URL = (
    "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/"
    "python/src/robyn/tutorials/resources/dt_simulated_weekly.csv"
)
ROBYN_FILE = RAW / "robyn_dt_simulated_weekly.csv"


@dataclass
class MMMData:
    name: str
    dates: pd.Series
    y: np.ndarray            # (T,) target in currency
    spend: np.ndarray        # (T, C) spend in currency
    controls: np.ndarray     # (T, K)
    channels: list[str]
    control_names: list[str]


def download_robyn(force: bool = False) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    if force or not ROBYN_FILE.exists():
        urllib.request.urlretrieve(ROBYN_URL, ROBYN_FILE)
    return ROBYN_FILE


def load_robyn() -> MMMData:
    df = pd.read_csv(download_robyn(), parse_dates=["DATE"])
    channels = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
    events = pd.get_dummies(df["events"], prefix="event").drop(columns="event_na").astype(float)
    ctrl = pd.concat([df[["competitor_sales_B", "newsletter"]], events], axis=1)
    return MMMData(
        name="robyn",
        dates=df["DATE"],
        y=df["revenue"].to_numpy(float),
        spend=df[channels].to_numpy(float),
        controls=ctrl.to_numpy(float),
        channels=[c.removesuffix("_S") for c in channels],
        control_names=list(ctrl.columns),
    )


def from_simulation(df: pd.DataFrame, channels: list[str]) -> MMMData:
    return MMMData(
        name="simulated",
        dates=df["date"],
        y=df["sales"].to_numpy(float),
        spend=df[channels].to_numpy(float),
        controls=df[["promo", "price_index"]].to_numpy(float),
        channels=list(channels),
        control_names=["promo", "price_index"],
    )
