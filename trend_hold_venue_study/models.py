"""Shared result schema; account inputs remain independent snapshots."""
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class Result:
    minute: pd.DataFrame
    daily_equity: pd.Series
    daily_returns: pd.Series
    orders: pd.DataFrame
    decisions: pd.DataFrame
    funding: pd.DataFrame
    metadata: dict
    attribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    pressure_events: pd.DataFrame = field(default_factory=pd.DataFrame)
