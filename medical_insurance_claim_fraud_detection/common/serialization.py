"""JSON serialization helpers."""
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime.datetime, datetime.date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)

def safe_json_dump(data, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_default)
