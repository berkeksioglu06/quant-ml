"""Ortak yardımcılar: yollar, evren, takvim, IO."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "data" / "universe.json"
NY_TZ = "America/New_York"


# --------------------------------------------------------------------------- evren
def load_universe(path: Path | str = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        u = json.load(f)
    for s in u["symbols"]:
        s.setdefault("optional", False)
        s.setdefault("expected_splits", [])
    return u


def symbols_with_role(u: dict, *roles: str) -> list[str]:
    return [s["symbol"] for s in u["symbols"] if set(roles) & set(s["roles"])]


def symbol_info(u: dict, symbol: str) -> dict:
    for s in u["symbols"]:
        if s["symbol"] == symbol:
            return s
    raise KeyError(symbol)


def safe_name(symbol: str) -> str:
    """^VIX -> _VIX, BTC-USD -> BTC-USD (dosya adı güvenli)."""
    return symbol.replace("^", "_").replace("/", "_")


# --------------------------------------------------------------------------- takvim
_CAL_CACHE: dict = {}


def us_sessions(start, end) -> tuple[pd.DatetimeIndex, str]:
    """XNYS seans günleri. exchange_calendars yoksa SPY tarihleri kullanılır (çağıran verir)."""
    try:
        import exchange_calendars as xc

        key = "XNYS"
        if key not in _CAL_CACHE:
            _CAL_CACHE[key] = xc.get_calendar("XNYS", start="2000-01-01")
        cal = _CAL_CACHE[key]
        s = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return pd.DatetimeIndex(
            s.tz_localize(None) if s.tz is not None else s
        ), f"exchange_calendars {xc.__version__} XNYS"
    except ImportError:
        return pd.DatetimeIndex([]), "YOK"


SESSION_BUFFER = pd.Timedelta(
    minutes=15
)  # kapanıştan sonra sağlayıcının son mumu kesinleştirmesi için pay


def session_close_utc(day) -> tuple[pd.Timestamp | None, str]:
    """O günün gerçek seans kapanışı (UTC). Erken kapanışları (ör. 13:00) takvimden alır.
    Seans değilse None. exchange_calendars yoksa 16:00 NY varsayılır ve kaynak 'varsayim' döner."""
    day = pd.Timestamp(day).normalize()
    try:
        import exchange_calendars as xc

        if "XNYS" not in _CAL_CACHE:
            _CAL_CACHE["XNYS"] = xc.get_calendar("XNYS", start="2000-01-01")
        cal = _CAL_CACHE["XNYS"]
        if not cal.is_session(day):
            return None, "takvim"
        c = cal.session_close(day)
        return (c.tz_convert("UTC") if c.tzinfo else c.tz_localize("UTC")), "takvim"
    except ImportError:
        if day.weekday() >= 5:
            return None, "varsayim"
        return pd.Timestamp(f"{day.date()} 16:00", tz=NY_TZ).tz_convert("UTC"), "varsayim"


def is_session_complete(day, at_utc) -> bool | None:
    """at_utc anında `day` seansı tamamlanmış mı? Seans değilse None."""
    close, _ = session_close_utc(day)
    if close is None:
        return None
    t = pd.Timestamp(at_utc)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t >= close + SESSION_BUFFER


def last_complete_session(at_utc) -> pd.Timestamp | None:
    t = pd.Timestamp(at_utc)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    today = t.tz_convert(NY_TZ).tz_localize(None).normalize()
    sess, _ = us_sessions(today - pd.Timedelta(days=14), today)
    for d in reversed(list(sess)):
        if is_session_complete(d, t):
            return d
    return None


# --------------------------------------------------------------------------- IO
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def make_readonly(path: Path):
    try:
        os.chmod(path, 0o444)
    except OSError:
        pass


def latest_snapshot(data_dir: Path) -> Path:
    raw = data_dir / "raw"
    snaps = sorted([p for p in raw.glob("yahoo_*") if (p / "manifest.json").exists()])
    if not snaps:
        raise FileNotFoundError(
            f"{raw} altında snapshot yok. Önce `uv run python -m quant_ml.data download`."
        )
    return snaps[-1]


@dataclass
class Paths:
    data_dir: Path
    snapshot_id: str

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw" / self.snapshot_id

    @property
    def clean(self) -> Path:
        return self.data_dir / "clean" / self.snapshot_id

    @property
    def reports(self) -> Path:
        return self.data_dir / "reports" / self.snapshot_id
