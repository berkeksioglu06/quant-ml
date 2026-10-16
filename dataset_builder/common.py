"""Proje deposundan bağımsız seans ve dosya yardımcıları."""

import hashlib
from functools import lru_cache
from pathlib import Path

import exchange_calendars as xc
import pandas as pd


@lru_cache(maxsize=1)
def calendar():
    return xc.get_calendar("XNYS", start="2000-01-01")


def us_sessions(start, end):
    return calendar().sessions_in_range(start, end).tz_localize(None), "XNYS"


def complete_session(at):
    at = pd.Timestamp(at)
    if at.tzinfo is None:
        raise ValueError("İndirme zamanı saat dilimi içermeli")
    at = at.tz_convert("UTC")
    day = at.tz_convert("America/New_York").tz_localize(None).normalize()
    sessions, _ = us_sessions(day - pd.Timedelta(days=14), day)
    for session in reversed(sessions):
        if at >= calendar().session_close(session) + pd.Timedelta(minutes=15):
            return session
    raise ValueError("Tamamlanmış son seans bulunamadı")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(symbol):
    return symbol.replace("^", "_").replace("/", "_")
