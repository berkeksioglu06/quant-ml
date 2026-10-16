"""Eski çok dosyalı ve yeni market_data.py düzenleri için salt okunur giriş."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .common import complete_session, safe_name, sha256


@dataclass
class Source:
    project: Path
    snapshot: str
    layout: str
    clean: Path
    raw: Path
    manifest_file: Path
    universe_file: Path
    quality_file: Path
    manifest: dict
    universe: dict
    statuses: dict
    source_files: dict
    last_session: pd.Timestamp
    global_issues: list

    def load(self, symbol, as_of=None):
        file = self.clean / f"{safe_name(symbol)}.parquet"
        frame = pd.read_parquet(file)
        required = {
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "eligible",
            "is_session",
            "splits",
        }
        if missing := required - set(frame.columns):
            raise ValueError(f"{symbol}: eksik kolonlar {sorted(missing)}")
        for name in required - {"eligible", "is_session"}:
            if not is_numeric_dtype(frame[name]):
                raise ValueError(f"{symbol}: {name} sayısal olmalı")
        for name in ("eligible", "is_session"):
            if not is_bool_dtype(frame[name]):
                raise ValueError(f"{symbol}: {name} boolean olmalı")
        if "raw_volume" in frame and not is_numeric_dtype(frame.raw_volume):
            raise ValueError(f"{symbol}: raw_volume sayısal olmalı")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
            raise ValueError(f"{symbol}: boş olmayan DatetimeIndex gerekli")
        if frame.index.tz is not None or frame.index.has_duplicates or frame.index.hasnans:
            raise ValueError(f"{symbol}: tarihler tekil, saat dilimsiz ve eksiksiz olmalı")
        if not frame.index.equals(frame.index.normalize()):
            raise ValueError(f"{symbol}: tarihler gece yarısına normalize edilmeli")
        cutoff = self.last_session
        if self.layout == "legacy":
            at = self.manifest["symbols"][symbol].get("downloaded_at_utc")
            if at:
                cutoff = complete_session(at)
        if as_of is not None:
            cutoff = min(cutoff, pd.Timestamp(as_of))
        return frame.sort_index().loc[:cutoff].copy()

    def check_unchanged(self):
        for file, expected in self.source_files.items():
            if sha256(file) != expected:
                raise ValueError(f"İşlem sırasında kaynak değişmiş: {file}")


def discover(project, snapshot, layout="auto"):
    project = Path(project).resolve()
    if (
        Path(snapshot).name != snapshot
        or snapshot in {".", ".."}
        or not snapshot.startswith("yahoo_")
    ):
        raise ValueError("Snapshot yalnız yahoo_ ile başlayan bir klasör adı olmalı")
    options = {
        "legacy": (
            project / "data/clean" / snapshot,
            project / "data/raw" / snapshot,
            project / "data/reports" / snapshot / "kalite_detay.json",
            project / "config/data/universe.json",
        ),
        "main": (
            project / "data" / snapshot / "clean",
            project / "data" / snapshot,
            project / "data" / snapshot / "issues.csv",
            # Snapshot kendi evren kopyasını taşıyorsa o okunur; yoksa repo kökündeki.
            (project / "data" / snapshot / "universe.json")
            if (project / "data" / snapshot / "universe.json").is_file()
            else project / "universe.json",
        ),
    }
    present = [
        key
        for key, (clean, raw, quality, universe) in options.items()
        if clean.is_dir()
        and (raw / "manifest.json").is_file()
        and quality.is_file()
        and universe.is_file()
    ]
    if layout == "auto":
        if len(present) != 1:
            raise ValueError(
                f"Tek bir kaynak düzeni bulunmalı; bulunan: {present}. --layout ile seçin"
            )
        layout = present[0]
    if layout not in present:
        raise ValueError(f"{layout} düzeninde eksiksiz snapshot/kalite/evren dosyası yok")
    clean, raw, quality, universe_file = options[layout]
    manifest_file = raw / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    universe = json.loads(universe_file.read_text(encoding="utf-8"))
    if not {"symbols", "horizons", "folds", "start", "test"} <= universe.keys():
        raise ValueError("Evren ayarları eksik")
    symbols = [s["symbol"] for s in universe["symbols"] if s["asset"] == "equity"] + ["SPY", "QQQ"]
    if len(set(symbols)) != len(symbols):
        raise ValueError("Tekrarlanan sembol")
    files = {str(p): sha256(p) for p in (manifest_file, universe_file, quality)}
    global_issues = []
    if layout == "legacy":
        q = json.loads(quality.read_text(encoding="utf-8"))
        if q.get("manifest", {}).get("snapshot_id") != snapshot:
            raise ValueError("Kalite raporu farklı snapshot'a ait")
        statuses = {s: q["results"].get(s, {}).get("durum", "UNREVIEWED") for s in symbols}
        last = complete_session(manifest["downloaded_at_utc"])
    else:
        issues = pd.read_csv(quality)
        if not {"symbol", "severity", "code", "msg"} <= set(issues):
            raise ValueError("issues.csv kolonları eksik")
        global_issues = issues.loc[issues.symbol.eq("*")].to_dict("records")
        if any(issue["severity"] == "HATA" for issue in global_issues):
            raise ValueError("Kaynak kalite raporunda genel HATA var; üretim durduruldu")
        statuses = {}
        for symbol in symbols:
            severity = set(issues.loc[issues.symbol.eq(symbol), "severity"])
            statuses[symbol] = (
                "HATA" if "HATA" in severity else "UYARI" if "UYARI" in severity else "OK"
            )
        last = min(
            complete_session(manifest["downloaded_at"]),
            pd.Timestamp(manifest["last_complete_session"]),
        )
    # Bütün ham kayıtları denetle; üretim için gerekli olmayan earnings kayıtları da dahil.
    for symbol, record in manifest["symbols"].items():
        entries = []
        if layout == "legacy":
            if record.get("status") == "OK":
                entries.append((raw / record["file"], record.get("sha256")))
            earnings = record.get("earnings", {})
            if earnings.get("status") == "OK":
                entries.append((raw / earnings["file"], earnings.get("sha256")))
        elif "error" not in record:
            entries.append((raw / "raw" / f"{safe_name(symbol)}.parquet", record.get("sha256")))
        for file, expected in entries:
            if not file.resolve().is_relative_to(raw.resolve()):
                raise ValueError("Ham dosya yolu snapshot klasörünün dışına çıkamaz")
            if not expected or not file.is_file() or sha256(file) != expected:
                raise ValueError(f"Ham dosya bütünlüğü doğrulanamadı: {symbol}")
            files[str(file)] = expected
    for record in manifest.get("auxiliary_files", []):
        file = (raw / record["file"]).resolve()
        expected = record.get("sha256")
        if (
            not file.is_relative_to(raw.resolve())
            or not expected
            or not file.is_file()
            or sha256(file) != expected
        ):
            raise ValueError("Yardımcı kaynak dosyasının bütünlüğü doğrulanamadı")
        files[str(file)] = expected
    for symbol in symbols:
        rec = manifest["symbols"].get(symbol, {})
        if (layout == "legacy" and rec.get("status") != "OK") or (
            layout == "main" and (not rec or "error" in rec)
        ):
            raise ValueError(f"Gerekli sembol indirilmemiş: {symbol}")
        file = clean / f"{safe_name(symbol)}.parquet"
        if not file.is_file():
            raise ValueError(f"Temiz veri eksik: {symbol}")
        files[str(file)] = sha256(file)
    return Source(
        project,
        snapshot,
        layout,
        clean,
        raw,
        manifest_file,
        universe_file,
        quality,
        manifest,
        universe,
        statuses,
        files,
        last,
        global_issues,
    )
