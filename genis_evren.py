"""Kural tabanlı geniş eğitim evreni: kimse hisse seçmez, kural her gün seçer.

    uv run python genis_evren.py symbols      # Nasdaq sembol dizininden aday havuzu
    uv run python genis_evren.py download     # aday havuzunu indir (kaldığı yerden devam eder)
    uv run python genis_evren.py membership   # her gün × hisse için kuralı uygula, liste ve rapor
    uv run python genis_evren.py snapshot     # evrene giren hisseleri market_data kaydına çevir

Aday havuzu: bugün ABD borsalarında işlem gören bütün adi hisseler (ETF, warrant, hak,
unit, tercihli ve tahvil hariç). Kaynak: nasdaqtrader.com sembol dizini.

Kural (sabit; sonuçlara bakılarak değiştirilmez). Bir hisse t gününde evrendedir, eğer
YALNIZ t kapanışına kadarki veriyle:
  1. split'e göre düzeltilmemiş gerçek kapanış > MIN_PRICE
  2. son WINDOW seansın medyan dolar hacmi > MIN_DOLLAR_VOLUME
  3. en az MIN_HISTORY seanslık işlem geçmişi var

Oynaklık kurala bilinçli olarak konmadı: evrenin büyüklüğü piyasa rejimiyle sallanır ve
etkisi dönemden döneme işaret değiştiren değişkene göre evren seçilmiş olur (Aşama 1
Deney 1–4). Oynaklık yalnız bilgi kolonu olarak hesaplanır.

Bilinen sınır: aday havuzu BUGÜN işlem gören hisselerdir. Batıp ya da satın alınıp borsadan
çıkmış hisseler Yahoo'da yoktur, bu yüzden evrende de yoktur (survivorship). Rapor bunu yazar.

Çıktılar (data/ Git'e girmez):
    data/genis_<snapshot>/symbols.csv          aday havuzu ve dizindeki bilgileri
    data/genis_<snapshot>/raw/*.parquet         yfinance tablosu, dokunulmaz
    data/genis_<snapshot>/manifest.json         indirme kayıtları, sha256
    data/genis_<snapshot>/membership.parquet    date × symbol: kural ölçüleri ve in_universe
    data/genis_<snapshot>/EVREN.md              özet rapor
    data/yahoo_genis_<zaman>/                   `snapshot`: evrene en az bir gün giren hisseler +
                                                SPY/QQQ/^VIX, market_data düzeninde (raw, universe.json,
                                                manifest); ardından market_data.build ve
                                                build_dataset çalışır. Günlük üyelik veri setine
                                                girmez; araştırma hattı `--membership` ile uygular.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

import market_data as md

ROOT = Path(__file__).parent
# Eğitim 2016'da başlar; "en az 1 yıl geçmiş" şartının 2016'da da uygulanabilmesi için bir yıl önce.
START = "2015-01-01"
DIRECTORY_URLS = {
    "nasdaqlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "otherlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}

# Kural: bir kez sabitlendi, sonuçlara bakılarak değiştirilmez.
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 10e6
MIN_HISTORY = 252
WINDOW = 60
RULE = {
    "min_price": MIN_PRICE,
    "min_dollar_volume": MIN_DOLLAR_VOLUME,
    "min_history_sessions": MIN_HISTORY,
    "window_sessions": WINDOW,
}

# Adi hisse olmayan menkul kıymetler ada göre ayıklanır.
NON_COMMON = re.compile(
    r"\b(?:warrants?|rights?|units?|preferred|preference|notes?|debentures?|bonds?|"
    r"depositary shares,? each representing a 1/|% |senior|subordinated|"
    r"trust preferred|perpetual|cumulative|convertible|exchange traded|"
    r"fund|municipal|etn)\b",
    re.IGNORECASE,
)


def snapshot_dir(snapshot: str | None) -> Path:
    if snapshot:
        return md.DATA / snapshot
    snaps = sorted(md.DATA.glob("genis_yahoo_*/symbols.csv"))
    if not snaps:
        raise SystemExit("data/ altında genis_yahoo_* yok; önce `symbols` çalıştır.")
    return snaps[-1].parent


# --------------------------------------------------------------------------- aday havuzu
def parse_directory(nasdaq_text: str, other_text: str) -> pd.DataFrame:
    """İki dizin dosyasından adi hisse adayları. Yahoo sembol biçimine çevrilir (BRK.B → BRK-B)."""

    def read(text):
        lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
        return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)

    nasdaq = read(nasdaq_text).rename(columns={"Symbol": "symbol", "Security Name": "name"})
    nasdaq["exchange"] = "Q"
    other = read(other_text).rename(
        columns={"ACT Symbol": "symbol", "Security Name": "name", "Exchange": "exchange"}
    )
    cols = ["symbol", "name", "exchange", "ETF", "Test Issue"]
    pool = pd.concat([nasdaq[cols], other[cols]], ignore_index=True)
    pool["name"] = pool["name"].str.strip()
    keep = (
        pool["ETF"].eq("N")
        & pool["Test Issue"].eq("N")
        & ~pool["symbol"].str.contains(r"[$^=+]", regex=True)
        & ~pool["name"].str.contains(NON_COMMON)
    )
    pool = pool.loc[keep, ["symbol", "name", "exchange"]].copy()
    pool["yahoo"] = pool["symbol"].str.replace(".", "-", regex=False)
    return pool.drop_duplicates("yahoo").sort_values("yahoo").reset_index(drop=True)


def build_symbols(snapshot: str | None = None) -> Path:
    import urllib.request

    now = datetime.now(UTC)
    snap = md.DATA / (snapshot or f"genis_yahoo_{now:%Y%m%dT%H%M%SZ}")
    (snap / "directory").mkdir(parents=True, exist_ok=False)
    texts = {}
    for name, url in DIRECTORY_URLS.items():
        with urllib.request.urlopen(url, timeout=60) as response:
            texts[name] = response.read().decode("utf-8")
        path = snap / "directory" / f"{name}.txt"
        path.write_text(texts[name], encoding="utf-8")
    pool = parse_directory(texts["nasdaqlisted"], texts["otherlisted"])
    pool.to_csv(snap / "symbols.csv", index=False)
    manifest = {
        "source": "Yahoo Finance via yfinance Ticker.history",
        "directory": {
            name: {"url": url, "sha256": md.sha256(snap / "directory" / f"{name}.txt")}
            for name, url in DIRECTORY_URLS.items()
        },
        "created_at": now.isoformat(),
        "last_complete_session": str(md.last_complete_session(now).date()),
        "start": START,
        "options": md.YF_OPTIONS,
        "rule": RULE,
        "symbols": {},
    }
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(pool)} aday → {snap}")
    return snap


# --------------------------------------------------------------------------- indirme
def download(snap: Path, limit: int | None = None, workers: int = 4, pause: float = 0.25) -> None:
    """Kaldığı yerden devam eder: ham dosyası olan veya kalıcı hata almış sembol atlanır."""
    manifest_path = snap / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool = pd.read_csv(snap / "symbols.csv", dtype=str, keep_default_na=False)
    (snap / "raw").mkdir(exist_ok=True)
    todo = [s for s in pool["yahoo"] if s not in manifest["symbols"]]
    if limit is not None:
        todo = todo[:limit]
    lock = threading.Lock()
    done = 0

    def one(sym):
        for attempt in range(4):
            try:
                df = md.fetch(sym, manifest["start"])
                if df.empty:
                    return sym, {"error": "boş tablo"}
                path = snap / "raw" / f"{md.safe(sym)}.parquet"
                df.reset_index().to_parquet(path)
                os.chmod(path, 0o444)
                return sym, {
                    "rows": len(df),
                    "first": str(df.index.min().date()),
                    "last": str(df.index.max().date()),
                    "sha256": md.sha256(path),
                }
            except Exception as exc:  # noqa: BLE001 - tek sembol indirmeyi durdurmaz
                text = repr(exc)
                if "Too Many Requests" in text or "Rate" in text:
                    time.sleep(30 * (attempt + 1))
                    continue
                return sym, {"error": text[:300]}
            finally:
                time.sleep(pause)
        return sym, {"error": "hız sınırı: 4 deneme başarısız"}

    with ThreadPoolExecutor(max_workers=workers) as pool_exec:
        futures = [pool_exec.submit(one, s) for s in todo]
        for future in as_completed(futures):
            sym, record = future.result()
            with lock:
                done += 1
                manifest["symbols"][sym] = record
                if done % 50 == 0 or done == len(todo):
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    ok = sum("sha256" in r for r in manifest["symbols"].values())
                    print(
                        f"[{done}/{len(todo)}] toplam kayıt {len(manifest['symbols'])}, başarılı {ok}",
                        flush=True,
                    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- kural
def rule_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Tek hissenin günlük tablosundan kural ölçüleri. Her değer yalnız t'ye kadarki veriyle.

    ``daily``: market_data.to_daily çıktısı (close split'e göre geri düzeltilmiş, volume aynı
    şekilde; split_factor = t'den SONRAKİ split'lerin çarpımı).
    """
    close = daily["close"].astype(float)
    raw_close = close * daily["split_factor"]  # o gün gerçekte görülen fiyat
    dollar_volume = close * daily["volume"].astype(float)  # split'ten bağımsız
    log_ret = np.log(daily["adj_close"].astype(float)).diff()
    out = pd.DataFrame(index=daily.index)
    out["raw_close"] = raw_close
    out["dollar_volume_med"] = dollar_volume.rolling(WINDOW, min_periods=WINDOW).median()
    out["history_sessions"] = np.arange(1, len(daily) + 1)
    out["volatility"] = log_ret.rolling(WINDOW, min_periods=WINDOW).std(ddof=1) * np.sqrt(252)
    out["in_universe"] = (
        out["raw_close"].gt(MIN_PRICE)
        & out["dollar_volume_med"].gt(MIN_DOLLAR_VOLUME)
        & out["history_sessions"].ge(MIN_HISTORY)
    )
    return out


def membership(snap: Path) -> pd.DataFrame:
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    last_session = pd.Timestamp(manifest["last_complete_session"])
    frames, problems = [], []
    for sym, rec in sorted(manifest["symbols"].items()):
        if "sha256" not in rec:
            continue
        path = snap / "raw" / f"{md.safe(sym)}.parquet"
        if md.sha256(path) != rec["sha256"]:
            raise SystemExit(f"Ham dosya checksum'ı uyuşmuyor: {sym}")
        try:
            daily, _ = md.to_daily(pd.read_parquet(path), last_session)
        except Exception as exc:  # noqa: BLE001 - bozuk tek dosya raporlanır
            problems.append((sym, repr(exc)[:200]))
            continue
        # Son tamamlanmış seansa kadar satırı kalmayan (bayat/çıkmış) hisse: takvim sorgusu NaT alır.
        daily = daily.loc[daily.index.notna()]
        if daily.empty:
            problems.append((sym, "tamamlanmış seans verisi yok"))
            continue
        daily = daily.loc[
            daily.index.isin(
                md.CAL.sessions_in_range(daily.index.min(), daily.index.max()).tz_localize(None)
            )
        ]
        if daily.empty:
            continue
        m = rule_metrics(daily)
        frames.append(m.reset_index().assign(symbol=sym))
    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(snap / "membership.parquet", index=False)
    # Uygulanan kural; manifest'teki kayıt aday havuzu oluşturulduğu andaki kuraldır.
    (snap / "rule.json").write_text(json.dumps(RULE, indent=2), encoding="utf-8")
    if problems:
        pd.DataFrame(problems, columns=["symbol", "error"]).to_csv(
            snap / "membership_problems.csv", index=False
        )
    return table


# --------------------------------------------------------------------------- rapor
def report(snap: Path, table: pd.DataFrame) -> None:
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    pool = pd.read_csv(snap / "symbols.csv", dtype=str, keep_default_na=False).set_index("yahoo")
    ok = [s for s, r in manifest["symbols"].items() if "sha256" in r]
    failed = {s: r["error"] for s, r in manifest["symbols"].items() if "error" in r}
    inu = table.loc[table["in_universe"]]
    last = table["date"].max()
    per_day = inu.groupby("date")["symbol"].size()
    yearly = per_day.groupby(per_day.index.year).agg(["min", "median", "max"])
    ever = inu.groupby("symbol")["date"].agg(first_in="min", last_in="max", days="size")

    today = inu.loc[inu["date"].eq(last)].copy()
    today = today.assign(name=today["symbol"].map(pool["name"]))
    today = today.sort_values("dollar_volume_med", ascending=False)
    today[["symbol", "name", "raw_close", "dollar_volume_med", "volatility"]].to_csv(
        snap / f"liste_{last:%Y%m%d}.csv", index=False
    )
    ever.join(pool["name"]).sort_values("first_in").to_csv(snap / "evrene_girenler.csv")

    current = {s["symbol"] for s in md.UNIVERSE["symbols"] if s["asset"] == "equity"}
    rows_current = [
        (
            s,
            ever.loc[s, "first_in"].date() if s in ever.index else None,
            int(ever.loc[s, "days"]) if s in ever.index else 0,
        )
        for s in sorted(current)
    ]
    lines = [
        f"# Kural tabanlı geniş evren — {snap.name}",
        "",
        (
            f"Aday havuzu: {len(pool)} adi hisse (bugünkü Nasdaq sembol dizini). "
            f"İndirilen: {len(ok)}, indirilemeyen: {len(failed)}."
        ),
        f"Son tamamlanmış seans: {manifest['last_complete_session']}.",
        "",
        "## Kural (sabit)",
        "",
        (
            f"t gününde, yalnız t kapanışına kadarki veriyle: gerçek fiyat > {MIN_PRICE:g}$; "
            f"son {WINDOW} seans medyan dolar hacmi > {MIN_DOLLAR_VOLUME / 1e6:g} M$; "
            f"en az {MIN_HISTORY} seans geçmiş. Oynaklık kurala konmadı, yalnız bilgi kolonu."
        ),
        "",
        (
            "**Bilinen sınır:** aday havuzu bugün işlem gören hisseler. Batmış veya satın "
            "alınıp borsadan çıkmış hisseler yok (survivorship). Geçmiş yıllarda evren "
            "gerçekte olduğundan biraz küçük ve 'hayatta kalanlara' doğru eğilimli."
        ),
        "",
        "## Yıllara göre günlük evren büyüklüğü",
        "",
        "| Yıl | En az | Medyan | En çok |",
        "|---|---:|---:|---:|",
        *[
            f"| {y} | {int(r['min'])} | {int(r['median'])} | {int(r['max'])} |"
            for y, r in yearly.iterrows()
        ],
        "",
        (
            f"Evrene en az bir gün giren hisse: {len(ever)}. "
            f"Son gün ({last.date()}) evrende: {len(today)}."
        ),
        "",
        "## Mevcut 29 hisse bu kurala göre",
        "",
        "| Hisse | Evrene ilk giriş | Evrende gün |",
        "|---|---|---:|",
        *[f"| {s} | {d or '—'} | {n} |" for s, d, n in rows_current],
        "",
        (
            f"Dosyalar: `liste_{last:%Y%m%d}.csv` (son günün evreni), `evrene_girenler.csv`, "
            "`membership.parquet`."
        ),
    ]
    if failed:
        lines += [
            "",
            f"İndirilemeyen {len(failed)} sembolün ilk 20'si: " + ", ".join(sorted(failed)[:20]),
        ]
    (snap / "EVREN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-12:]))


# --------------------------------------------------------------------------- market_data kaydı
CONTEXT = (("SPY", "etf"), ("QQQ", "etf"), ("^VIX", "index_vol"))


def universe_entries(symbols, first_dates, template, start):
    """Kural evreninden market_data/dataset_builder evren kayıtları.

    universe.json'da zaten tanımlı hisseler (eligible_from, beklenen split vb. ile) aynen alınır.
    Diğerleri "genis" rolüyle girer; veri başlangıçtan sonra başlıyorsa beklenen ilk seans
    sağlayıcının ilk çubuğudur (resmi halka arz tarihi değildir).
    """
    known = {s["symbol"]: s for s in template["symbols"] if s["asset"] == "equity"}
    entries = []
    for sym in sorted(symbols):
        if sym in known:
            entries.append(dict(known[sym]))
            continue
        rec = {"symbol": sym, "roles": ["genis"], "asset": "equity"}
        first = pd.Timestamp(first_dates[sym])
        if first > pd.Timestamp(start) + pd.Timedelta(days=5):
            rec["expected_first_date"] = str(first.date())
            rec["first_date_source"] = "yahoo_first_bar"
        entries.append(rec)
    return entries


def export_snapshot(snap: Path, out: Path | None = None) -> Path:
    """Evrene en az bir gün giren hisseleri market_data düzeninde yeni bir kayda kopyalar.

    Ham dosyalar ve sha256'ları değişmez. SPY, QQQ ve ^VIX aynı başlangıçtan indirilir.
    Sonra `market_data.build(out)` temizler ve kalite kontrolünü yapar; HATA alan hisse eğitim
    üyesi olmaz. Günlük üyelik (membership.parquet) kayda kopyalanır; veri seti bütün hisseleri
    üretir, araştırma hattı maskeyi `--membership` ile uygular.
    """
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    table = pd.read_parquet(snap / "membership.parquet", columns=["symbol", "in_universe"])
    ever = sorted(table.loc[table["in_universe"], "symbol"].unique())
    if not ever:
        raise SystemExit("Evrene giren hisse yok")
    now = datetime.now(UTC)
    out = out or md.DATA / f"yahoo_genis_{now:%Y%m%dT%H%M%SZ}"
    (out / "raw").mkdir(parents=True, exist_ok=False)
    records = {}
    for sym in ever:
        src = snap / "raw" / f"{md.safe(sym)}.parquet"
        rec = manifest["symbols"][sym]
        if md.sha256(src) != rec["sha256"]:
            raise SystemExit(f"Ham dosya checksum'ı uyuşmuyor: {sym}")
        dst = out / "raw" / src.name
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o444)
        records[sym] = {k: rec[k] for k in ("rows", "first", "last", "sha256")}
    for sym, _ in CONTEXT:
        df = md.fetch(sym, manifest["start"])
        if df.empty:
            raise SystemExit(f"{sym} indirilemedi")
        path = out / "raw" / f"{md.safe(sym)}.parquet"
        df.reset_index().to_parquet(path)
        os.chmod(path, 0o444)
        records[sym] = {
            "rows": len(df),
            "first": str(df.index.min().date()),
            "last": str(df.index.max().date()),
            "sha256": md.sha256(path),
        }
    shutil.copyfile(snap / "membership.parquet", out / "membership.parquet")

    # Kalite kontrolü başlangıcı seans takviminde arar; takvim dışı bir gün (2015-01-01) ilk seansa çekilir.
    requested = pd.Timestamp(manifest["start"])
    sessions, _ = md.us_sessions(
        max(requested, md.CAL.first_session.tz_localize(None)), requested + pd.Timedelta(days=10)
    )
    start = str(sessions[0].date())
    template = json.loads((ROOT / "universe.json").read_text(encoding="utf-8"))
    keys = (
        "start",
        "calendar",
        "warmup_sessions",
        "min_train_samples",
        "horizons",
        "folds",
        "test",
    )
    universe = {k: template[k] for k in keys}
    universe["version"] = (
        f"Kural tabanlı geniş evren ({snap.name}): evrene en az bir gün giren {len(ever)} hisse "
        f"+ {len(CONTEXT)} bağlam; günlük üyelik membership.parquet"
    )
    universe["rule"] = manifest["rule"]
    universe["_genis_not"] = (
        "Aday havuzu bugün işlem gören hisseler (survivorship). Üyelik her gün yalnız o güne "
        "kadarki veriyle; bu dosya evrene en az bir gün girenlerin tamamını listeler, günlük "
        "maske araştırma hattında uygulanır."
    )
    universe["symbols"] = universe_entries(
        ever, {s: records[s]["first"] for s in ever}, template, start
    ) + [{"symbol": s, "roles": ["baglam"], "asset": a} for s, a in CONTEXT]
    (out / "universe.json").write_text(
        json.dumps(universe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    new_manifest = {
        "source": manifest["source"],
        "downloaded_at": manifest["created_at"],
        "last_complete_session": manifest["last_complete_session"],
        "start": start,
        "options": manifest["options"],
        "versions": {
            p: version(p) for p in ["yfinance", "pandas", "numpy", "pyarrow", "exchange_calendars"]
        },
        "symbols": records,
        "auxiliary_files": [],
        "universe": {
            "file": "universe.json",
            "source": f"{snap.name}/membership.parquet",
            "sha256": md.sha256(out / "universe.json"),
        },
        "genis_source": snap.name,
        "rule": manifest["rule"],
    }
    (out / "manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    print(f"{len(ever)} hisse + {len(CONTEXT)} bağlam → {out}")
    return out


if __name__ == "__main__":
    md.utf8_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["symbols", "download", "membership", "snapshot"])
    ap.add_argument("--snapshot")
    ap.add_argument("--limit", type=int, help="yalnız ilk N sembol (deneme)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, help="snapshot: henüz var olmayan kayıt klasörü")
    a = ap.parse_args()
    if a.cmd == "symbols":
        build_symbols(a.snapshot)
    elif a.cmd == "download":
        download(snapshot_dir(a.snapshot), a.limit, a.workers)
    elif a.cmd == "snapshot":
        out = export_snapshot(snapshot_dir(a.snapshot), a.out)
        md.build(out)
    else:
        snap = snapshot_dir(a.snapshot)
        report(snap, membership(snap))
