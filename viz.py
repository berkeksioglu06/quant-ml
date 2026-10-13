"""Veri görselleştirici.

    uv run streamlit run viz.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import market_data as md

st.set_page_config(page_title="quant-ml veri", layout="wide")

ROLES = {"islem10": "İşlem 10", "ana20": "Ana 20", "kontrol8": "Kontrol 8", "nbis": "NBIS", "baglam": "Bağlam"}
TEST_START = pd.Timestamp(md.UNIVERSE["test"]["start"])
SYMBOLS = [s["symbol"] for s in md.UNIVERSE["symbols"]]


@st.cache_data
def load(snap: str, sym: str) -> pd.DataFrame | None:
    p = md.DATA / snap / "clean" / f"{md.safe(sym)}.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_csv(snap: str, name: str) -> pd.DataFrame:
    return pd.read_csv(md.DATA / snap / name)


def label(sym: str) -> str:
    return f"{sym}  ·  {', '.join(ROLES[r] for r in md.info(sym)['roles'])}"


def shade_test(fig, df, **kw):
    if show_test and df.index.max() >= TEST_START:
        fig.add_vrect(x0=max(TEST_START, df.index.min()), x1=df.index.max(), fillcolor="red",
                      opacity=0.06, line_width=0, annotation_text="TEST", annotation_position="top left", **kw)


# --------------------------------------------------------------------------- kenar çubuğu
snaps = sorted(p.parent.name for p in md.DATA.glob("yahoo_*/clean"))
if not snaps:
    st.error("Veri yok. Önce: `uv run python market_data.py all`")
    st.stop()

with st.sidebar:
    snap = st.selectbox("Snapshot", snaps[::-1])
    y0, y1 = st.slider("Dönem", 2016, pd.Timestamp.now().year, (2016, pd.Timestamp.now().year))
    show_test = st.checkbox("2025+ test dönemini göster", True,
                            help="Bakmak serbest; ama model, özellik, eşik veya evren seçiminde kullanılmaz.")
    period = slice(f"{y0}", f"{y1}" if show_test else min(f"{y1}", str(TEST_START.year - 1)))

tab_price, tab_compare, tab_quality = st.tabs(["Fiyat", "Karşılaştır", "Kalite ve kapsama"])

# --------------------------------------------------------------------------- tek sembol
with tab_price:
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    sym = c1.selectbox("Sembol", SYMBOLS, format_func=label)
    scale = c2.radio("Fiyat", ["Yahoo (split düzeltilmiş)", "O gün görülen"], horizontal=True)
    emas = c3.multiselect("EMA", [20, 50, 200], default=[50, 200])
    log = c4.toggle("Log", True)

    full = load(snap, sym)
    if full is None:
        st.warning(f"{sym} bu snapshot'ta yok.")
        st.stop()
    for n in emas:  # tüm seri üzerinde hesapla, sonra kes: dönem başında ısınma hatası olmasın
        full[f"ema{n}"] = full["close"].ewm(span=n, adjust=False).mean()
    df = full.loc[period].copy()
    if df.empty:
        st.warning("Seçilen dönemde veri yok.")
        st.stop()

    if scale == "O gün görülen":
        f = df["raw_close"] / df["close"]  # doğrulanmamış dönemde NaN -> grafikte boşluk
        for c in ["open", "high", "low", "close", *[f"ema{n}" for n in emas]]:
            df[c] = df[c] * f
        df["volume"] = df["raw_volume"]

    info = md.info(sym)
    ret = df["close"].pct_change()
    last = df.iloc[-1]
    m = st.columns(6)
    m[0].metric("Son kapanış", f"{last.close:,.2f}", f"{ret.iloc[-1]:+.2%}")
    m[1].metric("Dönem getirisi", f"{df.close.iloc[-1] / df.close.iloc[0] - 1:+.0%}")
    m[2].metric("Yıllık oynaklık", f"{ret.std() * 252 ** 0.5:.0%}")
    m[3].metric("60g ort. dolar hacmi", f"${(df.close * df.volume).tail(60).mean() / 1e6:,.0f}M")
    m[4].metric("Satır", f"{len(df):,}")
    m[5].metric("Eligible başlangıç", info.get("eligible_from", str(full.index.min().date())))
    if "event" in info:
        st.caption(info["event"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close,
                                 name=sym, showlegend=False), row=1, col=1)
    for n in emas:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"ema{n}"], name=f"EMA {n}", line_width=1.2), row=1, col=1)
    big = df[ret.abs() > md.BIG_MOVE]
    if len(big):
        fig.add_trace(go.Scatter(x=big.index, y=big.close, mode="markers", name=f"|getiri|>{md.BIG_MOVE:.0%}",
                                 marker={"symbol": "circle-open", "size": 11, "color": "orange", "line_width": 2}), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df.volume, name="Hacim", marker_color="rgba(100,110,130,0.5)",
                         showlegend=False), row=2, col=1)

    for d, r in df.loc[~df.splits.isin([0, 1]), "splits"].items():
        fig.add_vline(x=d, line_dash="dot", line_color="purple", annotation_text=f"split {r:g}:1")
    if not df.eligible.all():
        fig.add_vrect(x0=df.index.min(), x1=df.index[~df.eligible].max(), fillcolor="gray", opacity=0.15,
                      line_width=0, annotation_text="eligible değil (SPAC öncesi)", annotation_position="top left")
    shade_test(fig, df)

    gaps = pd.date_range(df.index.min(), df.index.max()).difference(df.index)  # hafta sonu/tatil boşlukları
    fig.update_xaxes(rangebreaks=[{"values": gaps}], rangeslider_visible=False)
    fig.update_yaxes(type="log" if log else "linear", row=1, col=1)
    fig.update_layout(height=680, margin={"t": 30, "b": 10}, legend={"orientation": "h", "y": 1.04})
    st.plotly_chart(fig, use_container_width=True)

    ctx = st.multiselect("Karşısında bağlam (getiri, %)", [s for s in SYMBOLS if s != sym],
                         default=["QQQ"], format_func=label)
    if ctx:
        series = {s: d["adj_close"] for s in [sym, *ctx] if (d := load(snap, s)) is not None}
        rel = pd.DataFrame(series).loc[period].dropna(subset=[sym])
        rel = (rel / rel.apply(lambda c: c.dropna().iloc[0]) - 1) * 100
        st.plotly_chart(px.line(rel, labels={"value": "%", "variable": ""}).update_layout(height=320),
                        use_container_width=True)

# --------------------------------------------------------------------------- çoklu sembol
with tab_compare:
    picks = st.multiselect("Semboller", SYMBOLS, default=["NVDA", "AMD", "MU", "MRVL", "SMH", "QQQ"],
                           format_func=label)
    if len(picks) >= 2:
        closes = pd.DataFrame({s: d["adj_close"] for s in picks if (d := load(snap, s)) is not None}).loc[period]
        rebased = closes / closes.apply(lambda c: c.dropna().iloc[0]) * 100
        fig = px.line(rebased, log_y=True, labels={"value": "100'den başlayarak (temettü dahil)", "variable": ""})
        shade_test(fig, rebased)
        st.plotly_chart(fig.update_layout(height=450), use_container_width=True)

        rets = closes.pct_change(fill_method=None)
        c1, c2 = st.columns(2)
        corr = rets.corr()
        c1.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                  title="Günlük getiri korelasyonu").update_layout(height=450),
                        use_container_width=True)
        win = c2.select_slider("Rolling oynaklık penceresi (seans)", [20, 60, 120, 252], 60)
        vol = rets.rolling(win).std() * 252 ** 0.5 * 100
        c2.plotly_chart(px.line(vol, labels={"value": "yıllık %", "variable": ""},
                                title=f"{win} seanslık oynaklık").update_layout(height=410),
                        use_container_width=True)
    else:
        st.info("En az iki sembol seç.")

# --------------------------------------------------------------------------- kalite
with tab_quality:
    cov = load_csv(snap, "coverage.csv").set_index("symbol")
    folds = [f["name"] for f in md.UNIVERSE["folds"]]
    st.subheader(f"Kapsama: H={max(md.UNIVERSE['horizons'])} eğitim örneği (≥{md.UNIVERSE['min_train_samples']} = yeter)")
    fig = px.imshow(cov[folds] >= md.UNIVERSE["min_train_samples"], color_continuous_scale=["#f4c7c3", "#b7e1cd"],
                    aspect="auto")
    fig.update_traces(text=cov[folds].values, texttemplate="%{text}", hovertemplate="%{y} %{x}: %{text}")
    fig.update_layout(height=26 * len(cov) + 80, coloraxis_showscale=False, margin={"t": 10})
    st.plotly_chart(fig, use_container_width=True)

    issues = load_csv(snap, "issues.csv")
    st.subheader("Bulgular")
    sev = st.multiselect("Önem", ["HATA", "UYARI", "BILGI"], default=["HATA", "UYARI"])
    st.dataframe(issues[issues.severity.isin(sev)], use_container_width=True, hide_index=True)
