# MC51_log_analyser_v3_st.py
# python3.12 -m streamlit run /Users/hayato44/Desktop/python/SUZUKA/MC51_log_analyser_v3_st.py 

import io
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# ====== 定数（元コード踏襲） ======
NE = [0, 5000, 6000, 7000, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 18000, 18001]
TH = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98, 100, 101]

TARGET_AF = 12.7
UPPER_LAF = 13.3
CHANGE_RATE = 0.2  # 1%あたりの変化量

# ====== カラーマップ（元コード踏襲） ======
COLORS = ["#606060", "#ff7f7f", "#ffbf7f", "#ffff7f", "#7fff7f", "#7fffff", "#7fbfff", "#bf7fff", "#e0e0e0"]
CMAP = LinearSegmentedColormap.from_list("custom_af", COLORS, N=len(COLORS))

green_min = TARGET_AF - 0.1
green_max = TARGET_AF + 0.1
boundaries = [
    green_min - 4*CHANGE_RATE, green_min - 3*CHANGE_RATE, green_min - 2*CHANGE_RATE, green_min - CHANGE_RATE,
    green_min, green_max,
    green_max + CHANGE_RATE, green_max + 2*CHANGE_RATE, green_max + 3*CHANGE_RATE, green_max + 4*CHANGE_RATE
]
NORM = BoundaryNorm(boundaries, len(COLORS), clip=True)

# ====== セッションステート 初期化 ======
if "df" not in st.session_state:
    st.session_state.df = None
if "target_lap" not in st.session_state:
    st.session_state.target_lap = None
if "heatmap" not in st.session_state:
    st.session_state.heatmap = None
if "last_sensor" not in st.session_state:
    st.session_state.last_sensor = None

# ====== ユーティリティ ======
def compute_heatmap(df: pd.DataFrame, sensor_col: str):
    """
    NE(回転数)×TH(スロットル)のビンごとに sensor_col の平均を計算し、12x18 行列を返す
    """
    required = {"NE", "THPTC", sensor_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {', '.join(sorted(missing))}")

    # UPPER_LAFでフィルタ
    df = df[df[sensor_col] < UPPER_LAF]

    # 行列をNaNで初期化
    mat = np.full((len(TH)-1, len(NE)-1), np.nan, dtype=float)

    # 各ビンで平均を計算
    for i in range(len(NE)-1):
        ne_low, ne_high = NE[i], NE[i+1]
        for j in range(len(TH)-1):
            th_low, th_high = TH[j], TH[j+1]
            mask = (
                (df["NE"] >= ne_low) & (df["NE"] <= ne_high) &
                (df["THPTC"] >= th_low) & (df["THPTC"] <= th_high)
            )
            sub = df.loc[mask, sensor_col]
            if not sub.empty:
                mat[j, i] = round(float(sub.mean()), 2)

    # 17列目のデータを18列目にコピー（元コード踏襲）
    if mat.shape[1] >= 18:
        mat[:, 17] = mat[:, 16]

    return mat

def style_af_dataframe(df_values: np.ndarray):
    """
    heatmap用のStylerを返す（BoundaryNorm + custom colormap）
    """
    df = pd.DataFrame(df_values, index=[str(v) for v in TH[:-1]], columns=[str(v) for v in NE[:-1]])

    def bg_css(val):
        if pd.isna(val):
            return "text-align: center;"
        idx = NORM([val])[0]
        idx = max(0, min(idx, len(COLORS)-1))
        color = COLORS[idx]
        return f"background-color: {color}; text-align: center;"

    styler = (
        df.style
        .format(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        .applymap(lambda v: "text-align: center;")
        .applymap(bg_css)
    )
    return styler

def style_correction_dataframe(heatmap: np.ndarray):
    """
    補正値 (実測 - TARGET_AF)/CHANGE_RATE を整数丸めして表示。0以外を薄緑でハイライト。
    """
    corr = (heatmap - TARGET_AF) / CHANGE_RATE
    corr_df = pd.DataFrame(corr, index=[str(v) for v in TH[:-1]], columns=[str(v) for v in NE[:-1]])

    def fmt_int(v):
        if pd.isna(v):
            return ""
        return f"{int(round(v, 0))}"

    def highlight(v):
        if pd.isna(v):
            return "text-align: center;"
        val = int(round(v, 0))
        if val != 0:
            return "background-color: #7fff7f; text-align: center;"
        return "text-align: center;"

    styler = corr_df.style.format(fmt_int).applymap(highlight)
    return styler

def filter_by_lap(df: pd.DataFrame, target_lap):
    if target_lap is None:
        return df
    return df[df["Lap"] == target_lap]

def extract_stats(df: pd.DataFrame):
    """
    最高速(GPSspd), 水温(TW)のmax/minを返す。列が無ければNone。
    """
    max_spd = None
    tw_max = None
    tw_min = None
    if "GPSspd" in df.columns:
        try:
            max_spd = round(float(df["GPSspd"].max()), 1)
        except Exception:
            pass
    if "TW" in df.columns:
        try:
            tw_max = round(float(df["TW"].max()), 1)
            tw_min = round(float(df["TW"].min()), 1)
        except Exception:
            pass
    return max_spd, tw_max, tw_min

def df_to_csv_download(df_vals: np.ndarray, name: str) -> bytes:
    df = pd.DataFrame(df_vals, index=[str(v) for v in TH[:-1]], columns=[str(v) for v in NE[:-1]])
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode("utf-8")


# ====== レイアウト ======
st.set_page_config(page_title="CBR250RR A/F Analyser", layout="wide")
st.title("CBR250RR A/F Analyser")

with st.sidebar:
    st.header("1) CSVを選択")
    uploaded = st.file_uploader("csvを無編集アップロードすること", type=["csv"])
    if uploaded:
        try:
            st.session_state.df = pd.read_csv(uploaded, skiprows=6)
            laps = sorted(st.session_state.df["Lap"].dropna().unique()) if "Lap" in st.session_state.df.columns else []
        except Exception as e:
            st.session_state.df = None
            st.error(f"CSVの読み込みに失敗しました: {e}")

    st.markdown("---")
    st.header("2) ラップ選択")
    target_lap = None
    if st.session_state.df is not None and "Lap" in st.session_state.df.columns:
        options = ["All"] + [str(int(l)) for l in laps]
        sel = st.selectbox("解析するラップ", options, index=0)
        target_lap = None if sel == "All" else int(sel)
    else:
        st.info("Lap列が無い場合は自動的に全データを対象にします。")

    st.markdown("---")
    st.header("3) センサー選択 & 解析")
    col1, col2 = st.columns(2)
    with col1:
        run_laf1 = st.button("LAF1解析を実行")
    with col2:
        run_laf2 = st.button("LAF2解析を実行")

# ====== 解析ロジック ======
if st.session_state.df is None:
    st.info("左のサイドバーからCSVを読み込んでください。")
else:
    df_target = filter_by_lap(st.session_state.df, target_lap)

    triggered = None
    if run_laf1:
        triggered = "LAF1"
    elif run_laf2:
        triggered = "LAF2"

    if triggered:
        sensor_col = triggered
        st.session_state.last_sensor = sensor_col

        need_cols = {"NE", "THPTC", sensor_col}
        missing = need_cols - set(df_target.columns)
        if missing:
            st.error(f"必要な列が不足しています: {', '.join(sorted(missing))}")
        else:
            try:
                heatmap = compute_heatmap(df_target, sensor_col)
                st.session_state.heatmap = heatmap
            except Exception as e:
                st.session_state.heatmap = None
                st.error(f"ヒートマップ計算でエラー: {e}")

    if st.session_state.heatmap is not None:
        max_spd, tw_max, tw_min = extract_stats(df_target)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="最高速 [km/h]", value=f"{max_spd} km/h" if max_spd is not None else "—")
        with m2:
            st.metric(label="水温 MAX [℃]", value=f"{tw_max} ℃" if tw_max is not None else "—")
        with m3:
            st.metric(label="水温 MIN [℃]", value=f"{tw_min} ℃" if tw_min is not None else "—")

        st.subheader(f"{st.session_state.last_sensor} 解析")
        st.dataframe(style_af_dataframe(st.session_state.heatmap),
                     width=1600,
                     height=460)

        st.subheader("A/F補正値")
        st.dataframe(style_correction_dataframe(st.session_state.heatmap),
                     width=1600,
                     height=460)

    else:
        st.info("左のボタンから LAF1 または LAF2 の解析を実行してください。")