# app_nd_factor_report.py
# -*- coding: utf-8 -*-
"""
Streamlit app: ND-standardized factor score reporting for a new subject
- Loads reference dataset (Excel) with DIAG + P01..P52
- Computes factor indices from P-items (per user's factor mapping)
- Standardizes by ND group (z), converts to 0–100 T-like scores (T=50+10*z, clipped)
- Locates subject relative to clinical group centroids (ASD/ADHD/SCD/HR) in Z-space
- Visualizes: bars, radar chart, and distance-based similarity
- Exports a concise JSON/CSV report

Run:
  streamlit run app_nd_factor_report.py

Dependencies:
  pip install streamlit pandas numpy plotly openpyxl scikit-learn

Author: Planapp helper
"""

import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# ---------------------------- UI SETUP ----------------------------
st.set_page_config(page_title="ND-Standardized Factor Report", layout="wide")
st.title("🧠 ND 표준화 기반 요인 리포트 / ND-Standardized Factor Report")
st.caption("참조 데이터로 ND 기준을 산출하고, 신규 대상자의 위치와 0–100 점수를 제공합니다.")

# ---------------------------- FACTOR DEFINITIONS ----------------------------
# User-provided mapping
FACTOR_ITEMS = {
    "Factor1": ["P04","P05","P06","P07","P09","P11","P12","P13","P15","P16","P19","P20","P22","P23","P24","P26","P27","P29","P32","P34","P36","P38","P39","P42","P44","P46","P49","P50","P52"],
    "Factor2": ["P14","P30","P31","P33","P37","P41","P43","P45","P47","P48","P51"],
    "Factor3": ["P08","P10","P15","P18","P21","P25","P26","P29","P34","P40"],
    "Factor4": ["P03","P20","P32"],
}
FACTOR_TITLES = {
    "Factor1": "사회적 의사소통 및 반복행동 / Social Communication & RRB",
    "Factor2": "사회적 인식 및 상호작용 조절 / Social Awareness & Reciprocity",
    "Factor3": "사회적 동기 및 정서표현 / Social Motivation & Emotion",
    "Factor4": "언어적 사회인지 / Verbal Social Cognition",
}
ALL_P = sorted({c for lst in FACTOR_ITEMS.values() for c in lst})
CLINICAL_GROUPS = ["ASD","ADHD","SCD","HR"]

# ---------------------------- HELPERS ----------------------------
def clean_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return s.astype(float)
    x = s.astype(str).str.strip()
    repl = {
        ",":"", "%":"", "−":"-", "–":"-", "—":"-", "±":" ",
        "≥":"", "≤":"", ">":"", "<":"", "=":"",
    }
    for k,v in repl.items():
        x = x.str.replace(k, v, regex=False)
    num = x.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(num, errors="coerce")

def clean_numeric_frame(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = clean_numeric_series(df[c])
    return out

@st.cache_data(show_spinner=False)
def load_reference(file_bytes: bytes, diag_col: str):
    df_raw = pd.read_excel(io.BytesIO(file_bytes))
    if diag_col not in df_raw.columns:
        raise ValueError(f"'{diag_col}' column not found in uploaded file.")
    Ps = clean_numeric_frame(df_raw, ALL_P)
    diag = df_raw[diag_col].astype(str)
    return df_raw, Ps, diag

def compute_factor_index(P_frame: pd.DataFrame, thresh_ratio: float = 0.5):
    idx = pd.DataFrame(index=P_frame.index)
    for fname, items in FACTOR_ITEMS.items():
        present = [c for c in items if c in P_frame.columns]
        sub = P_frame[present]
        cnt = sub.notna().sum(axis=1)
        thresh = max(1, int(np.ceil(thresh_ratio * len(present))))
        avg = sub.mean(axis=1)
        avg[cnt < thresh] = np.nan
        idx[fname] = avg
    return idx

def z_from_nd(factor_index: pd.DataFrame, diag: pd.Series, nd_label: str = "ND"):
    is_nd = (diag == nd_label)
    nd_mean = factor_index.loc[is_nd].mean()
    nd_std = factor_index.loc[is_nd].std(ddof=0).replace(0, np.nan)
    Z = (factor_index - nd_mean) / nd_std
    return Z, nd_mean, nd_std

def tscore_from_z(z: pd.Series | pd.DataFrame):
    # Map to 0–100 (T=50+10*z) and clip
    T = 50 + 10 * z
    return T.clip(lower=0, upper=100)

def group_centroids(Z: pd.DataFrame, diag: pd.Series, groups=CLINICAL_GROUPS):
    cents = {}
    for g in groups:
        mask = (diag == g)
        cents[g] = Z.loc[mask].mean(skipna=True)
    return cents

def distance_similarity(subject_z: pd.Series, cents: dict):
    dists = {}
    common = None
    for g, c in cents.items():
        cols = subject_z.index[subject_z.notna() & c.notna()]
        if len(cols) == 0:
            d = np.nan
        else:
            d = np.linalg.norm(subject_z[cols].values - c[cols].values)
            common = cols
        dists[g] = d
    # Convert to similarity via softmax on negative distances
    valid = {k:v for k,v in dists.items() if np.isfinite(v)}
    if not valid:
        return dists, {k: np.nan for k in dists.keys()}, common
    vals = np.array(list(valid.values()))
    # guard: if all zero, make uniform
    if np.allclose(vals, 0):
        probs = np.ones_like(vals) / len(vals)
    else:
        logits = -vals
        logits -= logits.max()
        ex = np.exp(logits)
        probs = ex / ex.sum()
    sim = {}
    for (k,_), p in zip(valid.items(), probs):
        sim[k] = float(p)
    for k in dists.keys():
        if k not in sim:
            sim[k] = np.nan
    return dists, sim, common

# ---------------------------- SIDEBAR: REFERENCE ----------------------------
st.sidebar.header("① 참조 데이터 업로드 / Reference data")
ref_file = st.sidebar.file_uploader("엑셀 파일(.xlsx) – DIAG, P01..P52 포함", type=["xlsx"]) 
DIAG_COL = st.sidebar.text_input("그룹 열 이름 / DIAG column", value="DIAG")
thresh_ratio = st.sidebar.slider("요인 평균 계산 최소 응답비율", 0.3, 1.0, 0.5, 0.1)

ref_loaded = False
if ref_file is not None:
    try:
        df_raw, P_ref, diag_ref = load_reference(ref_file.read(), DIAG_COL)
        ref_loaded = True
    except Exception as e:
        st.sidebar.error(f"참조 데이터 로드 오류: {e}")

if not ref_loaded:
    st.info("좌측에서 참조 데이터를 먼저 업로드하세요. (DIAG + P01..P52)")
    st.stop()

# Compute indices and ND-standardization
idx_ref = compute_factor_index(P_ref, thresh_ratio=thresh_ratio)
Z_ref, nd_mean, nd_std = z_from_nd(idx_ref, diag_ref, nd_label="ND")
centroids = group_centroids(Z_ref, diag_ref)

st.success("✅ 참조 데이터에서 ND 표준화 기준을 계산했습니다.")

# ---------------------------- SUBJECT INPUT ----------------------------
st.sidebar.header("② 신규 대상자 입력 / New subject")
mode = st.sidebar.radio("입력 방식 / Input mode", ["문항 점수 업로드(행 1개)", "요인 점수 직접 입력"], index=0)

subject_idx = None

if mode == "문항 점수 업로드(행 1개)":
    st.sidebar.caption("CSV/XLSX 한 행: P문항 열만 포함 (P01..P52 중 사용되는 항목)")
    subj_file = st.sidebar.file_uploader("대상자 파일", type=["csv","xlsx"], key="subj_upload")
    if subj_file is not None:
        try:
            if subj_file.type.endswith("excel") or subj_file.name.lower().endswith(".xlsx"):
                subj_df = pd.read_excel(subj_file)
            else:
                subj_df = pd.read_csv(subj_file)
            if len(subj_df) != 1:
                st.sidebar.warning("파일은 정확히 1행이어야 합니다. 첫 행만 사용합니다.")
            subj_row = subj_df.iloc[0:1].copy()
            P_subj = clean_numeric_frame(subj_row, ALL_P)
            subject_idx = compute_factor_index(P_subj, thresh_ratio=thresh_ratio).iloc[0]
        except Exception as e:
            st.sidebar.error(f"대상자 로드 오류: {e}")
else:
    cols = st.sidebar.columns(2)
    manual_vals = {}
    for i, fname in enumerate(FACTOR_ITEMS.keys()):
        with (cols[i % 2]):
            manual_vals[fname] = st.number_input(f"{fname}", value=float("nan"))
    subject_idx = pd.Series(manual_vals)

if subject_idx is None or subject_idx.isna().all():
    st.warning("신규 대상자 요인 점수가 필요합니다. 좌측에서 입력/업로드 해주세요.")
    st.stop()

# ---------------------------- SUBJECT COMPUTATIONS ----------------------------
subj_z = (subject_idx - nd_mean) / nd_std
subj_t = tscore_from_z(subj_z)

# Distances/similarity
D, S, common_cols = distance_similarity(subj_z, centroids)
closest = min(((g, d) for g,d in D.items() if np.isfinite(d)), key=lambda x: x[1])[0] if any(np.isfinite(list(D.values()))) else None

# ---------------------------- LAYOUT ----------------------------
left, mid, right = st.columns([1.1, 1.1, 0.9])

with left:
    st.subheader("🧾 신규 대상자 요인 점수 / Factor Index")
    disp_idx = pd.DataFrame({"Index": subject_idx, "Z(ND)": subj_z, "Score(0–100)": subj_t})
    st.dataframe(disp_idx.round(3))

with mid:
    st.subheader("🎯 그룹 근접도 / Proximity to Clinical Groups")
    prox_df = pd.DataFrame({
        "Distance(Z)": D,
        "Similarity": S,
    })
    st.dataframe(prox_df)

    if closest:
        st.success(f"가장 가까운 집단(거리 기준): **{closest}**")

with right:
    st.subheader("📤 결과 내보내기 / Export")
    report = {
        "subject_index": subject_idx.fillna(None).to_dict(),
        "subject_z": subj_z.fillna(None).round(4).to_dict(),
        "subject_score_0_100": subj_t.fillna(None).round(1).to_dict(),
        "group_distance": {k: (None if (v is None or not np.isfinite(v)) else float(v)) for k,v in D.items()},
        "group_similarity": {k: (None if (v is None or not np.isfinite(v)) else float(v)) for k,v in S.items()},
        "closest_group": closest,
        "nd_mean": nd_mean.round(4).to_dict(),
        "nd_std": nd_std.round(4).to_dict(),
    }
    json_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("⬇️ JSON 다운로드", data=json_bytes, file_name="subject_factor_report.json", mime="application/json")

    csv_bytes = disp_idx.round(4).to_csv().encode("utf-8-sig")
    st.download_button("⬇️ CSV 다운로드 (요인 점수)", data=csv_bytes, file_name="subject_factor_scores.csv", mime="text/csv")

# ---------------------------- VISUALS ----------------------------
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 0–100 점수 막대 / Bar Scores")
    bar = go.Figure()
    bar.add_trace(go.Bar(x=list(subj_t.index), y=list(subj_t.values), text=[f"{v:.1f}" if pd.notna(v) else "" for v in subj_t.values], textposition="outside"))
    bar.update_yaxes(range=[0,100])
    bar.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(bar, use_container_width=True)

with col2:
    st.subheader("🕸️ 레이더 / Radar (Z-space)")
    # Only include finite z
    zmask = subj_z.dropna()
    if not zmask.empty:
        cats = list(zmask.index)
        vals = list(zmask.values) + [zmask.values[0]]
        cats_close = cats + [cats[0]]
        rad = go.Figure()
        rad.add_trace(go.Scatterpolar(r=vals, theta=cats_close, fill='toself', name='Subject(Z)'))
        # add ND mean = 0 line
        rad.add_trace(go.Scatterpolar(r=[0]*len(cats_close), theta=cats_close, name='ND mean=0', line=dict(dash='dot')))
        # add closest group centroid if available
        if closest and centroids.get(closest) is not None:
            cen = centroids[closest][zmask.index].values
            rad.add_trace(go.Scatterpolar(r=list(cen)+[cen[0]], theta=cats_close, name=f'{closest} centroid(Z)'))
        rad.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=20), polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(rad, use_container_width=True)
    else:
        st.info("레이더를 그릴 유효한 Z점수가 없습니다.")

# ---------------------------- NOTES ----------------------------
st.markdown(
    """
**점수 산식**  
- ND 표준화 *z* = (대상자 요인점수 − ND평균) / ND표준편차  
- 0–100 변환: **Score = clip(50 + 10*z, 0, 100)**  
- 임상 근접도: Z-공간에서 임상 집단 **centroid(평균 벡터)** 에 대한 **유클리드 거리** 및 `softmax(-거리)` 유사도  

**입력 팁**  
- 문항 파일 업로드 시 **P문항만 포함된 한 행** 파일(CSV/XLSX)을 권장합니다.  
- 요인 평균은 기본적으로 각 요인 문항의 **50% 이상 응답** 시 계산합니다(좌측에서 변경 가능).  

**주의**  
- ND 집단 표본이 너무 작거나 특정 요인의 표준편차가 0이면(z 불가) 해당 요인은 제외됩니다.  
- 유사도는 상대적 지표이며, 임상적 진단을 대체하지 않습니다.
"""
)
