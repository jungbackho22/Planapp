# app_52item_assessment.py
# -*- coding: utf-8 -*-
"""
52문항 응답 → 4요인 계산 → 선택 기준선(예: ND) 표준화 → 0–100 점수/군집근접도 → 
PDF 리포트 다운로드 + 52문항 세션 저장/불러오기 + 자동 해석문 생성

배포: Streamlit Community Cloud 권장
필수: requirements.txt 에 아래 포함
  streamlit
  pandas
  numpy
  plotly
  openpyxl
  scikit-learn
  reportlab
  kaleido

폰트(한글 PDF 대응):
- 리포트랩(reportlab)에서 한글을 위해 TTF 등록이 필요합니다.
- 저장소에 fonts/NanumGothic.ttf 를 포함해 주세요. 없으면 시스템 폰트로 대체합니다.

실행:
  streamlit run app_52item_assessment.py
"""

import io
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# PDF (reportlab) & 이미지 내보내기(kaleido)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------- 페이지 설정 ----------------------------
st.set_page_config(page_title="52문항 요인 평가", layout="wide")
st.title("🧠 52문항 기반 요인 평가 · ND 표준화 리포트")
st.caption("참조 데이터로 기준선을 정한 뒤, 응답자의 위치와 0–100 점수 및 임상군 근접도를 제공합니다.")

# ---------------------------- 요인/문항 정의 ----------------------------
FACTOR_ITEMS = {
    "Factor1": ["P04","P05","P06","P07","P09","P11","P12","P13","P15","P16","P19","P20","P22","P23","P24","P26","P27","P29","P32","P34","P36","P38","P39","P42","P44","P46","P49","P50","P52"],
    "Factor2": ["P14","P30","P31","P33","P37","P41","P43","P45","P47","P48","P51"],
    "Factor3": ["P08","P10","P15","P18","P21","P25","P26","P29","P34","P40"],
    "Factor4": ["P03","P20","P32"],
}
FACTOR_TITLES = {
    "Factor1": "사회적 의사소통 및 반복행동",
    "Factor2": "사회적 인식 및 상호작용 조절",
    "Factor3": "사회적 동기 및 정서표현",
    "Factor4": "언어적 사회인지",
}
ALL_P = [f"P{str(i).zfill(2)}" for i in range(1,53)]
CLINICAL_GROUPS = ["ND","ASD","ADHD","SCD","HR"]

QUESTION_TEXTS = [
    "나는 어른들의 도움 없이도 다른 사람들과 어울리거나 이야기할  수 있다.",
    "모르는 것이 있어도 나는  되도록 다른 사람들에게 물어보지 않는다.",
    "나는 농담이나 유머를 자주 쓰는 편이다.",
    "나는 몸(손, 머리 등)이나 물건을 흔들거나 두드리는 습관이 있다.",
    "나는 빨리 대답하라는 말을 들을 때가 있다.",
    "나는 운동신경이 떨어진다.",
    "나는 다른 사람들이 무슨 생각을 하는지 잘 모르겠다.",
    "나는 조용한 곳보다 사람들이 많은 곳이 좋다.",
    "나는 다른 사람이 한 농담이 잘 이해가 안 될 때가 있다.",
    "나는 또래들에게 먼저 다가가거나 말을 걸 수 있다.",
    "나는 똑같은 얘기 좀 그만하라는 말을 자주 듣는다.",
    "나는 다른 사람과 대화를 길게 주고 받는 것이 어렵다.",
    "나는 소리나 빛, 촉감 등에 예민하다.",
    "나는 스스로 개인 위생을 관리할 수 있다.",
    "나는 다른 사람들과 같이 해야 하는 활동을 피한다.",
    "내 생각이 독특해서 잘 이해가 안 가거나 특이하다는 말을 들을 때가 있다.",
    "내 목소리가 너무 커서 다른 사람을 방해하고 있는지 알아차릴 수 있다.",
    "나는 혼자 있는 것보다 다른 사람들과 같이 있는 것이 좋다.",
    "한 가지만 지나치게 좋아한다고 주변 사람들이 나에게 뭐라고 한다.",
    "나는 다른 사람들의 대화에 적절히 끼어드는 것이 어렵다.",
    "나는 다른 사람들 보다 특별히 잘하는 것이 있다.",
    "나는 하루 대부분을 내가 좋아하는 것들에 대해 생각한다.",
    "하얀 거짓말도 나쁜 거짓말이라고 생각한다.",
    "나는 나의 생각을 말로 전달하는 것이 어렵다.",
    "나는 나의 기분을 표정과 행동으로 적절히 표현할 수 있다.",
    "나는 다른 사람들보다 못하는 것이 많다.",
    "나는 주변에 다른 사람이 있다는 것을 알아채지 못할 때가 있다",
    "나는 편식을 하는 편이다.",
    "나는 혼자 있는 것이 편해서 모임이나 단체 활동에서 빠진 적이 있다",
    "나는 다른 사람과 상호작용할 때 적절한 시선을 유지할 수 있다.",
    "나는 다른 사람의 반응을 보고 내가 실수했는지 알아차릴 수 있다.",
    "나는 책이나 말의 숨은 뜻을 이해하기가 어렵다.",
    "다른 사람과 상호작용할 때 나는 적절한 거리와 방향를 유지할 수 있다.",
    "나는 다른 사람들이 있는 곳에서는 긴장되거나 불안하다.",
    "나는 물건을 원래의 용도와 다르게 사용할 수 있다.",
    "시간표나 계획이 바뀌면 나는 생각과 마음이 많이 불편하다.",
    "나는 왜 대답을 안 하냐는 말을 들을 때가 있다.",
    "나는 눈치가 없거나 둔하다는 말을 듣는다.",
    "나는 다른 사람들과 어울리거나 이야기 하고 싶다.",
    "다른 사람과 상호작용할 때 나는 상황에 따라 적절한 행동을 할 수 있다.",
    "나는 책이나 대화 중에 나오는 관용적 표현이나 속담이 잘 이해되지 않을 때가 있다.",
    "나는 다른 사람의 목소리와 표정으로 그 사람의 기분이나 마음을 파악할 수 있다.",
    "나는 드라마나 영화를 볼 때 내용이 잘 이해가 안 된다.",
    "나는 다른 사람의 기분에 적절한 반응을 할 수 있다.",
    "나는 주변에서 무슨 일이 일어나는지 놓친다.",
    "다른 사람과 대화할 때 나는 적절한 어조, 말투, 말의 크기를 사용하여 말할 수 있다.",
    "다른 사람의 반응을 보고 무엇을 해야할지 알 수 있다.",
    "나는 다른 사람들의 기분이 어떤지 잘 모르겠다.",
    "내가 말귀를 잘 못 알아들어 답답하다는 말을 들을 때가 있다.",
    "나는 다른 사람의 기분이나 생각을 알아차릴 수 있다 .",
    "나는 어른이 옆에 없으면 불안하다."
]

# ---------------------------- 유틸 ----------------------------
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

def clean_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return s.astype(float)
    x = s.astype(str).str.strip()
    for a,b in {",":"", "%":"", "−":"-", "–":"-", "—":"-", "±":" ", "≥":"", "≤":"", ">":"", "<":"", "=":""}.items():
        x = x.str.replace(a,b, regex=False)
    num = x.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(num, errors="coerce")

def clean_numeric_frame(df: pd.DataFrame, cols):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = clean_numeric_series(df[c])
    return out

def z_standardize(factor_index: pd.DataFrame, base_mask: pd.Series):
    base_mean = factor_index.loc[base_mask].mean()
    base_std  = factor_index.loc[base_mask].std(ddof=0).replace(0, np.nan)
    Z = (factor_index - base_mean) / base_std
    return Z, base_mean, base_std

def tscore_from_z(z: pd.Series | pd.DataFrame):
    return (50 + 10*z).clip(lower=0, upper=100)

def group_centroids(Z: pd.DataFrame, diag: pd.Series, groups):
    cents = {}
    for g in groups:
        mask = (diag == g)
        cents[g] = Z.loc[mask].mean(skipna=True)
    return cents

def distance_similarity(subject_z: pd.Series, cents: dict):
    dists = {}
    for g, c in cents.items():
        cols = subject_z.index[subject_z.notna() & c.notna()]
        if len(cols) == 0:
            d = np.nan
        else:
            d = np.linalg.norm(subject_z[cols].values - c[cols].values)
        dists[g] = d
    valid = {k:v for k,v in dists.items() if np.isfinite(v)}
    if not valid:
        return dists, {k:np.nan for k in dists}, None
    vals = np.array(list(valid.values()))
    if np.allclose(vals, 0):
        probs = np.ones_like(vals)/len(vals)
    else:
        logits = -vals; logits -= logits.max(); ex = np.exp(logits); probs = ex/ex.sum()
    sim = {}
    for (k,_), p in zip(valid.items(), probs):
        sim[k] = float(p)
    for k in dists.keys():
        if k not in sim: sim[k] = np.nan
    return dists, sim, None

# ---------------------------- 사이드바: 참조/기준선 ----------------------------
st.sidebar.header("① 참조 데이터 / 기준선 설정")
ref_file = st.sidebar.file_uploader("참조 엑셀(.xlsx) — DIAG + P01..P52", type=["xlsx"], key="ref")
diag_col = st.sidebar.text_input("DIAG 열 이름", value="DIAG")
base_choice = st.sidebar.selectbox("기준선 레이블 선택 (Z 표준화에 사용)", options=["ND"] + ["사용자 지정"], index=0)
user_base_label = None
if base_choice == "사용자 지정":
    user_base_label = st.sidebar.text_input("기준으로 삼을 DIAG 라벨", value="ND")
thresh_ratio = st.sidebar.slider("요인 평균 최소 응답비율", 0.3, 1.0, 0.5, 0.1)

# ---------------------------- 참조 데이터 로드 ----------------------------
ref_loaded = False
if ref_file is not None:
    try:
        df_ref_raw = pd.read_excel(ref_file)
        if diag_col not in df_ref_raw.columns:
            st.sidebar.error(f"'{diag_col}' 열이 없습니다.")
        else:
            Ps_ref = clean_numeric_frame(df_ref_raw, ALL_P)
            idx_ref = compute_factor_index(Ps_ref, thresh_ratio=thresh_ratio)
            diag_ref = df_ref_raw[diag_col].astype(str)
            base_label = user_base_label if user_base_label else "ND"
            base_mask = (diag_ref == base_label)
            if base_mask.sum() < 5:
                st.sidebar.warning(f"기준선 '{base_label}' 표본이 적습니다(n={base_mask.sum()}).")
            Z_ref, base_mean, base_std = z_standardize(idx_ref, base_mask)
            cents = group_centroids(Z_ref, diag_ref, groups=[g for g in CLINICAL_GROUPS if g in diag_ref.unique()])
            st.sidebar.success("✅ 기준선 계산 완료")
            ref_loaded = True
    except Exception as e:
        st.sidebar.error(f"참조 데이터 로드 오류: {e}")

if not ref_loaded:
    st.info("좌측에서 참조 데이터를 업로드하고 기준선을 선택해 주세요.")
    st.stop()

# ---------------------------- 52문항 입력: 세션 저장/불러오기 ----------------------------
st.sidebar.header("② 응답 저장/불러오기")
# 초기화
if "responses" not in st.session_state:
    st.session_state["responses"] = {pid: None for pid in ALL_P}

col_json1, col_json2 = st.sidebar.columns(2)
with col_json1:
    if st.button("현재 응답 JSON 다운로드"):
        payload = json.dumps(st.session_state["responses"], ensure_ascii=False, indent=2)
        st.download_button("⬇️ responses.json", data=payload.encode("utf-8"), file_name="responses.json", mime="application/json")
with col_json2:
    uploaded_json = st.file_uploader("응답 불러오기(JSON)", type=["json"], key="loadjson")
    if uploaded_json is not None:
        try:
            data = json.load(uploaded_json)
            for k,v in data.items():
                if k in st.session_state["responses"]:
                    st.session_state["responses"][k] = v
            st.sidebar.success("✅ 응답 불러오기 완료")
        except Exception as e:
            st.sidebar.error(f"JSON 파싱 실패: {e}")

# ---------------------------- 52문항 폼 ----------------------------
st.subheader("🧩 52문항 설문 (1~5 Likert, 기본=3)")
with st.form("qform", clear_on_submit=False):
    sliders = {}
    cols = st.columns(2)
    for i, q in enumerate(QUESTION_TEXTS, start=1):
        pid = f"P{str(i).zfill(2)}"
        col = cols[(i-1)%2]
        with col:
            default_val = st.session_state["responses"].get(pid, 3)
            if default_val is None: default_val = 3
            sliders[pid] = st.slider(f"{pid}. {q}", 1, 5, int(default_val), 1)
    submitted = st.form_submit_button("결과 계산")

if not submitted:
    st.stop()

# 세션에 저장
st.session_state["responses"] = sliders.copy()

# ---------------------------- 점수 계산 ----------------------------
P_subj = pd.DataFrame([sliders])
idx_subj = compute_factor_index(P_subj, thresh_ratio=thresh_ratio).iloc[0]
subj_z = (idx_subj - base_mean) / base_std
subj_t = tscore_from_z(subj_z)

D, S, _ = distance_similarity(subj_z, cents)
closest = None
finite_d = {k:v for k,v in D.items() if np.isfinite(v)}
if finite_d:
    closest = min(finite_d.items(), key=lambda x:x[1])[0]

# ---------------------------- 자동 해석문 ----------------------------
def interpret_factor(zval: float, name: str):
    if pd.isna(zval):
        return f"{name}: 데이터 부족으로 해석 불가"
    if zval >= 1.5:
        return f"{name}: 매우 높은 편 (상위 약 7%)"
    elif zval >= 1.0:
        return f"{name}: 높은 편 (상위 약 16%)"
    elif zval >= 0.5:
        return f"{name}: 다소 높은 편"
    elif zval > -0.5:
        return f"{name}: 보통 범위"
    elif zval > -1.0:
        return f"{name}: 다소 낮은 편"
    elif zval > -1.5:
        return f"{name}: 낮은 편 (하위 약 16%)"
    else:
        return f"{name}: 매우 낮은 편 (하위 약 7%)"

interp_lines = [interpret_factor(subj_z.get(f), FACTOR_TITLES[f]+f" ({f})") for f in FACTOR_ITEMS.keys()]
if closest:
    interp_lines.append(f"임상군 근접도: 가장 가까운 집단은 **{closest}** 입니다.")

# ---------------------------- 시각화 ----------------------------
left, mid, right = st.columns([1.1, 1.1, 0.9])
with left:
    st.subheader("📊 요인 점수 (0–100)")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=list(subj_t.index), y=[None if pd.isna(v) else v for v in subj_t.values], text=["" if pd.isna(v) else f"{v:.1f}" for v in subj_t.values], textposition="outside"))
    fig_bar.update_yaxes(range=[0,100])
    fig_bar.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

with mid:
    st.subheader("🕸️ 레이더 (Z)")
    zmask = subj_z.dropna()
    if not zmask.empty:
        cats = list(zmask.index)
        vals = list(zmask.values) + [zmask.values[0]]
        catsc = cats + [cats[0]]
        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(r=vals, theta=catsc, fill='toself', name='Subject(Z)'))
        # 가장 가까운 집단 중심 표시
        if closest and cents.get(closest) is not None:
            cen = cents[closest][zmask.index].values
            fig_rad.add_trace(go.Scatterpolar(r=list(cen)+[cen[0]], theta=catsc, name=f'{closest} centroid(Z)'))
        fig_rad.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=20), polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig_rad, use_container_width=True)
    else:
        fig_rad = None
        st.info("레이더를 그릴 유효한 Z 점수가 없습니다.")

with right:
    st.subheader("🎯 임상군 근접도")
    prox_df = pd.DataFrame({"Distance": D, "Similarity": S})
    st.dataframe(prox_df)
    if closest:
        st.success(f"가장 가까운 집단: **{closest}**")

st.markdown("---")
st.subheader("📝 자동 해석")
st.markdown("\n".join([f"- {line}" for line in interp_lines]))

# ---------------------------- PDF 리포트 생성 ----------------------------
st.markdown("---")
st.subheader("📤 결과 리포트 PDF 다운로드")

# 폰트 등록 (한글)
FONT_PATHS = [
    "fonts/NanumGothic.ttf",                     # 저장소 포함 권장
    "/System/Library/Fonts/AppleSDGothicNeo.ttc" # macOS fallback
]
FONT_NAME = None
for fp in FONT_PATHS:
    try:
        if os.path.exists(fp):
            pdfmetrics.registerFont(TTFont("KFont", fp))
            FONT_NAME = "KFont"
            break
    except Exception:
        continue

if FONT_NAME is None:
    # 마지막 수단: 기본 폰트(영문). 한글은 이미지로 대체됨.
    FONT_NAME = "Helvetica"

# Plotly → 이미지 버퍼 (kaleido 필요)
def fig_to_png_bytes(fig):
    return fig.to_image(format="png", scale=2)

if st.button("PDF 생성 및 다운로드"):
    try:
        # 그림 PNG 준비
        bar_png = fig_to_png_bytes(fig_bar)
        rad_png = fig_to_png_bytes(fig_rad) if fig_rad is not None else None

        # PDF 메모리 버퍼
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        W, H = A4

        # 제목
        c.setFont(FONT_NAME, 16)
        c.drawString(40, H-60, "52문항 요인 평가 리포트 (ND 표준화)")

        # 요약 텍스트
        c.setFont(FONT_NAME, 10)
        y = H-90
        for line in interp_lines:
            c.drawString(40, y, line)
            y -= 14
            if y < 120:
                c.showPage(); c.setFont(FONT_NAME, 10); y = H-60

        # 바 차트
        c.showPage()
        c.setFont(FONT_NAME, 12)
        c.drawString(40, H-60, "요인 점수 (0–100)")
        img1 = ImageReader(io.BytesIO(bar_png))
        c.drawImage(img1, 40, 200, width=W-80, height=H-300, preserveAspectRatio=True, mask='auto')

        # 레이더
        if rad_png is not None:
            c.showPage()
            c.setFont(FONT_NAME, 12)
            c.drawString(40, H-60, "레이더 (Z)")
            img2 = ImageReader(io.BytesIO(rad_png))
            c.drawImage(img2, 80, 180, width=W-160, height=H-320, preserveAspectRatio=True, mask='auto')

        # 근접도 표 (텍스트 간략)
        c.showPage(); c.setFont(FONT_NAME, 12); c.drawString(40, H-60, "임상군 근접도")
        c.setFont(FONT_NAME, 10)
        y = H-90
        for g in prox_df.index:
            d = prox_df.loc[g, "Distance"]
            s = prox_df.loc[g, "Similarity"]
            c.drawString(40, y, f"{g}: 거리={d:.3f}  유사도={s:.3f}")
            y -= 14

        c.save()
        pdf_bytes = pdf_buffer.getvalue()
        st.download_button("⬇️ PDF 다운로드", data=pdf_bytes, file_name="factor_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF 생성 실패: {e}")

# ---------------------------- 주의/도움말 ----------------------------
st.markdown(
    """
**설정 메모**  
- *기준선 선택*: 기본은 ND이나, 사이드바에서 사용자 지정 라벨을 기준으로 표준화를 수행할 수 있습니다.  
- *0–100 변환*: Score = clip(50 + 10·z, 0, 100).  
- *근접도*: Z-공간 임상군 centroid와의 유클리드 거리 → softmax(-거리)로 유사도 환산.  
- *PDF 한글*: 리포트랩은 폰트 임베딩이 필요합니다. 저장소에 **fonts/NanumGothic.ttf** 를 포함하세요.  
- *세션 저장/불러오기*: 사이드바에서 JSON으로 내보내고 다시 불러올 수 있습니다.  
"""
)
