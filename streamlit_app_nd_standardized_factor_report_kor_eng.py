# app_52item_assessment_embedded.py
# -*- coding: utf-8 -*-
"""
Streamlit Cloud-ready (Internet only)
- 52문항 설문 → 4요인 계산 → (코드 내장 ND 기준) 표준화 → 0–100 점수 & 임상군 근접도
- 레이더: Z → 0–100 환산값으로 표시
- 요인명: 1=사회적 의사소통, 2=사회적 인식, 3=사회적 동기, 4=언어적 사회인지
- 바 차트: 요인별 서로 다른 색상
- PDF 리포트: 메모리에서 생성 후 즉시 다운로드(로컬 파일 저장 X)
- 자동 해석: 심리학 용어(
    높은 편 → 위험/고위험, 낮은 편 → 안정/매우 안정
  )으로 표기

requirements.txt 예시:
  streamlit
  pandas
  numpy
  plotly
  scikit-learn
  reportlab
  kaleido

폰트(한글 PDF용): 가능하면 fonts/NanumGothic.ttf 포함(없으면 시스템 폰트로 대체)
"""

import io, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------- 페이지/테마 ----------------------------
st.set_page_config(page_title="52문항 요인 평가 (ND 내장판)", layout="wide")
st.title("🧠 52문항 기반 요인 평가 · ND 표준화 (내장판)")
st.caption("ND 기준과 임상군 중심을 코드에 고정하여, 업로드 없이 즉시 평가합니다.")

# ---------------------------- 요인/문항 정의 ----------------------------
FACTOR_ITEMS = {
    "Factor1": ["P04","P05","P06","P07","P09","P11","P12","P13","P15","P16","P19","P20","P22","P23","P24","P26","P27","P29","P32","P34","P36","P38","P39","P42","P44","P46","P49","P50","P52"],
    "Factor2": ["P14","P30","P31","P33","P37","P41","P43","P45","P47","P48","P51"],
    "Factor3": ["P08","P10","P15","P18","P21","P25","P26","P29","P34","P40"],
    "Factor4": ["P03","P20","P32"],
}
FACTOR_TITLES = {
    "Factor1": "사회적 의사소통",
    "Factor2": "사회적 인식",
    "Factor3": "사회적 동기",
    "Factor4": "언어적 사회인지",
}
FACTOR_ORDER = ["Factor1","Factor2","Factor3","Factor4"]
ALL_P = [f"P{str(i).zfill(2)}" for i in range(1,53)]
CLINICAL_GROUPS = ["ND","ASD","ADHD","SCD","HR"]

# ---------------------------- ⛳ 내장 기준값 (실제 값으로 교체) ----------------------------
ND_BASE_MEAN = {"Factor1": 2.50,"Factor2": 2.12,"Factor3": 2.59,"Factor4": 3.09}
ND_BASE_STD = {"Factor1": 0.58,"Factor2": 0.74,"Factor3": 0.70,"Factor4": 1.01}
GROUP_CENTROIDS_Z = {
    "ND"  : {"Factor1": 0.0,  "Factor2": 0.0,  "Factor3": 0.0,  "Factor4": 0.0},
    "ASD" : {"Factor1": 2.29,  "Factor2": 0.93, "Factor3": 0.86, "Factor4": 1.05},
    "ADHD": {"Factor1": 1.34,  "Factor2": 0.63, "Factor3": 0.12,  "Factor4": 0.60},
    "SCD" : {"Factor1": 1.87,  "Factor2": 1.01, "Factor3": 0.71, "Factor4": 0.76},
    "HR"  : {"Factor1": 1.70,  "Factor2": 0.21, "Factor3": 1.26,  "Factor4": 0.26},
}

# ---------------------------- 52문항 텍스트 ----------------------------
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

def z_from_embedded(idx_row: pd.Series) -> pd.Series:
    z = {}
    for f in FACTOR_ORDER:
        m = ND_BASE_MEAN.get(f)
        s = ND_BASE_STD.get(f)
        val = idx_row.get(f)
        z[f] = (val - m) / s if (m is not None and s not in (None, 0) and pd.notna(val)) else np.nan
    return pd.Series(z)

def tscore_from_z(z):
    return (50 + 10*z).clip(lower=0, upper=100)

def distance_similarity(subject_z: pd.Series, cents: dict):
    dists = {}
    for g, c in cents.items():
        cols = [f for f in FACTOR_ORDER if pd.notna(subject_z.get(f)) and (f in c) and pd.notna(c[f])]
        if not cols:
            dists[g] = np.nan
            continue
        sv = np.array([subject_z[f] for f in cols])
        cv = np.array([c[f] for f in cols])
        dists[g] = float(np.linalg.norm(sv - cv))
    valid = {k:v for k,v in dists.items() if np.isfinite(v)}
    if not valid:
        return dists, {k:np.nan for k in dists}
    vals = np.array(list(valid.values()))
    if np.allclose(vals, 0):
        probs = np.ones_like(vals)/len(vals)
    else:
        logits = -vals; logits -= logits.max(); ex = np.exp(logits); probs = ex/ex.sum()
    sims = {}
    for (k,_), p in zip(valid.items(), probs):
        sims[k] = float(p)
    for k in dists.keys():
        if k not in sims: sims[k] = np.nan
    return dists, sims

# ---------------------------- 세션 상태 초기값 ----------------------------
if "responses" not in st.session_state:
    st.session_state["responses"] = {pid: 3 for pid in ALL_P}  # 기본값 3

# ---------------------------- 52문항 폼 ----------------------------
st.subheader("🧩 52문항 설문 (1~5 Likert)")
with st.form("qform"):
    sliders = {}
    cols = st.columns(2)
    for i, q in enumerate(QUESTION_TEXTS, start=1):
        pid = f"P{str(i).zfill(2)}"
        col = cols[(i-1)%2]
        with col:
            default_val = st.session_state["responses"].get(pid, 3)
            sliders[pid] = st.slider(f"{pid}. {q}", 1, 5, int(default_val), 1)
    submitted = st.form_submit_button("결과 계산")

if not submitted:
    st.stop()

st.session_state["responses"] = sliders.copy()

# ---------------------------- 점수 계산 ----------------------------
P_subj = pd.DataFrame([sliders])
idx_subj = compute_factor_index(P_subj, thresh_ratio=0.5).iloc[0]
subj_z = z_from_embedded(idx_subj)
subj_t = tscore_from_z(subj_z)

# 표시용 라벨
labels = [FACTOR_TITLES[f] for f in FACTOR_ORDER]
subj_t_display = pd.Series([subj_t.get(f) for f in FACTOR_ORDER], index=labels)
subj_z_display = pd.Series([subj_z.get(f) for f in FACTOR_ORDER], index=labels)

D, S = distance_similarity(subj_z, GROUP_CENTROIDS_Z)
closest = None
finite_d = {k:v for k,v in D.items() if np.isfinite(v)}
if finite_d:
    closest = min(finite_d.items(), key=lambda x:x[1])[0]

# ---------------------------- 자동 해석(심리학 용어) ----------------------------
def interpret_psych(zval: float, name: str):
    if pd.isna(zval):
        return f"{name}: 데이터 부족"
    # Z 기준 심리학적 위험/안정 레이블링
    if zval >= 2.0:
        return f"{name}: 고위험 (매우 높음)"
    elif zval >= 1.5:
        return f"{name}: 위험 (높음)"
    elif zval >= 1.0:
        return f"{name}: 주의 필요 (다소 높음)"
    elif zval > -0.5:
        return f"{name}: 평균 범위"
    elif zval > -1.0:
        return f"{name}: 안정 경향 (다소 낮음)"
    elif zval > -1.5:
        return f"{name}: 안정 (낮음)"
    else:
        return f"{name}: 매우 안정 (매우 낮음)"

interp_lines = [interpret_psych(subj_z_display.get(FACTOR_TITLES[f]), FACTOR_TITLES[f]) for f in FACTOR_ORDER]
if closest:
    interp_lines.append(f"임상군 근접도: 가장 가까운 집단은 **{closest}**")

# ---------------------------- 시각화 ----------------------------
bar_colors = {  # 요인별 바 색상
    "사회적 의사소통": "#1f77b4",
    "사회적 인식": "#ff7f0e",
    "사회적 동기": "#2ca02c",
    "언어적 사회인지": "#d62728",
}

left, mid, right = st.columns([1.1, 1.1, 0.9])
with left:
    st.subheader("📊 요인 점수 (0–100)")
    fig_bar = go.Figure()
    yvals = [None if pd.isna(v) else v for v in subj_t_display.values]
    colors = [bar_colors.get(name, "#888888") for name in subj_t_display.index]
    fig_bar.add_trace(go.Bar(x=list(subj_t_display.index), y=yvals,
                             marker_color=colors,
                             text=["" if pd.isna(v) else f"{v:.1f}" for v in subj_t_display.values],
                             textposition="outside"))
    fig_bar.update_yaxes(range=[0,100])
    fig_bar.update_layout(height=420,width=350,margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_bar, use_container_width=False) 

with mid:
    st.subheader("🕸️ 레이더 (0–100)")
    tmask = subj_t_display.dropna()
    if not tmask.empty:
        cats = list(tmask.index)
        vals = list(tmask.values) + [tmask.values[0]]
        catsc = cats + [cats[0]]
        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(r=vals, theta=catsc, fill='toself', name='Subject(0–100)'))
        if closest and GROUP_CENTROIDS_Z.get(closest) is not None:
            cen_z = np.array([GROUP_CENTROIDS_Z[closest][f] for f in FACTOR_ORDER])
            cen_t = np.clip(50 + 10*cen_z, 0, 100)
            cen_map = {FACTOR_TITLES[f]: cen_t[i] for i,f in enumerate(FACTOR_ORDER)}
            cen_vals = [cen_map[c] for c in cats] + [cen_map[cats[0]]]
            fig_rad.add_trace(go.Scatterpolar(r=cen_vals, theta=catsc, name=f'{closest} centroid(0–100)'))
        fig_rad.update_layout(
            height=420, margin=dict(l=20,r=20,t=30,b=20),
            polar=dict(radialaxis=dict(visible=True, range=[0,100], tick0=0, dtick=10))
        )
        st.plotly_chart(fig_rad, use_container_width=True)
    else:
        fig_rad = None
        st.info("레이더를 그릴 유효한 점수가 없습니다.")

with right:
    st.subheader("🎯 임상군 근접도")
    prox_df = pd.DataFrame({"Distance": D, "Similarity": S})
    st.dataframe(prox_df)
    if closest:
        st.success(f"가장 가까운 집단: **{closest}**")

st.markdown("---")
st.subheader("📝 자동 해석")
st.markdown("\n".join([f"- {line}" for line in interp_lines]))

# ---------------------------- PDF 리포트 (메모리 생성 → 다운로드) ----------------------------
st.markdown("---")
st.subheader("📤 결과 리포트 PDF 다운로드")

# 폰트 등록 (한글)
FONT_PATHS = ["fonts/NanumGothic.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc"]
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
    FONT_NAME = "Helvetica"

def fig_to_png_bytes(fig):
    if fig is None:
        return None
    return fig.to_image(format="png", scale=2)

if st.button("PDF 만들기"):
    try:
        bar_png = fig_to_png_bytes(fig_bar)
        rad_png = fig_to_png_bytes(fig_rad)
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        W, H = A4
        # 제목
        c.setFont(FONT_NAME, 16)
        c.drawString(40, H-60, "52문항 요인 평가 리포트 (ND 표준화·내장판)")
        # 요약(0–100)
        c.setFont(FONT_NAME, 10)
        y = H-90
        for name, val in subj_t_display.items():
            vtxt = "NaN" if pd.isna(val) else f"{val:.1f}"
            c.drawString(40, y, f"{name}: {vtxt}")
            y -= 14
            if y < 120:
                c.showPage(); c.setFont(FONT_NAME, 10); y = H-60
        # 자동 해석(심리학 용어)
        for line in interp_lines:
            c.drawString(40, y, line)
            y -= 14
            if y < 120:
                c.showPage(); c.setFont(FONT_NAME, 10); y = H-60
        # 바 차트
        c.showPage(); c.setFont(FONT_NAME, 12); c.drawString(40, H-60, "요인 점수 (0–100)")
        if bar_png:
            img1 = ImageReader(io.BytesIO(bar_png))
            c.drawImage(img1, 40, 200, width=W-80, height=H-300, preserveAspectRatio=True, mask='auto')
        # 레이더
        if rad_png:
            c.showPage(); c.setFont(FONT_NAME, 12); c.drawString(40, H-60, "레이더 (0–100)")
            img2 = ImageReader(io.BytesIO(rad_png))
            c.drawImage(img2, 80, 180, width=W-160, height=H-320, preserveAspectRatio=True, mask='auto')
        # 근접도
        c.showPage(); c.setFont(FONT_NAME, 12); c.drawString(40, H-60, "임상군 근접도")
        c.setFont(FONT_NAME, 10); y = H-90
        for g in prox_df.index:
            d = prox_df.loc[g, "Distance"]
            s = prox_df.loc[g, "Similarity"]
            d_txt = "NaN" if pd.isna(d) else f"{d:.3f}"
            s_txt = "NaN" if pd.isna(s) else f"{s:.3f}"
            c.drawString(40, y, f"{g}: 거리={d_txt}  유사도={s_txt}")
            y -= 14
        c.save()
        st.download_button("⬇️ PDF 다운로드", data=pdf_buffer.getvalue(), file_name="factor_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF 생성 실패: {e}")

