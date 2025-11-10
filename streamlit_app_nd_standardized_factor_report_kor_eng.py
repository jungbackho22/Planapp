# streamlit_app_nd_standardized_factor_report.py
# -*- coding: utf-8 -*-
"""
인터넷/Cloud 전용
- 52문항 → 4요인 평균 → ND 기준 표준화(Z) → 0–100 환산(T형식) 점수
- 시각화: 레이더(0–100), 바그래프(요인별 색상)
- 자동해석 표: 심리학 용어(고위험/위험/주의/중립/안정 경향/안정/매우 안정), 2줄 이상 설명
- PDF: 레이더 + 바그래프 + 해석표를 메모리에서 생성 후 즉시 다운로드
"""

import io, os
from datetime import datetime

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
st.set_page_config(page_title="52문항 요인 평가 (ND 표준화)", layout="wide")
st.title("🧠 52문항 기반 요인 평가 · ND 표준화")
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

# ---------------------------- ⛳ 내장 기준값 (실제 수치로 교체하세요) ----------------------------
ND_BASE_MEAN = {"Factor1": 3.0, "Factor2": 3.2, "Factor3": 3.1, "Factor4": 3.0}
ND_BASE_STD  = {"Factor1": 0.6, "Factor2": 0.5, "Factor3": 0.5, "Factor4": 0.4}

# ---------------------------- 색상 팔레트 ----------------------------
bar_colors = {
    "사회적 의사소통": "#1f77b4",
    "사회적 인식":   "#ff7f0e",
    "사회적 동기":   "#2ca02c",
    "언어적 사회인지": "#d62728",
}

# ---------------------------- 유틸 ----------------------------
def compute_factor_index(P_frame: pd.DataFrame, thresh_ratio: float = 0.5) -> pd.DataFrame:
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

def tscore_from_z(z: pd.Series) -> pd.Series:
    # 50 + 10*z 를 0~100로 클리핑
    return (50 + 10*z).clip(lower=0, upper=100)

# 위험수준 라벨
def level_from_z(z):
    if pd.isna(z): return "데이터 부족"
    if z >= 2.0:  return "고위험"
    if z >= 1.5:  return "위험"
    if z >= 1.0:  return "주의"
    if z > -0.5:  return "중립"
    if z > -1.0:  return "안정 경향"
    if z > -1.5:  return "안정"
    return "매우 안정"

# 요인별 위험수준 → 2줄 이상 문장 (HTML <br>로 줄바꿈)
INTERP_DICT = {
    "사회적 의사소통": {
        "고위험":"대인 대화의 지속·상호성이 현저히 저하될 수 있습니다.<br>의미 조율과 주고받기 실패가 반복될 가능성이 높습니다.",
        "위험":"상대 반응 조정과 대화 유지에 명확한 어려움이 보일 수 있습니다.<br>사회적 단서에 따른 발화 조절이 미흡할 수 있습니다.",
        "주의":"상호작용 품질이 고르게 유지되지 않을 수 있습니다.<br>의사소통 전략 훈련이 도움이 됩니다.",
        "중립":"연령 기대 수준의 상호작용이 대체로 유지됩니다.<br>특정 상황에서의 변동만 점검하세요.",
        "안정 경향":"의사소통 교환이 비교적 유연하게 이루어집니다.<br>또래 상호작용에서 보호 요인이 관찰됩니다.",
        "안정":"대화 주고받기와 조율 능력이 안정적으로 보입니다.<br>사회적 요구 변화에도 적응이 양호합니다.",
        "매우 안정":"높은 상호성으로 의미 조율이 원활합니다.<br>복잡한 대화 맥락에서도 기능이 견고합니다.",
    },
    "사회적 인식": {
        "고위험":"표정·시선·억양 등 비언어 단서 해석에 현저한 결함이 시사됩니다.<br>타인의 의도 추론에서 지속적인 실패가 우려됩니다.",
        "위험":"암묵적 규칙과 맥락 신호를 일관되게 포착하기 어렵습니다.<br>상황 오해로 관계 마찰이 증가할 수 있습니다.",
        "주의":"간접 신호 해석의 변동성이 큽니다.<br>명시적 피드백 제공과 시각적 단서가 유익합니다.",
        "중립":"비언어 단서와 맥락 이해가 대체로 적절합니다.<br>상황 복잡도에 따른 편차만 관찰하십시오.",
        "안정 경향":"타인의 정서·의도 파악이 비교적 정확합니다.<br>사회적 상황 추론에서 강점이 보입니다.",
        "안정":"맥락 추론과 신호 통합이 안정적입니다.<br>관계 조율에 긍정적 영향을 줍니다.",
        "매우 안정":"미묘한 단서도 정교하게 통합합니다.<br>복잡한 사회적 상황에서도 해석이 일관됩니다.",
    },
    "사회적 동기": {
        "고위험":"대인 접근/참여 동기가 크게 저하될 수 있습니다.<br>회피·단절이 생활 전반에 영향을 미칠 수 있습니다.",
        "위험":"또래 상호작용 회피가 두드러질 수 있습니다.<br>활동 시작·유지에 외적 촉진이 필요합니다.",
        "주의":"집단 활동 참여의 일관성이 떨어질 수 있습니다.<br>성공 경험 축적과 보상 설계가 권장됩니다.",
        "중립":"접근 동기가 평균 범위입니다.<br>활동 유형에 따른 선호 차이만 유의하세요.",
        "안정 경향":"대인 상호작용에 긍정적 접근이 관찰됩니다.<br>협동 상황에서 활력이 나타납니다.",
        "안정":"사회적 참여가 안정적·지속적으로 유지됩니다.<br>동료 관계 형성에 우호적입니다.",
        "매우 안정":"높은 참여 의지와 주도성이 두드러집니다.<br>새로운 또래 환경에도 빠르게 적응합니다.",
    },
    "언어적 사회인지": {
        "고위험":"은유·관용구·숨은 뜻 이해에서 심각한 곤란이 시사됩니다.<br>문맥 의도 추론의 실패로 의사소통 오해가 잦을 수 있습니다.",
        "위험":"간접화행과 상황함의 해석이 불안정합니다.<br>추론 요구가 높은 담화에서 난점이 두드러집니다.",
        "주의":"추론 단서를 명시화하면 개선 여지가 있습니다.<br>사례 중심의 의미 확장이 도움이 됩니다.",
        "중립":"문맥 기반 의미 이해가 대체로 적절합니다.<br>특정 난도에서만 점검하면 충분합니다.",
        "안정 경향":"의미 추론과 맥락 연결이 비교적 원활합니다.<br>복잡한 담화에서도 일관된 해석을 보입니다.",
        "안정":"언어적 사회인지가 안정적으로 유지됩니다.<br>다양한 의도 표현에도 적절히 반응합니다.",
        "매우 안정":"고난도 담화에서도 의미 추론이 정교합니다.<br>미묘한 뉘앙스까지 탄력적으로 이해합니다.",
    },
}

# ---------------------------- 세션 초기값 ----------------------------
if "responses" not in st.session_state:
    st.session_state["responses"] = {pid: 3 for pid in ALL_P}

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

# 표시용(한글 라벨)
labels = [FACTOR_TITLES[f] for f in FACTOR_ORDER]
subj_t_display = pd.Series([subj_t.get(f) for f in FACTOR_ORDER], index=labels)
subj_z_display = pd.Series([subj_z.get(f) for f in FACTOR_ORDER], index=labels)

# ---------------------------- 시각화 ----------------------------
left, mid, right = st.columns([0.9, 1.1, 1.0])

# 바그래프
with left:
    st.subheader("📊 요인 점수 (0–100)")
    fig_bar = go.Figure()
    yvals = [None if pd.isna(v) else float(v) for v in subj_t_display.values]
    colors = [bar_colors.get(name, "#888888") for name in subj_t_display.index]
    fig_bar.add_trace(go.Bar(
        x=list(subj_t_display.index),
        y=yvals,
        marker_color=colors,
        text=["" if v is None or np.isnan(v) else f"{v:.1f}" for v in yvals],
        textposition="outside"
    ))
    fig_bar.update_yaxes(range=[0, 100])
    fig_bar.update_layout(height=420, width=360,  # ▶ 가로 폭 축소
                          margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_bar, use_container_width=False)

# 레이더(0–100)
with mid:
    st.subheader("🕸️ 레이더 (0–100)")
    tmask = subj_t_display.dropna()
    if not tmask.empty:
        cats = list(tmask.index)
        vals = list(tmask.values) + [tmask.values[0]]
        catsc = cats + [cats[0]]
        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(r=vals, theta=catsc, fill='toself', name='Subject(0–100)'))
        fig_rad.update_layout(
            height=420, margin=dict(l=20, r=20, t=30, b=20),
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tick0=0, dtick=10))
        )
        st.plotly_chart(fig_rad, use_container_width=True)
    else:
        fig_rad = None
        st.info("레이더를 그릴 유효한 점수가 없습니다.")

# 해석 표
def make_interpret_table(z_series: pd.Series) -> go.Figure:
    rows_scale, rows_text = [], []
    for name, z in z_series.items():
        lv  = level_from_z(z)
        txt = INTERP_DICT.get(name, {}).get(lv, f"{name}: {lv}")
        rows_scale.append(name)
        rows_text.append(f"<b>[{lv}]</b> {txt}")
    table = go.Figure(data=[go.Table(
        columnorder=[1,2],
        columnwidth=[140, 540],
        header=dict(
            values=["<b>척도/하위척도</b>", "<b>특징</b>"],
            fill_color="#f2f2f2",
            align="left",
            font=dict(size=12)
        ),
        cells=dict(values=[rows_scale, rows_text],
                   align="left", height=28,
                   format=[None, None])
    )])
    table.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                        width=780, height=max(160, 36*len(rows_scale)+60))
    return table

with right:
    st.subheader("📝 자동 해석 (심리학 용어·요약표)")
    fig_table = make_interpret_table(subj_z_display)
    st.plotly_chart(fig_table, use_container_width=False)

# ---------------------------- PDF (메모리 생성 → 다운로드) ----------------------------
st.markdown("---")
st.subheader("📤 결과 리포트 PDF 다운로드")

# 한글 폰트 등록(가능하면 저장소에 fonts/NanumGothic.ttf 포함)
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
    return fig.to_image(format="png", scale=2)  # kaleido 필요

default_name = f"factor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
out_name = st.text_input("파일명", value=default_name)

if st.button("PDF 만들기"):
    try:
        bar_png = fig_to_png_bytes(fig_bar)
        rad_png = fig_to_png_bytes(fig_rad)
        table_png = fig_to_png_bytes(fig_table)

        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        W, H = A4

        # 제목
        c.setFont(FONT_NAME, 16)
        c.drawString(40, H-60, "52문항 요인 평가 리포트 (ND 표준화)")
        c.setFont(FONT_NAME, 10)
        y = H-90

        # 요인별 0–100 점수 요약
        for name, val in subj_t_display.items():
            vtxt = "NaN" if pd.isna(val) else f"{val:.1f}"
            c.drawString(40, y, f"{name}: {vtxt}")
            y -= 14
            if y < 120:
                c.showPage(); c.setFont(FONT_NAME, 10); y = H-60

        # 바그래프
        c.showPage(); c.setFont(FONT_NAME, 12); c.drawString(40, H-60, "요인 점수 (0–100)")
        if bar_png:
            img1 = ImageReader(io.BytesIO(bar_png))
            c.drawImage(img1, 50, 180, width=W-100, height=H-280, preserveAspectRatio=True, mask='auto')

        # 레이더
        if rad_png:
            c.showPage(); c.setFont(FONT_NAME, 12); c.drawString(40, H-60, "레이더 (0–100)")
            img2 = ImageReader(io.BytesIO(rad_png))
            c.drawImage(img2, 70, 160, width=W-140, height=H-320, preserveAspectRatio=True, mask='auto')

        # 해석표
        c.showPage(); c.setFont(FONT_NAME, 12); c.drawString(40, H-60, "자동 해석 요약표")
        if table_png:
            img3 = ImageReader(io.BytesIO(table_png))
            c.drawImage(img3, 40, 100, width=W-80, height=H-180, preserveAspectRatio=True, mask='auto')

        c.save()
        st.download_button("⬇️ PDF 다운로드", data=pdf_buffer.getvalue(),
                           file_name=out_name, mime="application/pdf")
    except Exception as e:
        st.error(f"PDF 생성 실패: {e}")

