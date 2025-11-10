# streamlit_app_nd_standardized_factor_report.py
# -*- coding: utf-8 -*-
"""
인터넷/Cloud 전용
- 52문항 → 4요인 평균 → ND 기준 표준화(Z) → 0–100 환산(T형식) 점수
- 프로파일(가로) 차트: K-CDI 스타일(축 40~90, 얇은 막대 표시)
- 자동해석: 심리학 용어(고위험/위험/주의/중립/안정 경향/안정/매우 안정) 표로 정리
- PDF: 프로파일 차트 + 해석표를 메모리에서 생성 후 즉시 다운로드
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
CLINICAL_GROUPS = ["ND","ASD","ADHD","SCD","HR"]

# ---------------------------- ⛳ 내장 기준값 (실제 수치로 교체하세요) ----------------------------
ND_BASE_MEAN = {"Factor1": 3.0, "Factor2": 3.2, "Factor3": 3.1, "Factor4": 3.0}
ND_BASE_STD  = {"Factor1": 0.6, "Factor2": 0.5, "Factor3": 0.5, "Factor4": 0.4}
GROUP_CENTROIDS_Z = {
    "ND"  : {"Factor1": 0.0,  "Factor2": 0.0,  "Factor3": 0.0,  "Factor4": 0.0},
    "ASD" : {"Factor1": 1.1,  "Factor2": -0.6, "Factor3": -0.2, "Factor4": -0.4},
    "ADHD": {"Factor1": 0.4,  "Factor2": -0.2, "Factor3": 0.6,  "Factor4": -0.1},
    "SCD" : {"Factor1": 0.7,  "Factor2": -1.0, "Factor3": -0.3, "Factor4": -0.8},
    "HR"  : {"Factor1": 0.3,  "Factor2": -0.1, "Factor3": 0.1,  "Factor4": 0.0},
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
    # 50 + 10*z 를 0~100로 클리핑 (표시는 40~90 축에 맞춰 별도 처리)
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

# 요인별 위험수준 → 문장
INTERP_DICT = {
    "사회적 의사소통": {
        "고위험":"대인 대화의 지속·상호성에서 현저한 어려움이 시사됩니다.",
        "위험":"의사소통 상호작용의 질적 저하가 관찰될 가능성이 큽니다.",
        "주의":"상대 반응조정/대화 유지에서 주의가 필요합니다.",
        "중립":"연령 기대 수준 내 기능으로 보입니다.",
        "안정 경향":"의사소통 상호작용에서 비교적 보호 요인이 관찰됩니다.",
        "안정":"사회적 의사소통 기능이 안정적입니다.",
        "매우 안정":"사회적 의사소통 기능이 매우 안정적입니다.",
    },
    "사회적 인식": {
        "고위험":"타인의 표정·의도 해석에 중대한 어려움이 시사됩니다.",
        "위험":"비언어적 단서 해석의 일관된 어려움이 예상됩니다.",
        "주의":"관계 맥락/암묵적 규칙 인식에 주의가 필요합니다.",
        "중립":"사회적 단서 인식이 중립 범위입니다.",
        "안정 경향":"단서 인식/상황 파악에서 비교적 보호적입니다.",
        "안정":"사회적 인식 기능이 안정적입니다.",
        "매우 안정":"사회적 인식 기능이 매우 안정적입니다.",
    },
    "사회적 동기": {
        "고위험":"대인 접근/참여 동기가 현저히 저하될 수 있습니다.",
        "위험":"또래 상호작용 회피 경향이 두드러질 수 있습니다.",
        "주의":"집단 활동 참여/지속에 주의가 필요합니다.",
        "중립":"대인 접근 동기가 중립 범위입니다.",
        "안정 경향":"대인 상호작용에 긍정적 접근이 관찰됩니다.",
        "안정":"사회적 동기가 안정적입니다.",
        "매우 안정":"사회적 동기가 매우 안정적입니다.",
    },
    "언어적 사회인지": {
        "고위험":"은유·관용구·숨은 뜻 이해에서 현저한 어려움이 시사됩니다.",
        "위험":"문맥 의도 추론의 일관된 어려움이 예상됩니다.",
        "주의":"간접화행/상황함의 이해에 주의가 필요합니다.",
        "중립":"언어적 사회인지가 중립 범위입니다.",
        "안정 경향":"의미 추론/맥락 이해가 비교적 보호적입니다.",
        "안정":"언어적 사회인지가 안정적입니다.",
        "매우 안정":"언어적 사회인지가 매우 안정적입니다.",
    },
}

# ---------------------------- K-CDI 스타일 프로파일 차트 ----------------------------
def make_profile_chart_t(t_series: pd.Series) -> go.Figure:
    """
    t_series: index=요인명(한글), values=0~100
    표시축: 40~90 (T척도 느낌), 얇은 수평 막대로 점만 표시
    """
    cats   = list(t_series.index)
    vals   = [None if pd.isna(v) else float(v) for v in t_series.values]
    xpos   = [None if v is None else max(40.0, min(90.0, v)) for v in vals]
    seg_w  = 1.8
    bases  = [None if x is None else x - seg_w/2 for x in xpos]
    widths = [0 if b is None else seg_w for b in bases]

    cats_rev   = cats[::-1]
    bases_rev  = bases[::-1]
    widths_rev = widths[::-1]
    colors_rev = [bar_colors.get(c, "#999999") for c in cats_rev]

    fig = go.Figure()

    # 배경 프레임
    fig.add_shape(type="rect", x0=40, x1=90, y0=-0.5, y1=len(cats)-0.5,
                  line=dict(color="#444", width=1), fillcolor="white")

    # 세로 점선 그리드
    for x in range(40, 91, 5):
        fig.add_vline(x=x, line=dict(color="#dddddd", width=1, dash="dot"))

    fig.add_trace(go.Bar(
        y=cats_rev,
        x=widths_rev,
        base=bases_rev,
        orientation="h",
        marker_color=colors_rev,
        marker_line=dict(width=0),
        hovertemplate="%{y} : T=%{customdata:.1f}<extra></extra>",
        customdata=[v for v in vals[::-1]],
        showlegend=False,
    ))

    # 왼쪽 T점수 텍스트
    for i, v in enumerate(vals[::-1]):
        if v is not None:
            fig.add_annotation(x=39.2, y=i, text=f"{int(round(v))}",
                               xanchor="right", yanchor="middle",
                               showarrow=False, font=dict(size=12))

    fig.add_annotation(x=90, y=len(cats)-0.9, text="단위: T점수",
                       xanchor="right", yanchor="bottom",
                       showarrow=False, font=dict(size=11, color="#444"))

    fig.update_xaxes(range=[39, 91], tickmode="array",
                     tickvals=list(range(40, 91, 5)),
                     showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_layout(height=max(260, 70*len(cats)),
                      width=680,  # ▶ 가로폭 (원하면 조절)
                      margin=dict(l=120, r=30, t=30, b=40))
    return fig

# ---------------------------- 해석표(Table) ----------------------------
def make_interpret_table(subj_z_display: pd.Series) -> go.Figure:
    rows_scale, rows_text = [], []
    for name, z in subj_z_display.items():
        lv  = level_from_z(z)
        txt = INTERP_DICT.get(name, {}).get(lv, f"{name}: {lv}")
        rows_scale.append(name)
        rows_text.append(f"[{lv}] {txt}")
    table = go.Figure(data=[go.Table(
        columnorder=[1,2],
        columnwidth=[140, 520],
        header=dict(
            values=["<b>척도/하위척도</b>", "<b>특징</b>"],
            fill_color="#f2f2f2",
            align="left",
            font=dict(size=12)
        ),
        cells=dict(values=[rows_scale, rows_text], align="left", height=26)
    )])
    table.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                        width=740, height=max(140, 32*len(rows_scale)+60))
    return table

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
left, right = st.columns([1.0, 1.0])

with left:
    st.subheader("📊 결과 프로파일 (T 40–90)")
    fig_profile = make_profile_chart_t(subj_t_display)
    st.plotly_chart(fig_profile, use_container_width=False)

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

# 파일명
default_name = f"factor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
out_name = st.text_input("파일명", value=default_name)

if st.button("PDF 만들기"):
    try:
        profile_png = fig_to_png_bytes(fig_profile)
        table_png   = fig_to_png_bytes(fig_table)

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

        # 프로파일 차트
        c.showPage(); c.setFont(FONT_NAME, 12); c.drawString(40, H-60, "결과 프로파일 (T 40–90)")
        if profile_png:
            img1 = ImageReader(io.BytesIO(profile_png))
            c.drawImage(img1, 40, 140, width=W-80, height=H-220, preserveAspectRatio=True, mask='auto')

        # 해석표
        c.showPage(); c.setFont(FONT_NAME, 12); c.drawString(40, H-60, "자동 해석 요약표")
        if table_png:
            img2 = ImageReader(io.BytesIO(table_png))
            c.drawImage(img2, 40, 100, width=W-80, height=H-180, preserveAspectRatio=True, mask='auto')

        c.save()
        st.download_button("⬇️ PDF 다운로드", data=pdf_buffer.getvalue(),
                           file_name=out_name, mime="application/pdf")
    except Exception as e:
        st.error(f"PDF 생성 실패: {e}")

# ---------------------------- 도움말 ----------------------------
st.markdown(
    """
**메모**  
- ND 기준/임상군 중심은 코드 상단 상수(`ND_BASE_MEAN`, `ND_BASE_STD`, `GROUP_CENTROIDS_Z`)를 실제 값으로 교체하세요.  
- 프로파일 차트의 가로폭은 함수 내부 `width=680`에서 조절할 수 있습니다.  
- PDF 한글을 위해 `fonts/NanumGothic.ttf` 포함을 권장합니다(없으면 시스템 기본 폰트로 대체).  
"""
)
