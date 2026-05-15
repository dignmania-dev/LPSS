import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 페이지 설정
st.set_page_config(page_title="LPSS 출력 예측 V.1.1", layout="centered")

# --- 헤더 섹션 (로고 우측 배치) ---
col_title, col_logo = st.columns([3, 1]) # 왼쪽 3, 오른쪽 1 비율

with col_title:
    st.title("LPSS 출력 예측 V.1.1")
    st.write("견적을 위한 예상 결과값으로 실제 견적과 다를 수 있습니다.")

with col_logo:
    try:
        # 로고를 불러와 우측에 배치
        logo = Image.open('layered&_wh.png')
        st.image(logo, use_container_width=True)
    except:
        st.caption("로고 파일 없음")

st.divider() # 구분선 추가

# --- 입력 섹션 ---
uploaded_file = st.file_uploader("로고 이미지 업로드 (화이트 배경에 블랙 로고 권장)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        real_width_mm = st.number_input("가로 폭 입력 (mm)", value=340.0, step=1.0)
    with col_in2:
        height_mm = st.number_input("두께 입력 (mm)", value=20.0, step=1.0)

    # 면적 계산 로직 (순수 로고 영역 추출)
    _, binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 가로 픽셀 폭 산출
        all_points = np.concatenate(contours)
        _, _, p_width, _ = cv2.boundingRect(all_points)
        mm_per_px = real_width_mm / p_width
        
        # 순수 단면적 계산
        pure_area_px = sum([cv2.contourArea(cnt) for cnt in contours])
        actual_area_mm2 = pure_area_px * (mm_per_px ** 2)

        # 실전 보정 계수 (K2 Plus / 벽 3겹 / 인필 10% 기준)
        fill_factor = 0.38 
        base_weight = (actual_area_mm2 * height_mm * 1.24 * 0.001) * fill_factor
        
        # 예상 무게 10% 안전 보정 추가
        estimated_weight_g = base_weight * 1.10
        
        # 시간 계산 (보정된 무게 기준 1g당 1.15분 + 가열 20분)
        total_minutes = (estimated_weight_g * 1.15) + 20
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)

        # --- 결과 출력 섹션 ---
        st.subheader("📊 예상 결과 (K2 Plus 기준)")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("예상 시간", f"{hours}시간 {minutes}분")
        res_col2.metric("예상 무게 (+10%)", f"{estimated_weight_g:.1f} g")
        res_col3.metric("순수 단면적", f"{actual_area_mm2