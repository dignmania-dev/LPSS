import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 페이지 설정
st.set_page_config(page_title="LPSS 출력 예측 V.1.1", layout="centered")

# --- 헤더 섹션 (로고 우측 끝 배치) ---
col_title, col_spacer, col_logo = st.columns([2.5, 0.5, 1]) 

with col_title:
    st.title("LPSS 출력 예측 V.1.1")
    st.write("견적을 위한 예상 결과값으로 실제 견적과 다를 수 있습니다.")

with col_logo:
    try:
        logo = Image.open('layered&_wh.png')
        st.image(logo, use_container_width=True)
    except:
        st.caption("로고 파일 없음")

st.divider()

# --- 입력 섹션 ---
uploaded_file = st.file_uploader("로고 이미지 업로드", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 기본 정보 입력
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        real_width_mm = st.number_input("가로 폭 입력 (mm)", value=340.0)
    with col_in2:
        height_mm = st.number_input("두께 입력 (mm)", value=20.0)

    # 견적 타입 설정
    st.write("### **견적 타입 및 단가 설정**")
    col_type, col_price = st.columns(2)
    with col_type:
        type_name = st.text_input("타입 입력 (예: A, B, C...)", value="A")
    with col_price:
        hourly_rate = st.number_input(f"{type_name} 타입 시간당 금액 (원)", value=5000, step=500)

    # 면적 계산 로직
    _, binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        all_points = np.concatenate(contours)
        _, _, p_width, _ = cv2.boundingRect(all_points)
        mm_per_px = real_width_mm / p_width
        
        pure_area_px = sum([cv2.contourArea(cnt) for cnt in contours])
        actual_area_mm2 = pure_area_px * (mm_per_px ** 2)

        # 실전 보정 계수 및 10% 무게 할증
        fill_factor = 0.38 
        base_weight = (actual_area_mm2 * height_mm * 1.24 * 0.001) * fill_factor
        estimated_weight_g = base_weight * 1.10 
        
        # 시간 계산
        total_minutes = (estimated_weight_g * 1.15) + 20
        total_hours_decimal = total_minutes / 60
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)

        # 견적 가격 계산
        total_price = total_hours_decimal * hourly_rate

        # --- 결과 출력 섹션 ---
        st.divider()
        st.write(f"### **예상 결과 (K2 Plus / {type_name} 타입)**")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("예상 시간", f"{hours}시간 {minutes}분")
        res_col2.metric("예상 무게 (+10%)", f"{estimated_weight_g:.1f} g")
        res_col3.metric("순수 단면적", f"{actual_area_mm2:.1f} mm²")