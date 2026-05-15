import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="LPSS 실전 출력 예측", layout="centered")
st.title("📟 K2 Plus 실전 출력 예측기 (V2)")
st.write("슬라이서의 '벽 3겹' 실제 데이터를 기준으로 보정한 버전입니다.")

uploaded_file = st.file_uploader("로고 이미지 업로드", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    col1, col2 = st.columns(2)
    with col1:
        real_width_mm = st.number_input("가로 폭 입력 (mm)", value=340.0)
    with col2:
        height_mm = st.number_input("두께 입력 (mm)", value=20.0)

    # 1. 면적 정밀 계산
    _, binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        all_points = np.concatenate(contours)
        _, _, p_width, _ = cv2.boundingRect(all_points)
        mm_per_px = real_width_mm / p_width
        
        # 순수 로고 단면적
        pure_area_px = sum([cv2.contourArea(cnt) for cnt in contours])
        actual_area_mm2 = pure_area_px * (mm_per_px ** 2)

        # 2. 실전 보정 계수 (슬라이서 190g 데이터에 맞춤)
        # 벽 3겹이 포함된 복잡한 로고는 전체 부피의 약 38%가 실제 재료량입니다.
        fill_factor = 0.38 
        estimated_weight_g = (actual_area_mm2 * height_mm * 1.24 * 0.001) * fill_factor
        
        # 시간 계산: 1g당 약 1.15분 + 예열/준비 20분
        total_minutes = (estimated_weight_g * 1.15) + 20
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)

        # 3. 결과 출력
        st.divider()
        st.subheader("📊 예상 결과 (K2 Plus 기준)")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("예상 시간", f"{hours}시간 {minutes}분")
        res_col2.metric("예상 무게", f"{estimated_weight_g:.1f} g")
        res_col3.metric("순수 단면적", f"{actual_area_mm2:.1f} mm²")

        st.info(f"💡 가로 {real_width_mm}mm, 두께 {height_mm}mm 기준 / 필라멘트 약 {estimated_weight_g / 1000:.2f}kg 소요")
        
        # 시각화
        cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 2)
        st.image(img_bgr, caption="인식된 로고 영역 (초록색 선 내부)", use_column_width=True)