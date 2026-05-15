import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="LPSS 출력 예측", layout="centered")
st.title("🖨️ LPSS 출력 예측")
st.write("슬라이서 데이터(벽 3겹, 10% 인필)를 기준으로 보정된 계산기입니다.")

uploaded_file = st.file_uploader("로고 이미지 업로드", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 이미지 처리
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. 사용자 입력
    col1, col2 = st.columns(2)
    with col1:
        real_width_mm = st.number_input("가로 폭 입력 (mm)", value=340.0)
    with col2:
        height_mm = st.number_input("두께 입력 (mm)", value=20.0)

    # 2. 면적 정밀 계산 (빈 공간 제외)
    # 이진화: 배경과 로고 분리
    _, binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 전체를 감싸는 가로 픽셀 폭 구하기
        all_points = np.concatenate(contours)
        _, _, p_width, _ = cv2.boundingRect(all_points)
        
        # 픽셀당 mm 비율
        mm_per_px = real_width_mm / p_width
        
        # 순수 로고 단면적 계산 (픽셀 단위 면적 합산)
        pure_area_px = sum([cv2.contourArea(cnt) for cnt in contours])
        actual_area_mm2 = pure_area_px * (mm_per_px ** 2)

        # 3. 보정 계수 적용 (슬라이서 데이터 기반)
        # 밀도: 1.24 (PLA), 충진 가중치: 0.22 (벽 3겹 + 인필 10%의 실질 채움 비율)
        fill_factor = 0.22 
        estimated_weight_g = (actual_area_mm2 * height_mm * 1.24 * 0.001) * fill_factor
        
        # 시간 계산: 1g당 1.3분 + 예열/준비 50분
        total_minutes = (estimated_weight_g * 1.3) + 50
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)

        # 4. 결과 출력
        st.divider()
        st.subheader("📊 예상 결과 (K2 Plus 기준)")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("예상 시간", f"{hours}시간 {minutes}분")
        res_col2.metric("예상 무게", f"{estimated_weight_g:.1f} g")
        res_col3.metric("순수 단면적", f"{actual_area_mm2:.1f} mm²")

        st.warning(f"💡 이 계산은 슬라이서의 **'벽 3겹, 0.25mm 레이어, 인필 10%'** 설정을 기준으로 보정되었습니다.")
        
        # 5. 시각화 (인식된 영역 확인용)
        st.write("### 로고 인식 범위 확인")
        cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 2)
        st.image(img_bgr, caption="초록색 선 내부 면적만 계산에 포함됩니다.", use_column_width=True)
    else:
        st.error("로고를 인식할 수 없습니다.")