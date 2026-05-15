import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 페이지 설정
st.set_page_config(page_title="LPSS 출력 예측 V.1.1", layout="centered")

# 1. 상단 로고 이미지 배치
# 파일 이름은 'layered&_wh.png'로 준비해 주세요.
try:
    logo = Image.open('layered&_wh.png')
    st.image(logo, width=200)
except:
    st.caption("로고 파일(layered&_wh.png)을 찾을 수 없습니다.")

st.title("LPSS 출력 예측 V.1.1")
st.write("견적을 위한 예상 결과값으로 실제 견적과 다를 수 있습니다.")

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

    # 면적 정밀 계산 (이진화 임계값 조절)
    _, binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        all_points = np.concatenate(contours)
        _, _, p_width, _ = cv2.boundingRect(all_points)
        mm_per_px = real_width_mm / p_width
        
        # 순수 로고 단면적
        pure_area_px = sum([cv2.contourArea(cnt) for cnt in contours])
        actual_area_mm2 = pure_area_px * (mm_per_px ** 2)

        # 실전 보정 계수 적용 (벽 3겹 기준 0.38)
        fill_factor = 0.38 
        base_weight = (actual_area_mm2 * height_mm * 1.24 * 0.001) * fill_factor
        
        # 요청사항: 예상 무게 10% 추가 보정
        estimated_weight_g = base_weight * 1.10
        
        # 시간 계산: 보정된 무게 기준 (1g당 1.15분 + 예열 20분)
        total_minutes = (estimated_weight_g * 1.15) + 20
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)

        # 결과 출력
        st.divider()
        st.subheader("📊 예상 결과 (K2 Plus 기준)")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("예상 시간", f"{hours}시간 {minutes}분")
        res_col2.metric("예상 무게 (+10%)", f"{estimated_weight_g:.1f} g")
        res_col3.metric("순수 단면적", f"{actual_area_mm2:.1f} mm²")

        st.info("💡 이 계산은 슬라이서의 '벽 3겹, 0.25mm 레이어, 인필 10%' 설정을 기준으로 보정되었습니다.")
        
        # 인식 범위 시각화
        cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 2)
        st.image(img_bgr, caption="인식된 로고 영역", use_column_width=True)
    else:
        st.error("로고 영역을 인식할 수 없습니다. 배경과의 대비를 확인해 주세요.")