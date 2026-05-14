import streamlit as st
import cv2
import numpy as np

def predict_k2_plus_time(image, real_width_mm, thickness_mm):
    # 1. 이미지 이진화 및 로고 영역 추출
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # 2. 기하학적 데이터 산출
    all_points = np.concatenate(contours)
    _, _, p_width, _ = cv2.boundingRect(all_points)
    mm_per_px = real_width_mm / p_width
    
    # 단면적(Area) 및 둘레(Perimeter) 계산
    total_area_mm2 = np.sum(binary == 255) * (mm_per_px ** 2)
    total_perimeter_mm = sum([cv2.arcLength(cnt, True) for cnt in contours]) * mm_per_px
    
    # 3. K2 Plus 실전 예측 수식 (슬라이서 데이터 반영)
    # 레이어 높이 0.25mm 기준 (스크린샷 설정 반영)
    total_layers = thickness_mm / 0.25
    
    # [A] 벽 출력 시간 (3겹 기준 가중치 적용)
    # 대형물일수록 코너 감속이 많아 가중 계수 1.8 적용
    wall_time_min = (total_perimeter_mm * 3 * total_layers / 150) / 60 * 1.8
    
    # [B] 인필 및 솔리드 채우기 시간 (10% 인필 + 상하단 스킨)
    # 스크린샷의 높은 인필 비중 반영을 위해 계수 2.2 적용
    infill_time_min = (total_area_mm2 * total_layers / (0.4 * 250)) / 60 * 2.2
    
    # [C] 고정 준비 시간 (베드 히팅 및 레벨링)
    prep_time_min = 25 
    
    # 총 합계
    total_min = wall_time_min + infill_time_min + prep_time_min
    
    return {
        "time_hr": int(total_min // 60),
        "time_min": int(total_min % 60),
        "area": total_area_mm2,
        "weight": total_area_mm2 * thickness_mm * 0.00125 * 1.5 # 1.5배 보정 계수 적용
    }

# --- UI 부분 ---
st.title("🖨️ K2 Plus 실전 출력 시간 예측기")
st.write("이미지를 분석하여 크리얼리티 슬라이서와 유사한 시간을 계산합니다.")

img_file = st.file_uploader("로고 이미지 업로드", type=['jpg', 'png'])
col1, col2 = st.columns(2)
with col1:
    in_w = st.number_input("가로 폭 입력 (mm)", value=350)
with col2:
    in_t = st.number_input("두께 입력 (mm)", value=20)

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    res = predict_k2_plus_time(gray_img, in_w, in_t)
    
    if res:
        st.divider()
        st.subheader("📊 예상 결과 (K2 Plus 기준)")
        c1, c2, c3 = st.columns(3)
        c1.metric("예상 시간", f"{res['time_hr']}시간 {res['time_min']}분")
        c2.metric("예상 무게", f"{res['weight']:.1f} g")
        c3.metric("단면적", f"{res['area']:.1f} mm²")
        
        st.warning(f"💡 이 계산은 슬라이서의 '벽 3겹, 0.25mm 레이어' 설정을 기준으로 보정되었습니다.")