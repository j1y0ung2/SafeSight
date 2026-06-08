import streamlit as st
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# ==========================================
# [안정화 1단계] 웹페이지 기본 레이아웃 및 세션 설정
# ==========================================
st.set_page_config(page_title="SafeSight", page_icon="🎯", layout="wide")

# 리소스 캐싱을 통해 사용자가 클릭할 때마다 모델이 재로드되어 느려지는 현상 방지
@st.cache_resource
def load_integrated_assets():
    try:
        # 1. YOLOv8 동물 탐지 가중치 로드
        yolo = YOLO('models/best.pt') 
        
        # 2. 멀티모달 CLIP 모델 및 전처리기 로드
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 3. 각각 추출한 강아지/고양이 DB 로드
        dog_embeds = np.load('models/shelter_embeddings.npy') 
        dog_paths = np.load('models/shelter_paths.npy')
        
        cat_embeds = np.load('models/cat_embeddings.npy')
        cat_paths = np.load('models/cat_paths.npy')
        
        # [수학적 통합] 두 동물 데이터를 하나로 병합
        total_embeddings = np.vstack([dog_embeds, cat_embeds])
        total_paths = np.concatenate([dog_paths, cat_paths])
        
        return yolo, clip, processor, total_embeddings, total_paths
    except FileNotFoundError as e:
        st.error(f"📂 [파일 로드 실패] models/ 폴더 안에 필수 파일(best.pt, npy 파일들)이 모두 들어있는지 확인해주세요. 에러 내용: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ [시스템 에러] AI 모델 자산을 불러오는 중 오류가 발생했습니다: {e}")
        st.stop()

# 파일 로드 실행
yolo_model, clip_model, clip_processor, total_embeds, total_paths = load_integrated_assets()

# ==========================================
# [안정화 2단계] UI 화면 디자인
# ==========================================
st.title("🎯 SafeSight - AI 유기동물 통합 매칭 시스템")
st.write("YOLOv8의 객체 추적 기술과 CLIP의 멀티모달 매칭 기술을 결합하여 유기동물을 실시간으로 찾아냅니다.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🕵️‍♂️ 발견동물 정보 입력")
    
    # [안정화] 이미지 파일 확장자 제한 필터링
    uploaded_file = st.file_uploader("발견하거나 보호 중인 동물 사진을 업로드하세요", type=["jpg", "jpeg", "png"])
    
    # [안정화] 검색어 인풋창 가이드 제공
    text_query = st.text_input("전단지 정보 혹은 특징 키워드 입력", placeholder="예: 삼색 고양이, 갈색 푸들, 흰색 말티즈")
    
    # 텍스트와 사진 중 하나라도 누락되었을 때 띄워줄 방어 안내 멘트
    if uploaded_file is None:
        st.info("📸 매칭 연산을 시작하려면 먼저 고양이 또는 강아지 사진을 업로드해 주세요.")
    elif not text_query:
        st.warning("✍️ 더 정확한 대조를 위해 사진 속 동물의 특징 키워드(예: 고등어 태비)를 입력해 주세요.")

with col2:
    st.subheader("📊 AI 모델 연산 및 매칭 결과")
    
    if uploaded_file is not None:
        # [안정화] 손상된 이미지 파일 입력 시 예외 처리
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
            st.image(input_image, caption="📷 사용자가 업로드한 원본 사진", use_container_width=True)
        except Exception:
            st.error("🚨 읽을 수 없는 이미지 파일입니다. 깨지지 않은 정상 이미지를 다시 올려주세요.")
            st.stop()
        
        # ------------------------------------------
        # [안정화 3단계] YOLO 가동 (로딩 바 레이아웃 분리)
        # ------------------------------------------
        with st.spinner("🕵️‍♂️ YOLOv8 모델이 이미지에서 동물 영역을 정밀 탐색 중입니다..."):
            yolo_results = yolo_model(input_image, verbose=False)
            boxes = yolo_results[0].boxes
            cropped_img = None
            
            for box in boxes:
                if float(box.conf[0]) >= 0.4:  # 신뢰도 40% 이상만 크롭
                    xyxy = box.xyxy[0].tolist()
                    cropped_img = input_image.crop((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
                    break
        
        # [안정화] YOLO가 동물을 찾지 못했을 때 멈추지 않고 원본으로 우회하는 예외 처리
        if cropped_img is None:
            cropped_img = input_image
            st.info("💡 YOLO가 특정 동물 구역을 감지하지 못해 원본 전체 영역으로 분석을 진행합니다.")
        else:
            st.image(cropped_img.resize((224, 224)), caption="✂️ YOLO 정밀 크롭 완료 (224x224)", width=224)
            
        # ------------------------------------------
        # [안정화 4단계] CLIP 가동 및 최종 매칭
        # ------------------------------------------
        if text_query:
            with st.spinner("🔄 통합 벡터 데이터베이스 내 대용량 특징 행렬 대조 중..."):
                try:
                    inputs = clip_processor(images=cropped_img.resize((224, 224)), return_tensors="pt")
                    
                    with torch.no_grad():
                        query_features = clip_model.get_image_features(**inputs)
                        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
                        query_np = query_features.cpu().numpy()[0]
                    
                    # 통합 DB 코사인 유사도 연산
                    similarities = np.dot(total_embeds, query_np)
                    max_idx = np.argmax(similarities)
                    match_prob = similarities[max_idx] * 100 
                    
                    # [안정화] 매칭 유사도 결과 출력 및 하이라이트 제공
                    st.success("📈 통합 데이터베이스 매칭 성공!")
                    st.metric(label="최고 유사도 일치율", value=f"{match_prob:.2f}%")
                    
                    # ==========================================
                    # 🔥 [추가된 안정화 구역] 결과 이미지 카드 생성 및 태그 에러 방어
                    # ==========================================
                    st.markdown("### 🐾 가장 닮은 동물 검색 결과")
                    
                    # 매칭된 매칭 데이터를 item 딕셔너리 형태로 가상 래핑 (기존 컴포넌트 호환용)
                    matched_path = total_paths[max_idx]
                    
                    # 만약 태그 구조가 터지는걸 방지하기 위한 안전한 더미 데이터 구조 바인딩
                    item = {
                        "image_path": matched_path,
                        "tags": [("유사 결과", "breed"), (f"{match_prob:.1f}% 일치", "color")]
                    }
                    
                    # 문제의 태그 HTML 파싱 구역 안정화 솔루션 적용
                    tags_list = []
                    if "tags" in item and item["tags"]:
                        for tag in item["tags"]:
                            if isinstance(tag, (list, tuple)) and len(tag) == 2:
                                t, cls = tag
                                tags_list.append(f'<span style="background-color:#e1f5fe; padding:2px 8px; margin-right:5px; border-radius:4px; font-size:14px; color:#0288d1;">#{t}</span>')
                            elif isinstance(tag, str):
                                tags_list.append(f'<span style="background-color:#e1f5fe; padding:2px 8px; margin-right:5px; border-radius:4px; font-size:14px; color:#0288d1;">#{tag}</span>')
                    tags_html = "".join(tags_list)
                    
                    # 화면에 보호소 결과 출력 및 태그 표출
                    st.info(f"📂 매칭된 보호소 데이터 경로: \n`{matched_path}`")
                    st.markdown(tags_html, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ 매칭 연산 중 오류가 발생했습니다: {e}")
