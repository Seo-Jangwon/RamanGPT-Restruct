"""
SAM3 테스트 스크립트 - 텍스트 프롬프트로 객체 자동 추출
"""

from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
import numpy as np
import os
from pathlib import Path


def _enhance_contrast(image):
    """[개선] CLAHE를 사용한 이미지 대비 향상"""
    # Lab 색상 공간으로 변환하여 밝기(L) 채널만 조정
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
        
    # CLAHE 적용 (Clip Limit을 조절하여 대비 강도 조절)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
        
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def segment_with_text_prompt(image_path, text_prompts, output_dir="outputs", conf_threshold=0.25):
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SAM3 텍스트 프롬프트 세그멘테이션 시작")
    print(f"{'='*60}")
    print(f"이미지: {image_path}")
    print(f"프롬프트: {text_prompts}")
    print(f"Confidence: {conf_threshold}")
    
    # 모델 설정
    print("\n[1/5] SAM3 모델 로딩...")
    overrides = dict(
        conf=conf_threshold,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=True,
        save=False,
        verbose=False
    )
    
    predictor = SAM3SemanticPredictor(overrides=overrides)
    
    # 이미지 로드 및 설정
    print("[2/5] 이미지 로딩...")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
    
    # image_enhanced = _enhance_contrast(image)
    
    h, w = image.shape[:2]
    print(f"   이미지 크기: {w} x {h}")
    
    # image = cv2.GaussianBlur(image, (5, 5), 3)

    predictor.set_image(image)
    
    # 추론
    print(f"[3/5] 추론 중... (프롬프트: {text_prompts})")
    results = predictor(text=text_prompts)
    
    # 결과 처리
    print("[4/5] 결과 처리 중...")
    detected_objects = []
    obj_id = 0
    
    # 시각화용 이미지 복사 (한 번만!)
    vis_image = image.copy()
    
    for result_idx, result in enumerate(results):
        if result.masks is None:
            print(f"   프롬프트 '{text_prompts[result_idx]}': 객체 없음")
            continue
        
        num_objects = len(result.masks)
        print(f"   프롬프트 '{text_prompts[result_idx]}': {num_objects}개 객체 발견")
        
        # 마스크와 박스 추출
        masks = result.masks.data.cpu().numpy()  # [N, H, W]
        boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
        
        for i in range(num_objects):
            mask = masks[i]  # [H, W]
            
            # 픽셀 좌표 추출
            ys, xs = np.where(mask > 0.5)
            
            if len(xs) == 0:
                continue
            
            # 마스크에서 직접 바운딩 박스 계산
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            bbox = [x_min, y_min, x_max, y_max]  # [x1, y1, x2, y2]
            
            # 중심점 계산
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)
            
            # 픽셀 좌표 리스트
            pixel_coords = [{"x": int(x), "y": int(y)} for x, y in zip(xs, ys)]
            
            # 객체 데이터 구조 (기존 SAM 형식과 호환)
            obj_data = {
                "id": obj_id,
                "center_x": center_x,
                "center_y": center_y,
                "center_type": "bbox_center",
                "pixels": pixel_coords,
                "bbox": bbox,  # 이미 리스트임
                "prompt": text_prompts[result_idx] if result_idx < len(text_prompts) else "unknown"
            }
            detected_objects.append(obj_data)
            
            # 시각화: 마스크 오버레이 (초록색)
            mask_bool = mask > 0.5
            vis_image[mask_bool] = vis_image[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4
            
            # # 바운딩 박스 그리기 (빨간색)
            # cv2.rectangle(vis_image, 
            #              (int(bbox[0]), int(bbox[1])), 
            #              (int(bbox[2]), int(bbox[3])), 
            #              (0, 0, 255), 2)
            
            # # 중심점 표시 (파란색)
            # cv2.circle(vis_image, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # ID 텍스트
            cv2.putText(vis_image, f"ID:{obj_id}", 
                       (center_x - 20, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            print(f"      객체 {obj_id}: 중심=({center_x}, {center_y}), 픽셀={len(pixel_coords)}개")
            
            obj_id += 1
    
    # 결과 저장
    print("[5/5] 결과 저장 중...")
    
    # 시각화 이미지 저장
    output_image_path = os.path.join(output_dir, "sam3_result.png")
    cv2.imwrite(output_image_path, vis_image)
    print(f"   시각화 이미지 저장: {output_image_path}")
    
    # JSON 저장
    import json
    json_path = os.path.join(output_dir, "sam3_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detected_objects, f, indent=2, ensure_ascii=False)
    print(f"   JSON 데이터 저장: {json_path}")
    
    print(f"\n{'='*60}")
    print(f"완료! 총 {len(detected_objects)}개 객체 감지됨")
    print(f"{'='*60}\n")
    
    return detected_objects


def main():
    """메인 함수"""
    
    # ========== 설정 ==========
    IMAGE_PATH = "C:\\Users\\seoja\\Desktop\\RamanGPT\\RamanGPT\\backend\\tests\\data\\path3.png"
    TEXT_PROMPTS = ["cell"]  # 텍스트 프롬프트 (여러 개 가능)
    OUTPUT_DIR = "./outputs"
    CONF_THRESHOLD = 0.5
    # ==========================
    
    try:
        # SAM3 실행
        detected_objects = segment_with_text_prompt(
            image_path=IMAGE_PATH,
            text_prompts=TEXT_PROMPTS,
            output_dir=OUTPUT_DIR,
            conf_threshold=CONF_THRESHOLD
        )
        
        # 결과 요약 출력
        print("\n📊 결과 요약:")
        print(f"   - 총 객체 수: {len(detected_objects)}")
        if detected_objects:
            print(f"   - 첫 번째 객체 정보:")
            obj = detected_objects[0]
            print(f"      ID: {obj['id']}")
            print(f"      중심: ({obj['center_x']}, {obj['center_y']})")
            print(f"      픽셀 수: {len(obj['pixels'])}")
            print(f"      프롬프트: {obj['prompt']}")
        
        return detected_objects
        
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
