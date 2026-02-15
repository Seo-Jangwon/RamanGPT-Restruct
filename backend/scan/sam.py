"""
서장원

SAM 
"""

import cv2
import numpy as np
import torch
import json
import os
import logging
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy

logger = logging.getLogger(__name__)

class SAMMultiSegmenter:
    def __init__(self, checkpoint_path, image_path, max_display_size=800):
        # 모델 로드
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 모델 초기화
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        
        self.predictor = SamPredictor(self.sam)
        
        # [개선 1] 자동 마스크 생성기 파라미터 튜닝 (배경 노이즈 제거용)
        # points_per_side를 줄이고, thresh를 높여 확실한 것만 잡도록 설정
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.92,      # 기본값 0.88 -> 0.92 상향 (확실한 것만)
            stability_score_thresh=0.95, # 기본값 0.95 유지 (불안정한 마스크 제거)
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100   # 너무 작은 잡티 제거
        )
        
        # 이미지 로드
        self.image_path = image_path
        self.image_original = self._imread_korean(image_path)
        if self.image_original is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        
        # [개선 2] 이미지 전처리 (Contrast Enhancement)
        # SAM이 투명한 객체를 더 잘 보게 하기 위해 CLAHE 적용된 이미지를 준비
        self.image_enhanced = self._enhance_contrast(self.image_original)
        
        # SAM에는 전처리된 이미지를 설정 (인식률 향상)
        print("이미지 로드 완료, SAM 인코딩 중 (Enhanced Image)...")
        image_rgb_enhanced = cv2.cvtColor(self.image_enhanced, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb_enhanced)
        print("준비 완료!")
        
        # 디스플레이용 이미지 크기 조정
        self.max_display_size = max_display_size
        self.scale = self._calculate_scale(self.image_original, max_display_size)
        self.image_display = self._resize_image(self.image_original, self.scale)
        
        print(f"원본 크기: {self.image_original.shape[:2]}")
        print(f"초기 스케일: {self.scale:.3f}")
        
        # 상태 변수들
        self.points = []
        self.labels = []
        self.mask = None
        self.all_masks = []
        self.selected_masks = []
        self.similarity_threshold = 0.9
        
        # Connected Components 관련
        self.labeled_objects = None
        self.num_objects = 0
        self.object_stats = None
        self.min_object_size = 50
        
        # Undo/Redo
        self.history = []
        self.history_index = -1
        self.max_history = 50
        
        # Zoom/Pan 기능 (원본 이미지 기준으로 작동)
        self.zoom_level = 1.0
        self.zoom_min = 0.3
        self.zoom_max = 8.0
        self.zoom_step = 0.15
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # 윈도우 크기
        self.window_width = 1200
        self.window_height = 800
        
        # ===== 누적 저장 기능 추가 =====
        self.accumulated_saves = []  # S키로 저장한 객체들을 누적
        self._save_state()

    def _enhance_contrast(self, image):
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

    def _imread_korean(self, image_path):
        """한글 경로 처리를 위한 이미지 읽기"""
        try:
            stream = open(image_path, "rb")
            bytes_data = bytearray(stream.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            stream.close()
            return image
        except Exception as e:
            print(f"이미지 로드 실패: {e}")
            return None
    
    def _imwrite_korean(self, image_path, image):
        """한글 경로 처리를 위한 이미지 저장"""
        try:
            extension = image_path.split('.')[-1]
            result, encoded_img = cv2.imencode(f'.{extension}', image)
            if result:
                with open(image_path, mode='w+b') as f:
                    encoded_img.tofile(f)
                return True
        except Exception as e:
            print(f"이미지 저장 실패: {e}")
            return False
    
    def _calculate_scale(self, image, max_size):
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > max_size:
            return max_size / max_dim
        return 1.0
    
    def _resize_image(self, image, scale):
        if scale == 1.0:
            return image
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _clamp_pan(self):
        """팬 좌표가 이미지 범위를 벗어나지 않도록 제한"""
        h, w = self.image_original.shape[:2]
        
        # 현재 줌 레벨에서 보여지는 창의 실제 크기 (원본 스케일)
        view_w = self.window_width / self.zoom_level
        view_h = self.window_height / self.zoom_level
        
        # 최대 이동 가능한 좌표 (이미지 크기 - 보여지는 크기)
        max_pan_x = max(0, w - view_w)
        max_pan_y = max(0, h - view_h)
        
        # 좌표 제한 (0 ~ max 사이로 고정)
        self.pan_x = max(0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0, min(self.pan_y, max_pan_y))

    def _scale_point(self, x, y):
        """화면 좌표를 원본 이미지 좌표로 변환 (줌/팬 고려)"""
        # 화면 좌표 -> 줌된 이미지 좌표
        zoomed_x = x / self.zoom_level + self.pan_x
        zoomed_y = y / self.zoom_level + self.pan_y
        return int(zoomed_x), int(zoomed_y)
    
    def _get_display_region(self):
        """현재 줌/팬 상태에서 표시할 이미지 영역 계산"""
        h, w = self.image_original.shape[:2]
        
        # 표시 영역 크기
        display_w = int(self.window_width / self.zoom_level)
        display_h = int(self.window_height / self.zoom_level)
        
        # 시작 좌표 (팬 적용)
        x1 = max(0, int(self.pan_x))
        y1 = max(0, int(self.pan_y))
        x2 = min(w, x1 + display_w)
        y2 = min(h, y1 + display_h)
        
        # 경계 체크
        if x2 - x1 < display_w:
            x1 = max(0, x2 - display_w)
        if y2 - y1 < display_h:
            y1 = max(0, y2 - display_h)
        
        return x1, y1, x2, y2
    
    def _save_state(self):
        self.history = self.history[:self.history_index + 1]
        state = (
            deepcopy(self.points),
            deepcopy(self.labels),
            self.mask.copy() if self.mask is not None else None,
            deepcopy(self.selected_masks)
        )
        self.history.append(state)
        self.history_index += 1
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
    
    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self._restore_state()
            print(f"Undo (히스토리: {self.history_index + 1}/{len(self.history)})")
        else:
            print("더 이상 되돌릴 수 없습니다")
    
    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self._restore_state()
            print(f"Redo (히스토리: {self.history_index + 1}/{len(self.history)})")
        else:
            print("더 이상 앞으로 갈 수 없습니다")
    
    def _restore_state(self):
        if 0 <= self.history_index < len(self.history):
            state = self.history[self.history_index]
            self.points = deepcopy(state[0])
            self.labels = deepcopy(state[1])
            self.mask = state[2].copy() if state[2] is not None else None
            self.selected_masks = deepcopy(state[3])
            if self.mask is not None:
                self.label_connected_objects()
            self.show_result()
    
    def label_connected_objects(self):
        if self.mask is None:
            return
        
        mask_uint8 = self.mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )
        
        valid_labels = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_object_size:
                valid_labels.append(i)
        
        new_labels = np.zeros_like(labels)
        for new_id, old_id in enumerate(valid_labels, start=1):
            new_labels[labels == old_id] = new_id
        
        self.labeled_objects = new_labels
        self.num_objects = len(valid_labels)
        
        self.object_stats = []
        for new_id, old_id in enumerate(valid_labels, start=1):
            stat = {
                'id': new_id,
                'area': int(stats[old_id, cv2.CC_STAT_AREA]),
                'centroid': (float(centroids[old_id, 0]), float(centroids[old_id, 1])),
                'bbox': {
                    'x': int(stats[old_id, cv2.CC_STAT_LEFT]),
                    'y': int(stats[old_id, cv2.CC_STAT_TOP]),
                    'width': int(stats[old_id, cv2.CC_STAT_WIDTH]),
                    'height': int(stats[old_id, cv2.CC_STAT_HEIGHT])
                }
            }
            self.object_stats.append(stat)
        
        print(f"✓ 총 {self.num_objects}개 객체 감지됨")

    def _extract_features(self, mask):
        image_rgb = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2RGB)
        
        masked_region = image_rgb[mask]
        if len(masked_region) == 0:
            return None
        
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        masked_hsv = hsv[mask]
        
        hist_h = np.histogram(masked_hsv[:, 0], bins=30, range=(0, 180))[0]
        hist_s = np.histogram(masked_hsv[:, 1], bins=32, range=(0, 256))[0]
        hist_v = np.histogram(masked_hsv[:, 2], bins=32, range=(0, 256))[0]
        
        hist_h = hist_h / (np.sum(hist_h) + 1e-6)
        hist_s = hist_s / (np.sum(hist_s) + 1e-6)
        hist_v = hist_v / (np.sum(hist_v) + 1e-6)
        
        mean_color = np.mean(masked_region, axis=0)
        
        features = np.concatenate([hist_h, hist_s, hist_v, mean_color / 255.0])
        return features
    
    def _calculate_similarity(self, features1, features2):
        if features1 is None or features2 is None:
            return 0.0
        return cosine_similarity([features1], [features2])[0][0]

    # 원래 찾기로직
    # def _detect_candidate_points(self):
    #     """Background Difference
    #     - 배경을 추정한 이미지 - 원본 이미지 = 객체만 남음
    #     """
    #     print("객체 후보 지점 탐색 중 (Background Difference)...")
        
    #     # 1. 그레이스케일 변환
    #     gray = cv2.cvtColor(self.image_enhanced, cv2.COLOR_BGR2GRAY)
        
    #     # 2. 배경 추정 (이미지를 심하게 뭉개서 객체를 지워버림)
    #     # ksize는 객체 크기보다 훨씬 커야 함(홀수)
    #     k_size = 101

    #     '''
    #     gray: 입력 이미지 (그레이스케일)
    #     (k_size, k_size): 커널(필터) 크기 - 반드시 홀수여야 함 (3, 5, 7, 9, ...), 클수록 더 많이 블러처리됨
    #     0: sigmaX (가우시안 커널의 표준편차), 0으로 설정하면 커널 크기에서 자동으로 계산됨
    #     '''
    #     bg_blur = cv2.GaussianBlur(gray, (k_size, k_size), 0)
        
    #     # 3. 차영상 구하기 (Diff = 배경 - 원본)
    #     # 배경은 밝고(값 높음), 객체는 어두우므로(값 낮음), 빼면 객체 부분이 양수로 남음
    #     diff = cv2.subtract(bg_blur, gray)
        
    #     # 4. Thresholding (이진화)
    #     # "주변 배경보다 30만큼 더 진한 놈만 남겨라"
    #     # 이 값을 조절하면 민감도가 변함 (20: 연한 것도 잡음, 50: 아주 진한 것만 잡음)
    #     sensitivity = 20 

    #     '''
    #     diff: 입력 이미지 (그레이스케일, 0~255 범위)
    #     sensitivity (20): 임계값(threshold) - 이 값을 기준으로 구분,  20보다 큰 변화만 의미있는 변화로 간주
    #     255: maxval - 임계값을 넘었을 때 설정할 값
    #     cv2.THRESH_BINARY: 이진화 방식
    #     ret: 사용된 임계값 (자동 계산 타입 아니면 그냥 sensitivity 반환)
    #     binary: 결과 이미지 (0 또는 255만 존재)
    #     '''
    #     ret, binary = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_OTSU)
        
    #     # 5. 노이즈 제거 (아주 작은 점 삭제)
    #     kernel = np.ones((5, 5), np.uint8)

    #     '''
    #     - binary: 입력 이진 이미지 (0 또는 255)
    #     - cv2.MORPH_OPEN: Opening 연산 (작은 노이즈 제거 -> 큰 객체는 복원)
    #     - kernel: (모두 1로 채워진 행렬)
    #     - iterations=1: 연산 반복 횟수
    #     '''
    #     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
    #     # 6. 윤곽선 검출 및 면적 필터링
    #     '''
    #     - binary: 입력 이진 이미지 (0과 255만 존재)
    #     - cv2.RETR_EXTERNAL: 검색 모드 (어떤 윤곽선을 찾을지, 여기선 외부 윤곽선만 검출)
    #     - cv2.CHAIN_APPROX_SIMPLE: 근사 방법 (윤곽선을 어떻게 저장할지, 여기선 직선 구간의 중간 점들을 생략해서 저장)
    #     - contours: 검출된 윤곽선 리스트 (각 contour는 좌표 배열)
    #     '''
    #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     candidate_points = []
    #     h, w = gray.shape
        
    #     # 필터링 기준
    #     min_area = 150        # 너무 작은 점(먼지/노이즈) 무시
    #     max_area = (h * w) * 0.2  # 화면 20% 이상 덮는 거대 그림자 무시
        
    #     # 디버깅용: 몇 개나 잡혔는지 면적과 함께 확인
    #     valid_cnt_count = 0
        
    #     # 면적 순 정렬 (큰 놈부터 우선 처리)
    #     contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
    #     # 상위 50개까지만 (너무 많이 잡히는 것 방지)
    #     contours = contours[:50]

    #     for cnt in contours:
    #         area = cv2.contourArea(cnt)
            
    #         if min_area < area < max_area:
    #             M = cv2.moments(cnt)
    #             if M["m00"] != 0:
    #                 cX = int(M["m10"] / M["m00"])
    #                 cY = int(M["m01"] / M["m00"])
    #                 candidate_points.append([cX, cY])
    #                 valid_cnt_count += 1
                
    #     print(f" -> 배경 제거 후 {valid_cnt_count}개의 유효 객체 발견")
        
    #     if valid_cnt_count == 0:
    #         print(" [알림] 검출된 객체가 없습니다. sensitivity(현재 30)를 낮춰보세요 (예: 20).")
    #         return np.array([])
            
    #     return np.array(candidate_points)

    # 개선된 찾기로직
    def _detect_candidate_points(self):
        print("객체 후보 지점 탐색 중 (Visual Debug Mode)...")
        
        if self.image_enhanced is None:
            print("Error: 이미지가 로드되지 않았습니다.")
            return np.array([])
            
        # # 화면에 보여줄 최대 너비
        # VIEW_WIDTH = 1000  
        
        # # 원본 크기 및 스케일 비율 계산
        # h, w = self.image_enhanced.shape[:2]
        # scale = 1.0
        # if w > VIEW_WIDTH:
        #     scale = VIEW_WIDTH / w
            
        # 1. 그레이스케일 변환
        gray = cv2.cvtColor(self.image_enhanced, cv2.COLOR_BGR2GRAY)
        
        # # [시각화 1] 그레이스케일 (리사이징 적용)
        # display_gray = cv2.resize(gray, None, fx=scale, fy=scale)
        # cv2.imshow("DEBUG Step 1: Grayscale", display_gray)
        # cv2.waitKey(0) 

        # 2. Adaptive Threshold
        block_size = 101  
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=20 
        )

        # # [시각화 2] 이진화 결과
        # display_binary = cv2.resize(binary, None, fx=scale, fy=scale)
        # cv2.imshow("DEBUG Step 2: Adaptive Threshold", display_binary)
        # cv2.waitKey(0)

        # 3. Morphology
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        kernel_close = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # # [시각화 3] 모폴로지 후 (리사이징 적용)
        # display_morph = cv2.resize(binary, None, fx=scale, fy=scale)
        # cv2.imshow("DEBUG Step 3: Morphology Result", display_morph)
        # cv2.waitKey(0)
        
        # 4. 윤곽선 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # # 그림 그릴 도화지 (컬러)
        # debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        candidate_points = []
        valid_cnt_count = 0
        
        # 면적 순 정렬
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]
        
        for i, cnt in enumerate(contours):
            # Moments 계산
            # 윤곽선 내부의 픽셀 분포를 계산하여 딕셔너리 형태인 M으로 반환
            # M['m00']은 면적, M['m10'], M['m01']은 위치 관련 값을 담고 있음
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                candidate_points.append([cX, cY])
                valid_cnt_count += 1
                
                # # 원본 크기 이미지에 그리기
                # cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
                # cv2.circle(debug_img, (cX, cY), 5, (0, 0, 255), -1)
                # cv2.putText(debug_img, f"{i}", (cX+5, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        print(f" -> {valid_cnt_count}개의 유효 객체 발견")
        
        # # [시각화 4] 최종 결과 (리사이징 적용)
        # # 그림은 원본 해상도에 그렸고, 보여줄 때만 줄여서 보여줌
        # display_final = cv2.resize(debug_img, None, fx=scale, fy=scale)
        # cv2.imshow("DEBUG Step 4: Final Candidates", display_final)
        # print(">> 확인 후 아무 키나 누르면 SAM이 실행됩니다.")
        # cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        if valid_cnt_count == 0:
            print(" [알림] 검출된 객체가 없습니다.")
            return np.array([])
        
        return np.array(candidate_points)

    def generate_all_masks(self):
        
        candidate_points = self._detect_candidate_points()
        
        if len(candidate_points) == 0:
            print("감지된 객체 후보가 없습니다.")
            return

        print(f"SAM 예측 시작 (총 {len(candidate_points)}개 지점)...")
        
        new_masks = []
        
        for i, point in enumerate(candidate_points):
            if i % 10 == 0:
                print(f"처리 중... {i}/{len(candidate_points)}", end='\r')
            
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            
            if scores[0] > 0.85:
                new_masks.append(masks[0])
                
        print(f"\n총 {len(new_masks)}개 유효 마스크 생성 완료")
        
        if len(new_masks) > 0:
            self.mask = np.zeros_like(new_masks[0], dtype=bool)
            for m in new_masks:
                self.mask = np.logical_or(self.mask, m)
            
            self.label_connected_objects()
            self._save_state()
            self.show_result()
        else:
            print("유효한 마스크를 찾지 못했습니다.")

    def find_similar_objects(self):
        if self.mask is None:
            print("먼저 객체를 선택해주세요")
            return
            
        print("유사 객체 탐색을 위해 전체 스캔 중...")
        
        candidate_points = self._detect_candidate_points()
        generated_masks = []
        
        for point in candidate_points:
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            if scores[0] > 0.80:
                generated_masks.append(masks[0])
        
        reference_features = self._extract_features(self.mask)
        if reference_features is None:
            return
            
        print(f"유사 객체 선별 중 (임계값: {self.similarity_threshold})...")
        self.selected_masks = [self.mask]
        
        count = 0
        for mask_candidate in generated_masks:
            features = self._extract_features(mask_candidate)
            if features is not None:
                similarity = self._calculate_similarity(reference_features, features)
                if similarity >= self.similarity_threshold:
                    overlap = np.logical_and(self.mask, mask_candidate)
                    if np.sum(overlap) == 0:
                        self.selected_masks.append(mask_candidate)
                        count += 1
        
        print(f"총 {count}개 유사 객체 추가됨")
        
        for m in self.selected_masks:
            self.mask = np.logical_or(self.mask, m)
            
        self.label_connected_objects()
        self._save_state()
        self.show_result()

    def mouse_callback(self, event, x, y, flags, param):
        # Ctrl 키가 눌렸는지 확인
        ctrl_pressed = flags & cv2.EVENT_FLAG_CTRLKEY
        
        # Ctrl + 스크롤: 줌
        if ctrl_pressed and event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # 스크롤 업
                self.zoom_level = min(self.zoom_max, self.zoom_level + self.zoom_step)
            else:  # 스크롤 다운
                self.zoom_level = max(self.zoom_min, self.zoom_level - self.zoom_step)
            print(f"줌 레벨: {self.zoom_level:.2f}x")
            self.show_result()
            return
        
        # 마우스 휠: 팬 (Ctrl 없이)
        if not ctrl_pressed and event == cv2.EVENT_MOUSEWHEEL:
            pan_speed = 30
            if flags > 0:  # 스크롤 업
                self.pan_y -= pan_speed
            else:  # 스크롤 다운
                self.pan_y += pan_speed
            self._clamp_pan()
            self.show_result()
            return
        
        # 마우스 중간 버튼 드래그: 팬
        if event == cv2.EVENT_MBUTTONDOWN:
            self.is_panning = True
            self.pan_start_x = x
            self.pan_start_y = y
            
        elif event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False
            
        elif event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            dx = x - self.pan_start_x
            dy = y - self.pan_start_y
            self.pan_x -= dx / self.zoom_level
            self.pan_y -= dy / self.zoom_level
            
            self._clamp_pan()
            
            self.pan_start_x = x
            self.pan_start_y = y
            self.show_result()
            return
        
        # 원본 좌표 변환
        orig_x, orig_y = self._scale_point(x, y)
        
        # 이미지 경계 체크
        h, w = self.image_original.shape[:2]
        if orig_x < 0 or orig_x >= w or orig_y < 0 or orig_y >= h:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([orig_x, orig_y])
            self.labels.append(1)
            print(f"✓ 포함 점 추가: ({orig_x}, {orig_y})")
            self.update_mask()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([orig_x, orig_y])
            self.labels.append(0)
            print(f"✗ 제외 점 추가: ({orig_x}, {orig_y}) - 이 영역은 마스크에서 제외됩니다")
            self.update_mask()
    
    def update_mask(self):
        if len(self.points) == 0:
            return
        
        input_points = np.array(self.points)
        input_labels = np.array(self.labels)
        
        # negative point가 있을 때는 단일 마스크 출력으로 더 정확함
        has_negative = 0 in input_labels
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=not has_negative  # negative point 있으면 단일 마스크
        )
        
        # 가장 높은 점수의 마스크 선택
        best_idx = np.argmax(scores)   
        self.mask = masks[best_idx]
        self.selected_masks = []
        
        # 디버그 정보
        if has_negative:
            print(f"  > 제외 점 적용: 마스크 점수 {scores[best_idx]:.3f}")
        
        self.label_connected_objects()
        self._save_state()
        self.show_result()
    
    def show_result(self):
        # 원본 이미지 기반으로 표시할 영역 추출
        x1, y1, x2, y2 = self._get_display_region()
        
        # 표시할 영역 추출
        display_original = self.image_original[y1:y2, x1:x2].copy()
        
        if self.mask is not None:
            # 마스크도 같은 영역 추출
            mask_region = self.mask[y1:y2, x1:x2]
            
            # 마스크 표시 (초록색)
            color_mask = np.zeros_like(display_original)
            color_mask[mask_region] = [0, 255, 0]
            display_original = cv2.addWeighted(display_original, 0.7, color_mask, 0.3, 0)
            
            # 객체 번호 표시
            if self.labeled_objects is not None and self.object_stats is not None:
                for stat in self.object_stats:
                    cx, cy = stat['centroid']
                    obj_id = stat['id']
                    
                    # 현재 표시 영역 내에 있는 경우만 표시
                    if x1 <= cx < x2 and y1 <= cy < y2:
                        display_cx = int(cx - x1)
                        display_cy = int(cy - y1)
                        cv2.putText(display_original, f"#{obj_id}", 
                                  (display_cx, display_cy),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 클릭한 점 표시 (더 크고 명확하게)
        for (x, y), label in zip(self.points, self.labels):
            if x1 <= x < x2 and y1 <= y < y2:
                display_x = int(x - x1)
                display_y = int(y - y1)
                
                if label == 1:  # 포함 점 (초록색)
                    color = (0, 255, 0)
                    # 초록 원 + 하얀 테두리
                    cv2.circle(display_original, (display_x, display_y), 8, (255, 255, 255), -1)
                    cv2.circle(display_original, (display_x, display_y), 6, color, -1)
                    cv2.circle(display_original, (display_x, display_y), 8, (0, 0, 0), 2)
                else:  # 제외 점 (빨간색)
                    color = (0, 0, 255)
                    # 빨간 X 표시
                    cv2.circle(display_original, (display_x, display_y), 8, (255, 255, 255), -1)
                    cv2.circle(display_original, (display_x, display_y), 6, color, -1)
                    # X 그리기
                    cv2.line(display_original, 
                            (display_x - 5, display_y - 5), 
                            (display_x + 5, display_y + 5), 
                            (255, 255, 255), 2)
                    cv2.line(display_original, 
                            (display_x + 5, display_y - 5), 
                            (display_x - 5, display_y + 5), 
                            (255, 255, 255), 2)
        
        # 줌 적용
        if self.zoom_level != 1.0:
            h, w = display_original.shape[:2]
            new_w = int(w * self.zoom_level)
            new_h = int(h * self.zoom_level)
            display_original = cv2.resize(display_original, (new_w, new_h), 
                                        interpolation=cv2.INTER_LINEAR if self.zoom_level > 1 else cv2.INTER_AREA)
        
        # 윈도우 크기에 맞게 조정 (필요시)
        h, w = display_original.shape[:2]
        if h > self.window_height or w > self.window_width:
            display_original = cv2.resize(display_original, (self.window_width, self.window_height),
                                         interpolation=cv2.INTER_AREA)
        
        # 단축키 가이드 플로팅 박스 (하단 왼쪽)
        guide_lines = [
            "L-Click: Select | R-Click: Exclude",
            "Ctrl+Wheel: Zoom | Wheel: Pan | I,J,K,L: Move",
            "a: Auto | f: Find Similar | c: Clear",
            "z/y: Undo/Redo | s: Save | r: Reset | q: Quit"
        ]
        
        # 박스 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        padding = 8
        line_height = 18
        
        # 텍스트 크기 계산
        max_width = 0
        for line in guide_lines:
            (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_w)
        
        box_width = max_width + padding * 2
        box_height = len(guide_lines) * line_height + padding * 2
        
        # 박스 위치 (하단 왼쪽)
        img_h, img_w = display_original.shape[:2]
        box_x = 10
        box_y = img_h - box_height - 10
        
        # 반투명 배경 박스
        overlay = display_original.copy()
        cv2.rectangle(overlay, 
                     (box_x, box_y), 
                     (box_x + box_width, box_y + box_height),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, display_original, 0.3, 0, display_original)
        
        # 테두리
        cv2.rectangle(display_original,
                     (box_x, box_y),
                     (box_x + box_width, box_y + box_height),
                     (100, 100, 100), 1)
        
        # 텍스트 그리기
        for i, line in enumerate(guide_lines):
            text_x = box_x + padding
            text_y = box_y + padding + (i + 1) * line_height - 5
            cv2.putText(display_original, line,
                       (text_x, text_y),
                       font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
        
        # 줌 레벨 정보 (상단 왼쪽, 더 작게)
        info_text = f"Zoom: {self.zoom_level:.2f}x | Saved: {len(self.accumulated_saves)}"
        cv2.putText(display_original, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(display_original, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imshow('SAM Segmentation', display_original)
    
    def save_results(self, base_path='outputs'):
        """저장 기능:
        1. 객체 ID
        2. 중심 좌표 (Bounding Box 중심 우선, 객체 밖이면 무게중심 사용)
        3. 객체를 구성하는 모든 픽셀 좌표
        
        ===== 누적 저장 기능 =====
        S키를 누를 때마다:
        - 현재 마스크의 모든 객체를 accumulated_saves에 추가
        - data.json을 누적된 전체 리스트로 덮어쓰기
        """
        if self.mask is None:
            print("저장할 마스크가 없습니다")
            return
        
        os.makedirs(base_path, exist_ok=True)
        
        # --- 1. JSON 데이터 생성 로직 시작 ---
        print("객체 데이터 추출 및 JSON 변환 중...")
        
        # 현재 라벨링된 객체 정보가 없으면 다시 계산
        if self.labeled_objects is None:
            self.label_connected_objects()
        
        # 현재 마스크의 객체들을 추출
        current_objects = []
        
        # 배경(0)을 제외한 모든 객체 ID 순회
        unique_ids = np.unique(self.labeled_objects)
        unique_ids = unique_ids[unique_ids != 0]
        
        for obj_id in unique_ids:
            # 해당 객체의 마스크만 추출 (Boolean Mask)
            obj_mask = (self.labeled_objects == obj_id)
            
            # 1. 모든 픽셀 좌표 추출 (Row=y, Col=x)
            ys, xs = np.where(obj_mask)
            
            # 픽셀 좌표 리스트 변환 [{'x': 10, 'y': 20}, ...]
            # int() 변환 필수 (numpy int는 json 직렬화 불가)
            pixel_coords = [{"x": int(x), "y": int(y)} for x, y in zip(xs, ys)]
            
            # 2. 중심 좌표 계산 로직
            # (A) Bounding Box 중심 계산
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            bbox_cx = int((x_min + x_max) / 2)
            bbox_cy = int((y_min + y_max) / 2)
            
            # (B) 무게 중심 (Center of Mass) 계산
            mass_cx = int(np.mean(xs))
            mass_cy = int(np.mean(ys))
            
            # (C) 결정 로직: BBox 중심이 실제 객체 마스크(True) 위에 있는가?
            # 주의: 배열 인덱스는 [y, x] 순서
            if obj_mask[bbox_cy, bbox_cx]:
                final_cx = bbox_cx
                final_cy = bbox_cy
                center_type = "bbox_center"
            else:
                final_cx = mass_cx
                final_cy = mass_cy
                center_type = "mass_centroid"

            # 데이터 구조체 생성
            obj_data = {
                "id": len(self.accumulated_saves) + len(current_objects),  # 전역 ID
                "center_x": final_cx,
                "center_y": final_cy,
                "center_type": center_type, 
                "pixels": pixel_coords
            }
            current_objects.append(obj_data)
        
        # 현재 객체들을 누적 리스트에 추가
        self.accumulated_saves.extend(current_objects)
        
        # JSON 파일 저장 (전체 누적 리스트)
        json_path = os.path.join(base_path, 'data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.accumulated_saves, f, indent=2)
        # --- JSON 데이터 생성 로직 끝 ---

        # 2. 마스크 저장
        np.save(os.path.join(base_path, 'mask.npy'), self.mask)
        
        # 3. 이미지 저장 (마스크 적용)
        masked_img = self.image_original.copy()
        masked_img[~self.mask] = 0
        self._imwrite_korean(os.path.join(base_path, 'result_masked.png'), masked_img)
        
        # 4. 오버레이 이미지
        overlay_img = self.image_original.copy()
        overlay_img[self.mask] = overlay_img[self.mask] * 0.7 + np.array([0, 255, 0]) * 0.3
        self._imwrite_korean(os.path.join(base_path, 'result_overlay.png'), overlay_img)
            
        print(f"✓ 저장 완료: {base_path}")
        print(f"  현재 세션: {len(current_objects)}개 객체")
        print(f"  누적 총계: {len(self.accumulated_saves)}개 객체")

    def get_accumulated_objects(self):
        """누적된 전체 객체 리스트 반환 (scanner_agent 연동용)"""
        return self.accumulated_saves

    def run(self):
        cv2.namedWindow('SAM Segmentation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SAM Segmentation', self.window_width, self.window_height)
        cv2.setMouseCallback('SAM Segmentation', self.mouse_callback)
        
        self.show_result()
        
        print("\n=== SAM Segmentation Tool ===")
        print("단축키는 화면 하단에 표시됩니다.")
        print("============================\n")
        
        while True:
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.points = []
                self.labels = []
                self.mask = None
                self.selected_masks = []
                self.show_result()
                print("초기화")
            elif key == ord('r'):
                self.zoom_level = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self.show_result()
                print("줌/팬 리셋")
            elif key == ord('f'):
                self.find_similar_objects()
            elif key == ord('a'):
                self.generate_all_masks()
            elif key == ord('z'):
                self.undo()
            elif key == ord('y'):
                self.redo()
            elif key == ord('s'):
                self.save_results()
            elif key == ord('+') or key == ord('='):
                self.zoom_level = min(self.zoom_max, self.zoom_level + self.zoom_step)
                print(f"줌 레벨: {self.zoom_level:.2f}x")
                self.show_result()
            elif key == ord('-') or key == ord('_'):
                self.zoom_level = max(self.zoom_min, self.zoom_level - self.zoom_step)
                print(f"줌 레벨: {self.zoom_level:.2f}x")
                self.show_result()
            # I,J,K,L로 팬 이동 (vim 스타일)
            elif key == ord('i'):  # 위로
                self.pan_y -= 30
                self._clamp_pan()
                self.show_result()
            elif key == ord('k'):  # 아래로
                self.pan_y += 30
                self._clamp_pan()
                self.show_result()
            elif key == ord('j'):  # 왼쪽으로
                self.pan_x -= 30
                self._clamp_pan()
                self.show_result()
            elif key == ord('l'):  # 오른쪽으로
                self.pan_x += 30
                self._clamp_pan()
                self.show_result()
                
        cv2.destroyAllWindows()
        
        # ===== 창 닫으면 누적된 객체 반환 =====
        return self.accumulated_saves