"""
서장원

SAM 결과를 기반으로 라만 스캐닝 최적 경로 생성 (PCA 기반 장축 스캔 최적화 - Robust Version)

전체 파이프라인:
1. SAM 인터랙티브 객체 선택
2. 픽셀 좌표 → Stage 좌표 변환
3. 경로 최적화 (Greedy + 2-opt) + PCA 기반 객체별 장축 스캔 생성
4. 시각화 생성
"""

import numpy as np
import cv2
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sklearn.decomposition import PCA

from backend.scan import sam

try:
    from backend.util.scanner_utils import CoordinateMapper, PathPlanner
    from backend.util.pattern_generator import generate_dense_grid, generate_circular_cluster
except ImportError:
    class CoordinateMapper:
        def __init__(self, mag_level):
            self.config = {"um_per_pixel": 0.5, "sign_x": 1, "sign_y": -1}
        def pixel_to_stage(self, px, py, curr_x, curr_y):
            return (curr_x + px * 0.5 * self.config["sign_x"], curr_y + py * 0.5 * self.config["sign_y"])

@dataclass
class ScanTarget:
    """스캔 작업 단위"""
    id: int
    centroid_stage: Tuple[float, float]
    scan_points_relative: List[Tuple[float, float]]
    obj_type: str
    pixel_coords: Tuple[float, float]
    entry_pixel: Tuple[float, float] = (0, 0)
    exit_pixel: Tuple[float, float] = (0, 0)
    entry_stage: Tuple[float, float] = (0.0, 0.0)
    exit_stage: Tuple[float, float] = (0.0, 0.0)
    pca_tips_stage: List[Tuple[float, float]] = None
    pca_angle: float = 0.0


class ScannerAgent:
    """
    SAM 기반 라만 스캐닝 경로 생성 에이전트
    """
    
    def __init__(self, mag_level: str = "20x", sam_checkpoint: str = "sam_vit_h_4b8939.pth"):
        self.sam_checkpoint = sam_checkpoint
        self.mapper = CoordinateMapper(mag_level)
        self.scan_step_um = 5.0

    def _precompute_endpoints(self, pixels: list) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], float]:
        if not pixels: return ((0, 0), (0, 0)), 0.0

        pts = np.array([[p['x'], p['y']] for p in pixels])
        if len(pts) < 2:
             return (tuple(pts[0]), tuple(pts[0])), 0.0

        pca = PCA(n_components=2)
        pca.fit(pts)
        
        angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        projected = pca.transform(pts)
        
        min_idx = np.argmin(projected[:, 0]) 
        max_idx = np.argmax(projected[:, 0]) 
        
        return (tuple(pts[min_idx]), tuple(pts[max_idx])), angle

    def generate_pca_snake_scan_path(self, pixels: list, step_px: int, cached_angle: float) -> list:
        if not pixels: return []
        pts = np.array([[p['x'], p['y']] for p in pixels])
        
        # PCA 각도로 회전
        angle = cached_angle
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_pts = pts @ rot_matrix.T
        
        y_min = rotated_pts[:, 1].min()
        y_max = rotated_pts[:, 1].max()

        internal_path = []
        # 테두리 여백을 고려하여 Y축 스텝 생성
        y_range = np.arange(y_min + step_px/2, y_max, step_px)
        
        for i, y in enumerate(y_range): 
            # 해당 Y축 라인 근처의 픽셀들 마스킹
            mask = np.abs(rotated_pts[:, 1] - y) < (step_px / 2 + 1)
            row_points = rotated_pts[mask]
            
            if len(row_points) == 0: continue
            
            # 해당 줄의 X축 범위 확인
            row_x_min, row_x_max = row_points[:, 0].min(), row_points[:, 0].max()
            
            # [수정된 부분] 양 끝점만 넣는 게 아니라, 사이를 step_px로 채움
            # np.arange는 마지막 값을 포함하지 않으므로 step_px/2를 더해줌
            x_range = np.arange(row_x_min, row_x_max + step_px/100, step_px)
            
            if len(x_range) == 0:
                # 혹시 범위가 너무 좁아서 점이 안 생기면, 중간값 하나라도 추가
                x_range = [ (row_x_min + row_x_max) / 2 ]

            # Snake 방향 처리 (짝수: 정방향, 홀수: 역방향)
            if i % 2 != 0:
                x_range = x_range[::-1] # 뒤집기
            
            # 경로에 추가
            for x in x_range:
                internal_path.append([x, y])

        if not internal_path: return []
        
        # 역회전 (원래 좌표계로 복귀)
        inv_rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        final_path = np.array(internal_path) @ inv_rot_matrix.T
        
        return [(int(round(p[0])), int(round(p[1]))) for p in final_path]

    def _convert_sam_to_stage_coords(self, optimized_targets: List[ScanTarget], objects_data: List[Dict], stage_pos: Dict[str, float]) -> List[ScanTarget]:
        print(f"\n[DEBUG] --- 내부 스캔 경로 생성 ---")
        final_targets = []
        um_per_px = self.mapper.config['um_per_pixel']
        step_px = max(1, int(self.scan_step_um / um_per_px))
        obj_map = {obj['id']: obj for obj in objects_data}

        for opt_target in optimized_targets:
            raw_obj = obj_map.get(opt_target.id)
            if not raw_obj: continue

            angle = opt_target.pca_angle

            # 1. 일단 경로 생성
            pixel_path = self.generate_pca_snake_scan_path(raw_obj['pixels'], step_px, angle)
            if not pixel_path:
                print(f"[DEBUG] Obj {raw_obj['id']}: 너무 작아서 경로 생성 실패 -> 중심점만 추가")
                pixel_path = [(raw_obj['center_x'], raw_obj['center_y'])]

            # 복잡한 변수 스와핑 대신, 실제 생성된 경로의 양 끝점을 기준으로 판단합니다.
            p_start = pixel_path[0]
            p_end = pixel_path[-1]
            
            # 현재 경로의 시작점과 끝점을 Stage 좌표로 변환
            s_start = self.mapper.pixel_to_stage(p_start[0], p_start[1], stage_pos['x'], stage_pos['y'])
            s_end = self.mapper.pixel_to_stage(p_end[0], p_end[1], stage_pos['x'], stage_pos['y'])
            
            # 2-Opt가 제안한 이상적인 진입점
            ideal_entry = opt_target.entry_stage
            
            dist_start_to_ideal = (s_start[0]-ideal_entry[0])**2 + (s_start[1]-ideal_entry[1])**2
            dist_end_to_ideal = (s_end[0]-ideal_entry[0])**2 + (s_end[1]-ideal_entry[1])**2
            
            # 현재 경로의 끝점(End)이 이상적인 진입점(Ideal)에 더 가깝다면, 경로를 뒤집어야 함

            is_reversed = False
            if dist_end_to_ideal < dist_start_to_ideal:
                pixel_path.reverse()
                # 뒤집었으므로 다시 리스트에서 추출
                actual_entry_px = pixel_path[0]
                actual_exit_px = pixel_path[-1]
                is_reversed = True
            else:
                actual_entry_px = pixel_path[0]
                actual_exit_px = pixel_path[-1]

            if is_reversed:
                print(f"[DEBUG] Obj {raw_obj['id']}: 내부 경로 역방향 정렬됨 (Global 경로와 일치시키기 위해)")

            # Stage 좌표도 픽셀 좌표를 기반으로 '다시' 계산 (정합성 보장)
            actual_entry_stage = self.mapper.pixel_to_stage(actual_entry_px[0], actual_entry_px[1], stage_pos['x'], stage_pos['y'])
            actual_exit_stage = self.mapper.pixel_to_stage(actual_exit_px[0], actual_exit_px[1], stage_pos['x'], stage_pos['y'])

            # 3. 상대 좌표 계산
            cx_px, cy_px = raw_obj['center_x'], raw_obj['center_y']
            rel_stage_points = []
            sx = self.mapper.config['sign_x']
            sy = self.mapper.config['sign_y']
            
            for px, py in pixel_path:
                dx_um = (px - cx_px) * um_per_px * sx
                dy_um = (py - cy_px) * um_per_px * sy
                rel_stage_points.append((dx_um, dy_um))

            stage_center = self.mapper.pixel_to_stage(cx_px, cy_px, stage_pos['x'], stage_pos['y'])
            
            new_target = ScanTarget(
                id=raw_obj['id'],
                centroid_stage=stage_center,
                scan_points_relative=rel_stage_points, # 이 리스트 순서는 이제 actual_entry_px와 무조건 일치함
                obj_type='cell',
                pixel_coords=(float(cx_px), float(cy_px)),
                entry_pixel=actual_entry_px, # 시각화 시 화살표가 가리킬 지점
                exit_pixel=actual_exit_px,
                entry_stage=actual_entry_stage,
                exit_stage=actual_exit_stage,
                pca_angle=angle
            )
            final_targets.append(new_target)
            
        return final_targets

    def run_full_pipeline(self, image_path: str, current_stage_pos: Dict[str, float], output_dir: str = './outputs') -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        segmenter = sam.SAMMultiSegmenter(checkpoint_path=self.sam_checkpoint, image_path=image_path)
        detected_objects = segmenter.run() 
        
        if not detected_objects:
            return None
        
        print("\nPHASE 2: Pre-calculation & Global Optimization")
        
        temp_targets = []
        for obj in detected_objects:
            (p_start, p_end), angle = self._precompute_endpoints(obj['pixels'])
            obj['pca_angle'] = angle

            s_center = self.mapper.pixel_to_stage(obj['center_x'], obj['center_y'], current_stage_pos['x'], current_stage_pos['y'])
            s_tip1 = self.mapper.pixel_to_stage(*p_start, current_stage_pos['x'], current_stage_pos['y'])
            s_tip2 = self.mapper.pixel_to_stage(*p_end, current_stage_pos['x'], current_stage_pos['y'])
            
            t = ScanTarget(
                id=obj['id'], 
                centroid_stage=s_center, 
                scan_points_relative=[], 
                obj_type='temp', 
                pixel_coords=(obj['center_x'], obj['center_y']),
                pca_tips_stage=[s_tip1, s_tip2],
                pca_angle=angle
            )
            temp_targets.append(t)
        
        start_tuple = (current_stage_pos['x'], current_stage_pos['y'])

        print("\n[INFO] Applying Spatial Sorting (X-major)...")
        temp_targets.sort(key=lambda t: t.centroid_stage[0])


        # ============================================================
        # 새로운 로직
        img_w = 0
        if detected_objects:
            img_w = max([obj['center_x'] for obj in detected_objects]) + 1000 # 여유분

        print("\n[INFO] Applying Serpentine (Snake) Sorting...")
        initial_path = self._sort_strip(temp_targets, img_w)
        optimized_targets, stats = self._two_opt(initial_path, 5000, start_tuple)

        # 새로운 로직 끝
        # ============================================================

        # greedy_path = self._greedy_nn(temp_targets, start_tuple)
        # optimized_targets, stats = self._two_opt(greedy_path, 5000, start_tuple)
        
        print("\nPHASE 3: Final Path Generation (Fixed)")
        scan_targets = self._convert_sam_to_stage_coords(
            optimized_targets,   
            detected_objects,    
            current_stage_pos
        )
        
        vis_path = self._visualize_scan_path(image_path, scan_targets, output_dir=output_dir)
        
        return {
            "total_objects": len(scan_targets),
            "scan_path": scan_targets,
            "visualization": vis_path,
            "total_distance": self._calc_total_distance(scan_targets, start_tuple),
            "stats": stats
        }

    # def _greedy_nn(self, targets, start_pos):
    #     print(f"\n[DEBUG] --- Greedy Nearest Neighbor 시작 ---")
    #     print(f"[DEBUG] 시작 위치: {start_pos}")

    #     unvisited = targets.copy()
    #     path = []
    #     curr_x, curr_y = start_pos
        
    #     step_count = 0
    #     while unvisited:
    #         best_idx = -1
    #         min_dist = float('inf')
    #         best_entry_idx = 0 
            
    #         for i, target in enumerate(unvisited):
    #             tips = target.pca_tips_stage 
    #             d1 = ((tips[0][0] - curr_x)**2 + (tips[0][1] - curr_y)**2) ** 0.5
    #             d2 = ((tips[1][0] - curr_x)**2 + (tips[1][1] - curr_y)**2) ** 0.5
                
    #             dist = min(d1, d2)
                
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 best_idx = i
    #                 best_entry_idx = 0 if d1 < d2 else 1
            
    #         next_target = unvisited.pop(best_idx)
    #         tips = next_target.pca_tips_stage
            
    #         print(f"[DEBUG] Step {step_count}: Target ID {next_target.id} 선택됨")
    #         print(f"       ㄴ 거리: {min_dist:.2f} um")
    #         print(f"       ㄴ 진입 팁: {'Tip 1' if best_entry_idx == 0 else 'Tip 2'}")

    #         if best_entry_idx == 0:
    #             next_target.entry_stage = tips[0]
    #             next_target.exit_stage = tips[1]
    #         else:
    #             next_target.entry_stage = tips[1]
    #             next_target.exit_stage = tips[0]
                
    #         path.append(next_target)
    #         curr_x, curr_y = next_target.exit_stage
    #         step_count += 1
            
    #     print(f"[DEBUG] --- Greedy 종료 (총 {len(path)}개 경로 생성) ---\n")
    #     return path
    
    def _sort_strip(self, targets: List[ScanTarget], image_width_px: int) -> List[ScanTarget]:

        if not targets: return []
        
        # 1. 화면을 몇 개의 세로 띠(Strip)로 나눌지 결정
        # 화면 너비의 1/5 ~ 1/10
        strip_width = 500 
        
        # 2. 버킷 생성
        strips = {}
        for t in targets:
            # X좌표를 기준으로 스트립 인덱스 결정
            s_idx = int(t.pixel_coords[0] // strip_width)
            if s_idx not in strips:
                strips[s_idx] = []
            strips[s_idx].append(t)
            
        # 3. 스트립 인덱스 순서대로 정렬 (왼쪽 -> 오른쪽)
        sorted_indices = sorted(strips.keys())
        
        final_path = []
        for i, s_idx in enumerate(sorted_indices):
            objs = strips[s_idx]
            
            # 4. 짝수 스트립은 내림차순(위->아래), 홀수 스트립은 오름차순(아래->위)
            # Y좌표(pixel_coords[1]) 기준 정렬
            if i % 2 == 0:
                objs.sort(key=lambda t: t.pixel_coords[1]) # Ascending (Top -> Bottom)
            else:
                objs.sort(key=lambda t: t.pixel_coords[1], reverse=True) # Descending (Bottom -> Top)
                
            final_path.extend(objs)
            
        print(f"[DEBUG] Snake Sort 완료: 총 {len(strips)}개 스트립으로 분할 정렬됨.")
        return final_path

    def _two_opt(self, path, max_iter, start_pos):
        print(f"[DEBUG] --- 2-Opt 최적화 시작 ---")
        best_path = path[:]
        best_dist = self._calc_path_length(best_path, start_pos)
        print(f"[DEBUG] 초기 경로 총 거리: {best_dist:.2f} um")
        
        swaps = 0
        for k in range(max_iter):
            improved = False
            for i in range(1, len(path) - 1):
                for j in range(i + 1, len(path)):
                    new_path = best_path[:i] + best_path[i:j][::-1] + best_path[j:]
                    new_dist = self._calc_path_length(new_path, start_pos)
                    if new_dist < best_dist:
                        print(f"[DEBUG] 2-Opt 개선됨! (Iter {k}, Swap {i}-{j})")
                        print(f"       ㄴ 거리 단축: {best_dist:.2f} -> {new_dist:.2f}")
                        best_path = new_path
                        best_dist = new_dist
                        improved = True
                        swaps += 1
                        break 
                if improved: break
            if not improved: 
                print(f"[DEBUG] 더 이상 개선 없음. 조기 종료 (Iter {k})")
                break
            
        print(f"[DEBUG] 2-Opt 종료. 총 {swaps}회 Swap 발생.")

        # Orientation 확정
        print(f"[DEBUG] --- 최종 방향(Orientation) 확정 ---")
        curr_x, curr_y = start_pos
        for idx, target in enumerate(best_path):
            tips = target.pca_tips_stage
            d1 = ((tips[0][0] - curr_x)**2 + (tips[0][1] - curr_y)**2)
            d2 = ((tips[1][0] - curr_x)**2 + (tips[1][1] - curr_y)**2)
            
            old_entry = target.entry_stage
            
            if d1 < d2:
                target.entry_stage = tips[0]
                target.exit_stage = tips[1]
                chosen = "Tip 1"
            else:
                target.entry_stage = tips[1]
                target.exit_stage = tips[0]
                chosen = "Tip 2"
            
            # 방향이 Greedy 단계와 달라졌는지 확인
            changed = " (방향 변경됨)" if target.entry_stage != old_entry else ""
            print(f"[DEBUG] Target {target.id}: {chosen} 진입 선택{changed}")
            
            curr_x, curr_y = target.exit_stage

        return best_path, {"optimized_distance": best_dist}
    
    def _calc_total_distance(self, path, start_pos):
        total = 0.0
        curr_x, curr_y = start_pos
        for target in path:
            tx, ty = target.entry_stage
            total += ((tx - curr_x)**2 + (ty - curr_y)**2) ** 0.5
            curr_x, curr_y = target.exit_stage
        return total
    
    def _calc_path_length(self, path, start_pos=None):
        if not path: return 0.0
        total = 0.0
        if start_pos:
            curr_x, curr_y = start_pos
        else:
            curr_x, curr_y = path[0].entry_stage

        for target in path:
            p1, p2 = target.pca_tips_stage
            dist_to_p1 = ((p1[0] - curr_x)**2 + (p1[1] - curr_y)**2)
            dist_to_p2 = ((p2[0] - curr_x)**2 + (p2[1] - curr_y)**2)
            
            if dist_to_p1 < dist_to_p2:
                total += dist_to_p1**0.5 
                curr_x, curr_y = p2 
            else:
                total += dist_to_p2**0.5
                curr_x, curr_y = p1
        return total
    
    def _visualize_scan_path(self, image_path, path, output_dir):
        image = cv2.imread(image_path)
        if image is None: return "Error"
        image = cv2.convertScaleAbs(image, alpha=0.9, beta=0) 
        
        um_per_px = self.mapper.config['um_per_pixel']
        sx, sy = self.mapper.config['sign_x'], self.mapper.config['sign_y']
        
        for i, target in enumerate(path):
            cx, cy = target.pixel_coords
            
            # 1. 스캔 경로 그리기
            pts = []
            for dx, dy in target.scan_points_relative:
                px = int(round(cx + dx / (um_per_px * sx)))
                py = int(round(cy + dy / (um_per_px * sy)))
                pts.append((px, py))
            
            for j in range(len(pts)):
                cv2.circle(image, pts[j], 1, (255, 255, 255), -1) 
                if j < len(pts) - 1:
                    cv2.line(image, pts[j], pts[j+1], (0, 0, 255), 1)
            
            # 2. 입구(Yellow) / 출구(Orange) 표시
            if pts:
                cv2.circle(image, pts[0], 4, (0, 255, 255), -1) 
                cv2.circle(image, pts[-1], 4, (0, 165, 255), -1)

            # 3. 이동 경로 화살표 (Cyan)
            # 이전 객체의 '실제 출구' -> 현재 객체의 '실제 입구'
            if i > 0:
                prev_target = path[i-1]
                start_pt = prev_target.exit_pixel
                end_pt = target.entry_pixel # == pts[0]
                
                p1 = (int(start_pt[0]), int(start_pt[1]))
                p2 = (int(end_pt[0]), int(end_pt[1]))
                
                cv2.arrowedLine(image, p1, p2, (255, 255, 0), 2, tipLength=0.1)

            cv2.putText(image, f"#{i}", (int(cx), int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = os.path.join(output_dir, 'refined_pca_path.png')
        cv2.imwrite(out_path, image)
        return out_path

# ============================================================
# 테스트용 메인
# ============================================================
if __name__ == "__main__":
    
    agent = ScannerAgent(
        mag_level="20x",
        sam_checkpoint='C:\\Users\\seoja\\Desktop\\RamanGPT\\RamanGPT\\sam_vit_h_4b8939.pth'
    )
    
    current_stage_position = {'x': 0.0, 'y': 0.0}
    
    result = agent.run_full_pipeline(
        image_path="C:\\Users\\seoja\\Desktop\\RamanGPT\\RamanGPT\\backend\\tests\\data\\path2big1.png",
        current_stage_pos=current_stage_position,
        output_dir="./outputs"
    )
    
    if result:
        print(f"\nFinal Distance: {result['total_distance']:.2f} um")