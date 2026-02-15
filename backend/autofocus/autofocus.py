"""
카메라 스트리밍 + 스테이지 이동 + 초점 점수 샘플링 모듈

1단계: start_move(), stop_move() 헬퍼 함수 분리
"""
import numpy as np
import cv2
import time
from ctypes import *
from typing import Optional, List, Dict
from pathlib import Path

from backend.util.cameras.TUCam import *


class StreamingTUCam:
    """TUCam 스트리밍 카메라 클래스"""
    
    def __init__(self, exposure_ms: float = 10.0):
        self.Path = './'
        self.TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)
        
        TUCAM_Api_Init(pointer(self.TUCAMINIT), 5000)
        if self.TUCAMINIT.uiCamCount == 0:
            raise RuntimeError("No camera found!")
        
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)
        TUCAM_Dev_Open(pointer(self.TUCAMOPEN))
        if self.TUCAMOPEN.hIdxTUCam == 0:
            raise RuntimeError("Camera open failed!")
        
        self.set_exposure(exposure_ms)
        self.is_streaming = False
        self.m_frame = TUCAM_FRAME()
        self.m_capmode = TUCAM_CAPTURE_MODES
        
    def set_exposure(self, ms: float):
        """노출 시간 설정 (밀리초)"""
        TUCAM_Capa_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value, 0)
        TUCAM_Prop_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDPROP.TUIDP_EXPOSURETM.value, c_double(ms), 0)

    def start_stream(self):
        """스트리밍 시작"""
        if self.is_streaming:
            return
        self.m_frame.pBuffer = 0
        self.m_frame.ucFormatGet = TUFRM_FORMATS.TUFRM_FMT_USUAl.value
        self.m_frame.uiRsdSize = 1
        TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(self.m_frame))
        TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, self.m_capmode.TUCCM_SEQUENCE.value)
        self.is_streaming = True

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """최신 프레임 획득"""
        if not self.is_streaming:
            return None
        try:
            TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(self.m_frame), 500)
        except:
            return None
        
        buf = create_string_buffer(self.m_frame.uiImgSize)
        pointer_data = c_void_p(self.m_frame.pBuffer + self.m_frame.usHeader)
        memmove(buf, pointer_data, self.m_frame.uiImgSize)
        
        dtype = np.uint8 if self.m_frame.ucElemBytes == 1 else np.uint16
        image_np = np.frombuffer(buf, dtype=dtype)
        
        if self.m_frame.ucChannels == 3:
            image_np = image_np.reshape((self.m_frame.usHeight, self.m_frame.usWidth, 3))
        else:
            image_np = image_np.reshape((self.m_frame.usHeight, self.m_frame.usWidth))
        
        return image_np

    def stop_stream(self):
        """스트리밍 중지"""
        if self.is_streaming:
            TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
            TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)
            TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)
            self.is_streaming = False

    def close(self):
        """카메라 연결 해제"""
        self.stop_stream()
        if self.TUCAMOPEN.hIdxTUCam != 0:
            TUCAM_Dev_Close(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Api_Uninit()


def show_live(image: np.ndarray, score: float, direction: str = "", speed_mode: str = "", index: int = 0):
    """실시간 카메라 스트리밍 + 초점 점수 오버레이"""
    if image is None:
        return
    
    disp = image.copy()
    
    if disp.dtype == np.uint16:
        disp = (disp / 256).astype(np.uint8)
    
    if len(disp.shape) == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    
    h, w = disp.shape[:2]
    if w > 1280:
        scale = 1280 / w
        disp = cv2.resize(disp, (1280, int(h * scale)))
        h, w = disp.shape[:2]
    
    # 반투명 배경 박스
    overlay = disp.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    disp = cv2.addWeighted(overlay, 0.6, disp, 0.4, 0)
    
    # 텍스트 오버레이
    cv2.putText(disp, f"Focus Score: {score:.1f}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    if direction and speed_mode:
        cv2.putText(disp, f"Direction: {direction.upper()} | Speed: {speed_mode}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(disp, f"Sample: {index}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Autofocus Live", disp)
    cv2.waitKey(1)


# =============================================================================
# 1단계: 헬퍼 함수 분리
# =============================================================================

def start_move(stage, direction: str, speed_mode: str = 'normal'):
    """
    스테이지 이동 시작 (버튼 누르기)
    
    Args:
        stage: GamepadController 인스턴스
        direction: 'up' 또는 'down'
        speed_mode: 'slow', 'normal', 'fast'
    """
    y_value = -32768 if direction == 'up' else 32767
    
    stage.press_button('R1', auto_update=False)
    if speed_mode == 'slow':
        stage.press_button('L1', auto_update=False)
    elif speed_mode == 'fast':
        stage.set_trigger('L2', value=255, auto_update=False)
    stage.set_stick('RIGHT', x=0, y=y_value, auto_update=False)
    stage.gamepad.update()


def stop_move(stage, speed_mode: str = 'normal'):
    """
    스테이지 이동 정지 (버튼 떼기)
    
    Args:
        stage: GamepadController 인스턴스
        speed_mode: 'slow', 'normal', 'fast'
    """
    stage.set_stick('RIGHT', x=0, y=0, auto_update=False)
    if speed_mode == 'slow':
        stage.release_button('L1', auto_update=False)
    elif speed_mode == 'fast':
        stage.set_trigger('L2', value=0, auto_update=False)
    stage.release_button('R1', auto_update=False)
    stage.gamepad.update()
    
    time.sleep(0.2)  # 안정화


# =============================================================================

def compute_focus_score(image: np.ndarray) -> float:
    """
    Tenengrad 초점 점수 계산 (Sobel gradient magnitude)
    """
    if image is None:
        return 0.0
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if gray.dtype == np.uint16:
        gray = (gray / 256).astype(np.uint8)
    
    h, w = gray.shape
    roi = gray[h//4:3*h//4, w//4:3*w//4]
    
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    score = np.mean(gx**2 + gy**2)
    
    return score


def sample_while_moving(
    cam: StreamingTUCam,
    stage,
    direction: str,
    duration: float = 2.0,
    interval: float = 0.1,
    speed_mode: str = 'normal',
    save_dir: str = "./testscreenshot"
) -> List[Dict]:
    """
    스테이지를 이동하면서 동시에 샘플링
    
    핵심: 버튼을 누른 채로 유지하면서 interval마다 샘플링
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    num_samples = int(duration / interval)
    
    print(f"\n[{direction.upper()}] {duration}s @ {speed_mode} ({num_samples} samples)")
    
    # === 이동 시작 (헬퍼 함수 사용) ===
    start_move(stage, direction, speed_mode)
    
    start_time = time.time()
    
    try:
        for i in range(num_samples):
            # 타이밍 맞추기
            target_time = start_time + (i * interval)
            wait_time = target_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
            
            # 프레임 획득
            frame = cam.get_latest_frame()
            if frame is None:
                print(f"  [{i+1:02d}/{num_samples}] Frame failed")
                continue
            
            score = compute_focus_score(frame)
            elapsed = time.time() - start_time
            
            # 저장
            save_img = frame.copy()
            if save_img.dtype == np.uint16:
                save_img = (save_img / 256).astype(np.uint8)
            
            filename = f"{direction}_{speed_mode}_{i:03d}_score_{score:.1f}.png"
            filepath = save_path / filename
            cv2.imwrite(str(filepath), save_img)
            
            results.append({
                'index': i,
                'time': elapsed,
                'direction': direction,
                'speed_mode': speed_mode,
                'score': score,
                'filename': filename
            })
            
            # 실시간 디스플레이
            show_live(frame, score, direction, speed_mode, i + 1)
            
            print(f"  [{i+1:02d}/{num_samples}] t={elapsed:.2f}s | Score: {score:>10.2f}")
        
        # duration 끝까지 이동 유지
        remaining = duration - (time.time() - start_time)
        if remaining > 0:
            time.sleep(remaining)
            
    finally:
        # === 이동 정지 (헬퍼 함수 사용) ===
        stop_move(stage, speed_mode)
    
    return results


def hill_climb_autofocus(
    cam: StreamingTUCam,
    stage,
    speed_mode: str = 'normal',
    interval: float = 0.1,
    patience_limit: int = 3,
    max_direction_changes: int = 2,
    max_samples: int = 100,
    save_dir: str = "./testscreenshot"
) -> Dict:
    """
    Hill Climbing 오토포커스
    
    버튼을 누른 채로 이동하면서 interval마다 샘플링.
    점수가 patience_limit번 연속 하락하면 방향 전환.
    방향 전환이 max_direction_changes번 발생하면 종료.
    
    Args:
        cam: StreamingTUCam 인스턴스
        stage: GamepadController 인스턴스
        speed_mode: 속도 모드 ('slow', 'normal', 'fast')
        interval: 샘플링 간격 (초)
        patience_limit: 연속 하락 허용 횟수
        max_direction_changes: 최대 방향 전환 횟수
        max_samples: 최대 샘플 수 (안전장치)
        save_dir: 스크린샷 저장 디렉토리
        
    Returns:
        Dict: 결과 정보
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 기존 파일 정리
    for f in save_path.glob("hc_*.png"):
        f.unlink()
    
    # 상태 초기화
    direction = 'up'  # 초기 방향
    prev_score = None
    best_score = 0
    best_filename = None
    patience = 0
    direction_changes = 0
    sample_count = 0
    results = []
    
    print("\n" + "=" * 60)
    print("HILL CLIMBING AUTOFOCUS")
    print("=" * 60)
    print(f"Speed: {speed_mode} | Interval: {interval}s")
    print(f"Patience: {patience_limit} | Max direction changes: {max_direction_changes}")
    print("=" * 60)
    
    # 이동 시작
    print(f"\n[START] Direction: {direction.upper()}")
    start_move(stage, direction, speed_mode)
    start_time = time.time()
    
    try:
        while direction_changes < max_direction_changes and sample_count < max_samples:
            # 타이밍 맞추기
            target_time = start_time + (sample_count * interval)
            wait_time = target_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
            
            # 프레임 획득
            frame = cam.get_latest_frame()
            if frame is None:
                print(f"  [{sample_count+1}] Frame failed")
                sample_count += 1
                continue
            
            score = compute_focus_score(frame)
            elapsed = time.time() - start_time
            
            # 이미지 저장
            save_img = frame.copy()
            if save_img.dtype == np.uint16:
                save_img = (save_img / 256).astype(np.uint8)
            
            filename = f"hc_{sample_count:03d}_{direction}_{score:.1f}.png"
            cv2.imwrite(str(save_path / filename), save_img)
            
            # 최고 점수 갱신
            if score > best_score:
                best_score = score
                best_filename = filename
            
            results.append({
                'index': sample_count,
                'time': elapsed,
                'direction': direction,
                'score': score,
                'filename': filename
            })
            
            # 실시간 디스플레이
            show_live(frame, score, direction, speed_mode, sample_count + 1)
            
            # === 점수 비교 로직 ===
            if prev_score is None:
                # 첫 번째 점수
                prev_score = score
                print(f"  [{sample_count+1:3d}] Score: {score:>10.2f} (initial)")
            elif score > prev_score:
                # 점수 상승 → 좋은 방향
                prev_score = score
                patience = 0
                print(f"  [{sample_count+1:3d}] Score: {score:>10.2f} ↑ (patience: {patience})")
            else:
                # 점수 하락 또는 유지
                patience += 1
                print(f"  [{sample_count+1:3d}] Score: {score:>10.2f} ↓ (patience: {patience}/{patience_limit})")
                
                if patience >= patience_limit:
                    # 방향 전환
                    stop_move(stage, speed_mode)
                    
                    direction = 'down' if direction == 'up' else 'up'
                    direction_changes += 1
                    
                    print(f"\n[DIRECTION CHANGE #{direction_changes}] → {direction.upper()}")
                    
                    if direction_changes < max_direction_changes:
                        start_move(stage, direction, speed_mode)
                        start_time = time.time()  # 타이밍 리셋
                        sample_count = -1  # 다음 루프에서 0이 됨
                    
                    patience = 0
                    prev_score = score
            
            sample_count += 1
    
    finally:
        stop_move(stage, speed_mode)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Direction changes: {direction_changes}")
    print(f"Best score: {best_score:.2f}")
    print(f"Best frame: {best_filename}")
    print("=" * 60)
    
    return {
        'best_score': best_score,
        'best_filename': best_filename,
        'total_samples': len(results),
        'direction_changes': direction_changes,
        'results': results
    }


def run_full_test(cam: StreamingTUCam, stage, save_dir: str = "./testscreenshot"):
    """
    모든 속도 모드에서 위/아래 이동하며 샘플링 테스트
    
    테스트 순서: slow/normal/fast × up/down = 6회
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 기존 파일 삭제
    for f in save_path.glob("*.png"):
        f.unlink()
    
    all_results = {}
    speed_modes = ['slow', 'normal', 'fast']
    directions = ['up', 'down']
    
    print("\n" + "=" * 60)
    print("STAGE MOVEMENT + FOCUS SAMPLING TEST")
    print("=" * 60)
    print(f"Duration: 2s per direction")
    print(f"Interval: 0.1s (20 samples)")
    print(f"Speed modes: {speed_modes}")
    print(f"Save dir: {save_path.absolute()}")
    print("=" * 60)
    
    for speed_mode in speed_modes:
        print(f"\n{'='*60}")
        print(f"SPEED MODE: {speed_mode.upper()}")
        print(f"{'='*60}")
        
        for direction in directions:
            key = f"{speed_mode}_{direction}"
            
            results = sample_while_moving(
                cam=cam,
                stage=stage,
                direction=direction,
                duration=2.0,
                interval=0.1,
                speed_mode=speed_mode,
                save_dir=save_dir
            )
            
            all_results[key] = results
            time.sleep(0.3)
        
        time.sleep(0.5)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for key, results in all_results.items():
        if results:
            scores = [r['score'] for r in results]
            best = max(results, key=lambda x: x['score'])
            print(f"{key:15s} | n={len(results):2d} | "
                  f"Score: {min(scores):>8.1f} ~ {max(scores):>8.1f} | "
                  f"Best: {best['filename']}")
    
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    from windows_receiver import GamepadController
    
    print("\n" + "=" * 60)
    print("HILL CLIMBING AUTOFOCUS TEST")
    print("=" * 60)
    
    # 1. 카메라 초기화
    print("\n[1/3] Initializing camera...")
    cam = StreamingTUCam(exposure_ms=10.0)
    cam.start_stream()
    print("Camera ready.")
    
    # 2. 스테이지 초기화
    print("\n[2/3] Initializing stage controller...")
    stage = GamepadController()
    print("Stage ready.")
    
    # 3. MantaRay 인식 대기
    print("\n[3/3] Waiting for MantaRay to recognize gamepad...")
    time.sleep(2)
    
    try:
        # Hill Climbing 오토포커스 실행
        result = hill_climb_autofocus(
            cam=cam,
            stage=stage,
            speed_mode='normal',      # 속도 모드
            interval=0.1,             # 샘플링 간격
            patience_limit=3,         # 연속 하락 허용 횟수
            max_direction_changes=2,  # 최대 방향 전환 횟수
            max_samples=100,          # 안전장치
            save_dir="./testscreenshot"
        )
        
        print("\nAutofocus complete! Check ./testscreenshot for saved images.")
        
    except KeyboardInterrupt:
        print("\nAutofocus interrupted.")
    finally:
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        cam.close()
        stage.reset()
        print("Done.")