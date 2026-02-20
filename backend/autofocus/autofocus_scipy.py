"""
Auto-Focus Agent — SciPy Brent's Method + Gemini Vision Validation

파이프라인:
  1단계 (Coarse Scan)  : Z축 전체를 큰 스텝으로 빠르게 훑어 초점 점수 프로파일 생성
  2단계 (Validation)   : 피크 구간 이미지를 Gemini Flash에 보내 타겟 일치 여부 확인
  3단계 (Fine Focus)   : 검증된 구간에서 SciPy minimize_scalar로 정밀 초점 탐색

하드웨어:
  - TangoController : Z축 절대좌표 이동 (move_absolute(x=0, y=0, z))
  - StreamingTUCam  : 프레임 캡처
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


class AutoFocusAgent:
    """
    SciPy + Gemini 기반 오토포커스 에이전트

    Args:
        camera:  StreamingTUCam 인스턴스 (start_stream 완료 상태)
        stage:   TangoController 인스턴스 (connect 완료 상태)
        gemini_api_key: Gemini API 키 (None이면 검증 스킵 → 점수만으로 판단)
        gemini_model:   Gemini 모델명
    """

    def __init__(
        self,
        camera,
        stage,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
    ):
        self.camera = camera
        self.stage = stage
        self.history: List[Dict] = []
        self._home_x = 0.0
        self._home_y = 0.0

        # Gemini 설정
        self._gemini_client = None
        self._gemini_model_name = gemini_model
        if gemini_api_key:
            try:
                from google import genai

                self._gemini_client = genai.Client(api_key=gemini_api_key)
                logger.info("Gemini loaded: %s", gemini_model)
            except ImportError:
                logger.warning("google-genai not installed, Gemini validation disabled")

    # ================================================================
    # 하드웨어 제어
    # ================================================================

    def _move_z(self, z: float):
        """Z축 절대 이동 (X, Y는 run() 시작 시점의 위치 유지)"""
        self.stage.move_absolute(self._home_x, self._home_y, z)

    def _capture_frame(self) -> Optional[np.ndarray]:
        return self.camera.get_latest_frame()

    # ================================================================
    # 초점 점수
    # ================================================================

    def _calculate_tenengrad(self, image: np.ndarray) -> float:
        """Tenengrad 초점 점수 (중앙 50% ROI, Sobel gradient magnitude)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if gray.dtype == np.uint16:
            gray = (gray / 256).astype(np.uint8)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        h, w = gray.shape
        roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

        gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

        return float(np.mean(gx**2 + gy**2))

    def _capture_and_score(self) -> Tuple[Optional[np.ndarray], float]:
        """프레임 캡처 + 초점 점수"""
        frame = self._capture_frame()
        if frame is None:
            return None, 0.0
        return frame, self._calculate_tenengrad(frame)

    # ================================================================
    # Gemini 검증
    # ================================================================

    def _frame_to_jpeg_bytes(self, frame: np.ndarray) -> bytes:
        img = frame
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    def validate_with_gemini(self, image: np.ndarray, target_description: str) -> str:
        """
        Gemini Flash로 이미지 검증

        Returns:
            "ok"   — 타겟 일치
            "up"   — 불일치, Z를 올려서 재탐색
            "down" — 불일치, Z를 내려서 재탐색
        """
        if self._gemini_client is None:
            logger.warning("Gemini not configured, skipping validation")
            return "ok"

        from google.genai import types

        jpeg_bytes = self._frame_to_jpeg_bytes(image)

        prompt = (
            f"You are an expert microscope image analyzer.\n"
            f"Examine this microscope image and determine if it clearly shows: {target_description}\n\n"
            f"Rules:\n"
            f"- If the image clearly shows the described targets in focus → respond: ok\n"
            f"- If the image does NOT show the targets and appears to need a higher focal plane → respond: down\n"
            f"- If the image does NOT show the targets and appears to need a lower focal plane → respond: up\n"
            f"- If unsure about direction → respond: up\n\n"
            f"Respond with ONLY one word: ok, up, or down."
        )

        try:
            response = self._gemini_client.models.generate_content(
                model=self._gemini_model_name,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            result = response.text.strip().lower()

            if result not in ("ok", "up", "down"):
                logger.warning("Unexpected Gemini response: '%s', defaulting to 'up'", result)
                result = "up"

            logger.info("Gemini validation result: %s", result)
            return result
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return "ok"

    # ================================================================
    # 1단계: Coarse Scan
    # ================================================================

    def coarse_scan(
        self,
        z_min: float,
        z_max: float,
        step: float = 50.0,
        settle_time: float = 0.05,
    ) -> List[Dict]:
        """
        Z축 전체를 큰 스텝으로 빠르게 스캔하여 초점 점수 프로파일 생성

        Args:
            z_min/z_max: 탐색 범위 (um)
            step: 이동 스텝 (um)
            settle_time: 이동 후 안정화 대기 (초)
        """
        logger.info("=== Coarse Scan: Z=[%.1f, %.1f], step=%.1f ===", z_min, z_max, step)

        scan_data = []
        z_positions = np.arange(z_min, z_max + step / 2, step)

        for i, z in enumerate(z_positions):
            self._move_z(float(z))
            time.sleep(settle_time)

            frame, score = self._capture_and_score()

            entry = {"z": float(z), "score": score, "index": i}
            scan_data.append(entry)

            logger.info(
                "  [%d/%d] Z=%7.1f | Score=%10.2f",
                i + 1, len(z_positions), z, score,
            )

        return scan_data

    # ================================================================
    # 2단계: 피크 탐지 + Gemini 검증
    # ================================================================

    def find_candidate_peaks(
        self,
        scan_data: List[Dict],
        score_threshold_ratio: float = 0.5,
    ) -> List[Dict]:
        """
        스캔 데이터에서 피크(유망 구간) 추출

        Args:
            score_threshold_ratio: 최대 점수 대비 임계값 (0.5 = 상위 50% 이상)
        """
        scores = np.array([d["score"] for d in scan_data])

        if len(scores) < 3:
            best_idx = int(np.argmax(scores))
            return [scan_data[best_idx]]

        threshold = scores.max() * score_threshold_ratio
        peak_indices, _ = find_peaks(scores, height=threshold)

        if len(peak_indices) == 0:
            best_idx = int(np.argmax(scores))
            return [scan_data[best_idx]]

        peaks = [scan_data[i] for i in peak_indices]
        peaks.sort(key=lambda x: x["score"], reverse=True)

        logger.info("Found %d candidate peaks", len(peaks))
        return peaks

    def select_valid_region(
        self,
        peaks: List[Dict],
        target_description: str,
        search_margin: float = 100.0,
    ) -> Optional[Tuple[float, float]]:
        """
        피크들을 순회하며 Gemini 검증 → 첫 번째 "ok" 구간 반환

        Args:
            peaks: 후보 피크 (점수 내림차순)
            target_description: 타겟 설명 텍스트 (예: "세포, 세포벽")
            search_margin: 확정 구간 여유 (um)

        Returns:
            (z_min, z_max) 유효 구간 또는 None
        """
        remaining = list(peaks)
        i = 0
        while i < len(remaining):
            peak = remaining[i]
            z = peak["z"]
            logger.info("Validating peak Z=%.4f (score=%.2f) ...", z, peak["score"])

            self._move_z(z)
            time.sleep(0.1)

            frame = self._capture_frame()
            if frame is None:
                i += 1
                continue

            result = self.validate_with_gemini(frame, target_description)

            if result == "ok":
                region = (z - search_margin, z + search_margin)
                logger.info("Validated! Fine focus region: [%.4f, %.4f]", *region)
                return region

            # up/down 힌트로 남은 피크 재정렬
            rest = remaining[i + 1 :]
            if result == "up":
                rest.sort(key=lambda p: p["z"], reverse=True)   # Z 높은 쪽 우선
                logger.info("Gemini hint=up → remaining peaks sorted high-Z first")
            elif result == "down":
                rest.sort(key=lambda p: p["z"])                  # Z 낮은 쪽 우선
                logger.info("Gemini hint=down → remaining peaks sorted low-Z first")
            remaining[i + 1 :] = rest
            i += 1

        logger.warning("No valid region found")
        return None

    # ================================================================
    # 3단계: Fine Focus (SciPy)
    # ================================================================

    def _objective_function(self, z: float) -> float:
        """SciPy 목적 함수 (부호 반전: 최소화 → 최대화)"""
        z = float(z)
        self._move_z(z)
        time.sleep(0.05)

        _, score = self._capture_and_score()

        self.history.append({"z": z, "score": score, "phase": "fine"})
        logger.info("  Fine Z=%7.2f | Score=%10.4f", z, score)

        return -score

    def fine_focus(
        self,
        z_min: float,
        z_max: float,
        tolerance: float = 1.0,
        max_iter: int = 50,
    ) -> Dict[str, Any]:
        """
        검증된 구간 내에서 SciPy Brent's method로 정밀 초점 탐색

        Args:
            z_min/z_max: 탐색 구간 (um)
            tolerance: 수렴 허용 오차 (um)
            max_iter: 최대 반복 횟수
        """
        logger.info("=== Fine Focus: Z=[%.1f, %.1f], tol=%.1f ===", z_min, z_max, tolerance)

        res = minimize_scalar(
            self._objective_function,
            bounds=(z_min, z_max),
            method="bounded",
            options={"xatol": tolerance, "maxiter": max_iter},
        )

        best_z = float(res.x)
        best_score = float(-res.fun)

        # 최적 위치로 최종 이동
        self._move_z(best_z)

        logger.info("Fine focus done: Z=%.2f, Score=%.4f, Calls=%d", best_z, best_score, res.nfev)
        return {
            "optimal_z": best_z,
            "best_score": best_score,
            "iterations": int(res.nfev),
        }

    # ================================================================
    # 전체 파이프라인
    # ================================================================

    def run(
        self,
        z_range: Tuple[float, float] = (0.0, 1000.0),
        target_description: str = "세포, 세포벽",
        coarse_step: float = 50.0,
        fine_tolerance: float = 1.0,
        score_threshold_ratio: float = 0.5,
        search_margin: float = 100.0,
    ) -> Dict[str, Any]:
        """
        전체 오토포커스 파이프라인

        1. Coarse Scan  → Z 전체 훑기
        2. Peak Detection → 초점 점수 피크 추출
        3. Gemini Validation → 피크 이미지 검증 ("ok" / "up" / "down")
        4. Fine Focus   → 검증 구간에서 SciPy 정밀 탐색

        Args:
            z_range: Z축 탐색 범위 (um)
            target_description: Gemini에 보낼 타겟 설명
            coarse_step: Coarse scan 스텝 크기 (um)
            fine_tolerance: Fine focus 수렴 허용치 (um)
            score_threshold_ratio: 피크 탐지 임계 비율 (0~1)
            search_margin: Fine focus 구간 여유 (um)

        Returns:
            결과 딕셔너리 (status, optimal_z, best_score, history 등)
        """
        self.history = []
        z_min, z_max = z_range

        # run() 시작 시점의 X, Y 좌표 저장 → _move_z에서 재사용
        pos = self.stage.get_position()
        if pos is not None:
            self._home_x, self._home_y = pos[0], pos[1]
            logger.info("Home XY set: X=%.3f, Y=%.3f", self._home_x, self._home_y)

        logger.info("=" * 60)
        logger.info("AutoFocus Pipeline Started")
        logger.info("  Range : [%.1f, %.1f] um", z_min, z_max)
        logger.info("  Target: %s", target_description)
        logger.info("  Step  : %.1f um (coarse), %.1f um (fine tol)", coarse_step, fine_tolerance)
        logger.info("=" * 60)

        # ── 1. Coarse Scan ──
        scan_data = self.coarse_scan(z_min, z_max, step=coarse_step)
        for d in scan_data:
            self.history.append({**d, "phase": "coarse"})

        # ── 2. Peak Detection ──
        peaks = self.find_candidate_peaks(scan_data, score_threshold_ratio)
        if not peaks:
            logger.error("No peaks found in coarse scan")
            return {"status": "failed", "reason": "no_peaks", "history": self.history}

        # ── 3. Gemini Validation ──
        region = self.select_valid_region(peaks, target_description, search_margin)
        if region is None:
            logger.error("No valid region after Gemini validation")
            return {"status": "failed", "reason": "validation_failed", "history": self.history}

        # 범위 클리핑
        fine_z_min = max(region[0], z_min)
        fine_z_max = min(region[1], z_max)

        # ── 4. Fine Focus ──
        result = self.fine_focus(fine_z_min, fine_z_max, tolerance=fine_tolerance)

        logger.info("=" * 60)
        logger.info("AutoFocus Complete")
        logger.info("  Optimal Z : %.2f um", result["optimal_z"])
        logger.info("  Best Score: %.4f", result["best_score"])
        logger.info("  Evaluations: %d coarse + %d fine", len(scan_data), result["iterations"])
        logger.info("=" * 60)

        return {
            "status": "ok",
            "optimal_z": result["optimal_z"],
            "best_score": result["best_score"],
            "coarse_samples": len(scan_data),
            "fine_iterations": result["iterations"],
            "history": self.history,
        }


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    # 프로젝트 루트를 sys.path에 추가 (어디서 실행하든 import 가능하도록)
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    _DLL_PATH = str(Path(__file__).resolve().parent.parent / "util" / "stage_move" / "Tango_DLL.dll")

    # 프로젝트 루트의 .env 파일에서 API 키 로드 (소스코드에 절대 하드코딩 금지)
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")
    _GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    _GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    if _GEMINI_API_KEY:
        print(f"  Gemini API 키 로드 완료 (.env) | 모델: {_GEMINI_MODEL}")
    else:
        print("  GEMINI_API_KEY 없음 → Gemini 검증 스킵 (.env 파일에 키를 입력하세요)")

    # ── 1. 스테이지 연결 ──
    print("[1/3] Tango 스테이지 연결 중...")
    from backend.util.stage_move.stage_test import TangoController

    _stage = TangoController(_DLL_PATH)
    if not _stage.load_dll():
        sys.exit("DLL 로드 실패")
    if not _stage.create_session():
        sys.exit("세션 생성 실패")
    if not _stage.connect():
        sys.exit("스테이지 연결 실패")

    _pos = _stage.get_position()
    _current_z = _pos[2]
    print(f"  현재 위치: X={_pos[0]:.3f}, Y={_pos[1]:.3f}, Z={_current_z:.3f} mm")

    # ── 2. 카메라 연결 ──
    print("[2/3] 카메라 초기화 중...")
    from backend.autofocus.autofocus import StreamingTUCam

    _cam = StreamingTUCam(exposure_ms=10.0)
    _cam.start_stream()
    time.sleep(0.5)

    _frame = _cam.get_latest_frame()
    if _frame is None:
        _cam.close()
        _stage.disconnect()
        _stage.free_session()
        sys.exit("카메라 프레임 획득 실패")
    print(f"  프레임 획득 OK: shape={_frame.shape}, dtype={_frame.dtype}")

    # ── 3. AutoFocus 실행 (실시간 스트리밍 오버레이 포함) ──
    print("[3/3] AutoFocus 시작... (창 이름: 'AutoFocus Live')")

    from backend.autofocus.autofocus import show_live

    class _LiveAutoFocusAgent(AutoFocusAgent):
        """프레임 캡처마다 cv2 창에 실시간으로 띄워주는 서브클래스"""

        def _capture_and_score(self):
            frame, score = super()._capture_and_score()
            if frame is not None:
                phase = self.history[-1]["phase"] if self.history else "coarse"
                show_live(frame, score, direction=phase, speed_mode="", index=len(self.history))
            return frame, score

    _agent = _LiveAutoFocusAgent(
        camera=_cam,
        stage=_stage,
        gemini_api_key=_GEMINI_API_KEY,
        gemini_model=_GEMINI_MODEL,
    )

    # Z 범위: 현재 위치 ±0.5 mm, 스텝 0.05 mm (TangoController 단위 = mm)
    _Z_MIN = _current_z - 2.0
    _Z_MAX = _current_z + 2.0

    try:
        _result = _agent.run(
            z_range=(_Z_MIN, _Z_MAX),
            target_description="눈금, 눈금자",
            coarse_step=0.2,
            fine_tolerance=0.001,
            search_margin=0.3,   # coarse_step의 1.5배 수준 (단위: mm)
        )

        print("\n" + "=" * 60)
        print("AutoFocus 결과")
        print("=" * 60)
        print(f"Status     : {_result['status']}")
        if _result['status'] == 'ok':
            print(f"Optimal Z  : {_result['optimal_z']:.4f} mm")
            print(f"Best Score : {_result['best_score']:.2f}")
            print(f"Coarse 횟수: {_result['coarse_samples']}")
            print(f"Fine 반복  : {_result['fine_iterations']}")
        else:
            print(f"실패 이유  : {_result.get('reason')}")
        print("=" * 60)

        print("창을 닫으려면 아무 키나 누르세요...")
        cv2.waitKey(0)

    except KeyboardInterrupt:
        print("\n중단됨 (Ctrl+C)")
    finally:
        cv2.destroyAllWindows()
        _cam.close()
        _stage.disconnect()
        _stage.free_session()
        print("정리 완료")
