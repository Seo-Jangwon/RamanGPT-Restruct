"""
Andor CCD 싱글톤 매니저

startup : connect → cooler ON → set temp -40°C
shutdown : cooler OFF → wait temp >= -20°C (max 1200s) → ShutDown

HW_andor_camera-master 폴더명에 하이픈이 있어 직접 import 불가.
sys.path 에 해당 디렉토리를 추가한 뒤 importlib 로 로드.
"""

import asyncio
import importlib.util
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# HW_andor_camera-master 절대 경로 (프로젝트 루트 기준)
_ANDOR_DIR = Path(__file__).resolve().parent.parent.parent / "HW_andor_camera-master"

# ── 글로벌 인스턴스 ──
_ccd = None
_ccd_lock = threading.Lock()
_ccd_init_in_progress = False          # TOCTOU 방지: 초기화 중 플래그
ccd_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ccd")


# ── 내부 유틸 ──

def _ensure_andor_in_path() -> None:
    """andor_ccd_consts 등 flat import가 동작하도록 디렉토리를 sys.path 앞에 추가."""
    andor_str = str(_ANDOR_DIR)
    if andor_str not in sys.path:
        sys.path.insert(0, andor_str)


def _load_andor_interface():
    """
    andor_ccd_interface 모듈을 파일 경로 기반으로 로드.
    모듈의 __file__ 이 HW_andor_camera-master/ 를 가리키므로
    DLL 자동 탐색(os.path.dirname(__file__))도 정상 동작.
    """
    _ensure_andor_in_path()
    module_path = str(_ANDOR_DIR / "andor_ccd_interface.py")
    spec = importlib.util.spec_from_file_location("andor_ccd_interface", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Getter ──

def get_ccd():
    return _ccd


# ── Connect / Disconnect ──

async def connect_ccd(target_temp: int = -40) -> None:
    """
    CCD 연결 → 쿨러 ON → 온도 목표 설정.
    온도 안정화는 비동기로 진행되므로 즉시 반환.
    """
    global _ccd, _ccd_init_in_progress

    with _ccd_lock:
        if _ccd is not None or _ccd_init_in_progress:
            raise RuntimeError("CCD already connected or connecting")
        _ccd_init_in_progress = True

    try:
        loop = asyncio.get_running_loop()

        def _init() -> object:
            mod = _load_andor_interface()
            ccd = mod.AndorCCD(initialize_to_defaults=False)
            ccd.set_cooler(True)
            ccd.set_temperature(target_temp)
            logger.info("CCD initialized, cooler ON, target temp=%d°C", target_temp)
            return ccd

        ccd = await loop.run_in_executor(ccd_executor, _init)

        with _ccd_lock:
            _ccd = ccd
            _ccd_init_in_progress = False
        logger.info("CCD connected")

    except Exception:
        with _ccd_lock:
            _ccd_init_in_progress = False
        raise


async def disconnect_ccd(warmup_temp: int = -20, max_wait_sec: int = 1200) -> None:
    """
    CCD 안전 종료:
      1) 쿨러 OFF  — TEC 냉각 중단, 자연 승온 시작
      2) 온도가 warmup_temp(-20°C) 이상 오를 때까지 대기 (최대 max_wait_sec초)
      3) ShutDown

    Andor 권장: ShutDown 전 검출기를 -20°C 이상으로 승온.
    warmup_temp=-20°C, max_wait_sec=1200(20분) 이 기본값.
    """
    global _ccd
    with _ccd_lock:
        ccd = _ccd
        _ccd = None         # 즉시 None으로 → 이후 요청은 연결 안 됨으로 처리

    if ccd is None:
        return

    loop = asyncio.get_running_loop()

    def _warmup_and_close() -> None:
        # 1) 쿨러 OFF — 능동 냉각 중단, 자연 승온 시작
        try:
            ccd.set_cooler_off()
            logger.info("CCD cooler OFF")
        except Exception as e:
            logger.warning("CCD set_cooler_off error: %s", e)

        # 2) 온도 상승 대기
        logger.info(
            "CCD warmup: waiting for temp >= %d°C (max %ds, poll 10s)",
            warmup_temp, max_wait_sec,
        )
        deadline = time.monotonic() + max_wait_sec
        while time.monotonic() < deadline:
            try:
                temp = ccd.get_temperature()
                logger.info("CCD warmup temp=%d°C (target >= %d°C)", temp, warmup_temp)
                if temp >= warmup_temp:
                    logger.info("CCD warmup target reached (%d°C)", temp)
                    break
            except Exception as e:
                logger.warning("CCD get_temperature error during warmup: %s", e)
                break
            time.sleep(10)
        else:
            logger.warning(
                "CCD warmup timeout after %ds — proceeding with ShutDown anyway",
                max_wait_sec,
            )

        # 3) ShutDown
        try:
            ccd.close()
            logger.info("CCD ShutDown complete")
        except Exception as e:
            logger.warning("CCD close error: %s", e)

    await loop.run_in_executor(ccd_executor, _warmup_and_close)


def shutdown_ccd_sync(warmup_temp: int = -20) -> None:
    """
    동기 강제 종료 — lifespan 외 비상 상황에서만 사용.
    쿨러 OFF 후 최대 30초 대기 뒤 ShutDown.
    """
    global _ccd
    with _ccd_lock:
        ccd = _ccd
        _ccd = None

    if ccd is None:
        return

    # 쿨러 OFF
    try:
        ccd.set_cooler_off()
        logger.info("CCD cooler OFF (sync shutdown)")
    except Exception as e:
        logger.warning("CCD cooler_off error: %s", e)

    # 짧게 온도 확인 (비상 종료이므로 최대 30초)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            temp = ccd.get_temperature()
            if temp >= warmup_temp:
                break
        except Exception:
            break
        time.sleep(5)

    try:
        ccd.close()
    except Exception as e:
        logger.warning("CCD close error: %s", e)

    logger.info("CCD force-shutdown complete")


# ── 상태 조회 헬퍼 ──

def read_ccd_temp_and_status() -> tuple:
    """
    온도와 안정화 상태를 단일 SDK 호출로 읽어 (temperature, temp_status) 튜플 반환.
    get_temperature_status() 내부에서 get_temperature()를 호출하므로
    SDK DLL 락 획득은 1회만 발생.
    """
    ccd = get_ccd()
    if ccd is None:
        return None, None
    try:
        status = ccd.get_temperature_status()   # 내부에서 get_temperature() 호출
        temp = ccd.temperature                  # 위 호출로 캐시된 값
        return temp, status
    except Exception:
        return None, None
