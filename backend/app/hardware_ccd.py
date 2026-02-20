"""
Andor CCD 싱글톤 매니저

startup : connect → cooler ON → set temp -40°C
shutdown : set temp -5°C → wait (max 30s) → ShutDown

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
    CCD 연결 → 쿨러 ON → 온도 목표 설정 (-40°C).
    온도 안정화는 비동기로 진행되므로 즉시 반환.
    """
    global _ccd
    with _ccd_lock:
        if _ccd is not None:
            raise RuntimeError("CCD already connected")

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
    logger.info("CCD connected")


async def disconnect_ccd(warmup_temp: int = -5, max_wait_sec: int = 30) -> None:
    """
    CCD 안전 종료:
      1) 온도 목표를 warmup_temp(-5°C) 로 설정 (열충격 방지)
      2) max_wait_sec 초 동안 온도 상승 대기
      3) ShutDown
    """
    global _ccd
    with _ccd_lock:
        ccd = _ccd
        _ccd = None

    if ccd is None:
        return

    loop = asyncio.get_running_loop()

    def _warmup_and_close() -> None:
        try:
            ccd.set_temperature(warmup_temp)
            logger.info("CCD warmup: target=%d°C, waiting up to %ds", warmup_temp, max_wait_sec)
        except Exception as e:
            logger.warning("CCD set_temperature error during shutdown: %s", e)

        deadline = time.monotonic() + max_wait_sec
        while time.monotonic() < deadline:
            try:
                temp = ccd.get_temperature()
                logger.info("CCD warmup temp=%d°C", temp)
                if temp >= warmup_temp:
                    break
            except Exception:
                break
            time.sleep(5)

        try:
            ccd.close()
            logger.info("CCD ShutDown complete")
        except Exception as e:
            logger.warning("CCD close error: %s", e)

    await loop.run_in_executor(ccd_executor, _warmup_and_close)


def shutdown_ccd_sync(warmup_temp: int = -5) -> None:
    """동기 강제 종료 — lifespan 외 비상 상황에서만 사용."""
    global _ccd
    with _ccd_lock:
        ccd = _ccd
        _ccd = None

    if ccd is None:
        return

    try:
        ccd.set_temperature(warmup_temp)
    except Exception as e:
        logger.warning("CCD temp set error: %s", e)

    try:
        ccd.close()
    except Exception as e:
        logger.warning("CCD close error: %s", e)

    logger.info("CCD force-shutdown complete")


# ── 상태 조회 헬퍼 ──

def read_ccd_temperature() -> Optional[int]:
    ccd = get_ccd()
    if ccd is None:
        return None
    try:
        return ccd.get_temperature()
    except Exception:
        return None


def read_ccd_temp_status() -> Optional[str]:
    ccd = get_ccd()
    if ccd is None:
        return None
    try:
        return ccd.get_temperature_status()
    except Exception:
        return None
