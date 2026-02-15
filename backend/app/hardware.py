"""
하드웨어 싱글톤 매니저

각 디바이스는 모듈 레벨 글로벌 변수로 관리.
연결/해제는 명시적 API 호출로만 수행.
모든 blocking HW 호출은 hw_executor를 통해 실행.
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# ── 글로벌 인스턴스 (None = 미연결) ──
_camera = None  # StreamingTUCam
_stage = None   # TangoController
_laser = None   # LaserController

# ── 디바이스별 Lock ──
_camera_lock = threading.Lock()
_stage_lock = threading.Lock()
_laser_lock = threading.Lock()

# ── 공유 ThreadPoolExecutor ──
hw_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hw")


# ── Getter ──

def get_camera():
    return _camera


def get_stage():
    return _stage


def get_laser():
    return _laser


# ── Camera ──

async def connect_camera(exposure_ms: float = 10.0):
    global _camera
    with _camera_lock:
        if _camera is not None:
            raise RuntimeError("Camera already connected")

    from backend.autofocus.autofocus import StreamingTUCam

    loop = asyncio.get_running_loop()
    cam = await loop.run_in_executor(hw_executor, StreamingTUCam, exposure_ms)

    with _camera_lock:
        _camera = cam
    logger.info("Camera connected (exposure=%.1fms)", exposure_ms)
    return cam


async def disconnect_camera():
    global _camera
    with _camera_lock:
        cam = _camera
        _camera = None

    if cam is not None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(hw_executor, cam.close)
        logger.info("Camera disconnected")


# ── Stage ──

async def connect_stage(dll_path: str = "./Tango_DLL.dll"):
    global _stage
    with _stage_lock:
        if _stage is not None:
            raise RuntimeError("Stage already connected")

    from backend.util.stage_move.stage_test import TangoController

    def _init_stage():
        tango = TangoController(dll_path)
        if not tango.load_dll():
            raise RuntimeError("Stage DLL load failed")
        if not tango.create_session():
            raise RuntimeError("Stage session creation failed")
        if not tango.connect():
            raise RuntimeError("Stage connection failed")
        return tango

    loop = asyncio.get_running_loop()
    stage = await loop.run_in_executor(hw_executor, _init_stage)

    with _stage_lock:
        _stage = stage
    logger.info("Stage connected (dll=%s)", dll_path)
    return stage


async def disconnect_stage():
    global _stage
    with _stage_lock:
        stage = _stage
        _stage = None

    if stage is not None:
        loop = asyncio.get_running_loop()

        def _cleanup():
            stage.disconnect()
            stage.free_session()

        await loop.run_in_executor(hw_executor, _cleanup)
        logger.info("Stage disconnected")


# ── Laser ──

async def connect_laser(port: str = "COM4", baud: int = 115200):
    global _laser
    with _laser_lock:
        if _laser is not None:
            raise RuntimeError("Laser already connected")

    from backend.scan.laser import LaserController

    loop = asyncio.get_running_loop()
    laser = await loop.run_in_executor(hw_executor, LaserController, port, baud)

    if laser.ser is None:
        raise RuntimeError(f"Laser serial connection failed on {port}")

    with _laser_lock:
        _laser = laser
    logger.info("Laser connected (port=%s)", port)
    return laser


async def disconnect_laser():
    global _laser
    with _laser_lock:
        laser = _laser
        _laser = None

    if laser is not None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(hw_executor, laser.close)
        logger.info("Laser disconnected")


# ── Shutdown ──

def shutdown_all():
    """lifespan shutdown에서 호출. 동기적으로 모든 디바이스 해제."""
    global _camera, _stage, _laser

    if _camera is not None:
        try:
            _camera.close()
        except Exception as e:
            logger.warning("Camera cleanup error: %s", e)
        _camera = None

    if _stage is not None:
        try:
            _stage.disconnect()
            _stage.free_session()
        except Exception as e:
            logger.warning("Stage cleanup error: %s", e)
        _stage = None

    if _laser is not None:
        try:
            _laser.close()
        except Exception as e:
            logger.warning("Laser cleanup error: %s", e)
        _laser = None

    hw_executor.shutdown(wait=False)
    logger.info("All hardware shut down")
