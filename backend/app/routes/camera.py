import asyncio
import io

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from backend.app.hardware import get_camera, connect_camera, disconnect_camera, hw_executor
from backend.app.schemas import (
    CameraConnectRequest,
    CameraStatus,
    ExposureRequest,
    FocusScoreResponse,
    StatusResponse,
)

router = APIRouter()


def _frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """프레임을 JPEG 바이트로 변환"""
    img = frame
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


@router.post("/connect", response_model=CameraStatus)
async def camera_connect(req: CameraConnectRequest):
    try:
        await connect_camera(exposure_ms=req.exposure_ms)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CameraStatus(connected=True, is_streaming=False, exposure_ms=req.exposure_ms)


@router.post("/disconnect", response_model=StatusResponse)
async def camera_disconnect():
    if get_camera() is None:
        raise HTTPException(status_code=400, detail="Camera not connected")
    await disconnect_camera()
    return StatusResponse(status="ok", message="Camera disconnected")


@router.get("/status", response_model=CameraStatus)
async def camera_status():
    cam = get_camera()
    if cam is None:
        return CameraStatus(connected=False)
    return CameraStatus(connected=True, is_streaming=cam.is_streaming)


@router.post("/exposure", response_model=StatusResponse)
async def camera_set_exposure(req: ExposureRequest):
    cam = get_camera()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not connected")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(hw_executor, cam.set_exposure, req.exposure_ms)
    return StatusResponse(status="ok", message=f"Exposure set to {req.exposure_ms}ms")


@router.post("/stream/start", response_model=StatusResponse)
async def camera_stream_start():
    cam = get_camera()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not connected")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(hw_executor, cam.start_stream)
    return StatusResponse(status="ok", message="Streaming started")


@router.post("/stream/stop", response_model=StatusResponse)
async def camera_stream_stop():
    cam = get_camera()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not connected")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(hw_executor, cam.stop_stream)
    return StatusResponse(status="ok", message="Streaming stopped")


@router.get("/capture")
async def camera_capture():
    """단일 프레임을 JPEG으로 반환 (Swagger 테스트용)"""
    cam = get_camera()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not connected")
    if not cam.is_streaming:
        raise HTTPException(status_code=400, detail="Camera not streaming. Call /stream/start first")

    loop = asyncio.get_running_loop()
    frame = await loop.run_in_executor(hw_executor, cam.get_latest_frame)
    if frame is None:
        raise HTTPException(status_code=500, detail="Frame capture failed")

    jpeg_bytes = await loop.run_in_executor(hw_executor, _frame_to_jpeg, frame)
    return StreamingResponse(io.BytesIO(jpeg_bytes), media_type="image/jpeg")


@router.get("/focus-score", response_model=FocusScoreResponse)
async def camera_focus_score():
    """현재 프레임의 초점 점수 반환"""
    cam = get_camera()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not connected")
    if not cam.is_streaming:
        raise HTTPException(status_code=400, detail="Camera not streaming")

    from backend.autofocus.autofocus import compute_focus_score

    loop = asyncio.get_running_loop()
    frame = await loop.run_in_executor(hw_executor, cam.get_latest_frame)
    if frame is None:
        raise HTTPException(status_code=500, detail="Frame capture failed")

    score = await loop.run_in_executor(hw_executor, compute_focus_score, frame)
    return FocusScoreResponse(score=score)


@router.websocket("/ws")
async def camera_ws_stream(websocket: WebSocket):
    """WebSocket MJPEG 스트림 (~30fps)"""
    cam = get_camera()
    if cam is None or not cam.is_streaming:
        await websocket.close(code=1008, reason="Camera not connected or not streaming")
        return

    await websocket.accept()
    loop = asyncio.get_running_loop()

    try:
        while True:
            frame = await loop.run_in_executor(hw_executor, cam.get_latest_frame)
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            jpeg_bytes = await loop.run_in_executor(hw_executor, _frame_to_jpeg, frame, 70)
            await websocket.send_bytes(jpeg_bytes)
            await asyncio.sleep(0.033)  # ~30 FPS cap
    except WebSocketDisconnect:
        pass
