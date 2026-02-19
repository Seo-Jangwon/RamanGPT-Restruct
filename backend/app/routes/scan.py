import asyncio
import os

from fastapi import APIRouter, HTTPException

from backend.app.hardware import get_camera, get_stage, hw_executor
from backend.app.schemas import (
    AutofocusRequest,
    AutofocusResponse,
    SAM3SegmentRequest,
    SAM3SegmentResponse,
)

router = APIRouter()


@router.post("/sam3/segment", response_model=SAM3SegmentResponse)
async def sam3_segment(req: SAM3SegmentRequest):
    """SAM3 텍스트 프롬프트 세그멘테이션"""
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"Image not found: {req.image_path}")

    from backend.scan.sam3 import segment_with_text_prompt

    def _run():
        return segment_with_text_prompt(
            image_path=req.image_path,
            text_prompts=req.text_prompts,
            output_dir=req.output_dir,
            conf_threshold=req.conf_threshold,
        )

    loop = asyncio.get_running_loop()
    try:
        detected_objects = await loop.run_in_executor(hw_executor, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    vis_path = os.path.join(req.output_dir, "sam3_result.png")
    return SAM3SegmentResponse(
        total_objects=len(detected_objects),
        objects=detected_objects,
        visualization_path=vis_path,
    )


@router.post("/autofocus", response_model=AutofocusResponse)
async def run_autofocus(req: AutofocusRequest):
    """
    SciPy + Gemini 오토포커스 파이프라인

    카메라와 스테이지가 모두 연결되어 있어야 합니다.
    카메라는 스트리밍 상태여야 합니다.
    GEMINI_API_KEY 환경변수가 설정되면 Gemini 검증이 활성화됩니다.
    """
    camera = get_camera()
    stage = get_stage()

    if camera is None:
        raise HTTPException(status_code=400, detail="Camera not connected")
    if not camera.is_streaming:
        raise HTTPException(status_code=400, detail="Camera not streaming. Call /api/camera/stream/start first")
    if stage is None:
        raise HTTPException(status_code=400, detail="Stage not connected")

    from backend.autofocus.autofocus_scipy import AutoFocusAgent

    gemini_key = os.environ.get("GEMINI_API_KEY")

    def _run():
        agent = AutoFocusAgent(
            camera=camera,
            stage=stage,
            gemini_api_key=gemini_key,
        )
        return agent.run(
            z_range=(req.z_min, req.z_max),
            target_description=req.target_description,
            coarse_step=req.coarse_step,
            fine_tolerance=req.fine_tolerance,
        )

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(hw_executor, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AutofocusResponse(
        status=result["status"],
        optimal_z=result.get("optimal_z"),
        best_score=result.get("best_score"),
        coarse_samples=result.get("coarse_samples"),
        fine_iterations=result.get("fine_iterations"),
        reason=result.get("reason"),
    )
