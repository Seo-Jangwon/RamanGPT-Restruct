import asyncio
import os

from fastapi import APIRouter, HTTPException

from backend.app.hardware import hw_executor
from backend.app.schemas import (
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
