import asyncio

from fastapi import APIRouter, HTTPException

from backend.app.hardware import get_stage, connect_stage, disconnect_stage, hw_executor
from backend.app.schemas import (
    StageConnectRequest,
    StageMoveAbsoluteRequest,
    StageMoveRelativeRequest,
    StageVelocityRequest,
    StagePosition,
    StageStatus,
    StatusResponse,
)

router = APIRouter()


@router.post("/connect", response_model=StageStatus)
async def stage_connect(req: StageConnectRequest):
    try:
        await connect_stage(dll_path=req.dll_path)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    stage = get_stage()
    loop = asyncio.get_running_loop()
    pos = await loop.run_in_executor(hw_executor, stage.get_position)

    position = StagePosition(x=pos[0], y=pos[1], z=pos[2], a=pos[3]) if pos else None
    return StageStatus(connected=True, position=position)


@router.post("/disconnect", response_model=StatusResponse)
async def stage_disconnect():
    if get_stage() is None:
        raise HTTPException(status_code=400, detail="Stage not connected")
    await disconnect_stage()
    return StatusResponse(status="ok", message="Stage disconnected")


@router.get("/status", response_model=StageStatus)
async def stage_status():
    stage = get_stage()
    if stage is None:
        return StageStatus(connected=False)

    loop = asyncio.get_running_loop()
    pos = await loop.run_in_executor(hw_executor, stage.get_position)
    position = StagePosition(x=pos[0], y=pos[1], z=pos[2], a=pos[3]) if pos else None
    return StageStatus(connected=True, position=position)


@router.get("/position", response_model=StagePosition)
async def stage_position():
    stage = get_stage()
    if stage is None:
        raise HTTPException(status_code=404, detail="Stage not connected")

    loop = asyncio.get_running_loop()
    pos = await loop.run_in_executor(hw_executor, stage.get_position)
    if pos is None:
        raise HTTPException(status_code=500, detail="Position query failed")
    return StagePosition(x=pos[0], y=pos[1], z=pos[2], a=pos[3])


@router.post("/move/absolute", response_model=StagePosition)
async def stage_move_absolute(req: StageMoveAbsoluteRequest):
    stage = get_stage()
    if stage is None:
        raise HTTPException(status_code=404, detail="Stage not connected")

    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        hw_executor,
        lambda: stage.move_absolute(req.x, req.y, req.z, req.a, req.wait),
    )
    if not success:
        raise HTTPException(status_code=500, detail="Move failed")

    pos = await loop.run_in_executor(hw_executor, stage.get_position)
    return StagePosition(x=pos[0], y=pos[1], z=pos[2], a=pos[3])


@router.post("/move/relative", response_model=StagePosition)
async def stage_move_relative(req: StageMoveRelativeRequest):
    stage = get_stage()
    if stage is None:
        raise HTTPException(status_code=404, detail="Stage not connected")

    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        hw_executor,
        lambda: stage.move_relative(req.dx, req.dy, req.dz, req.da, req.wait),
    )
    if not success:
        raise HTTPException(status_code=500, detail="Move failed")

    pos = await loop.run_in_executor(hw_executor, stage.get_position)
    return StagePosition(x=pos[0], y=pos[1], z=pos[2], a=pos[3])


@router.post("/velocity", response_model=StatusResponse)
async def stage_set_velocity(req: StageVelocityRequest):
    stage = get_stage()
    if stage is None:
        raise HTTPException(status_code=404, detail="Stage not connected")

    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        hw_executor,
        lambda: stage.set_velocity(req.vx, req.vy, req.vz, req.va),
    )
    if not success:
        raise HTTPException(status_code=500, detail="Velocity setting failed")
    return StatusResponse(status="ok", message=f"Velocity set to ({req.vx}, {req.vy}, {req.vz})")
