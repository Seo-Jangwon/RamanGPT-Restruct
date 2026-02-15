import asyncio

from fastapi import APIRouter, HTTPException

from backend.app.hardware import get_laser, connect_laser, disconnect_laser, hw_executor
from backend.app.schemas import (
    LaserConnectRequest,
    LaserFireRequest,
    LaserStatus,
    StatusResponse,
)

router = APIRouter()


@router.post("/connect", response_model=LaserStatus)
async def laser_connect(req: LaserConnectRequest):
    try:
        await connect_laser(port=req.port, baud=req.baud)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return LaserStatus(connected=True, port=req.port)


@router.post("/disconnect", response_model=StatusResponse)
async def laser_disconnect():
    if get_laser() is None:
        raise HTTPException(status_code=400, detail="Laser not connected")
    await disconnect_laser()
    return StatusResponse(status="ok", message="Laser disconnected")


@router.get("/status", response_model=LaserStatus)
async def laser_status():
    laser = get_laser()
    if laser is None:
        return LaserStatus(connected=False)
    return LaserStatus(connected=True, port=laser.port)


@router.post("/fire", response_model=StatusResponse)
async def laser_fire(req: LaserFireRequest):
    laser = get_laser()
    if laser is None:
        raise HTTPException(status_code=404, detail="Laser not connected")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(hw_executor, laser.laser_on, req.duration)
    return StatusResponse(status="ok", message=f"Laser fired for {req.duration}s")
