import asyncio

from fastapi import APIRouter, HTTPException

from backend.app.hardware_ccd import (
    get_ccd,
    connect_ccd,
    disconnect_ccd,
    read_ccd_temperature,
    read_ccd_temp_status,
    ccd_executor,
)
from backend.app.schemas import (
    CCDConnectRequest,
    CCDTemperatureRequest,
    CCDStatus,
    StatusResponse,
)

router = APIRouter()


@router.post("/connect", response_model=CCDStatus)
async def ccd_connect(req: CCDConnectRequest):
    # 이미 연결된 경우 현재 상태를 그대로 반환 (idempotent)
    if get_ccd() is not None:
        return CCDStatus(
            connected=True,
            temperature=read_ccd_temperature(),
            temp_status=read_ccd_temp_status(),
        )
    try:
        await connect_ccd(target_temp=req.target_temp)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return CCDStatus(
        connected=True,
        temperature=read_ccd_temperature(),
        temp_status=read_ccd_temp_status(),
    )


@router.post("/disconnect", response_model=StatusResponse)
async def ccd_disconnect():
    if get_ccd() is None:
        raise HTTPException(status_code=400, detail="CCD not connected")
    await disconnect_ccd()
    return StatusResponse(status="ok", message="CCD disconnected")


@router.get("/status", response_model=CCDStatus)
async def ccd_status():
    if get_ccd() is None:
        return CCDStatus(connected=False)
    return CCDStatus(
        connected=True,
        temperature=read_ccd_temperature(),
        temp_status=read_ccd_temp_status(),
    )


@router.post("/temperature", response_model=StatusResponse)
async def ccd_set_temperature(req: CCDTemperatureRequest):
    ccd = get_ccd()
    if ccd is None:
        raise HTTPException(status_code=404, detail="CCD not connected")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(ccd_executor, ccd.set_temperature, req.temperature)
    return StatusResponse(status="ok", message=f"CCD target temperature set to {req.temperature}°C")


@router.post("/cooler/{state}", response_model=StatusResponse)
async def ccd_cooler(state: str):
    ccd = get_ccd()
    if ccd is None:
        raise HTTPException(status_code=404, detail="CCD not connected")
    if state not in ("on", "off"):
        raise HTTPException(status_code=422, detail="state must be 'on' or 'off'")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(ccd_executor, ccd.set_cooler, state == "on")
    return StatusResponse(status="ok", message=f"CCD cooler turned {state}")
