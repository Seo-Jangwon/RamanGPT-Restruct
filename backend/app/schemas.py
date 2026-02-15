from pydantic import BaseModel, Field
from typing import Optional, List, Tuple


# ── 공통 ──

class StatusResponse(BaseModel):
    status: str
    message: str


# ── Camera ──

class CameraConnectRequest(BaseModel):
    exposure_ms: float = Field(default=10.0, ge=0.1, le=10000.0)


class CameraStatus(BaseModel):
    connected: bool
    is_streaming: bool = False
    exposure_ms: Optional[float] = None


class ExposureRequest(BaseModel):
    exposure_ms: float = Field(ge=0.1, le=10000.0)


class FocusScoreResponse(BaseModel):
    score: float


# ── Stage ──

class StageConnectRequest(BaseModel):
    dll_path: str = Field(default="./Tango_DLL.dll")


class StagePosition(BaseModel):
    x: float
    y: float
    z: float
    a: float = 0.0


class StageMoveAbsoluteRequest(BaseModel):
    x: float
    y: float
    z: float
    a: float = 0.0
    wait: bool = True


class StageMoveRelativeRequest(BaseModel):
    dx: float
    dy: float
    dz: float
    da: float = 0.0
    wait: bool = True


class StageVelocityRequest(BaseModel):
    vx: float
    vy: float
    vz: float
    va: float = 5.0


class StageStatus(BaseModel):
    connected: bool
    position: Optional[StagePosition] = None


# ── Laser ──

class LaserConnectRequest(BaseModel):
    port: str = Field(default="COM4")
    baud: int = Field(default=115200)


class LaserFireRequest(BaseModel):
    duration: float = Field(ge=0.01, le=60.0, description="Duration in seconds")


class LaserStatus(BaseModel):
    connected: bool
    port: Optional[str] = None


# ── Scan & AI ──

class SAM3SegmentRequest(BaseModel):
    image_path: str
    text_prompts: List[str]
    output_dir: str = "./outputs"
    conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class SAM3SegmentResponse(BaseModel):
    total_objects: int
    objects: List[dict]
    visualization_path: str
