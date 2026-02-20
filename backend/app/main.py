import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.hardware import shutdown_all
from backend.app.hardware_ccd import connect_ccd, disconnect_ccd
from backend.app.routes import camera, stage, laser, scan, ccd

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 (main.py 기준)
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info("Server started")
    logger.info("API docs: http://localhost:8000/docs")
    try:
        await connect_ccd(target_temp=-40)
        logger.info("CCD startup: target temp -40°C set")
    except Exception as e:
        logger.warning("CCD startup failed (continuing without CCD): %s", e)
    yield
    # shutdown
    try:
        await disconnect_ccd(warmup_temp=-5, max_wait_sec=30)
    except Exception as e:
        logger.warning("CCD shutdown error: %s", e)
    shutdown_all()
    logger.info("Server stopped")


app = FastAPI(
    title="RamanGPT",
    description="FastAPI + React 통합 프로젝트",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정 (개발 중에 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(camera.router, prefix="/api/camera", tags=["Camera"])
app.include_router(stage.router, prefix="/api/stage", tags=["Stage"])
app.include_router(laser.router, prefix="/api/laser", tags=["Laser"])
app.include_router(scan.router, prefix="/api/scan", tags=["Scan & AI"])
app.include_router(ccd.router, prefix="/api/ccd", tags=["CCD"])


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "message": "Server is running"}


# React 정적 파일 서빙 (빌드 후 활성화)
# catch-all은 반드시 라우터 등록 이후에 위치해야 함
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        if not full_path.startswith("api"):
            return FileResponse(str(STATIC_DIR / "index.html"))
