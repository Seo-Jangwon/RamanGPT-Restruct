import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 (main.py 기준)
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info("Server started")
    logger.info("API docs: http://localhost:8000/docs")
    yield
    # shutdown
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

# API 라우터 등록 (나중에 추가)
# from backend.app.api import users
# app.include_router(users.router, prefix="/api/v1/users", tags=["users"])


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "message": "Server is running"}


# React 정적 파일 서빙 (빌드 후 활성화)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        if not full_path.startswith("api"):
            return FileResponse(str(STATIC_DIR / "index.html"))
