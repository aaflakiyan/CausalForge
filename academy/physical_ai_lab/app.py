"""Single-user web lab. Server-side MuJoCo; browser draws real state data."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
from .engine import ReachingLab

WEB = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="Physical AI Academy — Lab 01", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB), name="static")
_lab: Optional[ReachingLab] = None


class Config(BaseModel):
    x: float = Field(0.45, ge=-0.7, le=0.7)
    y: float = Field(0.35, ge=-0.7, le=0.7)
    gain: float = Field(2.5, ge=0.1, le=6.0)
    noise: float = Field(0.0, ge=0, le=0.06)


class Step(BaseModel):
    steps: int = Field(10, ge=1, le=100)


def lab():
    global _lab
    if _lab is None:
        try:
            _lab = ReachingLab()
        except (ImportError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"MuJoCo engine unavailable: {exc}") from exc
    return _lab


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "mujoco", "single_user": True}


@app.get("/api/state")
def get_state():
    return lab().snapshot()


@app.post("/api/reset")
def reset():
    return lab().reset()


@app.post("/api/config")
def configure(cfg: Config):
    try:
        return lab().configure(target=(cfg.x, cfg.y), gain=cfg.gain, noise=cfg.noise)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/step")
def step(args: Step):
    return lab().step(args.steps)
