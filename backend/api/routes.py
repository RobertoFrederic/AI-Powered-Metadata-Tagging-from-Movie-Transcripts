from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import json
import os
from pathlib import Path

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "metadata-tagging-api"}

@router.get("/config")
async def get_config():
    from config import settings
    return {
        "api_keys_configured": settings.validate_api_keys(),
        "directories_ready": True,
        "max_file_size": settings.MAX_FILE_SIZE
    }
