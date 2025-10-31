from fastapi import APIRouter
from .endpoints import segmentation, health, roi

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(segmentation.router, prefix="/segmentation", tags=["Segmentation"])
api_router.include_router(roi.router, prefix="/roi", tags=["ROI"])
