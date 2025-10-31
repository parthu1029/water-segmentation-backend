from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
import uuid

router = APIRouter()

@router.post("/store-geo-cordinate")
async def store_geo_cordinate(geojson: Dict[str, Any]):
    if not isinstance(geojson, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid payload")
    geometry = geojson.get("geometry") if "geometry" in geojson else geojson
    if not isinstance(geometry, dict) or geometry.get("type") not in ("Polygon", "MultiPolygon"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="GeoJSON must contain a Polygon or MultiPolygon geometry")
    roi_id = str(uuid.uuid4())
    try:
        # optional properties if client sends Feature-like payload
        properties = geojson.get("properties") if isinstance(geojson, dict) else None
        from ....db.connection import insert_roi
        insert_roi(roi_id, geometry=geometry, properties=properties)
        return {"status": "stored", "roi_id": roi_id}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"DB error: {e}")
