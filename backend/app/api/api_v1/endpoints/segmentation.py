import os
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import numpy as np
from fastapi.responses import FileResponse
import importlib
import json
import hashlib

from ....core.config import settings

router = APIRouter()

# Lazy singletons for heavy services
_sentinel_service = None
_model_inference = None

def get_sentinel_service():
    global _sentinel_service
    if _sentinel_service is None:
        from ....services.sentinel import SentinelHubService
        _sentinel_service = SentinelHubService()
    return _sentinel_service

def get_model_inference():
    global _model_inference
    if _model_inference is None:
        from ....services.inference import ModelInference
        _model_inference = ModelInference(model_path=settings.MODEL_PATH)
    return _model_inference

@router.post("/predict")
async def predict_waterbody(geojson: Dict[str, Any]):
    """
    Process a GeoJSON polygon and return water segmentation results.
    
    Args:
        geojson: GeoJSON containing a polygon feature
        
    Returns:
        Dictionary containing paths to the results
    """
    try:
        # Parse GeoJSON and get coordinates
        geometry = geojson.get('geometry') if isinstance(geojson, dict) and 'geometry' in geojson else geojson
        if not geometry or geometry.get('type') not in ('Polygon', 'MultiPolygon'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GeoJSON must contain a Polygon or MultiPolygon geometry"
            )
        jpg_only = False
        try:
            jpg_only = bool(geojson.get('jpg_only')) if isinstance(geojson, dict) else False
        except Exception:
            jpg_only = False
        fast_jpg = False
        try:
            fast_jpg = bool(geojson.get('fast_jpg')) if isinstance(geojson, dict) else False
        except Exception:
            fast_jpg = False
        requested_date = None
        requested_max_cloud = None
        requested_max_px = None
        if isinstance(geojson, dict):
            requested_date = geojson.get('date')
            requested_max_cloud = geojson.get('max_cloud')
            try:
                requested_max_cloud = int(requested_max_cloud) if requested_max_cloud is not None else None
            except Exception:
                requested_max_cloud = None
            try:
                requested_max_px = geojson.get('max_px')
                requested_max_px = int(requested_max_px) if requested_max_px is not None else None
            except Exception:
                requested_max_px = None
        if jpg_only and fast_jpg and (requested_max_px is None or requested_max_px <= 0):
            requested_max_px = 1500

        cache_id = None
        try:
            payload = {
                'geometry': geometry,
                'date': requested_date or '',
                'max_cloud': requested_max_cloud if requested_max_cloud is not None else '',
                'res': 10,
                'max_px': requested_max_px if requested_max_px is not None else '',
            }
            m = hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
            cache_id = m.hexdigest()[:24]
        except Exception:
            cache_id = str(uuid.uuid4())

        request_id = cache_id if jpg_only else str(uuid.uuid4())
        output_dir = os.path.join(settings.OUTPUT_DIR, request_id)
        os.makedirs(output_dir, exist_ok=True)

        if jpg_only:
            rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
            rgb_jpg_path = os.path.join(output_dir, 'sentinel_rgb.jpg')
            rgb_tif_path = os.path.join(output_dir, 'sentinel_rgb.tif')
            ndwi_tif_path = os.path.join(output_dir, 'ndwi.tif')
            bounds = None
            acquisition_date = None
            if not (os.path.exists(rgb_jpg_path) or os.path.exists(rgb_png_path)):
                dl = await get_sentinel_service().download_sentinel_data(
                    geometry_geojson=geometry,
                    output_dir=output_dir,
                    date_iso=requested_date,
                    max_cloud=requested_max_cloud,
                    max_px=requested_max_px
                )
                rgb_png_path = dl.get('rgb_png_path')
                rgb_jpg_path = dl.get('rgb_jpg_path')
                rgb_tif_path = dl.get('rgb_tif_path')
                ndwi_tif_path = dl.get('ndwi_tif_path')
                bounds = dl.get('bounds')
                acquisition_date = dl.get('acquisition_date')
            else:
                try:
                    rasterio = importlib.import_module("rasterio")
                    if os.path.exists(rgb_tif_path):
                        with rasterio.open(rgb_tif_path) as src:
                            b = src.bounds
                            bounds = [[b.bottom, b.left], [b.top, b.right]]
                except Exception:
                    bounds = None
            rgb_url = f"/static/{request_id}/sentinel_rgb.png" if rgb_png_path and os.path.exists(rgb_png_path) else None
            rgb_jpg_url = f"/static/{request_id}/sentinel_rgb.jpg" if rgb_jpg_path and os.path.exists(rgb_jpg_path) else None
            rgb_tif_url = f"/static/{request_id}/sentinel_rgb.tif" if rgb_tif_path and os.path.exists(rgb_tif_path) else None
            ndwi_tif_url = f"/static/{request_id}/ndwi.tif" if ndwi_tif_path and os.path.exists(ndwi_tif_path) else None
            return {
                'status': 'success',
                'request_id': request_id,
                'timestamp': datetime.utcnow().isoformat(),
                'results': {
                    'rgb_url': rgb_url,
                    'rgb_jpg_url': rgb_jpg_url,
                    'rgb_tif_url': rgb_tif_url,
                    'ndwi_tif_url': ndwi_tif_url,
                    'bounds': bounds,
                    'acquisition_date': acquisition_date,
                }
            }

        print("Downloading Sentinel-2 data...")

        try:
            max_px_for_analyze = requested_max_px if (isinstance(requested_max_px, int) and requested_max_px > 0) else 1500
            dl = await get_sentinel_service().download_sentinel_data(
                geometry_geojson=geometry,
                output_dir=output_dir,
                date_iso=requested_date,
                max_cloud=requested_max_cloud,
                max_px=max_px_for_analyze
            )
        except ValueError as ve:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(ve)
            )

        rgb_tif_path = dl.get('rgb_tif_path')
        ndwi_tif_path = dl.get('ndwi_tif_path')
        ndwi_npy_path = dl.get('ndwi_npy_path')
        rgb_png_path = dl.get('rgb_png_path')
        rgb_jpg_path = dl.get('rgb_jpg_path')
        bounds = dl['bounds']
        acquisition_date = dl.get('acquisition_date')
        resolution = 10  # meters, keep in sync with sentinel service

        # 2. Try model inference if model is available and RGB GeoTIFF is available
        mask = None
        if get_model_inference().model is not None and rgb_tif_path and os.path.exists(rgb_tif_path):
            try:
                print("Preprocessing image for model...")
                processed_image = get_model_inference().preprocess_image(rgb_tif_path)
                print("Running model inference...")
                mask = get_model_inference().predict(processed_image)
                mask = (mask.squeeze() > 0).astype(np.uint8)
            except Exception as e:
                print(f"Model inference not available, falling back to NDWI: {e}")
                mask = None

        if mask is None:
            print("Using NDWI threshold fallback...")
            # Read NDWI and threshold (e.g., > 0.2)
            ndwi = None
            if ndwi_tif_path and os.path.exists(ndwi_tif_path):
                try:
                    rasterio = importlib.import_module("rasterio")
                    with rasterio.open(ndwi_tif_path) as src:
                        ndwi = src.read(1)
                except Exception:
                    ndwi = None
            if ndwi is None and ndwi_npy_path and os.path.exists(ndwi_npy_path):
                ndwi = np.load(ndwi_npy_path)
            if ndwi is None:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="NDWI data not available for processing")
            # -1 marks invalid from evalscript; ignore negatives by thresholding > 0.2
            mask = (ndwi > 0.2).astype(np.uint8)

        # 3. Save mask overlay PNG (transparent background)
        mask_overlay_path = os.path.join(output_dir, 'water_mask.png')
        overlay_saved = False
        try:
            get_model_inference().save_mask_as_overlay_png(mask, mask_overlay_path, color=(0, 150, 255, 255))
            overlay_saved = True
        except Exception as _e:
            overlay_saved = False

        # 4. Compute statistics
        # Water area from mask pixel count and pixel size
        water_pixels = int(mask.sum())
        total_pixels = int(mask.size)
        pixel_area_km2 = (resolution * resolution) / 1e6
        water_area_km2 = water_pixels * pixel_area_km2
        total_mask_area_km2 = total_pixels * pixel_area_km2

        # Total polygon area using geodesic area
        try:
            from pyproj import Geod
            coords = geometry.get('coordinates', [])
            ring = []
            if geometry.get('type') == 'Polygon' and coords:
                ring = coords[0]
            elif geometry.get('type') == 'MultiPolygon' and coords:
                ring = coords[0][0]
            lon = [p[0] for p in ring]
            lat = [p[1] for p in ring]
            geod = Geod(ellps="WGS84")
            area, _ = geod.polygon_area_perimeter(lon, lat)
            total_area_km2 = abs(area) / 1e6
        except Exception:
            # Fallback to mask-area estimate
            total_area_km2 = total_mask_area_km2

        water_percentage = (water_area_km2 / total_area_km2) * 100 if total_area_km2 > 0 else 0.0

        # 5. Build URLs for frontend
        rgb_url = f"/static/{request_id}/sentinel_rgb.png" if rgb_png_path and os.path.exists(rgb_png_path) else None
        rgb_jpg_url = f"/static/{request_id}/sentinel_rgb.jpg" if rgb_jpg_path and os.path.exists(rgb_jpg_path) else None
        mask_url = f"/static/{request_id}/water_mask.png" if overlay_saved and os.path.exists(mask_overlay_path) else None
        rgb_tif_url = f"/static/{request_id}/sentinel_rgb.tif" if rgb_tif_path and os.path.exists(rgb_tif_path) else None
        ndwi_tif_url = f"/static/{request_id}/ndwi.tif" if ndwi_tif_path and os.path.exists(ndwi_tif_path) else None

        return {
            'status': 'success',
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'results': {
                'rgb_url': rgb_url,
                'rgb_jpg_url': rgb_jpg_url,
                'mask_url': mask_url,
                'rgb_tif_url': rgb_tif_url,
                'ndwi_tif_url': ndwi_tif_url,
                'bounds': bounds,
                'total_area_km2': float(total_area_km2),
                'water_area_km2': float(water_area_km2),
                'water_percentage': float(water_percentage),
                'acquisition_date': acquisition_date,
            }
        }
        
    except HTTPException as he:
        # Pass through HTTP errors like 404 for no imagery
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/result/{request_id}")
async def get_result(request_id: str):
    """
    Get the result of a previous prediction by request ID.
    """
    mask_path = os.path.join(settings.OUTPUT_DIR, request_id, 'water_mask.png')
    if not os.path.exists(mask_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request ID not found or results not available"
        )
    
    return FileResponse(mask_path, media_type="image/png")
