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
import logging

from ....core.config import settings
from ....db.connection import insert_job, update_job_results, update_job_error, insert_artifact

router = APIRouter()
logger = logging.getLogger(__name__)

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

        # Log job payload when DB storage is enabled
        if settings.STORE_IN_DB:
            try:
                insert_job(request_id, payload={
                    'geometry': geometry,
                    'date': requested_date,
                    'max_cloud': requested_max_cloud,
                    'max_px': requested_max_px,
                    'res': 10,
                    'jpg_only': jpg_only,
                    'fast_jpg': fast_jpg,
                })
            except Exception as _je:
                # Don't fail request if logging fails
                pass

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
            # Optionally store artifacts in DB
            if settings.STORE_IN_DB:
                try:
                    if rgb_png_path and os.path.exists(rgb_png_path):
                        with open(rgb_png_path, 'rb') as f:
                            insert_artifact(request_id, 'sentinel_rgb.png', f.read(), 'image/png')
                    if rgb_jpg_path and os.path.exists(rgb_jpg_path):
                        with open(rgb_jpg_path, 'rb') as f:
                            insert_artifact(request_id, 'sentinel_rgb.jpg', f.read(), 'image/jpeg')
                    if rgb_tif_path and os.path.exists(rgb_tif_path):
                        with open(rgb_tif_path, 'rb') as f:
                            insert_artifact(request_id, 'sentinel_rgb.tif', f.read(), 'image/tiff')
                    if ndwi_tif_path and os.path.exists(ndwi_tif_path):
                        with open(ndwi_tif_path, 'rb') as f:
                            insert_artifact(request_id, 'ndwi.tif', f.read(), 'image/tiff')
                    # Optionally remove local files to avoid persisting to disk when storing in DB
                    for p in [rgb_png_path, rgb_jpg_path, rgb_tif_path, ndwi_tif_path]:
                        try:
                            if p and os.path.exists(p):
                                os.remove(p)
                        except Exception:
                            pass
                except Exception as _ae:
                    pass
            rgb_url = f"/static/{request_id}/sentinel_rgb.png" if rgb_png_path and os.path.exists(rgb_png_path) or settings.STORE_IN_DB else None
            rgb_jpg_url = f"/static/{request_id}/sentinel_rgb.jpg" if rgb_jpg_path and os.path.exists(rgb_jpg_path) or (settings.STORE_IN_DB and rgb_jpg_path) else None
            if rgb_url is None and rgb_jpg_url is not None:
                rgb_url = rgb_jpg_url
            rgb_tif_url = f"/static/{request_id}/sentinel_rgb.tif" if rgb_tif_path and os.path.exists(rgb_tif_path) or settings.STORE_IN_DB else None
            ndwi_tif_url = f"/static/{request_id}/ndwi.tif" if ndwi_tif_path and os.path.exists(ndwi_tif_path) or settings.STORE_IN_DB else None
            resp = {
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
            if settings.STORE_IN_DB:
                try:
                    update_job_results(request_id, resp.get('results', {}))
                except Exception:
                    pass
            return resp

        logger.info("Downloading Sentinel-2 data...")

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
        overlay_from_service = dl.get('overlay_png_path')
        bounds = dl['bounds']
        acquisition_date = dl.get('acquisition_date')
        resolution = 10  # meters, keep in sync with sentinel service

        # 2. If overlay already provided by service, use it; otherwise try model/NDWI paths
        mask = None
        mask_overlay_path = None
        overlay_saved = False
        if overlay_from_service and os.path.exists(overlay_from_service):
            mask_overlay_path = overlay_from_service
            overlay_saved = True
        else:
            if get_model_inference().model is not None and rgb_tif_path and os.path.exists(rgb_tif_path):
                try:
                    logger.info("Preprocessing image for model...")
                    processed_image = get_model_inference().preprocess_image(rgb_tif_path)
                    logger.info("Running model inference...")
                    mask = get_model_inference().predict(processed_image)
                    mask = (mask.squeeze() > 0).astype(np.uint8)
                except Exception as e:
                    logger.warning(f"Model inference not available, falling back to NDWI: {e}")
                    mask = None

            if mask is None:
                logger.info("Using NDWI threshold fallback...")
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
                if ndwi is not None:
                    mask = (ndwi > 0.2).astype(np.uint8)
                else:
                    # No NDWI data available; will proceed without mask generation
                    mask = None

        # 3. Save mask overlay PNG from computed mask if needed
        if mask_overlay_path is None and mask is not None:
            mask_overlay_path = os.path.join(output_dir, 'water_mask.png')
            try:
                get_model_inference().save_mask_as_overlay_png(mask, mask_overlay_path, color=(0, 150, 255, 255))
                overlay_saved = True
            except Exception as _e:
                overlay_saved = False

        # 4. Compute statistics
        # Water area from mask pixel count and pixel size
        pixel_area_km2 = (resolution * resolution) / 1e6
        if mask is not None:
            water_pixels = int(mask.sum())
            total_pixels = int(mask.size)
            water_area_km2 = water_pixels * pixel_area_km2
            total_mask_area_km2 = total_pixels * pixel_area_km2
        else:
            water_pixels = 0
            total_pixels = 0
            water_area_km2 = 0.0
            total_mask_area_km2 = 0.0

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

        # Optionally store artifacts in DB
        if settings.STORE_IN_DB:
            try:
                if rgb_png_path and os.path.exists(rgb_png_path):
                    with open(rgb_png_path, 'rb') as f:
                        insert_artifact(request_id, 'sentinel_rgb.png', f.read(), 'image/png')
                if rgb_jpg_path and os.path.exists(rgb_jpg_path):
                    with open(rgb_jpg_path, 'rb') as f:
                        insert_artifact(request_id, 'sentinel_rgb.jpg', f.read(), 'image/jpeg')
                if overlay_saved and mask_overlay_path and os.path.exists(mask_overlay_path):
                    with open(mask_overlay_path, 'rb') as f:
                        insert_artifact(request_id, 'water_mask.png', f.read(), 'image/png')
                if rgb_tif_path and os.path.exists(rgb_tif_path):
                    with open(rgb_tif_path, 'rb') as f:
                        insert_artifact(request_id, 'sentinel_rgb.tif', f.read(), 'image/tiff')
                if ndwi_tif_path and os.path.exists(ndwi_tif_path):
                    with open(ndwi_tif_path, 'rb') as f:
                        insert_artifact(request_id, 'ndwi.tif', f.read(), 'image/tiff')
                # Optionally remove local files to avoid persisting to disk when storing in DB
                for p in [rgb_png_path, rgb_jpg_path, mask_overlay_path, rgb_tif_path, ndwi_tif_path]:
                    try:
                        if p and os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
            except Exception:
                pass

        # 5. Build URLs for frontend
        rgb_url = f"/static/{request_id}/sentinel_rgb.png" if (rgb_png_path and os.path.exists(rgb_png_path)) or settings.STORE_IN_DB else None
        rgb_jpg_url = f"/static/{request_id}/sentinel_rgb.jpg" if (rgb_jpg_path and os.path.exists(rgb_jpg_path)) or (settings.STORE_IN_DB and rgb_jpg_path) else None
        if rgb_url is None and rgb_jpg_url is not None:
            rgb_url = rgb_jpg_url
        mask_url = f"/static/{request_id}/water_mask.png" if (overlay_saved and mask_overlay_path and os.path.exists(mask_overlay_path)) or (settings.STORE_IN_DB and overlay_saved) else None
        rgb_tif_url = f"/static/{request_id}/sentinel_rgb.tif" if (rgb_tif_path and os.path.exists(rgb_tif_path)) or settings.STORE_IN_DB else None
        ndwi_tif_url = f"/static/{request_id}/ndwi.tif" if (ndwi_tif_path and os.path.exists(ndwi_tif_path)) or settings.STORE_IN_DB else None

        resp = {
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
        if settings.STORE_IN_DB:
            try:
                update_job_results(request_id, resp.get('results', {}))
            except Exception:
                pass
        return resp
        
    except HTTPException as he:
        # Pass through HTTP errors like 404 for no imagery
        if settings.STORE_IN_DB:
            try:
                update_job_error(locals().get('request_id', 'unknown'), str(he.detail))
            except Exception:
                pass
        raise he
    except Exception as e:
        if settings.STORE_IN_DB:
            try:
                update_job_error(locals().get('request_id', 'unknown'), str(e))
            except Exception:
                pass
        logger.exception("Unhandled error in predict_waterbody")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/result/{request_id}")
async def get_result(request_id: str):
    """
    Get the result of a previous prediction by request ID.
    """
    if settings.STORE_IN_DB:
        try:
            from ....db.connection import get_artifact
            row = get_artifact(request_id, 'water_mask.png')
            if not row:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Results not available")
            content, content_type = row
            from fastapi.responses import Response
            return Response(content, media_type=content_type or "image/png")
        except HTTPException:
            raise
        except Exception as _e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(_e))
    else:
        mask_path = os.path.join(settings.OUTPUT_DIR, request_id, 'water_mask.png')
        if not os.path.exists(mask_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Request ID not found or results not available"
            )
        return FileResponse(mask_path, media_type="image/png")
