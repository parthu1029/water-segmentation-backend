import os
import numpy as np
import logging

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except Exception as e:
    logging.error("Failed to import rasterio: %s", e)
    rasterio = None
    from_bounds = None
    RASTERIO_AVAILABLE = False
import json
try:
    import urllib.request as urlrequest
    import urllib.parse as urlparse
except Exception:
    urlrequest = None
    urlparse = None
try:
    from shapely.geometry import shape as shp_shape
    SHAPELY_AVAILABLE = True
except Exception:
    shp_shape = None
    SHAPELY_AVAILABLE = False
from datetime import datetime, timedelta
try:
    from PIL import Image
    from PIL import features as PIL_features
    PIL_AVAILABLE = True
except Exception:
    Image = None
    PIL_features = None
    PIL_AVAILABLE = False

def _bounds_from_geojson(geometry_geojson):
    """Compute (minx, miny, maxx, maxy) from a Polygon/MultiPolygon GeoJSON without Shapely.
    Assumes coordinates are in lon/lat order (EPSG:4326).
    """
    def _flatten(coords):
        for c in coords:
            if isinstance(c[0], (float, int)) and isinstance(c[1], (float, int)):
                yield c
            else:
                yield from _flatten(c)

    coords = list(_flatten(geometry_geojson.get('coordinates', [])))
    if not coords:
        raise ValueError('Invalid geometry: no coordinates')
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))

logger = logging.getLogger(__name__)

class SentinelHubService:
    """Service for interacting with Sentinel Hub API"""

    def __init__(self):
        self.client_id = os.getenv('SENTINEL_HUB_CLIENT_ID', '')
        self.client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET', '')
        # Public WMS instance (can be overridden via env)
        self.wms_instance = os.getenv('SENTINEL_HUB_WMS_INSTANCE', 'ef5210ea-db25-42dc-afc1-b95646d1d02c')

    def _get_token(self) -> str:
        if not self.client_id or not self.client_secret:
            logger.warning("Sentinel Hub credentials not found. Set SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET.")
            return ""
        if urlrequest is None:
            raise RuntimeError("urllib not available for HTTP requests")
        data = urlparse.urlencode({
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }).encode()
        req = urlrequest.Request(
            url='https://services.sentinel-hub.com/oauth/token',
            data=data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        with urlrequest.urlopen(req) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
            return payload.get('access_token', '')

    def _process_png(self, token: str, bbox: tuple, width: int, height: int, evalscript: str, time_from: str, time_to: str, max_cloud: int | None):
        if urlrequest is None:
            raise RuntimeError("urllib not available for HTTP requests")
        data_filter = {
            'timeRange': { 'from': f"{time_from}T00:00:00Z", 'to': f"{time_to}T23:59:59Z" },
            'mosaickingOrder': 'leastCC',
        }
        if max_cloud is not None:
            try:
                v = max(0, min(100, int(max_cloud)))
                data_filter['maxCloudCoverage'] = float(v) / 100.0
            except Exception:
                pass
        payload = {
            'input': {
                'bounds': {
                    'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                    'properties': { 'crs': 'http://www.opengis.net/def/crs/EPSG/0/4326' }
                },
                'data': [
                    {
                        'type': 'S2L2A',
                        'dataFilter': data_filter
                    }
                ]
            },
            'output': {
                'width': int(max(1, width)),
                'height': int(max(1, height)),
                'responses': [
                    { 'identifier': 'default', 'format': { 'type': 'image/png' } }
                ]
            },
            'evalscript': evalscript
        }
        req = urlrequest.Request(
            url='https://services.sentinel-hub.com/api/v1/process',
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'Accept': 'image/png'
            }
        )
        with urlrequest.urlopen(req) as resp:
            return resp.read()

    def _process_wms_png(self, bbox: tuple, width: int, height: int, time_from: str, time_to: str, max_cloud: int | None, evalscript: str | None = None):
        if urlrequest is None or urlparse is None:
            raise RuntimeError("urllib not available for HTTP requests")
        base = f"https://services.sentinel-hub.com/ogc/wms/{self.wms_instance}"
        params = {
            'REQUEST': 'GetMap',
            'FORMAT': 'image/png',
            'TRANSPARENT': 'TRUE',
            'CRS': 'EPSG:4326',
            'BBOX': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            'WIDTH': str(int(max(1, width))),
            'HEIGHT': str(int(max(1, height))),
            'TIME': f"{time_from}/{time_to}",
            'SHOWLOGO': 'false',
        }
        if max_cloud is not None:
            try:
                v = max(0, min(100, int(max_cloud)))
                params['MAXCC'] = str(v)
            except Exception:
                pass
        # Use EVALSCRIPT when provided; otherwise default layer
        if evalscript:
            params['EVALSCRIPT'] = evalscript
        else:
            params['LAYERS'] = '1_TRUE_COLOR'
        # If not using custom evalscript (i.e. default TRUE_COLOR), make image opaque to avoid fully transparent tiles
        if not evalscript:
            params['TRANSPARENT'] = 'FALSE'
            params['BGCOLOR'] = 'FFFFFF'
        url = base + '?' + urlparse.urlencode(params)
        req = urlrequest.Request(url=url, headers={'Accept': 'image/png'})
        with urlrequest.urlopen(req) as resp:
            data = resp.read()
            # Basic sanity check: PNG magic header
            try:
                if not (isinstance(data, (bytes, bytearray)) and data.startswith(b"\x89PNG\r\n\x1a\n")):
                    raise ValueError("WMS response is not a PNG image")
            except Exception:
                raise
            return data

    async def download_sentinel_data(self, geometry_geojson, time_interval_days=(-30, 0), resolution=10, output_dir='.', date_iso: str | None = None, max_cloud: int | None = None, max_px: int | None = None):
        """
        Download Sentinel-2 data for the given GeoJSON geometry.

        Args:
            geometry_geojson: GeoJSON geometry (Polygon) in EPSG:4326
            time_interval_days: Tuple of (days_from, days_to) relative to current date
            resolution: Pixel resolution in meters
            output_dir: Directory to save the output

        Returns:
            dict containing paths and bounds for overlay
        """
        # Compute bounds (prefer Shapely if available)
        if SHAPELY_AVAILABLE and shp_shape is not None:
            geom = shp_shape(geometry_geojson)
            minx, miny, maxx, maxy = geom.bounds
        else:
            minx, miny, maxx, maxy = _bounds_from_geojson(geometry_geojson)

        # Compute output size proportionally to bbox extent, clamped by max_px
        dx = maxx - minx
        dy = maxy - miny
        limit = 2500 if (max_px is None or int(max_px) <= 0) else int(max_px)
        denom = max(dx, dy) if max(dx, dy) > 0 else 1.0
        width = max(1, int(round(limit * (dx / denom))))
        height = max(1, int(round(limit * (dy / denom))))

        # Define the evalscript for true color RGB + NDWI (single-sample, brightened)
        evalscript_truecolor = """
        //VERSION=3
        function setup() {
            return { input: ["B02","B03","B04","B08","dataMask"], output: { bands: 4 } };
        }
        function evaluatePixel(s) {
            let valid = s.dataMask === 1;
            let r = Math.min(1.0, Math.pow(s.B04 * 2.5, 1.0/1.2));
            let g = Math.min(1.0, Math.pow(s.B03 * 2.5, 1.0/1.2));
            let b = Math.min(1.0, Math.pow(s.B02 * 2.5, 1.0/1.2));
            return [r, g, b, valid ? 1 : 0];
        }
        """
        evalscript_overlay = """
        //VERSION=3
        function setup() {
            return { input: ["B03","B08","dataMask"], output: { bands: 4 } };
        }
        function evaluatePixel(s) {
            let ndwi = (s.B03 - 0.5 * s.B08) / (s.B03 + s.B08 + 0.0001);
            let valid = s.dataMask === 1;
            if (valid && ndwi > 0.2) {
                return [0.0, 0.588, 1.0, 1.0];
            }
            return [0.0, 0.0, 0.0, 0.0];
        }
        """

        token = self._get_token()
        acquisition_date = None
        rgb_png_path = None
        overlay_png_path = None
        if date_iso:
            try:
                base_dt = datetime.fromisoformat(date_iso[:10])
            except Exception:
                base_dt = datetime.utcnow()
            window_days_candidates = [15, 30, 60]
            cloud_candidates = []
            if max_cloud is not None:
                try:
                    cloud_candidates.append(int(max_cloud))
                except Exception:
                    pass
            cloud_candidates += [80, 100]
            seen = set()
            cc_list = []
            for v in cloud_candidates:
                if v is None:
                    continue
                vv = max(0, min(100, int(v)))
                if vv not in seen:
                    seen.add(vv)
                    cc_list.append(vv)
            for wd in window_days_candidates:
                start_dt = base_dt - timedelta(days=wd)
                t_from = start_dt.strftime('%Y-%m-%d')
                t_to = base_dt.strftime('%Y-%m-%d')
                tries = cc_list if cc_list else [100]
                for cc in tries:
                    got = False
                    # Try Process API if token available
                    if token:
                        try:
                            img = self._process_png(token, (minx, miny, maxx, maxy), width, height, evalscript_truecolor, t_from, t_to, cc)
                            os.makedirs(output_dir, exist_ok=True)
                            rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
                            with open(rgb_png_path, 'wb') as f:
                                f.write(img)
                            # Try overlay via Process API; failures are non-fatal
                            try:
                                ov = self._process_png(token, (minx, miny, maxx, maxy), width, height, evalscript_overlay, t_from, t_to, cc)
                                overlay_png_path = os.path.join(output_dir, 'water_mask.png')
                                with open(overlay_png_path, 'wb') as f:
                                    f.write(ov)
                            except Exception:
                                overlay_png_path = None
                            acquisition_date = t_to
                            got = True
                        except Exception:
                            got = False
                    # Fallback: public WMS GetMap for true color (default layer)
                    if not got:
                        try:
                            img = self._process_wms_png((minx, miny, maxx, maxy), width, height, t_from, t_to, cc, None)
                            os.makedirs(output_dir, exist_ok=True)
                            rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
                            with open(rgb_png_path, 'wb') as f:
                                f.write(img)
                            # Sanity check: ensure file is not trivially small
                            try:
                                if os.path.getsize(rgb_png_path) < 200:
                                    rgb_png_path = None
                                    raise ValueError('RGB WMS produced too small image')
                            except Exception:
                                raise
                            try:
                                ov = self._process_wms_png((minx, miny, maxx, maxy), width, height, t_from, t_to, cc, evalscript_overlay)
                                overlay_png_path = os.path.join(output_dir, 'water_mask.png')
                                with open(overlay_png_path, 'wb') as f:
                                    f.write(ov)
                                try:
                                    if os.path.getsize(overlay_png_path) < 200:
                                        overlay_png_path = None
                                except Exception:
                                    overlay_png_path = None
                            except Exception:
                                overlay_png_path = None
                            acquisition_date = t_to
                            got = True
                        except Exception:
                            got = False
                    if got:
                        break
                if rgb_png_path:
                    break
        else:
            end_date = datetime.utcnow()
            window_days_candidates = [30, 60]
            cloud_candidates = []
            if max_cloud is not None:
                try:
                    cloud_candidates.append(int(max_cloud))
                except Exception:
                    pass
            cloud_candidates += [80, 100]
            seen = set()
            cc_list = []
            for v in cloud_candidates:
                if v is None:
                    continue
                vv = max(0, min(100, int(v)))
                if vv not in seen:
                    seen.add(vv)
                    cc_list.append(vv)
            for wd in window_days_candidates:
                start_date = end_date - timedelta(days=wd)
                t_from = start_date.strftime('%Y-%m-%d')
                t_to = end_date.strftime('%Y-%m-%d')
                tries = cc_list if cc_list else [100]
                for cc in tries:
                    got = False
                    if token:
                        try:
                            img = self._process_png(token, (minx, miny, maxx, maxy), width, height, evalscript_truecolor, t_from, t_to, cc)
                            os.makedirs(output_dir, exist_ok=True)
                            rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
                            with open(rgb_png_path, 'wb') as f:
                                f.write(img)
                            try:
                                ov = self._process_png(token, (minx, miny, maxx, maxy), width, height, evalscript_overlay, t_from, t_to, cc)
                                overlay_png_path = os.path.join(output_dir, 'water_mask.png')
                                with open(overlay_png_path, 'wb') as f:
                                    f.write(ov)
                            except Exception:
                                overlay_png_path = None
                            acquisition_date = t_to
                            got = True
                        except Exception:
                            got = False
                    if not got:
                        try:
                            img = self._process_wms_png((minx, miny, maxx, maxy), width, height, t_from, t_to, cc, evalscript_truecolor)
                            os.makedirs(output_dir, exist_ok=True)
                            rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
                            with open(rgb_png_path, 'wb') as f:
                                f.write(img)
                            try:
                                ov = self._process_wms_png((minx, miny, maxx, maxy), width, height, t_from, t_to, cc, evalscript_overlay)
                                overlay_png_path = os.path.join(output_dir, 'water_mask.png')
                                with open(overlay_png_path, 'wb') as f:
                                    f.write(ov)
                            except Exception:
                                overlay_png_path = None
                            acquisition_date = t_to
                            got = True
                        except Exception:
                            got = False
                    if got:
                        break
                if rgb_png_path:
                    break

        if not rgb_png_path:
            raise ValueError("No imagery found for the selected time window and cloud coverage filter.")

        try:
            os.makedirs(output_dir, exist_ok=True)
            rgb_jpg_path = None
            rgb_tif_path = None
            ndwi_tif_path = None
            ndwi_npy_path = None
            bounds = [[miny, minx], [maxy, maxx]]

            return {
                'rgb_tif_path': rgb_tif_path,
                'ndwi_tif_path': ndwi_tif_path,
                'ndwi_npy_path': ndwi_npy_path,
                'rgb_png_path': rgb_png_path,
                'rgb_jpg_path': rgb_jpg_path,
                'bounds': bounds,
                'acquisition_date': acquisition_date,
                'overlay_png_path': overlay_png_path,
            }

        except Exception as e:
            logger.exception(f"Error downloading Sentinel-2 data: {str(e)}")
            raise
