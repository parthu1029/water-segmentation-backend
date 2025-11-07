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
    from sentinelhub import (
        CRS,
        BBox,
        DataCollection,
        MimeType,
        MosaickingOrder,
        SentinelHubRequest,
        bbox_to_dimensions,
        SHConfig,
    )
    SENTINELHUB_AVAILABLE = True
except Exception:
    SENTINELHUB_AVAILABLE = False
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
                data_filter['maxCloudCoverage'] = v
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
                        'dataFilter': data_filter,
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

    async def _download_via_sdk(self, geometry_geojson, time_interval_days=(-30, 0), resolution=10, output_dir='.', date_iso: str | None = None, max_cloud: int | None = None, max_px: int | None = None):
        if not SENTINELHUB_AVAILABLE:
            raise RuntimeError("sentinelhub SDK not available")
        # Bounds via shapely if available, else fallback
        if SHAPELY_AVAILABLE and shp_shape is not None:
            geom = shp_shape(geometry_geojson)
            minx, miny, maxx, maxy = geom.bounds
        else:
            minx, miny, maxx, maxy = _bounds_from_geojson(geometry_geojson)

        # Config with credentials
        config = SHConfig()
        config.sh_client_id = os.getenv('SENTINEL_HUB_CLIENT_ID', '')
        config.sh_client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET', '')

        bbox_sh = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)
        dims = bbox_to_dimensions(bbox_sh, resolution=resolution)
        limit = 2500 if (max_px is None or int(max_px) <= 0) else int(max_px)
        scale = (max(dims) / float(limit)) if max(dims) > limit else 1.0
        size_px = (
            max(1, int(round(dims[0] / scale))),
            max(1, int(round(dims[1] / scale)))
        )

        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B08", "SCL", "dataMask"],
                output: [
                    { id: "default", bands: 4, sampleType: "FLOAT32" },
                    { id: "ndwi", bands: 1, sampleType: "FLOAT32" }
                ]
            };
        }

        function evaluatePixel(sample) {
            let ndwi = (sample.B03 - 0.5 * sample.B08) / (sample.B03 + sample.B08 + 0.0001);
            let valid = sample.SCL !== 3 && sample.SCL !== 9 && sample.SCL !== 10 && sample.dataMask === 1;
            return {
                default: [sample.B04, sample.B03, sample.B02, valid ? 1 : 0],
                ndwi: [valid ? ndwi : -1]
            };
        }
        """

        evalscript_alt = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B08", "dataMask"],
                output: [
                    { id: "default", bands: 4, sampleType: "FLOAT32" },
                    { id: "ndwi", bands: 1, sampleType: "FLOAT32" }
                ]
            };
        }
        function evaluatePixel(s) {
            let ndwi = (s.B03 - 0.5 * s.B08) / (s.B03 + s.B08 + 0.0001);
            let r = 2.5 * s.B04;
            let g = 2.5 * s.B03;
            let b = 2.5 * s.B02;
            return { default: [r, g, b, s.dataMask], ndwi: [ndwi] };
        }
        """

        def build_request(time_interval, cloud_value):
            max_cc_fraction = None
            if cloud_value is not None:
                try:
                    v = max(0, min(100, int(cloud_value)))
                    max_cc_fraction = v / 100.0
                except Exception:
                    max_cc_fraction = None
            return SentinelHubRequest(
                evalscript=evalscript_alt,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                        other_args={
                            'dataFilter': { 'maxCloudCoverage': max_cc_fraction }
                        } if (max_cc_fraction is not None) else None,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF),
                    SentinelHubRequest.output_response('ndwi', MimeType.TIFF),
                ],
                bbox=bbox_sh,
                size=size_px,
                config=config,
            )

        def build_request_alt(time_interval, cloud_value):
            max_cc_fraction = None
            if cloud_value is not None:
                try:
                    v = max(0, min(100, int(cloud_value)))
                    max_cc_fraction = v / 100.0
                except Exception:
                    max_cc_fraction = None
            return SentinelHubRequest(
                evalscript=evalscript_alt,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                        other_args={
                            'dataFilter': { 'maxCloudCoverage': max_cc_fraction }
                        } if (max_cc_fraction is not None) else None,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF),
                    SentinelHubRequest.output_response('ndwi', MimeType.TIFF),
                ],
                bbox=bbox_sh,
                size=size_px,
                config=config,
            )

        acquisition_date = None
        data = None

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
                time_interval_tuple = (start_dt.strftime('%Y-%m-%d'), base_dt.strftime('%Y-%m-%d'))
                tries = cc_list if cc_list else [100]
                for cc in tries:
                    try:
                        req = build_request(time_interval_tuple, cc)
                        dd = req.get_data()
                        if dd and len(dd) >= 2 and dd[0] is not None and dd[1] is not None:
                            data = dd
                            acquisition_date = base_dt.strftime('%Y-%m-%d')
                            break
                    except Exception:
                        continue
                if data is not None:
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
                time_interval_tuple = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                tries = cc_list if cc_list else [100]
                for cc in tries:
                    try:
                        req = build_request(time_interval_tuple, cc)
                        dd = req.get_data()
                        if dd and len(dd) >= 2 and dd[0] is not None and dd[1] is not None:
                            data = dd
                            acquisition_date = end_date.strftime('%Y-%m-%d')
                            break
                    except Exception:
                        continue
                if data is not None:
                    break

        if not data or len(data) < 2 or data[0] is None or data[1] is None:
            raise ValueError("No imagery found for the selected time window and cloud coverage filter.")

        os.makedirs(output_dir, exist_ok=True)

        rgb_data = data[0]
        ndwi_data = data[1]  # H x W x 1

        try:
            const_alpha = float(np.mean((rgb_data[:, :, 3] > 0.01).astype(np.float32)))
        except Exception:
            const_alpha = 1.0
        if const_alpha < 0.01:
            try:
                if date_iso:
                    base_dt = datetime.fromisoformat(date_iso[:10]) if date_iso else datetime.utcnow()
                    start_dt = base_dt - timedelta(days=30)
                    ti = (start_dt.strftime('%Y-%m-%d'), base_dt.strftime('%Y-%m-%d'))
                else:
                    end_date = datetime.utcnow()
                    start_dt = end_date - timedelta(days=30)
                    ti = (start_dt.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                cc = max_cloud if (max_cloud is not None) else None
                req2 = build_request_alt(ti, cc)
                dd2 = req2.get_data()
                if dd2 and len(dd2) >= 2 and dd2[0] is not None and dd2[1] is not None:
                    rgb_data = dd2[0]
                    ndwi_data = dd2[1]
                    try:
                        acquisition_date = ti[1]
                    except Exception:
                        pass
            except Exception:
                pass

        # Prepare visualization RGB with contrast stretching (keep GeoTIFF data unmodified)
        rgb_vis = rgb_data.copy()
        alpha_coverage = None
        try:
            valid = (rgb_vis[:, :, 3] > 0.5)
            if np.any(valid):
                alpha_coverage = float(np.mean(valid))
                for i in range(3):
                    band = rgb_vis[:, :, i][valid]
                    if band.size > 0:
                        lo = float(np.percentile(band, 2))
                        hi = float(np.percentile(band, 98))
                        if hi > lo:
                            v = (rgb_vis[:, :, i] - lo) / (hi - lo)
                            rgb_vis[:, :, i] = np.clip(v, 0.0, 1.0)
        except Exception:
            pass

        # GeoTIFFs
        rgb_tif_path = None
        ndwi_tif_path = None
        ndwi_npy_path = None
        try:
            if RASTERIO_AVAILABLE and rasterio is not None and from_bounds is not None:
                transform = from_bounds(minx, miny, maxx, maxy, width=rgb_data.shape[1], height=rgb_data.shape[0])
                rgb_tif_path = os.path.join(output_dir, 'sentinel_rgb.tif')
                with rasterio.open(
                    rgb_tif_path,
                    'w',
                    driver='GTiff',
                    height=rgb_data.shape[0],
                    width=rgb_data.shape[1],
                    count=4,
                    dtype='float32',
                    crs='EPSG:4326',
                    transform=transform,
                ) as dst:
                    dst.write(rgb_data[:, :, 0], 1)
                    dst.write(rgb_data[:, :, 1], 2)
                    dst.write(rgb_data[:, :, 2], 3)
                    dst.write(rgb_data[:, :, 3], 4)
                ndwi_tif_path = os.path.join(output_dir, 'ndwi.tif')
                with rasterio.open(
                    ndwi_tif_path,
                    'w',
                    driver='GTiff',
                    height=ndwi_data.shape[0],
                    width=ndwi_data.shape[1],
                    count=1,
                    dtype='float32',
                    crs='EPSG:4326',
                    transform=transform,
                ) as dst:
                    dst.write(ndwi_data[:, :, 0], 1)
            else:
                # Fallback: save NDWI as NPY if GeoTIFF is not possible
                ndwi_npy_path = os.path.join(output_dir, 'ndwi.npy')
                np.save(ndwi_npy_path, ndwi_data[:, :, 0])
        except Exception:
            pass

        # PNG quicklook and JPG
        rgb_png_path = None
        rgb_jpg_path = None
        try:
            if PIL_AVAILABLE and Image is not None:
                rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
                rgb_uint8 = np.clip(rgb_vis[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
                alpha_uint8 = np.clip(rgb_vis[:, :, 3] * 255.0, 0, 255).astype(np.uint8)
                rgba = np.dstack([rgb_uint8, alpha_uint8])
                Image.fromarray(rgba, mode='RGBA').save(rgb_png_path)
                rgb_jpg_path = os.path.join(output_dir, 'sentinel_rgb.jpg')
                Image.fromarray(rgb_uint8, mode='RGB').save(rgb_jpg_path, format='JPEG', quality=95, subsampling=0, optimize=True)
        except Exception:
            pass

        # If PNG is still missing, attempt WMS fallback (no PIL required)
        if rgb_png_path is None or (alpha_coverage is not None and alpha_coverage < 0.05):
            try:
                # Choose a 15-day window ending on acquisition_date (or today)
                if acquisition_date:
                    try:
                        end_dt = datetime.fromisoformat(acquisition_date[:10])
                    except Exception:
                        end_dt = datetime.utcnow()
                else:
                    end_dt = datetime.utcnow()
                start_dt = end_dt - timedelta(days=15)
                t_from = start_dt.strftime('%Y-%m-%d')
                t_to = end_dt.strftime('%Y-%m-%d')
                cc = max_cloud if (max_cloud is not None) else None
                wms_png = self._process_wms_png((minx, miny, maxx, maxy), size_px[0], size_px[1], t_from, t_to, cc)
                rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
                with open(rgb_png_path, 'wb') as f:
                    f.write(wms_png)
            except Exception:
                pass

        bounds = [[miny, minx], [maxy, maxx]]
        return {
            'rgb_tif_path': rgb_tif_path,
            'ndwi_tif_path': ndwi_tif_path,
            'ndwi_npy_path': ndwi_npy_path,
            'rgb_png_path': rgb_png_path,
            'rgb_jpg_path': rgb_jpg_path,
            'bounds': bounds,
            'acquisition_date': acquisition_date,
        }

    def _process_wms_png(self, bbox: tuple, width: int, height: int, time_from: str, time_to: str, max_cloud: int | None):
        if urlrequest is None or urlparse is None:
            raise RuntimeError("urllib not available for HTTP requests")
        base = f"https://services.sentinel-hub.com/ogc/wms/{self.wms_instance}"
        params = {
            'REQUEST': 'GetMap',
            'LAYERS': '1_TRUE_COLOR',
            'FORMAT': 'image/png',
            'TRANSPARENT': 'FALSE',
            'CRS': 'EPSG:4326',
            'BBOX': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            'WIDTH': str(int(max(1, width))),
            'HEIGHT': str(int(max(1, height))),
            'TIME': f"{time_from}/{time_to}",
        }
        if max_cloud is not None:
            try:
                v = max(0, min(100, int(max_cloud)))
                params['MAXCC'] = str(v)
            except Exception:
                pass
        url = base + '?' + urlparse.urlencode(params)
        req = urlrequest.Request(url=url, headers={'Accept': 'image/png'})
        with urlrequest.urlopen(req) as resp:
            return resp.read()

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
        # Prefer SDK path identical to frontend backend
        if SENTINELHUB_AVAILABLE:
            try:
                return await self._download_via_sdk(geometry_geojson, time_interval_days, resolution, output_dir, date_iso, max_cloud, max_px)
            except Exception as _sdk_e:
                logger.warning("SentinelHub SDK path failed, falling back to HTTP/WMS: %s", _sdk_e)
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

        # Define the evalscript for true color RGB + NDWI
        evalscript_truecolor = """
        //VERSION=3
        function setup() {
            return { input: ["B02","B03","B04","dataMask"], output: { bands: 4 } };
        }
        function evaluatePixel(s) {
            // Approximate Sentinel Hub TRUE COLOR visualization by applying a simple gain
            let r = 2.5 * s.B04;
            let g = 2.5 * s.B03;
            let b = 2.5 * s.B02;
            return [r, g, b, s.dataMask];
        }
        """
        evalscript_overlay = """
        //VERSION=3
        function setup() {
            return { input: ["B03","B08","SCL","dataMask"], output: { bands: 4 } };
        }
        function evaluatePixel(s) {
            let ndwi = (s.B03 - 0.5 * s.B08) / (s.B03 + s.B08 + 0.0001);
            let valid = s.SCL !== 3 && s.SCL !== 9 && s.SCL !== 10 && s.dataMask === 1;
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
                    # Prefer: public WMS TRUE COLOR to match frontend appearance
                    try:
                        img = self._process_wms_png((minx, miny, maxx, maxy), width, height, t_from, t_to, cc)
                        os.makedirs(output_dir, exist_ok=True)
                        rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
                        with open(rgb_png_path, 'wb') as f:
                            f.write(img)
                        acquisition_date = t_to
                        got = True
                        # Optionally compute overlay via Process API (non-fatal)
                        if token:
                            try:
                                ov = self._process_png(token, (minx, miny, maxx, maxy), width, height, evalscript_overlay, t_from, t_to, cc)
                                overlay_png_path = os.path.join(output_dir, 'water_mask.png')
                                with open(overlay_png_path, 'wb') as f:
                                    f.write(ov)
                            except Exception:
                                pass
                    except Exception:
                        got = False
                    # Fallback: Process API if WMS fails
                    if not got and token:
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
                                pass
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
                    # Prefer WMS
                    try:
                        img = self._process_wms_png((minx, miny, maxx, maxy), width, height, t_from, t_to, cc)
                        os.makedirs(output_dir, exist_ok=True)
                        rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
                        with open(rgb_png_path, 'wb') as f:
                            f.write(img)
                        acquisition_date = t_to
                        got = True
                        if token:
                            try:
                                ov = self._process_png(token, (minx, miny, maxx, maxy), width, height, evalscript_overlay, t_from, t_to, cc)
                                overlay_png_path = os.path.join(output_dir, 'water_mask.png')
                                with open(overlay_png_path, 'wb') as f:
                                    f.write(ov)
                            except Exception:
                                pass
                    except Exception:
                        got = False
                    # Fallback: Process API
                    if not got and token:
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
                                pass
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
