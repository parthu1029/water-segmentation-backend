import os
import io
import numpy as np
import rasterio
from rasterio.transform import from_bounds
try:
    from sentinelhub import (
        CRS,
        BBox,
        DataCollection,
        MimeType,
        MosaickingOrder,
        SentinelHubRequest,
        bbox_to_dimensions,
        SHConfig
    )
    SH_AVAILABLE = True
except Exception:
    CRS = BBox = DataCollection = MimeType = MosaickingOrder = SentinelHubRequest = bbox_to_dimensions = SHConfig = None
    SH_AVAILABLE = False
from shapely.geometry import shape as shp_shape
from datetime import datetime, timedelta
from PIL import Image
import urllib.request as urlrequest
import urllib.parse as urlparse

class SentinelHubService:
    """Service for interacting with Sentinel Hub API"""

    def __init__(self):
        self.config = self._get_config()
        # Public WMS instance can be overridden by env
        self.wms_instance = os.getenv('SENTINEL_HUB_WMS_INSTANCE', 'ef5210ea-db25-42dc-afc1-b95646d1d02c')

    def _get_config(self):
        """Initialize Sentinel Hub configuration with credentials"""
        if not SH_AVAILABLE:
            return None
        config = SHConfig()
        config.sh_client_id = os.getenv('SENTINEL_HUB_CLIENT_ID', '')
        config.sh_client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET', '')

        if not config.sh_client_id or not config.sh_client_secret:
            print("Warning: Sentinel Hub credentials not found. Please set SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET environment variables.")

        return config

    def _wms_getmap_png(self, bbox: tuple, width: int, height: int, time_from: str, time_to: str, max_cloud: int | None, evalscript: str | None = None, transparent: bool = True) -> bytes:
        base = f"https://services.sentinel-hub.com/ogc/wms/{self.wms_instance}"
        params = {
            'REQUEST': 'GetMap',
            'FORMAT': 'image/png',
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
        if evalscript:
            params['EVALSCRIPT'] = evalscript
            params['TRANSPARENT'] = 'TRUE'
        else:
            params['LAYERS'] = '1_TRUE_COLOR'
            params['TRANSPARENT'] = 'FALSE'
            params['BGCOLOR'] = 'FFFFFF'
        url = base + '?' + urlparse.urlencode(params)
        req = urlrequest.Request(url=url, headers={'Accept': 'image/png'})
        with urlrequest.urlopen(req) as resp:
            data = resp.read()
            if not (isinstance(data, (bytes, bytearray)) and data.startswith(b"\x89PNG\r\n\x1a\n")):
                raise ValueError("WMS response is not a PNG image")
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
        # Convert geojson to shapely geometry
        geom = shp_shape(geometry_geojson)
        minx, miny, maxx, maxy = geom.bounds

        # BBox for SH will be created inside build_request if SH is available

        # Calculate image size in pixels (keep within max_px, preserving aspect ratio)
        dx = maxx - minx
        dy = maxy - miny
        limit = 2500 if (max_px is None or int(max_px) <= 0) else int(max_px)
        denom = max(dx, dy) if max(dx, dy) > 0 else 1.0
        width = max(1, int(round(limit * (dx / denom))))
        height = max(1, int(round(limit * (dy / denom))))
        size_px = (width, height)

        # Define the evalscript for true color RGB + NDWI
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
            // Calculate NDWI
            let ndwi = (sample.B03 - 0.5 * sample.B08) / (sample.B03 + sample.B08 + 0.0001);

            // Mask clouds and invalid using SCL (0/1/3/8/9/10/11) and dataMask
            let isCloud = (sample.SCL === 3) || (sample.SCL === 8) || (sample.SCL === 9) || (sample.SCL === 10) || (sample.SCL === 11);
            let isNoData = (sample.SCL === 0) || (sample.SCL === 1);
            let valid = (sample.dataMask === 1) && !(isCloud || isNoData);

            // Brighten RGB with gain and gamma
            let gain = 2.5;
            let gamma = 1.0/1.2;
            let r = Math.min(1.0, Math.pow(sample.B04 * gain, gamma));
            let g = Math.min(1.0, Math.pow(sample.B03 * gain, gamma));
            let b = Math.min(1.0, Math.pow(sample.B02 * gain, gamma));

            // Return RGB + NDWI + mask
            return {
                default: [r, g, b, valid ? 1 : 0],
                ndwi: [valid ? ndwi : -1]
            };
        }
        """

        # Helper to build request for a given time interval and cloud threshold
        def build_request(time_interval, cloud_value):
            max_cc_fraction = None
            if cloud_value is not None:
                try:
                    v = max(0, min(100, int(cloud_value)))
                    max_cc_fraction = v / 100.0
                except Exception:
                    max_cc_fraction = None
            bbox_sh = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)
            return SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                        other_args={
                            'dataFilter': {
                                'maxCloudCoverage': max_cc_fraction
                            }
                        } if (max_cc_fraction is not None) else None,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF),
                    SentinelHubRequest.output_response('ndwi', MimeType.TIFF)
                ],
                bbox=bbox_sh,
                size=size_px,
                config=self.config
            )

        acquisition_date = None
        data = None
        sh_ready = bool(SH_AVAILABLE and self.config and os.getenv('SENTINEL_HUB_CLIENT_ID') and os.getenv('SENTINEL_HUB_CLIENT_SECRET'))

        if date_iso and sh_ready:
            # Preferred end date is the provided date
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
            # Remove dups, clamp to 0..100 preserving order
            seen = set()
            cc_list = []
            for v in cloud_candidates:
                if v is None:
                    continue
                vv = max(0, min(100, int(v)))
                if vv not in seen:
                    seen.add(vv)
                    cc_list.append(vv)

            # Try progressively wider windows and looser clouds
            data = None
            for wd in window_days_candidates:
                start_dt = base_dt - timedelta(days=wd)
                time_interval_tuple = (start_dt.strftime('%Y-%m-%d'), base_dt.strftime('%Y-%m-%d'))
                # If user didn't supply a cloud limit, try with 100 only
                tries = cc_list if cc_list else [100]
                for cc in tries:
                    try:
                        request = build_request(time_interval_tuple, cc)
                        dd = request.get_data()
                        if dd and len(dd) >= 2 and dd[0] is not None and dd[1] is not None:
                            data = dd
                            acquisition_date = base_dt.strftime('%Y-%m-%d')
                            break
                    except Exception:
                        continue
                if data is not None:
                    break
        else:
            # No specific date: use now as end and try windows 30 then 60 days
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

            data = None
            for wd in window_days_candidates:
                start_date = end_date - timedelta(days=wd)
                time_interval_tuple = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                tries = cc_list if cc_list else [100]
                if sh_ready:
                    for cc in tries:
                        try:
                            request = build_request(time_interval_tuple, cc)
                            dd = request.get_data()
                            if dd and len(dd) >= 2 and dd[0] is not None and dd[1] is not None:
                                data = dd
                                acquisition_date = end_date.strftime('%Y-%m-%d')
                                break
                        except Exception:
                            continue
                if data is not None:
                    break

        # Validate data before proceeding
        if not data or len(data) < 2 or data[0] is None or data[1] is None:
            # WMS fallback: return opaque TRUE_COLOR PNG and transparent NDWI overlay if possible
            try:
                os.makedirs(output_dir, exist_ok=True)
                # Determine a reasonable time window around provided date or now
                if date_iso:
                    try:
                        end_dt = datetime.fromisoformat(date_iso[:10])
                    except Exception:
                        end_dt = datetime.utcnow()
                else:
                    end_dt = datetime.utcnow()
                start_dt = end_dt - timedelta(days=30)
                t_from = start_dt.strftime('%Y-%m-%d')
                t_to = end_dt.strftime('%Y-%m-%d')
                cc = max_cloud if max_cloud is not None else 100

                # RGB (opaque)
                rgb_bytes = self._wms_getmap_png((minx, miny, maxx, maxy), width, height, t_from, t_to, cc, evalscript=None, transparent=False)
                rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
                with open(rgb_png_path, 'wb') as f:
                    f.write(rgb_bytes)
                try:
                    if os.path.getsize(rgb_png_path) < 200:
                        raise ValueError('RGB WMS produced too small image')
                except Exception:
                    pass

                # Overlay via NDWI evalscript (transparent where not water)
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
                overlay_bytes = None
                try:
                    overlay_bytes = self._wms_getmap_png((minx, miny, maxx, maxy), width, height, t_from, t_to, cc, evalscript=evalscript_overlay, transparent=True)
                except Exception:
                    overlay_bytes = None
                overlay_png_path = None
                if overlay_bytes:
                    overlay_png_path = os.path.join(output_dir, 'water_mask.png')
                    with open(overlay_png_path, 'wb') as f:
                        f.write(overlay_bytes)
                    try:
                        if os.path.getsize(overlay_png_path) < 200:
                            overlay_png_path = None
                    except Exception:
                        overlay_png_path = None

                # Derive JPG from PNG for download convenience
                rgb_jpg_path = None
                try:
                    img = Image.open(io.BytesIO(rgb_bytes)).convert('RGB')
                    candidate = os.path.join(output_dir, 'sentinel_rgb.jpg')
                    img.save(candidate, format='JPEG', quality=95, subsampling=0, optimize=True)
                    rgb_jpg_path = candidate
                except Exception:
                    rgb_jpg_path = None

                bounds = [[miny, minx], [maxy, maxx]]
                acquisition_date = t_to
                return {
                    'rgb_tif_path': None,
                    'ndwi_tif_path': None,
                    'rgb_png_path': rgb_png_path,
                    'rgb_jpg_path': rgb_jpg_path,
                    'overlay_png_path': overlay_png_path,
                    'bounds': bounds,
                    'acquisition_date': acquisition_date,
                }
            except Exception as _wf:
                raise ValueError("No imagery found for the selected time window and cloud coverage filter.")


        try:
            os.makedirs(output_dir, exist_ok=True)

            # Decode TIFF bytes to arrays if needed
            if isinstance(data[0], (bytes, bytearray)):
                def _decode_tiff(b):
                    with rasterio.io.MemoryFile(b) as mem:
                        with mem.open() as ds:
                            arr = ds.read()  # (bands, H, W)
                            return np.moveaxis(arr, 0, -1)  # (H, W, bands)
                rgb_data = _decode_tiff(data[0])
                ndwi_data = _decode_tiff(data[1])
            else:
                rgb_data = data[0]  # H x W x 4 (R,G,B,mask) floats 0-1
                ndwi_data = data[1]  # H x W x 1 float NDWI

            # Write GeoTIFFs with georeferencing
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
                # R,G,B
                dst.write(rgb_data[:, :, 0], 1)
                dst.write(rgb_data[:, :, 1], 2)
                dst.write(rgb_data[:, :, 2], 3)
                # Alpha as 0..1, store as float32 as well
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

            # Save a PNG quicklook for overlay (scale 0-255, apply alpha)
            rgb_png_path = os.path.join(output_dir, 'sentinel_rgb.png')
            rgb_uint8 = np.clip(rgb_data[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
            alpha_uint8 = np.clip(rgb_data[:, :, 3] * 255.0, 0, 255).astype(np.uint8)
            rgba = np.dstack([rgb_uint8, alpha_uint8])
            Image.fromarray(rgba, mode='RGBA').save(rgb_png_path)

            rgb_jpg_path = None
            try:
                candidate = os.path.join(output_dir, 'sentinel_rgb.jpg')
                Image.fromarray(rgb_uint8, mode='RGB').save(candidate, format='JPEG', quality=95, subsampling=0, optimize=True)
                rgb_jpg_path = candidate
            except Exception:
                rgb_jpg_path = None

            bounds = [[miny, minx], [maxy, maxx]]  # for Leaflet imageOverlay

            return {
                'rgb_tif_path': rgb_tif_path,
                'ndwi_tif_path': ndwi_tif_path,
                'rgb_png_path': rgb_png_path,
                'rgb_jpg_path': rgb_jpg_path,
                'bounds': bounds,
                'acquisition_date': acquisition_date,
            }

        except Exception as e:
            print(f"Error downloading Sentinel-2 data: {str(e)}")
            raise
