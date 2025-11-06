import os
import numpy as np
try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except Exception:
    rasterio = None
    from_bounds = None
    RASTERIO_AVAILABLE = False
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
try:
    from shapely.geometry import shape as shp_shape
    SHAPELY_AVAILABLE = True
except Exception:
    shp_shape = None
    SHAPELY_AVAILABLE = False
from datetime import datetime, timedelta
from PIL import Image

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

class SentinelHubService:
    """Service for interacting with Sentinel Hub API"""

    def __init__(self):
        self.config = self._get_config()

    def _get_config(self):
        """Initialize Sentinel Hub configuration with credentials"""
        config = SHConfig()
        config.sh_client_id = os.getenv('SENTINEL_HUB_CLIENT_ID', '')
        config.sh_client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET', '')

        if not config.sh_client_id or not config.sh_client_secret:
            print("Warning: Sentinel Hub credentials not found. Please set SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET environment variables.")

        return config

    async def download_sentinel_data(self, geometry_geojson, time_interval_days=(-30, 0), resolution=10, output_dir='.', date_iso: str | None = None, max_cloud: int | None = None):
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

        # Convert to Sentinel Hub BBox
        bbox_sh = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)

        # Calculate image size in pixels
        dims = bbox_to_dimensions(bbox_sh, resolution=resolution)
        scale = (max(dims) / 2500.0) if max(dims) > 2500 else 1.0
        size_px = (max(1, int(round(dims[0] / scale))), max(1, int(round(dims[1] / scale))))

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

            // Mask clouds and cloud shadows using SCL
            let valid = sample.SCL !== 3 && sample.SCL !== 9 && sample.SCL !== 10 && sample.dataMask === 1;

            // Return RGB + NDWI + mask
            return {
                default: [sample.B04, sample.B03, sample.B02, valid ? 1 : 0],
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

        if date_iso:
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
            raise ValueError("No imagery found for the selected time window and cloud coverage filter.")


        try:
            os.makedirs(output_dir, exist_ok=True)

            rgb_data = data[0]  # H x W x 4 (R,G,B,mask) floats 0-1
            ndwi_data = data[1]  # H x W x 1 float NDWI

            rgb_tif_path = None
            ndwi_tif_path = None

            # Write GeoTIFFs with georeferencing if rasterio is available
            if RASTERIO_AVAILABLE and from_bounds is not None:
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

            rgb_jpg_path = os.path.join(output_dir, 'sentinel_rgb.jpg')
            Image.fromarray(rgb_uint8, mode='RGB').save(rgb_jpg_path, format='JPEG', quality=95, subsampling=0, optimize=True)

            # Always save NDWI as .npy for environments without rasterio
            ndwi_npy_path = os.path.join(output_dir, 'ndwi.npy')
            np.save(ndwi_npy_path, ndwi_data[:, :, 0])

            bounds = [[miny, minx], [maxy, maxx]]  # for Leaflet imageOverlay

            return {
                'rgb_tif_path': rgb_tif_path,
                'ndwi_tif_path': ndwi_tif_path,
                'ndwi_npy_path': ndwi_npy_path,
                'rgb_png_path': rgb_png_path,
                'rgb_jpg_path': rgb_jpg_path,
                'bounds': bounds,
                'acquisition_date': acquisition_date,
            }

        except Exception as e:
            print(f"Error downloading Sentinel-2 data: {str(e)}")
            raise
