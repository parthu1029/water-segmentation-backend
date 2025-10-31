import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';
import 'leaflet-draw';

const MapComponent = ({ onMapLoad, onPolygonDrawn, result, maskOpacity = 0.6, resetSignal, showRgbOverlay = true, showMaskOverlay = true, onRegisterActions, selectedDate, maxCloud = 30 }) => {
  const mapRef = useRef(null);
  const containerRef = useRef(null);
  const drawnItemsRef = useRef(null);
  const maskOverlayRef = useRef(null);
  const rgbOverlayRef = useRef(null);
  const drawControlRef = useRef(null);
  const sentinelLayerRef = useRef(null);
  const drawingPolygonRef = useRef(false);

  useEffect(() => {
    // Initialize the map only once
    if (!mapRef.current && containerRef.current) {
      const map = L.map(containerRef.current, {
        center: [20.5937, 78.9629], // India center
        zoom: 7,
        minZoom: 3,
        preferCanvas: true,
      });

      // Sentinel Hub WMS basemap
      const sentinelHubWMS = 'https://services.sentinel-hub.com/ogc/wms/ef5210ea-db25-42dc-afc1-b95646d1d02c';
      const fmt = (d) => d.toISOString().slice(0, 10);
      const end = selectedDate ? new Date(selectedDate) : new Date();
      const start = new Date(end);
      if (selectedDate) {
        start.setDate(end.getDate() - 30);
      } else {
        start.setDate(end.getDate() - 60);
      }
      const sentinelLayer = L.tileLayer.wms(sentinelHubWMS, {
        layers: '1_TRUE_COLOR',
        format: 'image/png',
        transparent: true,
        tileSize: 512,
        maxZoom: 19,
        minZoom: 7,
        time: `${fmt(start)}/${fmt(end)}`,
        MAXCC: typeof maxCloud === 'number' ? Math.max(0, Math.min(100, Math.round(maxCloud))) : 30,
        attribution: 'Imagery © Sentinel Hub',
      });
      sentinelLayerRef.current = sentinelLayer;
      // OpenStreetMap basemap for context (available via layer control)
      const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19,
      });
      const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Imagery © Esri',
        maxZoom: 19,
      });
      // Add OSM as base and Sentinel as overlay so transparent WMS areas show OSM beneath
      osm.addTo(map);
      osm.setZIndex(1);
      sentinelLayer.setZIndex(2);

      // Layer control: OSM as base, Sentinel as overlay
      const baseLayers = { 'Open Map': osm, 'Esri Map': esri };
      const overlays = { 'Sentinel-2': sentinelLayer };
      L.control.layers(baseLayers, overlays).addTo(map);


      // Feature group to store drawn polygon
      const drawnItems = new L.FeatureGroup();
      drawnItemsRef.current = drawnItems;
      map.addLayer(drawnItems);

      // Draw control
      const drawControl = new L.Control.Draw({
        edit: { featureGroup: drawnItems, remove: true, selectedPathOptions: { maintainColor: true } },
        draw: { polygon: true, polyline: false, rectangle: false, circle: false, marker: false, circlemarker: false },
      });
      map.addControl(drawControl);
      drawControlRef.current = drawControl;

      

      const getVertexCount = (layer) => {
        if (!layer || !layer.getLatLngs) return 0;
        const ll = layer.getLatLngs();
        // ll is [ring] for Polygon or [[ring1], [ring2], ...] for MultiPolygon
        if (Array.isArray(ll) && ll.length > 0) {
          if (Array.isArray(ll[0]) && ll[0].length && ll[0][0].lat !== undefined) {
            // Simple polygon: [LatLng, LatLng, ...]
            return ll[0].length;
          }
          if (Array.isArray(ll[0]) && Array.isArray(ll[0][0])) {
            // MultiPolygon-like nesting
            return ll.reduce((sum, ringGroup) => sum + (Array.isArray(ringGroup[0]) ? ringGroup[0].length : 0), 0);
          }
        }
        return 0;
      };

      const MIN_VERTICES = 3;

      // Handle draw created
      map.on(L.Draw.Event.CREATED, (e) => {
        const layer = e.layer;
        // Validate minimum vertices
        const v = getVertexCount(layer);
        if (v < MIN_VERTICES) {
          alert(`Polygon must have at least ${MIN_VERTICES} points. You drew ${v}.`);
          return;
        }
        // Only one polygon at a time
        drawnItems.clearLayers();
        drawnItems.addLayer(layer);

        if (onPolygonDrawn) {
          onPolygonDrawn(layer);
        }
      });

      // Handle edits
      map.on('draw:edited', (evt) => {
        if (evt && evt.layers) {
          evt.layers.eachLayer((layer) => {
            const v = getVertexCount(layer);
            if (v < MIN_VERTICES) {
              drawnItems.removeLayer(layer);
              alert(`Edited polygon must have at least ${MIN_VERTICES} points. It has ${v}.`);
            }
          });
        }
        const layers = drawnItems.getLayers();
        if (layers.length > 0 && onPolygonDrawn) {
          onPolygonDrawn(layers[0]);
        }
      });

      mapRef.current = map;
      if (onMapLoad) onMapLoad(map);

      // Expose actions to parent
      const startPolygonDraw = () => {
        if (!mapRef.current || !drawControlRef.current) return;
        const map = mapRef.current;
        const drawTb = drawControlRef.current._toolbars && drawControlRef.current._toolbars.draw;
        const polyMode = drawTb && drawTb._modes && drawTb._modes.polygon;
        if (polyMode && polyMode.handler) {
          polyMode.handler.enable();
          return;
        }
        const options = { ...(drawControlRef.current.options.draw.polygon || {}), repeatMode: false };
        const handler = new L.Draw.Polygon(map, options);
        handler.enable();
      };
      const resetView = () => {
        if (!mapRef.current) return;
        mapRef.current.setView([20.5937, 78.9629], 7);
      };
      if (onRegisterActions) onRegisterActions({ startPolygonDraw, resetView });
    }

    if (mapRef.current) {
      const map = mapRef.current;
      map.on(L.Draw.Event.DRAWSTART, (e) => {
        if (e && e.layerType === 'polygon') {
          drawingPolygonRef.current = true;
          const onKey = (ev) => {
            if (ev && ev.key === 'Enter') {
              ev.preventDefault();
              ev.stopPropagation();
            }
          };
          map._preventEnterHandler = onKey;
          window.addEventListener('keydown', onKey, true);
        }
      });
      map.on(L.Draw.Event.DRAWSTOP, () => {
        drawingPolygonRef.current = false;
        if (map._preventEnterHandler) {
          window.removeEventListener('keydown', map._preventEnterHandler, true);
          map._preventEnterHandler = null;
        }
      });
    }

    return () => {
      // Clean up (important for React StrictMode dev double-mount)
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
      if (onRegisterActions) onRegisterActions({ startPolygonDraw: null, resetView: null });
    };
  }, []);

  // Update Sentinel Hub WMS time parameter when selectedDate changes
  useEffect(() => {
    if (!sentinelLayerRef.current) return;
    const date = selectedDate ? new Date(selectedDate) : null;
    const fmt = (d) => d.toISOString().slice(0, 10);
    const time = date ? `${fmt(new Date(date.getTime() - 15*24*60*60*1000))}/${fmt(date)}` : undefined;
    if (time) {
      sentinelLayerRef.current.setParams({ time });
    }
  }, [selectedDate]);

  // Update MAXCC when cloud slider changes
  useEffect(() => {
    if (!sentinelLayerRef.current) return;
    const cc = typeof maxCloud === 'number' ? Math.max(0, Math.min(100, Math.round(maxCloud))) : undefined;
    if (cc !== undefined) {
      sentinelLayerRef.current.setParams({ MAXCC: cc });
    }
  }, [maxCloud]);

  // Invalidate size after mount and on window resize (fixes blank map in flex layouts)
  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    const handle = () => map.invalidateSize();
    const t = setTimeout(handle, 150);
    window.addEventListener('resize', handle);
    return () => {
      clearTimeout(t);
      window.removeEventListener('resize', handle);
    };
  }, []);

  // Clear overlays when resetSignal changes or result becomes null
  useEffect(() => {
    if (!mapRef.current) return;

    if (!result) {
      if (maskOverlayRef.current) {
        mapRef.current.removeLayer(maskOverlayRef.current);
        maskOverlayRef.current = null;
      }
      if (rgbOverlayRef.current) {
        mapRef.current.removeLayer(rgbOverlayRef.current);
        rgbOverlayRef.current = null;
      }
      return;
    }
  }, [resetSignal, result]);

  // Add/update overlays when result changes
  useEffect(() => {
    if (!mapRef.current || !result || !result.bounds) return;

    const map = mapRef.current;
    const boundsArr = result.bounds; // [[south, west],[north, east]]
    const bounds = L.latLngBounds([ [boundsArr[0][0], boundsArr[0][1]], [boundsArr[1][0], boundsArr[1][1]] ]);

    // Resolve URLs (prefix backend if needed)
    const ensureAbsolute = (url) => (url?.startsWith('http') ? url : `http://localhost:8000${url}`);

    // Add RGB base overlay if provided
    if (result.rgb_url && showRgbOverlay) {
      if (rgbOverlayRef.current) map.removeLayer(rgbOverlayRef.current);
      const rgbUrl = ensureAbsolute(result.rgb_url);
      rgbOverlayRef.current = L.imageOverlay(rgbUrl, bounds, { opacity: 1.0, interactive: false });
      rgbOverlayRef.current.addTo(map);
      // Fit map to bounds
      map.fitBounds(bounds, { padding: [20, 20] });
    } else if (rgbOverlayRef.current) {
      map.removeLayer(rgbOverlayRef.current);
      rgbOverlayRef.current = null;
    }

    // Add mask overlay
    if (result.mask_url && showMaskOverlay) {
      if (maskOverlayRef.current) map.removeLayer(maskOverlayRef.current);
      const maskUrl = ensureAbsolute(result.mask_url);
      maskOverlayRef.current = L.imageOverlay(maskUrl, bounds, { opacity: maskOpacity, interactive: false });
      maskOverlayRef.current.addTo(map);
    } else if (maskOverlayRef.current) {
      map.removeLayer(maskOverlayRef.current);
      maskOverlayRef.current = null;
    }
  }, [result, showRgbOverlay, showMaskOverlay]);

  // Update mask opacity when slider changes
  useEffect(() => {
    if (maskOverlayRef.current) {
      maskOverlayRef.current.setOpacity(maskOpacity);
    }
  }, [maskOpacity]);

  // If backend returned a specific acquisition_date, sync the WMS time to a 15-day window ending on it
  useEffect(() => {
    if (!sentinelLayerRef.current || !result || !result.acquisition_date) return;
    try {
      const d = new Date(result.acquisition_date);
      const fmt = (x) => x.toISOString().slice(0, 10);
      const start = new Date(d.getTime() - 15*24*60*60*1000);
      sentinelLayerRef.current.setParams({ time: `${fmt(start)}/${fmt(d)}` });
    } catch {}
  }, [result]);

  return (
    <div ref={containerRef} id="map" style={{ position: 'absolute', inset: 0 }} />
  );
};

export default MapComponent;
