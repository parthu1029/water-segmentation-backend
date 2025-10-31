import React, { useRef, useState, useEffect } from 'react';
import ErrorBoundary from './components/ErrorBoundary';
import MapComponent from './components/MapComponent.jsx';
import Sidebar from './components/Sidebar.jsx';
import Topbar from './components/Topbar.jsx';
import './index.css';

function App() {
  const mapRef = useRef(null);
  const [drawnPolygon, setDrawnPolygon] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [maskOpacity, setMaskOpacity] = useState(0.6);
  const [resetSignal, setResetSignal] = useState(0);
  const [showRgbOverlay, setShowRgbOverlay] = useState(true);
  const [showMaskOverlay, setShowMaskOverlay] = useState(true);
  const [selectedDate, setSelectedDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [maxCloud, setMaxCloud] = useState(30);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const mapActionsRef = useRef({ startPolygonDraw: null, resetView: null });
  const [backendStatus, setBackendStatus] = useState('checking');
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState(null);
  const [isDownloadingJPG, setIsDownloadingJPG] = useState(false);

  useEffect(() => {
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  const handleSaveROI = async () => {
    if (!drawnPolygon) return;
    setIsSaving(true);
    setSaveMessage(null);
    try {
      const geojson = drawnPolygon.toGeoJSON();
      const resp = await fetch('http://localhost:8000/api/v1/roi/store-geo-cordinate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(geojson),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with status ${resp.status}`);
      }
      const data = await resp.json();
      setSaveMessage(`ROI saved (ID: ${data.roi_id || 'ok'})`);
    } catch (e) {
      setSaveMessage(e.message || 'Failed to save ROI');
    } finally {
      setIsSaving(false);
    }
  };

  // When date changes, if we already have an ROI, auto re-run analyze for that date
  useEffect(() => {
    if (drawnPolygon && !isLoading) {
      handleAnalyze(selectedDate);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDate]);

  // When cloud coverage changes, if ROI exists, auto re-run analyze
  useEffect(() => {
    if (drawnPolygon && !isLoading) {
      handleAnalyze(selectedDate);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [maxCloud]);

  const handleStartDraw = () => {
    const fn = mapActionsRef.current?.startPolygonDraw;
    if (typeof fn === 'function') fn();
  };

  const handleResetView = () => {
    const fn = mapActionsRef.current?.resetView;
    if (typeof fn === 'function') fn();
  };

  const checkHealth = async () => {
    try {
      setBackendStatus('checking');
      const r = await fetch('http://localhost:8000/api/v1/health');
      if (!r.ok) throw new Error('offline');
      const j = await r.json();
      setBackendStatus(j.status === 'healthy' ? 'online' : 'offline');
    } catch {
      setBackendStatus('offline');
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const handlePolygonDrawn = (layer) => {
    setDrawnPolygon(layer);
  };

  const handleAnalyze = async (dateOverride) => {
    if (!drawnPolygon) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const geojson = drawnPolygon.toGeoJSON();
      const body = { ...geojson, date: (dateOverride || selectedDate), max_cloud: maxCloud };
      const resp = await fetch('http://localhost:8000/api/v1/segmentation/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with status ${resp.status}`);
      }
      const data = await resp.json();
      setResult(data.results);
    } catch (e) {
      console.error(e);
      setError(e.message || 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadJPG = async () => {
    if (!drawnPolygon) return;
    setIsDownloadingJPG(true);
    try {
      let jpgUrl = result?.rgb_jpg_url;
      if (!jpgUrl) {
        const geojson = drawnPolygon.toGeoJSON();
        const body = { ...geojson, date: selectedDate, max_cloud: maxCloud };
        const resp = await fetch('http://localhost:8000/api/v1/segmentation/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.detail || `Request failed with status ${resp.status}`);
        }
        const data = await resp.json();
        jpgUrl = data?.results?.rgb_jpg_url;
      }
      if (jpgUrl) {
        const abs = jpgUrl.startsWith('http') ? jpgUrl : `http://localhost:8000${jpgUrl}`;
        const a = document.createElement('a');
        a.href = abs;
        a.download = 'selected_region.jpg';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      } else {
        throw new Error('JPG not available');
      }
    } catch (e) {
      setError(e.message || 'Failed to download JPG');
    } finally {
      setIsDownloadingJPG(false);
    }
  };

  const handleClear = () => {
    if (drawnPolygon && mapRef.current) {
      mapRef.current.removeLayer(drawnPolygon);
    }
    setDrawnPolygon(null);
    setResult(null);
    setError(null);
    setResetSignal((x) => x + 1);
  };

  return (
    <div
      className="h-screen w-screen flex flex-col overflow-hidden bg-gradient-to-br from-slate-50 to-slate-100"
      style={{ height: '100vh', width: '100vw', display: 'flex', overflow: 'hidden' }}
    >
      <Topbar
        onResetView={handleResetView}
        backendStatus={backendStatus}
        onRetryHealth={checkHealth}
        onStartDraw={handleStartDraw}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
      />
      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          collapsed={!sidebarOpen}
          onToggleCollapse={() => setSidebarOpen((v) => !v)}
          selectedDate={selectedDate}
          onDateChange={setSelectedDate}
          maxCloud={maxCloud}
          onMaxCloudChange={setMaxCloud}
          onAnalyze={handleAnalyze}
          onClear={handleClear}
          result={result}
          isLoading={isLoading}
          error={error}
          hasPolygon={!!drawnPolygon}
          maskOpacity={maskOpacity}
          setMaskOpacity={setMaskOpacity}
          showRgbOverlay={showRgbOverlay}
          setShowRgbOverlay={setShowRgbOverlay}
          showMaskOverlay={showMaskOverlay}
          setShowMaskOverlay={setShowMaskOverlay}
          onStartDraw={handleStartDraw}
          onSaveROI={handleSaveROI}
          isSaving={isSaving}
          saveMessage={saveMessage}
          onDownloadJPG={handleDownloadJPG}
          isDownloadingJPG={isDownloadingJPG}
        />
        <div
          className="flex-1 relative min-h-0"
          style={{ flex: 1, position: 'relative', minHeight: 0, minWidth: 0 }}
        >
          {isLoading && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-100/60 backdrop-blur-sm">
              <div className="animate-spin rounded-full h-10 w-10 border-4 border-cyan-600 border-t-transparent" />
            </div>
          )}
          <ErrorBoundary>
            <MapComponent
              onMapLoad={(map) => { mapRef.current = map; }}
              onPolygonDrawn={handlePolygonDrawn}
              result={result}
              maskOpacity={maskOpacity}
              resetSignal={resetSignal}
              showRgbOverlay={showRgbOverlay}
              showMaskOverlay={showMaskOverlay}
              selectedDate={selectedDate}
              maxCloud={maxCloud}
              onRegisterActions={(actions) => { mapActionsRef.current = actions || {}; }}
            />
          </ErrorBoundary>
          {!!drawnPolygon && (
            <div className="absolute right-4 top-24 flex flex-col items-end gap-2" style={{ zIndex: 1500 }}>
              <button
                type="button"
                onClick={handleDownloadJPG}
                disabled={isDownloadingJPG}
                className={`px-3 py-2 rounded-md text-white shadow ${!isDownloadingJPG ? 'bg-[#16a34a] hover:bg-[#15803d]' : 'bg-slate-400 cursor-not-allowed'}`}
              >
                {isDownloadingJPG ? 'Downloading…' : 'Download JPG'}
              </button>
              <button
                type="button"
                onClick={handleSaveROI}
                disabled={isSaving}
                className={`px-3 py-2 rounded-md text-white shadow ${!isSaving ? 'bg-[#0077b6] hover:bg-[#00679f]' : 'bg-slate-400 cursor-not-allowed'}`}
              >
                {isSaving ? 'Saving…' : 'Save ROI'}
              </button>
              {saveMessage && (
                <div className="text-xs bg-white border border-slate-200 rounded px-2 py-1 text-slate-700 shadow">
                  {saveMessage}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
