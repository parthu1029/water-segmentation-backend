import React from 'react';

const Sidebar = ({
  collapsed,
  onToggleCollapse,
  onAnalyze,
  onClear,
  result,
  isLoading,
  error,
  hasPolygon,
  maskOpacity,
  setMaskOpacity,
  showRgbOverlay,
  setShowRgbOverlay,
  showMaskOverlay,
  setShowMaskOverlay,
  onStartDraw,
  selectedDate,
  onDateChange,
  maxCloud,
  onMaxCloudChange,
  onSaveROI,
  isSaving,
  saveMessage,
  onDownloadJPG,
  isDownloadingJPG,
}) => {
  const widthClass = collapsed ? 'w-20' : 'w-[380px]';
  return (
    <div
      className={`${widthClass} h-full bg-white/95 border-r border-slate-200 flex flex-col shadow-lg transition-all duration-300 ease-in-out`}
      style={{ background: '#fff', borderRight: '1px solid #e5e7eb', display: 'flex', flexDirection: 'column' }}
    >
      <div className="p-4 border-b border-slate-200 bg-gradient-to-r text-white shadow-sm" style={{ background: 'linear-gradient(90deg, #0077b6, #00b4d8)' }}>
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">Waterbody Segmentation</h1>
          <button
            type="button"
            onClick={onToggleCollapse}
            className="px-2 py-1 rounded bg-white/20 hover:bg-white/30 text-white text-xs"
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? '›' : '‹'}
          </button>
        </div>
        {!collapsed && (
          <p className="text-sm text-white/90 mt-1">Draw a polygon on the map, then analyze.</p>
        )}
      </div>

      {!collapsed && (
        <>
      <div className="p-4 space-y-4 overflow-y-auto">
        <div className="rounded-lg border border-cyan-200 bg-cyan-50 text-cyan-900 p-3 text-sm">
          <ol className="list-decimal pl-5 space-y-1">
            <li>Use the polygon tool on the map to draw your ROI (min 3 points).</li>
            <li>Click Analyze to fetch Sentinel-2 and run water detection.</li>
            <li>Use toggles and opacity to adjust overlays.</li>
          </ol>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-3">
          <label className="block text-sm font-medium text-slate-700 mb-1">Acquisition date</label>
          <input
            type="date"
            value={selectedDate || ''}
            onChange={(e) => onDateChange && onDateChange(e.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#00b4d8] focus:border-[#00b4d8]"
          />
          <p className="mt-1 text-xs text-slate-500">Pick a date to fetch imagery for that day. If no image is available, try nearby dates.</p>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-3">
          <label className="block text-sm font-medium text-slate-700">Max cloud coverage: {Math.round(maxCloud || 0)}%</label>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={maxCloud ?? 0}
            onChange={(e) => onMaxCloudChange && onMaxCloudChange(parseInt(e.target.value, 10))}
            className="w-full accent-[#00b4d8]"
          />
        </div>

        <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-3">
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onStartDraw}
              className="px-3 py-2 rounded-md border border-[#00b4d8] text-[#0077b6] hover:bg-cyan-50"
            >
              Draw ROI
            </button>
            <button
              type="button"
              onClick={onAnalyze}
              disabled={!hasPolygon || isLoading}
              className={`px-3 py-2 rounded-md text-white shadow ${hasPolygon && !isLoading ? 'bg-[#0077b6] hover:bg-[#00679f]' : 'bg-slate-400 cursor-not-allowed'}`}
            >
              {isLoading ? 'Analyzing...' : 'Analyze ROI'}
            </button>
            <button
              type="button"
              onClick={onDownloadJPG}
              disabled={!hasPolygon || isDownloadingJPG}
              className={`px-3 py-2 rounded-md text-white shadow ${hasPolygon && !isDownloadingJPG ? 'bg-[#16a34a] hover:bg-[#15803d]' : 'bg-slate-400 cursor-not-allowed'}`}
            >
              {isDownloadingJPG ? 'Downloading…' : 'Download JPG'}
            </button>
            <button
              type="button"
              onClick={onSaveROI}
              disabled={!hasPolygon || isSaving}
              className={`px-3 py-2 rounded-md text-white shadow ${hasPolygon && !isSaving ? 'bg-[#0ea5e9] hover:bg-[#0284c7]' : 'bg-slate-400 cursor-not-allowed'}`}
            >
              {isSaving ? 'Saving…' : 'Save ROI'}
            </button>
            <button
              type="button"
              onClick={onClear}
              className="px-3 py-2 rounded-md border border-slate-300 text-slate-700 hover:bg-slate-100"
            >
              Clear
            </button>
          </div>
          {saveMessage && (
            <div className="mt-2 text-xs text-slate-600">{saveMessage}</div>
          )}
        </div>

        <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-3">
          <div className="flex items-center gap-4 text-sm">
            <label className="flex items-center gap-2 text-slate-700">
              <input className="accent-[#00b4d8]" type="checkbox" checked={!!showRgbOverlay} onChange={(e) => setShowRgbOverlay(e.target.checked)} />
              True color base
            </label>
            <label className="flex items-center gap-2 text-slate-700">
              <input className="accent-[#00b4d8]" type="checkbox" checked={!!showMaskOverlay} onChange={(e) => setShowMaskOverlay(e.target.checked)} />
              Water mask
            </label>
          </div>
          <div className="mt-3">
            <label className="block text-sm font-medium text-slate-700">Mask opacity: {maskOpacity.toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={maskOpacity}
              onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
              className="w-full accent-[#00b4d8]"
            />
          </div>
        </div>

        {error && (
          <div className="text-sm text-red-600">{error}</div>
        )}
      </div>

      <div className="px-4 pb-4 overflow-y-auto" style={{ paddingLeft: 16, paddingRight: 16, paddingBottom: 16, overflowY: 'auto' }}>
        {result ? (
          <div className="space-y-4">
            <h2 className="text-sm font-semibold text-slate-700">Results</h2>
            <div className="grid grid-cols-2 gap-2 text-sm">
              {result.acquisition_date && (
                <>
                  <div className="text-slate-500">Acquisition date</div>
                  <div className="font-medium">{result.acquisition_date}</div>
                </>
              )}
              <div className="text-slate-500">Polygon area</div>
              <div className="font-medium">{result.total_area_km2?.toFixed(3)} km²</div>
              <div className="text-slate-500">Water area</div>
              <div className="font-medium">{result.water_area_km2?.toFixed(3)} km²</div>
              <div className="text-slate-500">Coverage</div>
              <div className="font-medium">{result.water_percentage?.toFixed(2)}%</div>
            </div>
              <div className="flex flex-col gap-2">
                {result.mask_url && (
                  <a className="text-cyan-700 hover:underline text-sm" href={`http://localhost:8000${result.mask_url}`} target="_blank" rel="noreferrer">Download Water Mask PNG</a>
                )}
                {result.rgb_jpg_url && (
                  <a className="text-cyan-700 hover:underline text-sm" href={`http://localhost:8000${result.rgb_jpg_url}`} target="_blank" rel="noreferrer">Download RGB JPG</a>
                )}
                {result.rgb_tif_url && (
                  <a className="text-cyan-700 hover:underline text-sm" href={`http://localhost:8000${result.rgb_tif_url}`} target="_blank" rel="noreferrer">Download RGB GeoTIFF</a>
                )}
                {result.ndwi_tif_url && (
                  <a className="text-cyan-700 hover:underline text-sm" href={`http://localhost:8000${result.ndwi_tif_url}`} target="_blank" rel="noreferrer">Download NDWI GeoTIFF</a>
                )}
              </div>
            </div>
        ) : (
          <div className="text-sm text-gray-500">No results yet. Draw a polygon and click Analyze.</div>
        )}
      </div>
      </>
      )}

      {!collapsed && (
        <div className="mt-auto p-4 text-xs text-slate-400 border-t border-slate-200" style={{ marginTop: 'auto', padding: 16, fontSize: 12, color: '#9ca3af', borderTop: '1px solid #e5e7eb' }}>
          Data: Sentinel‑2 • React + Leaflet • FastAPI
        </div>
      )}
    </div>
  );
};

export default Sidebar;
