import React from 'react';

const Topbar = ({ onResetView, backendStatus = 'checking', onRetryHealth, onStartDraw, onToggleSidebar }) => {
  const status = backendStatus;
  return (
    <div className="topbar h-12 flex items-center justify-between px-4 border-b border-slate-200 bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow" style={{ background: 'linear-gradient(90deg, #0891b2, #2563eb)' }}>
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="px-2 py-1 rounded-md bg-white/10 hover:bg-white/20 text-white text-sm"
          title="Toggle sidebar"
        >
          ☰
        </button>
        <div className="h-6 w-6 rounded bg-white/20 flex items-center justify-center font-bold">W</div>
        <div className="font-semibold">Waterbody Segmentation</div>
      </div>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-xs">
          {status === 'online' && (
            <span className="inline-flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-emerald-300"></span>
              <span className="text-white/90">API online</span>
            </span>
          )}
          {status === 'offline' && (
            <span className="inline-flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-red-300"></span>
              <span className="text-white/90">API offline</span>
              <button type="button" onClick={onRetryHealth} className="px-2 py-1 rounded bg-white/20 hover:bg-white/30">Retry</button>
            </span>
          )}
          {status === 'checking' && (
            <span className="inline-flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-white/50 animate-pulse"></span>
              <span className="text-white/90">Checking API…</span>
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={onStartDraw}
          className="btn btn-ghost px-3 py-1.5 rounded-md bg-white/10 hover:bg-white/20 text-white text-sm"
          title="Draw polygon"
        >
          Draw ROI
        </button>
        <button
          type="button"
          onClick={onResetView}
          className="btn btn-ghost px-3 py-1.5 rounded-md bg-white/20 hover:bg-white/30 text-white text-sm"
          title="Reset map view"
        >
          Reset view
        </button>
        <a
          href="#"
          className="btn btn-ghost px-3 py-1.5 rounded-md bg-white/10 hover:bg-white/20 text-white text-sm"
          title="Help"
          onClick={(e) => { e.preventDefault(); alert('Draw a polygon using the polygon tool, ensure at least 4 points, then click Analyze. Toggle overlays and adjust opacity from the sidebar.'); }}
        >
          Help
        </a>
      </div>
    </div>
  );
};

export default Topbar;
