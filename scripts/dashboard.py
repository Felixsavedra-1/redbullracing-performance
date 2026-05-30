import base64
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import text
from sqlalchemy.engine import Engine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analytics import ref_params, pit_stop_efficiency, dnf_rate_model, sector_deltas, tyre_degradation, pit_strategy

logger = logging.getLogger("f1_analytics")

_LOGO_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "redbullracinglogo.jpg")
)

_BG         = "#000000"
_BG_CARD    = "#0B0B0C"
_BG_HOVER   = "#141416"
_GRID       = "#1C1C1F"
_ZERO_LINE  = "#26262B"
_TICK       = "#6B7077"
_FONT_COLOR = "#FFFFFF"
_FONT       = "'Inter', 'Helvetica Neue', Arial, sans-serif"
_ACCENT     = "#1E41FF"
_ACCENT_DIM = "#2A2A30"
_STATUS_OK  = "#1E41FF"
_SPIKE      = "#1E41FF"

_DRIVER_COLORS = {
    "Verstappen": "#1E41FF",  # blue
    "Pérez":      "#FFFFFF",  # white
    "Tsunoda":    "#FF1800",  # red
    "Lawson":     "#FFDD00",  # yellow
}
_FALLBACK_COLORS = ["#1E41FF", "#FFFFFF", "#FF1800", "#FFDD00", "#888888"]


def _driver_color(name: str, idx: int) -> str:
    for surname, color in _DRIVER_COLORS.items():
        if surname in name:
            return color
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


# --------------------------------------------------------------------------- #
#  HTML template                                                                #
# --------------------------------------------------------------------------- #

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<title>PLACEHOLDER_TITLE · F1 Performance Analytics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<!-- Post-processing (r128 examples/js, pinned) — load order matters; failures are tolerated at runtime -->
<script src="https://unpkg.com/three@0.128.0/examples/js/shaders/CopyShader.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/shaders/LuminosityHighPassShader.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/postprocessing/ShaderPass.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/postprocessing/UnrealBloomPass.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/postprocessing/AfterimagePass.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/shaders/GammaCorrectionShader.js"></script>
<style>
:root{--bg:#000;--bg-card:#0B0B0C;--bg-hover:#141416;--accent:#1E41FF;--accent-2:#CC0000;--text:#FFFFFF;--dim:#9AA0A6;--border:#1C1C1F;--line:#26262B;--font:'Inter','Helvetica Neue','Helvetica',Arial,sans-serif}
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;color:#fff;font-family:var(--font);min-height:100vh;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.status-bar{display:flex;align-items:center;gap:22px;padding:9px 40px;background:#000;border-bottom:1px solid var(--border);font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:var(--dim);position:sticky;top:0;z-index:100;flex-wrap:wrap}
.sb-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);flex-shrink:0}
.sb-label{color:var(--dim)}
.sb-val{color:#fff;font-weight:500}
.sb-sep{color:#3A3A3F}
header{padding:48px 40px 34px;border-bottom:1px solid var(--border);background:#000}
.hd-team{font-size:.62rem;font-weight:600;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;margin-bottom:12px}
h1{font-size:2.4rem;font-weight:600;letter-spacing:-.02em;line-height:1.04}
h1 span.accent{color:var(--accent)}
.sub{color:var(--dim);font-size:.72rem;font-weight:500;letter-spacing:.16em;margin-top:14px;text-transform:uppercase}
.stats-row{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--border)}
.stat-card{padding:24px 26px;border-right:1px solid var(--border);transition:background .18s;position:relative}
.stat-card:last-child{border-right:none}
.stat-card:hover{background:var(--bg-card)}
.stat-card::after{content:'';position:absolute;bottom:8px;right:8px;width:6px;height:6px;border-color:var(--line);border-style:solid;border-width:0 1px 1px 0}
.stat-top{display:flex;align-items:baseline;gap:5px}
.stat-val{font-size:1.85rem;font-weight:600;color:#fff;letter-spacing:-.01em;line-height:1}
.stat-unit{font-size:.56rem;color:var(--dim);letter-spacing:.12em;text-transform:uppercase;align-self:flex-end;margin-bottom:3px}
.stat-lbl{font-size:.56rem;color:var(--dim);font-weight:500;letter-spacing:.16em;margin-top:9px;text-transform:uppercase}
.stat-status{display:flex;align-items:center;gap:6px;margin-top:7px}
.stat-dot{width:5px;height:5px;border-radius:50%;background:var(--accent)}
.stat-ok-text{font-size:.50rem;color:var(--dim);letter-spacing:.14em;text-transform:uppercase}
.car-viewer{padding:0;display:flex;justify-content:center;border-bottom:1px solid var(--border);background:radial-gradient(ellipse at 50% 42%,#141416 0%,#000 72%);cursor:pointer;position:relative}
.car-viewer::before{content:'RB \00B7 STUDIO RENDER';position:absolute;top:16px;left:24px;font-size:.56rem;font-weight:500;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;font-family:var(--font);z-index:2}
#f1car{display:block;width:100%;height:480px}
.race-cta{position:absolute;bottom:28px;left:50%;transform:translateX(-50%);display:inline-flex;align-items:center;gap:10px;font-family:var(--font);font-size:.74rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:#fff;background:var(--accent);border:1px solid var(--accent);border-radius:5px;padding:13px 26px;cursor:pointer;z-index:3;transition:transform .16s ease,background .16s ease,box-shadow .16s ease;box-shadow:0 6px 22px rgba(30,65,255,.28)}
.race-cta:hover{transform:translateX(-50%) translateY(-2px);background:#3358ff;box-shadow:0 10px 30px rgba(30,65,255,.42)}
.race-cta:active{transform:translateX(-50%) translateY(0)}
.race-cta .tri{width:0;height:0;border-style:solid;border-width:5px 0 5px 8px;border-color:transparent transparent transparent #fff}
.charts{padding:40px;display:grid;grid-template-columns:1fr;gap:40px}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:32px}
.chart-section{border-top:1px solid var(--border);padding-top:22px;position:relative}
.chart-section[data-section]::before{content:attr(data-section);position:absolute;top:-8px;left:0;font-size:.50rem;font-weight:500;letter-spacing:.20em;color:var(--dim);background:#000;padding-right:8px;font-family:var(--font);text-transform:uppercase}
.chart-label{font-size:.70rem;font-weight:600;letter-spacing:.16em;color:#fff;text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:10px}
.chart-label::before{content:'';width:8px;height:8px;background:var(--accent);border-radius:1px;flex-shrink:0}
.telemetry-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;padding-top:16px}
.telem-panel{border:1px solid var(--border);border-radius:6px;padding:16px 18px;position:relative;background:var(--bg-card)}
.telem-label{font-size:.56rem;font-weight:500;letter-spacing:.18em;color:var(--dim);text-transform:uppercase;margin-bottom:10px;font-family:var(--font)}
footer{padding:24px 40px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.ft-left{font-size:.58rem;color:var(--dim);letter-spacing:.12em;text-transform:uppercase}
.ft-right{font-size:.58rem;color:var(--dim);letter-spacing:.08em}
.ft-right span{color:var(--accent)}
@media(max-width:860px){.chart-row{grid-template-columns:1fr}.stats-row{grid-template-columns:1fr 1fr}.telemetry-row{grid-template-columns:1fr}}
@media(max-width:480px){h1{font-size:1.6rem}.stats-row{grid-template-columns:1fr}.charts{padding:24px}}
.logo-bar{display:flex;justify-content:center;padding:34px 0 18px}
.logo-img{height:150px;display:block;filter:invert(1)}
#game-overlay{display:none;position:fixed;inset:0;z-index:9999;background:#000}
#game-overlay.active{display:grid;grid-template-rows:1fr auto}
#game-canvas{width:100%;height:100%;display:block;outline:none;min-height:0}
#hud{display:flex;flex-direction:column;padding:0;font-family:'Space Mono','Courier New',monospace;font-size:13px;color:#E0F2FE;border-top:1px solid #00D4FF;flex-shrink:0;position:relative}
#hud-main{display:flex;align-items:center;gap:10px;padding:6px 14px;background:#030F1A}
#hud-sectors{display:flex;gap:16px;align-items:center;padding:2px 14px 4px;font-size:11px;background:#030F1A;border-top:1px solid #0A2035}
.hud-pos{color:#00D4FF;font-weight:700;font-size:17px;min-width:28px}
.hud-lap{color:#4A7FA5;min-width:72px}
.hud-timer{color:#E0F2FE;min-width:80px;font-weight:700}
.hud-speed{color:#E0F2FE;min-width:70px;font-weight:700}
.hud-gear-wrap{display:flex;flex-direction:column;align-items:center;min-width:52px}
.hud-gear{font-size:26px;font-weight:900;color:#E0F2FE;line-height:1;transition:color .05s}
.hud-gear.flash{color:#ff4400}
.hud-rpm-bar{width:48px;height:5px;background:#0A2035;border-radius:2px;margin-top:2px}
.hud-rpm-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,#00D4FF 0%,#00FF87 65%,#ff4400 100%);width:0%;transition:width .04s}
.hud-drs{color:#0A2035;border:1px solid #0A2035;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:700;letter-spacing:.12em;transition:color .15s,border-color .15s,text-shadow .15s}
.hud-drs.on{color:#00FF87;border-color:#00FF87;text-shadow:0 0 8px #00FF87}
.hud-tires{display:flex;align-items:center;gap:4px}
.hud-tire-label{color:#4A7FA5;font-size:10px;min-width:8px}
.hud-tire-wrap{width:52px;height:8px;background:#0A2035;border-radius:3px;overflow:hidden;border:1px solid #0F3050}
.hud-tire-bar{height:100%;width:100%;border-radius:3px;transition:width .1s,background .3s}
.hud-tire-temp{width:10px;height:10px;border-radius:50%;background:#4A7FA5;transition:background .4s;border:1px solid rgba(255,255,255,0.25)}
.hud-s1,.hud-s2,.hud-s3{color:#4A7FA5;min-width:88px;font-size:11px}
.hud-s1.purple,.hud-s2.purple,.hud-s3.purple{color:#cc00ff;font-weight:700}
.hud-s1.green,.hud-s2.green,.hud-s3.green{color:#00FF87;font-weight:700}
.hud-s1.yellow,.hud-s2.yellow,.hud-s3.yellow{color:#ffdd00}
.hud-delta{color:#4A7FA5;min-width:80px;font-size:11px}
.hud-delta.green{color:#00FF87}
.hud-delta.red{color:#ff4400}
.hud-msg{flex:1;text-align:center;color:#0A2035;font-size:10px;letter-spacing:.10em}
.hud-close{margin-left:auto;background:none;border:1px solid #CC0000;color:#CC0000;font-family:'Space Mono','Courier New',monospace;cursor:pointer;padding:3px 10px;font-size:12px;letter-spacing:.08em}
.hud-close:hover{background:#CC0000;color:#000}
#hud-minimap{position:absolute;bottom:8px;right:12px;border:1px solid #00D4FF;background:rgba(3,15,26,.80);border-radius:4px;pointer-events:none}
#hud-wheel{position:absolute;bottom:10px;right:182px;pointer-events:none}
#cockpit-wheel{display:none;position:fixed;left:50%;bottom:-34px;width:520px;margin-left:-260px;z-index:9998;pointer-events:none;filter:drop-shadow(0 -2px 30px rgba(0,0,0,.85))}
#cockpit-wheel.active{display:block}
.hud-drs.armed{color:#FFD500;border-color:#FFD500;text-shadow:0 0 8px #FFD500}
#hud-standings{position:fixed;top:12px;left:12px;width:188px;background:rgba(3,15,26,.86);border:1px solid #0A2035;border-radius:5px;pointer-events:none;overflow:hidden;z-index:10001;box-shadow:0 0 0 1px rgba(0,212,255,.10),0 8px 26px rgba(0,0,0,.55)}
#hud-standings .st-hd{display:flex;justify-content:space-between;align-items:center;padding:4px 8px;font-size:9px;letter-spacing:.22em;color:#00D4FF;background:linear-gradient(90deg,rgba(0,212,255,.18),rgba(0,212,255,0));border-bottom:1px solid #0A3550;text-transform:uppercase}
#hud-standings .st-hd b{color:#E0F2FE;font-weight:700;letter-spacing:.10em}
#hud-standings .st-row{display:flex;align-items:center;gap:6px;padding:1px 6px;font-size:10px;line-height:15px;letter-spacing:.04em;color:#9FC2D8;border-bottom:1px solid rgba(10,32,53,.5)}
#hud-standings .st-row.me{background:rgba(30,77,155,.45);color:#E0F2FE}
#hud-standings .st-pos{width:16px;color:#4A7FA5;text-align:right}
#hud-standings .st-chip{width:5px;height:10px;border-radius:1px;flex-shrink:0}
#hud-standings .st-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#hud-standings .st-gap{color:#4A7FA5;font-size:9px}
#podium-overlay{display:none;position:absolute;inset:0;z-index:10000;background:rgba(3,15,26,.93);flex-direction:column;align-items:center;justify-content:center;font-family:'Space Mono','Courier New',monospace;color:#E0F2FE}
#podium-overlay.active{display:flex}
#podium-overlay h2{font-size:1.4rem;letter-spacing:.25em;color:#00D4FF;margin-bottom:24px;text-transform:uppercase}
#podium-table{border-collapse:collapse;font-size:.85rem}
#podium-table td{padding:6px 20px;border-bottom:1px solid #0A2035}
#podium-table td:first-child{color:#4A7FA5;text-align:right}
#podium-table td:nth-child(2){color:#E0F2FE;font-weight:700}
#podium-table td:last-child{color:#4A7FA5;text-align:right}
.podium-p1 td{color:#ffd700 !important}
.podium-btn{margin-top:28px;background:none;border:1px solid #00D4FF;color:#00D4FF;font-family:'Space Mono','Courier New',monospace;cursor:pointer;padding:7px 22px;letter-spacing:.12em;font-size:.8rem}
.podium-btn:hover{background:#00D4FF;color:#030F1A}
#pause-overlay{display:none;position:absolute;inset:0;z-index:10000;background:rgba(3,15,26,.85);flex-direction:column;align-items:center;justify-content:center;font-family:'Space Mono','Courier New',monospace;color:#E0F2FE}
#pause-overlay.active{display:flex}
#pause-overlay h2{font-size:1.8rem;letter-spacing:.35em;color:#00D4FF;text-transform:uppercase;margin-bottom:14px}
#pause-overlay p{color:#4A7FA5;font-size:.72rem;letter-spacing:.22em;text-transform:uppercase}
#lights-bar{display:none;position:absolute;top:14px;left:50%;transform:translateX(-50%);z-index:20001;gap:10px;padding:10px 18px;background:rgba(3,15,26,.92);border:1px solid #330000;border-radius:8px;pointer-events:none}
#lights-bar.active{display:flex}
.light-bulb{width:30px;height:30px;border-radius:50%;background:#1a0000;border:2px solid #440000;transition:background .06s,box-shadow .06s}
.light-bulb.lit{background:#ff2200;border-color:#ff5500;box-shadow:0 0 14px #ff2200,0 0 32px #880000}
#go-flash{display:none;position:absolute;inset:0;background:rgba(255,255,255,.88);z-index:20002;pointer-events:none}
#grid-msg{display:none;position:absolute;bottom:120px;left:50%;transform:translateX(-50%);z-index:20001;color:#E0F2FE;font-family:'Space Mono','Courier New',monospace;font-size:.72rem;letter-spacing:.28em;text-transform:uppercase;text-align:center;text-shadow:0 0 8px #00D4FF;pointer-events:none}
#grid-msg.active{display:block}
</style>
</head>
<body>
<div class="status-bar">
  <div class="sb-dot"></div>
  <span><span class="sb-label">SYSTEM</span>&nbsp;<span class="sb-val">NOMINAL</span></span>
  <span class="sb-sep">&middot;</span>
  <span><span class="sb-label">MET</span>&nbsp;<span class="sb-val" id="met-clock">T+00:00:00</span></span>
  <span class="sb-sep">&middot;</span>
  <span><span class="sb-label">DRIVERS</span>&nbsp;<span class="sb-val">PLACEHOLDER_DRIVER_COUNT</span></span>
  <span class="sb-sep">&middot;</span>
  <span><span class="sb-label">DATA</span>&nbsp;<span class="sb-val">PLACEHOLDER_TS</span></span>
</div>
<div class="logo-bar">PLACEHOLDER_LOGO</div>
<header>
  <div class="hd-team">Oracle Red Bull Racing</div>
  <h1>Red Bull <span class="accent">F1</span> Analytics</h1>
  <p class="sub">PLACEHOLDER_SUBTITLE</p>
</header>
<div class="stats-row">PLACEHOLDER_STATS</div>
<div class="car-viewer">
  <canvas id="f1car"></canvas>
</div>
<div id="game-overlay">
  <canvas id="game-canvas" tabindex="0"></canvas>
  <div id="hud">
    <div id="hud-main">
      <span class="hud-pos">P4</span>
      <span class="hud-lap">LAP 1/3</span>
      <span class="hud-timer">0:00.000</span>
      <span class="hud-speed">0 km/h</span>
      <div class="hud-gear-wrap">
        <span class="hud-gear">N</span>
        <div class="hud-rpm-bar"><div class="hud-rpm-fill" id="rpm-fill"></div></div>
      </div>
      <span class="hud-drs">DRS</span>
      <div class="hud-tires">
        <span class="hud-tire-label">F</span>
        <div class="hud-tire-wrap"><div class="hud-tire-bar" id="tire-f"></div></div>
        <div class="hud-tire-temp" id="temp-f"></div>
        <span class="hud-tire-label">R</span>
        <div class="hud-tire-wrap"><div class="hud-tire-bar" id="tire-r"></div></div>
        <div class="hud-tire-temp" id="temp-r"></div>
      </div>
      <span class="hud-msg"></span>
      <button class="hud-close">&#x2715; ESC</button>
    </div>
    <div id="hud-sectors">
      <span class="hud-s1" id="hud-s1">S1 ---.---</span>
      <span class="hud-s2" id="hud-s2">S2 ---.---</span>
      <span class="hud-s3" id="hud-s3">S3 ---.---</span>
      <span class="hud-delta" id="hud-delta">&#916; ---</span>
      <span class="hud-msg" style="flex:1;font-size:10px;color:#444;letter-spacing:.10em">WASD/ARROWS · SHIFT=DRS · P=PIT · C=CAM · G=AUTO/MAN · Z/X=SHIFT · ESC=PAUSE</span>
    </div>
    <canvas id="hud-minimap" width="160" height="160"></canvas>
    <div id="hud-standings"></div>
    <svg id="hud-wheel" width="52" height="52" viewBox="-26 -26 52 52">
      <circle cx="0" cy="0" r="22" fill="none" stroke="#333" stroke-width="3"/>
      <line x1="-14" y1="0" x2="14" y2="0" stroke="#555" stroke-width="1.5"/>
      <line x1="0" y1="-14" x2="0" y2="14" stroke="#555" stroke-width="1.5"/>
      <g id="hud-wheel-ind">
        <line x1="-18" y1="0" x2="18" y2="0" stroke="#00D4FF" stroke-width="3" stroke-linecap="round"/>
      </g>
    </svg>
  </div>
  <div id="cockpit-wheel">
    <svg viewBox="-150 -90 300 165" width="520" height="286">
      <defs>
        <!-- carbon-fibre twill weave -->
        <pattern id="cwCarbon" width="9" height="9" patternUnits="userSpaceOnUse">
          <rect width="9" height="9" fill="#15181c"/>
          <path d="M0 0 L4.5 0 L0 4.5 Z M9 4.5 L9 9 L4.5 9 Z" fill="#21262d"/>
          <path d="M4.5 0 L9 0 L9 4.5 Z M0 4.5 L0 9 L4.5 9 Z" fill="#0d1014"/>
        </pattern>
        <linearGradient id="cwScreen" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#0c1c24"/><stop offset="1" stop-color="#04090c"/>
        </linearGradient>
        <radialGradient id="cwKnob" cx="0.5" cy="0.32" r="0.75">
          <stop offset="0" stop-color="#3a4047"/><stop offset="0.7" stop-color="#1a1d21"/><stop offset="1" stop-color="#070809"/>
        </radialGradient>
        <linearGradient id="cwGrip" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#24272b"/><stop offset="0.5" stop-color="#101316"/><stop offset="1" stop-color="#04060a"/>
        </linearGradient>
        <linearGradient id="cwBezel" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#2a2f35"/><stop offset="1" stop-color="#05070a"/>
        </linearGradient>
      </defs>
      <g id="cw-rot">
        <!-- shift paddle hints behind the body -->
        <path d="M -120 12 Q -150 16 -146 52 L -126 48 Q -122 28 -110 22 Z" fill="#0a0c0e" stroke="#000" stroke-width="1"/>
        <path d="M 120 12 Q 150 16 146 52 L 126 48 Q 122 28 110 22 Z" fill="#0a0c0e" stroke="#000" stroke-width="1"/>
        <!-- carbon body: wide modern F1 wheel — long flat top, short vertical sides, scooped centre between grip wings -->
        <path d="M -118 -78 Q -138 -78 -138 -58 L -138 6 Q -138 26 -120 36 L -92 50 Q -80 64 -58 60 Q -40 56 -38 28 Q -36 16 -18 16 L 18 16 Q 36 16 38 28 Q 40 56 58 60 Q 80 64 92 50 L 120 36 Q 138 26 138 6 L 138 -58 Q 138 -78 118 -78 Z" fill="url(#cwCarbon)" stroke="#04060a" stroke-width="3"/>
        <path d="M -118 -78 Q -138 -78 -138 -58 L -138 6 Q -138 26 -120 36 L -92 50 Q -80 64 -58 60 Q -40 56 -38 28 Q -36 16 -18 16 L 18 16 Q 36 16 38 28 Q 40 56 58 60 Q 80 64 92 50 L 120 36 Q 138 26 138 6 L 138 -58 Q 138 -78 118 -78 Z" fill="none" stroke="#1A3E82" stroke-width="1.4" opacity="0.85"/>
        <!-- contoured thumb grips on the vertical sides (navy Red Bull accent) -->
        <g>
          <rect x="-140" y="-48" width="26" height="74" rx="11" fill="url(#cwGrip)" stroke="#1A3E82" stroke-width="2.5"/>
          <rect x="114" y="-48" width="26" height="74" rx="11" fill="url(#cwGrip)" stroke="#1A3E82" stroke-width="2.5"/>
          <g stroke="#000" stroke-width="2" opacity="0.55">
            <line x1="-134" y1="-30" x2="-120" y2="-30"/><line x1="-134" y1="-12" x2="-120" y2="-12"/><line x1="-134" y1="6" x2="-120" y2="6"/><line x1="-134" y1="20" x2="-120" y2="20"/>
            <line x1="120" y1="-30" x2="134" y2="-30"/><line x1="120" y1="-12" x2="134" y2="-12"/><line x1="120" y1="6" x2="134" y2="6"/><line x1="120" y1="20" x2="134" y2="20"/>
          </g>
        </g>
        <!-- 12 o'clock dead-centre marker (yellow) -->
        <rect x="-7" y="-78" width="14" height="9" rx="2" fill="#FFC400"/>
        <!-- rev LED strip across the wide flat top -->
        <rect x="-92" y="-66" width="190" height="18" rx="5" fill="#04070a" stroke="#000" stroke-width="1.2"/>
        <g id="cw-leds"></g>
        <!-- central LCD with cyan bezel -->
        <rect x="-58" y="-42" width="116" height="56" rx="8" fill="url(#cwBezel)" stroke="#000" stroke-width="2"/>
        <rect x="-53" y="-37" width="106" height="46" rx="6" fill="url(#cwScreen)" stroke="#00D4FF" stroke-width="1" opacity="0.95"/>
        <text x="0" y="-29" text-anchor="middle" font-family="'Space Mono',monospace" font-size="6" letter-spacing="2.5" fill="#3d6a86">RED BULL RACING</text>
        <text id="cw-gear" x="0" y="4" text-anchor="middle" font-family="'Space Mono',monospace" font-size="32" font-weight="900" fill="#00D4FF">N</text>
        <!-- LCD readouts -->
        <text x="-47" y="-16" font-family="'Space Mono',monospace" font-size="6" fill="#2f5870">SPD</text>
        <text x="47" y="-16" text-anchor="end" font-family="'Space Mono',monospace" font-size="6" fill="#2f5870">LAP</text>
        <text x="-47" y="6" font-family="'Space Mono',monospace" font-size="6" fill="#7a5a10">DRS</text>
        <text x="47" y="6" text-anchor="end" font-family="'Space Mono',monospace" font-size="6" fill="#7a2010">ERS</text>
        <!-- button matrix on the flanks -->
        <g stroke="#000" stroke-width="1">
          <circle cx="-92" cy="-28" r="7.5" fill="#CC0000"/><circle cx="-92" cy="-6" r="7.5" fill="#1A3E82"/><circle cx="-92" cy="15" r="6.5" fill="#FFC400"/>
          <circle cx="92" cy="-28" r="7.5" fill="#13a05a"/><circle cx="92" cy="-6" r="7.5" fill="#c8ccd0"/><circle cx="92" cy="15" r="6.5" fill="#7a2fb0"/>
        </g>
        <g font-family="'Space Mono',monospace" font-size="5" fill="#9aa3ab" text-anchor="middle">
          <text x="-92" y="-26">1</text><text x="-92" y="-4">N</text><text x="92" y="-26">OK</text>
        </g>
        <!-- rotary dials on the lower flanks with labels -->
        <g font-family="'Space Mono',monospace" font-size="6" font-weight="700" text-anchor="middle">
          <circle cx="-72" cy="34" r="12" fill="url(#cwKnob)" stroke="#000" stroke-width="2"/>
          <line x1="-72" y1="34" x2="-72" y2="24" stroke="#FFC400" stroke-width="3" stroke-linecap="round"/>
          <text x="-72" y="52" fill="#FFC400">BB</text>
          <circle cx="72" cy="34" r="12" fill="url(#cwKnob)" stroke="#000" stroke-width="2"/>
          <line x1="72" y1="34" x2="79" y2="42" stroke="#CC0000" stroke-width="3" stroke-linecap="round"/>
          <text x="72" y="52" fill="#CC0000">DIFF</text>
        </g>
      </g>
    </svg>
  </div>
  <div id="lights-bar">
    <div class="light-bulb" id="lb0"></div>
    <div class="light-bulb" id="lb1"></div>
    <div class="light-bulb" id="lb2"></div>
    <div class="light-bulb" id="lb3"></div>
    <div class="light-bulb" id="lb4"></div>
  </div>
  <div id="go-flash"></div>
  <div id="grid-msg">PREPARE TO RACE &nbsp;·&nbsp; HOLD THROTTLE TO REV</div>
  <div id="podium-overlay">
    <h2>&#x1F3C6; Race Complete</h2>
    <table id="podium-table"></table>
    <button class="podium-btn" id="podium-close">RETURN TO DASHBOARD</button>
  </div>
  <div id="pause-overlay">
    <h2>PAUSED</h2>
    <p>ESC TO RESUME &nbsp;&#xB7;&nbsp; Q TO QUIT</p>
  </div>
</div>
<div class="charts">
  <div class="chart-section" data-section="SYS&middot;01">
    <div class="chart-label">Championship &middot; Points Trajectory</div>
    PLACEHOLDER_C1
  </div>
  <div class="chart-row">
    <div class="chart-section" data-section="SYS&middot;02">
      <div class="chart-label">Finish Positions &middot; Season</div>
      PLACEHOLDER_C2
    </div>
    <div class="chart-section" data-section="SYS&middot;03">
      <div class="chart-label">Driver Delta &middot; Points Gap</div>
      PLACEHOLDER_C3
    </div>
  </div>
  <div class="chart-section" data-section="SYS&middot;04">
    <div class="chart-label">Performance Matrix &middot; All Seasons</div>
    PLACEHOLDER_C4
  </div>
  <div class="chart-section" data-section="SYS&middot;05">
    <div class="chart-label">Pace &middot; Grid vs Finish</div>
    PLACEHOLDER_C5
  </div>
  <div class="chart-section" data-section="SYS&middot;06">
    <div class="chart-label">Telemetry &middot; Advanced Analytics</div>
    <div class="telemetry-row">
      <div class="telem-panel">
        <div class="telem-label">Pit Stop Efficiency &middot; Z-Score</div>
        PLACEHOLDER_C6
      </div>
      <div class="telem-panel">
        <div class="telem-label">Reliability Model &middot; DNF Rate</div>
        PLACEHOLDER_C7
      </div>
      <div class="telem-panel">
        <div class="telem-label">Sector Delta &middot; Green-Flag Laps</div>
        PLACEHOLDER_C8
      </div>
    </div>
  </div>
  <div class="chart-section" data-section="SYS&middot;07">
    <div class="chart-label">Lap Analysis &middot; FastF1</div>
    <div class="chart-row">
      <div>
        <div class="telem-label">Tyre Degradation &middot; Rate by Compound</div>
        PLACEHOLDER_C9
      </div>
      <div>
        <div class="telem-label">Race Strategy &middot; Stint Structure</div>
        PLACEHOLDER_C10
      </div>
    </div>
  </div>
</div>
<footer>
  <div class="ft-left">Oracle Red Bull Racing &middot; Performance Analytics</div>
  <div class="ft-right">Generated <span>PLACEHOLDER_TS</span> &nbsp;&middot;&nbsp; Oracle Red Bull Racing</div>
</footer>
<script>
(function(){
  var c=document.getElementById('f1car');
  if(!c||typeof THREE==='undefined') return;
  var H=480,W=c.parentElement.offsetWidth||900;
  c.width=W; c.height=H;

  // Renderer — PBR pipeline, ACES filmic, PCFSoft shadows
  var renderer=new THREE.WebGLRenderer({canvas:c,antialias:true,alpha:true});
  renderer.setSize(W,H); renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.shadowMap.enabled=true;
  renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.outputEncoding=THREE.sRGBEncoding;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure=1.28;
  renderer.physicallyCorrectLights=true;

  var scene=new THREE.Scene();
  var cam=new THREE.PerspectiveCamera(30,W/H,0.1,100);
  cam.position.set(5.4,1.75,4.5); cam.lookAt(0,-0.04,0);

  // Procedural studio environment map — bakes key/fill/rim into IBL
  (function(){
    var pmrem=new THREE.PMREMGenerator(renderer);
    pmrem.compileEquirectangularShader();
    var envMat=new THREE.ShaderMaterial({
      side:THREE.BackSide,
      vertexShader:'varying vec3 vN;void main(){vN=normalize(position);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
      fragmentShader:'varying vec3 vN;void main(){vec3 n=normalize(vN);vec3 col=mix(vec3(0.012,0.012,0.018),vec3(0.16,0.20,0.28),smoothstep(-0.4,0.7,n.y));float k=pow(max(0.,dot(n,normalize(vec3(0.50,0.78,0.35)))),10.);col+=k*vec3(1.00,0.95,0.86)*2.8;float f=pow(max(0.,dot(n,normalize(vec3(-0.72,0.18,-0.48)))),5.);col+=f*vec3(0.38,0.48,0.70)*0.58;float r=pow(max(0.,dot(n,normalize(vec3(-0.22,0.28,-0.94)))),7.);col+=r*vec3(0.50,0.62,1.00)*0.42;float fl=pow(max(0.,dot(n,vec3(0.,-1.,0.))),2.);col+=fl*vec3(0.08,0.06,0.04)*0.18;gl_FragColor=vec4(col,1.);}'
    });
    var envMesh=new THREE.Mesh(new THREE.SphereGeometry(50,32,16),envMat);
    var es=new THREE.Scene(); es.add(envMesh);
    var cubeRT=new THREE.WebGLCubeRenderTarget(512,{format:THREE.RGBFormat,generateMipmaps:true,minFilter:THREE.LinearMipmapLinearFilter});
    var cc=new THREE.CubeCamera(1,100,cubeRT);
    es.add(cc); cc.update(renderer,es);
    scene.environment=pmrem.fromCubemap(cubeRT.texture).texture;
    pmrem.dispose();
  })();

  // Lighting — clean studio key + cool fill
  scene.add(new THREE.HemisphereLight(0xe2ecff,0x101015,0.34));
  var s1=new THREE.DirectionalLight(0xffffff,1.95);
  s1.position.set(6,12,5); s1.castShadow=true;
  s1.shadow.mapSize.width=s1.shadow.mapSize.height=2048;
  s1.shadow.camera.near=1; s1.shadow.camera.far=40;
  s1.shadow.camera.left=-5; s1.shadow.camera.right=5;
  s1.shadow.camera.top=4; s1.shadow.camera.bottom=-4;
  s1.shadow.bias=-0.0005; scene.add(s1);
  var s2=new THREE.DirectionalLight(0x9fb4e0,0.50); s2.position.set(-5,3,-3); scene.add(s2);
  var s3=new THREE.DirectionalLight(0x6f8cff,0.40); s3.position.set(-2,2,-8); scene.add(s3);
  // Cool rim from behind to catch the rear-wing edge against black
  var s4=new THREE.DirectionalLight(0xbcd0ff,0.55); s4.position.set(1,4,-7); scene.add(s4);

  // Ground — subtly reflective studio floor
  var gnd=new THREE.Mesh(new THREE.PlaneGeometry(20,16),
    new THREE.MeshStandardMaterial({color:0x050507,metalness:0.62,roughness:0.20,envMapIntensity:0.9}));
  gnd.rotation.x=-Math.PI/2; gnd.position.y=-0.57; gnd.receiveShadow=true; scene.add(gnd);

  // Materials
  var PI=Math.PI;
  var mNav=new THREE.MeshPhysicalMaterial({color:0x0D1B8C,metalness:0.06,roughness:0.26,clearcoat:1.0,clearcoatRoughness:0.05});
  var mRed=new THREE.MeshPhysicalMaterial({color:0xCC0000,metalness:0.04,roughness:0.23,clearcoat:1.0,clearcoatRoughness:0.04});
  var mGold=new THREE.MeshPhysicalMaterial({color:0xC9A85C,metalness:0.84,roughness:0.12,clearcoat:0.55,clearcoatRoughness:0.14});
  var mC=new THREE.MeshPhysicalMaterial({color:0x0a0a0a,metalness:0.24,roughness:0.55,clearcoat:0.55,clearcoatRoughness:0.25});
  var mT=new THREE.MeshStandardMaterial({color:0x030303,metalness:0.0,roughness:0.98});
  var mR=new THREE.MeshPhysicalMaterial({color:0xBBBBBB,metalness:0.96,roughness:0.03,clearcoat:0.3});
  var mG=new THREE.MeshStandardMaterial({color:0x888888,metalness:0.74,roughness:0.28});
  var mB=new THREE.MeshPhysicalMaterial({color:0x1E41FF,metalness:0.04,roughness:0.32,clearcoat:0.85,clearcoatRoughness:0.07});

  function mk(geo,mat,x,y,z,rx,ry,rz){
    var m=new THREE.Mesh(geo,mat);
    m.position.set(x||0,y||0,z||0); m.rotation.set(rx||0,ry||0,rz||0); return m;
  }
  function bx(w,h,d,mat,x,y,z,rx,ry,rz){return mk(new THREE.BoxGeometry(w,h,d),mat,x,y,z,rx,ry,rz);}
  function cy(r1,r2,h,s,mat,x,y,z,rx,ry,rz){return mk(new THREE.CylinderGeometry(r1,r2,h,s),mat,x,y,z,rx,ry,rz);}
  function bar(ax,ay,az,ex,ey,ez,r,mat){
    var dx=ex-ax,dy=ey-ay,dz=ez-az,len=Math.sqrt(dx*dx+dy*dy+dz*dz);
    var m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,len,6),mat);
    m.position.set((ax+ex)/2,(ay+ey)/2,(az+ez)/2);
    var q=new THREE.Quaternion();
    q.setFromUnitVectors(new THREE.Vector3(0,1,0),new THREE.Vector3(dx/len,dy/len,dz/len));
    m.setRotationFromQuaternion(q); return m;
  }
  // Elliptical cross-section body section (CylinderGeometry scaled to oval, rotated along X)
  function el(rx,ry,depth,mat,x,y,z,rxr,ryr,rzr){
    var m=new THREE.Mesh(new THREE.CylinderGeometry(1,1,depth,32),mat);
    m.scale.set(rx,1,ry);
    m.position.set(x||0,y||0,z||0);
    m.rotation.set(rxr||0,ryr||0,rzr||0);
    return m;
  }
  // NACA-like airfoil extruded along Z for wing elements
  function wing(span,chord,thick,mat,x,y,z,ryRot){
    var sh=new THREE.Shape(),t=thick*0.5;
    sh.moveTo(0,0);
    sh.bezierCurveTo(chord*0.1,t, chord*0.4,t, chord,0);
    sh.bezierCurveTo(chord*0.4,-t, chord*0.1,-t, 0,0);
    var geo=new THREE.ExtrudeGeometry(sh,{depth:span,bevelEnabled:false,steps:1});
    geo.translate(0,0,-span*0.5);
    var m=new THREE.Mesh(geo,mat);
    m.position.set(x||0,y||0,z||0);
    m.rotation.y=ryRot||0;
    return m;
  }
  // Quadratic bezier tube — used for halo arch
  function tube3(ax,ay,az,bx2,by2,bz2,cx2,cy2,cz2,r,mat){
    var crv=new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(ax,ay,az),
      new THREE.Vector3(bx2,by2,bz2),
      new THREE.Vector3(cx2,cy2,cz2)
    );
    return new THREE.Mesh(new THREE.TubeGeometry(crv,20,r,8,false),mat);
  }

  var car=new THREE.Group();

  // Chassis — elliptical monocoque sections (Coke-bottle waist profile)
  car.add(el(0.31, 0.20, 0.70, mNav,  1.25, 0.05, 0, 0, 0, PI/2));
  car.add(el(0.35, 0.21, 0.80, mNav,  0.50, 0.05, 0, 0, 0, PI/2));
  car.add(el(0.26, 0.19, 0.50, mNav, -0.15, 0.05, 0, 0, 0, PI/2));
  car.add(el(0.31, 0.21, 0.50, mNav, -0.65, 0.05, 0, 0, 0, PI/2));
  car.add(el(0.275,0.18, 0.75, mNav, -1.28, 0.05, 0, 0, 0, PI/2));
  car.add(el(0.27, 0.105,1.10, mNav, -0.30, 0.31, 0, 0, 0, PI/2));
  car.add(el(0.30, 0.065,0.58, mNav,  0.53, 0.27, 0, 0, 0, PI/2));

  // Nose — LatheGeometry revolution profile
  var nosePts=[
    new THREE.Vector2(0.28,0.00),
    new THREE.Vector2(0.27,0.08),
    new THREE.Vector2(0.24,0.26),
    new THREE.Vector2(0.21,0.46),
    new THREE.Vector2(0.17,0.68),
    new THREE.Vector2(0.12,0.90),
    new THREE.Vector2(0.07,1.14),
    new THREE.Vector2(0.04,1.32),
    new THREE.Vector2(0.02,1.46)
  ];
  var noseMesh=new THREE.Mesh(new THREE.LatheGeometry(nosePts,32),mNav);
  noseMesh.rotation.z=-PI/2;
  noseMesh.position.set(1.60,-0.01,0);
  car.add(noseMesh);
  car.add(cy(0.022,0.022,0.06,8,mRed,3.10,-0.01,0,0,0,-PI/2));

  // Front wing — airfoil-section elements + red endplates
  car.add(wing(2.12, 0.30, 0.040, mC, 2.86, -0.245, 0, 0));
  car.add(wing(1.94, 0.24, 0.036, mC, 2.66, -0.200, 0, 0));
  car.add(wing(1.74, 0.18, 0.030, mC, 2.48, -0.158, 0, 0));
  [-1.03,1.03].forEach(function(z){
    car.add(bx(0.50,0.32,0.05,mRed,2.66,-0.095,z));
    car.add(bx(0.20,0.10,0.05,mRed,2.84,-0.30,z));
  });
  [-0.20,0.20].forEach(function(z){car.add(bx(0.06,0.24,0.04,mC,2.72,-0.075,z));});

  // Sidepods — tapered oval cross-sections, wider at intake than exit
  [-0.53,0.53].forEach(function(z){
    var sg=z>0?1:-1;
    var sf=new THREE.Mesh(new THREE.CylinderGeometry(0.175,0.145,0.85,24),mNav);
    sf.scale.set(1,1,1.9); sf.rotation.z=PI/2; sf.position.set(0.10,-0.03,z+sg*0.02); car.add(sf);
    var sr=new THREE.Mesh(new THREE.CylinderGeometry(0.130,0.090,0.77,24),mNav);
    sr.scale.set(1,1,1.7); sr.rotation.z=PI/2; sr.position.set(-0.565,-0.03,z+sg*0.02); car.add(sr);
    car.add(bx(0.90,0.32,0.012,mRed,-0.16,-0.03,z+sg*0.166));
    car.add(bx(0.09,0.21,0.09,mC,0.37,0.04,z));
    car.add(bx(1.22,0.08,0.26,mC,-0.39,-0.23,z));
  });

  // Bargeboards
  for(var bi=0;bi<3;bi++){
    [-0.34-bi*0.09,0.34+bi*0.09].forEach(function(z){car.add(bx(0.09,0.20,0.03,mC,0.84,-0.04,z));});
  }

  // Engine cover — LatheGeometry fin profile + gold stripe + oval intake
  var ecPts=[
    new THREE.Vector2(0.000,0.00),new THREE.Vector2(0.085,0.10),
    new THREE.Vector2(0.195,0.26),new THREE.Vector2(0.260,0.35),
    new THREE.Vector2(0.240,0.25),new THREE.Vector2(0.185,0.10),
    new THREE.Vector2(0.000,0.00)
  ];
  var ecMesh=new THREE.Mesh(new THREE.LatheGeometry(ecPts,20),mNav);
  ecMesh.rotation.z=PI/2; ecMesh.scale.set(1,0.054,1);
  ecMesh.position.set(-0.20,0.18,0); car.add(ecMesh);
  car.add(bx(0.70,0.040,0.054,mGold,-0.20,0.71,0));
  var riMesh=new THREE.Mesh(new THREE.CylinderGeometry(0.16,0.16,0.19,20),mNav);
  riMesh.scale.set(1,1,2.0); riMesh.position.set(0.29,0.39,0); car.add(riMesh);

  // Halo — cylindrical post + bezier arch + angled rear legs
  car.add(cy(0.028,0.028,0.34,12,mG,0.60,0.52,0));
  car.add(tube3(0.40,0.70,-0.28, 0.57,0.82,0, 0.40,0.70,0.28, 0.034,mG));
  car.add(bar(0.40,0.70,-0.28, 0.18,0.30,-0.26, 0.022,mG));
  car.add(bar(0.40,0.70, 0.28, 0.18,0.30, 0.26, 0.022,mG));

  // Mirrors — cylindrical stanchions + polished faces
  car.add(cy(0.016,0.016,0.28,8,mC,0.50,0.38,-0.26));
  car.add(cy(0.016,0.016,0.28,8,mC,0.50,0.38, 0.26));
  car.add(bx(0.11,0.07,0.17,mR,0.47,0.53,-0.26));
  car.add(bx(0.11,0.07,0.17,mR,0.47,0.53, 0.26));

  // Helmet (Verstappen blue)
  var helm=mk(new THREE.SphereGeometry(0.14,24,18),mB,0.41,0.36,0);
  helm.scale.set(1.2,0.92,1.1); car.add(helm);

  // Floor + diffuser strakes + diffuser ramp
  car.add(bx(3.12,0.042,1.80,mC,-0.08,-0.21,0));
  for(var ds=-3;ds<=3;ds++){car.add(bx(0.64,0.13,0.03,mC,-1.83,-0.15,ds*0.245));}
  car.add(bx(0.42,0.030,1.60,mC,-1.830,-0.146,0,0,0,2.719));

  // Rear wing — airfoil-section elements + swan-neck pylons + endplates
  car.add(bx(0.33,0.23,0.42,mNav,-1.71,0.07,0));
  car.add(bx(0.54,0.065,0.90,mC,-1.85,0.19,0));
  car.add(bar(-1.62,0.185,-0.13, -2.00,0.661,-0.18, 0.018,mC));
  car.add(bar(-1.62,0.185, 0.13, -2.00,0.661, 0.18, 0.018,mC));
  car.add(wing(1.60, 0.27, 0.058, mC, -2.03, 0.69, 0, 0));
  car.add(wing(1.48, 0.18, 0.046, mC, -1.87, 0.63, 0, 0));
  [-0.80,0.80].forEach(function(z){
    car.add(bx(0.31,0.58,0.048,mRed,-1.97,0.41,z));
  });

  // Suspension wishbones
  [[1.72,-0.80],[1.72,0.80]].forEach(function(w){
    var wx=w[0],wz=w[1],ci=wz>0?0.22:-0.22;
    car.add(bar(wx, 0.00,wz, 1.52, 0.08,ci, 0.016,mG));
    car.add(bar(wx, 0.00,wz, 1.18, 0.06,ci, 0.016,mG));
    car.add(bar(wx,-0.28,wz, 1.50,-0.22,ci, 0.016,mG));
    car.add(bar(wx,-0.28,wz, 1.16,-0.22,ci, 0.016,mG));
  });
  [[-1.52,-0.88],[-1.52,0.88]].forEach(function(w){
    var wx=w[0],wz=w[1],ci=wz>0?0.24:-0.24;
    car.add(bar(wx, 0.00,wz,-1.10, 0.06,ci, 0.016,mG));
    car.add(bar(wx, 0.00,wz,-1.42, 0.04,ci, 0.016,mG));
    car.add(bar(wx,-0.28,wz,-1.12,-0.20,ci, 0.016,mG));
    car.add(bar(wx,-0.28,wz,-1.44,-0.18,ci, 0.016,mG));
  });

  // Wheels — LatheGeometry tire profile
  function addWheel(x,z,tw){
    var g=new THREE.Group();
    var fs=(z>0)?1:-1,fY=fs*tw*0.46;
    var R=0.340,ri=0.260,hw=tw*0.50;
    var tp=[
      new THREE.Vector2(ri,          hw+0.004),
      new THREE.Vector2(ri+0.022,    hw-0.002),
      new THREE.Vector2(R-0.030,     hw-0.004),
      new THREE.Vector2(R-0.006,     hw-0.026),
      new THREE.Vector2(R,           hw-0.056),
      new THREE.Vector2(R,           0),
      new THREE.Vector2(R,         -(hw-0.056)),
      new THREE.Vector2(R-0.006,   -(hw-0.026)),
      new THREE.Vector2(R-0.030,   -(hw-0.004)),
      new THREE.Vector2(ri+0.022,  -(hw-0.002)),
      new THREE.Vector2(ri,        -(hw+0.004))
    ];
    g.add(mk(new THREE.LatheGeometry(tp,52),mT,0,0,0));
    g.add(mk(new THREE.CylinderGeometry(ri-0.002,ri-0.002,tw+0.006,44),mR,0,0,0));
    g.add(mk(new THREE.CylinderGeometry(ri-0.004,ri-0.004,0.026,44),mR,0,fY,0));
    g.add(mk(new THREE.CylinderGeometry(0.065,0.065,tw+0.032,16),mG,0,0,0));
    for(var i=0;i<5;i++){
      var pv=new THREE.Group();
      pv.rotation.y=i*2*PI/5; pv.position.y=fY;
      var sp=new THREE.Mesh(new THREE.BoxGeometry(ri-0.044,0.020,0.020),mR);
      sp.position.x=(ri-0.044)/2; pv.add(sp); g.add(pv);
    }
    g.rotation.x=PI/2; g.position.set(x,-0.22,z); car.add(g);
  }
  addWheel(1.72,-0.80,0.300); addWheel(1.72,0.80,0.300);
  addWheel(-1.52,-0.88,0.405); addWheel(-1.52,0.88,0.405);

  scene.add(car); car.rotation.y=PI/6;
  car.traverse(function(o){if(o.isMesh){o.castShadow=true;o.receiveShadow=true;}});

  // Soft elliptical contact shadow — grounds the car; a child of `car` so it follows the turntable
  (function(){
    var cc2=document.createElement('canvas');cc2.width=cc2.height=256;
    var g2d=cc2.getContext('2d');
    var grd=g2d.createRadialGradient(128,128,8,128,128,128);
    grd.addColorStop(0,'rgba(0,0,0,0.62)');grd.addColorStop(0.45,'rgba(0,0,0,0.30)');grd.addColorStop(1,'rgba(0,0,0,0)');
    g2d.fillStyle=grd;g2d.fillRect(0,0,256,256);
    var csTex=new THREE.CanvasTexture(cc2);
    var cs=new THREE.Mesh(new THREE.PlaneGeometry(8.2,3.6),
      new THREE.MeshBasicMaterial({map:csTex,transparent:true,depthWrite:false,opacity:0.9}));
    cs.rotation.x=-PI/2; cs.position.set(0,-0.563,0); cs.renderOrder=2; car.add(cs);
  })();

  // Expose car builder for the mini-game script
  window._f1Mats={mNav:mNav,mRed:mRed,mGold:mGold,mC:mC,mT:mT,mR:mR,mG:mG,mB:mB};
  window._f1Helpers={mk:mk,bx:bx,cy:cy,bar:bar,el:el,wing:wing,tube3:tube3,addWheel:addWheel,PI:PI};
  // Expose showcase objects so the mini-game can raycast "click the car"
  window._f1Showcase={canvas:c,renderer:renderer,camera:cam,car:car};

  function animate(){requestAnimationFrame(animate);car.rotation.y+=0.0032;renderer.render(scene,cam);}
  animate();
  window.addEventListener('resize',function(){
    var nW=c.parentElement.offsetWidth||900;
    cam.aspect=nW/H; cam.updateProjectionMatrix(); renderer.setSize(nW,H);
  });
})();
</script>
<script>
(function(){
'use strict';

/* ===================== DOM ===================== */
var overlay=document.getElementById('game-overlay');
var gameCanvas=document.getElementById('game-canvas');
var podiumOvl=document.getElementById('podium-overlay');
var podiumTable=document.getElementById('podium-table');
var pauseOvl=document.getElementById('pause-overlay');
var lightsBarEl=document.getElementById('lights-bar');
var goFlashEl=document.getElementById('go-flash');
var gridMsgEl=document.getElementById('grid-msg');
var hudPos=document.querySelector('.hud-pos');
var hudLap=document.querySelector('.hud-lap');
var hudTimer=document.querySelector('.hud-timer');
var hudSpeed=document.querySelector('.hud-speed');
var hudGear=document.querySelector('.hud-gear');
var rpmFill=document.getElementById('rpm-fill');
var hudDrs=document.querySelector('.hud-drs');
var tireF=document.getElementById('tire-f');
var tireR=document.getElementById('tire-r');
var tempF=document.getElementById('temp-f');
var tempR=document.getElementById('temp-r');
var hudS1=document.getElementById('hud-s1');
var hudS2=document.getElementById('hud-s2');
var hudS3=document.getElementById('hud-s3');
var minimapCanvas=document.getElementById('hud-minimap');
var minimapCtx=minimapCanvas?minimapCanvas.getContext('2d'):null;
var hudStandings=document.getElementById('hud-standings');
var wheelInd=document.getElementById('hud-wheel-ind');
// Cockpit (first-person) F1 steering wheel overlay
var cockpitWheel=document.getElementById('cockpit-wheel');
var cwRot=document.getElementById('cw-rot');
var cwGear=document.getElementById('cw-gear');
var cwSteer=0,cwLeds=[];
(function buildCwLeds(){
  var host=document.getElementById('cw-leds');if(!host) return;
  var N=13,x0=-84,gap=14,svgNS='http://www.w3.org/2000/svg';
  for(var i=0;i<N;i++){
    var c=i<5?'#00ff66':i<10?'#ff2a00':'#3aa0ff';
    var r=document.createElementNS(svgNS,'rect');
    r.setAttribute('x',x0+i*gap);r.setAttribute('y',-63);r.setAttribute('width',11);r.setAttribute('height',12);
    r.setAttribute('rx',2);r.setAttribute('fill',c);r.setAttribute('opacity',0.12);
    host.appendChild(r);cwLeds.push(r);
  }
})();
var hudClose=document.querySelector('.hud-close');
var podiumClose=document.getElementById('podium-close');

/* ===================== CONSTANTS ===================== */
var SEGS=900,TW=21,CURB=2.2,BH=3.2;
var MAX_SPD=30,ENG=12000,BRK=18000,DRAG=0.35,CAR_MASS=800;
var GRIP_LIMIT=4.0,GRIP_SCRUB=3.2;  // lateral load (yawRate*speed) the tires hold; above it, understeer scrubs speed — you must brake for corners
var RIDE_H=0.59,LAPS=3;
var AI_NUMS=[22,16,44,63,12,4,81,14,18,10,7,23,55,30,6,27,5,31,87];
var AI_COLORS=[
  0x1A3E82,0xCC0000,0xCC0000,0xC0C0C0,0xC0C0C0,
  0xFF5500,0xFF5500,0x1B5E38,0x1B5E38,0x0090FF,
  0x0090FF,0x005AFF,0x005AFF,0x1934DB,0x1934DB,
  0x52E252,0x52E252,0xDDDDDD,0xDDDDDD
];
var AI_ACCENTS=[
  0xCC1E1E,0xFFD700,0xFFD700,0x00D2BE,0x00D2BE,
  0x0D1F8C,0x0D1F8C,0xC5A028,0xC5A028,0xFF87BC,
  0xFF87BC,0xE8002D,0xE8002D,0xCC1E1E,0xCC1E1E,
  0xEEEEEE,0xEEEEEE,0xCC1E1E,0xCC1E1E
];
var AI_NAMES=['VERSTAPPEN','TSUNODA','LECLERC','HAMILTON','RUSSELL','ANTONELLI',
  'NORRIS','PIASTRI','ALONSO','STROLL','GASLY','DOOHAN',
  'ALBON','SAINZ','LAWSON','HADJAR','HULKENBERG','BORTOLETO','OCON','BEARMAN'];
var AI_TEAMS=[
  'RBR','FER','FER','MER','MER',
  'MCL','MCL','AMR','AMR','ALP',
  'ALP','WIL','WIL','RB','RB',
  'SAU','SAU','HAS','HAS'
];

/* ===================== TRACK ===================== */
/* Suzuka-flavoured technical circuit (clockwise). y carries gentle elevation —
   flat across the start/finish straight (the pit complex datum), climbing through
   the Esses to a crest, descending back to flat before the line. The layout is an
   unfolded loop (no true crossover — the flat track snaps cars by XZ only). */
var trackPts=[
  /* Start/finish straight — flat datum, pit complex sits alongside */
  new THREE.Vector3(-300, 0,  40),
  new THREE.Vector3(-120, 0,  30),
  new THREE.Vector3( 120, 0,  32),
  /* Turn 1-2 — fast right, road begins to climb */
  new THREE.Vector3( 260, 2,  60),
  new THREE.Vector3( 330, 5, 130),
  /* The Esses — snaking L-R-L-R up the hill */
  new THREE.Vector3( 285, 8, 185),
  new THREE.Vector3( 340,11, 240),
  new THREE.Vector3( 280,13, 300),
  new THREE.Vector3( 335,14, 355),
  /* Dunlop -> Degner — fast left across the crest */
  new THREE.Vector3( 270,13, 425),
  new THREE.Vector3( 140,11, 460),
  /* Hairpin — tight U at the far end */
  new THREE.Vector3(  40,10, 450),
  new THREE.Vector3( -30,10, 405),
  new THREE.Vector3( -10, 9, 345),
  /* 130R-style fast left, sweeping back downhill */
  new THREE.Vector3(-110, 8, 330),
  new THREE.Vector3(-240, 6, 330),
  new THREE.Vector3(-330, 4, 270),
  new THREE.Vector3(-360, 2, 180),
  /* Final flowing left back onto the straight (returns to flat) */
  new THREE.Vector3(-355, 1, 110),
  new THREE.Vector3(-330, 0,  68),
];
var trackCurve=new THREE.CatmullRomCurve3(trackPts,true,'catmullrom',0.5);
var wpAll=trackCurve.getSpacedPoints(SEGS);
var waypoints=wpAll.slice(0,SEGS);
var N=waypoints.length;

function wpDir(i){
  var a=waypoints[(i+3)%N],b=waypoints[((i-3)+N)%N];
  var dx=a.x-b.x,dz=a.z-b.z,len=Math.sqrt(dx*dx+dz*dz)||1;
  return {x:dx/len,z:dz/len};
}
function wpPerp(i){var d=wpDir(i);return {x:d.z,z:-d.x};}

function closestWP(x,z,hint){
  var best=hint||0,bd=1e9,start=((best-50)+N*2)%N;
  for(var ii=0;ii<100;ii++){
    var idx=(start+ii)%N,wp=waypoints[idx];
    var dd=(wp.x-x)*(wp.x-x)+(wp.z-z)*(wp.z-z);
    if(dd<bd){bd=dd;best=idx;}
  }
  return best;
}

/* ===================== THREE.JS SETUP ===================== */
if(!gameCanvas||typeof THREE==='undefined') return;
var renderer=new THREE.WebGLRenderer({canvas:gameCanvas,antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.shadowMap.enabled=true;
renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.outputEncoding=THREE.sRGBEncoding;
renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=1.0;
var scene=new THREE.Scene();
scene.background=null;
scene.fog=new THREE.FogExp2(0x05070e,0.0017);
var gameCam=new THREE.PerspectiveCamera(72,1,0.1,2000);
scene.add(gameCam);

/* Post-processing: bloom makes the floodlights / kerbs / car lights glow at night.
   Guarded — if the examples/js scripts failed to load we fall back to a plain render. */
var composer=null,useComposer=false,afterPass=null;
try{
  if(THREE.EffectComposer&&THREE.RenderPass&&THREE.UnrealBloomPass){
    composer=new THREE.EffectComposer(renderer);
    composer.addPass(new THREE.RenderPass(scene,gameCam));
    composer.addPass(new THREE.UnrealBloomPass(new THREE.Vector2(1280,720),0.50,0.4,0.88));
    // Speed motion blur: ghost-trail pass whose damp is driven by player speed in the loop
    // (near-zero at low speed so the static world stays crisp; ramps up toward top speed).
    if(THREE.AfterimagePass){afterPass=new THREE.AfterimagePass();afterPass.uniforms['damp'].value=0.0;composer.addPass(afterPass);}
    // Composer render targets are linear; convert back to sRGB on the way to screen
    // (otherwise the whole frame is shown un-gamma'd and midtones crush to black).
    if(THREE.GammaCorrectionShader&&THREE.ShaderPass) composer.addPass(new THREE.ShaderPass(THREE.GammaCorrectionShader));
    useComposer=true;
  }
}catch(e){useComposer=false;composer=null;}

function resizeCam(){
  var W=gameCanvas.clientWidth||900,CH=gameCanvas.clientHeight||600;
  renderer.setSize(W,CH,false);
  if(composer) composer.setSize(W,CH);
  gameCam.aspect=W/CH;gameCam.updateProjectionMatrix();
}
window.addEventListener('resize',resizeCam);resizeCam();

/* ===================== LIGHTING (night) ===================== */
/* Floodlit-stadium night: the track surface is brightly lit (cool white) under a
   dark sky. One shadow-casting key keeps cars grounded; floodlight PointLights add
   warm local pools, and the bloom pass makes the lamps/lights glow. */
scene.add(new THREE.AmbientLight(0x8aa0c0,0.16));
scene.add(new THREE.HemisphereLight(0x3a4a68,0x080a10,0.34));
var sun=new THREE.DirectionalLight(0xbecbf0,0.58);
sun.position.set(180,320,80);sun.castShadow=true;
sun.shadow.mapSize.width=sun.shadow.mapSize.height=2048;
sun.shadow.camera.near=1;sun.shadow.camera.far=1200;
sun.shadow.camera.left=-600;sun.shadow.camera.right=600;
sun.shadow.camera.top=600;sun.shadow.camera.bottom=-600;
sun.shadow.bias=-0.0004;
scene.add(sun);
var fill=new THREE.DirectionalLight(0x5a72a8,0.16);
fill.position.set(-120,60,-80);scene.add(fill);

/* ===================== WORLD BUILD ===================== */
// Populated inside buildWorld but animated from the game loop, so they live at the outer scope.
var clouds=[],standAnchors=[],crowdMats=[];
(function buildWorld(){
  var i,b,wp,p,hw;
  /* Sky dome — clear night gradient with moon + stars + Milky Way + drifting clouds */
  (function(){
    var sgeo=new THREE.SphereGeometry(1800,32,20);
    var sp=sgeo.getAttribute('position'),sc=[];
    for(var si=0;si<sp.count;si++){
      var ny=sp.getY(si)/1800;
      var sh=(ny+1)*0.5;                 // 0 = nadir, 1 = zenith
      var lo=1-Math.min(1,Math.max(0,sh));// brighter toward horizon
      // deep navy zenith -> faint blue horizon glow
      sc.push(0.012+0.040*lo, 0.022+0.055*lo, 0.055+0.095*lo);
    }
    sgeo.setAttribute('color',new THREE.Float32BufferAttribute(sc,3));
    scene.add(new THREE.Mesh(sgeo,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.BackSide})));
    // Moon disc + soft halo (bright -> blooms)
    var moon=new THREE.Mesh(new THREE.SphereGeometry(34,20,14),new THREE.MeshBasicMaterial({color:0xeef2ff}));
    moon.position.set(1380,640,420);scene.add(moon);
    var moonGlow=new THREE.Mesh(new THREE.SphereGeometry(95,16,10),new THREE.MeshBasicMaterial({color:0xb8c8ff,transparent:true,opacity:0.13}));
    moonGlow.position.copy(moon.position);scene.add(moonGlow);
    // Star field — sparse points on the upper dome
    var stN=700,stPos=[];
    for(var sti=0;sti<stN;sti++){
      var u=Math.random()*Math.PI*2,v=Math.random()*0.9+0.06,rr=1700;
      var sy=Math.cos(v*Math.PI*0.5);            // bias to upper sky
      var sxz=Math.sqrt(1-sy*sy);
      stPos.push(Math.cos(u)*sxz*rr,sy*rr,Math.sin(u)*sxz*rr);
    }
    var stGeo=new THREE.BufferGeometry();
    stGeo.setAttribute('position',new THREE.Float32BufferAttribute(stPos,3));
    scene.add(new THREE.Points(stGeo,new THREE.PointsMaterial({color:0xcfe0ff,size:3.2,sizeAttenuation:false,transparent:true,opacity:0.9,fog:false})));
    // Milky Way — dense faint star band across a tilted great circle
    var mwPos=[],mwRR=1690,tilt=0.55;
    for(var mwi=0;mwi<650;mwi++){
      var th=Math.random()*Math.PI*2,band=(Math.random()-0.5)*0.30+(Math.random()-0.5)*0.16;
      var bx=Math.cos(th),bz=Math.sin(th),by=band;
      var ty2=by*Math.cos(tilt)-bz*Math.sin(tilt),tz2=by*Math.sin(tilt)+bz*Math.cos(tilt);
      var ln=Math.sqrt(bx*bx+ty2*ty2+tz2*tz2)||1;
      if(ty2/ln<0.04) continue;
      mwPos.push(bx/ln*mwRR,ty2/ln*mwRR,tz2/ln*mwRR);
    }
    var mwGeo=new THREE.BufferGeometry();
    mwGeo.setAttribute('position',new THREE.Float32BufferAttribute(mwPos,3));
    scene.add(new THREE.Points(mwGeo,new THREE.PointsMaterial({color:0xbfd0ff,size:2.0,sizeAttenuation:false,transparent:true,opacity:0.55,fog:false})));
    // Drifting clouds — large soft sprites high overhead, slowly tracked across the sky in the loop
    var cc2=document.createElement('canvas');cc2.width=cc2.height=128;var cg2=cc2.getContext('2d');
    var crg=cg2.createRadialGradient(64,64,4,64,64,64);
    crg.addColorStop(0,'rgba(60,74,110,0.55)');crg.addColorStop(0.5,'rgba(40,52,82,0.25)');crg.addColorStop(1,'rgba(40,52,82,0)');
    cg2.fillStyle=crg;cg2.fillRect(0,0,128,128);
    var cloudTex=new THREE.CanvasTexture(cc2);
    for(var cli=0;cli<6;cli++){
      var clm=new THREE.SpriteMaterial({map:cloudTex,transparent:true,opacity:0.10,depthWrite:false,fog:false});
      var clsp=new THREE.Sprite(clm);
      clsp.scale.set(720+Math.random()*520,260+Math.random()*170,1);
      clsp.position.set(-1400+Math.random()*2800,540+Math.random()*240,-700+Math.random()*1400);
      scene.add(clsp);clouds.push({sp:clsp,vx:7+Math.random()*9});
    }
  })();
  /* Procedural ground texture: speckled noise so asphalt/grass read less flat up close.
     Returned greyscale-ish so it multiplies cleanly over a base material colour / vertex colours. */
  function groundTex(size,baseHex,specks,sLo,sHi,rep){
    var cc=document.createElement('canvas');cc.width=cc.height=size;var gg=cc.getContext('2d');
    gg.fillStyle=baseHex;gg.fillRect(0,0,size,size);
    for(var qi=0;qi<specks;qi++){
      var v=Math.floor(sLo+Math.random()*(sHi-sLo));
      gg.fillStyle='rgba('+v+','+v+','+v+','+(0.05+Math.random()*0.12)+')';
      var rr2=1+Math.random()*size*0.045;
      gg.beginPath();gg.arc(Math.random()*size,Math.random()*size,rr2,0,6.283);gg.fill();
    }
    var t=new THREE.CanvasTexture(cc);t.wrapS=t.wrapT=THREE.RepeatWrapping;
    if(rep) t.repeat.set(rep,rep);return t;
  }
  /* Road */
  var rPos=[],rIdx=[],rUv=[];
  for(i=0;i<N;i++){
    wp=waypoints[i];p=wpPerp(i);hw=TW*0.5;
    rPos.push(wp.x+p.x*hw,wp.y+0.06,wp.z+p.z*hw,
              wp.x-p.x*hw,wp.y+0.06,wp.z-p.z*hw);
    rUv.push(0,i*0.20, 1,i*0.20);   // tile asphalt along the lap length
    if(i<N-1){b=i*2;rIdx.push(b,b+1,b+2,b+2,b+1,b+3);}
  }
  b=(N-1)*2;rIdx.push(b,b+1,0,0,b+1,1);
  var rGeo=new THREE.BufferGeometry();
  rGeo.setAttribute('position',new THREE.Float32BufferAttribute(rPos,3));
  rGeo.setAttribute('uv',new THREE.Float32BufferAttribute(rUv,2));
  rGeo.setIndex(rIdx);rGeo.computeVertexNormals();
  var asphaltTex=groundTex(256,'#d8d8dc',900,150,255,1);
  var road=new THREE.Mesh(rGeo,new THREE.MeshStandardMaterial({color:0x17171b,map:asphaltTex,roughness:0.82,metalness:0.04}));
  road.receiveShadow=true;scene.add(road);
  /* Painted track lines: white limit lines (catch the lights) + dark rubbered-in racing band.
     Reuses the road's vertex/index winding so face normals point +Y. */
  (function(){
    function strip(latC,wid,yOff,mat){
      var pos=[],idx=[],ii,ww,pp;
      for(ii=0;ii<N;ii++){
        ww=waypoints[ii];pp=wpPerp(ii);
        pos.push(ww.x+pp.x*(latC+wid*0.5),ww.y+yOff,ww.z+pp.z*(latC+wid*0.5),
                 ww.x+pp.x*(latC-wid*0.5),ww.y+yOff,ww.z+pp.z*(latC-wid*0.5));
        if(ii<N-1){var bb2=ii*2;idx.push(bb2,bb2+1,bb2+2,bb2+2,bb2+1,bb2+3);}
      }
      var be=(N-1)*2;idx.push(be,be+1,0,0,be+1,1);
      var geo=new THREE.BufferGeometry();
      geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
      geo.setIndex(idx);geo.computeVertexNormals();
      var m=new THREE.Mesh(geo,mat);m.receiveShadow=true;scene.add(m);
    }
    strip(0,TW*0.30,0.068,new THREE.MeshStandardMaterial({color:0x0d0d0f,roughness:0.7}));
    var lineMat=new THREE.MeshStandardMaterial({color:0xffffff,emissive:0x3a3a44,emissiveIntensity:1.0,roughness:0.55});
    var lEdge=TW*0.5-0.5;
    strip( lEdge,0.34,0.074,lineMat);
    strip(-lEdge,0.34,0.074,lineMat);
  })();
  /* S/F line */
  (function(){
    var sfwp=waypoints[0],sfd=wpDir(0);
    var sfAngle=Math.atan2(sfd.x,sfd.z);
    var sfMat=new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xffffff,emissiveIntensity:0.6,roughness:0.7});
    for(var li=0;li<5;li++){
      var sOff=(li-2)*0.55;
      var sw=new THREE.Mesh(new THREE.BoxGeometry(TW+CURB*2,0.06,0.32),sfMat);
      sw.position.set(sfwp.x+sfd.x*sOff,sfwp.y+0.075,sfwp.z+sfd.z*sOff);
      sw.rotation.y=sfAngle;scene.add(sw);
    }
  })();

  /* Kerbs — alternating red/white stripes via vertex colors */
  var kPos=[],kIdx=[],kCol=[];
  for(i=0;i<N;i++){
    wp=waypoints[i];p=wpPerp(i);hw=TW*0.5;
    kPos.push(wp.x+p.x*(hw+CURB),wp.y+0.07,wp.z+p.z*(hw+CURB),
              wp.x+p.x*hw,        wp.y+0.07,wp.z+p.z*hw,
              wp.x-p.x*hw,        wp.y+0.07,wp.z-p.z*hw,
              wp.x-p.x*(hw+CURB), wp.y+0.07,wp.z-p.z*(hw+CURB));
    var isRed=Math.floor(i/3)%2===0;
    var kr=isRed?0.80:0.94,kg=isRed?0.13:0.94,kb=isRed?0.00:0.94;
    for(var vi=0;vi<4;vi++) kCol.push(kr,kg,kb);
    if(i<N-1){
      b=i*4;
      kIdx.push(b,b+1,b+4,b+4,b+1,b+5,b+2,b+3,b+6,b+6,b+3,b+7);
    }
  }
  var kGeo=new THREE.BufferGeometry();
  kGeo.setAttribute('position',new THREE.Float32BufferAttribute(kPos,3));
  kGeo.setAttribute('color',new THREE.Float32BufferAttribute(kCol,3));
  kGeo.setIndex(kIdx);kGeo.computeVertexNormals();
  scene.add(new THREE.Mesh(kGeo,new THREE.MeshStandardMaterial({vertexColors:true,emissive:0x2a2a2a,emissiveIntensity:1.0,roughness:0.8})));

  /* Armco barriers */
  var baPos=[],baIdx=[];
  hw=TW*0.5+CURB;
  for(i=0;i<N;i++){
    wp=waypoints[i];p=wpPerp(i);
    baPos.push(wp.x+p.x*hw,wp.y,    wp.z+p.z*hw,
               wp.x+p.x*hw,wp.y+BH, wp.z+p.z*hw,
               wp.x-p.x*hw,wp.y,    wp.z-p.z*hw,
               wp.x-p.x*hw,wp.y+BH, wp.z-p.z*hw);
    if(i<N-1){
      b=i*4;
      baIdx.push(b,b+4,b+1,b+1,b+4,b+5,b+2,b+6,b+3,b+3,b+6,b+7);
    }
  }
  var baGeo=new THREE.BufferGeometry();
  baGeo.setAttribute('position',new THREE.Float32BufferAttribute(baPos,3));
  baGeo.setIndex(baIdx);baGeo.computeVertexNormals();
  scene.add(new THREE.Mesh(baGeo,new THREE.MeshStandardMaterial({color:0xe0e0e0,roughness:0.3,metalness:0.45})));

  /* Armco red stripe — curvature-aware offset matches armco */
  var strPos=[],strIdx=[];
  var sY=BH*0.58,sH=0.28,strHw=TW*0.5+CURB;
  for(i=0;i<N;i++){
    wp=waypoints[i];p=wpPerp(i);
    strPos.push(wp.x+p.x*strHw,wp.y+sY,    wp.z+p.z*strHw,
                wp.x+p.x*strHw,wp.y+sY+sH, wp.z+p.z*strHw,
                wp.x-p.x*strHw,wp.y+sY,    wp.z-p.z*strHw,
                wp.x-p.x*strHw,wp.y+sY+sH, wp.z-p.z*strHw);
    if(i<N-1){b=i*4;strIdx.push(b,b+4,b+1,b+1,b+4,b+5,b+2,b+6,b+3,b+3,b+6,b+7);}
  }
  var strGeo=new THREE.BufferGeometry();
  strGeo.setAttribute('position',new THREE.Float32BufferAttribute(strPos,3));
  strGeo.setIndex(strIdx);strGeo.computeVertexNormals();
  scene.add(new THREE.Mesh(strGeo,new THREE.MeshStandardMaterial({color:0xCC0000,roughness:0.5})));

  /* Terrain — vertex-colored: run-off grey near track, lush green mid, golden-dry at edge */
  var TS=9,TER=900,cx=-20,cz=185;
  var cols=Math.floor(TER/TS)+1,rows=cols;
  var gPos2=new Float32Array(cols*rows*3),gIdx2=[],gCol2=[],gUv2=[];
  var halfTW=TW*0.5+CURB+2;
  for(var r=0;r<rows;r++){
    for(var c=0;c<cols;c++){
      var vx=cx-TER/2+c*TS,vz=cz-TER/2+r*TS;
      var vi=r*cols+c;
      var bd=1e9,bY=0;
      for(var k=0;k<N;k+=6){
        var wk=waypoints[k],ddx=wk.x-vx,ddz=wk.z-vz,dd=ddx*ddx+ddz*ddz;
        if(dd<bd){bd=dd;bY=wk.y;}
      }
      var dist=Math.sqrt(bd);
      var t=Math.max(0,Math.min(1,(dist-halfTW)/30));
      var hill=Math.sin(vx*0.020)*2.0+Math.cos(vz*0.016)*1.6+Math.sin((vx-vz)*0.011)*0.9;
      var hy=(bY-0.8)*(1-t)+(hill-2.5)*t;
      gPos2[vi*3]=vx;gPos2[vi*3+1]=hy;gPos2[vi*3+2]=vz;
      var tFar=Math.max(0,Math.min(1,(dist-halfTW-10)/100));
      var tNear=Math.max(0,Math.min(1,1-(dist-halfTW)/12));
      var hillN=Math.max(0,Math.min(1,(hy+2.5)/5.0));
      var gcR=0.22+0.30*tFar+0.07*hillN,gcG=0.50-0.06*tFar+0.06*hillN,gcB=0.14-0.06*tFar;
      if(tNear>0){gcR=gcR*(1-tNear*0.35)+0.42*tNear;gcG=gcG*(1-tNear*0.35)+0.42*tNear;gcB=gcB*(1-tNear*0.35)+0.26*tNear;}
      gCol2.push(gcR*0.17+0.008,gcG*0.19+0.013,gcB*0.22+0.024);
      gUv2.push(c*0.5,r*0.5);
      if(r<rows-1&&c<cols-1){gIdx2.push(vi,vi+cols,vi+1,vi+1,vi+cols,vi+cols+1);}
    }
  }
  var tGeo=new THREE.BufferGeometry();
  tGeo.setAttribute('position',new THREE.Float32BufferAttribute(gPos2,3));
  tGeo.setAttribute('color',new THREE.Float32BufferAttribute(gCol2,3));
  tGeo.setAttribute('uv',new THREE.Float32BufferAttribute(gUv2,2));
  tGeo.setIndex(gIdx2);tGeo.computeVertexNormals();
  var grassTex=groundTex(256,'#cfcfcf',1400,150,240,1);
  scene.add(new THREE.Mesh(tGeo,new THREE.MeshStandardMaterial({vertexColors:true,map:grassTex,roughness:1.0})));
  /* Outer ground ring — fills the gap between terrain edge and mountain base */
  (function(){
    var NS=64,igR=200,ogR=690,ogPos=[],ogIdx=[],ogCol=[];
    for(var oi=0;oi<=NS;oi++){
      var oAng=oi/NS*Math.PI*2,oCa=Math.cos(oAng),oSa=Math.sin(oAng);
      ogPos.push(oCa*igR,-2,oSa*igR, oCa*ogR,-4,oSa*ogR);
      ogCol.push(0.10,0.16,0.09, 0.08,0.10,0.11);
      if(oi<NS){var ob=oi*2;ogIdx.push(ob,ob+2,ob+1,ob+1,ob+2,ob+3);}
    }
    var ogGeo=new THREE.BufferGeometry();
    ogGeo.setAttribute('position',new THREE.Float32BufferAttribute(ogPos,3));
    ogGeo.setAttribute('color',new THREE.Float32BufferAttribute(ogCol,3));
    ogGeo.setIndex(ogIdx);ogGeo.computeVertexNormals();
    scene.add(new THREE.Mesh(ogGeo,new THREE.MeshStandardMaterial({vertexColors:true,roughness:1.0})));
  })();

  /* Trees & roadside foliage — layered crowns (stacked cones / clustered blobs) + low shrubs */
  var mTrunk=new THREE.MeshStandardMaterial({color:0x3d2008,roughness:1});
  var leafMats=[
    new THREE.MeshStandardMaterial({color:0x0a2406,roughness:1}),
    new THREE.MeshStandardMaterial({color:0x123010,roughness:1}),
    new THREE.MeshStandardMaterial({color:0x1a3e12,roughness:1}),
    new THREE.MeshStandardMaterial({color:0x0f3a1a,roughness:1}),
    new THREE.MeshStandardMaterial({color:0x244a16,roughness:1})
  ];
  var shrubMat=new THREE.MeshStandardMaterial({color:0x16380f,roughness:1});
  for(var ti=0;ti<N;ti+=6){
    var twp=waypoints[ti],tp=wpPerp(ti);
    var offs=TW*0.5+CURB+10;
    var isPine=(Math.floor(ti/6)%3!==1);
    var szJ=0.7+((ti*17)%13)/13*0.7;
    var lm=leafMats[(ti*5)%leafMats.length];
    [1,-1].forEach(function(s){
      var jit=((ti*7+s*3)%9)-4;
      var ox=twp.x+tp.x*s*(offs+jit),oz=twp.z+tp.z*s*(offs+jit);
      var trH=3.2+szJ;
      var tr=new THREE.Mesh(new THREE.CylinderGeometry(0.28,0.55,trH,7),mTrunk);
      tr.position.set(ox,twp.y+trH*0.5,oz);scene.add(tr);
      if(isPine){
        // stacked tiers — full at the base, tapering to a point
        for(var tier=0;tier<3;tier++){
          var tr2=1-tier*0.30,cone=new THREE.Mesh(new THREE.ConeGeometry(3.0*szJ*tr2,3.2*szJ,8),lm);
          cone.position.set(ox,twp.y+trH+1.0+tier*2.0*szJ,oz);cone.castShadow=true;scene.add(cone);
        }
      } else {
        // clustered canopy of overlapping blobs
        var cby=twp.y+trH+2.2*szJ;
        var blobs=[[0,0,0,1.0],[1.4,0.3,0.6,0.7],[-1.2,0.1,-0.7,0.7],[0.4,1.3,-0.4,0.65]];
        for(var bbi=0;bbi<blobs.length;bbi++){
          var bo=blobs[bbi],bl=new THREE.Mesh(new THREE.SphereGeometry(2.2*szJ*bo[3],7,6),lm);
          bl.position.set(ox+bo[0]*szJ,cby+bo[1]*szJ,oz+bo[2]*szJ);bl.castShadow=true;scene.add(bl);
        }
      }
      // low shrub tucked between barrier and treeline (every other slot)
      if((ti/6|0)%2===0){
        var sox=twp.x+tp.x*s*(offs-5.5),soz=twp.z+tp.z*s*(offs-5.5);
        var shrub=new THREE.Mesh(new THREE.SphereGeometry(1.0+Math.random()*0.5,6,5),shrubMat);
        shrub.position.set(sox,twp.y+0.7,soz);shrub.scale.y=0.7;scene.add(shrub);
      }
    });
  }

  /* Distant mountain ring — low-poly ridgeline fading into the golden haze */
  (function(){
    var NM=80,MR=630,mPos=[],mIdx=[],mCol=[];
    for(var mi=0;mi<=NM;mi++){
      var mAng=mi/NM*Math.PI*2;
      var mH=Math.max(8,30+Math.sin(mAng*5.3+0.5)*22+Math.sin(mAng*11.7+1.3)*14+Math.sin(mAng*19.1+2.7)*7);
      var mCx=Math.cos(mAng)*MR,mCz=Math.sin(mAng)*MR;
      mPos.push(mCx,-5,mCz, mCx,mH,mCz);
      var mBf=Math.max(0,Math.min(1,mH/70));
      mCol.push(0.11,0.13,0.19, 0.20+mBf*0.14,0.24+mBf*0.14,0.36+mBf*0.10);
      if(mi<NM){var mb=mi*2;mIdx.push(mb,mb+2,mb+1,mb+1,mb+2,mb+3);}
    }
    var mGeo=new THREE.BufferGeometry();
    mGeo.setAttribute('position',new THREE.Float32BufferAttribute(mPos,3));
    mGeo.setAttribute('color',new THREE.Float32BufferAttribute(mCol,3));
    mGeo.setIndex(mIdx);mGeo.computeVertexNormals();
    scene.add(new THREE.Mesh(mGeo,new THREE.MeshStandardMaterial({vertexColors:true,roughness:1.0,side:THREE.DoubleSide})));
  })();

  /* Grandstands */
  var mk2=function(geo,mat,x,y,z){var m=new THREE.Mesh(geo,mat);m.position.set(x||0,y||0,z||0);return m;};
  var mConc2=new THREE.MeshStandardMaterial({color:0x888878,roughness:0.85});
  var mRoof2=new THREE.MeshStandardMaterial({color:0x4a4e52,roughness:0.55,metalness:0.4});
  // Shared crowd materials (8 colours) — reused across every seat so the field of spectators is
  // cheap, and so the loop can pulse a faint emissive shimmer across the whole crowd.
  var crowdCols=[0xcc1111,0x1133cc,0xddcc00,0xffffff,0x118833,0xcc6600,0x880099,0x009988];
  crowdMats=crowdCols.map(function(hx){return new THREE.MeshStandardMaterial({color:hx,emissive:hx,emissiveIntensity:0.0,roughness:1});});
  function buildGrandstand(wpIdx,side,width,seatHex){
    var wp2=waypoints[wpIdx],p2=wpPerp(wpIdx);
    var ry2=Math.atan2(-p2.x*side,-p2.z*side);
    var c2=Math.cos(ry2),s2=Math.sin(ry2);
    // Auto-clearance: push the stand outward until its whole footprint clears the drivable track.
    // Without this, where the loop folds back near itself (e.g. the Esses at wp280) a fixed offset
    // can drop a stand corner inside the track edge.
    var halfFp=(width+4)*0.5,fpZ=[-6.3,-2,2,7.5,9.5],clearMin=TW*0.5+CURB+8;
    var dist2=TW*0.5+CURB+BH+22;
    for(var gi=0;gi<14;gi++){
      var gox=wp2.x+p2.x*side*dist2,goz=wp2.z+p2.z*side*dist2,nearest=1e9;
      for(var fxi=-1;fxi<=1;fxi++){
        var lx=fxi*halfFp;
        for(var fzi=0;fzi<fpZ.length;fzi++){
          var lz=fpZ[fzi];
          var wx=gox+lx*c2+lz*s2,wz=goz-lx*s2+lz*c2;
          for(var wi=0;wi<N;wi+=3){
            var ddx=wx-waypoints[wi].x,ddz=wz-waypoints[wi].z,d2=ddx*ddx+ddz*ddz;
            if(d2<nearest) nearest=d2;
          }
        }
      }
      if(nearest>=clearMin*clearMin) break;
      dist2+=3;
    }
    var ox2=wp2.x+p2.x*side*dist2,oz2=wp2.z+p2.z*side*dist2;
    var mS2=new THREE.MeshStandardMaterial({color:seatHex,roughness:0.8});
    var mAd2=new THREE.MeshStandardMaterial({color:0x1E41FF,emissive:0x0c1f70,emissiveIntensity:2.0,roughness:0.6});
    var g2=new THREE.Group();
    g2.add(mk2(new THREE.BoxGeometry(width,4,9  ),mConc2,0,2,   -1.5));
    g2.add(mk2(new THREE.BoxGeometry(width,3,7  ),mConc2,0,5.5,  3.5));
    g2.add(mk2(new THREE.BoxGeometry(width,2.5,5),mConc2,0,8.25, 7.5));
    g2.add(mk2(new THREE.BoxGeometry(width-1,0.3,8.5),mS2,0,4.15,-1.5));
    g2.add(mk2(new THREE.BoxGeometry(width-1,0.3,6.5),mS2,0,7.05, 3.5));
    g2.add(mk2(new THREE.BoxGeometry(width-1,0.3,4.5),mS2,0,9.50, 7.5));
    g2.add(mk2(new THREE.BoxGeometry(width+4,0.5,15),mRoof2,0,11.5,3));
    [-(width*0.5-2),width*0.5-2].forEach(function(cx){
      g2.add(mk2(new THREE.BoxGeometry(0.8,11.5,0.8),mConc2,cx,5.75,3));
    });
    g2.add(mk2(new THREE.BoxGeometry(width-4,2.5,0.25),mAd2,0,1.6,-6.2));
    g2.position.set(ox2,wp2.y,oz2);g2.rotation.y=ry2;scene.add(g2);
    var crowdGeo=new THREE.BoxGeometry(0.65,0.95,0.45);
    var cg=new THREE.Group(),ci2=0;
    [0,3.0,6.0].forEach(function(tZ2){
      [0,2.4,4.8].forEach(function(tY2){
        for(var cx3=-(width-3)*0.5;cx3<(width-3)*0.5;cx3+=1.6){
          var cm=new THREE.Mesh(crowdGeo,crowdMats[(ci2*7)%crowdMats.length]);
          cm.position.set(cx3,4.8+tY2,tZ2-1.0);cg.add(cm);ci2++;
        }
      });
    });
    cg.position.set(ox2,wp2.y,oz2);cg.rotation.y=ry2;scene.add(cg);
    standAnchors.push({x:ox2,y:wp2.y+9,z:oz2,w:width*0.5});
  }
  buildGrandstand( 10,+1,70,0x1E41FF);
  buildGrandstand( 10,-1,55,0xCC0000);
  buildGrandstand(130,+1,50,0xFFDD00);
  buildGrandstand(280,-1,45,0xffffff);
  buildGrandstand(450,+1,60,0x1E41FF);
  buildGrandstand(640,-1,40,0xCC0000);
  buildGrandstand(790,+1,50,0x22aa44);

  /* Floodlight towers — ring the circuit; emissive lamp banks bloom, a capped
     set of PointLights give real local fill so the track reads under the lights */
  (function(){
    var poleMat=new THREE.MeshStandardMaterial({color:0x2a2e33,roughness:0.5,metalness:0.55});
    var headMat=new THREE.MeshStandardMaterial({color:0x16181c,roughness:0.6,metalness:0.4});
    var lampMat=new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xdfeaff,emissiveIntensity:2.4,roughness:0.4});
    var lampGeo=new THREE.BoxGeometry(1.4,1.4,0.4);
    var towerWps=[40,150,260,370,480,590,700,810];
    for(var fi=0;fi<towerWps.length;fi++){
      var wpI=towerWps[fi],side=(fi%2===0)?1:-1;
      var fwp=waypoints[wpI],fp=wpPerp(wpI);
      var fdist=TW*0.5+CURB+BH+34;
      var fbx=fwp.x+fp.x*side*fdist,fbz=fwp.z+fp.z*side*fdist;
      var poleH=27;
      var pole=new THREE.Mesh(new THREE.CylinderGeometry(0.45,0.75,poleH,8),poleMat);
      pole.position.set(fbx,fwp.y+poleH*0.5,fbz);pole.castShadow=true;scene.add(pole);
      var faceY=Math.atan2(-fp.x*side,-fp.z*side);
      var head=new THREE.Group();
      head.add(new THREE.Mesh(new THREE.BoxGeometry(6.2,3.0,0.7),headMat));
      for(var lr=0;lr<2;lr++)for(var lc=0;lc<4;lc++){
        var lamp=new THREE.Mesh(lampGeo,lampMat);
        lamp.position.set((lc-1.5)*1.45,(lr-0.5)*1.35,0.45);head.add(lamp);
      }
      head.position.set(fbx,fwp.y+poleH+0.6,fbz);head.rotation.y=faceY;head.rotation.x=0.34;
      scene.add(head);
      // Local fill toward the track (no shadows — cheap)
      var pl=new THREE.PointLight(0xeaf0ff,1.35,235,1.4);
      pl.position.set(fwp.x+fp.x*side*(TW*0.5+6),fwp.y+22,fwp.z+fp.z*side*(TW*0.5+6));
      scene.add(pl);
      // Volumetric light shaft — faint additive cone from the lamp bank down onto the track
      var coneH=poleH+10,vc=new THREE.ConeGeometry(15,coneH,16,1,true);
      vc.translate(0,-coneH*0.5,0);  // apex at origin, opens downward (-Y)
      var vcone=new THREE.Mesh(vc,new THREE.MeshBasicMaterial({color:0xc4d6ff,transparent:true,opacity:0.045,side:THREE.DoubleSide,depthWrite:false,blending:THREE.AdditiveBlending,fog:false}));
      vcone.position.set(fbx,fwp.y+poleH+0.4,fbz);
      var vdir=new THREE.Vector3(fwp.x-fbx,fwp.y-(fwp.y+poleH+0.4),fwp.z-fbz).normalize();
      vcone.quaternion.setFromUnitVectors(new THREE.Vector3(0,-1,0),vdir);
      scene.add(vcone);
    }
  })();

  /* S/F gantry */
  (function(){
    var wp0=waypoints[0],p0=wpPerp(0);
    var hw=TW*0.5+CURB+BH+1.0;
    var mSt=new THREE.MeshStandardMaterial({color:0xdddddd,roughness:0.4,metalness:0.35});
    var mBl=new THREE.MeshStandardMaterial({color:0x1E41FF,roughness:0.5,metalness:0.1});
    var mLt=new THREE.MeshStandardMaterial({color:0xff2200,emissive:0xff2200,emissiveIntensity:2.4,roughness:0.4});
    [1,-1].forEach(function(s){
      var tx=wp0.x+p0.x*s*hw,tz=wp0.z+p0.z*s*hw;
      var gt=new THREE.Mesh(new THREE.BoxGeometry(1.2,18,1.2),mSt);
      gt.position.set(tx,wp0.y+9,tz);scene.add(gt);
    });
    var lx=wp0.x+p0.x*hw,lz=wp0.z+p0.z*hw;
    var rx=wp0.x-p0.x*hw,rz=wp0.z-p0.z*hw;
    var barLen=Math.sqrt((rx-lx)*(rx-lx)+(rz-lz)*(rz-lz))+1.2;
    var bar=new THREE.Mesh(new THREE.BoxGeometry(1.4,1.6,barLen),mBl);
    bar.position.set((lx+rx)*0.5,wp0.y+18,(lz+rz)*0.5);
    bar.rotation.y=Math.atan2(rx-lx,rz-lz);scene.add(bar);
    for(var gli=0;gli<5;gli++){
      var lt=(gli/4-0.5)*hw*1.6;
      var gl=new THREE.Mesh(new THREE.BoxGeometry(0.7,0.7,0.35),mLt);
      gl.position.set(wp0.x+p0.x*lt,wp0.y+16.5,wp0.z+p0.z*lt);scene.add(gl);
    }
  })();
  /* Pit complex */
  (function(){
    var plPos2=[],plIdx2=[];
    var plZ1=52,plZ2=144,plSteps=25,plX=-17,plW=7;
    for(var pli=0;pli<=plSteps;pli++){
      var plz2=plZ1+(plZ2-plZ1)*pli/plSteps;
      plPos2.push(plX-plW*0.5,0.02,plz2,plX+plW*0.5,0.02,plz2);
      if(pli<plSteps){var pb2=pli*2;plIdx2.push(pb2,pb2+2,pb2+1,pb2+1,pb2+2,pb2+3);}
    }
    var plGeo2=new THREE.BufferGeometry();
    plGeo2.setAttribute('position',new THREE.Float32BufferAttribute(plPos2,3));
    plGeo2.setIndex(plIdx2);plGeo2.computeVertexNormals();
    function pm(geo,mat,x,y,z,ry){var m=new THREE.Mesh(geo,mat);m.position.set(x,y,z);if(ry)m.rotation.y=ry;scene.add(m);}
    pm(plGeo2,new THREE.MeshStandardMaterial({color:0x252525,roughness:0.75}),0,0,0);
    pm(new THREE.BoxGeometry(0.45,1.3,93),new THREE.MeshStandardMaterial({color:0xc0c0c0,roughness:0.4,metalness:0.3}),-13.5,0.65,98);
    pm(new THREE.BoxGeometry(0.45,0.08,93),new THREE.MeshStandardMaterial({color:0xffffff,roughness:0.7}),-13.5,1.34,98);
    var pgCols=[0x1E41FF,0xff2200,0xffdd00,0x00ccff,0xff6600,0x22cc22,0xaaaaaa,0xcc00cc];
    for(var pg=0;pg<8;pg++){
      var pgz=60+pg*11;
      pm(new THREE.BoxGeometry(14,8.5,10.5),new THREE.MeshStandardMaterial({color:0x555560,roughness:0.7,metalness:0.1}),-36,4.25,pgz);
      pm(new THREE.BoxGeometry(9,6.5,0.3),new THREE.MeshStandardMaterial({color:pgCols[pg],emissive:pgCols[pg],emissiveIntensity:0.9,roughness:0.55}),-29.4,3.5,pgz);
      pm(new THREE.BoxGeometry(16,0.5,12.5),new THREE.MeshStandardMaterial({color:0x2a2a30,roughness:0.8,metalness:0.2}),-35,8.75,pgz);
      pm(new THREE.BoxGeometry(0.2,0.2,5),new THREE.MeshStandardMaterial({color:0x333333,metalness:0.5,roughness:0.5}),-28.5,2.5,pgz);
    }
    pm(new THREE.BoxGeometry(14,4,11),new THREE.MeshStandardMaterial({color:0x888878,roughness:0.85}),-42,2,147);
    pm(new THREE.BoxGeometry(9,22,7),new THREE.MeshStandardMaterial({color:0x6a7888,roughness:0.7,metalness:0.2}),-42,15,147);
    pm(new THREE.BoxGeometry(8,5,6.2),new THREE.MeshStandardMaterial({color:0x1a3344,roughness:0.08,metalness:0.7,transparent:true,opacity:0.85}),-42,28,147);
    pm(new THREE.BoxGeometry(8.2,1.5,0.3),new THREE.MeshStandardMaterial({color:0x1E41FF,emissive:0x1030a0,emissiveIntensity:1.8,roughness:0.5}),-42,23.5,143.8);
  })();
  /* Animated billboard posts */
  (function(){
    var CW=1024,CH=256;
    function _bbOracle(ctx){
      var g=ctx.createLinearGradient(0,0,CW,0);
      g.addColorStop(0,'#0c1f3d');g.addColorStop(0.5,'#1a3a6b');g.addColorStop(1,'#0c1f3d');
      ctx.fillStyle=g;ctx.fillRect(0,0,CW,CH);
      ctx.fillStyle='#C74634';ctx.fillRect(0,0,22,CH);
      ctx.fillStyle='#ffffff';ctx.fillRect(22,0,7,CH);
      ctx.fillStyle='#ffffff';ctx.font='bold 110px Arial,sans-serif';
      ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillText('ORACLE',100,CH*0.41);
      ctx.fillStyle='rgba(255,255,255,0.55)';ctx.font='400 36px Arial,sans-serif';
      ctx.fillText('RED BULL RACING PARTNER',100,CH*0.73);
      ctx.fillStyle='#C74634';ctx.font='bold 30px Arial,sans-serif';
      ctx.textAlign='right';ctx.fillText('ORACLE.COM',CW-24,CH*0.18);
    }
    function _bbF1(ctx){
      var g=ctx.createLinearGradient(0,0,0,CH);
      g.addColorStop(0,'#e8002d');g.addColorStop(1,'#9a0020');
      ctx.fillStyle=g;ctx.fillRect(0,0,CW,CH);
      ctx.fillStyle='rgba(255,255,255,0.14)';
      ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(CW*0.40,0);ctx.lineTo(CW*0.29,CH);ctx.lineTo(0,CH);ctx.fill();
      ctx.fillStyle='#ffffff';ctx.font='900 142px Arial,sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('F1',CW*0.5,CH*0.42);
      ctx.fillStyle='rgba(255,255,255,0.72)';ctx.font='600 34px Arial,sans-serif';
      ctx.fillText('FORMULA ONE WORLD CHAMPIONSHIP',CW*0.5,CH*0.78);
    }
    function _bbMcLaren(ctx){
      var g=ctx.createLinearGradient(0,0,CW,CH);
      g.addColorStop(0,'#ff8000');g.addColorStop(0.6,'#e06000');g.addColorStop(1,'#ff8000');
      ctx.fillStyle=g;ctx.fillRect(0,0,CW,CH);
      ctx.fillStyle='rgba(0,0,0,0.20)';
      ctx.beginPath();ctx.moveTo(CW*0.56,0);ctx.lineTo(CW,0);ctx.lineTo(CW*0.73,CH);ctx.lineTo(CW*0.18,CH);ctx.fill();
      ctx.fillStyle='#000000';ctx.font='900 118px Arial,sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('McLAREN',CW*0.5,CH*0.41);
      ctx.fillStyle='rgba(0,0,0,0.65)';ctx.font='500 36px Arial,sans-serif';
      ctx.fillText('BRITISH RACING · SINCE 1963',CW*0.5,CH*0.76);
    }
    function _bbPetronas(ctx){
      var g=ctx.createLinearGradient(0,0,CW,0);
      g.addColorStop(0,'#009e97');g.addColorStop(0.5,'#00b4ac');g.addColorStop(1,'#009e97');
      ctx.fillStyle=g;ctx.fillRect(0,0,CW,CH);
      ctx.fillStyle='rgba(0,0,0,0.15)';ctx.fillRect(0,CH*0.10,CW,CH*0.045);
      ctx.fillStyle='rgba(255,255,255,0.10)';ctx.fillRect(0,CH*0.15,CW,CH*0.045);
      ctx.fillStyle='#ffffff';ctx.font='bold 108px Arial,sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('PETRONAS',CW*0.5,CH*0.41);
      ctx.fillStyle='rgba(255,255,255,0.65)';ctx.font='400 36px Arial,sans-serif';
      ctx.fillText('PRIMAX · MOTORSPORT FUELS',CW*0.5,CH*0.76);
    }
    function _bbPirelli(ctx){
      ctx.fillStyle='#181818';ctx.fillRect(0,0,CW,CH);
      ctx.fillStyle='#FFD700';ctx.fillRect(0,CH*0.70,CW,CH*0.30);
      ctx.fillStyle='#FFD700';ctx.font='900 118px Arial,sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('PIRELLI',CW*0.5,CH*0.39);
      ctx.fillStyle='#181818';ctx.font='bold 38px Arial,sans-serif';
      ctx.fillText('P ZERO · THE POWER OF DREAMS',CW*0.5,CH*0.84);
    }
    function _bbAramco(ctx){
      var g=ctx.createLinearGradient(0,0,CW,0);
      g.addColorStop(0,'#004d2a');g.addColorStop(0.5,'#006b3c');g.addColorStop(1,'#004d2a');
      ctx.fillStyle=g;ctx.fillRect(0,0,CW,CH);
      ctx.strokeStyle='rgba(255,255,255,0.06)';ctx.lineWidth=1;
      for(var gx=0;gx<CW;gx+=64){ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,CH);ctx.stroke();}
      ctx.fillStyle='#ffffff';ctx.font='bold 112px Arial,sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('ARAMCO',CW*0.5,CH*0.40);
      ctx.fillStyle='rgba(255,255,255,0.60)';ctx.font='400 34px Arial,sans-serif';
      ctx.fillText("WORLD'S LEADING ENERGY COMPANY",CW*0.5,CH*0.76);
    }
    var adDrawers=[_bbOracle,_bbF1,_bbMcLaren,_bbPetronas,_bbPirelli,_bbAramco];
    var adEmissive=[0x0a1830,0x500010,0x3a2000,0x003533,0x1a1400,0x002018];
    var bbTextures=[],bbMats=[];
    adDrawers.forEach(function(fn,idx){
      var cv=document.createElement('canvas');cv.width=CW;cv.height=CH;
      fn(cv.getContext('2d'));
      var tex=new THREE.CanvasTexture(cv);tex.anisotropy=4;
      bbTextures.push(tex);
      bbMats.push(new THREE.MeshStandardMaterial({
        map:tex,emissiveMap:tex,
        emissive:new THREE.Color(adEmissive[idx]),
        emissiveIntensity:0.45,roughness:0.55,metalness:0.05,side:THREE.DoubleSide
      }));
    });
    var mPost=new THREE.MeshStandardMaterial({color:0x555555,roughness:0.5,metalness:0.6});
    var pGeo=new THREE.CylinderGeometry(0.18,0.22,9,7);
    var bOff=TW*0.5+CURB+BH+2.5;
    var _bbList=[];
    for(var bi=0;bi<N;bi+=20){
      var bwp=waypoints[bi],bp=wpPerp(bi),bd=wpDir(bi);
      var bAng=Math.atan2(bd.x,bd.z);
      var slot=Math.floor(bi/20)%adDrawers.length;
      var phase=Math.floor(bi/20)%8;
      [-1,1].forEach(function(bs){
        var bx=bwp.x+bp.x*bs*bOff,bz=bwp.z+bp.z*bs*bOff;
        var post=new THREE.Mesh(pGeo,mPost);
        post.position.set(bx,bwp.y+4.5,bz);scene.add(post);
        var bMat=bbMats[slot].clone();
        var board=new THREE.Mesh(new THREE.BoxGeometry(10,3,0.3),bMat);
        board.position.set(bx,bwp.y+7.5,bz);board.rotation.y=bAng;scene.add(board);
        _bbList.push({mat:bMat,slot:slot,phase:phase});
        slot=(slot+3)%adDrawers.length;
      });
    }
    var _bbTick=0;
    setInterval(function(){
      _bbTick++;
      _bbList.forEach(function(bb){
        if((_bbTick+bb.phase)%8===0){
          bb.slot=(bb.slot+1)%adDrawers.length;
          bb.mat.map=bbTextures[bb.slot];
          bb.mat.emissiveMap=bbTextures[bb.slot];
          bb.mat.emissive.set(adEmissive[bb.slot]);
          bb.mat.needsUpdate=true;
        }
      });
    },500);
  })();
  /* Catch fencing posts */
  (function(){
    var mFP=new THREE.MeshStandardMaterial({color:0x777777,roughness:0.5,metalness:0.45});
    var fGeo=new THREE.CylinderGeometry(0.07,0.07,5,5);
    var fOff=TW*0.5+CURB+BH+1.8;
    for(var fi=0;fi<N;fi+=5){
      var fwp=waypoints[fi],fp=wpPerp(fi);
      [-1,1].forEach(function(fs){
        var fm=new THREE.Mesh(fGeo,mFP);
        fm.position.set(fwp.x+fp.x*fs*fOff,fwp.y+2.5,fwp.z+fp.z*fs*fOff);
        scene.add(fm);
      });
    }
  })();
  /* Tire stacks at hairpin outer wall — dynamically positioned from spline */
  var mTireW=new THREE.MeshStandardMaterial({color:0xffffff,roughness:0.9});
  var mTireR=new THREE.MeshStandardMaterial({color:0xcc0000,roughness:0.9});
  (function(){
    var hpIdx=490;
    var hpP=wpPerp(hpIdx);
    var stackOff=TW*0.5+CURB+3.5;
    [-2,-0.7,0.7,2].forEach(function(along){
      var aWp=waypoints[(hpIdx+Math.round(along*8)+N)%N];
      [mTireW,mTireR,mTireW].forEach(function(mt,ti2){
        var tc=new THREE.Mesh(new THREE.CylinderGeometry(0.6,0.6,0.9,12),mt);
        tc.position.set(aWp.x+hpP.x*stackOff,aWp.y+ti2*0.9+0.45,aWp.z+hpP.z*stackOff);
        scene.add(tc);
      });
    });
  })();
  /* Tyre walls lining the outside of every corner (outer side resolved from the turn vector) */
  (function(){
    var off=TW*0.5+CURB+1.1,last=-100;
    for(var ci=0;ci<N;ci+=5){
      var dA=wpDir((ci-8+N)%N),dB=wpDir((ci+8)%N);
      if(Math.abs(dA.x*dB.z-dA.z*dB.x)<0.45||ci-last<55) continue;
      last=ci;
      var inx=dB.x-dA.x,inz=dB.z-dA.z;        // points toward the corner's inside
      for(var a=-3;a<=3;a++){
        var wi=(ci+a*3+N)%N,wp=waypoints[wi],pp=wpPerp(wi);
        var side=(pp.x*inx+pp.z*inz)>0?-1:1;  // place on the opposite (outer) side
        var bx=wp.x+pp.x*side*off,bz=wp.z+pp.z*side*off;
        [mTireW,mTireR,mTireW].forEach(function(mt,h){
          var tc=new THREE.Mesh(new THREE.CylinderGeometry(0.55,0.55,0.85,12),mt);
          tc.position.set(bx,wp.y+0.45+h*0.85,bz);scene.add(tc);
        });
      }
    }
  })();
  /* Brake distance boards — 100/75/50m markers at each corner */
  (function(){
    var bmPole=new THREE.MeshStandardMaterial({color:0xbbbbbb,roughness:0.5,metalness:0.4});
    var bmPGeo=new THREE.CylinderGeometry(0.06,0.07,5,5);
    var bmDists=[{d:41,n:'100'},{d:31,n:'75'},{d:20,n:'50'}];
    var bmCorners=[],bmLast=-100;
    for(var bqi=0;bqi<N;bqi+=5){
      var bmDA=wpDir(bqi),bmDB=wpDir((bqi+10)%N);
      var bmCross=Math.abs(bmDA.x*bmDB.z-bmDA.z*bmDB.x);
      if(bmCross>0.42&&bqi-bmLast>60){bmCorners.push(bqi);bmLast=bqi;}
    }
    bmCorners.slice(0,10).forEach(function(bci){
      bmDists.forEach(function(bmd){
        var bmwi=(bci-bmd.d+N)%N;
        var bmwp=waypoints[bmwi],bmpp=wpPerp(bmwi),bmdir=wpDir(bmwi);
        var bmAng=Math.atan2(bmdir.x,bmdir.z);
        var bmOff=TW*0.5+CURB+BH+4.0;
        [1,-1].forEach(function(bms){
          var bmpx=bmwp.x+bmpp.x*bms*bmOff,bmpz=bmwp.z+bmpp.z*bms*bmOff;
          var bmPl=new THREE.Mesh(bmPGeo,bmPole);
          bmPl.position.set(bmpx,bmwp.y+2.5,bmpz);scene.add(bmPl);
          var bCV=document.createElement('canvas');bCV.width=128;bCV.height=192;
          var bCTX=bCV.getContext('2d');
          bCTX.fillStyle='#111111';bCTX.fillRect(0,0,128,192);
          bCTX.fillStyle='#ffffff';bCTX.font='bold 120px Arial';
          bCTX.textAlign='center';bCTX.textBaseline='middle';bCTX.fillText(bmd.n,64,100);
          var bTX=new THREE.CanvasTexture(bCV);
          var bmBd=new THREE.Mesh(new THREE.BoxGeometry(0.6,0.9,0.08),
            new THREE.MeshStandardMaterial({map:bTX,roughness:0.8}));
          bmBd.position.set(bmpx,bmwp.y+4.5,bmpz);bmBd.rotation.y=bmAng;scene.add(bmBd);
        });
      });
    });
  })();
})();

/* NB: we deliberately do NOT set scene.environment. A PMREM env map of the night
   scene zeroes out direct/ambient lighting on every MeshStandardMaterial in r128,
   turning the whole world black. Car paint instead picks up glossy specular glints
   from the floodlights/moonlight via its clearcoat + the direct lights. */

/* ===================== CAR BUILDER ===================== */
// Shared procedural carbon-weave map for the floor/wings/diffuser — built once, reused on every car.
var _carbonTex=null;
function carbonTex(){
  if(_carbonTex) return _carbonTex;
  var s=128,cv=document.createElement('canvas');cv.width=cv.height=s;var x=cv.getContext('2d');
  x.fillStyle='#101013';x.fillRect(0,0,s,s);
  for(var yy=0;yy<s;yy+=8)for(var xx=0;xx<s;xx+=8){
    var over=(((xx/8)+(yy/8))%2)===0,g=x.createLinearGradient(xx,yy,xx+8,yy+8);
    var a=over?['#2a2a30','#15151a','#0c0c0f']:['#0c0c0f','#15151a','#2a2a30'];
    g.addColorStop(0,a[0]);g.addColorStop(0.5,a[1]);g.addColorStop(1,a[2]);
    x.fillStyle=g;x.fillRect(xx,yy,8,8);
  }
  var t=new THREE.CanvasTexture(cv);t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(3,3);
  _carbonTex=t;return t;
}
// Race-number roundel texture (cached per number/colour) for the nose flanks.
var _numTexCache={};
function numTex(n,bgHex,fgHex){
  var key=n+bgHex+fgHex;if(_numTexCache[key]) return _numTexCache[key];
  var s=128,cv=document.createElement('canvas');cv.width=cv.height=s;var x=cv.getContext('2d');
  x.clearRect(0,0,s,s);
  x.fillStyle=bgHex;x.beginPath();x.arc(s/2,s/2,s*0.47,0,6.283);x.fill();
  x.lineWidth=s*0.05;x.strokeStyle=fgHex;x.stroke();
  x.fillStyle=fgHex;x.font='bold '+Math.round(s*0.6)+'px Arial,sans-serif';
  x.textAlign='center';x.textBaseline='middle';x.fillText(''+n,s/2,s*0.56);
  var t=new THREE.CanvasTexture(cv);_numTexCache[key]=t;return t;
}
function buildCar(bodyColor,accentColor,carNum){
  var g=new THREE.Group();
  var PI=Math.PI;
  var mNav=new THREE.MeshPhysicalMaterial({color:bodyColor,metalness:0.12,roughness:0.22,clearcoat:1.0,clearcoatRoughness:0.05});
  var mRed=new THREE.MeshPhysicalMaterial({color:accentColor,metalness:0.04,roughness:0.23,clearcoat:1.0,clearcoatRoughness:0.04});
  var mGold=new THREE.MeshPhysicalMaterial({color:0xC9A85C,metalness:0.84,roughness:0.12,clearcoat:0.55,clearcoatRoughness:0.14});
  var mC=new THREE.MeshPhysicalMaterial({color:0xffffff,map:carbonTex(),metalness:0.30,roughness:0.42,clearcoat:0.55,clearcoatRoughness:0.22});
  var mT=new THREE.MeshStandardMaterial({color:0x030303,metalness:0.0,roughness:0.98});
  var mR=new THREE.MeshPhysicalMaterial({color:0xBBBBBB,metalness:0.96,roughness:0.03,clearcoat:0.3});
  var mG=new THREE.MeshStandardMaterial({color:0x888888,metalness:0.74,roughness:0.28});
  var mRim=new THREE.MeshPhysicalMaterial({color:0x202024,metalness:0.92,roughness:0.34,clearcoat:0.4});
  var mRimAcc=new THREE.MeshStandardMaterial({color:accentColor,metalness:0.55,roughness:0.40});
  var mPir=new THREE.MeshStandardMaterial({color:0xE10600,emissive:0x300000,emissiveIntensity:0.45,roughness:0.7});
  function mk(geo,mat,x,y,z,rx,ry,rz){var m=new THREE.Mesh(geo,mat);m.position.set(x||0,y||0,z||0);m.rotation.set(rx||0,ry||0,rz||0);return m;}
  function bx(w,h,d,mat,x,y,z,rx,ry,rz){return mk(new THREE.BoxGeometry(w,h,d),mat,x,y,z,rx,ry,rz);}
  function cy(r1,r2,h,s,mat,x,y,z,rx,ry,rz){return mk(new THREE.CylinderGeometry(r1,r2,h,s),mat,x,y,z,rx,ry,rz);}
  function el(erx,ery,depth,mat,x,y,z,rxr,ryr,rzr){var m=new THREE.Mesh(new THREE.CylinderGeometry(1,1,depth,32),mat);m.scale.set(erx,1,ery);m.position.set(x||0,y||0,z||0);m.rotation.set(rxr||0,ryr||0,rzr||0);return m;}
  function wing(span,chord,thick,mat,x,y,z,ryRot){var sh=new THREE.Shape(),t=thick*0.5;sh.moveTo(0,0);sh.bezierCurveTo(chord*0.1,t,chord*0.4,t,chord,0);sh.bezierCurveTo(chord*0.4,-t,chord*0.1,-t,0,0);var geo=new THREE.ExtrudeGeometry(sh,{depth:span,bevelEnabled:false,steps:1});geo.translate(0,0,-span*0.5);var m=new THREE.Mesh(geo,mat);m.position.set(x||0,y||0,z||0);m.rotation.y=ryRot||0;return m;}
  function tube3(ax,ay,az,bx2,by2,bz2,cx2,cy2,cz2,r,mat){var crv=new THREE.QuadraticBezierCurve3(new THREE.Vector3(ax,ay,az),new THREE.Vector3(bx2,by2,bz2),new THREE.Vector3(cx2,cy2,cz2));return new THREE.Mesh(new THREE.TubeGeometry(crv,20,r,8,false),mat);}
  function bar(ax,ay,az,ex,ey,ez,r,mat){var dx=ex-ax,dy=ey-ay,dz=ez-az,len=Math.sqrt(dx*dx+dy*dy+dz*dz);var m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,len,6),mat);m.position.set((ax+ex)/2,(ay+ey)/2,(az+ez)/2);var q=new THREE.Quaternion();q.setFromUnitVectors(new THREE.Vector3(0,1,0),new THREE.Vector3(dx/len,dy/len,dz/len));m.setRotationFromQuaternion(q);return m;}
  function add(m){g.add(m);}
  // Chassis
  add(el(0.31,0.20,0.70,mNav, 1.25,0.05,0,0,0,PI/2));
  add(el(0.35,0.21,0.80,mNav, 0.50,0.05,0,0,0,PI/2));
  add(el(0.26,0.19,0.50,mNav,-0.15,0.05,0,0,0,PI/2));
  add(el(0.31,0.21,0.50,mNav,-0.65,0.05,0,0,0,PI/2));
  add(el(0.275,0.18,0.75,mNav,-1.28,0.05,0,0,0,PI/2));
  add(el(0.27,0.105,1.10,mNav,-0.30,0.31,0,0,0,PI/2));
  add(el(0.30,0.065,0.58,mNav, 0.53,0.27,0,0,0,PI/2));
  // Nose
  var nosePts=[new THREE.Vector2(0.28,0),new THREE.Vector2(0.27,0.08),new THREE.Vector2(0.24,0.26),new THREE.Vector2(0.21,0.46),new THREE.Vector2(0.17,0.68),new THREE.Vector2(0.12,0.90),new THREE.Vector2(0.07,1.14),new THREE.Vector2(0.04,1.32),new THREE.Vector2(0.02,1.46)];
  var noseMesh=new THREE.Mesh(new THREE.LatheGeometry(nosePts,32),mNav);noseMesh.rotation.z=-PI/2;noseMesh.position.set(1.60,-0.01,0);add(noseMesh);
  add(cy(0.022,0.022,0.06,8,mRed,3.10,-0.01,0,0,0,-PI/2));
  // Front wing
  add(wing(2.12,0.30,0.040,mC,2.86,-0.245,0,0));add(wing(1.94,0.24,0.036,mC,2.66,-0.200,0,0));add(wing(1.74,0.18,0.030,mC,2.48,-0.158,0,0));
  [-1.03,1.03].forEach(function(z){add(bx(0.50,0.32,0.05,mRed,2.66,-0.095,z));add(bx(0.20,0.10,0.05,mRed,2.84,-0.30,z));});
  [-0.20,0.20].forEach(function(z){add(bx(0.06,0.24,0.04,mC,2.72,-0.075,z));});
  // Sidepods
  [-0.53,0.53].forEach(function(z){
    var sg=z>0?1:-1;
    var sf=new THREE.Mesh(new THREE.CylinderGeometry(0.175,0.145,0.85,24),mNav);sf.scale.set(1,1,1.9);sf.rotation.z=PI/2;sf.position.set(0.10,-0.03,z+sg*0.02);add(sf);
    var sr2=new THREE.Mesh(new THREE.CylinderGeometry(0.130,0.090,0.77,24),mNav);sr2.scale.set(1,1,1.7);sr2.rotation.z=PI/2;sr2.position.set(-0.565,-0.03,z+sg*0.02);add(sr2);
    add(bx(0.90,0.32,0.012,mRed,-0.16,-0.03,z+sg*0.166));add(bx(0.09,0.21,0.09,mC,0.37,0.04,z));add(bx(1.22,0.08,0.26,mC,-0.39,-0.23,z));
  });
  // Bargeboards
  for(var bi=0;bi<3;bi++){[-0.34-bi*0.09,0.34+bi*0.09].forEach(function(z){add(bx(0.09,0.20,0.03,mC,0.84,-0.04,z));});}
  // Engine cover
  var ecPts=[new THREE.Vector2(0,0),new THREE.Vector2(0.085,0.10),new THREE.Vector2(0.195,0.26),new THREE.Vector2(0.260,0.35),new THREE.Vector2(0.240,0.25),new THREE.Vector2(0.185,0.10),new THREE.Vector2(0,0)];
  var ecMesh=new THREE.Mesh(new THREE.LatheGeometry(ecPts,20),mNav);ecMesh.rotation.z=PI/2;ecMesh.scale.set(1,0.054,1);ecMesh.position.set(-0.20,0.18,0);add(ecMesh);
  add(bx(0.70,0.040,0.054,mGold,-0.20,0.71,0));
  var riMesh=new THREE.Mesh(new THREE.CylinderGeometry(0.16,0.16,0.19,20),mNav);riMesh.scale.set(1,1,2.0);riMesh.position.set(0.29,0.39,0);add(riMesh);
  // Halo
  add(cy(0.028,0.028,0.34,12,mG,0.60,0.52,0));
  add(tube3(0.40,0.70,-0.28,0.57,0.82,0,0.40,0.70,0.28,0.034,mG));
  add(bar(0.40,0.70,-0.28,0.18,0.30,-0.26,0.022,mG));add(bar(0.40,0.70,0.28,0.18,0.30,0.26,0.022,mG));
  // Mirrors
  add(cy(0.016,0.016,0.28,8,mC,0.50,0.38,-0.26));add(cy(0.016,0.016,0.28,8,mC,0.50,0.38,0.26));
  add(bx(0.11,0.07,0.17,mR,0.47,0.53,-0.26));add(bx(0.11,0.07,0.17,mR,0.47,0.53,0.26));
  // Helmet
  var helm=mk(new THREE.SphereGeometry(0.14,24,18),mNav,0.41,0.36,0);helm.scale.set(1.2,0.92,1.1);add(helm);
  // Floor + diffuser
  add(bx(3.12,0.042,1.80,mC,-0.08,-0.21,0));
  for(var ds=-3;ds<=3;ds++){add(bx(0.64,0.13,0.03,mC,-1.83,-0.15,ds*0.245));}
  add(bx(0.42,0.030,1.60,mC,-1.830,-0.146,0,0,0,2.719));
  // Rear wing
  add(bx(0.33,0.23,0.42,mNav,-1.71,0.07,0));add(bx(0.54,0.065,0.90,mC,-1.85,0.19,0));
  add(bar(-1.62,0.185,-0.13,-2.00,0.661,-0.18,0.018,mC));add(bar(-1.62,0.185,0.13,-2.00,0.661,0.18,0.018,mC));
  add(wing(1.60,0.27,0.058,mC,-2.03,0.69,0,0));add(wing(1.48,0.18,0.046,mC,-1.87,0.63,0,0));
  [-0.80,0.80].forEach(function(z){add(bx(0.31,0.58,0.048,mRed,-1.97,0.41,z));});
  // Lights — emissive so bloom picks them up; intensities pulsed from the loop
  var mTail=new THREE.MeshStandardMaterial({color:0x3a0000,emissive:0xff0000,emissiveIntensity:1.5,roughness:0.5});
  var mDrs=new THREE.MeshStandardMaterial({color:0x0a0a0a,emissive:0x00ff66,emissiveIntensity:0.0,roughness:0.5});
  var mHead=new THREE.MeshStandardMaterial({color:0x222222,emissive:0xfff2dc,emissiveIntensity:0.45,roughness:0.5});
  add(bx(0.07,0.16,0.13,mTail,-2.07,0.00,0));                         // rear rain light
  [-0.80,0.80].forEach(function(z){add(bx(0.05,0.10,0.16,mTail,-2.05,0.30,z));}); // endplate tail strips
  add(bx(0.04,0.05,0.16,mDrs,-2.07,0.58,0));                          // DRS tell-tale
  [-0.12,0.12].forEach(function(z){add(bx(0.04,0.05,0.06,mHead,2.90,-0.05,z));}); // nose markers
  g.userData.tailMat=mTail;g.userData.drsMat=mDrs;
  // Suspension
  [[1.72,-0.80],[1.72,0.80]].forEach(function(w){var wx=w[0],wz=w[1],ci=wz>0?0.22:-0.22;add(bar(wx,0.00,wz,1.52,0.08,ci,0.016,mG));add(bar(wx,0.00,wz,1.18,0.06,ci,0.016,mG));add(bar(wx,-0.28,wz,1.50,-0.22,ci,0.016,mG));add(bar(wx,-0.28,wz,1.16,-0.22,ci,0.016,mG));});
  [[-1.52,-0.88],[-1.52,0.88]].forEach(function(w){var wx=w[0],wz=w[1],ci=wz>0?0.24:-0.24;add(bar(wx,0.00,wz,-1.10,0.06,ci,0.016,mG));add(bar(wx,0.00,wz,-1.42,0.04,ci,0.016,mG));add(bar(wx,-0.28,wz,-1.12,-0.20,ci,0.016,mG));add(bar(wx,-0.28,wz,-1.44,-0.18,ci,0.016,mG));});
  // Wheels
  function addWheel(x,z,tw){
    var wg=new THREE.Group();var fs=(z>0)?1:-1,fY=fs*tw*0.46;
    var R=0.340,ri=0.260,hw=tw*0.50;
    var tp=[new THREE.Vector2(ri,hw+0.004),new THREE.Vector2(ri+0.022,hw-0.002),new THREE.Vector2(R-0.030,hw-0.004),new THREE.Vector2(R-0.006,hw-0.026),new THREE.Vector2(R,hw-0.056),new THREE.Vector2(R,0),new THREE.Vector2(R,-(hw-0.056)),new THREE.Vector2(R-0.006,-(hw-0.026)),new THREE.Vector2(R-0.030,-(hw-0.004)),new THREE.Vector2(ri+0.022,-(hw-0.002)),new THREE.Vector2(ri,-(hw+0.004))];
    wg.add(mk(new THREE.LatheGeometry(tp,52),mT,0,0,0));
    wg.add(mk(new THREE.CylinderGeometry(ri-0.002,ri-0.002,tw+0.006,44),mRim,0,0,0));
    wg.add(mk(new THREE.CylinderGeometry(ri-0.004,ri-0.004,0.026,44),mRimAcc,0,fY,0));
    wg.add(mk(new THREE.CylinderGeometry(0.065,0.065,tw+0.032,16),mG,0,0,0));
    wg.add(mk(new THREE.CylinderGeometry(R-0.008,R-0.008,0.05,40,1,true),mPir,0,fY*0.55,0)); // Pirelli compound stripe
    for(var wi=0;wi<5;wi++){var pv=new THREE.Group();pv.rotation.y=wi*2*PI/5;pv.position.y=fY;var sp=new THREE.Mesh(new THREE.BoxGeometry(ri-0.044,0.020,0.020),mRim);sp.position.x=(ri-0.044)/2;pv.add(sp);wg.add(pv);}
    wg.rotation.x=PI/2;wg.position.set(x,-0.22,z);g.add(wg);
  }
  addWheel(1.72,-0.80,0.300);addWheel(1.72,0.80,0.300);
  addWheel(-1.52,-0.88,0.405);addWheel(-1.52,0.88,0.405);
  // Race-number roundel on each nose flank
  if(carNum!=null){
    var nbg='#'+('000000'+accentColor.toString(16)).slice(-6);
    var nMat=new THREE.MeshBasicMaterial({map:numTex(carNum,nbg,'#ffffff'),transparent:true});
    [0.205,-0.205].forEach(function(nz){
      var np=new THREE.Mesh(new THREE.PlaneGeometry(0.34,0.34),nMat);
      np.position.set(2.02,0.12,nz);np.rotation.y=nz>0?0:PI;g.add(np);
    });
  }
  g.traverse(function(o){if(o.isMesh){o.castShadow=true;o.receiveShadow=true;}});
  return g;
}

/* ===================== GAME STATE ===================== */
var gameState='IDLE',paused=false,raceTime=0,animRunning=false;
var playerGrp=null,aiGrps=[],aiLabels=[];
var camSmooth=new THREE.Vector3(),camVel=new THREE.Vector3();
var cockpitMode=false;

var P={x:0,z:0,y:0,heading:0,speed:0,yawRate:0,tIdx:0,lap:1,
  sector:0,sectorStart:0,lapStart:0,s1:0,s2:0,s3:0,
  bestS1:Infinity,bestS2:Infinity,bestS3:Infinity,bestLap:Infinity,
  rpmVal:0,gear:1,drs:false,_drsKey:false,
  tireWear:1.0,tireTempF:0.3,tireTempR:0.3,_lapGuard:false};

// Standing-grid slots: staggered 2-wide on the flat S/F straight (wp ~47..104, clear of Turn 1 ~135).
// Position p (0=pole) -> row/column; consecutive slots step back ~3 wp so no two cars sit exactly abreast.
var GRID_FRONT=104,GRID_DY=6,PLAYER_GRID=9;
function gridSlot(p){
  var row=Math.floor(p/2),col=p%2;
  return {ti:GRID_FRONT-row*GRID_DY-col*3,lat:(col===0?-1:1)*TW*0.22};
}
var AI=(function(){
  var a=[];
  // 5 distinct lane positions, wide enough that TW*0.16 same-lane detection doesn't bleed across lanes
  var LANES=[-TW*0.36,-TW*0.18,0,TW*0.18,TW*0.36];
  for(var i=0;i<19;i++){
    var bl=LANES[i%5];
    // Unique apex depth per car: 0.42–0.74, deterministic so fast cars stay consistent
    var apxD=0.42+((i*3)%7)*0.090+Math.random()*0.03;
    var aggr=0.90+Math.random()*0.50;
    // Fastest cars (low i) start nearest pole; player takes grid slot PLAYER_GRID (~P10).
    var gp=i<PLAYER_GRID?i:i+1,slot=gridSlot(gp);
    var initTIdx=slot.ti,initSide=(i%2===0)?1:-1;
    a.push({tIdx:initTIdx,_initTIdx:initTIdx,latOff:slot.lat,_gridLat:slot.lat,spdFac:1.36-i*0.012,
            _randFac:Math.random()*0.025,apexD:apxD,_aggr:aggr,_ovSide:initSide,_initOvSide:initSide,
            _ovTgt:bl,_ovTimer:0.0,finishTime:Infinity,
            x:0,z:0,y:0,heading:0,speed:0,yawRate:0,lap:1,_inStart:false,
            _stuckTimer:0,
            _latTarget:bl,_baseLat:bl,_initLat:bl});
  }
  return a;
})();

var keys={};
document.addEventListener('keydown',function(e){
  keys[e.code]=true;
  if(e.code==='Escape'&&gameState!=='IDLE'){
    if(gameState==='FINISHED'){closeGame();return;}
    paused=!paused;
    if(pauseOvl) pauseOvl.className=paused?'active':'';
  }
  if(e.code==='KeyQ'&&paused) closeGame();
  if(e.code==='KeyC') cockpitMode=!cockpitMode;
  if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space'].indexOf(e.code)>=0&&gameState!=='IDLE') e.preventDefault();
});
document.addEventListener('keyup',function(e){keys[e.code]=false;});

/* ===================== SPAWN ===================== */
function spawnPos(ti,lat){
  var idx=((ti%N)+N)%N;
  var wp=waypoints[idx],p=wpPerp(idx),d=wpDir(idx);
  return {x:wp.x+p.x*lat,z:wp.z+p.z*lat,y:wp.y,heading:Math.atan2(d.x,d.z)};
}

function makeLabel(driverName,abbr,bodyHex,accentHex){
  var cv=document.createElement('canvas');cv.width=256;cv.height=72;
  var ctx=cv.getContext('2d');
  function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();}
  // Dark background pill
  ctx.fillStyle='rgba(0,0,0,0.78)';rr(0,0,256,72,10);ctx.fill();
  // Team badge (left block)
  var br=(bodyHex>>16)&255,bg=(bodyHex>>8)&255,bb=bodyHex&255;
  ctx.fillStyle='rgb('+br+','+bg+','+bb+')';rr(5,5,58,62,7);ctx.fill();
  // Badge text
  var ar=(accentHex>>16)&255,ag=(accentHex>>8)&255,ab=accentHex&255;
  ctx.fillStyle='rgb('+ar+','+ag+','+ab+')';
  ctx.font='bold 15px Arial';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(abbr,34,36);
  // Driver name
  ctx.fillStyle='#ffffff';ctx.font='bold 21px Arial';ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillText(driverName,74,36);
  var tex=new THREE.CanvasTexture(cv);
  var spr=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false}));
  spr.scale.set(3.2,0.9,1);
  return spr;
}


function initRace(){
  if(playerGrp){scene.remove(playerGrp);playerGrp=null;}
  aiGrps.forEach(function(gr){scene.remove(gr);});aiGrps=[];
  aiLabels.forEach(function(sl){scene.remove(sl);});aiLabels=[];
  playerGrp=buildCar(0x1A3E82,0xCC1E1E,1);scene.add(playerGrp);
  var pslot=gridSlot(PLAYER_GRID),sp=spawnPos(pslot.ti,pslot.lat);
  P.x=sp.x;P.z=sp.z;P.y=sp.y;P.heading=sp.heading;
  P.speed=0;P.yawRate=0;P.tIdx=pslot.ti;P.lap=1;P.rpmVal=0;P.gear=1;
  P.drs=false;P._drsKey=false;P.drsArmed=false;P.tireWear=1;P.tireTempF=0.3;P.tireTempR=0.3;
  P.sector=0;P.sectorStart=0;P.lapStart=0;P.s1=0;P.s2=0;P.s3=0;
  P.bestS1=Infinity;P.bestS2=Infinity;P.bestS3=Infinity;P.bestLap=Infinity;P._lapGuard=false;P.finishTime=Infinity;
  AI.forEach(function(ai,i){
    var ag=buildCar(AI_COLORS[i],AI_ACCENTS[i],AI_NUMS[i]);scene.add(ag);aiGrps.push(ag);
    var lbl=makeLabel(AI_NAMES[i+1],AI_TEAMS[i],AI_COLORS[i],AI_ACCENTS[i]);scene.add(lbl);aiLabels.push(lbl);
    ai.tIdx=ai._initTIdx;ai._latTarget=ai._gridLat;ai._baseLat=ai._initLat;
    ai._ovTimer=0;ai._stuckTimer=0;ai._ovSide=ai._initOvSide;ai._ovTgt=ai._initLat;
    var as=spawnPos(ai.tIdx,ai._gridLat);
    ai.x=as.x;ai.z=as.z;ai.y=as.y;ai.heading=as.heading;
    ai.speed=0;ai.yawRate=0;ai.lap=1;ai._inStart=false;ai.finishTime=Infinity;
    ag.position.set(ai.x,ai.y+RIDE_H,ai.z);ag.rotation.y=ai.heading-Math.PI/2;
    lbl.position.set(ai.x,ai.y+RIDE_H+3.2,ai.z);
  });
  playerGrp.position.set(P.x,P.y+RIDE_H,P.z);playerGrp.rotation.y=P.heading-Math.PI/2;
  var d0=wpDir(P.tIdx);
  gameCam.position.set(P.x-d0.x*28,P.y+13,P.z-d0.z*28);
  camSmooth.copy(gameCam.position);camVel.set(0,0,0);
  gameCam.lookAt(new THREE.Vector3(P.x+d0.x*8,P.y+0.5,P.z+d0.z*8));
  gameCam.fov=80;gameCam.updateProjectionMatrix();
}

/* ===================== COUNTDOWN ===================== */
var lightEls=[
  document.getElementById('lb0'),document.getElementById('lb1'),
  document.getElementById('lb2'),document.getElementById('lb3'),
  document.getElementById('lb4')
];

function startCountdown(){
  gameState='COUNTDOWN';raceTime=0;
  if(lightsBarEl) lightsBarEl.className='active';
  if(gridMsgEl) gridMsgEl.className='active';
  lightEls.forEach(function(l){if(l) l.className='light-bulb';});
  for(var li=0;li<5;li++){
    (function(idx){
      setTimeout(function(){if(lightEls[idx]) lightEls[idx].className='light-bulb lit';},idx*750);
    })(li);
  }
  setTimeout(function(){
    lightEls.forEach(function(l){if(l) l.className='light-bulb';});
    if(lightsBarEl) lightsBarEl.className='';
    if(gridMsgEl) gridMsgEl.className='';
    gameState='RACING';raceTime=0;P.lapStart=0;P.sectorStart=0;
    if(goFlashEl){goFlashEl.style.display='block';setTimeout(function(){goFlashEl.style.display='none';},180);}
  },5*750+400);
}

/* ===================== PLAYER PHYSICS ===================== */
/* ===================== TIRE SMOKE ===================== */
// Pooled sprite particles: lockups (front axle), wheelspin/slides (rear axle), wall brushes.
var SMOKE=(function(){
  var POOL=150,parts=[],tex=null,idx=0;
  function mkTex(){
    var c=document.createElement('canvas');c.width=c.height=64;var g=c.getContext('2d');
    var rg=g.createRadialGradient(32,32,1,32,32,32);
    rg.addColorStop(0,'rgba(224,228,232,0.85)');rg.addColorStop(0.5,'rgba(198,204,212,0.40)');
    rg.addColorStop(1,'rgba(198,204,212,0)');
    g.fillStyle=rg;g.fillRect(0,0,64,64);return new THREE.CanvasTexture(c);
  }
  function init(){
    if(tex) return;tex=mkTex();
    for(var i=0;i<POOL;i++){
      var m=new THREE.SpriteMaterial({map:tex,transparent:true,depthWrite:false,opacity:0});
      var sp=new THREE.Sprite(m);sp.visible=false;scene.add(sp);
      parts.push({sp:sp,life:0,max:1,vx:0,vy:0,vz:0,s0:1,s1:3});
    }
  }
  function puff(x,y,z,vx,vy,vz,s0,s1,life){
    if(!tex) init();
    var p=parts[idx];idx=(idx+1)%POOL;
    p.life=life;p.max=life;p.vx=vx;p.vy=vy;p.vz=vz;p.s0=s0;p.s1=s1;
    p.sp.position.set(x,y,z);p.sp.scale.set(s0,s0,s0);p.sp.material.opacity=0;p.sp.visible=true;
  }
  function emit(car,axle,n,intensity){
    if(!tex) init();
    var fx=Math.sin(car.heading),fz=Math.cos(car.heading),sx=fz,sz=-fx;
    var fwd=axle>0?2.0:-1.9;
    for(var w=-1;w<=1;w+=2){
      var wx=car.x+fx*fwd+sx*w*1.0,wz=car.z+fz*fwd+sz*w*1.0,wy=(car.y||0)+0.25;
      for(var k=0;k<n;k++){
        var sp=(Math.random()-0.5);
        puff(wx,wy,wz,
          -fx*car.speed*0.035+sx*sp*1.1+(Math.random()-0.5)*0.4,
          0.6+Math.random()*1.1,
          -fz*car.speed*0.035+sz*sp*1.1+(Math.random()-0.5)*0.4,
          0.9,2.6+intensity*2.2,0.45+Math.random()*0.3);
      }
    }
  }
  function update(dt){
    if(!tex) return;
    for(var i=0;i<parts.length;i++){
      var p=parts[i];if(p.life<=0) continue;
      p.life-=dt;
      if(p.life<=0){p.sp.visible=false;p.sp.material.opacity=0;continue;}
      var t=1-p.life/p.max;
      p.sp.position.x+=p.vx*dt;p.sp.position.y+=p.vy*dt;p.sp.position.z+=p.vz*dt;
      p.vy*=(1-dt*0.6);p.vx*=(1-dt*1.5);p.vz*=(1-dt*1.5);
      var sc=p.s0+(p.s1-p.s0)*t;p.sp.scale.set(sc,sc,sc);
      p.sp.material.opacity=Math.max(0,1-t)*0.5;
    }
  }
  return {emit:emit,update:update,init:init};
})();

/* ===================== CROWD CAMERA FLASHES ===================== */
var FLASH=(function(){
  var POOL=18,fl=[],tex=null,idx=0,timer=0;
  function mkTex(){
    var c=document.createElement('canvas');c.width=c.height=32;var g=c.getContext('2d');
    var rg=g.createRadialGradient(16,16,0,16,16,16);
    rg.addColorStop(0,'rgba(255,255,255,1)');rg.addColorStop(0.4,'rgba(255,255,235,0.6)');rg.addColorStop(1,'rgba(255,255,235,0)');
    g.fillStyle=rg;g.fillRect(0,0,32,32);return new THREE.CanvasTexture(c);
  }
  function init(){
    if(tex) return;tex=mkTex();
    for(var i=0;i<POOL;i++){
      var m=new THREE.SpriteMaterial({map:tex,transparent:true,depthWrite:false,opacity:0,blending:THREE.AdditiveBlending,fog:false});
      var sp=new THREE.Sprite(m);sp.scale.set(2.4,2.4,1);sp.visible=false;scene.add(sp);fl.push({sp:sp,life:0});
    }
  }
  function update(dt){
    if(!standAnchors.length) return;
    if(!tex) init();
    if(gameState==='RACING'||gameState==='COUNTDOWN'){
      timer-=dt;
      if(timer<=0){timer=0.04+Math.random()*0.11;
        var a=standAnchors[(Math.random()*standAnchors.length)|0],f=fl[idx];idx=(idx+1)%POOL;
        f.life=0.11;
        f.sp.position.set(a.x+(Math.random()-0.5)*a.w*1.6,a.y+Math.random()*5,a.z+(Math.random()-0.5)*a.w*1.6);
        f.sp.visible=true;f.sp.material.opacity=1;
      }
    }
    for(var i=0;i<fl.length;i++){var f=fl[i];if(f.life<=0) continue;f.life-=dt;
      if(f.life<=0){f.sp.visible=false;f.sp.material.opacity=0;}else f.sp.material.opacity=f.life/0.11;}
  }
  return {update:update};
})();

/* ===================== DRS ZONES ===================== */
// Activation zones on the two fast straights; player must be within ~1s of the car ahead to arm.
var DRS_ZONES=[{s:868,e:112},{s:545,e:645}];
function drsZoneIndex(ti){
  for(var i=0;i<DRS_ZONES.length;i++){
    var z=DRS_ZONES[i];
    if(z.s<=z.e){if(ti>=z.s&&ti<=z.e) return i;}
    else{if(ti>=z.s||ti<=z.e) return i;}
  }
  return -1;
}
function gapAheadSec(car){
  var best=1e9;
  for(var i=0;i<AI.length;i++){
    var o=AI[i];if(o===car) continue;
    var g=((o.tIdx-car.tIdx)+N)%N;if(g>0&&g<best) best=g;
  }
  if(car!==P){var gp=((P.tIdx-car.tIdx)+N)%N;if(gp>0&&gp<best) best=gp;}
  return best*2.31/Math.max(car.speed,1);
}

function updatePlayer(dt){
  var thr=(keys['ArrowUp']||keys['KeyW'])?1:0;
  var brk=(keys['ArrowDown']||keys['KeyS'])?1:0;
  P.thr=thr;P.brk=brk;
  var sl=(keys['ArrowLeft']||keys['KeyA'])?1:0;
  var sr=(keys['ArrowRight']||keys['KeyD'])?1:0;
  var si=sl-sr;P.steerIn=si;
  // DRS: armed only inside a zone when within ~1s of the car ahead; opens on Shift, shuts on brake/zone-exit
  var inZone=drsZoneIndex(P.tIdx)>=0;
  P.drsArmed=inZone&&gapAheadSec(P)<1.0;
  if((keys['ShiftLeft']||keys['ShiftRight'])&&P.drsArmed) P.drs=true;
  if(brk>0||!inZone||!P.drsArmed) P.drs=false;
  if(hudDrs) hudDrs.className='hud-drs'+(P.drs?' on':(P.drsArmed?' armed':''));
  if(gameState==='COUNTDOWN'){
    P.rpmVal=thr?Math.min(P.rpmVal+dt*3,1):Math.max(P.rpmVal-dt*4,0);
    return;
  }
  if(gameState!=='RACING') return;
  var maxSpd=P.drs?MAX_SPD*1.08:MAX_SPD;
  var acc=(thr*ENG-brk*BRK-DRAG*P.speed*P.speed)/CAR_MASS;
  P.speed=Math.max(0,Math.min(P.speed+acc*dt,maxSpd));
  var ms=Math.PI/5*(1-0.6*P.speed/MAX_SPD);
  var ty=si*ms*Math.min(P.speed/5,1);
  P.yawRate+=(ty-P.yawRate)*Math.min(dt*6,1);P.yawRate*=Math.max(0,1-dt*7.5);
  // Cornering grip: too much lateral load and the tires give up — the car understeers and scrubs speed,
  // so you have to brake for the Esses/hairpin instead of holding top speed through them.
  var latLoad=Math.abs(P.yawRate)*P.speed;
  if(latLoad>GRIP_LIMIT){P.speed=Math.max(0,P.speed-(latLoad-GRIP_LIMIT)*GRIP_SCRUB*dt);P.gripLoss=true;}
  else P.gripLoss=false;
  P.heading+=P.yawRate*P.speed*0.6*dt;
  P.x+=Math.sin(P.heading)*P.speed*dt;
  P.z+=Math.cos(P.heading)*P.speed*dt;
  P.tIdx=closestWP(P.x,P.z,P.tIdx);
  P.y=waypoints[P.tIdx].y;
  var _bwp=waypoints[P.tIdx],_bp=wpPerp(P.tIdx);
  var _lat=(P.x-_bwp.x)*_bp.x+(P.z-_bwp.z)*_bp.z;
  var _maxL=TW*0.5+CURB;
  if(Math.abs(_lat)>_maxL){var _sgn=_lat>0?1:-1;P.x=_bwp.x+_bp.x*_sgn*_maxL;P.z=_bwp.z+_bp.z*_sgn*_maxL;P.speed*=0.45;}
  var GB=[0,0.12,0.25,0.38,0.54,0.70,0.86,1.0];
  var rpm=P.speed/maxSpd;P.gear=7;
  for(var gi=1;gi<GB.length-1;gi++){if(rpm<GB[gi+1]){P.gear=gi;break;}}
  var gLo=GB[Math.max(P.gear-1,0)],gHi=GB[P.gear];
  P.rpmVal=gHi>gLo?Math.max(0,Math.min(1,(rpm-gLo)/(gHi-gLo))):0;
  var slp=Math.abs(P.yawRate)*P.speed;
  P.tireWear=Math.max(0.02,P.tireWear-slp*dt*0.00004);
  P.tireTempF=Math.min(1,Math.max(0.1,P.tireTempF+(thr*0.3+slp*0.08)*dt*0.08-dt*0.02));
  P.tireTempR=Math.min(1,Math.max(0.1,P.tireTempR+(thr*0.5+brk*0.3)*dt*0.08-dt*0.02));
  // Tire smoke: lockup under heavy braking (front), wheelspin on a low-speed launch + slides (rear)
  if(brk===1&&P.speed>12) SMOKE.emit(P,1,1,1.0);
  if((thr===1&&P.speed<7&&P.speed>0.2)||slp>4.5) SMOKE.emit(P,-1,1,slp>4.5?1.2:0.6);
  var s1i=Math.floor(N/3),s2i=Math.floor(2*N/3);
  if(P.sector===0&&P.tIdx>=s1i&&P.tIdx<s1i+15){
    P.s1=raceTime-P.sectorStart;P.sector=1;P.sectorStart=raceTime;
    setSec(hudS1,'hud-s1',P.s1,P.bestS1);if(P.s1<P.bestS1) P.bestS1=P.s1;
  }
  if(P.sector===1&&P.tIdx>=s2i&&P.tIdx<s2i+15){
    P.s2=raceTime-P.sectorStart;P.sector=2;P.sectorStart=raceTime;
    setSec(hudS2,'hud-s2',P.s2,P.bestS2);if(P.s2<P.bestS2) P.bestS2=P.s2;
  }
  if(P.sector===2&&P.tIdx<15&&!P._lapGuard){
    P._lapGuard=true;
    P.s3=raceTime-P.sectorStart;
    setSec(hudS3,'hud-s3',P.s3,P.bestS3);if(P.s3<P.bestS3) P.bestS3=P.s3;
    var lapT=P.s1+P.s2+P.s3;if(lapT>0&&lapT<P.bestLap) P.bestLap=lapT;
    P.lap++;P.sector=0;P.sectorStart=raceTime;P.lapStart=raceTime;
    if(P.lap>LAPS){P.lap=LAPS;P.finishTime=raceTime;gameState='FINISHED';showPodium();}
  }
  if(P.tIdx>=20) P._lapGuard=false;
}

function setSec(el,cls,t,best){
  if(!el) return;
  el.className=cls+(t>0&&t<best?' purple':' yellow');
  el.textContent=(cls==='hud-s1'?'S1 ':cls==='hud-s2'?'S2 ':'S3 ')+(t>0?t.toFixed(3):'---.---');
}

/* ===================== AI ===================== */
function updateAI(ai,dt){
  if(gameState!=='RACING') return;
  var maxSpd=MAX_SPD*(ai.spdFac+ai._randFac);

  // Curvature scan: max curvature over 5–55 wp for speed; nearest significant turn within 35 wp for apex direction
  var curvature=0,turnSign=0;
  for(var k=5;k<=55;k+=5){
    var dA=wpDir((ai.tIdx+k)%N),dB=wpDir((ai.tIdx+k+5)%N);
    var cr=dA.x*dB.z-dA.z*dB.x,absCr=Math.abs(cr);
    if(absCr>curvature) curvature=absCr;
    if(!turnSign&&k<=35&&absCr>0.018) turnSign=cr>0?1:-1;
  }
  curvature=Math.min(1,curvature*5.5);
  var onStraight=curvature<0.12;

  // Target speed: corner-limited — brake hard for the Esses/hairpin so the line holds inside the wall
  var brkFac=0.40-(ai._aggr-1.0)*0.12;
  var spdFloor=Math.max(0.36,0.46-ai._aggr*0.06);
  var targetSpd=Math.max(maxSpd*spdFloor,maxSpd*(1-curvature*brkFac));
  var launching=raceTime<2.0;

  // Gap-proportional following: closing rate limited by gap size — never cascades to zero
  var bp0=wpPerp(ai.tIdx),bw0=waypoints[ai.tIdx];
  var myLat=ai._latTarget;
  var allCars=[P].concat(AI);
  var fwdCar=null,fwdGap=999;
  if(!launching){
    for(var j=0;j<allCars.length;j++){
      var oth=allCars[j];if(oth===ai) continue;
      var g=((oth.tIdx-ai.tIdx)+N)%N;
      if(g<1||g>40) continue;
      var othLat=oth._latTarget!==undefined?oth._latTarget:((oth.x-bw0.x)*bp0.x+(oth.z-bw0.z)*bp0.z);
      if(Math.abs(myLat-othLat)<TW*0.24&&g<fwdGap){fwdGap=g;fwdCar=oth;}
    }
    if(fwdCar){
      var closingSpd=ai.speed-fwdCar.speed;
      if(closingSpd>0){
        var allowedClose=Math.min(fwdGap*0.7,9.0);
        if(closingSpd>allowedClose) targetSpd=Math.min(targetSpd,fwdCar.speed+allowedClose);
      }
      var desiredGap=9;
      if(fwdGap<desiredGap){
        var gapFac=fwdGap/desiredGap;
        targetSpd=Math.min(targetSpd,fwdCar.speed*(0.90+0.10*gapFac));
      }
    }
    if(fwdCar&&fwdGap<40) ai._stuckTimer+=dt;
    else ai._stuckTimer=Math.max(0,ai._stuckTimer-dt*1.4);
  }
  // (No full-speed launch override: keep curvature braking active so cars still slow for Turn 1.)

  // Slipstream: +7% maxSpd when directly behind another car in the same lane
  var slipBoost=1.0;
  if(!launching){
    for(var js=0;js<allCars.length;js++){
      var oS=allCars[js];if(oS===ai) continue;
      var gS=((oS.tIdx-ai.tIdx)+N)%N;
      if(gS<1||gS>25) continue;
      var oSLat=oS._latTarget!==undefined?oS._latTarget:((oS.x-bw0.x)*bp0.x+(oS.z-bw0.z)*bp0.z);
      if(Math.abs(myLat-oSLat)<TW*0.22){slipBoost=1.10;break;}
    }
  }
  if(slipBoost>1.0&&drsZoneIndex(ai.tIdx)>=0&&gapAheadSec(ai)<1.0) slipBoost=1.16;  // DRS in zone
  maxSpd*=slipBoost;

  // Accelerate / brake toward target speed
  var force=targetSpd>ai.speed?ENG:-BRK;
  ai.speed=Math.max(0,Math.min(ai.speed+(force/CAR_MASS)*dt,maxSpd));
  if(force<0&&ai.speed>15&&curvature>0.35&&Math.random()<0.18) SMOKE.emit(ai,1,1,0.7);  // lockup smoke

  // Lateral target — committed-overtake state machine
  var halfTW=TW*0.44;
  var desiredLat=ai._baseLat;

  if(raceTime<2.5){
    // Formation: hold start lane
    desiredLat=ai._baseLat;ai._ovTimer=0;

  } else if(ai._ovTimer>0){
    // COMMITTED OVERTAKE: hold _ovTgt until timer expires or gap opens
    ai._ovTimer-=dt;
    desiredLat=ai._ovTgt;
    if(!fwdCar||fwdGap>55){ai._ovTimer=0;ai._baseLat=ai._ovTgt;}  // Lock new lane

  } else if(curvature>0.22){
    // CORNER: each car takes its own apex; only fall back if nose-to-tail
    desiredLat=(fwdCar&&fwdGap<25)?ai._baseLat:-turnSign*halfTW*ai.apexD;

  } else {
    // STRAIGHT / GENTLE CURVE: follow or commit to overtake
    if(fwdCar&&fwdGap<40){
      var fwdLat=fwdCar._latTarget!==undefined?fwdCar._latTarget:((fwdCar.x-bw0.x)*bp0.x+(fwdCar.z-bw0.z)*bp0.z);
      // Direction: go to whichever side we're already biased toward — prevents flip-flopping
      var oDir=Math.abs(ai._latTarget)>halfTW*0.65?-Math.sign(ai._latTarget):ai._ovSide;
      var oTgt=Math.max(-halfTW,Math.min(halfTW,fwdLat+oDir*TW*0.35));
      // Faster cars force past; others only go if the gap is clear
      var doOvertake=ai._stuckTimer>1.5||((fwdCar.spdFac!==undefined)&&(ai.spdFac-fwdCar.spdFac)>0.05);
      if(!doOvertake){
        doOvertake=true;
        for(var kk=0;kk<allCars.length;kk++){
          var ocb=allCars[kk];if(ocb===ai||ocb===fwdCar) continue;
          var og2=((ocb.tIdx-ai.tIdx)+N)%N;
          if(og2>22||(N-og2)>10) continue;  // Window: 10 wp behind to 22 wp ahead
          var ocbLat=ocb._latTarget!==undefined?ocb._latTarget:((ocb.x-bw0.x)*bp0.x+(ocb.z-bw0.z)*bp0.z);
          if(Math.abs(oTgt-ocbLat)<TW*0.20){doOvertake=false;break;}
        }
      }
      if(doOvertake){ai._ovTgt=oTgt;ai._ovTimer=3.5;ai._ovSide*=-1;desiredLat=oTgt;}
      else desiredLat=ai._baseLat;
    } else {
      desiredLat=ai._baseLat;
    }
  }
  var wallEdge=halfTW*0.78;
  if(Math.abs(desiredLat)>wallEdge) desiredLat-=Math.sign(desiredLat)*(Math.abs(desiredLat)-wallEdge)*1.4;
  desiredLat=Math.max(-halfTW,Math.min(halfTW,desiredLat));
  ai._latTarget+=(desiredLat-ai._latTarget)*Math.min(dt*5.0,1);
  ai._latTarget=Math.max(-halfTW,Math.min(halfTW,ai._latTarget));

  // Steer toward lookahead waypoint with lateral offset — short lookahead in corners so the
  // target tracks the current apex instead of aiming across the next Esses reversal
  var look=Math.max(6,Math.round(28-curvature*22));
  var tI=(ai.tIdx+look)%N,tWp=waypoints[tI],tPp=wpPerp(tI);
  var tx=tWp.x+tPp.x*ai._latTarget,tz=tWp.z+tPp.z*ai._latTarget;
  var dh=Math.atan2(tx-ai.x,tz-ai.z)-ai.heading;
  while(dh>Math.PI) dh-=2*Math.PI;while(dh<-Math.PI) dh+=2*Math.PI;
  var maxTurn=Math.PI*2.4*dt;
  ai.heading+=Math.max(-maxTurn,Math.min(maxTurn,dh*dt*7.6));

  ai.x+=Math.sin(ai.heading)*ai.speed*dt;
  ai.z+=Math.cos(ai.heading)*ai.speed*dt;
  ai.tIdx=closestWP(ai.x,ai.z,ai.tIdx);
  ai.y=waypoints[ai.tIdx].y;

  // Track boundary clamp
  var bp=wpPerp(ai.tIdx),bw=waypoints[ai.tIdx];
  var lat=(ai.x-bw.x)*bp.x+(ai.z-bw.z)*bp.z;
  if(Math.abs(lat)>TW*0.5+CURB){
    var s=lat>0?1:-1;
    ai.x=bw.x+bp.x*s*(TW*0.5+CURB);
    ai.z=bw.z+bp.z*s*(TW*0.5+CURB);
    ai.speed*=0.80;                 // glancing brush, not a full stop
    ai._latTarget=-s*halfTW*0.5;    // steer back toward the racing line, not just toward centre
    ai._ovTimer=0;
    SMOKE.emit(ai,1,2,1.2);
  }

  // Lap detection
  if(ai.tIdx<12&&!ai._inStart){
    ai._inStart=true;
    ai.lap++;if(ai.lap>LAPS){ai.lap=LAPS;if(ai.finishTime===Infinity)ai.finishTime=raceTime;}
  }
  if(ai.tIdx>=20) ai._inStart=false;
}

/* ===================== COLLISION ===================== */
var CAR_HL=2.55,CAR_HW=1.05;
function resolveCollisions(){
  var cars=[P].concat(AI);
  for(var pass=0;pass<2;pass++){
    for(var i=0;i<cars.length;i++){
      for(var j=i+1;j<cars.length;j++){
        var a=cars[i],b=cars[j];
        var dx=b.x-a.x,dz=b.z-a.z;
        if(dx*dx+dz*dz>49) continue;
        var fx=Math.sin(a.heading),fz=Math.cos(a.heading);
        var sx=-fz,sz=fx;
        var projL=dx*fx+dz*fz,projS=dx*sx+dz*sz;
        var ovL=CAR_HL*2-Math.abs(projL),ovS=CAR_HW*2-Math.abs(projS);
        if(ovL<0||ovS<0) continue;
        var nx,nz,pen;
        if(ovL<ovS){
          nx=projL>0?fx:-fx;nz=projL>0?fz:-fz;pen=ovL*0.5;
          var dv=Math.max(0,a.speed-b.speed);
          var imp=Math.min(dv*0.35,pen*5+dv*0.08);
          var flatPen=Math.min(pen*1.2,0.8);
          a.speed=Math.max(0,a.speed-imp*0.65-flatPen*0.4);
          b.speed=Math.min(b.speed+imp*0.25,MAX_SPD*1.05);
          b.speed=Math.max(0,b.speed-flatPen*0.15);
        } else {
          nx=projS>0?sx:-sx;nz=projS>0?sz:-sz;pen=ovS*0.5;
          var bumpY=pen*0.08;
          if(a.yawRate!==undefined) a.yawRate+=(projS>0?bumpY:-bumpY);
          if(b.yawRate!==undefined) b.yawRate+=(projS>0?-bumpY:bumpY);
          // Strong speed penalty on contact — proportional to penetration depth
          var rub=Math.min(pen*0.6,0.45);
          a.speed=Math.max(0,a.speed-rub);b.speed=Math.max(0,b.speed-rub);
          // Force lateral targets AND base lanes to diverge — prevents immediate re-merge
          var latPush=2.5,hTW=TW*0.44;
          if(a._latTarget!==undefined){a._latTarget+=(projS>0?-latPush:latPush);a._latTarget=Math.max(-hTW,Math.min(hTW,a._latTarget));}
          if(b._latTarget!==undefined){b._latTarget+=(projS>0?latPush:-latPush);b._latTarget=Math.max(-hTW,Math.min(hTW,b._latTarget));}
          if(a._baseLat!==undefined){a._baseLat+=(projS>0?-latPush*0.5:latPush*0.5);a._baseLat=Math.max(-hTW,Math.min(hTW,a._baseLat));}
          if(b._baseLat!==undefined){b._baseLat+=(projS>0?latPush*0.5:-latPush*0.5);b._baseLat=Math.max(-hTW,Math.min(hTW,b._baseLat));}
        }
        a.x-=nx*pen;a.z-=nz*pen;b.x+=nx*pen;b.z+=nz*pen;
      }
    }
  }
}

/* ===================== AUDIO ===================== */
var _AC=null,_masterComp=null,_playerEng=null,_aiEngs=[];
var _prevRpm=0,_popCooldown=0,_prevTickT=0,_noiseBuffer=null;
var _audioRpm=0,_prevGear=0,_crackleBurst=0;

function _initAudio(){
  if(_AC) return;
  _AC=new(window.AudioContext||window.webkitAudioContext)();

  // Aggressive compressor — catches any transient peak before it reaches the output
  _masterComp=_AC.createDynamicsCompressor();
  _masterComp.threshold.value=-18;_masterComp.knee.value=8;
  _masterComp.ratio.value=6;_masterComp.attack.value=0.003;_masterComp.release.value=0.12;
  _masterComp.connect(_AC.destination);

  // 8 harmonics, near-zero above the 4th. The 16-harmonic wave was generating
  // 640–6400 Hz content that the distortion then multiplied — the entire screech source.
  var eReal=new Float32Array(8),eImag=new Float32Array(8);
  eImag[1]=1.00;eImag[2]=0.92;eImag[3]=0.65;eImag[4]=0.38;
  eImag[5]=0.14;eImag[6]=0.05;eImag[7]=0.02;
  var _exhaustWave=_AC.createPeriodicWave(eReal,eImag);

  _noiseBuffer=_AC.createBuffer(1,_AC.sampleRate,_AC.sampleRate);
  var nd=_noiseBuffer.getChannelData(0);
  for(var i=0;i<nd.length;i++) nd[i]=Math.random()*2-1;

  function _makeDistCurve(k){
    var n=512,c=new Float32Array(n);
    for(var i=0;i<n;i++){var x=(i*2)/n-1;c[i]=Math.tanh(k*x)/Math.tanh(k);}
    return c;
  }

  function buildEng(){
    // Main chain: oscillator (200-1200 Hz) → soft clip → LP that SWEEPS OPEN with RPM.
    // The LP opening is what makes the sound brighten as revs rise — no noise sweeps needed.
    var exhaustOsc=_AC.createOscillator();
    exhaustOsc.setPeriodicWave(_exhaustWave);
    exhaustOsc.frequency.value=80;
    var dist=_AC.createWaveShaper();
    dist.curve=_makeDistCurve(2.5);dist.oversample='2x';
    var exhaustLP=_AC.createBiquadFilter();
    exhaustLP.type='lowpass';exhaustLP.frequency.value=400;exhaustLP.Q.value=0.5;

    // Parallel low-body: boost 100-160 Hz band to keep deep chest weight at all RPM.
    var bodyBPF=_AC.createBiquadFilter();
    bodyBPF.type='bandpass';bodyBPF.frequency.value=70;bodyBPF.Q.value=1.0;
    var bodyGain=_AC.createGain();bodyGain.gain.value=0.60;

    // Mechanical texture: broadband noise at very low gain — Q=0.4 so it is diffuse, not tonal.
    var ns=_AC.createBufferSource();ns.buffer=_noiseBuffer;ns.loop=true;ns.start();
    var nBPF=_AC.createBiquadFilter();
    nBPF.type='bandpass';nBPF.frequency.value=220;nBPF.Q.value=0.4;
    var nGain=_AC.createGain();nGain.gain.value=0.04;

    var gain=_AC.createGain();gain.gain.value=0;
    // LFO: cylinder-pulse burble — strong at idle, fades to texture at redline
    var lfo=_AC.createOscillator();lfo.frequency.value=3.5;
    var lfoDepth=_AC.createGain();lfoDepth.gain.value=0;
    lfo.connect(lfoDepth);lfoDepth.connect(gain.gain);
    lfo.start();
    exhaustOsc.connect(dist);dist.connect(exhaustLP);exhaustLP.connect(gain);
    exhaustOsc.connect(bodyBPF);bodyBPF.connect(bodyGain);bodyGain.connect(gain);
    ns.connect(nBPF);nBPF.connect(nGain);nGain.connect(gain);
    gain.connect(_masterComp);
    exhaustOsc.start();
    return {exhaustOsc:exhaustOsc,exhaustLP:exhaustLP,gain:gain,lfo:lfo,lfoDepth:lfoDepth};
  }

  _playerEng=buildEng();
  _aiEngs=[];
  for(var i=0;i<AI.length;i++) _aiEngs.push(buildEng());
}

// RPM-scaled burst of exhaust crackle pops — fires count pops via Web Audio scheduling.
function _fireExhaustCrackle(count){
  var now=_AC.currentTime;
  var gap=0.06+Math.max(0,0.5-P.rpmVal)*0.04;
  for(var i=0;i<count;i++){
    var t=now+i*gap;
    var ns=_AC.createBufferSource();ns.buffer=_noiseBuffer;
    var bpf=_AC.createBiquadFilter();bpf.type='bandpass';
    bpf.frequency.value=1200+Math.random()*1600;bpf.Q.value=1.8;
    var env=_AC.createGain();
    var amp=0.32+Math.random()*0.18;
    env.gain.setValueAtTime(0,t);
    env.gain.setValueAtTime(amp,t+0.001);
    env.gain.exponentialRampToValueAtTime(0.001,t+0.055);
    ns.connect(bpf);bpf.connect(env);env.connect(_masterComp);
    ns.start(t);ns.stop(t+0.08);
  }
}

// Sharp flat-cut bang on upshift — high-frequency exhaust blowback.
function _fireUpshiftCrack(){
  var ns=_AC.createBufferSource();ns.buffer=_noiseBuffer;
  var bpf=_AC.createBiquadFilter();bpf.type='bandpass';
  bpf.frequency.value=1400+Math.random()*800;bpf.Q.value=4.5;
  var env=_AC.createGain();var t=_AC.currentTime;
  env.gain.setValueAtTime(0.55,t);
  env.gain.exponentialRampToValueAtTime(0.001,t+0.022);
  ns.connect(bpf);bpf.connect(env);env.connect(_masterComp);
  ns.start();ns.stop(t+0.035);
}

// Cascading 3-pop crackle sequence on downshift — characteristic F1 downchange sound.
function _fireDownshiftCrackle(){
  var now=_AC.currentTime;
  var amps=[0.42,0.35,0.28];var freqs=[1800,1400,2100];
  for(var i=0;i<3;i++){
    var t=now+i*0.075;
    var ns=_AC.createBufferSource();ns.buffer=_noiseBuffer;
    var bpf=_AC.createBiquadFilter();bpf.type='bandpass';
    bpf.frequency.value=freqs[i]+Math.random()*200;bpf.Q.value=2.5;
    var env=_AC.createGain();
    env.gain.setValueAtTime(0,t);
    env.gain.setValueAtTime(amps[i],t+0.001);
    env.gain.exponentialRampToValueAtTime(0.001,t+0.065);
    ns.connect(bpf);bpf.connect(env);env.connect(_masterComp);
    ns.start(t);ns.stop(t+0.09);
  }
}

function _tickAudio(){
  if(!_AC) return;
  if(_AC.state==='suspended') _AC.resume();
  var now=_AC.currentTime;
  var active=(gameState==='RACING'||gameState==='COUNTDOWN');
  var thr=P.thr||0,brk=P.brk||0;
  var dt=now-_prevTickT;if(dt<=0||dt>0.2)dt=0.016;_prevTickT=now;

  // Gear-change transient: audible RPM drop/spike makes shifts feel physical
  if(_prevGear!==0&&P.gear!==_prevGear&&gameState==='RACING'&&_crackleBurst<=0){
    if(P.gear>_prevGear){
      _audioRpm=Math.max(0.05,_audioRpm-0.38);  // upshift: RPM falls
      _fireUpshiftCrack();_crackleBurst=0.18;
    } else {
      _audioRpm=Math.min(0.95,_audioRpm+0.22);  // downshift: RPM spikes
      _fireDownshiftCrackle();_crackleBurst=0.30;
    }
  }
  _prevGear=P.gear;

  // _audioRpm chases P.rpmVal with inertia — prevents the instant-drone effect
  var rpmT=P.rpmVal;
  var riseR=thr>0?2.2:5.0,fallR=brk>0?6.0:3.5;
  if(rpmT>_audioRpm) _audioRpm=Math.min(rpmT,_audioRpm+riseR*dt);
  else _audioRpm=Math.max(rpmT,_audioRpm-fallR*dt);

  // Small random jitter — breaks up the perfect monotone between gear changes
  var wobble=(Math.random()-0.5)*(0.08-_audioRpm*0.06);
  var effRpm=Math.max(0,Math.min(1,_audioRpm+wobble));

  // Oscillator 80→300 Hz, LP 280→2500 Hz — wider sweep = dramatic tonal shift
  var crankHz=45+effRpm*155;
  var lpHz=300+effRpm*1300;
  if(brk>0)  lpHz*=0.50;   // braking: very dark and muted
  else if(!thr) lpHz*=0.72; // coasting: slightly muffled
  _playerEng.exhaustOsc.frequency.setTargetAtTime(crankHz,now,0.02);
  _playerEng.exhaustLP.frequency.setTargetAtTime(lpHz,now,0.04);

  // LFO cylinder pulse: 2 Hz burble at idle → 12 Hz texture at redline
  var lfoHz=2+effRpm*10;
  var lfoAmt=active?(0.07-effRpm*0.06):0;
  _playerEng.lfo.frequency.setTargetAtTime(lfoHz,now,0.08);
  _playerEng.lfoDepth.gain.setTargetAtTime(lfoAmt,now,0.08);

  var targetGain;
  if(brk>0)      targetGain=0.18+effRpm*0.08;
  else if(thr>0) targetGain=0.38+effRpm*0.24;
  else           targetGain=0.22+effRpm*0.12;
  if(active) _playerEng.gain.gain.setTargetAtTime(targetGain,now,0.06);
  else _playerEng.gain.gain.setTargetAtTime(0,now,0.06);

  // Rev limiter stutter at redline under full throttle
  if(active&&effRpm>0.95&&thr===1){
    var fl=now%0.067<0.033?0.88:1.0;
    _playerEng.gain.gain.setValueAtTime(targetGain*fl,now);
  }

  _crackleBurst=Math.max(0,_crackleBurst-dt);
  _popCooldown=Math.max(0,_popCooldown-dt);

  // Throttle-lift crackle
  if(_prevRpm>0.45&&(_audioRpm-_prevRpm)<-0.04&&_popCooldown<=0&&_crackleBurst<=0&&gameState==='RACING'){
    _fireExhaustCrackle(1+Math.floor(_audioRpm*3));
    _popCooldown=0.10;_crackleBurst=0.35;
  }
  _prevRpm=_audioRpm;

  // AI engines — distance-attenuated, same LP sweep range
  for(var i=0;i<AI.length;i++){
    var ai=AI[i],e=_aiEngs[i];if(!e) continue;
    var aiNorm=Math.min(1,ai.speed/(MAX_SPD*(ai.spdFac||1.0)));
    var aiCrank=45+aiNorm*155;
    var aiLP=300+aiNorm*1300;
    e.exhaustOsc.frequency.setTargetAtTime(aiCrank,now,0.12);
    e.exhaustLP.frequency.setTargetAtTime(aiLP,now,0.15);
    var ddx=ai.x-P.x,ddz=ai.z-P.z;
    var vol=(active&&gameState==='RACING')?Math.max(0,0.10*(1-Math.sqrt(ddx*ddx+ddz*ddz)/160)):0;
    e.gain.gain.setTargetAtTime(vol,now,0.15);
  }
}

function _stopAudio(){
  if(!_AC) return;
  var now=_AC.currentTime;
  if(_playerEng) _playerEng.gain.gain.setTargetAtTime(0,now,0.25);
  _aiEngs.forEach(function(e){e.gain.gain.setTargetAtTime(0,now,0.25);});
}

/* ============ COCKPIT STEERING WHEEL OVERLAY ============ */
function updateCockpitWheel(dt){
  if(!cockpitWheel) return;
  var show=cockpitMode&&(gameState==='RACING'||gameState==='COUNTDOWN'||gameState==='FINISHED');
  if(cockpitWheel.className!==(show?'active':'')) cockpitWheel.className=show?'active':'';
  if(!show) return;
  // smooth the raw steering input and rotate the wheel up to ~110° of lock
  cwSteer+=((P.steerIn||0)-cwSteer)*Math.min(dt*9,1);
  if(cwRot) cwRot.setAttribute('transform','rotate('+(-cwSteer*110).toFixed(1)+')');
  if(cwGear) cwGear.textContent=P.speed<0.5?'N':P.gear;
  var lit=Math.floor(P.rpmVal*cwLeds.length+0.0001);
  for(var i=0;i<cwLeds.length;i++) cwLeds[i].setAttribute('opacity',i<lit?'1':'0.12');
}

/* ===================== CAMERA ===================== */
function updateCamera(dt){
  var spd=P.speed;
  if(gameState==='COUNTDOWN'){
    var d0=wpDir(P.tIdx),wy0=waypoints[P.tIdx].y;
    gameCam.position.set(P.x-d0.x*28,wy0+13,P.z-d0.z*28);
    gameCam.lookAt(new THREE.Vector3(P.x+d0.x*8,wy0+0.5,P.z+d0.z*8));
    gameCam.fov=80;gameCam.updateProjectionMatrix();
    camSmooth.copy(gameCam.position);camVel.set(0,0,0);return;
  }
  if(gameState!=='RACING'&&gameState!=='FINISHED') return;
  if(cockpitMode){
    // Driver's-eye: sit up in the cockpit and look up the track along the car's heading.
    var fxh=Math.sin(P.heading),fzh=Math.cos(P.heading);
    gameCam.position.set(P.x+fxh*0.1,P.y+RIDE_H+0.92,P.z+fzh*0.1);
    gameCam.lookAt(new THREE.Vector3(P.x+fxh*22,P.y+RIDE_H+1.15,P.z+fzh*22));
    gameCam.fov=84;gameCam.updateProjectionMatrix();return;
  }
  var dist=14+(spd/MAX_SPD)*8,ht=4+(spd/MAX_SPD)*2;
  var fx=Math.sin(P.heading),fz=Math.cos(P.heading);
  var tgt=new THREE.Vector3(P.x-fx*dist,P.y+RIDE_H+ht,P.z-fz*dist);
  var om=9,ze=0.88,c2=2*ze*om,kk=om*om;
  camVel.x+=(tgt.x-camSmooth.x)*kk*dt-camVel.x*c2*dt;
  camVel.y+=(tgt.y-camSmooth.y)*kk*dt-camVel.y*c2*dt;
  camVel.z+=(tgt.z-camSmooth.z)*kk*dt-camVel.z*c2*dt;
  camSmooth.x+=camVel.x*dt;camSmooth.y+=camVel.y*dt;camSmooth.z+=camVel.z*dt;
  gameCam.position.copy(camSmooth);
  gameCam.lookAt(new THREE.Vector3(P.x+fx*20,P.y+RIDE_H+0.5,P.z+fz*20));
  gameCam.fov=P.drs?82:72+(spd/90)*6;gameCam.updateProjectionMatrix();
}

/* ===================== HUD ===================== */
function fmt(t){
  var m=Math.floor(t/60),s=Math.floor(t%60),ms=Math.floor((t%1)*1000);
  return m+':'+(s<10?'0':'')+s+'.'+(ms<100?(ms<10?'00':'0'):'')+ms;
}

// Single source of truth for race order: finished cars rank by finishTime (earliest
// first) ahead of still-running cars, which fall through to laps-then-progress.
function classifyCmp(a,b){
  if(a.ft<Infinity&&b.ft<Infinity) return a.ft-b.ft;
  if(a.ft<Infinity) return -1;
  if(b.ft<Infinity) return 1;
  return b.lap!==a.lap?b.lap-a.lap:b.ti-a.ti;
}

function getPos(){
  var all=[{n:'P',lap:P.lap,ti:P.tIdx,ft:P.finishTime}];
  AI.forEach(function(ai,i){all.push({n:'A'+i,lap:ai.lap,ti:ai.tIdx,ft:ai.finishTime});});
  all.sort(classifyCmp);
  var pp=0;all.forEach(function(e,i){if(e.n==='P') pp=i;});return pp;
}

var _stLast=0;
function updateStandings(){
  if(!hudStandings) return;
  var now=Date.now();if(now-_stLast<200) return;_stLast=now;
  var arr=[{isP:true,name:AI_NAMES[0],abbr:'YOU',color:'#2D7DFF',lap:P.lap,ti:P.tIdx,sp:P.speed,ft:P.finishTime}];
  for(var i=0;i<AI.length;i++){
    arr.push({isP:false,name:AI_NAMES[i+1],abbr:AI_TEAMS[i],
      color:'#'+(AI_COLORS[i]||0xAAAAAA).toString(16).padStart(6,'0'),
      lap:AI[i].lap,ti:AI[i].tIdx,sp:AI[i].speed,ft:AI[i].finishTime});
  }
  arr.sort(classifyCmp);
  var html='<div class="st-hd"><span>Race Order</span><b>LAP '+Math.min(P.lap,LAPS)+'/'+LAPS+'</b></div>';
  for(var k=0;k<arr.length;k++){
    var e=arr[k],gap='';
    if(k===0) gap='LEADER';
    else{var ah=arr[k-1];
      if(e.lap<ah.lap) gap='+'+(ah.lap-e.lap)+'L';
      else{var wg=((ah.ti-e.ti)+N)%N;gap='+'+((wg*2.31)/Math.max(e.sp,4)).toFixed(1);}
    }
    html+='<div class="st-row'+(e.isP?' me':'')+'"><span class="st-pos">'+(k+1)+'</span>'+
      '<span class="st-chip" style="background:'+e.color+'"></span>'+
      '<span class="st-name">'+e.abbr+' '+e.name.slice(0,9)+'</span>'+
      '<span class="st-gap">'+gap+'</span></div>';
  }
  hudStandings.innerHTML=html;
}

function updateHUD(){
  if(hudPos) hudPos.textContent='P'+(getPos()+1);
  if(hudLap) hudLap.textContent='LAP '+P.lap+'/'+LAPS;
  if(hudTimer) hudTimer.textContent=fmt(gameState==='RACING'?raceTime-P.lapStart:0);
  if(hudSpeed) hudSpeed.textContent=((P.speed*3.6)|0)+' km/h';
  if(hudGear) hudGear.textContent=P.speed<0.5?'N':P.gear;
  if(rpmFill) rpmFill.style.width=(P.rpmVal*100)+'%';
  var tw=P.tireWear;
  var tireColor=tw>0.5?'#00ff44':tw>0.2?'#ffdd00':'#ff4400';
  if(tireF){tireF.style.width=(tw*100)+'%';tireF.style.background=tireColor;}
  if(tireR){tireR.style.width=(tw*100)+'%';tireR.style.background=tireColor;}
  function tc(t2){return t2>0.7?'#ff4400':t2>0.4?'#ffdd00':'#1E41FF';}
  if(tempF) tempF.style.background=tc(P.tireTempF);
  if(tempR) tempR.style.background=tc(P.tireTempR);
  if(wheelInd) wheelInd.setAttribute('transform','rotate('+(-P.yawRate*20*180/Math.PI)+')');
  // Best lap display
  var hudMsgEl=document.querySelector('#hud-main .hud-msg');
  if(hudMsgEl){
    if(P.bestLap<Infinity){hudMsgEl.textContent='BEST '+fmt(P.bestLap);hudMsgEl.style.color='#cc00ff';}
    else{hudMsgEl.textContent='';hudMsgEl.style.color='';}
  }
  // Live gap to car ahead
  var hudDeltaEl=document.getElementById('hud-delta');
  if(hudDeltaEl&&gameState==='RACING'){
    var myRank=getPos();
    if(myRank===0){
      hudDeltaEl.textContent='Δ LEADER';hudDeltaEl.className='hud-delta green';
    } else {
      var allCarsD=[{tIdx:P.tIdx,lap:P.lap,isP:true}].concat(AI.map(function(a){return {tIdx:a.tIdx,lap:a.lap,isP:false};}));
      allCarsD.sort(function(a,b){if(a.lap!==b.lap)return b.lap-a.lap;return((b.tIdx-a.tIdx)+N)%N<((a.tIdx-b.tIdx)+N)%N?-1:1;});
      var ahead=allCarsD[myRank-1];
      if(ahead){
        var wpGap=((ahead.tIdx-P.tIdx)+N)%N;
        var gapS=(wpGap*2.2)/Math.max(P.speed,1);
        hudDeltaEl.textContent='Δ +'+(gapS<99?gapS.toFixed(1)+'s':'lap');
        hudDeltaEl.className='hud-delta'+(gapS<1?' green':gapS>5?' red':'');
      }
    }
  }
  drawMinimap();
  updateStandings();
}

var mmB=null;
function drawMinimap(){
  if(!minimapCtx) return;
  if(!mmB){
    var mnx=Infinity,mxx=-Infinity,mnz=Infinity,mxz=-Infinity;
    for(var i=0;i<N;i++){
      mnx=Math.min(mnx,waypoints[i].x);mxx=Math.max(mxx,waypoints[i].x);
      mnz=Math.min(mnz,waypoints[i].z);mxz=Math.max(mxz,waypoints[i].z);
    }
    mmB={mnx:mnx,mxx:mxx,mnz:mnz,mxz:mxz};
  }
  var mw=160,mh=160,bb=mmB;
  var sc=Math.min((mw-16)/(bb.mxx-bb.mnx),(mh-16)/(bb.mxz-bb.mnz));
  var ox=(mw-(bb.mxx-bb.mnx)*sc)/2,oz=(mh-(bb.mxz-bb.mnz)*sc)/2;
  function wx(x){return ox+(x-bb.mnx)*sc;}
  function wz2(z){return oz+(z-bb.mnz)*sc;}
  minimapCtx.clearRect(0,0,mw,mh);
  minimapCtx.strokeStyle='#1A6A90';minimapCtx.lineWidth=3;minimapCtx.beginPath();
  for(var j=0;j<N;j+=2){
    var ww=waypoints[j];
    if(j===0) minimapCtx.moveTo(wx(ww.x),wz2(ww.z));
    else minimapCtx.lineTo(wx(ww.x),wz2(ww.z));
  }
  minimapCtx.closePath();minimapCtx.stroke();
  // DRS zones highlighted in green over the track outline
  minimapCtx.strokeStyle='#00FF87';minimapCtx.lineWidth=3;
  for(var zi=0;zi<DRS_ZONES.length;zi++){
    var z=DRS_ZONES[zi],pen=false;minimapCtx.beginPath();
    for(var ji=0;ji<N;ji+=2){
      var inz=z.s<=z.e?(ji>=z.s&&ji<=z.e):(ji>=z.s||ji<=z.e);
      if(inz){var wz=waypoints[ji];
        if(pen) minimapCtx.lineTo(wx(wz.x),wz2(wz.z));else{minimapCtx.moveTo(wx(wz.x),wz2(wz.z));pen=true;}
      } else pen=false;
    }
    minimapCtx.stroke();
  }
  minimapCtx.fillStyle='#1E41FF';minimapCtx.beginPath();
  minimapCtx.arc(wx(P.x),wz2(P.z),5,0,Math.PI*2);minimapCtx.fill();
  AI.forEach(function(ai,i){
    minimapCtx.fillStyle='#'+(AI_COLORS[i]||0xAAAAAA).toString(16).padStart(6,'0');
    minimapCtx.beginPath();
    minimapCtx.arc(wx(ai.x),wz2(ai.z),3,0,Math.PI*2);minimapCtx.fill();
  });
}

/* ===================== PODIUM ===================== */
function showPodium(){
  var all=[{n:AI_NAMES[0],lap:P.lap,ti:P.tIdx,ft:P.finishTime}];
  AI.forEach(function(ai,i){all.push({n:AI_NAMES[i+1],lap:ai.lap,ti:ai.tIdx,ft:ai.finishTime});});
  all.sort(classifyCmp);
  var rows='',medals=['1ST','2ND','3RD','4TH','5TH','6TH','7TH','8TH','9TH','10TH',
    '11TH','12TH','13TH','14TH','15TH','16TH','17TH','18TH','19TH','20TH'];
  all.forEach(function(e,i){
    if(i>=10) return;
    rows+='<tr'+(i===0?' class="podium-p1"':'')+'><td>'+medals[i]+'</td><td>'+e.n+'</td><td>LAP '+e.lap+'</td></tr>';
  });
  if(podiumTable) podiumTable.innerHTML=rows;
  if(podiumOvl) podiumOvl.className='active';
}

/* ===================== OPEN / CLOSE ===================== */
function openGame(){
  if(!overlay) return;
  overlay.className='active';
  gameCanvas.focus();requestAnimationFrame(resizeCam);
  _initAudio();
  initRace();startCountdown();
  if(!animRunning){animRunning=true;requestAnimationFrame(loop);}
}

function closeGame(){
  overlay.className='';
  if(podiumOvl) podiumOvl.className='';
  if(pauseOvl) pauseOvl.className='';
  if(lightsBarEl) lightsBarEl.className='';
  if(gridMsgEl) gridMsgEl.className='';
  _stopAudio();
  gameState='IDLE';paused=false;animRunning=false;
}

(function(){
  var sc=document.getElementById('f1car');
  if(sc){
    sc.style.cursor='pointer';
    sc.addEventListener('click',function(){if(gameState==='IDLE') openGame();});
  }
  var v=document.querySelector('.car-viewer');
  if(v){
    v.style.position='relative';
    var hint=document.createElement('button');
    hint.className='race-cta';
    hint.innerHTML='<span class="tri"></span>START RACE';
    hint.addEventListener('click',function(){if(gameState==='IDLE') openGame();});
    v.appendChild(hint);
  }
})();

if(podiumClose) podiumClose.addEventListener('click',closeGame);
if(hudClose) hudClose.addEventListener('click',closeGame);

/* ===================== GAME LOOP ===================== */
var prevTs=0;
function loop(ts){
  if(!animRunning) return;
  requestAnimationFrame(loop);
  if(paused) return;
  var dt=Math.min((ts-prevTs)/1000,0.05);prevTs=ts;
  if(dt<=0) return;
  if(gameState==='RACING') raceTime+=dt;
  updatePlayer(dt);
  AI.forEach(function(ai){updateAI(ai,dt);});
  resolveCollisions();
  if(playerGrp){
    playerGrp.visible=!cockpitMode;  // don't see your own chassis from the driver's seat
    playerGrp.position.set(P.x,P.y+RIDE_H,P.z);playerGrp.rotation.y=P.heading-Math.PI/2;
    if(playerGrp.userData.tailMat) playerGrp.userData.tailMat.emissiveIntensity=P.brk?3.6:1.4;
    if(playerGrp.userData.drsMat) playerGrp.userData.drsMat.emissiveIntensity=P.drs?2.2:0.0;
  }
  AI.forEach(function(ai,i){
    if(aiGrps[i]){aiGrps[i].position.set(ai.x,ai.y+RIDE_H,ai.z);aiGrps[i].rotation.y=ai.heading-Math.PI/2;}
    if(aiLabels[i]){aiLabels[i].position.set(ai.x,ai.y+RIDE_H+3.2,ai.z);}
  });
  SMOKE.update(dt);
  FLASH.update(dt);
  // Crowd twinkle + slow cloud drift
  var _t=(typeof performance!=='undefined'&&performance.now)?performance.now()*0.001:Date.now()*0.001;
  for(var _ci=0;_ci<crowdMats.length;_ci++) crowdMats[_ci].emissiveIntensity=0.03+0.03*Math.sin(_t*2.2+_ci*0.8);
  for(var _cl=0;_cl<clouds.length;_cl++){var _c=clouds[_cl];_c.sp.position.x+=_c.vx*dt;if(_c.sp.position.x>1500) _c.sp.position.x=-1500;}
  updateCamera(dt);
  updateHUD();
  updateCockpitWheel(dt);
  _tickAudio();
  if(afterPass){
    var spN=(gameState==='RACING')?P.speed/MAX_SPD:0;
    afterPass.uniforms['damp'].value=Math.min(0.62,Math.max(0,(spN-0.45)/0.55)*0.62);
  }
  if(useComposer) composer.render();else renderer.render(scene,gameCam);
}

})();
(function(){
  var s0=Date.now();
  function tick(){
    var s=Math.floor((Date.now()-s0)/1000);
    var el=document.getElementById('met-clock');
    if(el) el.textContent='T+'+[Math.floor(s/3600),Math.floor(s%3600/60),s%60]
      .map(function(v){return String(v).padStart(2,'0')}).join(':');
  }
  tick(); setInterval(tick,1000);
})();
</script>

</body>
</html>
"""

# --------------------------------------------------------------------------- #
#  2D chart helpers                                                             #
# --------------------------------------------------------------------------- #

def _axis_2d(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=_TICK, family=_FONT, size=9)),
        gridcolor=_GRID,
        zerolinecolor=_ZERO_LINE,
        tickfont=dict(color=_TICK, family=_FONT, size=9),
        showgrid=True,
        showspikes=True,
        spikecolor=_SPIKE,
        spikethickness=1,
        spikedash="solid",
        spikemode="across",
    )


def _layout_2d(title: str, xaxis_title: str = "", yaxis_title: str = "",
               height: int = 420, hovermode: str = "x unified") -> go.Layout:
    return go.Layout(
        title=dict(
            text=title,
            font=dict(color=_FONT_COLOR, family=_FONT, size=13),
            x=0,
            xanchor="left",
            pad=dict(t=4, l=0),
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        xaxis=_axis_2d(xaxis_title),
        yaxis=_axis_2d(yaxis_title),
        font=dict(family=_FONT, color=_FONT_COLOR),
        margin=dict(l=52, r=24, t=56, b=48),
        height=height,
        showlegend=True,
        legend=dict(
            font=dict(family=_FONT, color=_TICK, size=10),
            bgcolor="rgba(3,15,26,0)",
            bordercolor=_GRID,
            borderwidth=1,
        ),
        hovermode=hovermode,
        hoverdistance=80,
        spikedistance=400,
        hoverlabel=dict(
            bgcolor=_BG_HOVER,
            bordercolor=_ACCENT,
            font=dict(family=_FONT, size=11, color=_FONT_COLOR),
            namelength=-1,
        ),
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------- #
#  SQL helpers                                                                  #
# --------------------------------------------------------------------------- #

def _round_by_round_df(engine: Engine, team_refs: list[str]) -> pd.DataFrame:
    """Cumulative points and finish position per driver per round, all seasons.
    Returns: year | round | driver | points | position
    Computed from results — independent of driver_standings population."""
    placeholders, params = ref_params(team_refs)
    sql = f"""
    SELECT
        ra.year, ra.round,
        COALESCE(da.forename,'') || ' ' || COALESCE(da.surname,'') AS driver,
        SUM(COALESCE(res.points, 0)) OVER (
            PARTITION BY res.driver_id, ra.year
            ORDER BY ra.round
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS points,
        CAST(res.position_order AS INTEGER) AS position
    FROM results res
    JOIN constructors c ON res.constructor_id = c.constructor_id
    JOIN drivers da     ON res.driver_id       = da.driver_id
    JOIN races ra       ON res.race_id         = ra.race_id
    WHERE c.constructor_ref IN ({placeholders})
    ORDER BY da.driver_id, ra.year, ra.round
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def _grid_finish_df(engine: Engine, team_refs: list[str]) -> pd.DataFrame:
    placeholders, params = ref_params(team_refs)
    sql = (
        "SELECT COALESCE(da.forename,'') || ' ' || COALESCE(da.surname,'') AS driver,"
        " ra.year, CAST(r.grid AS INTEGER) AS grid,"
        " CAST(r.position_order AS INTEGER) AS finish"
        " FROM results r"
        " JOIN constructors c ON r.constructor_id = c.constructor_id"
        " JOIN drivers da     ON r.driver_id       = da.driver_id"
        " JOIN races ra       ON r.race_id          = ra.race_id"
        f" WHERE c.constructor_ref IN ({placeholders})"
        "   AND r.grid > 0 AND r.position_order < 999"
    )
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# --------------------------------------------------------------------------- #
#  2D chart builders                                                            #
# --------------------------------------------------------------------------- #

def chart_championship_2d(traj_df: pd.DataFrame) -> go.Figure:
    """Championship points per round — latest season. Area fills + win markers."""
    if traj_df.empty:
        return go.Figure(layout=_layout_2d(
            "CHAMPIONSHIP TRAJECTORY", xaxis_title="ROUND", yaxis_title="POINTS", height=460,
        ))

    latest = int(traj_df["year"].max())
    df = traj_df[traj_df["year"] == latest].sort_values("round")
    layout = _layout_2d(
        f"CHAMPIONSHIP TRAJECTORY · {latest}",
        xaxis_title="ROUND",
        yaxis_title="POINTS",
        height=460,
    )
    fig = go.Figure(layout=layout)

    for i, (driver, g) in enumerate(df.groupby("driver")):
        color = _driver_color(driver, i)
        surname = driver.split()[-1]

        fig.add_trace(go.Scatter(
            x=g["round"], y=g["points"],
            mode="none",
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, 0.12),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=g["round"], y=g["points"],
            mode="lines+markers",
            name=surname,
            line=dict(color=color, width=2.5, shape="spline"),
            marker=dict(size=6, color=color, line=dict(color="#030F1A", width=1)),
            hovertemplate=f"<b>{surname}</b>  %{{y}} pts<extra></extra>",
        ))

        wins = g[g["position"] == 1]
        if not wins.empty:
            fig.add_trace(go.Scatter(
                x=wins["round"], y=wins["points"],
                mode="markers",
                name=f"{surname} win",
                marker=dict(
                    symbol="star-diamond", size=12,
                    color="#FFD700",
                    line=dict(color="#030F1A", width=1),
                ),
                showlegend=False,
                hovertemplate=f"<b>{surname}</b>  WIN  %{{y}} pts<extra></extra>",
            ))

        last = g.sort_values("round").iloc[-1]
        fig.add_annotation(
            x=last["round"], y=last["points"],
            text=f" {int(last['points'])}",
            font=dict(color=color, family=_FONT, size=10),
            showarrow=False, xanchor="left",
        )

    return fig


def chart_positions_bump_2d(traj_df: pd.DataFrame) -> go.Figure:
    """Bump chart of race finish positions — latest season. Thick spline lines, P1 zone shaded."""
    if traj_df.empty:
        layout = _layout_2d("FINISH POSITIONS", xaxis_title="ROUND", yaxis_title="FINISH", height=420)
        layout.yaxis.update(autorange="reversed", dtick=5)
        return go.Figure(layout=layout)

    latest = int(traj_df["year"].max())
    df = traj_df[(traj_df["year"] == latest) & (traj_df["position"] < 999)].sort_values("round")
    layout = _layout_2d(
        f"FINISH POSITIONS · {latest}",
        xaxis_title="ROUND",
        yaxis_title="FINISH",
        height=420,
    )
    layout.yaxis.update(autorange="reversed", dtick=5)
    fig = go.Figure(layout=layout)

    fig.add_hrect(y0=0.5, y1=1.5, fillcolor="rgba(255,215,0,0.10)", layer="below", line_width=0)
    fig.add_hrect(y0=0.5, y1=3.5, fillcolor="rgba(255,255,255,0.04)", layer="below", line_width=0)

    for i, (driver, g) in enumerate(df.groupby("driver")):
        color = _driver_color(driver, i)
        fig.add_trace(go.Scatter(
            x=g["round"], y=g["position"],
            mode="lines+markers",
            name=driver.split()[-1],
            line=dict(color=color, width=2.5, shape="linear"),
            marker=dict(size=7, color=color, line=dict(color="#030F1A", width=1.5)),
            hovertemplate="<b>%{fullData.name}</b>  P%{y}<extra></extra>",
        ))
    return fig


def chart_points_gap_2d(traj_df: pd.DataFrame) -> go.Figure:
    """Points gap between the two drivers — latest season.
    Gap = d_a (season leader) minus d_b.  Positive = d_a leads, negative = d_b leads.
    Step interpolation: gap is a discrete step function that changes only at race events.
    Line color tracks the current leader so the viewer always knows who's ahead."""
    layout = _layout_2d(
        "POINTS GAP · DRIVERS",
        xaxis_title="ROUND",
        yaxis_title="GAP (PTS)",
        height=420,
        hovermode="closest",
    )
    fig = go.Figure(layout=layout)

    if traj_df.empty:
        return fig

    latest = int(traj_df["year"].max())
    df = traj_df[traj_df["year"] == latest].sort_values("round")
    drivers = df["driver"].unique()
    if len(drivers) < 2:
        return fig

    pivot = df.pivot_table(index="round", columns="driver", values="points", aggfunc="last").ffill()
    # Sort by total season points descending so d_a is always the championship leader
    total_pts = df.groupby("driver")["points"].max()
    sorted_drivers = total_pts.sort_values(ascending=False).index.tolist()
    d_a = next((d for d in sorted_drivers if d in pivot.columns), None)
    remaining = [d for d in sorted_drivers if d in pivot.columns and d != d_a]
    d_b = remaining[0] if remaining else None
    if d_a is None or d_b is None:
        return fig

    rounds = pivot.index.tolist()
    gap = (pivot[d_a] - pivot[d_b]).tolist()
    # Resolve white to a visible grey so it renders on dark background
    color_a = _driver_color(d_a, 0) if _driver_color(d_a, 0) != "#FFFFFF" else "#AAAAAA"
    color_b = _driver_color(d_b, 1) if _driver_color(d_b, 1) != "#FFFFFF" else "#888888"
    surname_a = d_a.split()[-1]
    surname_b = d_b.split()[-1]

    # Zero reference
    fig.add_hline(y=0, line=dict(color=_ZERO_LINE, width=1.5, dash="dot"))

    # Area fills — show which driver is leading at a glance
    pos_gap = [g if g >= 0 else 0 for g in gap]
    neg_gap = [g if g < 0 else 0 for g in gap]
    fig.add_trace(go.Scatter(
        x=rounds, y=pos_gap,
        mode="none", fill="tozeroy",
        fillcolor=_hex_to_rgba(color_a, 0.25),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=neg_gap,
        mode="none", fill="tozeroy",
        fillcolor=_hex_to_rgba(color_b, 0.25),
        showlegend=False, hoverinfo="skip",
    ))

    # Gap line — step interpolation (gap only changes at race events)
    # Color segments by leader: build separate traces per contiguous leader block
    leader_color = color_a if gap[0] >= 0 else color_b
    seg_x: list = [rounds[0]]
    seg_y: list = [gap[0]]
    for i in range(1, len(gap)):
        cur_color = color_a if gap[i] >= 0 else color_b
        if cur_color != leader_color:
            # Close the segment and start a new one, sharing the boundary point
            seg_x.append(rounds[i])
            seg_y.append(gap[i])
            fig.add_trace(go.Scatter(
                x=seg_x, y=seg_y,
                mode="lines",
                line=dict(color=leader_color, width=2.5, shape="hv"),
                showlegend=False, hoverinfo="skip",
            ))
            seg_x = [rounds[i - 1], rounds[i]]
            seg_y = [gap[i - 1], gap[i]]
            leader_color = cur_color
        else:
            seg_x.append(rounds[i])
            seg_y.append(gap[i])
    # Final segment
    fig.add_trace(go.Scatter(
        x=seg_x, y=seg_y,
        mode="lines",
        line=dict(color=leader_color, width=2.5, shape="hv"),
        showlegend=False, hoverinfo="skip",
    ))

    # Markers per round with full hover context
    fig.add_trace(go.Scatter(
        x=rounds, y=gap,
        mode="markers",
        name=f"{surname_a} vs {surname_b}",
        marker=dict(
            size=7,
            color=[color_a if g >= 0 else color_b for g in gap],
            line=dict(color="#030F1A", width=1),
        ),
        customdata=[[surname_a if (g or 0) >= 0 else surname_b, abs(int(g)) if pd.notna(g) else 0] for g in gap],
        hovertemplate=(
            "<b>R%{x}</b>  %{customdata[0]} leads by %{customdata[1]} pts<extra></extra>"
        ),
    ))

    # End annotation — show the final gap prominently
    final_gap = gap[-1]
    final_round = rounds[-1]
    leader_name = surname_a if final_gap >= 0 else surname_b
    leader_col = color_a if final_gap >= 0 else color_b
    fig.add_annotation(
        x=final_round, y=final_gap,
        text=f"  {leader_name} +{abs(int(final_gap))} pts",
        font=dict(color=leader_col, family=_FONT, size=10),
        showarrow=True, arrowcolor=leader_col, arrowwidth=1,
        arrowhead=0, ax=20, ay=-20,
        xanchor="left",
    )

    # Zone labels — anchored to right edge, clearly in each zone
    fig.add_annotation(
        x=0.98, y=0.93, xref="paper", yref="paper",
        text=f"↑ {surname_a} leads",
        font=dict(color=color_a, family=_FONT, size=9),
        showarrow=False, xanchor="right",
    )
    fig.add_annotation(
        x=0.98, y=0.07, xref="paper", yref="paper",
        text=f"↓ {surname_b} leads",
        font=dict(color=color_b, family=_FONT, size=9),
        showarrow=False, xanchor="right",
    )
    return fig


def chart_heatmap_2d(traj_df: pd.DataFrame) -> go.Figure:
    """Performance matrix — finish position per driver per round, all seasons as color cells."""
    layout = _layout_2d(
        "PERFORMANCE MATRIX · ALL SEASONS",
        xaxis_title="ROUND",
        height=400,
        hovermode="closest",
    )
    fig = go.Figure(layout=layout)
    fig.update_layout(yaxis_showgrid=False)

    if traj_df.empty:
        return fig

    df = traj_df[traj_df["position"] < 999].copy()
    df["pos_clip"] = df["position"].clip(upper=20)
    df["label"] = df["position"].apply(lambda p: "DNF" if p >= 999 else f"P{int(p)}")
    df["yr_round"] = df["year"].astype(str) + "·R" + df["round"].astype(str).str.zfill(2)

    cols = sorted(df["yr_round"].unique())
    rows = sorted(df["driver"].unique())

    z = [[None] * len(cols) for _ in rows]
    text = [[""] * len(cols) for _ in rows]

    col_idx = {c: i for i, c in enumerate(cols)}
    row_idx = {r: i for i, r in enumerate(rows)}

    for _, row in df.iterrows():
        ri = row_idx[row["driver"]]
        ci = col_idx[row["yr_round"]]
        z[ri][ci] = float(row["pos_clip"])
        text[ri][ci] = row["label"]

    colorscale = [
        [0.00, "#00D4FF"],
        [0.10, "#1E41FF"],
        [0.40, "#0A2035"],
        [0.75, "#8B0000"],
        [1.00, "#4A0000"],
    ]

    fig.add_trace(go.Heatmap(
        z=z, x=cols, y=rows,
        text=text,
        colorscale=colorscale,
        zmin=1, zmax=20,
        showscale=True,
        colorbar=dict(
            title=dict(text="POS", font=dict(family=_FONT, color=_TICK, size=9)),
            tickfont=dict(family=_FONT, color=_TICK, size=8),
            thickness=10, len=0.8,
        ),
        xgap=2, ygap=2,
        hovertemplate="<b>%{y}</b>  %{x}<br>%{text}<extra></extra>",
    ))

    fig.update_layout(margin=dict(l=100, r=24, t=56, b=80))
    fig.update_xaxes(tickfont=dict(size=7), tickangle=90)
    return fig


def chart_grid_finish_2d(df: pd.DataFrame) -> go.Figure:
    """Grid vs finish scatter with per-driver OLS regression lines — all seasons."""
    layout = _layout_2d(
        "PACE · GRID vs FINISH · ALL SEASONS",
        xaxis_title="GRID POSITION",
        yaxis_title="FINISH POSITION",
        height=440,
        hovermode="closest",
    )
    fig = go.Figure(layout=layout)
    if df.empty:
        return fig

    mx = int(max(df["grid"].max(), df["finish"].max())) + 1
    fig.add_trace(go.Scatter(
        x=[1, mx], y=[1, mx],
        mode="lines",
        line=dict(color=_ZERO_LINE, width=1.5, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
        name="grid = finish",
    ))
    fig.add_annotation(
        x=mx * 0.72, y=mx * 0.72,
        text="grid = finish",
        font=dict(color=_TICK, family=_FONT, size=8),
        showarrow=False, xanchor="left", yanchor="bottom",
        textangle=-45,
    )
    fig.update_yaxes(autorange="reversed")

    years = sorted(df["year"].unique())
    year_norm = {y: i / max(len(years) - 1, 1) for i, y in enumerate(years)}

    for i, (driver, g) in enumerate(df.groupby("driver")):
        color = _driver_color(driver, i)
        point_colors = [
            _hex_to_rgba(color if color != "#FFFFFF" else "#888888",
                         0.4 + 0.55 * year_norm[y])
            for y in g["year"]
        ]
        delta = g["grid"] - g["finish"]
        fig.add_trace(go.Scatter(
            x=g["grid"], y=g["finish"],
            mode="markers",
            name=driver.split()[-1],
            customdata=list(zip(g["year"], delta)),
            marker=dict(
                size=7, color=point_colors,
                line=dict(color="#030F1A", width=0.5),
            ),
            hovertemplate=(
                "<b>%{fullData.name}</b>  %{customdata[0]}<br>"
                "Grid P%{x} → Finish P%{y}<br>"
                "%{customdata[1]:+d} positions<extra></extra>"
            ),
        ))

        if len(g) >= 3:
            m, b = np.polyfit(g["grid"], g["finish"], 1)
            x_r = np.linspace(g["grid"].min(), g["grid"].max(), 40)
            ols_line = dict(
                color=color if color != "#FFFFFF" else "#888888",
                width=1.5, dash="dot",
            )
            fig.add_trace(go.Scatter(
                x=x_r, y=m * x_r + b,
                mode="lines",
                line=ols_line,
                showlegend=False,
                hoverinfo="skip",
            ))

    return fig


# --------------------------------------------------------------------------- #
#  Telemetry chart builders                                                     #
# --------------------------------------------------------------------------- #

def chart_pit_efficiency_2d(df: pd.DataFrame) -> go.Figure:
    """Pit stop efficiency — z-score horizontal bar per driver.
    Error bars show SEM (precision of the mean estimate, not spread of observations).
    Bar color encodes z-score magnitude: cyan = fast, red = slow."""
    layout = _layout_2d(
        "PIT STOP EFFICIENCY · Z-SCORE",
        xaxis_title="← faster than field  ·  Z-SCORE (σ)  ·  slower →",
        height=280,
        hovermode="closest",
    )
    layout.margin.update(dict(l=90, r=60, t=48, b=48))
    fig = go.Figure(layout=layout)
    if df.empty:
        return fig

    # SEM — precision of the mean, not spread of individual stops
    sem = (df["std_z"] / np.sqrt(df["n_stops"])).fillna(0)

    # Continuous color: red (slow, high z) → grey (average) → cyan (fast, low z)
    z_min, z_max = df["mean_z"].min(), df["mean_z"].max()
    z_range = max(z_max - z_min, 1e-9)
    norm = ((df["mean_z"] - z_min) / z_range).tolist()  # 0 = fastest, 1 = slowest
    bar_colors = [
        f"rgba({int(255 * v)},{int(212 * (1 - v))},{int(255 * (1 - v))},0.90)"
        for v in norm
    ]

    surnames = df["driver"].apply(lambda d: d.split()[-1])
    fig.add_trace(go.Bar(
        x=df["mean_z"],
        y=surnames,
        orientation="h",
        marker=dict(color=bar_colors),
        error_x=dict(
            type="data", array=sem.tolist(), visible=True,
            color="#FFFFFF", thickness=1.8, width=5,
        ),
        customdata=list(zip(df["n_stops"], df["std_z"].fillna(0))),
        hovertemplate=(
            "<b>%{y}</b>  z = %{x:.3f}σ<br>"
            "SEM ±%{error_x.array:.3f}σ<br>"
            "n = %{customdata[0]} stops<extra></extra>"
        ),
    ))

    # Zero reference — prominent, not just a dotted line
    fig.add_vline(x=0, line=dict(color=_ACCENT_DIM, width=1.5, dash="dot"))

    # n_stops count to the right of each bar
    x_offset = (z_max - z_min) * 0.04 + 0.02
    for _, row in df.iterrows():
        fig.add_annotation(
            x=max(row["mean_z"], 0) + x_offset,
            y=row["driver"].split()[-1],
            text=f"n={int(row['n_stops'])}",
            font=dict(color=_TICK, family=_FONT, size=8),
            showarrow=False, xanchor="left", yanchor="middle",
        )
    return fig


def chart_dnf_reliability_2d(df: pd.DataFrame) -> go.Figure:
    """DNF rate per driver — dot plot with asymmetric Poisson CI.
    Marker color encodes reliability: green (0 DNFs) → red (high rate).
    CI bars are white for contrast against dark background.
    X/Y count shown beside each marker in the static view."""
    layout = _layout_2d(
        "RELIABILITY MODEL · DNF RATE",
        xaxis_title="DNF RATE  (95% Poisson CI)",
        height=280,
        hovermode="closest",
    )
    layout.margin.update(dict(l=90, r=72, t=48, b=36))
    layout.xaxis.update(tickformat=".0%")
    fig = go.Figure(layout=layout)
    if df.empty:
        return fig

    surnames = df["driver"].apply(lambda d: d.split()[-1])
    err_upper = (df["ci_upper"] - df["rate"]).clip(lower=0)
    err_lower = (df["rate"] - df["ci_lower"]).clip(lower=0)

    # Gradient: green (reliable, rate≈0) → red (unreliable, rate≈max)
    max_rate = max(df["rate"].max(), 1e-9)
    dot_colors = [
        f"rgba({int(220 * r / max_rate)},{int(200 * (1 - r / max_rate))},60,0.95)"
        for r in df["rate"]
    ]

    fig.add_trace(go.Scatter(
        x=df["rate"],
        y=surnames,
        mode="markers",
        marker=dict(
            size=11,
            color=dot_colors,
            line=dict(color="#030F1A", width=1.5),
        ),
        error_x=dict(
            type="data", symmetric=False,
            array=err_upper.tolist(),
            arrayminus=err_lower.tolist(),
            visible=True,
            color="rgba(255,255,255,0.65)",
            thickness=2.5, width=5,
        ),
        customdata=list(zip(df["ci_lower"], df["ci_upper"], df["races"], df["dnfs"])),
        hovertemplate=(
            "<b>%{y}</b>  %{x:.1%} DNF rate<br>"
            "95% CI [%{customdata[0]:.1%}, %{customdata[1]:.1%}]<br>"
            "%{customdata[3]:.0f} DNFs / %{customdata[2]:.0f} races<extra></extra>"
        ),
    ))

    # Static count label — visible without hovering
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row["ci_upper"],
            y=row["driver"].split()[-1],
            text=f"  {int(row['dnfs'])}/{int(row['races'])}",
            font=dict(color=_TICK, family=_FONT, size=8),
            showarrow=False, xanchor="left", yanchor="middle",
        )
    return fig


def chart_sector_delta_2d(df: pd.DataFrame) -> go.Figure:
    """Sector delta from best — grouped horizontal bar (S1/S2/S3).
    Each bar shows how many seconds slower than the fastest driver in that sector.
    Shorter bar = closer to best pace. Best driver per sector annotated with ★."""
    n_drivers = max(len(df), 1)
    chart_height = max(280, n_drivers * 72)
    layout = _layout_2d(
        "SECTOR DELTA · Δ FROM BEST (s)",
        xaxis_title="SECONDS SLOWER THAN BEST",
        height=chart_height,
        hovermode="closest",
    )
    layout.margin.update(dict(l=90, r=48, t=48, b=48))
    layout.update(barmode="group")
    layout.xaxis.update(tickformat=".2f")
    fig = go.Figure(layout=layout)
    if df.empty:
        return fig

    surnames = df["driver"].apply(lambda d: d.split()[-1]).tolist()
    # S2 was near-white (#E0F2FE) — invisible on dark background; use orange instead
    sector_cfg = [
        ("s1_mean", "S1", _ACCENT),       # cyan
        ("s2_mean", "S2", "#FF9500"),      # orange
        ("s3_mean", "S3", "#FFD700"),      # gold
    ]
    for col, label, color in sector_cfg:
        if col not in df.columns:
            continue
        best = df[col].min()
        delta = (df[col] - best).tolist()
        fig.add_trace(go.Bar(
            x=delta,
            y=surnames,
            orientation="h",
            name=label,
            marker=dict(color=color, opacity=0.88),
            customdata=list(zip(df[col], df["n"])),
            hovertemplate=(
                f"<b>%{{y}}</b>  {label}: %{{customdata[0]:.3f}}s"
                f"  (+%{{x:.2f}}s vs best)"
                f"  (n=%{{customdata[1]}} laps)<extra></extra>"
            ),
        ))
        # ★ annotation for the best driver in this sector
        best_idx = int(df[col].idxmin())
        best_surname = df.loc[best_idx, "driver"].split()[-1]
        fig.add_annotation(
            x=0, y=best_surname,
            text=f"★ {label} best",
            font=dict(color=color, family=_FONT, size=7),
            showarrow=False, xanchor="right", xshift=-4, yanchor="middle",
        )
    return fig


# --------------------------------------------------------------------------- #
#  FastF1 lap analysis chart builders                                           #
# --------------------------------------------------------------------------- #

_COMPOUND_COLORS = {
    "SOFT":         "#FF3333",
    "MEDIUM":       "#FFD700",
    "HARD":         "#CCCCCC",
    "INTERMEDIATE": "#39B54A",
    "WET":          "#0067FF",
}


def _no_data_fig(title: str, note: str = "Run pipeline with --telemetry to load lap data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=note, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(color=_TICK, family=_FONT, size=10),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(family=_FONT, size=11, color=_ACCENT), x=0),
        paper_bgcolor=_BG_CARD, plot_bgcolor=_BG_CARD,
        height=240, margin=dict(l=12, r=12, t=36, b=12),
        font=dict(family=_FONT, color=_FONT_COLOR),
    )
    return fig


def chart_tyre_degradation_2d(deg_df: pd.DataFrame) -> go.Figure:
    if deg_df.empty:
        return _no_data_fig("Tyre Degradation · Rate by Compound")

    compounds = [c for c in ("SOFT", "MEDIUM", "HARD") if c in deg_df["compound"].unique()]
    drivers_sorted = (
        deg_df.groupby("driver")["deg_rate_s"].mean()
        .sort_values().index.tolist()
    )
    surnames = [d.split()[-1] for d in drivers_sorted]

    fig = go.Figure()
    for compound in compounds:
        cdf = deg_df[deg_df["compound"] == compound].set_index("driver")
        y_vals = [
            float(cdf.loc[d, "deg_rate_s"]) if d in cdf.index else None
            for d in drivers_sorted
        ]
        r2_vals = [
            f"{float(cdf.loc[d,'r2']):.3f}" if d in cdf.index else "—"
            for d in drivers_sorted
        ]
        n_vals = [
            str(int(cdf.loc[d, "n"])) if d in cdf.index else "—"
            for d in drivers_sorted
        ]
        fig.add_trace(go.Bar(
            x=surnames, y=y_vals,
            name=compound,
            marker=dict(color=_COMPOUND_COLORS.get(compound, "#888888"), opacity=0.88),
            customdata=list(zip(r2_vals, n_vals)),
            hovertemplate=(
                "<b>%{x}</b> — " + compound +
                "<br>Rate: %{y:+.4f} s/lap"
                "<br>R²: %{customdata[0]}  n=%{customdata[1]}"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line=dict(color=_ZERO_LINE, width=1, dash="dot"))
    fig.update_layout(
        barmode="group",
        title=dict(text="Tyre Degradation · Rate by Compound", font=dict(family=_FONT, size=11, color=_ACCENT), x=0),
        paper_bgcolor=_BG_CARD, plot_bgcolor=_BG_CARD,
        height=300,
        margin=dict(l=48, r=16, t=40, b=40),
        font=dict(family=_FONT, color=_FONT_COLOR, size=9),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8)),
        xaxis=dict(gridcolor=_GRID, tickfont=dict(size=9, color=_TICK)),
        yaxis=dict(
            title=dict(text="s / lap", font=dict(size=9, color=_TICK)),
            gridcolor=_GRID, tickfont=dict(size=9, color=_TICK), zeroline=False,
        ),
        hovermode="closest",
    )
    return fig


def chart_pit_strategy_2d(strategy_df: pd.DataFrame) -> go.Figure:
    if strategy_df.empty:
        return _no_data_fig("Race Strategy · Stint Structure")

    race_name = strategy_df["race_name"].iloc[0]
    year = int(strategy_df["year"].iloc[0])

    finish_order = (
        strategy_df.groupby("driver")["finish_pos"].min()
        .sort_values().index.tolist()
    )
    surnames_ordered = [d.split()[-1] for d in finish_order]

    fig = go.Figure()
    seen: set[str] = set()

    for _, row in strategy_df.iterrows():
        compound = str(row["compound"]).upper()
        color = _COMPOUND_COLORS.get(compound, "#888888")
        surname = row["driver"].split()[-1]
        show_legend = compound not in seen
        seen.add(compound)
        width = int(row["end_lap"]) - int(row["start_lap"]) + 1

        fig.add_trace(go.Bar(
            x=[width],
            y=[surname],
            base=int(row["start_lap"]) - 1,
            orientation="h",
            name=compound,
            marker=dict(color=color, opacity=0.88, line=dict(color=_BG, width=0.8)),
            showlegend=show_legend,
            hovertemplate=(
                f"<b>{surname}</b> — Stint {int(row['stint'])}<br>"
                f"{compound}<br>"
                f"Laps {int(row['start_lap'])}–{int(row['end_lap'])} "
                f"({int(row['stint_laps'])} laps)<extra></extra>"
            ),
        ))

    n_drivers = max(len(finish_order), 1)
    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"Race Strategy · {race_name} {year}",
            font=dict(family=_FONT, size=11, color=_ACCENT), x=0,
        ),
        paper_bgcolor=_BG_CARD, plot_bgcolor=_BG_CARD,
        height=max(220, n_drivers * 52 + 80),
        margin=dict(l=60, r=16, t=40, b=36),
        font=dict(family=_FONT, color=_FONT_COLOR, size=9),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8)),
        xaxis=dict(
            title=dict(text="Lap", font=dict(size=9, color=_TICK)),
            gridcolor=_GRID, tickfont=dict(size=9, color=_TICK), zeroline=False,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=surnames_ordered[::-1],
            tickfont=dict(size=9, color=_TICK), gridcolor=_GRID,
        ),
        hovermode="closest",
    )
    return fig


# --------------------------------------------------------------------------- #
#  Dashboard generator                                                          #
# --------------------------------------------------------------------------- #

def generate_dashboard(
    engine: Engine,
    team_refs: list[str],
    team_name: str,
    output_path: str,
) -> None:
    traj = _round_by_round_df(engine, team_refs)
    if traj.empty:
        logger.warning("_round_by_round_df returned no data — dashboard charts will be blank")

    gf = _grid_finish_df(engine, team_refs)

    try:
        pit_df = pit_stop_efficiency(engine, team_refs)
    except Exception:
        pit_df = pd.DataFrame()
    try:
        dnf_df = dnf_rate_model(engine, team_refs)
    except Exception:
        dnf_df = pd.DataFrame()
    try:
        sec_df = sector_deltas(engine, team_refs)
    except Exception:
        sec_df = pd.DataFrame()
    try:
        deg_df = tyre_degradation(engine, team_refs)
    except Exception:
        deg_df = pd.DataFrame()
    try:
        strat_df = pit_strategy(engine, team_refs)
    except Exception:
        strat_df = pd.DataFrame()

    fig1 = chart_championship_2d(traj)
    fig2 = chart_positions_bump_2d(traj)
    fig3 = chart_points_gap_2d(traj)
    fig4 = chart_heatmap_2d(traj)
    fig5 = chart_grid_finish_2d(gf)
    fig6 = chart_pit_efficiency_2d(pit_df)
    fig7 = chart_dnf_reliability_2d(dnf_df)
    fig8  = chart_sector_delta_2d(sec_df)
    fig9  = chart_tyre_degradation_2d(deg_df)
    fig10 = chart_pit_strategy_2d(strat_df)

    _cfg = {"displayModeBar": "hover", "scrollZoom": False}
    div1  = fig1.to_html(full_html=False, include_plotlyjs="cdn",  config=_cfg)
    div2  = fig2.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div3  = fig3.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div4  = fig4.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div5  = fig5.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div6  = fig6.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div7  = fig7.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div8  = fig8.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div9  = fig9.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div10 = fig10.to_html(full_html=False, include_plotlyjs=False, config=_cfg)

    years = sorted(traj["year"].unique()) if not traj.empty else []
    year_range = f"{years[0]}–{years[-1]}" if years else ""
    subtitle = f"PERFORMANCE DASHBOARD \xb7 {year_range}" if year_range else "PERFORMANCE DASHBOARD"

    total_races = int(traj["round"].nunique()) if not traj.empty else "—"
    total_wins  = int((gf["finish"] == 1).sum()) if not gf.empty else "—"
    driver_count = str(len(traj["driver"].unique())) if not traj.empty else "—"

    def _stat(val: str, unit: str, lbl: str, ok_text: str) -> str:
        return (
            f'<div class="stat-card">'
            f'<div class="stat-top"><div class="stat-val">{val}</div>'
            f'<div class="stat-unit">{unit}</div></div>'
            f'<div class="stat-lbl">{lbl}</div>'
            f'<div class="stat-status"><div class="stat-dot"></div>'
            f'<div class="stat-ok-text">{ok_text}</div></div>'
            f'</div>'
        )

    stat_html = (
        _stat(year_range or "—", "YRS", "Data Coverage", "NOMINAL")
        + _stat(str(total_races), "RND", "Race Rounds Analyzed", "LOADED")
        + _stat(str(total_wins), "W", "Wins in Dataset", "CONFIRMED")
        + _stat("4", "WCC", "Constructors Titles", "VERIFIED")
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    logo_html = ""
    if os.path.exists(_LOGO_PATH):
        with open(_LOGO_PATH, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        logo_html = f'<img src="data:image/jpeg;base64,{b64}" class="logo-img" alt="Red Bull Racing">'

    html = (
        _HTML_TEMPLATE
        .replace("PLACEHOLDER_TITLE",        team_name)
        .replace("PLACEHOLDER_SUBTITLE",     subtitle)
        .replace("PLACEHOLDER_LOGO",         logo_html)
        .replace("PLACEHOLDER_STATS",        stat_html)
        .replace("PLACEHOLDER_C1",           div1)
        .replace("PLACEHOLDER_C2",           div2)
        .replace("PLACEHOLDER_C3",           div3)
        .replace("PLACEHOLDER_C4",           div4)
        .replace("PLACEHOLDER_C5",           div5)
        .replace("PLACEHOLDER_C6",           div6)
        .replace("PLACEHOLDER_C7",           div7)
        .replace("PLACEHOLDER_C8",           div8)
        .replace("PLACEHOLDER_C9",           div9)
        .replace("PLACEHOLDER_C10",          div10)
        .replace("PLACEHOLDER_DRIVER_COUNT", driver_count)
        .replace("PLACEHOLDER_TS",           ts)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
