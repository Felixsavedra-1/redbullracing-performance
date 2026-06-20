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
from constants import DNF_POSITION_ORDER

logger = logging.getLogger("f1_analytics")

_LOGO_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "redbullracinglogo.jpg")
)

_BG         = "#000000"
_BG_CARD    = "#0B0A08"
_BG_HOVER   = "#16140F"
_GRID       = "#1A1814"
_GRID_SOFT  = "#141210"
_ZERO_LINE  = "#26231C"
_TICK       = "#6A7176"
_FONT_COLOR = "#ECE5D5"
_FONT       = "'Inter', 'Helvetica Neue', Arial, sans-serif"
_MONO       = "'Space Mono', 'SFMono-Regular', Menlo, monospace"
_ACCENT     = "#B3122B"
_ACCENT_HI  = "#D6203F"
_ACCENT_DIM = "#3A1A1F"
_NEUTRAL    = "#9DA3A8"
_NEUTRAL_DIM = "#6A7176"
_POSITIVE   = "#C7A06A"
_STATUS_OK  = "#B3122B"
_SPIKE      = "#6A7176"
_GLOW       = "#B3122B"

_DRIVER_COLORS = {
    "Verstappen": "#B3122B",
    "Pérez":      "#ECE5D5",
    "Tsunoda":    "#9DA3A8",
    "Lawson":     "#C7A06A",
}
_FALLBACK_COLORS = ["#B3122B", "#ECE5D5", "#9DA3A8", "#C7A06A", "#6A7176"]


def _driver_color(name: str, idx: int) -> str:
    for surname, color in _DRIVER_COLORS.items():
        if surname in name:
            return color
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]



_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
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
:root{--bg:#000;--bg-card:#0B0A08;--bg-hover:#16140F;--accent:#B3122B;--accent-soft:rgba(179,18,43,.14);--text:#ECE5D5;--dim:#7E858B;--neutral:#9DA3A8;--positive:#C7A06A;--border:#1A1814;--line:#26231C;--elev-1:#0E0D0A;--elev-2:#080706;--glow:rgba(179,18,43,.12);--hair:rgba(236,229,213,.05);--font:'Inter','Helvetica Neue','Helvetica',Arial,sans-serif;--mono:'Space Mono','SFMono-Regular',ui-monospace,Menlo,Consolas,monospace}
*{margin:0;padding:0;box-sizing:border-box}
::selection{background:rgba(179,18,43,.45);color:#fff}
html{scrollbar-color:#26231C #000;scrollbar-width:thin}
::-webkit-scrollbar{width:10px}
::-webkit-scrollbar-track{background:#000}
::-webkit-scrollbar-thumb{background:#1A1814;border:2px solid #000;border-radius:5px}
::-webkit-scrollbar-thumb:hover{background:#26231C}
body{background:#000;color:var(--text);font-family:var(--font);min-height:100vh;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body::before{content:'';position:fixed;inset:0;z-index:1;pointer-events:none;background:radial-gradient(circle,rgba(236,229,213,.045) 1px,transparent 1.4px) 0 0/26px 26px;mask-image:radial-gradient(ellipse at 50% 16%,rgba(0,0,0,.6),transparent 70%);-webkit-mask-image:radial-gradient(ellipse at 50% 16%,rgba(0,0,0,.6),transparent 70%)}
.status-bar{display:flex;align-items:center;gap:18px;padding:11px 44px;background:rgba(0,0,0,.74);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid var(--border);font-size:.58rem;letter-spacing:.18em;text-transform:uppercase;color:var(--dim);position:sticky;top:0;z-index:100;flex-wrap:wrap}
.sb-dot{width:5px;height:5px;border-radius:50%;background:var(--accent);flex-shrink:0}
.sb-label{color:var(--dim)}
.sb-val{color:var(--text);font-weight:700;font-family:var(--mono);letter-spacing:.06em}
.sb-sep{color:#282C2E}
.sb-spacer{margin-left:auto}
.sb-rec{display:inline-flex;align-items:center;gap:6px;color:var(--accent);font-weight:700}
.sb-rec b{width:5px;height:5px;border-radius:50%;background:var(--accent)}
.brand-badge{display:flex;flex-direction:column;gap:3px;padding:8px 14px;background:linear-gradient(180deg,rgba(11,10,8,.78),rgba(8,7,6,.86));backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid rgba(214,32,63,.18);border-radius:7px;box-shadow:0 0 0 1px rgba(214,32,63,.10),0 10px 30px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.05)}
.brand-name{font-family:var(--font);font-size:.74rem;font-weight:700;letter-spacing:.12em;color:var(--text);line-height:1;text-transform:none}
.brand-sub{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:.46rem;font-weight:600;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;line-height:1}
.brand-sub::before{content:'';width:4px;height:4px;border-radius:50%;background:var(--accent);box-shadow:0 0 5px rgba(179,18,43,.8);flex-shrink:0}
.brand-badge.in-game{position:fixed;top:12px;left:12px;z-index:10002;pointer-events:none}
header{padding:56px 44px 40px;border-bottom:1px solid var(--border);background:#000;position:relative;overflow:hidden}
.hd-team{font-size:.62rem;font-weight:600;letter-spacing:.24em;color:var(--dim);text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:10px}
.hd-team::before{content:'';width:18px;height:1px;background:var(--accent)}
.hd-ghost{position:absolute;right:38px;bottom:-14px;font-family:var(--mono);font-size:7rem;font-weight:700;line-height:1;letter-spacing:-.04em;color:transparent;-webkit-text-stroke:1px rgba(236,229,213,.07);pointer-events:none;user-select:none;white-space:nowrap}
h1{font-size:2.6rem;font-weight:700;letter-spacing:-.03em;line-height:1.04}
h1 span.accent{color:var(--accent)}
.sub{color:var(--dim);font-size:.72rem;font-weight:500;letter-spacing:.16em;margin-top:14px;text-transform:uppercase}
.hero{display:grid;grid-template-columns:1fr 1fr;align-items:stretch;border-bottom:1px solid var(--border)}
.hero .cluster{border-bottom:none;border-right:1px solid var(--border);display:flex;flex-direction:column;justify-content:center}
.hero .car-viewer{border-bottom:none}
.hero .cluster-body{grid-template-columns:1fr;gap:30px;align-items:start}
.hero .gauges{justify-content:space-between}
.cluster{position:relative;border-bottom:1px solid var(--border);background:#000;padding:38px 44px 42px;overflow:hidden}
/* faint HUD grid backdrop */
.cluster::before{content:'';position:absolute;inset:0;pointer-events:none;z-index:0;background:linear-gradient(rgba(236,229,213,.025) 1px,transparent 1px) 0 0/100% 30px,linear-gradient(90deg,rgba(236,229,213,.025) 1px,transparent 1px) 0 0/30px 100%,radial-gradient(ellipse at 28% 50%,rgba(179,18,43,.06),transparent 60%);-webkit-mask:linear-gradient(#000,#000);opacity:.7}
/* 4 corner brackets — targeting frame */
.cluster::after{content:'';position:absolute;inset:14px;pointer-events:none;z-index:0;background:linear-gradient(var(--accent),var(--accent)) 0 0/18px 1px,linear-gradient(var(--accent),var(--accent)) 0 0/1px 18px,linear-gradient(var(--accent),var(--accent)) 100% 0/18px 1px,linear-gradient(var(--accent),var(--accent)) 100% 0/1px 18px,linear-gradient(var(--accent),var(--accent)) 0 100%/18px 1px,linear-gradient(var(--accent),var(--accent)) 0 100%/1px 18px,linear-gradient(var(--accent),var(--accent)) 100% 100%/18px 1px,linear-gradient(var(--accent),var(--accent)) 100% 100%/1px 18px;background-repeat:no-repeat;opacity:.45}
.cluster-hd{display:flex;align-items:center;gap:12px;font-size:.56rem;font-weight:600;letter-spacing:.26em;color:var(--dim);text-transform:uppercase;margin-bottom:28px;font-family:var(--font);position:relative;z-index:2}
.cluster-hd>.dot{width:6px;height:6px;background:var(--accent);flex-shrink:0;animation:hdPulse 2.4s ease-in-out infinite}
@keyframes hdPulse{0%,100%{opacity:1;box-shadow:0 0 8px rgba(179,18,43,.9)}50%{opacity:.45;box-shadow:0 0 2px rgba(179,18,43,.3)}}
.cluster-hd>.ln{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}
.cluster-hd>.tag{font-family:var(--mono);color:var(--accent);letter-spacing:.10em}
.cluster-body{display:grid;grid-template-columns:auto 1fr;gap:42px;align-items:center;position:relative;z-index:2}
.gauges{display:flex;gap:28px;flex-shrink:0}
.gauge{display:flex;flex-direction:column;align-items:center;animation:fadeUp .6s ease both}
.gauge:nth-child(2){animation-delay:.08s}.gauge:nth-child(3){animation-delay:.16s}
.g-ring{position:relative;width:124px;height:124px}
.g-ring::before{content:'';position:absolute;inset:-14px;border-radius:50%;background:radial-gradient(circle,rgba(179,18,43,.10),transparent 68%);pointer-events:none;animation:gHalo 5s ease-in-out infinite}
@keyframes gHalo{0%,100%{opacity:.5}50%{opacity:1}}
.g-ring svg{position:absolute;inset:0;transform:rotate(-90deg)}
/* counter-rotating segmented holo rings — irregular arc fragments at independent speeds */
.g-seg,.g-seg2,.g-coil,.g-flow{transform-box:fill-box;transform-origin:center}
.g-seg{fill:none;stroke:var(--accent);stroke-width:1.5;opacity:.55;animation:gSpin 34s linear infinite}
.g-seg2{fill:none;stroke:var(--neutral);stroke-width:1;opacity:.35;animation:gSpinR 52s linear infinite}
/* inner reactor coil — chunky segments, slow counter-rotation */
.g-coil{fill:none;stroke:var(--accent);stroke-width:2.5;opacity:.26;animation:gSpinR 18s linear infinite}
/* energy circulating along the gauge ring */
.g-flow{fill:none;stroke:var(--text);stroke-width:7;stroke-dasharray:2.5 35;stroke-linecap:round;opacity:.28;mix-blend-mode:screen;animation:gFlow 2.8s linear infinite}
@keyframes gFlow{to{stroke-dashoffset:-37.5}}
.gauge:hover .g-seg{animation-duration:7s}
.gauge:hover .g-seg2{animation-duration:11s}
/* steady reactor-core glow — bone-hot centre, crimson falloff, slow breathing */
.g-core{position:absolute;inset:33px;border-radius:50%;pointer-events:none;background:radial-gradient(circle,rgba(236,229,213,.18),rgba(179,18,43,.20) 38%,rgba(179,18,43,.05) 62%,transparent 72%);animation:gCore 4.2s ease-in-out infinite}
@keyframes gCore{0%,100%{opacity:.65;transform:scale(.98)}50%{opacity:1;transform:scale(1.02)}}
@keyframes gSpin{to{transform:rotate(360deg)}}
@keyframes gSpinR{to{transform:rotate(-360deg)}}
.g-val{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:1.55rem;font-weight:700;color:var(--text);letter-spacing:-.02em;font-variant-numeric:tabular-nums;text-shadow:0 0 12px rgba(179,18,43,.55),0 0 2px rgba(236,229,213,.4);z-index:2}
.g-val small{font-size:.72rem;color:var(--dim);margin-left:1px;font-weight:400;text-shadow:none}
.g-lbl{margin-top:11px;font-size:.50rem;font-weight:600;letter-spacing:.18em;color:var(--dim);text-transform:uppercase;text-align:center}
.g-track{fill:none;stroke:#16140F}
.g-arc{fill:none;stroke:var(--accent);stroke-linecap:round;filter:drop-shadow(0 0 4px rgba(179,18,43,.65));transition:stroke-dashoffset 1.3s cubic-bezier(.16,1,.3,1)}
.g-tick{fill:none;stroke:#26231C;stroke-width:4}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:6px;overflow:hidden}
.kpi{background:var(--elev-1);padding:16px 17px;position:relative;transition:background .18s;animation:fadeUp .55s ease both}
.kpi:nth-child(n+2){animation-delay:.05s}.kpi:nth-child(n+5){animation-delay:.12s}
.kpi:hover{background:var(--bg-hover)}
.kpi::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--accent);transform:scaleY(0);transform-origin:top;transition:transform .28s cubic-bezier(.16,1,.3,1)}
.kpi:hover::before{transform:scaleY(1)}
.kpi::after{content:'';position:absolute;top:12px;right:12px;width:3px;height:3px;border-radius:50%;background:var(--accent);box-shadow:0 0 5px rgba(179,18,43,.8);animation:kpiPulse 3.2s ease-in-out infinite}
@keyframes kpiPulse{0%,100%{opacity:.9}50%{opacity:.22}}
.kpi:nth-child(2)::after{animation-delay:.3s}.kpi:nth-child(3)::after{animation-delay:.6s}.kpi:nth-child(4)::after{animation-delay:.9s}.kpi:nth-child(5)::after{animation-delay:1.2s}.kpi:nth-child(6)::after{animation-delay:.5s}.kpi:nth-child(7)::after{animation-delay:1.5s}.kpi:nth-child(8)::after{animation-delay:.8s}
.kpi-top{display:flex;align-items:baseline;gap:4px}
.kpi-val{font-family:var(--mono);font-size:1.3rem;font-weight:700;color:var(--text);letter-spacing:-.01em;line-height:1;font-variant-numeric:tabular-nums}
.kpi-unit{font-size:.5rem;color:var(--dim);letter-spacing:.12em;text-transform:uppercase}
.kpi-lbl{font-size:.5rem;color:var(--dim);font-weight:500;letter-spacing:.14em;margin-top:9px;text-transform:uppercase}
.ticker{position:relative;border-bottom:1px solid var(--border);background:#040406;overflow:hidden;white-space:nowrap}
.ticker::before,.ticker::after{content:'';position:absolute;top:0;bottom:0;width:60px;z-index:2;pointer-events:none}
.ticker::before{left:0;background:linear-gradient(90deg,#040406,transparent)}
.ticker::after{right:0;background:linear-gradient(270deg,#040406,transparent)}
.ticker-track{display:inline-flex;align-items:center;will-change:transform;animation:tickerScroll 42s linear infinite}
.ticker:hover .ticker-track{animation-play-state:paused}
.ticker-item{display:inline-flex;align-items:center;gap:8px;padding:8px 22px;font-family:var(--mono);font-size:.6rem;letter-spacing:.10em;color:var(--dim);text-transform:uppercase}
.ticker-item b{color:var(--text);font-weight:700}
.ticker-item .k{color:var(--accent)}
.ticker-item::before{content:'';width:4px;height:4px;border-radius:50%;background:var(--accent);opacity:.7;margin-right:4px}
@keyframes tickerScroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.car-viewer{padding:0;display:flex;justify-content:center;border-bottom:1px solid var(--border);background:#000;cursor:pointer;position:relative}
.car-viewer::before{content:'RB \\00B7 STUDIO RENDER';position:absolute;top:16px;left:24px;font-size:.56rem;font-weight:500;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;font-family:var(--font);z-index:2}
#f1car{display:block;width:100%;height:480px}
.race-cta{position:absolute;bottom:28px;left:50%;transform:translateX(-50%);display:inline-flex;align-items:center;gap:11px;font-family:var(--font);font-size:.76rem;font-weight:700;letter-spacing:.20em;text-transform:uppercase;color:#FCEFF1;background:linear-gradient(180deg,#D6203F 0%,#9E1226 55%,#7A0C1C 100%);border:1px solid #E03A52;border-radius:7px;padding:15px 34px;cursor:pointer;z-index:3;transition:transform .16s ease,filter .16s ease,box-shadow .16s ease;box-shadow:0 8px 22px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.32),inset 0 -2px 4px rgba(0,0,0,.4),0 0 0 1px rgba(0,0,0,.25)}
.race-cta::before{content:'';position:absolute;inset:-9px;border-radius:13px;background:radial-gradient(ellipse at 50% 50%,rgba(214,32,63,.55),rgba(179,18,43,.18) 60%,transparent 75%);box-shadow:0 0 30px 6px rgba(214,32,63,.45);z-index:-1;filter:blur(3px);pointer-events:none;animation:ctaGlow 2.4s ease-in-out infinite}
.race-cta:hover{transform:translateX(-50%) translateY(-2px);filter:brightness(1.12);box-shadow:0 12px 28px rgba(0,0,0,.55),inset 0 1px 0 rgba(255,255,255,.4),inset 0 -2px 4px rgba(0,0,0,.4),0 0 0 1px rgba(0,0,0,.25)}
.race-cta:active{transform:translateX(-50%) translateY(0);filter:brightness(.95);box-shadow:0 4px 12px rgba(0,0,0,.5),inset 0 2px 5px rgba(0,0,0,.45)}
.race-cta .tri{width:0;height:0;border-style:solid;border-width:5px 0 5px 8px;border-color:transparent transparent transparent #FCEFF1;filter:drop-shadow(0 1px 0 rgba(0,0,0,.35))}
@keyframes ctaGlow{0%,100%{opacity:.35;transform:scale(.96)}50%{opacity:.9;transform:scale(1.06)}}
@media(prefers-reduced-motion:reduce){.race-cta::before{animation:none;opacity:.6;transform:scale(1)}}
.charts{padding:48px 44px;display:grid;grid-template-columns:1fr;gap:44px}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:36px}
.chart-section{background:linear-gradient(180deg,#100E0B 0%,var(--elev-1) 110px);border:1px solid var(--border);border-radius:6px;padding:28px 24px 18px;position:relative;transition:border-color .25s ease,box-shadow .25s ease}
.chart-section:hover{border-color:#2A2620;box-shadow:0 0 0 1px rgba(179,18,43,.07),0 18px 50px -30px rgba(0,0,0,.9)}
.scan{position:absolute;inset:0;border-radius:6px;pointer-events:none;overflow:hidden;z-index:1}
.scan::before,.scan::after{content:'';position:absolute;inset:7px;background:linear-gradient(currentColor,currentColor) 0 0/14px 1px,linear-gradient(currentColor,currentColor) 0 0/1px 14px,linear-gradient(currentColor,currentColor) 100% 0/14px 1px,linear-gradient(currentColor,currentColor) 100% 0/1px 14px,linear-gradient(currentColor,currentColor) 0 100%/14px 1px,linear-gradient(currentColor,currentColor) 0 100%/1px 14px,linear-gradient(currentColor,currentColor) 100% 100%/14px 1px,linear-gradient(currentColor,currentColor) 100% 100%/1px 14px;background-repeat:no-repeat}
.scan::before{color:#403A31}
.scan::after{color:var(--accent);opacity:0;transition:opacity .35s ease}
.chart-section:hover .scan::after{opacity:.9}
.chart-section:hover .scan{animation:scanSweep 1.5s cubic-bezier(.4,0,.2,1) 1}
@keyframes scanSweep{from{background:linear-gradient(180deg,transparent,rgba(236,229,213,.05) 50%,rgba(179,18,43,.04) 52%,transparent) no-repeat 0 -90px/100% 90px}to{background:linear-gradient(180deg,transparent,rgba(236,229,213,.05) 50%,rgba(179,18,43,.04) 52%,transparent) no-repeat 0 calc(100% + 90px)/100% 90px}}
.chart-section[data-section]::before{content:attr(data-section);position:absolute;top:-7px;left:18px;font-size:.50rem;font-weight:500;letter-spacing:.20em;color:var(--dim);background:var(--elev-1);padding:0 8px;font-family:var(--mono);text-transform:uppercase;z-index:2}
.chart-section[data-readout]::after{content:attr(data-readout);position:absolute;top:14px;right:18px;font-size:.48rem;font-weight:700;letter-spacing:.16em;color:var(--accent);font-family:var(--mono);text-transform:uppercase;z-index:2;opacity:.75}
.chart-label{font-size:.70rem;font-weight:600;letter-spacing:.18em;color:var(--text);text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:10px}
.chart-label::before{content:'';width:6px;height:6px;background:var(--accent);border-radius:1px;flex-shrink:0}
.telemetry-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;padding-top:16px}
.telem-panel{border:1px solid var(--border);border-radius:6px;padding:16px 18px;position:relative;background:var(--elev-2);transition:border-color .18s}
.telem-panel:hover{border-color:#2A2E30}
.telem-label{font-size:.56rem;font-weight:500;letter-spacing:.18em;color:var(--dim);text-transform:uppercase;margin-bottom:10px;font-family:var(--font)}
footer{padding:28px 44px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
.ft-left{font-size:.58rem;color:var(--dim);letter-spacing:.14em;text-transform:uppercase}
.ft-status{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:.56rem;letter-spacing:.18em;color:var(--accent);text-transform:uppercase}
.ft-status b{width:5px;height:5px;border-radius:50%;background:var(--accent)}
.ft-right{font-size:.58rem;color:var(--dim);letter-spacing:.08em;font-family:var(--mono)}
.ft-right span{color:var(--accent)}
@media(max-width:860px){.hero{grid-template-columns:1fr}.hero .cluster{border-right:none;border-bottom:1px solid var(--border)}.chart-row{grid-template-columns:1fr}.telemetry-row{grid-template-columns:1fr}.cluster-body{grid-template-columns:1fr;gap:28px}.gauges{justify-content:center;flex-wrap:wrap}.kpis{grid-template-columns:repeat(2,1fr)}}
@media(max-width:480px){h1{font-size:1.6rem}.kpis{grid-template-columns:1fr}.charts{padding:24px}.cluster{padding:24px}}
.logo-bar{display:flex;justify-content:center;padding:34px 0 18px}
.logo-img{height:135px;display:block;filter:grayscale(1) invert(1) brightness(1.05);opacity:.92}
#game-overlay{display:none;position:fixed;inset:0;z-index:9999;background:#000}
#game-overlay.active{display:grid;grid-template-rows:1fr auto}
#game-canvas{width:100%;height:100%;display:block;outline:none;min-height:0}
#hud{display:flex;flex-direction:column;padding:0;font-family:'Space Mono','Courier New',monospace;font-size:13px;color:#ECE5D5;border-top:1px solid rgba(214,32,63,.55);flex-shrink:0;position:relative;box-shadow:0 -1px 0 rgba(214,32,63,.25),0 -10px 30px rgba(0,0,0,.5)}
#hud-main{display:flex;align-items:center;gap:10px;padding:7px 14px;background:linear-gradient(180deg,rgba(14,13,10,.82),rgba(8,7,6,.93));backdrop-filter:blur(11px);-webkit-backdrop-filter:blur(11px);box-shadow:inset 0 1px 0 rgba(255,255,255,.06)}
#hud-sectors{display:flex;gap:16px;align-items:center;padding:3px 14px 4px;font-size:11px;background:linear-gradient(180deg,rgba(8,7,6,.9),rgba(8,7,6,.96));backdrop-filter:blur(11px);-webkit-backdrop-filter:blur(11px);border-top:1px solid rgba(214,32,63,.12)}
.hud-pos{color:#D6203F;font-weight:700;font-size:17px;min-width:28px;font-variant-numeric:tabular-nums;text-shadow:0 0 12px rgba(214,32,63,.45);transition:color .2s,text-shadow .2s}
.hud-lap{color:#7E858B;min-width:72px;font-variant-numeric:tabular-nums}
.hud-timer{color:#ECE5D5;min-width:80px;font-weight:700;font-variant-numeric:tabular-nums}
.hud-speed{color:#ECE5D5;min-width:70px;font-weight:700;font-variant-numeric:tabular-nums}
.hud-gear-wrap{display:flex;flex-direction:column;align-items:center;min-width:52px}
.hud-gear{font-size:27px;font-weight:900;color:#ECE5D5;line-height:1;font-variant-numeric:tabular-nums;text-shadow:0 0 14px rgba(214,32,63,.4);transition:color .05s}
.hud-gear.flash{color:#D6203F}
.hud-rpm-bar{width:52px;height:6px;background:#16140F;border-radius:3px;margin-top:3px;overflow:hidden;box-shadow:inset 0 0 3px rgba(0,0,0,.65)}
.hud-rpm-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#9DA3A8 0%,#C7A06A 55%,#D6203F 82%,#B3122B 100%);width:0%;transition:width .04s;box-shadow:0 0 7px rgba(214,32,63,.5)}
.hud-drs{color:#16140F;border:1px solid #16140F;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:700;letter-spacing:.12em;transition:color .15s,border-color .15s,text-shadow .15s}
.hud-drs.on{color:#C7A06A;border-color:#C7A06A;text-shadow:0 0 8px #C7A06A}
.hud-tires{display:flex;align-items:center;gap:4px}
.hud-tire-label{color:#7E858B;font-size:10px;min-width:8px}
.hud-tire-wrap{width:52px;height:8px;background:#16140F;border-radius:3px;overflow:hidden;border:1px solid #2A2E30}
.hud-tire-bar{height:100%;width:100%;border-radius:3px;transition:width .1s,background .3s}
.hud-tire-temp{width:10px;height:10px;border-radius:50%;background:#7E858B;transition:background .4s;border:1px solid rgba(255,255,255,0.25)}
.hud-tire-cmp{width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;color:#0a0a0a;border:1px solid rgba(255,255,255,0.4)}
.hud-tire-pct{color:#ECE5D5;font-size:11px;min-width:30px;font-weight:700;font-variant-numeric:tabular-nums}
.hud-box{color:#16140F;border:1px solid #16140F;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:800;letter-spacing:.12em}
.hud-box.advise{color:#D6203F;border-color:#D6203F;animation:boxPulse .7s ease-in-out infinite}
@keyframes boxPulse{0%,100%{opacity:.35}50%{opacity:1;text-shadow:0 0 9px #D6203F}}
@media(prefers-reduced-motion:reduce){.hud-box.advise{animation:none;opacity:1}}
.hud-s1,.hud-s2,.hud-s3{color:#7E858B;min-width:88px;font-size:11px;font-variant-numeric:tabular-nums}
.hud-s1.purple,.hud-s2.purple,.hud-s3.purple{color:#C7A06A;font-weight:700}
.hud-s1.green,.hud-s2.green,.hud-s3.green{color:#ECE5D5;font-weight:700}
.hud-s1.yellow,.hud-s2.yellow,.hud-s3.yellow{color:#9DA3A8}
.hud-delta{color:#7E858B;min-width:80px;font-size:11px;font-variant-numeric:tabular-nums}
.hud-delta.green{color:#C7A06A}
.hud-delta.red{color:#D6203F}
.hud-msg{flex:1;text-align:center;color:#16140F;font-size:10px;letter-spacing:.10em}
.hud-close{margin-left:auto;background:none;border:1px solid #D6203F;color:#D6203F;font-family:'Space Mono','Courier New',monospace;cursor:pointer;padding:3px 10px;font-size:12px;letter-spacing:.08em}
.hud-close:hover{background:#D6203F;color:#000}
#hud-minimap{position:absolute;bottom:8px;right:12px;border:1px solid rgba(214,32,63,.55);background:rgba(8,7,6,.6);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);border-radius:6px;pointer-events:none;box-shadow:0 8px 26px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.05)}
#hud-wheel{position:absolute;bottom:10px;right:182px;pointer-events:none}
.hud-drs.armed{color:#C7A06A;border-color:#C7A06A;text-shadow:0 0 8px #C7A06A}
#hud-standings{position:fixed;top:70px;left:12px;width:188px;background:linear-gradient(180deg,rgba(11,10,8,.78),rgba(8,7,6,.86));backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid rgba(214,32,63,.18);border-radius:7px;pointer-events:none;overflow:hidden;z-index:10001;box-shadow:0 0 0 1px rgba(214,32,63,.10),0 10px 30px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.05)}
#hud-standings .st-hd{display:flex;justify-content:space-between;align-items:center;padding:4px 8px;font-size:9px;letter-spacing:.22em;color:#D6203F;background:linear-gradient(90deg,rgba(214,32,63,.18),rgba(214,32,63,0));border-bottom:1px solid #2A2E30;text-transform:uppercase}
#hud-standings .st-hd b{color:#ECE5D5;font-weight:700;letter-spacing:.10em}
#hud-standings .st-row{display:flex;align-items:center;gap:6px;padding:1px 6px;font-size:10px;line-height:15px;letter-spacing:.04em;color:#7E858B;border-bottom:1px solid rgba(22,20,16,.5)}
#hud-standings .st-row.me{background:rgba(236,229,213,.14);color:#ECE5D5}
#hud-standings .st-pos{width:16px;color:#7E858B;text-align:right}
#hud-standings .st-chip{width:5px;height:10px;border-radius:1px;flex-shrink:0}
#hud-standings .st-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#hud-standings .st-gap{color:#7E858B;font-size:9px}
#hud-standings .st-pit{color:#C7A06A;font-weight:700;letter-spacing:.08em}
#hud-standings .st-dnf{color:#D6203F;font-weight:700;letter-spacing:.08em}
#podium-overlay{display:none;position:absolute;inset:0;z-index:10000;background:rgba(8,7,6,.93);flex-direction:column;align-items:center;justify-content:center;font-family:'Space Mono','Courier New',monospace;color:#ECE5D5}
#podium-overlay.active{display:flex}
#podium-overlay h2{font-size:1.4rem;letter-spacing:.25em;color:#D6203F;margin-bottom:24px;text-transform:uppercase}
#podium-table{border-collapse:collapse;font-size:.85rem}
#podium-table td{padding:6px 20px;border-bottom:1px solid #16140F}
#podium-table td:first-child{color:#7E858B;text-align:right}
#podium-table td:nth-child(2){color:#ECE5D5;font-weight:700}
#podium-table td:last-child{color:#7E858B;text-align:right}
.podium-p1 td{color:#C7A06A !important}
.podium-btn{margin-top:28px;background:none;border:1px solid #D6203F;color:#D6203F;font-family:'Space Mono','Courier New',monospace;cursor:pointer;padding:7px 22px;letter-spacing:.12em;font-size:.8rem}
.podium-btn:hover{background:#D6203F;color:#0B0A08}
#dnf-overlay{display:none;position:absolute;inset:0;z-index:10002;background:rgba(22,4,4,.58);flex-direction:column;align-items:center;justify-content:center;font-family:'Space Mono','Courier New',monospace;color:#ECE5D5;-webkit-backdrop-filter:blur(2px);backdrop-filter:blur(2px)}
#dnf-overlay.active{display:flex}
#dnf-overlay .dnf-tag{font-size:2.6rem;font-weight:800;letter-spacing:.4em;color:#D6203F;text-shadow:0 0 22px rgba(255,59,48,.7);margin-bottom:4px}
#dnf-overlay h2{font-size:1.0rem;letter-spacing:.34em;color:#fff;text-transform:uppercase;margin:0 0 14px}
#dnf-overlay #dnf-reason{color:#D98B8B;font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;max-width:72%;text-align:center;line-height:1.6}
#dnf-overlay .podium-btn{border-color:#D6203F;color:#D6203F}
#dnf-overlay .podium-btn:hover{background:#D6203F;color:#160404}
#pause-overlay{display:none;position:absolute;inset:0;z-index:10000;background:rgba(8,7,6,.85);flex-direction:column;align-items:center;justify-content:center;font-family:'Space Mono','Courier New',monospace;color:#ECE5D5}
#pause-overlay.active{display:flex}
#pause-overlay h2{font-size:1.8rem;letter-spacing:.35em;color:#D6203F;text-transform:uppercase;margin-bottom:14px}
#pause-overlay p{color:#7E858B;font-size:.72rem;letter-spacing:.22em;text-transform:uppercase}
#tire-select{display:none;position:absolute;inset:0;z-index:10001;background:rgba(6,5,4,.93);flex-direction:column;align-items:center;justify-content:center;font-family:'Space Mono','Courier New',monospace;color:#ECE5D5}
#tire-select.active{display:flex}
#ts-weather{display:flex;flex-direction:column;align-items:center;gap:6px;margin-bottom:10px}
#ts-wlabel{font-size:1.6rem;letter-spacing:.3em;font-weight:800;color:#D6203F;text-shadow:0 0 16px rgba(214,32,63,.5)}
#ts-wnote{font-size:.72rem;letter-spacing:.12em;color:#7E858B;max-width:540px;text-align:center}
#ts-title{font-size:.8rem;letter-spacing:.34em;color:#7E858B;margin:14px 0 18px;text-transform:uppercase}
#ts-cards{display:flex;gap:18px}
.tire-card{position:relative;width:190px;padding:20px 16px 18px;background:rgba(14,13,10,.85);border:1px solid #2A2E30;border-radius:10px;cursor:pointer;display:flex;flex-direction:column;align-items:center;text-align:center;transition:transform .14s,border-color .14s,box-shadow .14s}
.tire-card:hover{transform:translateY(-4px);border-color:#D6203F;box-shadow:0 10px 30px rgba(214,32,63,.22)}
.tc-disc{width:74px;height:74px;border-radius:50%;border:6px solid #222;display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:900;color:#0a0a0a;margin-bottom:12px}
.tc-name{font-size:1rem;letter-spacing:.18em;font-weight:800;margin-bottom:8px}
.tc-key{position:absolute;top:8px;left:10px;font-size:.7rem;color:#7E858B}
.tc-desc{font-size:.66rem;line-height:1.5;color:#7E858B;letter-spacing:.04em}
.tc-rec{margin-top:12px;font-size:.6rem;letter-spacing:.18em;color:#C7A06A;border:1px solid #C7A06A;border-radius:3px;padding:2px 8px;visibility:hidden}
.tire-card.rec .tc-rec{visibility:visible}
.tire-card.rec{border-color:#C7A06A}
#ts-hint{margin-top:22px;font-size:.66rem;letter-spacing:.2em;color:#7E858B;text-transform:uppercase}
#map-select{display:none;position:absolute;inset:0;z-index:10001;background:rgba(6,5,4,.93);flex-direction:column;align-items:center;justify-content:center;font-family:'Space Mono','Courier New',monospace;color:#ECE5D5}
#map-select.active{display:flex}
#ms-title{font-size:.8rem;letter-spacing:.34em;color:#7E858B;margin:0 0 18px;text-transform:uppercase}
#ms-cards{display:flex;gap:18px}
.map-card{position:relative;width:210px;padding:18px 14px 16px;background:rgba(14,13,10,.85);border:1px solid #2A2E30;border-radius:10px;cursor:pointer;display:flex;flex-direction:column;align-items:center;text-align:center;transition:transform .14s,border-color .14s,box-shadow .14s}
.map-card.sel{border-color:#22D3EE;box-shadow:0 10px 30px rgba(34,211,238,.2)}
.map-card:not(.locked):hover{transform:translateY(-4px);border-color:#22D3EE;box-shadow:0 10px 30px rgba(34,211,238,.28)}
.map-card.locked{cursor:default;opacity:.45;filter:grayscale(.85)}
.mc-key{position:absolute;top:8px;left:10px;font-size:.7rem;color:#7E858B}
.mc-thumb{width:178px;height:96px;border-radius:6px;margin-bottom:12px;display:flex;align-items:center;justify-content:center;font-size:2.1rem;font-weight:900;color:#ECE5D5;background:linear-gradient(160deg,#0b0614 0%,#1a0a2e 45%,#3d0a28 100%);text-shadow:0 0 14px #22D3EE,0 0 30px #e94fd8;border:1px solid #2A2E30}
.map-card.locked .mc-thumb{background:linear-gradient(160deg,#101013 0%,#1a1a20 100%);text-shadow:none;color:#5A5F66;font-size:1.4rem}
.mc-name{font-size:.92rem;letter-spacing:.16em;font-weight:800;margin-bottom:6px}
.mc-sub{font-size:.62rem;line-height:1.5;color:#7E858B;letter-spacing:.06em}
.mc-tag{margin-top:10px;font-size:.6rem;letter-spacing:.18em;border-radius:3px;padding:2px 8px;color:#C7A06A;border:1px solid #C7A06A}
.map-card.locked .mc-tag{color:#7E858B;border-color:#3A3E42}
#ms-hint{margin-top:22px;font-size:.66rem;letter-spacing:.2em;color:#7E858B;text-transform:uppercase}
#lights-bar{display:none;position:absolute;top:14px;left:50%;transform:translateX(-50%);z-index:20001;gap:10px;padding:10px 18px;background:rgba(8,7,6,.92);border:1px solid #3A0A0F;border-radius:8px;pointer-events:none}
#lights-bar.active{display:flex}
.light-bulb{width:30px;height:30px;border-radius:50%;background:#1a0000;border:2px solid #440000;transition:background .06s,box-shadow .06s}
.light-bulb.lit{background:#D6203F;border-color:#E03A52;box-shadow:0 0 14px #D6203F,0 0 32px #5A0A14}
#go-flash{display:none;position:absolute;inset:0;background:rgba(255,255,255,.88);z-index:20002;pointer-events:none}
#grid-msg{display:none;position:absolute;bottom:120px;left:50%;transform:translateX(-50%);z-index:20001;color:#ECE5D5;font-family:'Space Mono','Courier New',monospace;font-size:.72rem;letter-spacing:.28em;text-transform:uppercase;text-align:center;text-shadow:0 0 8px #D6203F;pointer-events:none}
#grid-msg.active{display:block}
body::after{content:'';position:fixed;inset:0;z-index:1;pointer-events:none;background:radial-gradient(ellipse at 50% 22%,transparent 62%,rgba(0,0,0,.45) 100%)}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
header{animation:fadeUp .55s ease both}
.reveal{opacity:0;transform:translateY(18px);transition:opacity .7s ease,transform .8s cubic-bezier(.16,1,.3,1)}
.reveal.in{opacity:1;transform:none}
@media(prefers-reduced-motion:reduce){.reveal{opacity:1;transform:none;transition:none}}
#f1car{position:relative;z-index:1}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style>
</head>
<body>
<div class="status-bar">
  <div class="brand-badge">
    <span class="brand-name">Vedra Research</span>
    <span class="brand-sub">Powered by Claude</span>
  </div>
  <div class="sb-dot"></div>
  <span><span class="sb-label">SYSTEM</span>&nbsp;<span class="sb-val">NOMINAL</span></span>
  <span class="sb-sep">&middot;</span>
  <span><span class="sb-label">MET</span>&nbsp;<span class="sb-val" id="met-clock">T+00:00:00</span></span>
  <span class="sb-sep">&middot;</span>
  <span><span class="sb-label">SEASONS</span>&nbsp;<span class="sb-val">PLACEHOLDER_SEASONS</span></span>
  <span class="sb-sep">&middot;</span>
  <span><span class="sb-label">ROUNDS</span>&nbsp;<span class="sb-val">PLACEHOLDER_ROUNDS</span></span>
  <span class="sb-sep">&middot;</span>
  <span><span class="sb-label">DRIVERS</span>&nbsp;<span class="sb-val">PLACEHOLDER_DRIVER_COUNT</span></span>
  <span class="sb-spacer"></span>
  <span><span class="sb-label">BUILD</span>&nbsp;<span class="sb-val">PLACEHOLDER_BUILD</span></span>
  <span class="sb-sep">&middot;</span>
  <span class="sb-rec"><b></b>&nbsp;LIVE</span>
</div>
<div class="logo-bar">PLACEHOLDER_LOGO</div>
<header>
  <span class="hd-ghost" aria-hidden="true">PLACEHOLDER_YEAR_RANGE</span>
  <div class="hd-team">Oracle Red Bull Racing</div>
  <h1>Red Bull <span class="accent">F1</span> Analytics</h1>
  <p class="sub">PLACEHOLDER_SUBTITLE</p>
</header>
<div class="hero">
<section class="cluster">
  <div class="cluster-hd"><span class="dot"></span>Vehicle Telemetry &middot; Performance Cluster<span class="ln"></span><span class="tag">PLACEHOLDER_CLUSTER_TAG</span></div>
  <div class="cluster-body">
    <div class="gauges">PLACEHOLDER_GAUGES</div>
    <div class="kpis">PLACEHOLDER_KPIS</div>
  </div>
</section>
<div class="car-viewer">
  <canvas id="f1car"></canvas>
</div>
</div>
<div id="game-overlay">
  <canvas id="game-canvas" tabindex="0"></canvas>
  <div id="hud">
    <div class="brand-badge in-game">
      <span class="brand-name">Vedra Research</span>
      <span class="brand-sub">Powered by Claude</span>
    </div>
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
        <span class="hud-tire-cmp" id="tire-cmp">M</span>
        <span class="hud-tire-label">F</span>
        <div class="hud-tire-wrap"><div class="hud-tire-bar" id="tire-f"></div></div>
        <div class="hud-tire-temp" id="temp-f"></div>
        <span class="hud-tire-label">R</span>
        <div class="hud-tire-wrap"><div class="hud-tire-bar" id="tire-r"></div></div>
        <div class="hud-tire-temp" id="temp-r"></div>
        <span class="hud-tire-pct" id="tire-pct">100%</span>
        <span class="hud-tire-label" style="min-width:auto;margin-left:6px">DMG</span>
        <div class="hud-tire-wrap"><div class="hud-tire-bar" id="dmg-bar" style="width:0%"></div></div>
        <span class="hud-tire-pct" id="dmg-pct">0%</span>
        <span class="hud-box" id="hud-box">BOX</span>
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
        <line x1="-18" y1="0" x2="18" y2="0" stroke="#D6203F" stroke-width="3" stroke-linecap="round"/>
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
  <div id="dnf-overlay">
    <div class="dnf-tag">DNF</div>
    <h2>Retired</h2>
    <p id="dnf-reason"></p>
    <button class="podium-btn" id="dnf-classify">SEE CLASSIFICATION</button>
  </div>
  <div id="pause-overlay">
    <h2>PAUSED</h2>
    <p>ESC TO RESUME &nbsp;&#xB7;&nbsp; Q TO QUIT</p>
  </div>
  <div id="tire-select">
    <div id="ts-weather"><span id="ts-wlabel">DRY</span><span id="ts-wnote"></span></div>
    <h3 id="ts-title">SELECT YOUR TIRES</h3>
    <div id="ts-cards"></div>
    <p id="ts-hint">CLICK A COMPOUND &nbsp;&#xB7;&nbsp; OR PRESS 1 / 2 / 3</p>
  </div>
  <div id="map-select">
    <h3 id="ms-title">SELECT CIRCUIT</h3>
    <div id="ms-cards"></div>
    <p id="ms-hint">CLICK TOKYO &nbsp;&#xB7;&nbsp; OR PRESS 1 / ENTER</p>
  </div>
</div>
<div class="ticker"><div class="ticker-track">PLACEHOLDER_TICKER</div></div>
<div class="charts">
  <div class="chart-section" data-section="SYS&middot;01" data-readout="&#9679; LIVE"><div class="scan"></div>
    <div class="chart-label">Championship &middot; Points Trajectory</div>
    PLACEHOLDER_C1
  </div>
  <div class="chart-row">
    <div class="chart-section" data-section="SYS&middot;02" data-readout="&#9679; LIVE"><div class="scan"></div>
      <div class="chart-label">Finish Positions &middot; Season</div>
      PLACEHOLDER_C2
    </div>
    <div class="chart-section" data-section="SYS&middot;03" data-readout="&#9679; &#916;"><div class="scan"></div>
      <div class="chart-label">Driver Delta &middot; Points Gap</div>
      PLACEHOLDER_C3
    </div>
  </div>
  <div class="chart-section" data-section="SYS&middot;04" data-readout="&#9679; MATRIX"><div class="scan"></div>
    <div class="chart-label">Performance Matrix &middot; All Seasons</div>
    PLACEHOLDER_C4
  </div>
  <div class="chart-section" data-section="SYS&middot;05" data-readout="&#9679; PACE"><div class="scan"></div>
    <div class="chart-label">Pace &middot; Grid vs Finish</div>
    PLACEHOLDER_C5
  </div>
  <div class="chart-section" data-section="SYS&middot;06" data-readout="&#9679; TELEMETRY"><div class="scan"></div>
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
  <div class="chart-section" data-section="SYS&middot;07" data-readout="&#9679; FASTF1"><div class="scan"></div>
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
  <div class="ft-status"><b></b>All Systems Nominal &middot; Data Integrity Verified</div>
  <div class="ft-right">Generated <span>PLACEHOLDER_TS</span> &nbsp;&middot;&nbsp; Oracle Red Bull Racing</div>
</footer>
<script>
window.F1FX=(function(){
  if(typeof THREE==='undefined') return null;
  var CACHE={},_carbon=null,_carbonN=null,_paintN=null,_rubber=null;
  function cvs(w,h){var c=document.createElement('canvas');c.width=w;c.height=h||w;return c;}
  function hex(n){return '#'+('000000'+((n>>>0)&0xffffff).toString(16)).slice(-6);}
  function rgbOf(n){return {r:(n>>16)&255,g:(n>>8)&255,b:n&255};}
  function lerp(a,b,t){return a+(b-a)*t;}
  function shade(n,f){var c=rgbOf(n),k=function(v){return Math.max(0,Math.min(255,v*f|0));};return 'rgb('+k(c.r)+','+k(c.g)+','+k(c.b)+')';}
  function inkOn(n){var c=rgbOf(n);return (0.299*c.r+0.587*c.g+0.114*c.b)>140?'#0a0a0d':'#f4f4f6';}

  function carbon(){
    if(_carbon) return _carbon;
    var s=256,c=cvs(s),x=c.getContext('2d'),cell=10;
    x.fillStyle='#0b0b0e';x.fillRect(0,0,s,s);
    for(var yy=0;yy<s;yy+=cell)for(var xx=0;xx<s;xx+=cell){
      var over=((((xx/cell)|0)+((yy/cell)|0))&1)===0,g=x.createLinearGradient(xx,yy,xx+cell,yy+cell);
      if(over){g.addColorStop(0,'#2c2c34');g.addColorStop(0.5,'#141418');g.addColorStop(1,'#0a0a0d');}
      else{g.addColorStop(0,'#0a0a0d');g.addColorStop(0.5,'#141418');g.addColorStop(1,'#2c2c34');}
      x.fillStyle=g;x.fillRect(xx,yy,cell,cell);
    }
    x.globalAlpha=0.05;for(var i=0;i<1600;i++){x.fillStyle=(i&1)?'#ffffff':'#000000';x.fillRect(Math.random()*s,Math.random()*s,1,1);}x.globalAlpha=1;
    var t=new THREE.CanvasTexture(c);t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(4,4);if('anisotropy' in t)t.anisotropy=4;
    _carbon=t;return t;
  }
  function carbonNormal(){
    if(_carbonN) return _carbonN;
    var s=256,c=cvs(s),x=c.getContext('2d'),cell=10;
    x.fillStyle='#8080ff';x.fillRect(0,0,s,s);
    for(var yy=0;yy<s;yy+=cell)for(var xx=0;xx<s;xx+=cell){
      var over=((((xx/cell)|0)+((yy/cell)|0))&1)===0;
      var g=over?x.createLinearGradient(xx,yy,xx+cell,yy):x.createLinearGradient(xx,yy,xx,yy+cell);
      g.addColorStop(0,'#9aa0ff');g.addColorStop(0.5,'#8080ff');g.addColorStop(1,'#5e64ff');
      x.fillStyle=g;x.fillRect(xx,yy,cell,cell);
    }
    var t=new THREE.CanvasTexture(c);t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(4,4);
    _carbonN=t;return t;
  }
  function paintNormal(){
    if(_paintN) return _paintN;
    var s=256,c=cvs(s),x=c.getContext('2d');
    x.fillStyle='#8080ff';x.fillRect(0,0,s,s);
    for(var i=0;i<16000;i++){var v=120+(Math.random()*18|0);x.fillStyle='rgba('+v+','+v+',255,0.5)';x.fillRect(Math.random()*s,Math.random()*s,1,1);}
    var t=new THREE.CanvasTexture(c);t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(8,8);
    _paintN=t;return t;
  }

  function liveryFlank(bodyHex,accentHex){
    var k='F'+bodyHex+'_'+accentHex;if(CACHE[k]) return CACHE[k];
    // Sponsor flank: individual brand marks (colours, weights, badge chips) drawn over the
    // body paint. 2048px wide so the lettering stays crisp in the studio orbit and chase cam.
    var W=2048,H=512,c=cvs(W,H),x=c.getContext('2d'),acc=hex(accentHex),ink=inkOn(bodyHex);
    x.clearRect(0,0,W,H);
    function rr(x0,y0,w2,h2,r2){x.beginPath();x.moveTo(x0+r2,y0);x.arcTo(x0+w2,y0,x0+w2,y0+h2,r2);x.arcTo(x0+w2,y0+h2,x0,y0+h2,r2);x.arcTo(x0,y0+h2,x0,y0,r2);x.arcTo(x0,y0,x0+w2,y0,r2);x.closePath();}
    function spaced(txt,x0,y0,ls){var sx=x0;for(var i=0;i<txt.length;i++){x.fillText(txt[i],sx,y0);sx+=x.measureText(txt[i]).width+ls;}return sx-ls;}
    x.textBaseline='middle';x.textAlign='left';
    // accent speed swoosh with a bright machined top edge
    x.fillStyle=acc;x.beginPath();x.moveTo(0,H*0.64);x.bezierCurveTo(W*0.30,H*0.585,W*0.70,H*0.515,W,H*0.49);x.lineTo(W,H*0.80);x.bezierCurveTo(W*0.70,H*0.825,W*0.30,H*0.885,0,H*0.93);x.closePath();x.fill();
    x.strokeStyle='rgba(255,255,255,0.8)';x.lineWidth=5;x.beginPath();x.moveTo(0,H*0.64);x.bezierCurveTo(W*0.30,H*0.585,W*0.70,H*0.515,W,H*0.49);x.stroke();
    x.fillStyle='rgba(10,10,14,0.92)';x.fillRect(0,H*0.93,W,H*0.07);
    // wordmarks riding the swoosh
    x.fillStyle='#ffffff';x.font='italic 900 '+(H*0.155|0)+'px Arial,Helvetica,sans-serif';
    x.fillText('Red Bull Racing',W*0.045,H*0.775);
    x.font='900 '+(H*0.115|0)+'px Arial,Helvetica,sans-serif';x.fillText('BYBIT',W*0.60,H*0.685);
    x.font='italic 900 '+(H*0.115|0)+'px Arial,Helvetica,sans-serif';x.fillText('VISA',W*0.82,H*0.66);
    // headline ORACLE wordmark, letterspaced, with accent rule
    x.fillStyle=ink;x.font='900 '+(H*0.21|0)+'px Arial,Helvetica,sans-serif';
    var oEnd=spaced('ORACLE',W*0.035,H*0.255,H*0.045);
    x.fillStyle=acc;x.fillRect(W*0.035,H*0.405,oEnd-W*0.035,H*0.022);
    // partner badge chips: white plates keep the small marks readable, like real decals
    var chips=[
      function(){x.textAlign='left';x.fillStyle='#0058A8';x.font='italic 900 70px Arial,Helvetica,sans-serif';var mw=x.measureText('Mobil').width;x.fillText('Mobil',-(mw+58)*0.5,4);x.fillStyle='#EE3124';x.font='900 104px Arial,Helvetica,sans-serif';x.fillText('1',-(mw+58)*0.5+mw+12,-2);},
      function(){x.textAlign='center';x.fillStyle='#CC0000';x.font='900 62px Arial,Helvetica,sans-serif';x.fillText('HONDA',0,16);x.fillRect(-104,-52,208,10);},
      function(){x.textAlign='center';x.fillStyle='#E20613';rr(-138,-38,82,72,10);x.fill();x.fillStyle='#ffffff';x.font='900 42px Arial,Helvetica,sans-serif';x.fillText('TAG',-97,-1);x.fillStyle='#16161c';x.font='900 54px Arial,Helvetica,sans-serif';x.textAlign='left';x.fillText('HEUER',-40,-1);},
      function(){x.textAlign='center';x.fillStyle='#16161c';x.font='900 64px Arial,Helvetica,sans-serif';x.fillText('PIRELLI',0,-8);x.fillStyle='#FCD700';x.fillRect(-118,36,236,14);}
    ];
    for(var ci=0;ci<4;ci++){
      var cx0=W*0.355+ci*W*0.162,cw2=W*0.150,cy0=H*0.07,ch2=H*0.345;
      x.save();
      x.fillStyle='rgba(244,244,247,0.97)';rr(cx0,cy0,cw2,ch2,18);x.fill();
      x.strokeStyle='rgba(0,0,0,0.25)';x.lineWidth=3;rr(cx0,cy0,cw2,ch2,18);x.stroke();
      x.translate(cx0+cw2*0.5,cy0+ch2*0.5);chips[ci]();
      x.restore();
    }
    var t=new THREE.CanvasTexture(c);if('anisotropy' in t)t.anisotropy=8;CACHE[k]=t;return t;
  }
  function roundel(num,bgHex,fgHex){
    var k='R'+num+'_'+bgHex+'_'+fgHex;if(CACHE[k]) return CACHE[k];
    var s=512,c=cvs(s),x=c.getContext('2d');x.clearRect(0,0,s,s);
    x.fillStyle='rgba(0,0,0,0.35)';x.beginPath();x.arc(s/2,s/2+s*0.012,s*0.465,0,6.283);x.fill();
    x.fillStyle=hex(bgHex);x.beginPath();x.arc(s/2,s/2,s*0.46,0,6.283);x.fill();
    var gl=x.createLinearGradient(0,s*0.06,0,s*0.96);gl.addColorStop(0,'rgba(255,255,255,0.28)');gl.addColorStop(0.45,'rgba(255,255,255,0.04)');gl.addColorStop(1,'rgba(0,0,0,0.26)');
    x.fillStyle=gl;x.beginPath();x.arc(s/2,s/2,s*0.46,0,6.283);x.fill();
    x.lineWidth=s*0.045;x.strokeStyle=hex(fgHex);x.beginPath();x.arc(s/2,s/2,s*0.432,0,6.283);x.stroke();
    x.lineWidth=s*0.012;x.strokeStyle='rgba(0,0,0,0.45)';x.beginPath();x.arc(s/2,s/2,s*0.462,0,6.283);x.stroke();
    x.fillStyle=hex(fgHex);x.font='italic 900 '+(s*0.56|0)+'px Arial,Helvetica,sans-serif';x.textAlign='center';x.textBaseline='middle';
    x.shadowColor='rgba(0,0,0,0.5)';x.shadowBlur=s*0.02;x.shadowOffsetY=s*0.008;
    x.fillText(''+num,s/2,s*0.55);
    x.shadowColor='transparent';x.shadowBlur=0;x.shadowOffsetY=0;
    var t=new THREE.CanvasTexture(c);if('anisotropy' in t)t.anisotropy=8;CACHE[k]=t;return t;
  }
  function liveryFin(bodyHex,accentHex){
    var k='N'+bodyHex+'_'+accentHex;if(CACHE[k]) return CACHE[k];
    // Transparent ORACLE wordmark with an accent rule — decaled onto the shark fin and the
    // rear-wing endplates.
    var W=1024,H=256,c=cvs(W,H),x=c.getContext('2d'),acc=hex(accentHex),ink=inkOn(bodyHex);
    x.clearRect(0,0,W,H);
    x.fillStyle=ink;x.font='900 150px Arial,Helvetica,sans-serif';x.textBaseline='middle';x.textAlign='left';
    var sx=30,txt='ORACLE';
    for(var i=0;i<txt.length;i++){x.fillText(txt[i],sx,H*0.42);sx+=x.measureText(txt[i]).width+22;}
    x.fillStyle=acc;x.fillRect(30,H*0.78,sx-52,14);
    var t=new THREE.CanvasTexture(c);if('anisotropy' in t)t.anisotropy=8;CACHE[k]=t;return t;
  }
  function tyre(colHex){
    var k='T'+colHex;if(CACHE[k]) return CACHE[k];
    // Pirelli-style sidewall at 1024px: compound-colour PIRELLI wordmarks + P ZERO marks
    // (real C-compound styling — the colour lives in the lettering, not a painted hoop),
    // mould rings, radial striations and a moulding barcode.
    var s=1024,c=cvs(s),x=c.getContext('2d'),cx=s/2,col=hex(colHex);
    x.fillStyle='#0b0b0e';x.beginPath();x.arc(cx,cx,s*0.5,0,6.283);x.fill();
    var rg=x.createRadialGradient(cx,cx,s*0.28,cx,cx,s*0.5);
    rg.addColorStop(0,'rgba(56,56,62,0.32)');rg.addColorStop(0.7,'rgba(22,22,26,0.3)');rg.addColorStop(1,'rgba(4,4,7,0.78)');
    x.fillStyle=rg;x.beginPath();x.arc(cx,cx,s*0.5,0,6.283);x.fill();
    x.strokeStyle='rgba(255,255,255,0.025)';x.lineWidth=1.5;
    for(var st=0;st<150;st++){var a0=(st+0.13)/150*6.283;x.beginPath();x.moveTo(cx+Math.cos(a0)*s*0.385,cx+Math.sin(a0)*s*0.385);x.lineTo(cx+Math.cos(a0)*s*0.495,cx+Math.sin(a0)*s*0.495);x.stroke();}
    x.strokeStyle='rgba(0,0,0,0.55)';x.lineWidth=3;
    [0.388,0.452,0.490].forEach(function(rr2){x.beginPath();x.arc(cx,cx,s*rr2,0,6.283);x.stroke();});
    x.strokeStyle='rgba(150,150,158,0.10)';x.lineWidth=4;x.beginPath();x.arc(cx,cx,s*0.468,0,6.283);x.stroke();
    var R=s*0.434;
    function side(txt,a,font,fill){x.save();x.translate(cx+Math.cos(a)*R,cx+Math.sin(a)*R);x.rotate(a+1.5708);x.font=font;x.fillStyle=fill;x.textAlign='center';x.textBaseline='middle';x.fillText(txt,0,0);x.restore();}
    for(var i=0;i<4;i++){
      side('PIRELLI',(i/4)*6.283,'italic 900 '+(s*0.072|0)+'px Arial,Helvetica,sans-serif',col);
      side('P ZERO',(i/4+0.125)*6.283,'700 '+(s*0.038|0)+'px Arial,Helvetica,sans-serif','#d8d8de');
    }
    x.save();x.translate(cx,cx);x.rotate(0.32);x.fillStyle='rgba(235,235,240,0.85)';
    for(var bc=0;bc<14;bc++){x.fillRect(R-14,-58+bc*3.4,((bc*7)%3)?2:5,2.2);}
    x.restore();
    x.globalCompositeOperation='destination-out';x.beginPath();x.arc(cx,cx,s*0.375,0,6.283);x.fill();x.globalCompositeOperation='source-over';
    var t=new THREE.CanvasTexture(c);if('anisotropy' in t)t.anisotropy=8;CACHE[k]=t;return t;
  }
  function rubber(){
    if(_rubber) return _rubber;
    // Tiled rubber grain for the tyre lathe: colour + bump. u wraps the circumference and v
    // runs across the profile, so the horizontal bands become mould rings around the tread.
    var s=256,c=cvs(s),x=c.getContext('2d');
    x.fillStyle='#141418';x.fillRect(0,0,s,s);
    for(var i=0;i<8000;i++){var v=12+(Math.random()*26|0);x.fillStyle='rgba('+v+','+v+','+(v+4)+',0.6)';x.fillRect(Math.random()*s,Math.random()*s,1.6,1.6);}
    x.fillStyle='rgba(0,0,0,0.4)';x.fillRect(0,s*0.44,s,3);x.fillRect(0,s*0.56,s,3);
    x.fillStyle='rgba(255,255,255,0.06)';x.fillRect(0,s*0.50,s,1.5);
    var t=new THREE.CanvasTexture(c);t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(10,1);
    _rubber=t;return t;
  }
  function rimFace(accentHex){
    var k='W'+accentHex;if(CACHE[k]) return CACHE[k];
    // Baked dished 10-twin-spoke wheel face: deep AO between blades, top-lit machined lip,
    // team-accent pinstripe, centre hub with drive pegs.
    var s=512,c=cvs(s),x=c.getContext('2d'),cx=s/2,acc=hex(accentHex);
    var dish=x.createRadialGradient(cx,cx,s*0.05,cx,cx,s*0.5);
    dish.addColorStop(0,'#22242a');dish.addColorStop(0.5,'#0c0d10');dish.addColorStop(0.88,'#121318');dish.addColorStop(1,'#34373f');
    x.fillStyle=dish;x.fillRect(0,0,s,s);
    for(var i=0;i<10;i++){
      x.save();x.translate(cx,cx);x.rotate(i/10*6.283);
      [-1,1].forEach(function(sg){
        var g2=x.createLinearGradient(0,0,0,sg*26);
        g2.addColorStop(0,'#41454f');g2.addColorStop(0.55,'#23252c');g2.addColorStop(1,'#0d0e12');
        x.fillStyle=g2;x.beginPath();
        x.moveTo(s*0.115,sg*4);x.lineTo(s*0.435,sg*15);x.lineTo(s*0.435,sg*4.5);x.lineTo(s*0.115,sg*1.5);x.closePath();x.fill();
        x.strokeStyle='rgba(205,215,232,0.30)';x.lineWidth=1.6;
        x.beginPath();x.moveTo(s*0.118,sg*4);x.lineTo(s*0.432,sg*15);x.stroke();
      });
      x.restore();
    }
    var lip=x.createLinearGradient(0,0,0,s);lip.addColorStop(0,'#4a4e58');lip.addColorStop(0.5,'#23252b');lip.addColorStop(1,'#15161a');
    x.strokeStyle=lip;x.lineWidth=s*0.052;x.beginPath();x.arc(cx,cx,s*0.468,0,6.283);x.stroke();
    x.globalAlpha=0.95;x.strokeStyle=acc;x.lineWidth=s*0.016;x.beginPath();x.arc(cx,cx,s*0.428,0,6.283);x.stroke();x.globalAlpha=1;
    var hub=x.createRadialGradient(cx,cx,2,cx,cx,s*0.13);hub.addColorStop(0,'#3a3d45');hub.addColorStop(0.7,'#1c1e23');hub.addColorStop(1,'#101115');
    x.fillStyle=hub;x.beginPath();x.arc(cx,cx,s*0.125,0,6.283);x.fill();
    x.strokeStyle='rgba(180,190,205,0.25)';x.lineWidth=2;x.beginPath();x.arc(cx,cx,s*0.122,0,6.283);x.stroke();
    for(var p=0;p<6;p++){var pa=p/6*6.283+0.5;x.fillStyle='#0a0a0d';x.beginPath();x.arc(cx+Math.cos(pa)*s*0.085,cx+Math.sin(pa)*s*0.085,s*0.016,0,6.283);x.fill();}
    x.strokeStyle='rgba(255,255,255,0.05)';x.lineWidth=s*0.07;
    x.beginPath();x.arc(cx,cx,s*0.30,5.6,0.6);x.stroke();
    x.beginPath();x.arc(cx,cx,s*0.30,2.5,3.6);x.stroke();
    var t=new THREE.CanvasTexture(c);if('anisotropy' in t)t.anisotropy=8;
    CACHE[k]=t;return t;
  }

  function roundedBox(w,h,d,r,seg){
    r=Math.min(r,w/2-0.001,h/2-0.001);seg=seg||1;
    var sh=new THREE.Shape(),hw=w/2,hh=h/2;
    sh.moveTo(-hw+r,-hh);sh.lineTo(hw-r,-hh);sh.quadraticCurveTo(hw,-hh,hw,-hh+r);
    sh.lineTo(hw,hh-r);sh.quadraticCurveTo(hw,hh,hw-r,hh);
    sh.lineTo(-hw+r,hh);sh.quadraticCurveTo(-hw,hh,-hw,hh-r);
    sh.lineTo(-hw,-hh+r);sh.quadraticCurveTo(-hw,-hh,-hw+r,-hh);
    var bev=Math.min(r*0.6,d*0.45);
    var geo=new THREE.ExtrudeGeometry(sh,{depth:d-bev*2,bevelEnabled:true,bevelThickness:bev,bevelSize:bev,bevelSegments:seg,steps:1});
    geo.translate(0,0,-(d-bev*2)/2);geo.computeVertexNormals();
    return geo;
  }

  // Single shared F1 chassis builder used by BOTH the studio hero render and the in-game
  // cars (player/AI/safety). All the realistic bodywork lives here once; game-only FX
  // (brake-glow sprites, contact shadow) are layered on top by the game's buildCar wrapper.
  // Realism extras (sidepod intakes, beam wing, diffuser fins, brake discs, deeper paint)
  // are gated behind opts.detailed so the player+studio get the full treatment while AI
  // cars keep the lighter build until that flag is flipped.
  function buildChassis(opts){
    opts=opts||{};
    var bodyColor=(opts.bodyColor==null)?0x1A3E82:opts.bodyColor;
    var accentColor=(opts.accentColor==null)?0xCC1E1E:opts.accentColor;
    // Optional second livery accent (Red Bull yellow). Defaults to the primary accent so
    // non-RB cars just get accent-coloured trim rather than a stray yellow pinstripe.
    var accent2=(opts.accent2==null)?accentColor:opts.accent2;
    var carNum=opts.carNum;
    var envMap=opts.envMap||null;
    var detailed=!!opts.detailed;
    var tyreCol=(opts.tyreCol==null)?0xE10600:opts.tyreCol;
    var g=new THREE.Group();
    var PI=Math.PI,pN=paintNormal();
    // Satin painted-carbon body: rough-ish pigment base under a glossy clearcoat,
    // matching how real car paint reads.
    var mNav=new THREE.MeshPhysicalMaterial({color:bodyColor,metalness:0.0,roughness:0.50,clearcoat:1.0,clearcoatRoughness:detailed?0.04:0.06,normalMap:pN,normalScale:new THREE.Vector2(0.12,0.12)});
    var mRed=new THREE.MeshPhysicalMaterial({color:accentColor,metalness:0.0,roughness:0.40,clearcoat:1.0,clearcoatRoughness:detailed?0.04:0.05,normalMap:pN,normalScale:new THREE.Vector2(0.10,0.10)});
    var mYel=new THREE.MeshPhysicalMaterial({color:accent2,metalness:0.0,roughness:0.42,clearcoat:1.0,clearcoatRoughness:0.06,normalMap:pN,normalScale:new THREE.Vector2(0.10,0.10)});
    var mGold=new THREE.MeshPhysicalMaterial({color:0xC9A85C,metalness:0.84,roughness:0.12,clearcoat:0.55,clearcoatRoughness:0.14});
    // Dark exposed carbon for wings/floor/halo — base darkened so reflections sit over carbon,
    // not white plastic; stronger weave normal so the fibre reads up close.
    var mC=new THREE.MeshPhysicalMaterial({color:0x17181b,map:carbon(),normalMap:carbonNormal(),normalScale:new THREE.Vector2(0.7,0.7),metalness:0.30,roughness:0.50,clearcoat:0.6,clearcoatRoughness:0.22});
    var mFloor=new THREE.MeshPhysicalMaterial({color:0x111113,map:carbon(),normalMap:carbonNormal(),normalScale:new THREE.Vector2(0.65,0.65),metalness:0.28,roughness:0.56,clearcoat:0.22,clearcoatRoughness:0.32});
    var mFl=detailed?mFloor:mC;
    var rubT=rubber();
    var mT=new THREE.MeshStandardMaterial({color:0xCFCFD4,map:rubT,bumpMap:rubT,bumpScale:0.0035,metalness:0.0,roughness:0.90});
    var mR=new THREE.MeshPhysicalMaterial({color:0xBBBBBB,metalness:0.96,roughness:0.03,clearcoat:0.3});
    var mG=new THREE.MeshStandardMaterial({color:0x888888,metalness:0.74,roughness:0.28});
    var mRim=new THREE.MeshPhysicalMaterial({color:0x202024,metalness:0.92,roughness:0.34,clearcoat:0.4});
    var mRimAcc=new THREE.MeshStandardMaterial({color:accentColor,metalness:0.55,roughness:0.40});
    var faceT=rimFace(accentColor);
    var mFace=new THREE.MeshStandardMaterial({map:faceT,bumpMap:faceT,bumpScale:0.0045,metalness:0.72,roughness:0.36});
    var mDisc=new THREE.MeshStandardMaterial({color:0x2c2e33,metalness:0.55,roughness:0.55});
    var mCal=new THREE.MeshStandardMaterial({color:accentColor,metalness:0.35,roughness:0.5});
    var mIntake=new THREE.MeshStandardMaterial({color:0x050507,metalness:0.2,roughness:0.88});
    if(envMap){[mNav,mRed,mYel,mGold,mC,mFloor,mR,mG,mRim,mRimAcc,mFace].forEach(function(m){m.envMap=envMap;m.envMapIntensity=(m===mRed?(detailed?1.55:1.5):(m.metalness>0.6?1.1:0.9));});}
    // Tyre sidewall lip: matte compound colour (no emissive glow), recoloured by setTyre.
    var mPir=new THREE.MeshStandardMaterial({color:tyreCol,roughness:0.82,metalness:0.0});
    function mk(geo,mat,x,y,z,rx,ry,rz){var m=new THREE.Mesh(geo,mat);m.position.set(x||0,y||0,z||0);m.rotation.set(rx||0,ry||0,rz||0);return m;}
    function bx(w,h,d,mat,x,y,z,rx,ry,rz){return mk(new THREE.BoxGeometry(w,h,d),mat,x,y,z,rx,ry,rz);}
    function cy(r1,r2,h,s,mat,x,y,z,rx,ry,rz){return mk(new THREE.CylinderGeometry(r1,r2,h,s),mat,x,y,z,rx,ry,rz);}
    function wing(span,chord,thick,mat,x,y,z,ryRot){var sh=new THREE.Shape(),t=thick*0.5;sh.moveTo(0,0);sh.bezierCurveTo(chord*0.1,t,chord*0.4,t,chord,0);sh.bezierCurveTo(chord*0.4,-t,chord*0.1,-t,0,0);var geo=new THREE.ExtrudeGeometry(sh,{depth:span,bevelEnabled:false,steps:1});geo.translate(0,0,-span*0.5);var m=new THREE.Mesh(geo,mat);m.position.set(x||0,y||0,z||0);m.rotation.y=ryRot||0;return m;}
    function tube3(ax,ay,az,bx2,by2,bz2,cx2,cy2,cz2,r,mat){var crv=new THREE.QuadraticBezierCurve3(new THREE.Vector3(ax,ay,az),new THREE.Vector3(bx2,by2,bz2),new THREE.Vector3(cx2,cy2,cz2));return new THREE.Mesh(new THREE.TubeGeometry(crv,20,r,8,false),mat);}
    function bar(ax,ay,az,ex,ey,ez,r,mat){var dx=ex-ax,dy=ey-ay,dz=ez-az,len=Math.sqrt(dx*dx+dy*dy+dz*dz);var m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,len,6),mat);m.position.set((ax+ex)/2,(ay+ey)/2,(az+ez)/2);var q=new THREE.Quaternion();q.setFromUnitVectors(new THREE.Vector3(0,1,0),new THREE.Vector3(dx/len,dy/len,dz/len));m.setRotationFromQuaternion(q);return m;}
    function add(m){g.add(m);}
    function rbx(w,h,d,r,mat,x,y,z,rx,ry,rz){return mk(roundedBox(w,h,d,r),mat,x,y,z,rx,ry,rz);}
    var decals=new THREE.Group();g.userData.decals=decals;
    function decal(tex,w,h,x,y,z,rx,ry,rz){var m=new THREE.Mesh(new THREE.PlaneGeometry(w,h),new THREE.MeshStandardMaterial({map:tex,transparent:true,roughness:0.4,metalness:0.0,polygonOffset:true,polygonOffsetFactor:-2,polygonOffsetUnits:-2}));m.position.set(x||0,y||0,z||0);m.rotation.set(rx||0,ry||0,rz||0);m.userData.isDecal=true;decals.add(m);return m;}
    // Superellipse cross-section loft. p<1 squares off the section (flat top deck + flatter
    // sides + defined shoulder line) so the body reads as a real slab-sided monocoque, not a
    // round tube. p defaults to 1 (plain ellipse).
    function loftBody(secs,mat,rad,p){rad=rad||18;p=p||1;function se(t){return (t<0?-1:1)*Math.pow(Math.abs(t),p);}var n=secs.length,pos=[],uv=[],idx=[],i,j;for(i=0;i<n;i++){var s=secs[i];for(j=0;j<=rad;j++){var a=j/rad*PI*2;pos.push(s[0],s[3]+se(Math.cos(a))*s[2],se(Math.sin(a))*s[1]);uv.push(j/rad,i/(n-1));}}for(i=0;i<n-1;i++)for(j=0;j<rad;j++){var aa=i*(rad+1)+j,bb=aa+rad+1;idx.push(aa,bb,aa+1,bb,bb+1,aa+1);}var geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));geo.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));geo.setIndex(idx);geo.computeVertexNormals();return new THREE.Mesh(geo,mat);}
    // Slab-sided superellipse hull (halfW > halfH, flat top deck) with a pronounced coke-bottle
    // waist that necks hard ahead of the rear wheels.
    // Section = [x, halfWidth, halfHeight, yCentre]; p=0.62 squares off the cross-section.
    add(loftBody([
      [ 1.62,0.13,0.10,0.00],[ 1.34,0.23,0.15,0.01],[ 1.02,0.34,0.20,0.02],
      [ 0.64,0.44,0.24,0.02],[ 0.30,0.50,0.23,0.02],[-0.05,0.50,0.22,0.02],
      [-0.40,0.42,0.20,0.03],[-0.74,0.28,0.18,0.05],[-1.06,0.21,0.17,0.06],
      [-1.40,0.16,0.14,0.06],[-1.74,0.10,0.09,0.06]
    ],mNav,26,0.62));
    add(cy(0.205,0.225,0.05,22,mC,0.32,0.305,0));
    // Slim blade nose: a rounded-rectangle section (taller-than-wide chisel) that tapers and
    // drops in Y from the chassis front down toward the front-wing mount — the modern F1 look.
    // Same superellipse squaring (p=0.62).
    add(loftBody([
      [1.62,0.13,0.105,0.000],[1.95,0.115,0.105,-0.005],[2.30,0.095,0.095,-0.018],
      [2.62,0.072,0.082,-0.035],[2.88,0.050,0.064,-0.052],[3.02,0.030,0.044,-0.066]
    ],mNav,22,0.62));
    add(cy(0.022,0.022,0.06,8,mRed,3.10,-0.05,0,0,0,-PI/2));
    var fw=new THREE.Group();function addFW(m){fw.add(m);}
    addFW(wing(2.12,0.30,0.040,mC,2.86,-0.245,0,0));addFW(wing(1.94,0.24,0.036,mC,2.66,-0.200,0,0));addFW(wing(1.74,0.18,0.030,mC,2.48,-0.158,0,0));
    [-1.03,1.03].forEach(function(z){
      addFW(rbx(0.52,0.34,0.034,0.04,mC,2.66,-0.10,z));      // carbon endplate (taller, thinner, sharper)
      addFW(rbx(0.05,0.34,0.045,0.02,mRed,2.905,-0.10,z));   // thin red leading-edge strip
      addFW(rbx(0.52,0.028,0.05,0.012,mYel,2.66,0.055,z));   // yellow top pinstripe
      addFW(rbx(0.20,0.10,0.05,0.04,mC,2.84,-0.30,z));       // carbon footplate
    });
    [-0.20,0.20].forEach(function(z){addFW(rbx(0.06,0.24,0.04,0.02,mC,2.72,-0.075,z));});
    if(carNum!=null){[0.205,-0.205].forEach(function(nz){decal(roundel(carNum,accentColor,0xffffff),0.34,0.34,2.02,0.12,nz,0,nz>0?0:PI,0);});}
    add(fw);g.userData.fw=fw;
    [-0.53,0.53].forEach(function(z){
      var sg=z>0?1:-1;
      // Ground-effect sidepod: a tall, narrow inlet mouth at the front whose top surface ramps
      // down and inward (downwash) into the coke-bottle, with the section squared off (p=0.5)
      // for the flat-topped, undercut look.
      var pod=loftBody([
        [ 0.68,0.090,0.190,0.015],[ 0.48,0.190,0.205,0.000],[ 0.12,0.225,0.175,-0.015],
        [-0.26,0.205,0.135,-0.030],[-0.56,0.150,0.092,-0.040],[-0.74,0.085,0.055,-0.040]
      ],mNav,22,0.5);
      pod.position.z=sg*0.62;add(pod);
      add(cy(0.105,0.135,0.09,16,mC,0.70,0.02,sg*0.62,0,0,PI/2));
      add(bx(1.32,0.09,0.17,mC,-0.08,-0.17,z*0.72,0,0,sg*0.20));
      add(rbx(0.84,0.30,0.012,0.04,mRed,-0.14,0.00,z+sg*0.172));
      add(bx(0.09,0.20,0.08,mC,0.42,0.07,z));add(rbx(1.18,0.07,0.24,0.03,mFl,-0.40,-0.23,z));
      decal(liveryFlank(bodyColor,accentColor),1.10,0.30,0.02,0.04,z*1.58,0,z>0?0:PI,0);
      if(detailed){var intake=new THREE.Mesh(new THREE.CylinderGeometry(0.080,0.150,0.07,18),mIntake);intake.scale.set(1.0,1,1.10);intake.rotation.z=PI/2;intake.position.set(0.685,0.02,sg*0.62);add(intake);}
    });
    for(var bi=0;bi<3;bi++){[-0.34-bi*0.09,0.34+bi*0.09].forEach(function(z){add(bx(0.09,0.20,0.03,mC,0.84,-0.04,z));});}
    var ecPts=[new THREE.Vector2(0,0),new THREE.Vector2(0.085,0.10),new THREE.Vector2(0.195,0.26),new THREE.Vector2(0.260,0.35),new THREE.Vector2(0.240,0.25),new THREE.Vector2(0.185,0.10),new THREE.Vector2(0,0)];
    var ecMesh=new THREE.Mesh(new THREE.LatheGeometry(ecPts,20),mNav);ecMesh.rotation.z=PI/2;ecMesh.scale.set(1,0.054,1);ecMesh.position.set(-0.20,0.18,0);add(ecMesh);
    add(bx(0.70,0.040,0.050,mGold,-0.22,0.485,0,0,0,0.07));
    // Shark-fin engine cover: a thin tall carbon fin on the centreline sweeping back from behind
    // the airbox down to the rear wing — the signature modern-F1 spine.
    var finSh=new THREE.Shape();
    finSh.moveTo(0.05,0.26);
    finSh.lineTo(0.05,0.52);
    finSh.bezierCurveTo(-0.55,0.52,-1.15,0.46,-1.55,0.40);
    finSh.lineTo(-1.86,0.245);
    finSh.lineTo(-1.86,0.185);
    finSh.lineTo(-1.0,0.185);
    finSh.lineTo(0.05,0.26);
    var finGeo=new THREE.ExtrudeGeometry(finSh,{depth:0.020,bevelEnabled:false});finGeo.translate(0,0,-0.010);
    add(new THREE.Mesh(finGeo,mC));
    // ORACLE on both faces of the shark fin, tilted (rz) to follow the fin's sloping top edge.
    [-1,1].forEach(function(sg){decal(liveryFin(0x111111,accentColor),0.72,0.18,-0.86,0.335,sg*0.014,0,sg>0?0:PI,sg*0.05);});
    [-0.30,0.30].forEach(function(z){decal(liveryFlank(bodyColor,accentColor),1.30,0.28,-0.25,0.13,z,0,z>0?0:PI,0);});
    var abx=new THREE.Mesh(new THREE.CylinderGeometry(0.11,0.16,0.46,3),mNav);abx.rotation.z=PI/2;abx.rotation.x=PI;abx.position.set(0.28,0.46,0);add(abx);
    add(cy(0.085,0.085,0.05,16,mC,0.50,0.49,0,0,0,PI/2));
    // Halo: matte-black carbon, real teardrop hoop. Front aero blade (haloPost) rises from the
    // cockpit rim to the hoop apex; two side rails bow out around the driver's head and converge
    // to low rear mounts; short struts foot the mounts to the chassis deck. All grouped under
    // haloGrp so cockpit mode can hide the whole hoop out of the driver's eyeline; haloPost is
    // kept as a single mesh reference because other code grabs it via userData.haloPost.
    var haloGrp=new THREE.Group();
    var haloPost=bx(0.085,0.345,0.050,mC,0.665,0.495,0,0,0,0.19);haloGrp.add(haloPost);g.userData.haloPost=haloPost;
    [-1,1].forEach(function(sg){haloGrp.add(tube3(0.64,0.66,0, 0.34,0.64,sg*0.34, -0.02,0.54,sg*0.20, 0.030,mC));});
    haloGrp.add(bar(-0.02,0.54,-0.20,0.06,0.30,-0.20,0.022,mC));haloGrp.add(bar(-0.02,0.54,0.20,0.06,0.30,0.20,0.022,mC));
    add(haloGrp);g.userData.halo=haloGrp;
    [-0.26,0.26].forEach(function(mz){
      var sg=mz>0?1:-1;
      add(bar(0.40,0.30,mz*0.92,0.53,0.44,mz,0.012,mC));
      var hous=mk(new THREE.BoxGeometry(0.085,0.090,0.150),mNav,0.535,0.455,mz);hous.rotation.y=sg*0.20;add(hous);
      add(mk(new THREE.BoxGeometry(0.020,0.094,0.154),mRed,0.560,0.455,mz,0,sg*0.20,0));
      var face=mk(new THREE.PlaneGeometry(0.066,0.090),mR,0.498,0.455,mz);face.rotation.y=-PI/2+sg*0.22;add(face);
    });
    var helm=mk(new THREE.SphereGeometry(0.14,24,18),mNav,0.41,0.36,0);helm.scale.set(1.2,0.92,1.1);g.userData.helmet=helm;add(helm);
    add(rbx(3.12,0.042,1.80,0.05,mFl,-0.08,-0.21,0));
    // Floor-edge fences + front splitter: the raised carbon floor edge and forward lip that read
    // as ground-effect aero rather than a flat slab.
    [-0.86,0.86].forEach(function(z){add(rbx(2.60,0.090,0.022,0.02,mFl,-0.15,-0.172,z));});
    add(rbx(0.34,0.046,1.24,0.03,mFl,1.55,-0.205,0));
    for(var ds=-3;ds<=3;ds++){add(bx(0.64,0.13,0.03,mFl,-1.83,-0.15,ds*0.245));}
    add(rbx(0.42,0.030,1.60,0.03,mFl,-1.830,-0.146,0,0,0,2.719));
    if(detailed){for(var dfi=-2;dfi<=2;dfi++){add(bx(0.36,0.13,0.022,mFloor,-1.92,-0.13,dfi*0.155,0,0,0.20));}}
    if(detailed){
      // Bargeboards / turning vanes ahead of the sidepods (angled carbon fins).
      [-0.46,0.46].forEach(function(z){var sg=z>0?1:-1;
        add(rbx(0.34,0.22,0.010,0.02,mC,0.88,-0.03,z,0,sg*0.34,0));
        add(rbx(0.22,0.15,0.010,0.02,mC,0.97,-0.06,z*0.82,0,sg*0.30,0));
      });
      // Suspension pushrods: thin diagonal struts from the upright up to the chassis.
      [[1.72,0.80,1.40,0.20],[-1.52,0.88,-1.20,0.18]].forEach(function(s){
        [1,-1].forEach(function(sgz){add(bar(s[0],-0.10,sgz*s[1],s[2],0.16,sgz*s[3],0.013,mG));});
      });
    }
    add(bx(0.33,0.23,0.42,mNav,-1.71,0.07,0));add(bx(0.54,0.065,0.90,mC,-1.85,0.19,0));
    add(bar(-1.62,0.185,-0.13,-2.00,0.661,-0.18,0.018,mC));add(bar(-1.62,0.185,0.13,-2.00,0.661,0.18,0.018,mC));
    var rw=new THREE.Group();function addRW(m){rw.add(m);}
    addRW(wing(1.60,0.27,0.058,mC,-2.03,0.69,0,0));addRW(wing(1.48,0.18,0.046,mC,-1.87,0.63,0,0));
    [-0.80,0.80].forEach(function(z){
      addRW(rbx(0.31,0.60,0.030,0.04,mC,-1.97,0.41,z));      // carbon endplate (thinner)
      addRW(rbx(0.05,0.60,0.040,0.02,mRed,-2.10,0.41,z));    // thin red trailing-edge strip
      addRW(rbx(0.31,0.030,0.045,0.012,mYel,-1.97,0.69,z));  // yellow top pinstripe
    });
    if(detailed){addRW(wing(1.30,0.17,0.034,mC,-2.00,0.20,0,0));}
    add(rw);g.userData.rw=rw;
    // small ORACLE marks on the rear-wing endplate outer faces — reparented into rw so they
    // follow the wing when damage tilts it (decal() parks them in the decals group otherwise)
    [-0.80,0.80].forEach(function(z){var so=z>0?1:-1;rw.add(decal(liveryFin(0x111111,accentColor),0.26,0.065,-1.97,0.55,z+so*0.017,0,z>0?0:PI,0));});
    // FIA rain light: a vertical LED-matrix panel (white-hot centres bleeding to neon red)
    // drawn once and used as map+emissiveMap so individual LEDs resolve up close. emissive
    // stays white so the loop drives brightness purely via emissiveIntensity.
    var ledCv=document.createElement('canvas');ledCv.width=40;ledCv.height=80;var ledG=ledCv.getContext('2d');
    ledG.fillStyle='#08080b';ledG.fillRect(0,0,40,80);
    for(var lr=0;lr<7;lr++)for(var lc=0;lc<3;lc++){var ldx=8+lc*12,ldy=9+lr*10.4;
      var lg=ledG.createRadialGradient(ldx,ldy,0.5,ldx,ldy,4.4);
      lg.addColorStop(0,'#ffe6ee');lg.addColorStop(0.4,'#ff2540');lg.addColorStop(1,'rgba(255,20,60,0)');
      ledG.fillStyle=lg;ledG.beginPath();ledG.arc(ldx,ldy,4.4,0,6.283);ledG.fill();}
    var ledTex=new THREE.CanvasTexture(ledCv);
    var mTail=new THREE.MeshStandardMaterial({color:0xffffff,map:ledTex,emissiveMap:ledTex,emissive:0xffffff,emissiveIntensity:0.7,roughness:0.35});
    // Endplate brake lights: slim full-height neon blades on the wing trailing edges.
    var mTail2=new THREE.MeshStandardMaterial({color:0x30050c,emissive:0xff2348,emissiveIntensity:0.7,roughness:0.4});
    var mDrs=new THREE.MeshStandardMaterial({color:0x0a0a0a,emissive:0x00ff66,emissiveIntensity:0.0,roughness:0.5});
    var mHead=new THREE.MeshStandardMaterial({color:0x222222,emissive:0xfff2dc,emissiveIntensity:0.45,roughness:0.5});
    add(bx(0.075,0.26,0.145,mTail,-2.07,0.02,0));
    [-0.80,0.80].forEach(function(z){add(bx(0.030,0.56,0.046,mTail2,-2.125,0.41,z));});
    add(bx(0.04,0.05,0.16,mDrs,-2.07,0.58,0));
    [-0.12,0.12].forEach(function(z){add(bx(0.04,0.05,0.06,mHead,2.90,-0.05,z));});
    g.userData.tailMat=mTail;g.userData.tailMat2=mTail2;g.userData.drsMat=mDrs;g.userData.bodyMat=mNav;
    [[1.72,-0.80],[1.72,0.80]].forEach(function(w){var wx=w[0],wz=w[1],ci=wz>0?0.22:-0.22;add(bar(wx,0.00,wz,1.52,0.08,ci,0.016,mG));add(bar(wx,0.00,wz,1.18,0.06,ci,0.016,mG));add(bar(wx,-0.28,wz,1.50,-0.22,ci,0.016,mG));add(bar(wx,-0.28,wz,1.16,-0.22,ci,0.016,mG));});
    [[-1.52,-0.88],[-1.52,0.88]].forEach(function(w){var wx=w[0],wz=w[1],ci=wz>0?0.24:-0.24;add(bar(wx,0.00,wz,-1.10,0.06,ci,0.016,mG));add(bar(wx,0.00,wz,-1.42,0.04,ci,0.016,mG));add(bar(wx,-0.28,wz,-1.12,-0.20,ci,0.016,mG));add(bar(wx,-0.28,wz,-1.44,-0.18,ci,0.016,mG));});
    var mSide=new THREE.MeshStandardMaterial({map:tyre(tyreCol),transparent:true,roughness:0.9,metalness:0.0,polygonOffset:true,polygonOffsetFactor:-2,polygonOffsetUnits:-2});
    g.userData.setTyre=function(c){mSide.map=tyre(c);mSide.needsUpdate=true;mPir.color.setHex(c);};
    // Wheels: the rotating parts live in an inner `spin` group (exposed via userData.wheels
    // so the game can roll them with speed) and the fronts sit inside a steer group
    // (userData.steerW). The rim barrel stops short of the face so its end caps never hide
    // it; the visible face is a dished, bump-mapped 10-spoke rimFace disc, recessed inside
    // the tyre shoulder, with a hex centre-lock nut in team accent.
    function addWheel(x,z,tw,steerable){
      var wg=new THREE.Group(),spin=new THREE.Group();
      var fs=(z>0)?1:-1,fY=fs*tw*0.46;
      var R=0.340,ri=0.260,hw=tw*0.50;
      var tp=[new THREE.Vector2(ri,hw+0.004),new THREE.Vector2(ri+0.022,hw-0.002),new THREE.Vector2(R-0.030,hw-0.004),new THREE.Vector2(R-0.006,hw-0.026),new THREE.Vector2(R,hw-0.056),new THREE.Vector2(R,0),new THREE.Vector2(R,-(hw-0.056)),new THREE.Vector2(R-0.006,-(hw-0.026)),new THREE.Vector2(R-0.030,-(hw-0.004)),new THREE.Vector2(ri+0.022,-(hw-0.002)),new THREE.Vector2(ri,-(hw+0.004))];
      spin.add(mk(new THREE.LatheGeometry(tp,52),mT,0,0,0));
      spin.add(mk(new THREE.CylinderGeometry(ri-0.002,ri-0.002,tw*0.84,44),mRim,0,0,0));
      spin.add(mk(new THREE.CylinderGeometry(0.065,0.065,tw+0.032,16),mG,0,0,0));
      spin.add(mk(new THREE.CylinderGeometry(R-0.008,R-0.008,0.05,40,1,true),mPir,0,fY*0.55,0));
      spin.add(mk(new THREE.CircleGeometry(ri+0.002,48),mFace,0,fY-fs*0.006,0,-fs*PI/2,0,0));
      spin.add(mk(new THREE.CylinderGeometry(0.042,0.046,0.034,6),mRimAcc,0,fY,0));
      spin.add(mk(new THREE.CircleGeometry(R-0.004,36),mSide,0,fs*(hw-0.002),0,-fs*PI/2,0,0));
      if(detailed){wg.add(mk(new THREE.CylinderGeometry(ri-0.045,ri-0.045,0.026,28),mDisc,0,0,0));wg.add(mk(new THREE.BoxGeometry(0.05,0.11,0.05),mCal,ri*0.55,0,0));}
      wg.add(spin);wg.rotation.x=PI/2;
      wheelSpins.push(spin);
      if(steerable){var sgr=new THREE.Group();sgr.position.set(x,-0.22,z);sgr.add(wg);g.add(sgr);wheelSteers.push(sgr);}
      else{wg.position.set(x,-0.22,z);g.add(wg);}
    }
    var wheelSpins=[],wheelSteers=[];
    addWheel(1.72,-0.80,0.300,true);addWheel(1.72,0.80,0.300,true);
    addWheel(-1.52,-0.88,0.405);addWheel(-1.52,0.88,0.405);
    g.userData.wheels=wheelSpins;g.userData.steerW=wheelSteers;
    add(decals);
    g.traverse(function(o){if(o.isMesh){o.castShadow=!o.userData.isDecal;o.receiveShadow=!o.userData.isDecal;}});
    return g;
  }

  return {carbon:carbon,carbonNormal:carbonNormal,paintNormal:paintNormal,
          liveryFlank:liveryFlank,liveryFin:liveryFin,roundel:roundel,tyre:tyre,roundedBox:roundedBox,
          buildChassis:buildChassis,
          hex:hex,shade:shade,inkOn:inkOn};
})();
</script>
<script>
(function(){
  var c=document.getElementById('f1car');
  if(!c||typeof THREE==='undefined') return;
  var H=480,W=c.parentElement.offsetWidth||900;
  c.width=W; c.height=H;

  var renderer=new THREE.WebGLRenderer({canvas:c,antialias:true,alpha:true});
  renderer.setSize(W,H); renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.outputEncoding=THREE.sRGBEncoding;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure=1.42;
  renderer.physicallyCorrectLights=true;

  var scene=new THREE.Scene();
  var cam=new THREE.PerspectiveCamera(30,W/H,0.1,100);
  cam.position.set(7.71,2.50,6.43); cam.lookAt(0,-0.04,0);

  (function(){
    var pmrem=new THREE.PMREMGenerator(renderer);
    pmrem.compileEquirectangularShader();
    var envMat=new THREE.ShaderMaterial({
      side:THREE.BackSide,
      vertexShader:'varying vec3 vN;void main(){vN=normalize(position);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
      fragmentShader:'varying vec3 vN;void main(){vec3 n=normalize(vN);vec3 col=mix(vec3(0.04,0.045,0.06),vec3(0.42,0.47,0.56),smoothstep(-0.5,0.85,n.y));float box=smoothstep(0.58,0.99,n.y);col+=box*vec3(1.05,1.06,1.12)*0.92;float k=pow(max(0.,dot(n,normalize(vec3(0.50,0.78,0.35)))),10.);col+=k*vec3(1.00,0.95,0.86)*2.8;float f=pow(max(0.,dot(n,normalize(vec3(-0.72,0.18,-0.48)))),5.);col+=f*vec3(0.42,0.52,0.74)*0.70;float r=pow(max(0.,dot(n,normalize(vec3(-0.22,0.28,-0.94)))),7.);col+=r*vec3(0.55,0.66,1.00)*0.50;float fl=pow(max(0.,dot(n,vec3(0.,-1.,0.))),2.);col+=fl*vec3(0.10,0.10,0.12)*0.30;gl_FragColor=vec4(col,1.);}'
    });
    var envMesh=new THREE.Mesh(new THREE.SphereGeometry(50,32,16),envMat);
    var es=new THREE.Scene(); es.add(envMesh);
    var cubeRT=new THREE.WebGLCubeRenderTarget(512,{format:THREE.RGBFormat,generateMipmaps:true,minFilter:THREE.LinearMipmapLinearFilter});
    var cc=new THREE.CubeCamera(1,100,cubeRT);
    es.add(cc); cc.update(renderer,es);
    scene.environment=pmrem.fromCubemap(cubeRT.texture).texture;
    pmrem.dispose();
  })();

  scene.add(new THREE.HemisphereLight(0xe2ecff,0x101015,0.46));
  var s1=new THREE.DirectionalLight(0xffffff,2.55);
  s1.position.set(6,12,5); scene.add(s1);
  var s2=new THREE.DirectionalLight(0x9fb4e0,0.50); s2.position.set(-5,3,-3); scene.add(s2);
  var s3=new THREE.DirectionalLight(0x6f8cff,0.40); s3.position.set(-2,2,-8); scene.add(s3);
  var s4=new THREE.DirectionalLight(0xbcd0ff,0.55); s4.position.set(1,4,-7); scene.add(s4);

  var PI=Math.PI;
  // True Red Bull palette: near-black race navy base, red + yellow accents.
  var FX=window.F1FX,BODY=0x0A1530,ACCENT=0xD7202E,YEL=0xFBC900;
  // Studio hero render shares the exact in-game chassis (F1FX.buildChassis) at full
  // detail; passing no envMap means it reflects scene.environment. buildChassis already
  // runs the cast/receive-shadow traverse, so we don't repeat it here.
  var car=FX.buildChassis({bodyColor:BODY,accentColor:ACCENT,accent2:YEL,carNum:1,detailed:true});
  // Studio-only: the hero sits under a bright softbox env. Keep metals reflective but hold the
  // painted/carbon shells back so the softbox reads as deep lacquer, not blown-out plastic.
  car.traverse(function(o){if(o.isMesh&&o.material&&'envMapIntensity' in o.material){o.material.envMapIntensity=(o.material===car.userData.bodyMat)?0.72:(o.material.metalness>0.6?1.25:1.18);}});
  scene.add(car); car.rotation.y=PI/6;

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

var overlay=document.getElementById('game-overlay');
var gameCanvas=document.getElementById('game-canvas');
var podiumOvl=document.getElementById('podium-overlay');
var podiumTable=document.getElementById('podium-table');
var dnfOvl=document.getElementById('dnf-overlay');
var dnfReasonEl=document.getElementById('dnf-reason');
var dnfClassify=document.getElementById('dnf-classify');
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
var tireCmp=document.getElementById('tire-cmp');
var tirePct=document.getElementById('tire-pct');
var dmgBar=document.getElementById('dmg-bar');
var dmgPct=document.getElementById('dmg-pct');
var hudBox=document.getElementById('hud-box');
var hudS1=document.getElementById('hud-s1');
var hudS2=document.getElementById('hud-s2');
var hudS3=document.getElementById('hud-s3');
var minimapCanvas=document.getElementById('hud-minimap');
var minimapCtx=minimapCanvas?minimapCanvas.getContext('2d'):null;
var hudStandings=document.getElementById('hud-standings');
var wheelInd=document.getElementById('hud-wheel-ind');
var cwSteer=0,cwRev=0,cwPrevGear=0;
var cockpitWheel=null,cwSpin=null,cwLeds=null,cwDisp=null;
var hudClose=document.querySelector('.hud-close');
var podiumClose=document.getElementById('podium-close');

var SEGS=900,TW=16.8,CURB=2.2,BH=3.2;
var MAX_SPD=30,ENG=12000,BRK=18000,DRAG=0.35,CAR_MASS=800;
var GRIP_LIMIT=4.0,GRIP_SCRUB=3.2;
var RIDE_H=0.59,LAPS=5;
var AI_NUMS=[22,16,44,63,12,4,81,14,18,10,7,23,55,30,6,27,5,31,87];
var AI_COLORS=[
  0x0A1530,0xCC0000,0xCC0000,0xC0C0C0,0xC0C0C0,
  0xFF5500,0xFF5500,0x1B5E38,0x1B5E38,0x0090FF,
  0x0090FF,0x005AFF,0x005AFF,0x1934DB,0x1934DB,
  0x52E252,0x52E252,0xDDDDDD,0xDDDDDD
];
var AI_ACCENTS=[
  0xD7202E,0xFFD700,0xFFD700,0x00D2BE,0x00D2BE,
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
var TEAM_TIER={RBR:1.09,MCL:1.09,FER:1.08,MER:1.07,AMR:1.01,
               WIL:0.99,RB:0.99,ALP:0.98,SAU:0.96,HAS:0.96};

var TEAM_DRIVERS={RBR:['VERSTAPPEN']};
for(var _di=0;_di<AI_TEAMS.length;_di++){var _tid=AI_TEAMS[_di];(TEAM_DRIVERS[_tid]=TEAM_DRIVERS[_tid]||[]).push(AI_NAMES[_di+1]);}

var COMPOUNDS=[
  {id:'S',name:'SOFT',  col:0xe1342f, grip:1.07, wear:1.8, warm:0.65, desc:'Most grip, fastest lap — fades fast. Short stints.'},
  {id:'M',name:'MEDIUM',col:0xf2c037, grip:1.00, wear:1.0, warm:0.48, desc:'Balanced pace and life. The all-rounder.'},
  {id:'H',name:'HARD',  col:0xe8e8e8, grip:0.94, wear:0.6, warm:0.32, desc:'Least grip, longest life — go long, stop less.'}
];
var WEATHERS=[
  {id:'DRY',  label:'DRY',   gripMul:1.00, wearMul:1.00, tempBias:0.00, best:1, note:'Balanced track — the Medium is the safe call.'},
  {id:'HOT',  label:'HOT',   gripMul:0.97, wearMul:1.45, tempBias:0.22, best:2, note:'Track is roasting — tires overheat and wear fast. Go Hard.'},
  {id:'WINDY',label:'WINDY', gripMul:0.97, wearMul:1.10, tempBias:-0.04,best:1, note:'Gusts unsettle the car — keep it tidy on a Medium.'},
  {id:'WET',  label:'RAIN',  gripMul:0.74, wearMul:0.65, tempBias:-0.20,best:0, note:'Low grip, slow wear — Soft warms quickest in the cold.'}
];
var weather=WEATHERS[0];
function tempGrip(c){
  var warm=COMPOUNDS[c.compound].warm;
  return 0.82+0.18*Math.min(1,((c.tireTempF+c.tireTempR)*0.5)/Math.max(0.2,warm));
}

var trackPts=[
  new THREE.Vector3(-300, 0,  40),
  new THREE.Vector3(-120, 0,  30),
  new THREE.Vector3( 120, 0,  32),
  new THREE.Vector3( 260, 2,  60),
  new THREE.Vector3( 330, 5, 130),
  new THREE.Vector3( 285, 8, 185),
  new THREE.Vector3( 340,11, 240),
  new THREE.Vector3( 280,13, 300),
  new THREE.Vector3( 335,14, 355),
  new THREE.Vector3( 270,13, 425),
  new THREE.Vector3( 140,11, 460),
  new THREE.Vector3(  40,10, 450),
  new THREE.Vector3( -30,10, 405),
  new THREE.Vector3( -10, 9, 345),
  new THREE.Vector3(-110, 8, 330),
  new THREE.Vector3(-240, 6, 330),
  new THREE.Vector3(-330, 4, 270),
  new THREE.Vector3(-360, 2, 180),
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

var PIT_Z=58,PIT_W=9,PIT_X0=-250,PIT_X1=70,GARAGE_Z=74,PIT_LIMIT=8.3,PIT_STOP_DUR=2.7;
var REPAIR_TIME=2.2;
var PIT_BOX_Z=64,CREW_STOW_Z=72;var PIT_BOX_OFF=PIT_BOX_Z-PIT_Z;
var PIT_TEAMS=[
  {id:'RBR',col:0x1A3E82,acc:0xCC1E1E,num:1},
  {id:'FER',col:0xCC0000,acc:0xFFD700,num:16},
  {id:'MER',col:0xC0C0C0,acc:0x00D2BE,num:63},
  {id:'MCL',col:0xFF5500,acc:0x0D1F8C,num:4},
  {id:'AMR',col:0x1B5E38,acc:0xC5A028,num:14},
  {id:'ALP',col:0x0090FF,acc:0xFF87BC,num:10},
  {id:'WIL',col:0x005AFF,acc:0xE8002D,num:23},
  {id:'RB', col:0x1934DB,acc:0xCC1E1E,num:30},
  {id:'SAU',col:0x52E252,acc:0xEEEEEE,num:27},
  {id:'HAS',col:0xDDDDDD,acc:0xCC1E1E,num:31}
];
var BOX_SP=(PIT_X1-PIT_X0)/PIT_TEAMS.length;
var BOX_X=PIT_TEAMS.map(function(c,i){return PIT_X0+BOX_SP*0.5+i*BOX_SP;});
var TEAM_IDX={};PIT_TEAMS.forEach(function(c,i){TEAM_IDX[c.id]=i;});

// Pit path: a clean, monotonic-in-x corridor running alongside the S/F straight. The ends sit on
// the racing line so cars merge on/off without a kink; the lateral move onto the lane is eased in
// by an err-decay in pitMove (no teleport). The entry diverges late and steep — clear of the final
// corner exit, off the racing surface by x~-268 — so it reads as a deliberate entry road rather
// than a long shallow merge overlapping the track. Start must sit at x<=-295 (earliest P-key
// commit) so pitSFromX never snaps a committing car forward.
var pitCtrl=[
  new THREE.Vector3(-310,0,43),
  new THREE.Vector3(-288,0,48),
  new THREE.Vector3(-268,0,53.5),
  new THREE.Vector3(PIT_X0,0,PIT_Z),
  new THREE.Vector3(-90,0,PIT_Z),
  new THREE.Vector3(PIT_X1,0,PIT_Z),
  new THREE.Vector3(98,0,53.0),
  new THREE.Vector3(124,0,47.5),
  new THREE.Vector3(150,0,44.0),
  new THREE.Vector3(168,0,44.5)
];
var pitCurve=new THREE.CatmullRomCurve3(pitCtrl,false,'centripetal');
var PIT_SEGS=260;
var pitPts=pitCurve.getSpacedPoints(PIT_SEGS);
var PIT_LEN=pitCurve.getLength();
function pitPointAt(s){
  s=Math.max(0,Math.min(1,s));
  var f=s*PIT_SEGS,i0=Math.floor(f),i1=Math.min(PIT_SEGS,i0+1),fr=f-i0;
  var a=pitPts[i0],b=pitPts[i1];
  var dx=b.x-a.x,dz=b.z-a.z,len=Math.sqrt(dx*dx+dz*dz)||1;
  return {x:a.x+dx*fr,z:a.z+dz*fr,heading:Math.atan2(dx/len,dz/len)};
}
// Param along the path whose x best matches the given x (the path is monotonic in x), so a car can
// enter the lane at its current position instead of teleporting to s=0.
function pitSFromX(x){
  var bi=0,bd=1e9;
  for(var i=0;i<=PIT_SEGS;i++){var d=Math.abs(pitPts[i].x-x);if(d<bd){bd=d;bi=i;}}
  return bi/PIT_SEGS;
}
var BOX_S=BOX_X.map(function(bx){
  var bi=0,bd=1e9;
  for(var i=0;i<=PIT_SEGS;i++){var d=Math.abs(pitPts[i].x-bx)+Math.abs(pitPts[i].z-PIT_Z);if(d<bd){bd=d;bi=i;}}
  return bi/PIT_SEGS;
});
var teamCrew={},pitClock=0,idleFolk=[];
function inPit(c){return c.pitState&&c.pitState!=='NONE';}
function boxXOf(c){var i=TEAM_IDX[c.team];return i==null?BOX_X[0]:BOX_X[i];}
function boxSOf(c){var i=TEAM_IDX[c.team];return i==null?BOX_S[0]:BOX_S[i];}

function pitMove(c,dt,isPlayer){
  c.pitLift=c.pitLift||0;
  if(c.pitState==='LANE'||c.pitState==='EXIT'){
    var lim=PIT_LIMIT,bx=boxXOf(c),boxS=boxSOf(c);
    if(c.pitState==='LANE'){
      var dToBox=bx-c.x;
      if(dToBox<20) lim=Math.min(lim,Math.max(1.2,dToBox*0.85));
      if(isPlayer&&c.brk) lim*=0.4;
    } else if(c.x>=PIT_X1){
      lim=MAX_SPD;
    }
    var sRate=(c.pitState==='LANE'&&(bx-c.x)>=20&&c.speed>lim)?1.6:3;
    c.speed+=(lim-c.speed)*Math.min(dt*sRate,1);if(c.speed<0)c.speed=0;
    c.pitS+=(c.speed*dt)/PIT_LEN;
    var pp=pitPointAt(c.pitS);
    var perr=Math.max(0,1-dt*3);
    c._pitErrZ=(c._pitErrZ||0)*perr;c._pitErrH=(c._pitErrH||0)*perr;
    c.x=pp.x;c.z=pp.z+c._pitErrZ;c.y=0;c.heading=pp.heading+c._pitErrH;
    c.tIdx=closestWP(c.x,35,c.tIdx);
    if(c.pitState==='LANE'&&c.pitS>=boxS){
      var bp=pitPointAt(boxS);c._laneX=bp.x;c._boxHead=c.heading;
      c.pitState='ENTER';c._boxT=0;
      if(isPlayer) radio('Box, box — peel into the box, crew is ready.',false);
    }
    if(c.pitState==='EXIT'&&c.pitS>=0.999){
      c.pitState='NONE';c.pitLift=0;
      c.tIdx=closestWP(c.x,40,c.tIdx);
      var _ed=wpDir(c.tIdx);c.heading=Math.atan2(_ed.x,_ed.z);
      if(isPlayer){P._pitExitGrace=0.8;radio('Out clean, P'+(getPos()+1)+'. Hammer it.',false);}
      else {c._latTarget=-(TW*0.44);c._baseLat=c._initLat;c._ovTimer=0;c._inStart=false;}
    }
  } else if(c.pitState==='ENTER'||c.pitState==='LEAVE'){
    c._boxT=Math.min(1,(c._boxT||0)+dt/0.7);
    var e=c._boxT*c._boxT*(3-2*c._boxT),bx2=boxXOf(c);
    var fromX,toX,fromZ,toZ;
    if(c.pitState==='ENTER'){fromX=c._laneX;toX=bx2;fromZ=PIT_Z;toZ=PIT_BOX_Z;}
    else {fromX=bx2;toX=c._laneX;fromZ=PIT_BOX_Z;toZ=PIT_Z;}
    c.x=fromX+(toX-fromX)*e;c.z=fromZ+(toZ-fromZ)*e;c.y=0;c.heading=c._boxHead;
    c.speed=0;c.tIdx=closestWP(c.x,35,c.tIdx);
    if(c._boxT>=1){
      if(c.pitState==='ENTER'){c.pitState='BOX';c.pitTimer=0;c._dmgAtBox=c.damage;}
      else {c.pitState='EXIT';c.pitS=boxSOf(c);c.speed=PIT_LIMIT;}
    }
  } else if(c.pitState==='BOX'){
    c.speed=0;c.pitTimer=(c.pitTimer||0)+dt;
    if(c.pitTimer>=PIT_STOP_DUR+(c._dmgAtBox||0)*REPAIR_TIME){
      if(isPlayer){P.compound=(P._pitCompound==null?P.compound:P._pitCompound);P._pitCompound=null;P.tireWear=1;P.tireTempF=0.45;P.tireTempR=0.45;P._boxWarned=false;
        if(playerGrp&&playerGrp.userData.setTyre)playerGrp.userData.setTyre(COMPOUNDS[P.compound].col);
        if((c._dmgAtBox||0)>0.15) radio('New front wing on, damage fixed — back to full pace.',false);}
      else {c.compound=(c._nextCompound==null?c.compound:c._nextCompound);c._nextCompound=null;c._deg=0;
        var _ag=aiGrps[c._idx];if(_ag&&_ag.userData.setTyre)_ag.userData.setTyre(COMPOUNDS[c.compound].col);}
      c.damage=0;c.dmgFront=0;c.dmgRear=0;c._dmgAtBox=0;
      c.pitsDone=(c.pitsDone||0)+1;c.pitState='LEAVE';c._boxT=0;
    }
  }
}

function buildPitCrew(col,bx){
  var rec={grp:new THREE.Group(),guns:[],crouchers:[],jacks:[],lolly:null,_busy:false,_emerge:0};
  rec.grp.position.set(bx,0,CREW_STOW_Z);
  var mSuit=new THREE.MeshStandardMaterial({color:col,roughness:0.62,metalness:0.06});
  var mDark=new THREE.MeshStandardMaterial({color:0x14171c,roughness:0.85});
  var mSkin=new THREE.MeshStandardMaterial({color:0xd9b18c,roughness:0.85});
  var mGun=new THREE.MeshStandardMaterial({color:0xffd000,emissive:0x231a00,roughness:0.5,metalness:0.4});
  function fig(lx,lz,withGun){
    var G=new THREE.Group();
    function part(geo,mat,x,y,z,rx){var m=new THREE.Mesh(geo,mat);m.position.set(x,y,z);if(rx)m.rotation.x=rx;m.castShadow=true;G.add(m);return m;}
    part(new THREE.CylinderGeometry(0.21,0.16,0.6,8),mSuit,0,1.16,0);
    part(new THREE.BoxGeometry(0.32,0.32,0.22),mDark,0,0.8,0);
    part(new THREE.CylinderGeometry(0.07,0.07,0.52,6),mDark,-0.1,0.5,0);
    part(new THREE.CylinderGeometry(0.07,0.07,0.52,6),mDark,0.1,0.5,0);
    part(new THREE.SphereGeometry(0.14,10,8),mSkin,0,1.58,0);
    var hel=part(new THREE.SphereGeometry(0.17,12,9),mSuit,0,1.6,0.01);hel.scale.set(1,0.82,1.05);
    part(new THREE.CylinderGeometry(0.06,0.06,0.5,6),mSuit,-0.25,1.12,0.18,0.7);
    part(new THREE.CylinderGeometry(0.06,0.06,0.5,6),mSuit,0.25,1.12,0.18,0.7);
    if(withGun){
      var gun=new THREE.Mesh(new THREE.CylinderGeometry(0.07,0.07,0.5,8),mGun);
      gun.rotation.x=Math.PI/2;gun.position.set(0,0.95,0.42);G.add(gun);rec.guns.push(gun);
    }
    G.position.set(lx,0,lz);G.rotation.y=Math.atan2(-lx,-lz);
    G.userData.baseY=0;rec.grp.add(G);return G;
  }
  rec.crouchers.push(fig(1.7,-1.8,true),fig(1.7,1.8,true),fig(-1.5,-1.8,true),fig(-1.5,1.8,true));
  fig(2.5,2.7,false);fig(-2.3,2.7,false);
  rec.jacks.push(fig(3.5,0,false),fig(-3.7,0,false));
  var lo=new THREE.Group();
  var pole=new THREE.Mesh(new THREE.CylinderGeometry(0.04,0.04,2.4,6),mDark);pole.position.y=1.2;lo.add(pole);
  var board=new THREE.Mesh(new THREE.CylinderGeometry(0.34,0.34,0.06,18),new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:0.5,roughness:0.5}));
  board.rotation.x=Math.PI/2;board.position.set(0,2.4,-0.5);lo.add(board);
  lo.position.set(4.4,0,0);rec.grp.add(lo);rec.lolly=lo;
  return rec;
}

function buildGarage(cfg,bx,m){
  function pm(geo,mat,x,y,z){var me=new THREE.Mesh(geo,mat);me.position.set(x,y,z);me.castShadow=true;me.receiveShadow=true;scene.add(me);return me;}
  var W=28,D=13,H=7,gz=GARAGE_Z;
  pm(new THREE.BoxGeometry(W,H,0.5),m.conc,bx,H/2,gz+D/2);
  pm(new THREE.BoxGeometry(0.5,H,D),m.conc,bx-W/2,H/2,gz);
  pm(new THREE.BoxGeometry(0.5,H,D),m.conc,bx+W/2,H/2,gz);
  pm(new THREE.BoxGeometry(W+1.5,0.5,D+3),m.roof,bx,H+0.25,gz-1.5);
  var fas=new THREE.MeshStandardMaterial({color:cfg.col,emissive:cfg.col,emissiveIntensity:0.85,roughness:0.5});
  pm(new THREE.BoxGeometry(W-2,1.7,0.4),fas,bx,H-0.5,gz-D/2+0.2);
  pm(new THREE.BoxGeometry(5,2.4,0.2),m.screen,bx,4.2,gz+D/2-0.35);
  for(var s=0;s<2;s++)for(var u=0;u<3;u++) pm(new THREE.CylinderGeometry(0.5,0.5,0.34,16),m.tyre,bx-W/2+2.2+s*1.2,0.2+u*0.36,gz+2.4);
  pm(new THREE.BoxGeometry(1.5,1.0,0.9),m.metal,bx+W/2-2.6,0.6,gz+1.5);
  var bxw=3.4,bzd=2.4,bw=0.16;
  pm(new THREE.BoxGeometry(bxw*2,0.05,bw),m.line,bx,0.07,PIT_BOX_Z-bzd);
  pm(new THREE.BoxGeometry(bxw*2,0.05,bw),m.line,bx,0.07,PIT_BOX_Z+bzd);
  pm(new THREE.BoxGeometry(bw,0.05,bzd*2),m.line,bx-bxw,0.07,PIT_BOX_Z);
  pm(new THREE.BoxGeometry(bw,0.05,bzd*2),m.line,bx+bxw,0.07,PIT_BOX_Z);
  var nMat=new THREE.MeshBasicMaterial({map:numTex(cfg.num,'#111','#fff'),transparent:true});
  var npl=new THREE.Mesh(new THREE.PlaneGeometry(2.0,2.0),nMat);npl.rotation.x=-Math.PI/2;npl.position.set(bx,0.075,PIT_BOX_Z+bzd-1.4);scene.add(npl);
  var red=new THREE.MeshStandardMaterial({color:0xb71c1c,roughness:0.5,metalness:0.3});
  var blk=new THREE.MeshStandardMaterial({color:0x16181c,roughness:0.6,metalness:0.4});
  pm(new THREE.BoxGeometry(2.4,1.6,0.9),red,bx-W/2+3.4,0.85,gz+1.0);
  for(var dr=0;dr<3;dr++) pm(new THREE.BoxGeometry(2.2,0.07,0.92),blk,bx-W/2+3.4,0.55+dr*0.42,gz+1.0);
  pm(new THREE.BoxGeometry(1.3,2.6,1.0),blk,bx+W/2-2.4,1.3,gz+1.6);
  pm(new THREE.BoxGeometry(0.9,0.5,0.06),m.screen,bx+W/2-2.4,1.95,gz+1.6-0.52);
  for(var s2=0;s2<2;s2++)for(var u2=0;u2<3;u2++) pm(new THREE.CylinderGeometry(0.5,0.5,0.34,16),m.tyre,bx+W/2-2.4-s2*1.2,0.2+u2*0.36,gz+3.0);
  var ban=new THREE.Mesh(new THREE.PlaneGeometry(W-3,1.2),new THREE.MeshStandardMaterial({color:cfg.col,emissive:cfg.col,emissiveIntensity:0.5,roughness:0.6,side:THREE.DoubleSide}));
  ban.position.set(bx,H-2.0,gz-D/2+0.05);ban.rotation.y=Math.PI;scene.add(ban);
  var wm=new THREE.Mesh(new THREE.PlaneGeometry(2.2,1.4),new THREE.MeshBasicMaterial({map:numTex(cfg.num,'#06101f','#36a0ff'),transparent:true}));
  wm.position.set(bx+5,3.6,gz-D/2+0.16);wm.rotation.y=Math.PI;scene.add(wm);
  pm(new THREE.BoxGeometry(W-4,0.12,0.6),new THREE.MeshStandardMaterial({color:0xcfe0ff,emissive:0xbcd2ff,emissiveIntensity:1.2,roughness:0.4}),bx,H-0.4,gz);
  var drv=TEAM_DRIVERS[cfg.id]||['',''];
  for(var nbI=0;nbI<2;nbI++){
    var nb=new THREE.Mesh(new THREE.PlaneGeometry(W*0.42,1.0),new THREE.MeshBasicMaterial({map:nameTex(drv[nbI]||'',cfg.col),transparent:true}));
    nb.position.set(bx+(nbI===0?-1:1)*W*0.24,H-0.55,gz-D/2-0.1);nb.rotation.y=Math.PI;scene.add(nb);
  }
  // Garage interior life: a parked show car, idle mechanics, extra glowing screens and a
  // team-colour ceiling strip so the bays read busy instead of empty. The show car sits on
  // the +x side, clear of the crew stow corridor around bx (crew slide spans bx±3.7).
  var show=buildCar(cfg.col,cfg.acc,cfg.num,gameEnvMap(),false,null);
  show.position.set(bx+5.5,RIDE_H,gz+1);show.rotation.y=Math.PI/2;
  show.traverse(function(o){o.castShadow=false;});scene.add(show);
  [[bx+5.5,gz-3.6,0.3],[bx-7.6,gz+1.8,2.6],[bx-4.8,gz-2.2,-2.2]].forEach(function(mp){
    var mech=pitPerson(cfg.col,false);
    mech.grp.position.set(mp[0],0,mp[1]);mech.grp.rotation.y=mp[2];scene.add(mech.grp);
    idleFolk.push({head:mech.head,grp:mech.grp,mode:'stand',ph:Math.random()*6.28,baseRot:mp[2]});
  });
  pm(new THREE.BoxGeometry(1.8,1.0,0.1),m.screen,bx-8,2.8,gz+D/2-0.4);
  pm(new THREE.BoxGeometry(1.8,1.0,0.1),m.screen,bx+8.6,2.8,gz+D/2-0.4);
  pm(new THREE.BoxGeometry(1.2,1.5,0.7),m.metal,bx-2.6,0.75,gz-1.8);
  pm(new THREE.BoxGeometry(1.05,0.6,0.07),m.screen,bx-2.6,1.65,gz-1.8-0.32);
  var glow=new THREE.MeshStandardMaterial({color:cfg.col,emissive:cfg.col,emissiveIntensity:1.4,roughness:0.5});
  pm(new THREE.BoxGeometry(W-6,0.15,1.2),glow,bx,H-0.25,gz+2.5);
}

function pitPerson(col,seated){
  var G=new THREE.Group();
  var suit=new THREE.MeshStandardMaterial({color:col,roughness:0.6,metalness:0.05});
  var dark=new THREE.MeshStandardMaterial({color:0x14171c,roughness:0.85});
  var skin=new THREE.MeshStandardMaterial({color:0xd9b18c,roughness:0.85});
  var hs=new THREE.MeshStandardMaterial({color:0x101216,roughness:0.5,metalness:0.3});
  function p(geo,mat,x,y,z,rx){var me=new THREE.Mesh(geo,mat);me.position.set(x,y,z);if(rx)me.rotation.x=rx;me.castShadow=true;G.add(me);return me;}
  var hipY=seated?0.5:0.0;
  p(new THREE.CylinderGeometry(0.2,0.16,0.58,8),suit,0,hipY+1.12,0);
  p(new THREE.BoxGeometry(0.3,0.3,0.2),dark,0,hipY+0.78,0);
  if(seated){
    p(new THREE.BoxGeometry(0.13,0.12,0.5),dark,-0.1,hipY+0.6,0.26,-1.2);
    p(new THREE.BoxGeometry(0.13,0.12,0.5),dark,0.1,hipY+0.6,0.26,-1.2);
  } else {
    p(new THREE.CylinderGeometry(0.07,0.07,0.62,6),dark,-0.1,0.45,0);
    p(new THREE.CylinderGeometry(0.07,0.07,0.62,6),dark,0.1,0.45,0);
  }
  var head=p(new THREE.SphereGeometry(0.14,10,8),skin,0,hipY+1.52,0);
  var cap=p(new THREE.SphereGeometry(0.16,12,8),suit,0,hipY+1.55,0.01);cap.scale.set(1,0.7,1.05);
  var band=p(new THREE.TorusGeometry(0.15,0.022,6,14),hs,0,hipY+1.5,0);band.rotation.y=Math.PI/2;
  p(new THREE.SphereGeometry(0.05,6,6),hs,0.14,hipY+1.48,0.02);
  var ar=seated?-1.1:0.2;
  p(new THREE.CylinderGeometry(0.055,0.055,0.5,6),suit,-0.22,hipY+1.08,seated?0.16:0.0,ar);
  p(new THREE.CylinderGeometry(0.055,0.055,0.5,6),suit,0.22,hipY+1.08,seated?0.16:0.0,ar);
  return {grp:G,head:head};
}

function buildPitWallStand(cfg,bx){
  var gz=PIT_Z-PIT_W*0.5-2.6;
  var topY=1.5,segW=BOX_SP*0.82,half=segW/2;
  var body=new THREE.MeshStandardMaterial({color:0x16181d,roughness:0.6,metalness:0.3});
  var trim=new THREE.MeshStandardMaterial({color:cfg.col,emissive:cfg.col,emissiveIntensity:0.7,roughness:0.5});
  var scr=new THREE.MeshStandardMaterial({color:0x0a1830,emissive:0x2060ff,emissiveIntensity:1.3,roughness:0.35});
  function m(geo,mat,x,y,z){var me=new THREE.Mesh(geo,mat);me.position.set(x,y,z);me.castShadow=true;me.receiveShadow=true;scene.add(me);return me;}
  m(new THREE.BoxGeometry(segW,0.18,3.0),body,bx,topY,gz);
  [-half+0.4,half-0.4].forEach(function(lx){m(new THREE.BoxGeometry(0.3,topY,0.3),body,bx+lx,topY*0.5,gz+1.2);});
  m(new THREE.BoxGeometry(segW,0.9,1.0),body,bx,topY+0.55,gz-0.9);
  m(new THREE.BoxGeometry(segW,0.14,0.06),trim,bx,topY+0.95,gz-1.4);
  var nMon=3;for(var mi=0;mi<nMon;mi++){
    var sm=m(new THREE.BoxGeometry(segW/nMon-0.2,0.7,0.07),scr,bx-half+segW/nMon*(mi+0.5),topY+1.4,gz-1.3);sm.rotation.x=-0.18;
  }
  m(new THREE.BoxGeometry(segW+0.6,0.1,3.4),body,bx,topY+2.5,gz);
  m(new THREE.BoxGeometry(segW+0.6,0.12,0.12),trim,bx,topY+2.5,gz-1.7);
  [-half,half].forEach(function(lx){m(new THREE.BoxGeometry(0.12,1.0,0.12),body,bx+lx,topY+2.0,gz-1.6);});
  [-segW*0.22,segW*0.22].forEach(function(ox){
    var pn=pitPerson(cfg.col,true);
    pn.grp.position.set(bx+ox,topY+0.18,gz+0.4);pn.grp.rotation.y=Math.PI;scene.add(pn.grp);
    idleFolk.push({head:pn.head,mode:'type',ph:Math.random()*6.28});
  });
}

function animateStop(cr,tt){
  var working=tt>=0.45&&tt<PIT_STOP_DUR-0.35;
  var jacked=tt>=0.35&&tt<PIT_STOP_DUR-0.3;
  var k;
  for(k=0;k<cr.crouchers.length;k++){
    var fg=cr.crouchers[k],ty=working?-0.22:0;
    fg.position.y+=(ty-fg.position.y)*0.3;
  }
  for(k=0;k<cr.guns.length;k++) if(working) cr.guns[k].rotation.y+=0.6;
  for(k=0;k<cr.jacks.length;k++){var jr=jacked?0.5:0;cr.jacks[k].rotation.x+=(jr-cr.jacks[k].rotation.x)*0.3;}
  if(cr.lolly){var up=(tt<0||tt>=PIT_STOP_DUR-0.45)?-1.2:0;cr.lolly.rotation.x+=(up-cr.lolly.rotation.x)*0.25;}
}
function idleCrew(cr,clock){
  for(var i=0;i<cr.crouchers.length;i++) cr.crouchers[i].position.y=Math.sin(clock*1.6+i*1.3)*0.03;
  if(cr.lolly) cr.lolly.rotation.x+=(-1.2-cr.lolly.rotation.x)*0.1;
  for(var j=0;j<cr.jacks.length;j++) cr.jacks[j].rotation.x+=(0-cr.jacks[j].rotation.x)*0.1;
}
function updatePits(dt){
  pitClock+=dt;
  var id;for(id in teamCrew) teamCrew[id]._busy=false;
  var cars=[P].concat(AI),i;
  for(i=0;i<cars.length;i++){
    var c=cars[i];if(!inPit(c))continue;
    var cr=teamCrew[c.team];if(!cr)continue;
    var dApproach=boxXOf(c)-c.x;
    var ready=(c.pitState==='ENTER'||c.pitState==='BOX'||c.pitState==='LEAVE')||
              (c.pitState==='LANE'&&dApproach>-4&&dApproach<45);
    if(!ready)continue;
    cr._busy=true;
    if(c.pitState==='BOX'){
      animateStop(cr,c.pitTimer);
      c.pitLift=(c.pitTimer>0.45&&c.pitTimer<PIT_STOP_DUR-0.35)?0.26:0;
      if(c===P&&c.pitTimer>0.5&&c.pitTimer<2.0){c._gunT=(c._gunT||0)-dt;if(c._gunT<=0){_fireWheelGun();c._gunT=0.13;}}
    } else {animateStop(cr,-1);c.pitLift=0;}
  }
  for(id in teamCrew){
    var cr2=teamCrew[id];
    cr2._emerge+=((cr2._busy?1:0)-cr2._emerge)*Math.min(dt*4,1);
    cr2.grp.position.z=CREW_STOW_Z+(PIT_BOX_Z-CREW_STOW_Z)*cr2._emerge;
    if(cr2._busy)continue;
    idleCrew(cr2,pitClock);
  }
  for(var fi=0;fi<idleFolk.length;fi++){
    var fk=idleFolk[fi];
    if(fk.mode==='type'){fk.head.rotation.x=Math.sin(pitClock*1.3+fk.ph)*0.12;fk.head.rotation.y=Math.sin(pitClock*0.5+fk.ph)*0.18;}
    else{
      fk.head.rotation.y=Math.sin(pitClock*0.6+fk.ph)*0.4;
      fk.head.rotation.x=Math.sin(pitClock*0.9+fk.ph*1.7)*0.12;
      if(fk.grp){
        fk.grp.position.y=Math.abs(Math.sin(pitClock*1.4+fk.ph))*0.03;
        fk.grp.rotation.z=Math.sin(pitClock*0.8+fk.ph)*0.05;
        fk.grp.rotation.y=(fk.baseRot||0)+Math.sin(pitClock*0.3+fk.ph)*0.28;
      }
    }
  }
}

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
// --- Tokyo night palette: warm interior whites/ambers carry the city mass, with layered
// cyan/magenta/red signage neon on top; fog is a cool blue-violet smog. One shared set of
// hues reused across the whole world (signs, stands, garages) so the city reads as one place.
var NEON_RED=0xFF3B4E, NEON_AMBER=0xFFB347, NEON_CYAN=0x22D3EE, NEON_MAG=0xE94FD8;
var WARM_SIGN=0xFFE2B0, TOKYO_VIOLET=0x8B7BFF;
var CITY_FOG=0x141A33, CITY_GROUND=0x0a0c14;
var NEON_CYCLE=[NEON_CYAN,WARM_SIGN,NEON_MAG,NEON_AMBER,NEON_RED];
scene.fog=new THREE.FogExp2(CITY_FOG,0.0017);
var gameCam=new THREE.PerspectiveCamera(72,1,0.1,2000);
scene.add(gameCam);

var composer=null,useComposer=false,afterPass=null;
try{
  if(THREE.EffectComposer&&THREE.RenderPass&&THREE.UnrealBloomPass){
    composer=new THREE.EffectComposer(renderer);
    composer.addPass(new THREE.RenderPass(scene,gameCam));
    composer.addPass(new THREE.UnrealBloomPass(new THREE.Vector2(1280,720),0.72,0.42,0.80));
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

scene.add(new THREE.AmbientLight(0x16203a,0.30));
scene.add(new THREE.HemisphereLight(0x2a3658,0x0c0a14,0.42));
var sun=new THREE.DirectionalLight(0xbfd0ff,0.34);
sun.position.set(180,320,80);sun.castShadow=true;
sun.shadow.mapSize.width=sun.shadow.mapSize.height=2048;
sun.shadow.camera.near=1;sun.shadow.camera.far=1200;
sun.shadow.camera.left=-600;sun.shadow.camera.right=600;
sun.shadow.camera.top=600;sun.shadow.camera.bottom=-600;
sun.shadow.bias=-0.0004;sun.shadow.normalBias=0.03;
sun.shadow.radius=2.5;
scene.add(sun);
var fill=new THREE.DirectionalLight(0xFFB36B,0.18);
fill.position.set(-120,60,-80);scene.add(fill);

var clouds=[],standAnchors=[],crowdMats=[],crowdFolk=[];
var fanWalk=[],fanBody=null,fanHead=null,_fanDummy=new THREE.Object3D();
var staffWalk=[],staffBody=null,staffHead=null,fanFlags=[];
var marshals=[],skyObjs=[];
// Shared instanced-people updater: 'walk' ping-pongs a line, 'mill' orbits a point,
// 'stand' idles in place with a slow sway. Used by both fan and staff meshes.
function _updateFolk(arr,bodyIM,headIM,t,dt){
  for(var i=0;i<arr.length;i++){
    var w=arr[i],fx,fy,fz,fa;
    if(w.kind==='walk'){
      w.t+=w.spd*w.dir*dt;
      if(w.t>1){w.t=1;w.dir=-1;}else if(w.t<0){w.t=0;w.dir=1;}
      fx=w.x0+w.dx*w.len*w.t;fz=w.z0+w.dz*w.len*w.t;
      fa=Math.atan2(w.dx*w.dir,w.dz*w.dir);
      fy=w.y+Math.abs(Math.sin(t*4.2+w.ph))*0.06;
    } else if(w.kind==='stand'){
      fx=w.x;fz=w.z;
      fa=w.ry+Math.sin(t*0.6+w.ph)*0.35;
      fy=w.y+Math.sin(t*1.4+w.ph)*0.02;
    } else {
      w.a+=w.va*dt;
      fx=w.cx+Math.cos(w.a)*w.r;fz=w.cz+Math.sin(w.a)*w.r;
      fa=-w.a;
      fy=w.y+Math.abs(Math.sin(t*4.2+w.ph))*0.06;
    }
    _fanDummy.position.set(fx,fy,fz);_fanDummy.rotation.y=fa;_fanDummy.updateMatrix();
    bodyIM.setMatrixAt(i,_fanDummy.matrix);headIM.setMatrixAt(i,_fanDummy.matrix);
  }
  bodyIM.instanceMatrix.needsUpdate=true;headIM.instanceMatrix.needsUpdate=true;
}
var flagState='none',flagTimer=0;
function setFlag(state,secs){flagState=state;flagTimer=secs||0;}
function raiseYellow(c){if(flagState==='chequered') return;flagState='yellow';flagTimer=6.0;}
var FLAG_COLORS={green:0x18c83c,yellow:0xffd400,blue:0x1466ff,chequered:0xffffff,none:0x222222};
var chequerTexture=(function(){
  var cv=document.createElement('canvas');cv.width=cv.height=64;var g=cv.getContext('2d');
  for(var yy=0;yy<8;yy++)for(var xx=0;xx<8;xx++){g.fillStyle=((xx+yy)%2)?'#111':'#fff';g.fillRect(xx*8,yy*8,8,8);}
  return new THREE.CanvasTexture(cv);
})();
var _carbonTex=null,_numTexCache={},_bgTex=null;
function buildWorld(){
  var i,b,wp,p,hw;
  (function(){
    var sgeo=new THREE.SphereGeometry(1800,32,20);
    var sp=sgeo.getAttribute('position'),sc=[];
    for(var si=0;si<sp.count;si++){
      var ny=sp.getY(si)/1800;
      var sh=(ny+1)*0.5;
      var lo=1-Math.min(1,Math.max(0,sh));
      // Tokyo night: near-black overhead bleeding into a blue-violet city-glow haze at the
      // horizon (light pollution). lo=1 at the horizon, 0 at the zenith.
      sc.push(0.010+0.045*lo, 0.012+0.060*lo, 0.028+0.155*lo);
    }
    sgeo.setAttribute('color',new THREE.Float32BufferAttribute(sc,3));
    scene.add(new THREE.Mesh(sgeo,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.BackSide})));
    var moon=new THREE.Mesh(new THREE.SphereGeometry(34,20,14),new THREE.MeshBasicMaterial({color:0xF2ECD8}));
    moon.position.set(1380,640,420);scene.add(moon);
    var moonGlow=new THREE.Mesh(new THREE.SphereGeometry(110,16,10),new THREE.MeshBasicMaterial({color:0xAFC4FF,transparent:true,opacity:0.10}));
    moonGlow.position.copy(moon.position);scene.add(moonGlow);
    var stN=700,stPos=[];
    for(var sti=0;sti<stN;sti++){
      var u=Math.random()*Math.PI*2,v=Math.random()*0.9+0.06,rr=1700;
      var sy=Math.cos(v*Math.PI*0.5);
      var sxz=Math.sqrt(1-sy*sy);
      stPos.push(Math.cos(u)*sxz*rr,sy*rr,Math.sin(u)*sxz*rr);
    }
    var stGeo=new THREE.BufferGeometry();
    stGeo.setAttribute('position',new THREE.Float32BufferAttribute(stPos,3));
    scene.add(new THREE.Points(stGeo,new THREE.PointsMaterial({color:0xE8EEFF,size:2.6,sizeAttenuation:false,transparent:true,opacity:0.50,fog:false})));
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
    scene.add(new THREE.Points(mwGeo,new THREE.PointsMaterial({color:0xAFC0E8,size:2.0,sizeAttenuation:false,transparent:true,opacity:0.25,fog:false})));
    var cc2=document.createElement('canvas');cc2.width=cc2.height=128;var cg2=cc2.getContext('2d');
    var crg=cg2.createRadialGradient(64,64,4,64,64,64);
    crg.addColorStop(0,'rgba(56,64,110,0.50)');crg.addColorStop(0.5,'rgba(34,40,72,0.22)');crg.addColorStop(1,'rgba(34,40,72,0)');
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
  // --- Ground texture module: large soft tonal mottle + directional grain (speckle noise
  // shimmers at distance). Blobs are drawn wrapped at the canvas edges so the textures
  // tile seamlessly.
  function mottle(g,size,count,aLo,aHi){
    for(var mi=0;mi<count;mi++){
      var mx=Math.random()*size,my=Math.random()*size,mr=size*(0.15+Math.random()*0.20);
      var col=Math.random()<0.5?'0,0,0':'255,255,255';
      var al=aLo+Math.random()*(aHi-aLo);
      for(var ox=-1;ox<=1;ox++)for(var oy=-1;oy<=1;oy++){
        var bx=mx+ox*size,by=my+oy*size;
        if(bx+mr<0||bx-mr>size||by+mr<0||by-mr>size) continue;
        var gr=g.createRadialGradient(bx,by,0,bx,by,mr);
        gr.addColorStop(0,'rgba('+col+','+al.toFixed(3)+')');
        gr.addColorStop(1,'rgba('+col+',0)');
        g.fillStyle=gr;g.fillRect(bx-mr,by-mr,mr*2,mr*2);
      }
    }
  }
  function canvasTex(cc,rep){
    var t=new THREE.CanvasTexture(cc);t.wrapS=t.wrapT=THREE.RepeatWrapping;
    t.anisotropy=renderer.capabilities.getMaxAnisotropy();
    if(rep) t.repeat.set(rep,rep);return t;
  }
  function normalFromHeight(hc,size,strength){
    var hd=hc.getContext('2d').getImageData(0,0,size,size).data;
    var nc=document.createElement('canvas');nc.width=nc.height=size;var ng=nc.getContext('2d');
    var out=ng.createImageData(size,size),od=out.data;
    function H(x,y){x=(x+size)%size;y=(y+size)%size;return hd[((y*size+x)*4)]/255;}
    for(var y=0;y<size;y++)for(var x=0;x<size;x++){
      var nx=-(H(x+1,y)-H(x-1,y))*strength,ny=-(H(x,y+1)-H(x,y-1))*strength,nz=1;
      var l=Math.sqrt(nx*nx+ny*ny+nz*nz)||1,o=(y*size+x)*4;
      od[o]=(nx/l*0.5+0.5)*255;od[o+1]=(ny/l*0.5+0.5)*255;od[o+2]=(nz/l*0.5+0.5)*255;od[o+3]=255;
    }
    ng.putImageData(out,0,0);
    return canvasTex(nc,0);
  }
  // Road UVs: u spans the track width 0..1, v runs along the track (one tile per 5 waypoints),
  // so full-height vertical strokes = longitudinal grain/lines that tile seamlessly lap-round.
  function asphaltGrain(g,size,count,aLo,aHi,vLo,vHi){
    for(var gi=0;gi<count;gi++){
      var v=Math.floor(vLo+Math.random()*(vHi-vLo));
      g.fillStyle='rgba('+v+','+v+','+v+','+(aLo+Math.random()*(aHi-aLo)).toFixed(3)+')';
      g.fillRect(Math.random()*size,0,1,size);
    }
  }
  function rubberBands(g,size,centerAlphaCss){
    for(var bi=0;bi<2;bi++){
      var bc=(bi===0?0.30:0.70)*size,bw=0.12*size;
      var bg=g.createLinearGradient(bc-bw,0,bc+bw,0);
      bg.addColorStop(0,'rgba(0,0,0,0)');bg.addColorStop(0.5,centerAlphaCss);bg.addColorStop(1,'rgba(0,0,0,0)');
      g.fillStyle=bg;g.fillRect(bc-bw,0,bw*2,size);
    }
  }
  function asphaltColorTex(){
    var S=1024,cc=document.createElement('canvas');cc.width=cc.height=S;var g=cc.getContext('2d');
    g.fillStyle='#d8d8dc';g.fillRect(0,0,S,S);
    mottle(g,S,30,0.03,0.06);
    asphaltGrain(g,S,420,0.04,0.09,160,235);
    rubberBands(g,S,'rgba(16,16,20,0.30)');
    for(var ps=0;ps<6;ps++){
      g.fillStyle='rgba(0,0,0,'+(0.05+Math.random()*0.04).toFixed(3)+')';
      g.fillRect(Math.random()*S,0,2+Math.random()*3,S);
    }
    return canvasTex(cc,0);
  }
  function asphaltNormalTex(){
    var S=512,hc=document.createElement('canvas');hc.width=hc.height=S;var hg=hc.getContext('2d');
    hg.fillStyle='#808080';hg.fillRect(0,0,S,S);
    mottle(hg,S,18,0.04,0.08);
    asphaltGrain(hg,S,300,0.05,0.11,96,170);
    return normalFromHeight(hc,S,1.0);
  }
  function asphaltRoughTex(){
    var S=512,cc=document.createElement('canvas');cc.width=cc.height=S;var g=cc.getContext('2d');
    g.fillStyle='#d6d6d6';g.fillRect(0,0,S,S);
    mottle(g,S,14,0.04,0.08);
    rubberBands(g,S,'rgba(110,110,110,0.45)');
    return canvasTex(cc,0);
  }
  function mottleTex(size,baseHex){
    var cc=document.createElement('canvas');cc.width=cc.height=size;var g=cc.getContext('2d');
    g.fillStyle=baseHex;g.fillRect(0,0,size,size);
    mottle(g,size,26,0.03,0.07);
    return canvasTex(cc,0);
  }
  // --- Tokyo texture module: seeded procedural canvases for street-wall facades, storefront
  // rows and vertical neon signs. Seeded srnd() keeps the layout identical run-to-run; all
  // canvases draw once and materials are cached per (variant, repeat).
  function srnd(si){var x=Math.sin(si*127.1+311.7)*43758.5453;return x-Math.floor(x);}
  var JFONT='"Hiragino Sans","Hiragino Kaku Gothic ProN","Yu Gothic","Meiryo","Noto Sans CJK JP","MS PGothic",sans-serif';
  var JWORDS=['ラーメン','居酒屋','カラオケ','寿司','焼肉','薬局','ホテル','パチンコ','喫茶','酒場','電器','カメラ','銀座','東京','新宿','クラブ','本屋','うどん'];
  function floorTile(vi){
    // Drawn in 512x256 coordinate space but rasterized at 2x (1024x512) — crisp window edges
    // and seams up close instead of bilinear mush.
    var c=document.createElement('canvas');c.width=1024;c.height=512;var g=c.getContext('2d');
    g.scale(2,2);
    var sd=vi*977+13;
    var walls=['#07080d','#0a0a10','#090b12','#0b0a0e'];
    g.fillStyle=walls[vi%walls.length];g.fillRect(0,0,512,256);
    for(var gi=0;gi<4;gi++){
      g.fillStyle='rgba(0,0,0,'+(0.05+srnd(sd+90+gi)*0.10).toFixed(2)+')';
      g.fillRect(srnd(sd+110+gi)*512,0,3+srnd(sd+130+gi)*9,256);
    }
    g.fillStyle='#101218';
    for(var fx=0;fx<=4;fx++)g.fillRect(fx*128-4,0,8,256);
    for(var fy=0;fy<=2;fy++)g.fillRect(0,fy*128-4,512,8);
    var wx=20+((vi*37)%3)*12,wy=18+((vi*53)%3)*12,ww=128-wx*2,wh=128-wy*2;
    var warm=['#ffd9a0','#ffe8c8','#fff4e0'];
    for(var ry=0;ry<2;ry++)for(var cx2=0;cx2<4;cx2++){
      var s2=sd+ry*17+cx2*5,r2=srnd(s2),px=cx2*128+wx,py=ry*128+wy;
      g.globalAlpha=1;g.fillStyle='#181b22';g.fillRect(px-2,py-2,ww+4,wh+4);
      if(r2<0.45){g.globalAlpha=0.55+srnd(s2+1)*0.45;g.fillStyle=warm[s2%3];}
      else if(r2<0.53){g.globalAlpha=0.6+srnd(s2+1)*0.4;g.fillStyle='#bfd8ff';}
      else if(r2<0.56){g.globalAlpha=0.9;g.fillStyle=(s2%2)?'#22d3ee':'#e94fd8';}
      else{g.globalAlpha=1;g.fillStyle='#0b0d13';}
      g.fillRect(px,py,ww,wh);
      if(r2>=0.56){
        g.globalAlpha=0.10;g.fillStyle='#9fb4d8';
        g.beginPath();g.moveTo(px,py+wh);g.lineTo(px+ww*0.45,py);g.lineTo(px+ww*0.75,py);g.lineTo(px+ww*0.30,py+wh);g.closePath();g.fill();
      } else if(srnd(s2+5)<0.25){
        g.globalAlpha=0.35;g.fillStyle='#0a0c12';
        for(var bl=0;bl<4;bl++)g.fillRect(px,py+4+bl*9,ww,4);
      }
      g.globalAlpha=0.9;g.fillStyle='#15171c';g.fillRect(px-5,py+wh,ww+10,7);
      if(srnd(s2+2)<0.28){
        g.globalAlpha=0.95;g.fillStyle='#1d2026';g.fillRect(px+ww*0.55,py+wh-17,ww*0.34,15);
        g.fillStyle='#2a2e36';g.fillRect(px+ww*0.58,py+wh-13,ww*0.28,4);
      }
    }
    g.globalAlpha=1;return c;
  }
  function shopRow(vi){
    // 1024x256 coordinate space rasterized at 2x (2048x512) — kanji signboards and storefront
    // detail stay legible at racing distance.
    var c=document.createElement('canvas');c.width=2048;c.height=512;var g=c.getContext('2d');
    g.scale(2,2);
    var sd=vi*1543+71;
    g.fillStyle='#0a0b10';g.fillRect(0,0,1024,256);
    var boards=[['#d22a35','#ffffff'],['#16204a','#ffe2b0'],['#0e0e10','#ffd400'],['#0e7a8a','#ffffff'],['#5a1030','#ff9ad2'],['#1c3a14','#d8ffb0']];
    for(var si2=0;si2<8;si2++){
      var x0=si2*128,s2=sd+si2*29,ty=srnd(s2);
      if(ty<0.22){
        g.fillStyle='#3a3d44';g.fillRect(x0+10,60,108,184);
        g.fillStyle='rgba(0,0,0,0.35)';
        for(var sh=0;sh<11;sh++)g.fillRect(x0+10,68+sh*16,108,5);
        g.fillStyle='rgba(20,22,26,0.8)';g.fillRect(x0+10,196,108,48);
        if(srnd(s2+17)<0.4){
          g.fillStyle='rgba(190,60,140,0.45)';g.font='900 30px '+JFONT;g.textAlign='center';
          g.save();g.translate(x0+64,150);g.rotate(-0.12);g.fillText(JWORDS[(s2+3)%JWORDS.length],0,0);g.restore();
        }
      } else if(ty<0.42){
        g.fillStyle='#16181f';g.fillRect(x0+8,90,112,154);
        g.fillStyle='rgba(255,220,160,0.85)';
        for(var aw=0;aw<3;aw++)g.fillRect(x0+18+aw*34,110+srnd(s2+5+aw)*14,26,40);
        g.fillStyle='rgba(255,236,200,0.9)';g.fillRect(x0+14,170,100,52);
        g.fillStyle='rgba(35,22,12,0.8)';
        for(var cr=0;cr<4;cr++)g.fillRect(x0+20+cr*24,182+srnd(s2+9+cr)*10,12,34);
        var awc=(s2%2)?'#c43a2e':'#0e6a4a';
        g.fillStyle=awc;
        for(var st2=0;st2<7;st2++){g.fillStyle=(st2%2)?'#efe8da':awc;g.fillRect(x0+6+st2*17,60,17,26);}
        g.fillStyle='rgba(0,0,0,0.3)';g.fillRect(x0+6,84,119,6);
      } else if(ty<0.56){
        g.fillStyle='#101116';g.fillRect(x0+8,60,112,184);
        var nrc=(s2%2)?'#1d2f6e':'#5a1420';
        g.fillStyle=nrc;
        for(var nr=0;nr<3;nr++)g.fillRect(x0+14+nr*36,60,32,96);
        g.fillStyle='rgba(255,244,220,0.95)';g.font='700 26px '+JFONT;g.textAlign='center';g.textBaseline='middle';
        g.fillText(JWORDS[(s2+7)%JWORDS.length].charAt(0),x0+64,106);
        var dg=g.createLinearGradient(0,160,0,244);
        dg.addColorStop(0,'#ffedca');dg.addColorStop(1,'#e8a050');
        g.fillStyle=dg;g.fillRect(x0+34,160,60,84);
        g.fillStyle='rgba(30,18,10,0.8)';g.fillRect(x0+58,178,12,66);
        g.fillStyle='rgba(255,180,80,0.9)';g.beginPath();g.arc(x0+22,150,7,0,6.283);g.fill();
        g.beginPath();g.arc(x0+106,150,7,0,6.283);g.fill();
      } else {
        var ig=g.createLinearGradient(0,60,0,244);
        ig.addColorStop(0,'#fff1d6');ig.addColorStop(1,'#ffb45e');
        g.fillStyle=ig;g.fillRect(x0+12,60,104,184);
        g.fillStyle='rgba(255,255,255,0.25)';
        for(var sf=0;sf<3;sf++)g.fillRect(x0+12,96+sf*46,104,4);
        g.fillStyle='rgba(30,18,10,0.85)';
        for(var fi2=0;fi2<3;fi2++){var fx2=x0+22+srnd(s2+3+fi2)*68;g.fillRect(fx2,116+srnd(s2+7+fi2)*60,18+srnd(s2+11+fi2)*22,68);}
        g.fillStyle='rgba(40,26,14,0.55)';g.fillRect(x0+88,104,28,140);
        if(srnd(s2+13)<0.4){g.fillStyle='rgba(180,40,50,0.9)';g.fillRect(x0+12,60,104,22);}
      }
      var bc=boards[s2%boards.length];
      g.fillStyle=bc[0];g.fillRect(x0+6,4,116,50);
      g.strokeStyle='rgba(255,255,255,0.25)';g.lineWidth=2;g.strokeRect(x0+8,6,112,46);
      g.fillStyle=bc[1];g.font='700 38px '+JFONT;g.textAlign='center';g.textBaseline='middle';
      g.fillText(JWORDS[s2%JWORDS.length],x0+64,30);
    }
    return c;
  }
  function vertSign(word,bg,fg,cap){
    // 96x384 coordinate space rasterized at 128x512 (power-of-two for clean mips).
    var c=document.createElement('canvas');c.width=128;c.height=512;var g=c.getContext('2d');
    g.scale(128/96,512/384);
    g.fillStyle='#05060a';g.fillRect(0,0,96,384);
    g.fillStyle=bg;g.fillRect(8,6,80,372);
    g.strokeStyle='rgba(255,255,255,0.45)';g.lineWidth=3;g.strokeRect(11,9,74,366);
    var iv=g.createLinearGradient(8,6,88,6);
    iv.addColorStop(0,'rgba(255,255,255,0.10)');iv.addColorStop(0.5,'rgba(255,255,255,0)');iv.addColorStop(1,'rgba(0,0,0,0.18)');
    g.fillStyle=iv;g.fillRect(8,6,80,372);
    g.fillStyle=fg;g.textAlign='center';g.textBaseline='middle';
    var L=word.length,yy0=(330-L*66)/2+50;
    g.font='900 56px '+JFONT;
    for(var ci=0;ci<L;ci++)g.fillText(word[ci],48,yy0+ci*66);
    g.font='700 22px '+JFONT;
    g.fillText(cap,48,358);
    return c;
  }
  var UPPER_CVS=[floorTile(0),floorTile(1),floorTile(2),floorTile(3),floorTile(4),floorTile(5),floorTile(6),floorTile(7)];
  var SHOP_CVS=[shopRow(0),shopRow(1),shopRow(2),shopRow(3),shopRow(4),shopRow(5)];
  // tint buckets: subtle base-color + glow variation so identical facade canvases still
  // read differently block to block (materials stay cached per variant+repeat+tint)
  var FAC_TINTS=[0x0a0b13,0x0e0f1a,0x0b0d10,0x12101c];
  var _facC={};
  function _cityMat(cv,rx,ry,ei,tint){
    var t=new THREE.CanvasTexture(cv);t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(rx,ry);
    t.anisotropy=renderer.capabilities.getMaxAnisotropy();
    return new THREE.MeshStandardMaterial({color:tint,map:t,emissiveMap:t,emissive:0xffffff,emissiveIntensity:ei,roughness:0.8,metalness:0.25});
  }
  function upperMat(vi,rx,ry,ti){var k='U'+vi+'_'+rx+'_'+ry+'_'+ti;return _facC[k]||(_facC[k]=_cityMat(UPPER_CVS[vi],rx,ry,0.56+ti*0.09,FAC_TINTS[ti]));}
  function shopMat(vi,rx,ti){var k='S'+vi+'_'+rx+'_'+ti;return _facC[k]||(_facC[k]=_cityMat(SHOP_CVS[vi],rx,1,0.82+ti*0.08,FAC_TINTS[ti]));}
  var SIGN_COMBOS=[['#d22a35','#ffffff'],['#16204a','#ffe2b0'],['#0e7a8a','#ffffff'],['#0a0a0c','#ffd400'],['#14060f','#e94fd8']];
  var SIGN_CAPS=['BAR','24h','3F','CAFE','HOTEL','CLUB'];
  var signPool=[];
  for(var sgi=0;sgi<12;sgi++){
    var sgc=SIGN_COMBOS[sgi%SIGN_COMBOS.length];
    var sgt=new THREE.CanvasTexture(vertSign(JWORDS[(sgi*5+2)%JWORDS.length],sgc[0],sgc[1],SIGN_CAPS[sgi%SIGN_CAPS.length]));
    sgt.anisotropy=renderer.capabilities.getMaxAnisotropy();
    signPool.push(new THREE.MeshStandardMaterial({color:0xffffff,map:sgt,emissiveMap:sgt,emissive:0xffffff,emissiveIntensity:1.15,roughness:0.5,metalness:0.1}));
  }
  var rPos=[],rIdx=[],rUv=[];
  for(i=0;i<N;i++){
    wp=waypoints[i];p=wpPerp(i);hw=TW*0.5;
    rPos.push(wp.x+p.x*hw,wp.y+0.06,wp.z+p.z*hw,
              wp.x-p.x*hw,wp.y+0.06,wp.z-p.z*hw);
    rUv.push(0,i*0.20, 1,i*0.20);
    if(i<N-1){b=i*2;rIdx.push(b,b+1,b+2,b+2,b+1,b+3);}
  }
  b=(N-1)*2;rIdx.push(b,b+1,0,0,b+1,1);
  var rGeo=new THREE.BufferGeometry();
  rGeo.setAttribute('position',new THREE.Float32BufferAttribute(rPos,3));
  rGeo.setAttribute('uv',new THREE.Float32BufferAttribute(rUv,2));
  rGeo.setIndex(rIdx);rGeo.computeVertexNormals();
  var asphaltTex=asphaltColorTex();
  var asphaltNrm=asphaltNormalTex();
  var asphaltRgh=asphaltRoughTex();
  var road=new THREE.Mesh(rGeo,new THREE.MeshStandardMaterial({color:0x17171b,map:asphaltTex,normalMap:asphaltNrm,normalScale:new THREE.Vector2(0.35,0.35),roughnessMap:asphaltRgh,roughness:0.95,metalness:0.05}));
  road.receiveShadow=true;scene.add(road);
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
    var lineMat=new THREE.MeshStandardMaterial({color:0xffffff,emissive:0x3a3a44,emissiveIntensity:0.25,roughness:0.55});
    var lEdge=TW*0.5-0.5;
    strip( lEdge,0.34,0.074,lineMat);
    strip(-lEdge,0.34,0.074,lineMat);
  })();
  (function(){
    var sfwp=waypoints[0],sfd=wpDir(0);
    var sfAngle=Math.atan2(sfd.x,sfd.z);
    var sfMat=new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xffffff,emissiveIntensity:0.3,roughness:0.7});
    for(var li=0;li<5;li++){
      var sOff=(li-2)*0.55;
      var sw=new THREE.Mesh(new THREE.BoxGeometry(TW+CURB*2,0.06,0.32),sfMat);
      sw.position.set(sfwp.x+sfd.x*sOff,sfwp.y+0.075,sfwp.z+sfd.z*sOff);
      sw.rotation.y=sfAngle;scene.add(sw);
    }
  })();

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
  scene.add(new THREE.Mesh(kGeo,new THREE.MeshStandardMaterial({vertexColors:true,emissive:0x2a2a2a,emissiveIntensity:0.25,roughness:0.8})));

  // True when a barrier point sits in the pit entry/exit corridor: the wall quads there are
  // skipped so the pit splines pass through open gaps instead of through the wall. The lane
  // mid-section stays walled (the narrower track moved the divider further from the lane
  // centerline, so the 5.5u radius still clears the entry/exit gaps without opening it).
  function nearPitPath(x,z){
    if(x<-335||x>190||z<38||z>66) return false;
    for(var pi=0;pi<=PIT_SEGS;pi+=2){
      var dx=pitPts[pi].x-x,dz=pitPts[pi].z-z;
      if(dx*dx+dz*dz<30.25) return true;
    }
    return false;
  }

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
      var wp2=waypoints[i+1],p2=wpPerp(i+1);
      if(!nearPitPath(wp.x+p.x*hw,wp.z+p.z*hw)&&!nearPitPath(wp2.x+p2.x*hw,wp2.z+p2.z*hw))
        baIdx.push(b,b+4,b+1,b+1,b+4,b+5);
      if(!nearPitPath(wp.x-p.x*hw,wp.z-p.z*hw)&&!nearPitPath(wp2.x-p2.x*hw,wp2.z-p2.z*hw))
        baIdx.push(b+2,b+6,b+3,b+3,b+6,b+7);
    }
  }
  var baGeo=new THREE.BufferGeometry();
  baGeo.setAttribute('position',new THREE.Float32BufferAttribute(baPos,3));
  baGeo.setIndex(baIdx);baGeo.computeVertexNormals();
  scene.add(new THREE.Mesh(baGeo,new THREE.MeshStandardMaterial({color:0x44464e,roughness:0.28,metalness:0.7})));

  var strPos=[],strIdx=[];
  var sY=BH*0.58,sH=0.28,strHw=TW*0.5+CURB;
  for(i=0;i<N;i++){
    wp=waypoints[i];p=wpPerp(i);
    strPos.push(wp.x+p.x*strHw,wp.y+sY,    wp.z+p.z*strHw,
                wp.x+p.x*strHw,wp.y+sY+sH, wp.z+p.z*strHw,
                wp.x-p.x*strHw,wp.y+sY,    wp.z-p.z*strHw,
                wp.x-p.x*strHw,wp.y+sY+sH, wp.z-p.z*strHw);
    if(i<N-1){
      b=i*4;
      var wpS=waypoints[i+1],pS=wpPerp(i+1);
      if(!nearPitPath(wp.x+p.x*strHw,wp.z+p.z*strHw)&&!nearPitPath(wpS.x+pS.x*strHw,wpS.z+pS.z*strHw))
        strIdx.push(b,b+4,b+1,b+1,b+4,b+5);
      if(!nearPitPath(wp.x-p.x*strHw,wp.z-p.z*strHw)&&!nearPitPath(wpS.x-pS.x*strHw,wpS.z-pS.z*strHw))
        strIdx.push(b+2,b+6,b+3,b+3,b+6,b+7);
    }
  }
  var strGeo=new THREE.BufferGeometry();
  strGeo.setAttribute('position',new THREE.Float32BufferAttribute(strPos,3));
  strGeo.setIndex(strIdx);strGeo.computeVertexNormals();
  scene.add(new THREE.Mesh(strGeo,new THREE.MeshStandardMaterial({color:0xCC0000,roughness:0.5})));

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
      // Flat urban ground: the shoulder stays at track height out to the building line (~46u)
      // so street-wall facades sit on level footings, then eases down to a flat city floor.
      var t=Math.max(0,Math.min(1,(dist-46)/40));
      var hy=(bY-0.8)*(1-t)+(-2.0)*t;
      gPos2[vi*3]=vx;gPos2[vi*3+1]=hy;gPos2[vi*3+2]=vz;
      var tFar=Math.max(0,Math.min(1,(dist-halfTW-10)/100));
      var tNear=Math.max(0,Math.min(1,1-(dist-halfTW)/12));
      // Wet city asphalt: cool blue-violet near-black, warm streetlight pool right at the track.
      var gcR=0.024+0.012*tFar,gcG=0.028+0.016*tFar,gcB=0.040+0.030*tFar;
      if(tNear>0){gcR=gcR*(1-tNear*0.4)+0.045*tNear;gcG=gcG*(1-tNear*0.4)+0.038*tNear;gcB=gcB*(1-tNear*0.4)+0.030*tNear;}
      gCol2.push(gcR,gcG,gcB);
      gUv2.push(c*0.5,r*0.5);
      if(r<rows-1&&c<cols-1){gIdx2.push(vi,vi+cols,vi+1,vi+1,vi+cols,vi+cols+1);}
    }
  }
  var tGeo=new THREE.BufferGeometry();
  tGeo.setAttribute('position',new THREE.Float32BufferAttribute(gPos2,3));
  tGeo.setAttribute('color',new THREE.Float32BufferAttribute(gCol2,3));
  tGeo.setAttribute('uv',new THREE.Float32BufferAttribute(gUv2,2));
  tGeo.setIndex(gIdx2);tGeo.computeVertexNormals();
  var cityFloorTex=mottleTex(512,'#cfcfcf');
  scene.add(new THREE.Mesh(tGeo,new THREE.MeshStandardMaterial({vertexColors:true,map:cityFloorTex,roughness:0.62,metalness:0.06})));
  (function(){
    // Outer ground: must run all the way to the fog horizon (1600u, where FogExp2 has fully
    // taken over) — any gap reads as the city floating on a void against the glowing
    // below-horizon sky. Far vertex colors lean into the haze tone so geometry and fog
    // meet seamlessly.
    var NS=64,rR=[200,690,1600],rY=[-2,-4,-5],ogPos=[],ogIdx=[],ogCol=[];
    var rC=[[0.020,0.024,0.040],[0.026,0.033,0.060],[0.046,0.060,0.115]];
    for(var oi=0;oi<=NS;oi++){
      var oAng=oi/NS*Math.PI*2,oCa=Math.cos(oAng),oSa=Math.sin(oAng);
      for(var rk=0;rk<3;rk++){
        ogPos.push(oCa*rR[rk],rY[rk],oSa*rR[rk]);
        ogCol.push(rC[rk][0],rC[rk][1],rC[rk][2]);
      }
      if(oi<NS){var ob=oi*3;
        ogIdx.push(ob,ob+3,ob+1, ob+1,ob+3,ob+4, ob+1,ob+4,ob+2, ob+2,ob+4,ob+5);}
    }
    var ogGeo=new THREE.BufferGeometry();
    ogGeo.setAttribute('position',new THREE.Float32BufferAttribute(ogPos,3));
    ogGeo.setAttribute('color',new THREE.Float32BufferAttribute(ogCol,3));
    ogGeo.setIndex(ogIdx);ogGeo.computeVertexNormals();
    scene.add(new THREE.Mesh(ogGeo,new THREE.MeshStandardMaterial({vertexColors:true,roughness:1.0})));
    // Distant city-light carpet: a couple thousand dim sodium/cool-white points (plus rare
    // neon) scattered across the far plain, so the dark ground reads as a metropolis seen
    // from above the streets, not empty space. Fog melts them into the horizon glow.
    var clPos=[],clCol=[];
    for(var li=0;li<2400;li++){
      var lAng=Math.random()*Math.PI*2,lR=340+Math.pow(Math.random(),0.8)*1150;
      clPos.push(Math.cos(lAng)*lR,-1.6,Math.sin(lAng)*lR);
      var lv=0.35+Math.random()*0.65,lt=Math.random();
      if(lt<0.62)clCol.push(1.0*lv,0.76*lv,0.45*lv);
      else if(lt<0.90)clCol.push(0.72*lv,0.82*lv,1.0*lv);
      else if(lt<0.96)clCol.push(0.13*lv,0.83*lv,0.93*lv);
      else clCol.push(0.91*lv,0.31*lv,0.85*lv);
    }
    var clGeo=new THREE.BufferGeometry();
    clGeo.setAttribute('position',new THREE.Float32BufferAttribute(clPos,3));
    clGeo.setAttribute('color',new THREE.Float32BufferAttribute(clCol,3));
    scene.add(new THREE.Points(clGeo,new THREE.PointsMaterial({vertexColors:true,size:2.0,sizeAttenuation:false,transparent:true,opacity:0.8,depthWrite:false})));
  })();

  // --- Tokyo street circuit: continuous building walls line the track in three districts
  // (Neo-Shinjuku towers / izakaya alley / Shibuya media), with masked gaps reserved for
  // grandstands, light towers, the pit complex and the trackside landmarks. blocked[] holds
  // reservations; walled[] records where facades actually landed so billboards and street
  // furniture can key off real coverage.
  var mProp=new THREE.MeshStandardMaterial({color:0x0a0a0d,roughness:0.55,metalness:0.45});
  var unit=new THREE.BoxGeometry(1,1,1);
  var capMat=new THREE.MeshStandardMaterial({color:0x090a12,roughness:0.7,metalness:0.5});
  var beaconMat=new THREE.MeshBasicMaterial({color:0xff1e3c,fog:false});
  var roofClutterMat=new THREE.MeshStandardMaterial({color:0x101218,roughness:0.85,metalness:0.3});
  var SEG_LEN=waypoints[0].distanceTo(waypoints[1]);
  var blocked=[new Uint8Array(N),new Uint8Array(N)];
  var walled=[new Uint8Array(N),new Uint8Array(N)];
  function blockArc(side,wpI,halfWp){
    var si=(side>0)?0:1;
    for(var bi=-halfWp;bi<=halfWp;bi++)blocked[si][((wpI+bi)%N+N)%N]=1;
  }
  [[10,1,70],[10,-1,55],[130,1,50],[280,-1,45],[450,1,60],[640,-1,40],[790,1,50]].forEach(function(gs){
    blockArc(gs[1],gs[0],Math.ceil((gs[2]*0.5+10)/SEG_LEN));
  });
  [40,150,260,370,480,590,700,810].forEach(function(twI,ti2){blockArc((ti2%2===0)?1:-1,twI,5);});
  blockArc(1,352,4);blockArc(-1,352,4);
  blockArc(1,250,3);blockArc(-1,250,3);
  blockArc(1,715,3);blockArc(-1,715,3);
  blockArc(1,605,9);
  function inPitRect(x,z){return x>-315&&x<190&&z>38&&z<102;}
  // True when (x,z) sits clear of the racing surface by `margin` units beyond the
  // track edge. Coarse full-loop scan (build-time only) so a prop projected off one
  // waypoint can't land on a neighbouring track section on a tight inside corner.
  function clearsTrack(x,z,margin){
    var lim=TW*0.5+CURB+(margin||0);lim*=lim;
    for(var wi=0;wi<N;wi+=3){
      var dx=x-waypoints[wi].x,dz=z-waypoints[wi].z;
      if(dx*dx+dz*dz<lim)return false;
    }
    return true;
  }
  var DISTRICTS=[
    {a:60,b:300,face:26,w0:18,w1:24,f0:8,f1:16,depth:16,sg:1},
    {a:300,b:560,face:21,w0:12,w1:15,f0:2,f1:4,depth:10,sg:3},
    {a:560,b:845,face:24,w0:16,w1:20,f0:6,f1:10,depth:13,sg:2}
  ];
  function placeBuilding(bx,bz,rotY,w,dst,mid,s){
    var wpB=waypoints[mid],y0=wpB.y-1.3,shopH=5.8,dep=dst.depth;
    var svi=(srnd(mid*61+s*7)*SHOP_CVS.length)|0,uvi=(srnd(mid*67+s*11)*UPPER_CVS.length)|0;
    var fti=(srnd(mid*71+s*13)*FAC_TINTS.length)|0;
    var fSpan=(dst.f1-dst.f0)/2+1;
    var floors=dst.f0+2*Math.floor(srnd(mid*13+s)*fSpan);
    if(floors>dst.f1)floors=dst.f1;
    var upH=floors*3;
    var p2=wpPerp(mid);
    var shop=new THREE.Mesh(unit,shopMat(svi,Math.max(1,Math.round(w/12)),fti));
    shop.position.set(bx,y0+shopH*0.5,bz);shop.scale.set(dep,shopH,w);shop.rotation.y=rotY;scene.add(shop);
    var up=new THREE.Mesh(unit,upperMat(uvi,Math.max(1,Math.round(w/4.5)),floors/2,fti));
    up.position.set(bx,y0+shopH+upH*0.5,bz);up.scale.set(dep*0.96,upH,w*0.98);up.rotation.y=rotY;scene.add(up);
    var topY=y0+shopH+upH;
    var capH=1.5+srnd(mid*29+s)*1.5;
    var cap=new THREE.Mesh(unit,capMat);
    cap.position.set(bx,topY+capH*0.5,bz);cap.scale.set(dep*0.7,capH,w*0.7);cap.rotation.y=rotY;scene.add(cap);
    if(floors>=12){
      var ant=new THREE.Mesh(new THREE.CylinderGeometry(0.3,0.7,9,5),mProp);
      ant.position.set(bx,topY+capH+4.5,bz);scene.add(ant);
      var bcn=new THREE.Mesh(new THREE.SphereGeometry(0.9,8,6),beaconMat);
      bcn.position.set(bx,topY+capH+9.4,bz);scene.add(bcn);
    } else if(srnd(mid*31+s)<0.4){
      var ac=new THREE.Mesh(unit,roofClutterMat);
      var aoff=(srnd(mid*41+s)-0.5)*w*0.4;
      var dd2=wpDir(mid);
      ac.position.set(bx+dd2.x*aoff,topY+0.8,bz+dd2.z*aoff);
      ac.scale.set(dep*0.22,1.6,w*0.16);ac.rotation.y=rotY;scene.add(ac);
    }
    var nSg=Math.round(dst.sg*(0.6+srnd(mid*43+s)*0.8));
    for(var sg2=0;sg2<nSg;sg2++){
      var signH=Math.min(9,Math.max(4,shopH+upH-3));
      var lat=((sg2%2===0)?-1:1)*(0.18+srnd(mid*37+sg2)*0.24)*w;
      var sdd=wpDir(mid);
      var sx=bx-p2.x*s*(dep*0.5+0.55)+sdd.x*lat;
      var sz=bz-p2.z*s*(dep*0.5+0.55)+sdd.z*lat;
      var sgn=new THREE.Mesh(unit,signPool[((mid*7+sg2*5+s)%12+12)%12]);
      sgn.position.set(sx,topY-capH-signH*0.5-0.8,sz);
      sgn.scale.set(1.7,signH,0.35);sgn.rotation.y=rotY;scene.add(sgn);
    }
  }
  var placedBlds=[];
  DISTRICTS.forEach(function(dst,di){
    [1,-1].forEach(function(s){
      var sideIdx=(s>0)?0:1;
      var cur=dst.a;
      while(cur<dst.b-4){
        if(blocked[sideIdx][cur]){cur++;continue;}
        var w=(srnd(cur*3+sideIdx)<0.5)?dst.w0:dst.w1;
        var span=Math.ceil(w/SEG_LEN);
        var hitBlock=false;
        for(var q2=cur;q2<=cur+span&&q2<dst.b;q2++){if(blocked[sideIdx][q2%N]){hitBlock=true;cur=q2+1;break;}}
        if(hitBlock)continue;
        var mid=(cur+(span>>1))%N,wp2=waypoints[mid],p3=wpPerp(mid),dd3=wpDir(mid);
        var off=dst.face+dst.depth*0.5;
        var bx=wp2.x+p3.x*s*off,bz=wp2.z+p3.z*s*off;
        var rotY=Math.atan2(dd3.x,dd3.z);
        var ok=true;
        // (1) pit complex + exit corridor exclusion: test all four footprint corners + center
        var hwF=w*0.5,hdF=dst.depth*0.5;
        var corners=[[1,1],[1,-1],[-1,1],[-1,-1],[0,0]];
        for(var pc=0;pc<corners.length&&ok;pc++){
          var ccx=bx+dd3.x*hwF*corners[pc][0]+p3.x*hdF*corners[pc][1];
          var ccz=bz+dd3.z*hwF*corners[pc][0]+p3.z*hdF*corners[pc][1];
          if(inPitRect(ccx,ccz))ok=false;
        }
        // (2) track clearance, sampled every 4th waypoint over the whole loop: front corners
        // must stay >= face-1.5 from any track section, and back corners must stay outside
        // the barrier zone of any OTHER section the building might back onto (>= 15)
        if(ok){
          var fOff=dst.face,bOffc=dst.face+dst.depth;
          var minD2=(dst.face-1.5)*(dst.face-1.5),minB2=15*15;
          var f1x=wp2.x+p3.x*s*fOff+dd3.x*hwF,f1z=wp2.z+p3.z*s*fOff+dd3.z*hwF;
          var f2x=wp2.x+p3.x*s*fOff-dd3.x*hwF,f2z=wp2.z+p3.z*s*fOff-dd3.z*hwF;
          var b1x=wp2.x+p3.x*s*bOffc+dd3.x*hwF,b1z=wp2.z+p3.z*s*bOffc+dd3.z*hwF;
          var b2x=wp2.x+p3.x*s*bOffc-dd3.x*hwF,b2z=wp2.z+p3.z*s*bOffc-dd3.z*hwF;
          for(var q3=0;q3<N&&ok;q3+=4){
            var wq=waypoints[q3];
            var dx1=wq.x-f1x,dz1=wq.z-f1z,dx2=wq.x-f2x,dz2=wq.z-f2z;
            if(dx1*dx1+dz1*dz1<minD2||dx2*dx2+dz2*dz2<minD2)ok=false;
            if(ok){
              var dx3=wq.x-b1x,dz3=wq.z-b1z,dx4=wq.x-b2x,dz4=wq.z-b2z;
              if(dx3*dx3+dz3*dz3<minB2||dx4*dx4+dz4*dz4<minB2)ok=false;
            }
          }
        }
        // (3) overlap vs every placed building (also catches infield collisions where two
        // track sections run close and their inner walls would interpenetrate)
        for(var nb=0;nb<placedBlds.length&&ok;nb++){
          var ob=placedBlds[nb],nbx=bx-ob.x,nbz=bz-ob.z;
          var minSep=0.5*(w+ob.w)+1;
          if(nbx*nbx+nbz*nbz<minSep*minSep)ok=false;
        }
        if(!ok){cur+=2;continue;}
        placeBuilding(bx,bz,rotY,w,dst,mid,s);
        for(var q4=cur;q4<=cur+span;q4++)walled[sideIdx][q4%N]=1;
        placedBlds.push({x:bx,z:bz,w:w});
        cur+=span+1;
        // inside-of-corner convergence: skip a few extra waypoints when the track bends
        // toward this side so parcels don't fold into each other (gaps read as alleys)
        var dA=wpDir(cur%N),dB=wpDir((cur+span)%N);
        var crossT=dA.x*dB.z-dA.z*dB.x;
        if(crossT*s>0)cur+=Math.ceil(Math.abs(crossT)*6);
      }
    });
  });
  // --- Tokyo street furniture in the sidewalk zone between barrier and facades: power poles
  // strung with sagging cables (one LineSegments draw call), vending machines glowing by the
  // storefronts, lantern strings in the izakaya district and two kanji wayfinding gantries.
  (function(){
    function districtAt(wpI){
      for(var di=0;di<DISTRICTS.length;di++){if(wpI>=DISTRICTS[di].a&&wpI<DISTRICTS[di].b)return DISTRICTS[di];}
      return null;
    }
    var mPole=new THREE.MeshStandardMaterial({color:0x2a2c30,roughness:0.85,metalness:0.15});
    var poleGeo=new THREE.CylinderGeometry(0.16,0.22,8.5,6);
    var armGeo=new THREE.BoxGeometry(2.0,0.16,0.16);
    var cablePts=[];
    var prevTop=[null,null];
    for(var pi=0;pi<N;pi+=28){
      var dstP=districtAt(pi);if(!dstP)continue;
      var sP=((pi/28)|0)%2===0?1:-1,siP=(sP>0)?0:1;
      if(blocked[siP][pi]&&!walled[siP][pi]){prevTop[siP]=null;continue;}
      var wpP=waypoints[pi],ppP=wpPerp(pi);
      var pxP=wpP.x+ppP.x*sP*(dstP.face-2),pzP=wpP.z+ppP.z*sP*(dstP.face-2);
      if(inPitRect(pxP,pzP)||!clearsTrack(pxP,pzP,BH+0.5)){prevTop[siP]=null;continue;}
      var pole=new THREE.Mesh(poleGeo,mPole);
      pole.position.set(pxP,wpP.y+4.25,pzP);scene.add(pole);
      var arm=new THREE.Mesh(armGeo,mPole);
      var ddP=wpDir(pi);
      arm.position.set(pxP,wpP.y+7.6,pzP);arm.rotation.y=Math.atan2(ddP.x,ddP.z);scene.add(arm);
      var arm2=new THREE.Mesh(armGeo,mPole);
      arm2.position.set(pxP,wpP.y+6.8,pzP);arm2.rotation.y=Math.atan2(ddP.x,ddP.z);scene.add(arm2);
      var top={x:pxP,y:wpP.y+8.3,z:pzP};
      var pv=prevTop[siP];
      if(pv){
        var spanD=Math.sqrt((top.x-pv.x)*(top.x-pv.x)+(top.z-pv.z)*(top.z-pv.z));
        if(spanD<28*SEG_LEN*1.7){
          for(var cseg=0;cseg<4;cseg++){
            var t0=cseg/4,t1=(cseg+1)/4;
            var sag0=0.9*4*t0*(1-t0),sag1=0.9*4*t1*(1-t1);
            cablePts.push(pv.x+(top.x-pv.x)*t0,pv.y+(top.y-pv.y)*t0-sag0,pv.z+(top.z-pv.z)*t0,
                          pv.x+(top.x-pv.x)*t1,pv.y+(top.y-pv.y)*t1-sag1,pv.z+(top.z-pv.z)*t1);
          }
        }
      }
      prevTop[siP]=top;
    }
    if(cablePts.length){
      var cGeo=new THREE.BufferGeometry();
      cGeo.setAttribute('position',new THREE.Float32BufferAttribute(cablePts,3));
      scene.add(new THREE.LineSegments(cGeo,new THREE.LineBasicMaterial({color:0x05060a})));
    }
    // vending machines: warm-lit boxes against the storefronts, facing the street
    function vendCanvas(bg,word){
      var c=document.createElement('canvas');c.width=64;c.height=96;var g=c.getContext('2d');
      g.fillStyle=bg;g.fillRect(0,0,64,96);
      g.fillStyle='rgba(255,255,255,0.92)';g.fillRect(6,8,52,34);
      g.fillStyle='#15171c';g.font='700 13px '+JFONT;g.textAlign='center';g.textBaseline='middle';
      g.fillText(word,32,25);
      for(var vr=0;vr<2;vr++)for(var vc=0;vc<3;vc++){g.fillStyle='rgba(20,24,30,0.85)';g.fillRect(9+vc*17,48+vr*16,14,12);}
      g.fillStyle='rgba(0,0,0,0.4)';g.fillRect(0,82,64,14);
      return c;
    }
    var vendMats=[['#d22a35','コーラ'],['#f2f2f4','ドリンク']].map(function(vd){
      var t=new THREE.CanvasTexture(vendCanvas(vd[0],vd[1]));
      return new THREE.MeshStandardMaterial({color:0xffffff,map:t,emissiveMap:t,emissive:0xffffff,emissiveIntensity:0.8,roughness:0.6,metalness:0.2});
    });
    var vendGeo=new THREE.BoxGeometry(0.95,1.7,0.85);
    for(var vi2=7;vi2<N;vi2+=35){
      var dstV=districtAt(vi2);if(!dstV)continue;
      [1,-1].forEach(function(sV){
        var siV=(sV>0)?0:1;
        if(!walled[siV][vi2])return;
        var wpV=waypoints[vi2],ppV=wpPerp(vi2);
        var vxV=wpV.x+ppV.x*sV*(dstV.face-1.3),vzV=wpV.z+ppV.z*sV*(dstV.face-1.3);
        if(inPitRect(vxV,vzV)||!clearsTrack(vxV,vzV,BH+0.3))return;
        var vm=new THREE.Mesh(vendGeo,vendMats[(vi2+siV)%2]);
        vm.position.set(vxV,wpV.y+0.85,vzV);
        vm.rotation.y=Math.atan2(-ppV.x*sV,-ppV.z*sV);scene.add(vm);
      });
    }
    // izakaya lantern strings (D2 only): warm paper-lantern rows on a shallow sag
    var lanGeo=new THREE.SphereGeometry(0.28,8,6);
    var lanMat=new THREE.MeshStandardMaterial({color:0xFF8C42,emissive:0xFF8C42,emissiveIntensity:1.6,roughness:0.6});
    var lanPole=new THREE.CylinderGeometry(0.07,0.09,3.2,5);
    for(var li=305;li<560;li+=45){
      var dstL=districtAt(li);if(!dstL)continue;
      var sL=((li/45)|0)%2===0?1:-1,siL=(sL>0)?0:1;
      if(!walled[siL][li])continue;
      var wpL=waypoints[li],ppL=wpPerp(li),ddL=wpDir(li);
      var lxL=wpL.x+ppL.x*sL*(dstL.face-3),lzL=wpL.z+ppL.z*sL*(dstL.face-3);
      if(inPitRect(lxL,lzL)||!clearsTrack(lxL,lzL,BH+0.3))continue;
      var pA=new THREE.Mesh(lanPole,mPole),pB=new THREE.Mesh(lanPole,mPole);
      pA.position.set(lxL-ddL.x*3,wpL.y+1.6,lzL-ddL.z*3);scene.add(pA);
      pB.position.set(lxL+ddL.x*3,wpL.y+1.6,lzL+ddL.z*3);scene.add(pB);
      for(var ln2=0;ln2<7;ln2++){
        var tL=ln2/6;
        var lan=new THREE.Mesh(lanGeo,lanMat);
        lan.position.set(lxL+ddL.x*(tL-0.5)*6,wpL.y+3.0-0.5*4*tL*(1-tL),lzL+ddL.z*(tL-0.5)*6);
        scene.add(lan);
      }
    }
    // kanji wayfinding gantries spanning the verge (wp 250 / 715), start-gantry family
    function wayBoard(){
      var c=document.createElement('canvas');c.width=512;c.height=128;var g=c.getContext('2d');
      g.fillStyle='#0b3d2e';g.fillRect(0,0,512,128);
      g.strokeStyle='rgba(255,255,255,0.85)';g.lineWidth=5;g.strokeRect(8,8,496,112);
      g.fillStyle='#ffffff';g.font='700 40px '+JFONT;g.textAlign='center';g.textBaseline='middle';
      g.fillText('新宿 Shinjuku ↑',256,40);
      g.font='700 34px '+JFONT;
      g.fillText('渋谷 Shibuya →',256,90);
      return new THREE.CanvasTexture(c);
    }
    var wbTex=wayBoard();
    var wbMat=new THREE.MeshStandardMaterial({color:0xffffff,map:wbTex,emissiveMap:wbTex,emissive:0xffffff,emissiveIntensity:0.7,roughness:0.6,metalness:0.2});
    [250,715].forEach(function(gwp){
      var wpG=waypoints[gwp],ppG=wpPerp(gwp),ddG=wpDir(gwp);
      var gOff=TW*0.5+CURB+BH+1.2;
      var gxA=wpG.x+ppG.x*gOff,gzA=wpG.z+ppG.z*gOff;
      var gxB=wpG.x-ppG.x*gOff,gzB=wpG.z-ppG.z*gOff;
      if(inPitRect(gxA,gzA)||inPitRect(gxB,gzB))return;
      var postG=new THREE.CylinderGeometry(0.22,0.3,10.5,7);
      var pgA=new THREE.Mesh(postG,mPole),pgB=new THREE.Mesh(postG,mPole);
      pgA.position.set(gxA,wpG.y+5.25,gzA);scene.add(pgA);
      pgB.position.set(gxB,wpG.y+5.25,gzB);scene.add(pgB);
      var bar=new THREE.Mesh(unit,mPole);
      bar.position.set(wpG.x,wpG.y+10.2,wpG.z);
      bar.scale.set(gOff*2,0.35,0.35);bar.rotation.y=Math.atan2(ddG.x,ddG.z);scene.add(bar);
      var board=new THREE.Mesh(unit,wbMat);
      board.position.set(wpG.x,wpG.y+8.9,wpG.z);
      board.scale.set(9,2.2,0.2);board.rotation.y=Math.atan2(ddG.x,ddG.z);scene.add(board);
    });
  })();

  (function(){
    // Distant Tokyo skyline: towers gathered into 6 ward-like clusters (not a uniform ring)
    // with floor-by-floor window façades, rooftop caps, aircraft-warning beacons and a few
    // holo billboards rising out of the blue-violet haze. A gap is left at the Tokyo Tower
    // azimuth so the landmark reads clean against the sky.
    function facadeCanvas(){
      // 128x256 with an 8x32 window grid (was 64x128 / 6x20 — single-digit-pixel windows
      // turned to mush when tiled up tall towers). Whole-pixel cell math keeps edges crisp.
      var c=document.createElement('canvas');c.width=128;c.height=256;var g=c.getContext('2d');
      g.fillStyle='#04050a';g.fillRect(0,0,128,256);
      var cols=8,rows=32,cw=128/cols,ch=256/rows;
      var warm=['#ffe6c0','#ffd9a0','#fff2dc'],neon=['#22d3ee','#e94fd8','#8b7bff','#ffb347'];
      var accent=neon[(Math.random()*neon.length)|0];
      for(var ry=0;ry<rows;ry++){
        if(Math.random()<0.05){ g.globalAlpha=0.9;g.fillStyle=accent;g.fillRect(3,ry*ch+3,122,2);g.globalAlpha=1;continue; }
        var dark=Math.random()<0.28;
        for(var cx=0;cx<cols;cx++){
          if(dark||Math.random()<0.42)continue;
          g.globalAlpha=0.5+Math.random()*0.5;
          g.fillStyle=(Math.random()<0.82)?warm[(Math.random()*warm.length)|0]:accent;
          g.fillRect(cx*cw+3,ry*ch+2,cw-6,ch-4);
        }
      }
      g.globalAlpha=1;return c;
    }
    // shared façade pool (6 textures: 3 patterns × slim/wide window density) instead of a
    // canvas+texture per tower — towers are distant and fogged, so a fixed per-bucket window
    // scale reads fine while saving ~66 canvas creations + GPU uploads at build time.
    function facadeTex(rx,ry){var t=new THREE.CanvasTexture(facadeCanvas());t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(rx,ry);t.anisotropy=renderer.capabilities.getMaxAnisotropy();return t;}
    function facadeMat(tx){return new THREE.MeshStandardMaterial({color:0x0a0b13,map:tx,emissiveMap:tx,emissive:0xffffff,emissiveIntensity:0.6,roughness:0.8,metalness:0.3});}
    var slimMat=[facadeTex(2,15),facadeTex(2,17),facadeTex(3,16),facadeTex(2,14),facadeTex(3,18)].map(facadeMat);
    var wideMat=[facadeTex(4,7),facadeTex(5,6),facadeTex(4,8),facadeTex(5,8),facadeTex(6,7)].map(facadeMat);
    var NB=64;
    var TOWER_AZ=Math.atan2(-160,520);
    var clusterAz=[0.45,1.42,2.30,3.35,4.30,5.35];
    for(var bi2=0;bi2<NB;bi2++){
      var ang=clusterAz[bi2%6]+(srnd(bi2*17+3)-0.5)*0.30;
      var azd=Math.abs(Math.atan2(Math.sin(ang-TOWER_AZ),Math.cos(ang-TOWER_AZ)));
      if(azd<0.12)continue;
      var rad=540+srnd(bi2*23+7)*180;
      var bxp=Math.cos(ang)*rad,bzp=Math.sin(ang)*rad;
      var slim=srnd(bi2*29+1)<0.55;
      var bw=slim?(15+srnd(bi2*31+2)*16):(34+srnd(bi2*31+2)*36);
      var bd=slim?(15+srnd(bi2*37+4)*16):(34+srnd(bi2*37+4)*36);
      var bh=slim?(95+srnd(bi2*41+5)*srnd(bi2*43+6)*220):(38+srnd(bi2*41+5)*85);
      var bMat=slim?slimMat[bi2%slimMat.length]:wideMat[bi2%wideMat.length];
      var bld=new THREE.Mesh(unit,bMat);
      bld.position.set(bxp,bh*0.5-4,bzp);bld.scale.set(bw,bh,bd);bld.rotation.y=ang;scene.add(bld);
      // dark rooftop cap (mechanical penthouse) for silhouette variety
      var capH=4+srnd(bi2*47+8)*9;
      var cap=new THREE.Mesh(unit,capMat);
      cap.position.set(bxp,bh-4+capH*0.5,bzp);cap.scale.set(bw*(0.55+srnd(bi2*53+9)*0.3),capH,bd*(0.55+srnd(bi2*59+10)*0.3));cap.rotation.y=ang;scene.add(cap);
      if(bh>150){
        var ant=new THREE.Mesh(new THREE.CylinderGeometry(0.4,1.0,20,5),mProp);
        ant.position.set(bxp,bh-4+capH+10,bzp);scene.add(ant);
        var beac=new THREE.Mesh(new THREE.SphereGeometry(1.8,8,6),beaconMat);
        beac.position.set(bxp,bh-4+capH+20,bzp);scene.add(beac);
        // Skyward light beam off the tallest roofs: a faint additive cone — the cyberpunk
        // "searchlight into the smog" read, cheap enough to leave static.
        if(srnd(bi2*67+12)<0.55){
          var beam=new THREE.Mesh(new THREE.CylinderGeometry(1.4,4.2,280,8,1,true),
            new THREE.MeshBasicMaterial({color:(bi2%2)?NEON_CYAN:TOKYO_VIOLET,transparent:true,opacity:0.055,blending:THREE.AdditiveBlending,depthWrite:false,side:THREE.DoubleSide,fog:false}));
          beam.position.set(bxp,bh-4+capH+140,bzp);
          beam.rotation.z=(srnd(bi2*71+13)-0.5)*0.16;
          scene.add(beam);
        }
      }
      if(srnd(bi2*61+11)<0.16){
        var hcol=NEON_CYCLE[bi2%NEON_CYCLE.length];
        var hy=bh*0.5-4;
        var holo=new THREE.Mesh(new THREE.PlaneGeometry(Math.min(bw,bd)*0.9,bh*0.4),new THREE.MeshBasicMaterial({color:hcol,transparent:true,opacity:0.35,side:THREE.DoubleSide,fog:false}));
        var inset=Math.max(bw,bd)*0.5+1.5;
        holo.position.set(bxp-Math.cos(ang)*inset,hy,bzp-Math.sin(ang)*inset);
        holo.lookAt(0,hy,0);scene.add(holo);
      }
    }
  })();
  (function(){
    // Tokyo Tower: red/white lattice landmark on the skyline at the reserved azimuth —
    // 4 tilted legs, cross-brace rings, two observation decks and a beaconed mast. Emissive
    // tints keep it readable through the fog without real lights.
    var TX=520,TZ=-160,TY=-3;
    var mOr=new THREE.MeshStandardMaterial({color:0xE8492A,emissive:0xE8492A,emissiveIntensity:0.5,roughness:0.6,metalness:0.3});
    var mWh=new THREE.MeshStandardMaterial({color:0xF4F1E8,emissive:0xF4F1E8,emissiveIntensity:0.35,roughness:0.6,metalness:0.2});
    var legGeo=new THREE.CylinderGeometry(1.6,3.2,126,6);
    [[1,1],[1,-1],[-1,1],[-1,-1]].forEach(function(lc){
      var x0=TX+lc[0]*34,z0=TZ+lc[1]*34,x1=TX+lc[0]*7,z1=TZ+lc[1]*7;
      var leg=new THREE.Mesh(legGeo,mOr);
      leg.position.set((x0+x1)/2,TY+60,(z0+z1)/2);
      var dy=120,dxl=x1-x0,dzl=z1-z0;
      leg.rotation.z=Math.atan2(dxl,dy);leg.rotation.x=-Math.atan2(dzl,Math.sqrt(dy*dy+dxl*dxl));
      scene.add(leg);
    });
    for(var br=0;br<4;br++){
      var by=TY+24+br*24,hwB=30-(br*22/3);
      var brMat=(br%2===0)?mWh:mOr;
      [[TX,TZ-hwB,hwB*2,1],[TX,TZ+hwB,hwB*2,1],[TX-hwB,TZ,1,hwB*2],[TX+hwB,TZ,1,hwB*2]].forEach(function(ed){
        var brc=new THREE.Mesh(unit,brMat);
        brc.position.set(ed[0],by,ed[1]);
        brc.scale.set(ed[2],1.0,ed[3]);
        scene.add(brc);
      });
    }
    var deck=new THREE.Mesh(unit,mWh);
    deck.position.set(TX,TY+125,TZ);deck.scale.set(22,9,22);scene.add(deck);
    var deckBand=new THREE.Mesh(unit,mOr);
    deckBand.position.set(TX,TY+129,TZ);deckBand.scale.set(23,1.6,23);scene.add(deckBand);
    var mast1=new THREE.Mesh(new THREE.CylinderGeometry(4.5,7,54,6),mOr);
    mast1.position.set(TX,TY+156,TZ);scene.add(mast1);
    var deck2=new THREE.Mesh(unit,mWh);
    deck2.position.set(TX,TY+183,TZ);deck2.scale.set(10,5,10);scene.add(deck2);
    var mast2=new THREE.Mesh(new THREE.CylinderGeometry(0.9,2.2,40,5),mOr);
    mast2.position.set(TX,TY+205,TZ);scene.add(mast2);
    var tBeac=new THREE.Mesh(new THREE.SphereGeometry(1.6,8,6),new THREE.MeshBasicMaterial({color:0xff2030,fog:false}));
    tBeac.position.set(TX,TY+226,TZ);scene.add(tBeac);
  })();
  (function(){
    // Giant vermilion torii gate spanning the track in the izakaya district (wp 352, arc
    // reserved in the mask). Legs sit outside the barriers, lowest beam ~8.6u over the road.
    var gwp=352,wpT=waypoints[gwp],ppT=wpPerp(gwp),ddT=wpDir(gwp);
    var rotT=Math.atan2(ddT.x,ddT.z);
    var legOff=TW*0.5+CURB+BH+2;
    var mVer=new THREE.MeshStandardMaterial({color:0xD93A2B,emissive:0xFF4830,emissiveIntensity:0.85,roughness:0.55,metalness:0.15});
    var mInk=new THREE.MeshStandardMaterial({color:0x14161c,roughness:0.7,metalness:0.3});
    var legG=new THREE.CylinderGeometry(1.1,1.4,12.5,10);
    [1,-1].forEach(function(sT){
      var leg=new THREE.Mesh(legG,mVer);
      leg.position.set(wpT.x+ppT.x*sT*legOff,wpT.y+6.25,wpT.z+ppT.z*sT*legOff);
      scene.add(leg);
    });
    var nuki=new THREE.Mesh(unit,mVer);
    nuki.position.set(wpT.x,wpT.y+9.2,wpT.z);
    nuki.scale.set(legOff*2+2,1.2,1.1);nuki.rotation.y=rotT;scene.add(nuki);
    var shimaki=new THREE.Mesh(unit,mInk);
    shimaki.position.set(wpT.x,wpT.y+11.4,wpT.z);
    shimaki.scale.set(legOff*2+3,1.0,1.4);shimaki.rotation.y=rotT;scene.add(shimaki);
    var kasagi=new THREE.Mesh(unit,mVer);
    kasagi.position.set(wpT.x,wpT.y+12.6,wpT.z);
    kasagi.scale.set(legOff*2+7.2,1.5,1.6);kasagi.rotation.y=rotT;scene.add(kasagi);
    var plq=document.createElement('canvas');plq.width=64;plq.height=96;var pg=plq.getContext('2d');
    pg.fillStyle='#101216';pg.fillRect(0,0,64,96);
    pg.strokeStyle='#c8a44a';pg.lineWidth=4;pg.strokeRect(4,4,56,88);
    pg.fillStyle='#f2ead8';pg.font='900 34px '+JFONT;pg.textAlign='center';pg.textBaseline='middle';
    pg.fillText('東',32,30);pg.fillText('京',32,66);
    var plqTex=new THREE.CanvasTexture(plq);
    var plaque=new THREE.Mesh(unit,new THREE.MeshStandardMaterial({color:0xffffff,map:plqTex,emissiveMap:plqTex,emissive:0xffffff,emissiveIntensity:0.6,roughness:0.6}));
    plaque.position.set(wpT.x,wpT.y+10.4,wpT.z);
    plaque.scale.set(1.4,1.8,0.5);plaque.rotation.y=rotT;scene.add(plaque);
  })();
  (function(){
    // Shibuya media wall: a 12-floor host tower at wp 605 carrying a giant animated screen.
    // Four pre-rendered frames cycle by swapping map/emissiveMap on the already-uploaded
    // textures — same no-needsUpdate trick as the sponsor boards (avoids shader recompiles).
    var mwp=605,sM=1,wpM=waypoints[mwp],ppM=wpPerp(mwp),ddM=wpDir(mwp);
    var dstM={face:25,w0:30,w1:30,f0:12,f1:12,depth:14,sg:0};
    var offM=dstM.face+dstM.depth*0.5;
    var bxM=wpM.x+ppM.x*sM*offM,bzM=wpM.z+ppM.z*sM*offM;
    placeBuilding(bxM,bzM,Math.atan2(ddM.x,ddM.z),30,dstM,mwp,sM);
    for(var qm=mwp-9;qm<=mwp+9;qm++)walled[0][((qm%N)+N)%N]=1;
    function frame(draw){
      var c=document.createElement('canvas');c.width=512;c.height=288;var g=c.getContext('2d');
      g.fillStyle='#060608';g.fillRect(0,0,512,288);
      draw(g);
      g.fillStyle='rgba(0,0,0,0.22)';
      for(var sl=0;sl<288;sl+=6)g.fillRect(0,sl,512,2);
      return new THREE.CanvasTexture(c);
    }
    var frames=[
      frame(function(g){
        g.fillStyle='#e94fd8';g.font='900 120px '+JFONT;g.textAlign='center';g.textBaseline='middle';
        g.fillText('トーキョー',256,144);
        g.strokeStyle='rgba(233,79,216,0.6)';g.lineWidth=6;g.strokeRect(16,16,480,256);
      }),
      frame(function(g){
        g.fillStyle='#0A1530';g.fillRect(0,0,512,288);
        g.fillStyle='#D7202E';
        for(var ch=0;ch<5;ch++){g.beginPath();g.moveTo(40+ch*90,250);g.lineTo(85+ch*90,60);g.lineTo(130+ch*90,250);g.lineTo(108+ch*90,250);g.lineTo(85+ch*90,140);g.lineTo(62+ch*90,250);g.closePath();g.fill();}
        g.fillStyle='#ffffff';g.font='900 54px Arial,sans-serif';g.textAlign='center';
        g.fillText('RED BULL RACING',256,52);
      }),
      frame(function(g){
        g.strokeStyle='#22d3ee';g.lineWidth=10;
        for(var wv=0;wv<4;wv++){
          g.beginPath();
          for(var wx2=0;wx2<=512;wx2+=16)g.lineTo(wx2,150+wv*26-60*Math.sin(wx2/512*Math.PI*2+wv*0.8));
          g.stroke();
        }
        g.fillStyle='#ffffff';g.font='900 88px '+JFONT;g.textAlign='center';g.textBaseline='middle';
        g.fillText('東京GP',256,84);
      }),
      frame(function(g){
        g.fillStyle='#f2ead8';g.fillRect(0,0,512,288);
        g.fillStyle='#d22a35';
        for(var ra=0;ra<12;ra++){
          var a0=ra/12*Math.PI*2,a1=a0+0.13;
          g.beginPath();g.moveTo(256,144);
          g.lineTo(256+Math.cos(a0)*400,144+Math.sin(a0)*400);
          g.lineTo(256+Math.cos(a1)*400,144+Math.sin(a1)*400);
          g.closePath();g.fill();
        }
        g.beginPath();g.arc(256,144,58,0,6.283);g.fill();
      })
    ];
    // Pre-upload all four frames so the first swap cycle does not hitch on a GPU upload.
    frames.forEach(function(t){if(renderer.initTexture)renderer.initTexture(t);});
    var scrMat=new THREE.MeshStandardMaterial({color:0x111111,map:frames[0],emissiveMap:frames[0],emissive:0xffffff,emissiveIntensity:1.25,roughness:0.4,metalness:0.1});
    var scr=new THREE.Mesh(new THREE.PlaneGeometry(24,13),scrMat);
    var scrOff=dstM.face-0.4;
    scr.position.set(wpM.x+ppM.x*sM*scrOff,wpM.y+13.5,wpM.z+ppM.z*sM*scrOff);
    scr.rotation.y=Math.atan2(-ppM.x*sM,-ppM.z*sM);
    scene.add(scr);
    var scrFrame=new THREE.Mesh(unit,new THREE.MeshStandardMaterial({color:0x0c0d12,roughness:0.6,metalness:0.5}));
    // Frame box sits 0.75 behind the screen centre so its front face (24.85) clears the
    // screen plane (24.6) — they were exactly coplanar before, which z-fought and flickered.
    scrFrame.position.set(wpM.x+ppM.x*sM*(scrOff+0.75),wpM.y+13.5,wpM.z+ppM.z*sM*(scrOff+0.75));
    scrFrame.scale.set(1.0,14.4,25.4);scrFrame.rotation.y=Math.atan2(ddM.x,ddM.z);scene.add(scrFrame);
    var scrTick=0;
    setInterval(function(){
      scrTick=(scrTick+1)%frames.length;
      scrMat.map=frames[scrTick];scrMat.emissiveMap=frames[scrTick];
    },700);
  })();

  var mk2=function(geo,mat,x,y,z){var m=new THREE.Mesh(geo,mat);m.position.set(x||0,y||0,z||0);return m;};
  var mConc2=new THREE.MeshStandardMaterial({color:0x141016,roughness:0.8,metalness:0.2});
  var mRoof2=new THREE.MeshStandardMaterial({color:0x18141e,roughness:0.5,metalness:0.5});
  var crowdCols=[0xcc1111,0x1133cc,0xddcc00,0xffffff,0x118833,0xcc6600,0x880099,0x009988];
  crowdMats=crowdCols.map(function(hx){return new THREE.MeshStandardMaterial({color:hx,emissive:hx,emissiveIntensity:0.0,roughness:1});});
  function buildGrandstand(wpIdx,side,width,seatHex){
    var wp2=waypoints[wpIdx],p2=wpPerp(wpIdx);
    var ry2=Math.atan2(p2.x*side,p2.z*side);
    var c2=Math.cos(ry2),s2=Math.sin(ry2);
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
    var mS2=new THREE.MeshStandardMaterial({color:seatHex,emissive:seatHex,emissiveIntensity:0.55,roughness:0.6});
    var mLED=new THREE.MeshStandardMaterial({color:seatHex,emissive:seatHex,emissiveIntensity:1.8,roughness:0.4});
    var mAd2=new THREE.MeshStandardMaterial({color:NEON_RED,emissive:NEON_RED,emissiveIntensity:1.8,roughness:0.5});
    var g2=new THREE.Group();
    g2.add(mk2(new THREE.BoxGeometry(width,4,9  ),mConc2,0,2,   -1.5));
    g2.add(mk2(new THREE.BoxGeometry(width,3,7  ),mConc2,0,5.5,  3.5));
    g2.add(mk2(new THREE.BoxGeometry(width,2.5,5),mConc2,0,8.25, 7.5));
    g2.add(mk2(new THREE.BoxGeometry(width-1,0.3,8.5),mS2,0,4.15,-1.5));
    g2.add(mk2(new THREE.BoxGeometry(width-1,0.3,6.5),mS2,0,7.05, 3.5));
    g2.add(mk2(new THREE.BoxGeometry(width-1,0.3,4.5),mS2,0,9.50, 7.5));
    g2.add(mk2(new THREE.BoxGeometry(width+4,0.5,15),mRoof2,0,11.5,3));
    // neon LED edge strips along the roof lip + step fronts so the stand reads as a lit terrace
    g2.add(mk2(new THREE.BoxGeometry(width+4,0.22,0.22),mLED,0,11.25,10.4));
    g2.add(mk2(new THREE.BoxGeometry(width+4,0.22,0.22),mLED,0,11.25,-4.4));
    g2.add(mk2(new THREE.BoxGeometry(width-1,0.12,0.12),mLED,0,4.35,-5.7));
    g2.add(mk2(new THREE.BoxGeometry(width-1,0.12,0.12),mLED,0,7.25,0.3));
    [-(width*0.5-2),width*0.5-2].forEach(function(cx){
      g2.add(mk2(new THREE.BoxGeometry(0.8,11.5,0.8),mConc2,cx,5.75,3));
      g2.add(mk2(new THREE.BoxGeometry(0.26,11.5,0.26),mLED,cx,5.75,3.5));
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
          crowdFolk.push({m:cm,by:cm.position.y,ph:cx3*0.22});
        }
      });
    });
    cg.position.set(ox2,wp2.y,oz2);cg.rotation.y=ry2;scene.add(cg);
    standAnchors.push({x:ox2,y:wp2.y+9,z:oz2,w:width*0.5});
  }
  buildGrandstand( 10,+1,70,NEON_RED);
  buildGrandstand( 10,-1,55,NEON_CYAN);
  buildGrandstand(130,+1,50,NEON_AMBER);
  buildGrandstand(280,-1,45,NEON_MAG);
  buildGrandstand(450,+1,60,NEON_CYAN);
  buildGrandstand(640,-1,40,NEON_RED);
  buildGrandstand(790,+1,50,NEON_AMBER);

  (function(){
    var poleMat=new THREE.MeshStandardMaterial({color:0x0c0c10,roughness:0.5,metalness:0.6});
    var headMat=new THREE.MeshStandardMaterial({color:0x101014,roughness:0.6,metalness:0.4});
    var lampGeo=new THREE.BoxGeometry(1.4,1.4,0.4);
    var towerWps=[40,150,260,370,480,590,700,810];
    for(var fi=0;fi<towerWps.length;fi++){
      var wpI=towerWps[fi],side=(fi%2===0)?1:-1;
      var fcol=NEON_CYCLE[fi%NEON_CYCLE.length];
      var lampMat=new THREE.MeshStandardMaterial({color:fcol,emissive:fcol,emissiveIntensity:2.6,roughness:0.4});
      var fwp=waypoints[wpI],fp=wpPerp(wpI);
      var fdist=TW*0.5+CURB+BH+34;
      var fbx=fwp.x+fp.x*side*fdist,fbz=fwp.z+fp.z*side*fdist;
      // Keep floodlights out of the pit complex (the wp40 tower landed in the paddock).
      if(inPitRect(fbx,fbz)){side=-side;fbx=fwp.x+fp.x*side*fdist;fbz=fwp.z+fp.z*side*fdist;}
      var poleH=27;
      var pole=new THREE.Mesh(new THREE.CylinderGeometry(0.45,0.75,poleH,8),poleMat);
      pole.position.set(fbx,fwp.y+poleH*0.5,fbz);scene.add(pole);
      var pstrip=new THREE.Mesh(new THREE.BoxGeometry(0.18,poleH-1,0.18),lampMat);
      pstrip.position.set(fbx,fwp.y+poleH*0.5,fbz);scene.add(pstrip);
      var faceY=Math.atan2(-fp.x*side,-fp.z*side);
      var head=new THREE.Group();
      head.add(new THREE.Mesh(new THREE.BoxGeometry(6.2,3.0,0.7),headMat));
      for(var lr=0;lr<2;lr++)for(var lc=0;lc<4;lc++){
        var lamp=new THREE.Mesh(lampGeo,lampMat);
        lamp.position.set((lc-1.5)*1.45,(lr-0.5)*1.35,0.45);head.add(lamp);
      }
      head.position.set(fbx,fwp.y+poleH+0.6,fbz);head.rotation.y=faceY;head.rotation.x=0.34;
      scene.add(head);
      var coneH=poleH+10,vc=new THREE.ConeGeometry(15,coneH,16,1,true);
      vc.translate(0,-coneH*0.5,0);
      var vcone=new THREE.Mesh(vc,new THREE.MeshBasicMaterial({color:fcol,transparent:true,opacity:0.06,side:THREE.DoubleSide,depthWrite:false,blending:THREE.AdditiveBlending,fog:false}));
      vcone.position.set(fbx,fwp.y+poleH+0.4,fbz);
      var vdir=new THREE.Vector3(fwp.x-fbx,fwp.y-(fwp.y+poleH+0.4),fwp.z-fbz).normalize();
      vcone.quaternion.setFromUnitVectors(new THREE.Vector3(0,-1,0),vdir);
      scene.add(vcone);
    }
  })();

  (function(){
    var wp0=waypoints[0],p0=wpPerp(0);
    var hw=TW*0.5+CURB+BH+1.0;
    var mSt=new THREE.MeshStandardMaterial({color:0x14141a,roughness:0.4,metalness:0.6});
    var mBl=new THREE.MeshStandardMaterial({color:0x101018,roughness:0.5,metalness:0.4});
    var mTrim=new THREE.MeshStandardMaterial({color:NEON_CYAN,emissive:NEON_CYAN,emissiveIntensity:1.8,roughness:0.4});
    var mLt=new THREE.MeshStandardMaterial({color:0xff2200,emissive:0xff2200,emissiveIntensity:2.4,roughness:0.4});
    // The leg on the pit side stood at z~57, mid pit-lane corridor — push it past the garage
    // line so the gantry bridges the whole pit complex and the lane runs under it untouched.
    var ps=(p0.z>0)?1:-1;
    function legOff(s){return (s===ps)?hw+29:hw;}
    [1,-1].forEach(function(s){
      var o=legOff(s);
      var tx=wp0.x+p0.x*s*o,tz=wp0.z+p0.z*s*o;
      var gt=new THREE.Mesh(new THREE.BoxGeometry(1.2,18,1.2),mSt);
      gt.position.set(tx,wp0.y+9,tz);scene.add(gt);
    });
    var lx=wp0.x+p0.x*legOff(1),lz=wp0.z+p0.z*legOff(1);
    var rx=wp0.x-p0.x*legOff(-1),rz=wp0.z-p0.z*legOff(-1);
    var barLen=Math.sqrt((rx-lx)*(rx-lx)+(rz-lz)*(rz-lz))+1.2;
    var bar=new THREE.Mesh(new THREE.BoxGeometry(1.4,1.6,barLen),mBl);
    bar.position.set((lx+rx)*0.5,wp0.y+18,(lz+rz)*0.5);
    bar.rotation.y=Math.atan2(rx-lx,rz-lz);scene.add(bar);
    var trim=new THREE.Mesh(new THREE.BoxGeometry(0.3,0.3,barLen),mTrim);
    trim.position.set((lx+rx)*0.5,wp0.y+18.95,(lz+rz)*0.5);trim.rotation.y=bar.rotation.y;scene.add(trim);
    var trim2=new THREE.Mesh(new THREE.BoxGeometry(0.3,0.3,barLen),mTrim);
    trim2.position.set((lx+rx)*0.5,wp0.y+17.05,(lz+rz)*0.5);trim2.rotation.y=bar.rotation.y;scene.add(trim2);
    for(var gli=0;gli<5;gli++){
      var lt=(gli/4-0.5)*hw*1.6;
      var gl=new THREE.Mesh(new THREE.BoxGeometry(0.7,0.7,0.35),mLt);
      gl.position.set(wp0.x+p0.x*lt,wp0.y+16.5,wp0.z+p0.z*lt);scene.add(gl);
    }
  })();
  (function(){
    // Pit-lane surfaces vs the racing surface (track half-width + curb of the centerline):
    // painted lines are clipped wherever they'd land within 1u of it, and the asphalt ribbon —
    // which IS the entry/exit road and must stay full-length — is laid slightly BELOW the road
    // surface (road y+0.06, curbs y+0.07), so wherever the lane still overlaps the racing
    // surface at the merge ends, the track and curb simply cover it instead of a different-
    // shade wedge carpeting across the corner exit.
    var EDGE=TW*0.5+CURB;
    function onTrackAt(x,z){
      var bd2=1e9;
      for(var ow=0;ow<N;ow++){
        var odx=waypoints[ow].x-x,odz=waypoints[ow].z-z,od2=odx*odx+odz*odz;
        if(od2<bd2)bd2=od2;
      }
      return Math.sqrt(bd2)<EDGE+1.0;
    }
    function ribbon(off,half,y,mat,clipTrack){
      var pos=[],on=[];
      for(var i=0;i<=PIT_SEGS;i++){
        var a=pitPts[i],b=pitPts[Math.min(PIT_SEGS,i+1)];
        var dx=b.x-a.x,dz=b.z-a.z,L=Math.sqrt(dx*dx+dz*dz)||1,nx=dz/L,nz=-dx/L;
        var cx=a.x+nx*off,cz=a.z+nz*off;
        pos.push(cx+nx*half,y,cz+nz*half,cx-nx*half,y,cz-nz*half);
        on.push(clipTrack?onTrackAt(cx,cz):false);
      }
      // CCW winding from above so face normals point +Y (visible from above) — matches the main road.
      var tri=[];
      for(var q=0;q<PIT_SEGS;q++){
        if(on[q]||on[q+1])continue;
        var bb=q*2;tri.push(bb,bb+1,bb+2,bb+2,bb+1,bb+3);
      }
      var g=new THREE.BufferGeometry();
      g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setIndex(tri);g.computeVertexNormals();
      var me=new THREE.Mesh(g,mat);me.receiveShadow=true;scene.add(me);
    }
    var mAsph=new THREE.MeshStandardMaterial({color:0x202024,roughness:0.82,metalness:0.05});
    var apW=(PIT_X1-PIT_X0)+60,apCx=(PIT_X0+PIT_X1)/2,apZ0=PIT_Z-PIT_W*0.5-1,apZ1=GARAGE_Z-6.5,apCz=(apZ0+apZ1)/2;
    var apron=new THREE.Mesh(new THREE.BoxGeometry(apW,0.08,apZ1-apZ0),mAsph);apron.position.set(apCx,0.0,apCz);apron.receiveShadow=true;scene.add(apron);
    ribbon(0,PIT_W*0.5,0.03,mAsph,false);
    var mLine=new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xffffff,emissiveIntensity:0.3,roughness:0.6});
    ribbon(PIT_W*0.5-0.2,0.16,0.07,mLine,true);ribbon(-(PIT_W*0.5-0.2),0.16,0.07,mLine,true);
    // Pit wall stops ~14 units short of the exit (PIT_X1) so it never juts into the merge corridor
    // where cars rejoin the racing line — fixes the "wall stuck at the pit exit" snag.
    var wStartX=PIT_X0-10,wEndX=PIT_X1-14,wlen=wEndX-wStartX,wcx=(wStartX+wEndX)/2,wz=PIT_Z-PIT_W*0.5-0.4;
    var pwNeon=new THREE.MeshStandardMaterial({color:NEON_RED,emissive:NEON_RED,emissiveIntensity:1.6,roughness:0.4});
    var pw=new THREE.Mesh(new THREE.BoxGeometry(wlen,1.3,0.5),new THREE.MeshStandardMaterial({color:0x18181e,roughness:0.4,metalness:0.4}));pw.position.set(wcx,0.65,wz);pw.castShadow=true;scene.add(pw);
    var pwt=new THREE.Mesh(new THREE.BoxGeometry(wlen,0.12,0.54),pwNeon);pwt.position.set(wcx,1.35,wz);scene.add(pwt);
    var slm=new THREE.Mesh(new THREE.BoxGeometry(0.4,0.06,PIT_W),new THREE.MeshStandardMaterial({color:0xffcc00,emissive:0xffcc00,emissiveIntensity:0.5,roughness:0.6}));slm.position.set(PIT_X0,0.06,PIT_Z);scene.add(slm);
    // Pit entry signage: neon "PIT" board on a mast where the entry road diverges, plus angled
    // chevron dashes on the apron funnel, so steering in reads as a deliberate choice.
    (function(){
      var sc=document.createElement('canvas');sc.width=256;sc.height=96;var sg=sc.getContext('2d');
      sg.fillStyle='#0a0508';sg.fillRect(0,0,256,96);
      sg.strokeStyle='rgba(255,80,95,0.85)';sg.lineWidth=4;sg.strokeRect(6,6,244,84);
      sg.fillStyle='#FF3B4E';sg.textAlign='center';sg.textBaseline='middle';
      sg.font='900 58px Arial,sans-serif';sg.fillText('PIT →',128,52);
      var st=new THREE.CanvasTexture(sc);
      var sm=new THREE.MeshStandardMaterial({color:0x14060a,map:st,emissiveMap:st,emissive:0xffffff,emissiveIntensity:1.4,roughness:0.5,metalness:0.3});
      var board=new THREE.Mesh(new THREE.BoxGeometry(7,2.6,0.3),sm);
      board.position.set(-291,5.2,62);board.rotation.y=Math.PI+0.26;scene.add(board);
      var pole=new THREE.Mesh(new THREE.CylinderGeometry(0.18,0.22,4.2,8),new THREE.MeshStandardMaterial({color:0x22232a,roughness:0.5,metalness:0.6}));
      pole.position.set(-291,2.0,62);scene.add(pole);
      var mDash=new THREE.MeshStandardMaterial({color:0xffcc00,emissive:0xffcc00,emissiveIntensity:0.55,roughness:0.6});
      [[-278,52.6],[-271,54.4],[-264,56.1]].forEach(function(dp){
        var d=new THREE.Mesh(new THREE.BoxGeometry(2.6,0.06,0.5),mDash);
        d.position.set(dp[0],0.07,dp[1]);d.rotation.y=-0.26;scene.add(d);
      });
    })();
    // Pit control tower relocated to centre of the pit straight, well behind the garage line
    // (was at the exit at PIT_X1+24, crowding cars rejoining the track).
    var twr=new THREE.MeshStandardMaterial({color:0x14141c,roughness:0.6,metalness:0.4});
    var twrX=(PIT_X0+PIT_X1)/2,twrZ=GARAGE_Z+18;
    var t1=new THREE.Mesh(new THREE.BoxGeometry(9,22,7),twr);t1.position.set(twrX,15,twrZ);t1.castShadow=true;scene.add(t1);
    var t2=new THREE.Mesh(new THREE.BoxGeometry(8,5,6.2),new THREE.MeshStandardMaterial({color:0x1a0008,emissive:NEON_CYAN,emissiveIntensity:1.3,roughness:0.3,metalness:0.6}));t2.position.set(twrX,28,twrZ);scene.add(t2);
    var gm={conc:new THREE.MeshStandardMaterial({color:0x16161c,roughness:0.7,metalness:0.2}),
            roof:new THREE.MeshStandardMaterial({color:0x101014,roughness:0.8,metalness:0.3}),
            metal:new THREE.MeshStandardMaterial({color:0x33343a,roughness:0.4,metalness:0.6}),
            screen:new THREE.MeshStandardMaterial({color:0x1a0008,emissive:NEON_RED,emissiveIntensity:1.7,roughness:0.4}),
            tyre:new THREE.MeshStandardMaterial({color:0x111111,roughness:0.95}),line:mLine};
    for(var ti=0;ti<PIT_TEAMS.length;ti++){
      var cfg=PIT_TEAMS[ti],bx=BOX_X[ti];
      buildGarage(cfg,bx,gm);
      var crew=buildPitCrew(cfg.col,bx);scene.add(crew.grp);
      teamCrew[cfg.id]=crew;
      buildPitWallStand(cfg,bx);
      [-9,0.5,9].forEach(function(ox,qi){
        var pn=pitPerson(cfg.col,false);
        var baseRot=Math.PI+(qi===0?-0.4:qi===1?0.0:0.5);
        pn.grp.position.set(bx+ox,0,GARAGE_Z-7.5);pn.grp.rotation.y=baseRot;scene.add(pn.grp);
        idleFolk.push({head:pn.head,grp:pn.grp,mode:'stand',ph:Math.random()*6.28,baseRot:baseRot});
      });
      var trolley=new THREE.Group();
      var trM=new THREE.MeshStandardMaterial({color:0x2a2d33,roughness:0.6,metalness:0.4});
      var trBase=new THREE.Mesh(new THREE.BoxGeometry(1.5,0.2,0.85),trM);trBase.position.y=0.45;trolley.add(trBase);
      var trH=new THREE.Mesh(new THREE.CylinderGeometry(0.04,0.04,1.1,6),trM);trH.position.set(-0.7,0.95,0);trH.rotation.z=-0.5;trolley.add(trH);
      var trW=new THREE.Mesh(new THREE.CylinderGeometry(0.5,0.5,0.34,16),gm.tyre);trW.rotation.x=Math.PI/2;trW.position.set(0.2,0.95,0);trolley.add(trW);
      // Trolley sits behind the box line, clear of the lane (was at PIT_BOX_Z-1.4, inside it).
      trolley.position.set(bx-6.5,0,PIT_BOX_Z+0.5);scene.add(trolley);
    }
    // Red marker posts beside each box, behind the box line — clear of the lane's outer edge.
    var redPost=new THREE.MeshStandardMaterial({color:0xcc1c1c,roughness:0.5});
    for(var fp=0;fp<PIT_TEAMS.length;fp++){
      var ext=new THREE.Mesh(new THREE.CylinderGeometry(0.18,0.18,0.7,10),redPost);
      ext.position.set(BOX_X[fp]+5.2,0.55,PIT_BOX_Z+0.6);ext.castShadow=true;scene.add(ext);
    }
    // Cones removed entirely: the exit pair clipped cars rejoining the track and the entrance
    // pair (z=54.1) sat inside the lane's inner edge — the lane reads clean without them.
  })();
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
    function _bbAmex(ctx){
      ctx.fillStyle='#016FD0';ctx.fillRect(0,0,CW,CH);
      ctx.fillStyle='rgba(255,255,255,0.08)';
      ctx.beginPath();ctx.moveTo(CW*0.62,0);ctx.lineTo(CW,0);ctx.lineTo(CW*0.84,CH);ctx.lineTo(CW*0.46,CH);ctx.fill();
      ctx.strokeStyle='#ffffff';ctx.lineWidth=9;ctx.strokeRect(40,34,CH-68,CH-68);
      ctx.fillStyle='#ffffff';ctx.font='900 46px Arial,sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('AMEX',40+(CH-68)*0.5,CH*0.5);
      ctx.font='900 78px Arial,sans-serif';ctx.textAlign='left';
      ctx.fillText('AMERICAN EXPRESS',CH+40,CH*0.40);
      ctx.fillStyle='rgba(255,255,255,0.6)';ctx.font='400 34px Arial,sans-serif';
      ctx.fillText('OFFICIAL PARTNER OF FORMULA 1',CH+40,CH*0.75);
    }
    function _bbLV(ctx){
      var g=ctx.createLinearGradient(0,0,CW,0);
      g.addColorStop(0,'#33241b');g.addColorStop(0.5,'#52392a');g.addColorStop(1,'#33241b');
      ctx.fillStyle=g;ctx.fillRect(0,0,CW,CH);
      ctx.strokeStyle='rgba(200,164,74,0.18)';ctx.lineWidth=2;
      for(var dg=-CH;dg<CW;dg+=72){ctx.beginPath();ctx.moveTo(dg,0);ctx.lineTo(dg+CH,CH);ctx.stroke();}
      ctx.fillStyle='#C8A44A';ctx.font='900 120px Georgia,serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('LV',CW*0.11,CH*0.48);
      ctx.font='bold 84px Georgia,serif';ctx.fillText('LOUIS VUITTON',CW*0.57,CH*0.40);
      ctx.fillStyle='rgba(200,164,74,0.62)';ctx.font='400 32px Georgia,serif';
      ctx.fillText('MALLETIER \\u00B7 MAISON FONDEE EN 1854',CW*0.57,CH*0.76);
    }
    var adDrawers=[_bbOracle,_bbF1,_bbMcLaren,_bbPetronas,_bbPirelli,_bbAramco,_bbAmex,_bbLV];
    var adEmissive=[0x0a1830,0x500010,0x3a2000,0x003533,0x1a1400,0x002018,0x002a50,0x241409];
    var bbTextures=[],bbMats=[];
    adDrawers.forEach(function(fn,idx){
      var cv=document.createElement('canvas');cv.width=CW;cv.height=CH;
      fn(cv.getContext('2d'));
      var tex=new THREE.CanvasTexture(cv);tex.anisotropy=4;
      bbTextures.push(tex);
      bbMats.push(new THREE.MeshStandardMaterial({
        map:tex,emissiveMap:tex,
        emissive:new THREE.Color(adEmissive[idx]),
        emissiveIntensity:0.9,roughness:0.55,metalness:0.05,side:THREE.DoubleSide
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
        // sponsor boards only where no street wall landed (plaza, grandstand/tower gaps) —
        // a board in front of a facade would clip it and double-clutter the canyon
        if(walled[(bs>0)?0:1][bi])return;
        var bx=bwp.x+bp.x*bs*bOff,bz=bwp.z+bp.z*bs*bOff;
        if(inPitRect(bx,bz)||!clearsTrack(bx,bz,BH+0.5))return;
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
          // swap to an already-uploaded texture + change a uniform: no material.needsUpdate
          // here on purpose — setting it forces a shader recompile every interval, which
          // stalls the main thread (and clicks the audio graph) ~twice a second.
          bb.slot=(bb.slot+1)%adDrawers.length;
          bb.mat.map=bbTextures[bb.slot];
          bb.mat.emissiveMap=bbTextures[bb.slot];
          bb.mat.emissive.set(adEmissive[bb.slot]);
        }
      });
    },500);
  })();
  (function(){
    var mFP=new THREE.MeshStandardMaterial({color:0x777777,roughness:0.5,metalness:0.45});
    var fGeo=new THREE.CylinderGeometry(0.07,0.07,5,5);
    var fOff=TW*0.5+CURB+BH+1.8;
    for(var fi=0;fi<N;fi+=5){
      var fwp=waypoints[fi],fp=wpPerp(fi);
      [-1,1].forEach(function(fs){
        var fmx=fwp.x+fp.x*fs*fOff,fmz=fwp.z+fp.z*fs*fOff;
        if(inPitRect(fmx,fmz)||!clearsTrack(fmx,fmz,BH))return;
        var fm=new THREE.Mesh(fGeo,mFP);
        fm.position.set(fmx,fwp.y+2.5,fmz);
        scene.add(fm);
      });
    }
  })();
  // --- Fan life: neon event tents in the plaza gaps and ~80 walking spectators rendered as
  // two InstancedMesh draws (body + head). Walkers ping-pong along the sidewalk strip near
  // the grandstands; a share of them mill in circles around the tents.
  (function(){
    var standDefs=[[10,1],[10,-1],[130,1],[280,-1],[450,1],[640,-1],[790,1]];
    var standW=[70,55,50,45,60,40,50];
    var sideOff=TW*0.5+CURB+BH+8;
    // Fan-village spots: try the plaza fringes flanking each grandstand first, then fill any
    // long facade-free stretch around the lap (corners, the pre-district S/F run) so most of
    // the circuit has somewhere alive. Min spacing keeps villages from merging.
    var tentSpots=[];
    function tryTentSpot(twI,sT){
      twI=((twI%N)+N)%N;
      var siT=(sT>0)?0:1;
      if(walled[siT][twI]||tentSpots.length>=8) return false;
      var wpT2=waypoints[twI],pT2=wpPerp(twI);
      var tx2=wpT2.x+pT2.x*sT*(sideOff+3),tz2=wpT2.z+pT2.z*sT*(sideOff+3);
      if(inPitRect(tx2,tz2)||!clearsTrack(tx2,tz2,BH)) return false;
      // The built cluster spans ~±16.5 along the row (tents+flags) and -7.5..+7 across it
      // (picnic tables out front, stalls behind) — clear the whole footprint, not just the
      // centre, or row ends land on the track on the inside of corners.
      var ryT=Math.atan2(pT2.x*sT,pT2.z*sT),caT=Math.cos(ryT),saT=Math.sin(ryT);
      for(var fa=-16.5;fa<=16.5;fa+=8.25)for(var fb=-7.5;fb<=7;fb+=14.5){
        if(!clearsTrack(tx2+fa*caT+fb*saT,tz2-fa*saT+fb*caT,1.2)) return false;
      }
      for(var ei=0;ei<tentSpots.length;ei++){
        var ddx=tentSpots[ei].x-tx2,ddz=tentSpots[ei].z-tz2;
        if(ddx*ddx+ddz*ddz<70*70) return false;
      }
      tentSpots.push({x:tx2,y:wpT2.y,z:tz2,ry:Math.atan2(pT2.x*sT,pT2.z*sT),hue:tentSpots.length});
      return true;
    }
    standDefs.forEach(function(sd,ti3){
      var halfWp=Math.ceil((standW[ti3]*0.5+4)/SEG_LEN);
      [halfWp+2,-(halfWp+2),halfWp+7,-(halfWp+7),25,-25].some(function(o){return tryTentSpot(sd[0]+o,sd[1]);});
    });
    [0,1].forEach(function(siT){
      var sT=siT===0?1:-1,runT=0;
      for(var twI=0;twI<N;twI++){
        if(!walled[siT][twI]&&!blocked[siT][twI]) runT++; else runT=0;
        if(runT>=14){ tryTentSpot(twI-7,sT); runT=0; }
      }
    });
    var tPos=[],tIdx2=[],tCol=[];
    function pushBox(cx,cy,cz,w,h,d,ry,r,g,b2){
      var hw2=w/2,hd=d/2,c5=Math.cos(ry),s5=Math.sin(ry),base=tPos.length/3;
      [[-hw2,0,-hd],[hw2,0,-hd],[hw2,0,hd],[-hw2,0,hd],[-hw2,h,-hd],[hw2,h,-hd],[hw2,h,hd],[-hw2,h,hd]].forEach(function(v){
        tPos.push(cx+v[0]*c5+v[2]*s5,cy+v[1],cz-v[0]*s5+v[2]*c5);tCol.push(r,g,b2);
      });
      [[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7],[4,5,6],[4,6,7]].forEach(function(f){
        tIdx2.push(base+f[0],base+f[1],base+f[2]);
      });
    }
    function pushRoof(cx,cy,cz,w,h,d,ry,r,g,b2){
      var hw2=w/2,hd=d/2,c5=Math.cos(ry),s5=Math.sin(ry),base=tPos.length/3;
      [[-hw2,0,-hd],[hw2,0,-hd],[hw2,0,hd],[-hw2,0,hd],[0,h,0]].forEach(function(v){
        tPos.push(cx+v[0]*c5+v[2]*s5,cy+v[1],cz-v[0]*s5+v[2]*c5);tCol.push(r,g,b2);
      });
      [[0,4,1],[1,4,2],[2,4,3],[3,4,0]].forEach(function(f){tIdx2.push(base+f[0],base+f[1],base+f[2]);});
    }
    var signNames=['FAN ZONE','TOKYO EATS','TEAM MERCH','PIT VIEW','GRID WALK'];
    var signMats=signNames.map(function(nm,si4){
      var cv=document.createElement('canvas');cv.width=512;cv.height=128;var g4=cv.getContext('2d');
      g4.fillStyle='#0a0a10';g4.fillRect(0,0,512,128);
      var nc=['#22d3ee','#ffb52e','#ff3b4e','#8b7bff','#3ee07a'][si4];
      g4.strokeStyle=nc;g4.lineWidth=5;g4.strokeRect(8,8,496,112);
      g4.fillStyle=nc;g4.font='900 56px '+JFONT;g4.textAlign='center';g4.textBaseline='middle';
      g4.fillText(nm,256,66);
      var tx3=new THREE.CanvasTexture(cv);
      return new THREE.MeshStandardMaterial({map:tx3,emissiveMap:tx3,emissive:0xffffff,emissiveIntensity:1.1,roughness:0.5});
    });
    // Fan villages: each surviving spot becomes a cluster — a row of neon tents, a food-stall
    // row behind, picnic tables in front, string lights between tent peaks (one Points + one
    // LineSegments draw for all zones) and a couple of waving team flags.
    var bulbPts=[],wirePts=[];
    var flagPoleGeo=new THREE.CylinderGeometry(0.05,0.07,4.2,5);
    var flagPoleMat=new THREE.MeshStandardMaterial({color:0x2a2c30,roughness:0.85,metalness:0.15});
    tentSpots.forEach(function(ts,ti4){
      var ca=Math.cos(ts.ry),sa=Math.sin(ts.ry);
      function vx(a,b){return ts.x+a*ca+b*sa;}
      function vz(a,b){return ts.z-a*sa+b*ca;}
      var nT=3+(ti4%3),half=(nT-1)*0.5;
      var peaks=[];
      for(var tk=0;tk<nT;tk++){
        var ta=(tk-half)*6.4;
        var hue=NEON_CYCLE[(ts.hue+tk)%NEON_CYCLE.length];
        var r=((hue>>16)&255)/255,g=((hue>>8)&255)/255,b2=(hue&255)/255;
        pushBox(vx(ta,0),ts.y,vz(ta,0),4.6,2.6,3.2,ts.ry,0.07,0.07,0.10);
        pushRoof(vx(ta,0),ts.y+2.6,vz(ta,0),5.4,1.6,4.0,ts.ry,r*0.55,g*0.55,b2*0.55);
        peaks.push({x:vx(ta,0),z:vz(ta,0)});
      }
      // food stalls behind the tents, warm canopies
      for(var sk=0;sk<3;sk++){
        var sa2=(sk-1)*4.4;
        pushBox(vx(sa2,5.6),ts.y,vz(sa2,5.6),2.0,1.2,1.2,ts.ry,0.10,0.10,0.13);
        pushRoof(vx(sa2,5.6),ts.y+1.2,vz(sa2,5.6),2.6,0.7,1.8,ts.ry,0.55,0.38,0.16);
      }
      // picnic tables out front
      for(var pk=0;pk<4;pk++){
        var pa=(pk-1.5)*3.6,pb=-4.6-(pk%2)*1.6;
        pushBox(vx(pa,pb),ts.y,vz(pa,pb),0.4,0.66,0.4,ts.ry,0.13,0.11,0.09);
        pushBox(vx(pa,pb),ts.y+0.66,vz(pa,pb),1.8,0.12,0.9,ts.ry,0.22,0.18,0.13);
      }
      // string lights between tent peaks
      for(var lk=0;lk<peaks.length-1;lk++){
        var pA=peaks[lk],pB=peaks[lk+1];
        for(var bk=0;bk<=4;bk++){
          var bt=bk/4,sag=0.55*4*bt*(1-bt);
          var bxp=pA.x+(pB.x-pA.x)*bt,bzp=pA.z+(pB.z-pA.z)*bt,byp=ts.y+4.25-sag;
          if(bk>0){
            var pt=(bk-1)/4,ps=0.55*4*pt*(1-pt);
            wirePts.push(pA.x+(pB.x-pA.x)*pt,ts.y+4.25-ps,pA.z+(pB.z-pA.z)*pt,bxp,byp,bzp);
          }
          bulbPts.push(bxp,byp,bzp);
        }
      }
      // waving team flags flanking the village
      for(var fk=0;fk<2;fk++){
        var fa2=(fk===0?-1:1)*(half*6.4+3.4);
        var fxp=vx(fa2,-2.5),fzp=vz(fa2,-2.5);
        var fp=new THREE.Mesh(flagPoleGeo,flagPoleMat);
        fp.position.set(fxp,ts.y+2.1,fzp);scene.add(fp);
        var fhue=NEON_CYCLE[(ts.hue+fk*3+1)%NEON_CYCLE.length];
        var fl=new THREE.Mesh(new THREE.PlaneGeometry(1.5,0.9),
          new THREE.MeshStandardMaterial({color:fhue,emissive:fhue,emissiveIntensity:0.45,side:THREE.DoubleSide,roughness:0.6}));
        fl.position.set(fxp,ts.y+3.7,fzp);fl.rotation.y=ts.ry;
        scene.add(fl);fanFlags.push({flag:fl,ry:ts.ry,ph:Math.random()*6.28});
      }
      var sgn=new THREE.Mesh(new THREE.PlaneGeometry(3.6,0.9),signMats[ti4%signMats.length]);
      sgn.position.set(vx(0,-1.65),ts.y+2.25,vz(0,-1.65));
      sgn.rotation.y=ts.ry+Math.PI;scene.add(sgn);
    });
    if(bulbPts.length){
      var blGeo=new THREE.BufferGeometry();
      blGeo.setAttribute('position',new THREE.Float32BufferAttribute(bulbPts,3));
      scene.add(new THREE.Points(blGeo,new THREE.PointsMaterial({color:0xffe2b0,size:0.46,sizeAttenuation:true})));
      var wrGeo=new THREE.BufferGeometry();
      wrGeo.setAttribute('position',new THREE.Float32BufferAttribute(wirePts,3));
      scene.add(new THREE.LineSegments(wrGeo,new THREE.LineBasicMaterial({color:0x101014})));
    }
    if(tPos.length){
      var tGeo=new THREE.BufferGeometry();
      tGeo.setAttribute('position',new THREE.Float32BufferAttribute(tPos,3));
      tGeo.setAttribute('color',new THREE.Float32BufferAttribute(tCol,3));
      tGeo.setIndex(tIdx2);tGeo.computeVertexNormals();
      scene.add(new THREE.Mesh(tGeo,new THREE.MeshStandardMaterial({vertexColors:true,emissive:0x18181c,emissiveIntensity:0.4,roughness:0.8})));
    }
    // Walkers: sidewalk strips flanking each grandstand, plus millers circling the tents.
    var crowdCols2=[0xcc1111,0x1133cc,0xddcc00,0xeeeeee,0x118833,0xcc6600,0x880099,0x009988];
    standDefs.forEach(function(sd){
      for(var k=-2;k<=2;k++){
        var wI=((sd[0]+k*14)%N+N)%N,siW=(sd[1]>0)?0:1;
        var wpW=waypoints[wI],pW=wpPerp(wI),dW=wpDir(wI);
        var off2=sideOff+1.5+Math.random()*4;
        var wx2=wpW.x+pW.x*sd[1]*off2,wz2=wpW.z+pW.z*sd[1]*off2;
        if(inPitRect(wx2,wz2)||!clearsTrack(wx2,wz2,BH)) continue;
        if(walled[siW][wI]&&!blocked[siW][wI]) continue;
        fanWalk.push({kind:'walk',x0:wx2,y:wpW.y,z0:wz2,dx:dW.x,dz:dW.z,
          len:9+Math.random()*14,t:Math.random(),spd:(0.045+Math.random()*0.05),dir:1,ph:Math.random()*6.28});
      }
    });
    // circuit-wide strollers: open verge where no facade landed, otherwise the sidewalk strip
    // between the barrier zone and the storefronts (skipped where the alley is too narrow)
    var walkBase=TW*0.5+CURB+BH+1.0;
    for(var cwI=0;cwI<N;cwI+=10){
      var sC=((cwI/10)|0)%2===0?1:-1,siC=(sC>0)?0:1;
      var wpC=waypoints[cwI],pC=wpPerp(cwI),dC=wpDir(cwI);
      var offC;
      if(walled[siC][cwI]){
        var faceC=null;
        for(var dii=0;dii<DISTRICTS.length;dii++){if(cwI>=DISTRICTS[dii].a&&cwI<DISTRICTS[dii].b){faceC=DISTRICTS[dii].face;break;}}
        if(faceC===null||faceC-3.0<=walkBase) continue;
        offC=walkBase+Math.random()*(faceC-3.0-walkBase);
      } else if(blocked[siC][cwI]){
        continue;
      } else {
        offC=sideOff+1.0+Math.random()*4.5;
      }
      var cwx=wpC.x+pC.x*sC*offC,cwz=wpC.z+pC.z*sC*offC;
      if(inPitRect(cwx,cwz)||!clearsTrack(cwx,cwz,BH)) continue;
      fanWalk.push({kind:'walk',x0:cwx,y:wpC.y,z0:cwz,dx:dC.x,dz:dC.z,
        len:8+Math.random()*16,t:Math.random(),spd:0.04+Math.random()*0.055,dir:Math.random()<0.5?1:-1,ph:Math.random()*6.28});
    }
    tentSpots.forEach(function(ts){
      var caM=Math.cos(ts.ry),saM=Math.sin(ts.ry);
      for(var mi2=0;mi2<7;mi2++){
        fanWalk.push({kind:'mill',cx:ts.x+(Math.random()-0.5)*7,y:ts.y,cz:ts.z+(Math.random()-0.5)*5,
          r:1.6+Math.random()*2.2,a:Math.random()*6.28,va:(0.25+Math.random()*0.3)*(Math.random()<0.5?-1:1),ph:Math.random()*6.28});
      }
      // a couple queueing at the food stalls behind the tents
      for(var qi2=0;qi2<2;qi2++){
        var qa=(Math.random()-0.5)*8;
        fanWalk.push({kind:'mill',cx:ts.x+qa*caM+4.6*saM,y:ts.y,cz:ts.z-qa*saM+4.6*caM,
          r:0.8+Math.random()*0.9,a:Math.random()*6.28,va:(0.18+Math.random()*0.2)*(Math.random()<0.5?-1:1),ph:Math.random()*6.28});
      }
    });
    if(fanWalk.length){
      var fbGeo=new THREE.BoxGeometry(0.55,1.05,0.35);fbGeo.translate(0,0.85,0);
      var fhGeo=new THREE.SphereGeometry(0.13,8,6);fhGeo.translate(0,1.62,0);
      fanBody=new THREE.InstancedMesh(fbGeo,new THREE.MeshStandardMaterial({roughness:0.9,emissive:0xffffff,emissiveIntensity:0.10}),fanWalk.length);
      fanHead=new THREE.InstancedMesh(fhGeo,new THREE.MeshStandardMaterial({color:0xd9b18c,roughness:0.85}),fanWalk.length);
      for(var fi2=0;fi2<fanWalk.length;fi2++) fanBody.setColorAt(fi2,new THREE.Color(crowdCols2[fi2%crowdCols2.length]));
      fanBody.castShadow=false;fanHead.castShadow=false;
      scene.add(fanBody);scene.add(fanHead);
    }
    // FIA staff & officials: hi-vis orange marshals at the corner posts, white officials and
    // yellow security at the gantries, grandstand entrances and pit entry/exit. Slightly
    // emissive bodies so the vests read under the floodlights; same two-draw-call pattern.
    var STAFF_ORANGE=0xff7a00,STAFF_WHITE=0xf2f2f5,STAFF_YELLOW=0xffd400;
    function faceTrack(pp,sde){return Math.atan2(-pp.x*sde,-pp.z*sde);}
    for(var mp2=70;mp2<N;mp2+=120){
      var sdM=((mp2/120)|0)%2?1:-1;
      var wpM=waypoints[mp2%N],ppM=wpPerp(mp2%N),ddM=wpDir(mp2%N);
      var offM=TW*0.5+CURB+BH+0.8;
      for(var sm2=0;sm2<2;sm2++){
        var alM=(sm2===0?-1.3:1.3);
        var sxM=wpM.x+ppM.x*sdM*offM+ddM.x*alM,szM=wpM.z+ppM.z*sdM*offM+ddM.z*alM;
        if(inPitRect(sxM,szM)||!clearsTrack(sxM,szM,BH*0.5))continue;
        staffWalk.push({kind:'stand',x:sxM,y:wpM.y,z:szM,ry:faceTrack(ppM,sdM),ph:Math.random()*6.28,col:STAFF_ORANGE});
      }
    }
    [250,715].forEach(function(gwp2){
      [1,-1].forEach(function(sG){
        var wpG2=waypoints[gwp2],ppG2=wpPerp(gwp2),ddG2=wpDir(gwp2);
        var gx2=wpG2.x+ppG2.x*sG*(sideOff-2),gz2=wpG2.z+ppG2.z*sG*(sideOff-2);
        if(inPitRect(gx2,gz2)||!clearsTrack(gx2,gz2,BH))return;
        staffWalk.push({kind:'stand',x:gx2,y:wpG2.y,z:gz2,ry:faceTrack(ppG2,sG),ph:Math.random()*6.28,col:STAFF_WHITE});
        staffWalk.push({kind:'walk',x0:gx2+ddG2.x*2,y:wpG2.y,z0:gz2+ddG2.z*2,dx:ddG2.x,dz:ddG2.z,
          len:6,t:Math.random(),spd:0.05,dir:1,ph:Math.random()*6.28,col:STAFF_YELLOW});
      });
    });
    standDefs.forEach(function(sd2){
      var wpS=waypoints[sd2[0]],ppS=wpPerp(sd2[0]),ddS=wpDir(sd2[0]);
      var bxS=wpS.x+ppS.x*sd2[1]*(sideOff-3),bzS=wpS.z+ppS.z*sd2[1]*(sideOff-3);
      if(inPitRect(bxS,bzS)||!clearsTrack(bxS,bzS,BH))return;
      [-8,8].forEach(function(alS){
        staffWalk.push({kind:'stand',x:bxS+ddS.x*alS,y:wpS.y,z:bzS+ddS.z*alS,ry:faceTrack(ppS,sd2[1])+Math.PI,ph:Math.random()*6.28,col:STAFF_YELLOW});
      });
      staffWalk.push({kind:'walk',x0:bxS-ddS.x*5,y:wpS.y,z0:bzS-ddS.z*5,dx:ddS.x,dz:ddS.z,
        len:10,t:Math.random(),spd:0.045,dir:1,ph:Math.random()*6.28,col:STAFF_WHITE});
    });
    // pit entry/exit marshals: first and last waypoint whose verge borders the pit complex
    var pitWps=[];
    for(var pwi=0;pwi<N;pwi++){
      var wpp2=waypoints[pwi],ppp2=wpPerp(pwi);
      [1,-1].forEach(function(sp2){
        if(inPitRect(wpp2.x+ppp2.x*sp2*14,wpp2.z+ppp2.z*sp2*14))pitWps.push({i:pwi,s:sp2});
      });
    }
    if(pitWps.length>1){
      [pitWps[0],pitWps[pitWps.length-1]].forEach(function(pe){
        var wpE=waypoints[pe.i],ppE=wpPerp(pe.i),ddE=wpDir(pe.i);
        var exE=wpE.x+ppE.x*pe.s*(TW*0.5+CURB+2.0),ezE=wpE.z+ppE.z*pe.s*(TW*0.5+CURB+2.0);
        staffWalk.push({kind:'stand',x:exE,y:wpE.y,z:ezE,ry:faceTrack(ppE,pe.s),ph:Math.random()*6.28,col:STAFF_ORANGE});
        staffWalk.push({kind:'stand',x:exE+ddE.x*1.4,y:wpE.y,z:ezE+ddE.z*1.4,ry:faceTrack(ppE,pe.s),ph:Math.random()*6.28,col:STAFF_ORANGE});
      });
    }
    if(staffWalk.length){
      var sbGeo=new THREE.BoxGeometry(0.55,1.05,0.35);sbGeo.translate(0,0.85,0);
      var shGeo=new THREE.SphereGeometry(0.13,8,6);shGeo.translate(0,1.62,0);
      staffBody=new THREE.InstancedMesh(sbGeo,new THREE.MeshStandardMaterial({roughness:0.85,emissive:0xffffff,emissiveIntensity:0.12}),staffWalk.length);
      staffHead=new THREE.InstancedMesh(shGeo,new THREE.MeshStandardMaterial({color:0xe8e8ec,roughness:0.8}),staffWalk.length);
      for(var si5=0;si5<staffWalk.length;si5++) staffBody.setColorAt(si5,new THREE.Color(staffWalk[si5].col));
      staffBody.castShadow=false;staffHead.castShadow=false;
      scene.add(staffBody);scene.add(staffHead);
    }
  })();
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
  (function(){
    var off=TW*0.5+CURB+1.1,last=-100;
    for(var ci=0;ci<N;ci+=5){
      var dA=wpDir((ci-8+N)%N),dB=wpDir((ci+8)%N);
      if(Math.abs(dA.x*dB.z-dA.z*dB.x)<0.45||ci-last<55) continue;
      last=ci;
      var inx=dB.x-dA.x,inz=dB.z-dA.z;
      for(var a=-3;a<=3;a++){
        var wi=(ci+a*3+N)%N,wp=waypoints[wi],pp=wpPerp(wi);
        var side=(pp.x*inx+pp.z*inz)>0?-1:1;
        var bx=wp.x+pp.x*side*off,bz=wp.z+pp.z*side*off;
        [mTireW,mTireR,mTireW].forEach(function(mt,h){
          var tc=new THREE.Mesh(new THREE.CylinderGeometry(0.55,0.55,0.85,12),mt);
          tc.position.set(bx,wp.y+0.45+h*0.85,bz);scene.add(tc);
        });
      }
    }
  })();
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

  function buildMarshalPost(wpIdx,side){
    var wp=waypoints[wpIdx%N],pp=wpPerp(wpIdx%N);
    var off=TW*0.5+CURB+2.4;
    var bx=wp.x+pp.x*side*off,bz=wp.z+pp.z*side*off;
    var faceAng=Math.atan2(-pp.x*side,-pp.z*side);
    var grp=new THREE.Group();grp.position.set(bx,wp.y,bz);grp.rotation.y=faceAng;
    var mSuit=new THREE.MeshStandardMaterial({color:0xe8e8ec,roughness:0.7});
    var mDark=new THREE.MeshStandardMaterial({color:0x20242b,roughness:0.85});
    var mSkin=new THREE.MeshStandardMaterial({color:0xd9b18c,roughness:0.85});
    function part(geo,mat,x,y,z,rx,rz){var m=new THREE.Mesh(geo,mat);m.position.set(x,y,z);if(rx)m.rotation.x=rx;if(rz)m.rotation.z=rz;m.castShadow=true;grp.add(m);return m;}
    part(new THREE.CylinderGeometry(0.2,0.16,0.62,8),mSuit,0,1.14,0);
    part(new THREE.CylinderGeometry(0.07,0.07,0.52,6),mDark,-0.1,0.5,0);
    part(new THREE.CylinderGeometry(0.07,0.07,0.52,6),mDark,0.1,0.5,0);
    part(new THREE.SphereGeometry(0.13,10,8),mSkin,0,1.54,0);
    var hel=part(new THREE.SphereGeometry(0.16,12,9),mDark,0,1.57,0.01);hel.scale.set(1,0.85,1.05);
    part(new THREE.CylinderGeometry(0.05,0.05,0.5,6),mSuit,-0.24,1.1,0.16,0.7);
    var pole=new THREE.Mesh(new THREE.CylinderGeometry(0.028,0.028,1.5,6),mDark);
    pole.position.set(0.34,1.85,0.2);pole.rotation.z=-0.5;grp.add(pole);
    part(new THREE.CylinderGeometry(0.05,0.05,0.6,6),mSuit,0.2,1.34,0.16,0.8,0.5);
    var flagMat=new THREE.MeshStandardMaterial({color:0xffd400,side:THREE.DoubleSide,roughness:0.6,emissive:0x000000});
    var flag=new THREE.Mesh(new THREE.PlaneGeometry(0.95,0.6,1,1),flagMat);
    flag.position.set(0.95,2.25,0.2);flag.visible=false;grp.add(flag);
    scene.add(grp);
    marshals.push({flag:flag,mat:flagMat,pivotX:0.5,ph:Math.random()*6.28});
  }
  for(var _mp=70;_mp<N;_mp+=120) buildMarshalPost(_mp,((_mp/120)|0)%2?1:-1);

  (function buildSky(){
    var cx=0,cz=0;for(var si=0;si<N;si++){cx+=waypoints[si].x;cz+=waypoints[si].z;}cx/=N;cz/=N;
    var mBody=new THREE.MeshStandardMaterial({color:0x1b1f26,roughness:0.5,metalness:0.3});
    var mGlass=new THREE.MeshStandardMaterial({color:0x223044,roughness:0.2,metalness:0.6,emissive:0x05101f});
    var mRot=new THREE.MeshStandardMaterial({color:0x111111,roughness:0.7});
    var heli=new THREE.Group();
    var body=new THREE.Mesh(new THREE.SphereGeometry(3.2,12,10),mBody);body.scale.set(1.6,1,1);heli.add(body);
    var nose=new THREE.Mesh(new THREE.SphereGeometry(2.2,12,10),mGlass);nose.scale.set(1.1,0.8,0.9);nose.position.set(4.2,-0.2,0);heli.add(nose);
    var boom=new THREE.Mesh(new THREE.CylinderGeometry(0.5,0.3,8,8),mBody);boom.rotation.z=Math.PI/2;boom.position.set(-6,0.4,0);heli.add(boom);
    var fin=new THREE.Mesh(new THREE.BoxGeometry(1.2,2.2,0.3),mBody);fin.position.set(-9.4,1,0);heli.add(fin);
    var skid1=new THREE.Mesh(new THREE.CylinderGeometry(0.18,0.18,7,6),mBody);skid1.rotation.z=Math.PI/2;skid1.position.set(0,-3,1.8);heli.add(skid1);
    var skid2=skid1.clone();skid2.position.z=-1.8;heli.add(skid2);
    var rotor=new THREE.Group();
    for(var ri=0;ri<2;ri++){var bl=new THREE.Mesh(new THREE.BoxGeometry(16,0.12,0.7),mRot);bl.rotation.y=ri*Math.PI/2;rotor.add(bl);}
    rotor.position.set(0,3.4,0);heli.add(rotor);
    heli.add(new THREE.Mesh(new THREE.CylinderGeometry(0.2,0.2,1.4,6),mBody)).position.set(0,2.9,0);
    var trotor=new THREE.Group();var tbl=new THREE.Mesh(new THREE.BoxGeometry(0.1,3,0.4),mRot);trotor.add(tbl);var tbl2=tbl.clone();tbl2.rotation.x=Math.PI/2;trotor.add(tbl2);
    trotor.position.set(-9.4,1,0.6);heli.add(trotor);
    var navL=new THREE.Mesh(new THREE.SphereGeometry(0.3,6,6),new THREE.MeshBasicMaterial({color:0xff3030}));navL.position.set(0,-3.2,0);heli.add(navL);
    scene.add(heli);
    skyObjs.push({grp:heli,rotor:rotor,trotor:trotor,nav:navL,kind:'heli',cx:cx,cz:cz,ang:0,radius:340,height:175,speed:0.12});

    var mEnv=new THREE.MeshStandardMaterial({color:0x0a1a1e,roughness:0.5,emissive:0x00E5FF,emissiveIntensity:0.7});
    var blimp=new THREE.Group();
    var env=new THREE.Mesh(new THREE.SphereGeometry(14,16,12),mEnv);env.scale.set(2.2,1,1);blimp.add(env);
    var tailA=new THREE.Mesh(new THREE.BoxGeometry(5,7,0.5),mEnv);tailA.position.set(-27,0,0);blimp.add(tailA);
    var tailB=tailA.clone();tailB.rotation.x=Math.PI/2;blimp.add(tailB);
    var gond=new THREE.Mesh(new THREE.BoxGeometry(6,2.2,2.4),mBody);gond.position.set(2,-13,0);blimp.add(gond);
    blimp.position.set(-650,265,-520);scene.add(blimp);
    skyObjs.push({grp:blimp,kind:'blimp',vx:9,xmin:-650,xmax:650});

    for(var bi=0;bi<7;bi++){
      var bird=new THREE.Group();
      var mWing=new THREE.MeshStandardMaterial({color:0x15171c,roughness:0.9,emissive:0x8b7bff,emissiveIntensity:0.25});
      var wl=new THREE.Mesh(new THREE.BoxGeometry(3,0.1,0.9),mWing);wl.position.set(-1.6,0,0);wl.rotation.z=0.3;bird.add(wl);
      var wr=new THREE.Mesh(new THREE.BoxGeometry(3,0.1,0.9),mWing);wr.position.set(1.6,0,0);wr.rotation.z=-0.3;bird.add(wr);
      bird.scale.setScalar(0.6+Math.random()*0.6);scene.add(bird);
      skyObjs.push({grp:bird,wl:wl,wr:wr,kind:'bird',cx:cx,cz:cz,ang:Math.random()*6.28,radius:120+Math.random()*120,height:120+Math.random()*40,speed:0.5+Math.random()*0.3,ph:Math.random()*6.28});
    }

    // Broadcast drones: small neon quad-rotors drifting low over the circuit — cross-arm
    // body, four rotor disks, an underglow lamp in the signage palette and a blinking
    // tail nav light (faded via material.opacity in the loop).
    var glowTex=brakeGlowTex();
    for(var di=0;di<5;di++){
      var drone=new THREE.Group();
      var dcol=NEON_CYCLE[di%NEON_CYCLE.length];
      drone.add(new THREE.Mesh(new THREE.BoxGeometry(1.3,0.34,1.3),mBody));
      [Math.PI/4,-Math.PI/4].forEach(function(da){
        var arm=new THREE.Mesh(new THREE.BoxGeometry(2.8,0.08,0.18),mRot);arm.rotation.y=da;drone.add(arm);
      });
      [[1,1],[1,-1],[-1,1],[-1,-1]].forEach(function(dq){
        var rot=new THREE.Mesh(new THREE.CylinderGeometry(0.52,0.52,0.05,10),mRot);
        rot.position.set(dq[0]*0.99,0.12,dq[1]*0.99);drone.add(rot);
      });
      var lamp=new THREE.Mesh(new THREE.SphereGeometry(0.16,8,6),new THREE.MeshBasicMaterial({color:dcol,fog:false}));
      lamp.position.y=-0.26;drone.add(lamp);
      var lampGlow=new THREE.Sprite(new THREE.SpriteMaterial({map:glowTex,color:dcol,transparent:true,opacity:0.5,blending:THREE.AdditiveBlending,depthWrite:false,fog:false}));
      lampGlow.scale.set(2.4,2.4,1);lampGlow.position.y=-0.3;drone.add(lampGlow);
      var nav=new THREE.Mesh(new THREE.SphereGeometry(0.10,6,5),new THREE.MeshBasicMaterial({color:0xff3040,transparent:true,fog:false}));
      nav.position.set(-0.75,0.1,0);drone.add(nav);
      scene.add(drone);
      skyObjs.push({grp:drone,nav:nav,kind:'drone',cx:cx+(Math.random()-0.5)*240,cz:cz+(Math.random()-0.5)*240,
        ang:Math.random()*6.28,radius:45+Math.random()*90,height:20+Math.random()*22,speed:(0.18+Math.random()*0.14)*(di%2?1:-1),ph:Math.random()*6.28});
    }
  })();
}
// Build the (heavy) 3D world lazily so it never blocks the dashboard's first paint. It is
// kicked off in the background just after load, and openGame() calls ensureWorld() as a
// guaranteed fallback before the first frame renders.
var _worldBuilt=false;
function ensureWorld(){ if(_worldBuilt) return; _worldBuilt=true; buildWorld(); }
(window.requestIdleCallback||function(f){setTimeout(f,200);})(ensureWorld);

/* NB: we deliberately do NOT set scene.environment. A PMREM env map of the night
   scene zeroes out direct/ambient lighting on every MeshStandardMaterial in r128,
   turning the whole world black. Instead we bake a dedicated night-sky probe below and
   assign it PER-MATERIAL on the cars only, so paint/chrome gain real reflections while the
   world keeps its direct lighting. */
// One-time night-sky reflection probe (sky gradient + moon highlight + warm floodlight keys), baked
// into a PMREM cubemap. Applied per-material on cars via buildCar — never as scene.environment.
var _gameEnv=null;
function gameEnvMap(){
  if(_gameEnv!==null) return _gameEnv||undefined;
  try{
    var pmrem=new THREE.PMREMGenerator(renderer);pmrem.compileEquirectangularShader();
    var em=new THREE.ShaderMaterial({side:THREE.BackSide,
      vertexShader:'varying vec3 vN;void main(){vN=normalize(position);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
      fragmentShader:'varying vec3 vN;void main(){vec3 n=normalize(vN);vec3 col=mix(vec3(0.008,0.010,0.024),vec3(0.045,0.060,0.130),smoothstep(-0.3,0.8,n.y));float m=pow(max(0.,dot(n,normalize(vec3(0.86,0.40,0.26)))),60.);col+=m*vec3(0.95,0.93,1.00)*2.6;float fa=pow(max(0.,dot(n,normalize(vec3(0.20,0.92,0.10)))),3.);col+=fa*vec3(0.13,0.83,0.93)*0.60;float fb=pow(max(0.,dot(n,normalize(vec3(-0.50,0.72,-0.42)))),4.);col+=fb*vec3(1.00,0.30,0.85)*0.50;gl_FragColor=vec4(col,1.);}'
    });
    var es=new THREE.Scene();es.add(new THREE.Mesh(new THREE.SphereGeometry(50,32,16),em));
    var rt=new THREE.WebGLCubeRenderTarget(256,{format:THREE.RGBFormat,generateMipmaps:true,minFilter:THREE.LinearMipmapLinearFilter});
    var cc=new THREE.CubeCamera(1,100,rt);es.add(cc);cc.update(renderer,es);
    _gameEnv=pmrem.fromCubemap(rt.texture).texture;pmrem.dispose();
  }catch(e){_gameEnv=false;}
  return _gameEnv||undefined;
}

var _csTex=null;
function contactShadowTex(){
  if(_csTex) return _csTex;
  var cc=document.createElement('canvas');cc.width=cc.height=128;var g=cc.getContext('2d');
  var grd=g.createRadialGradient(64,64,4,64,64,64);
  grd.addColorStop(0,'rgba(0,0,0,0.60)');grd.addColorStop(0.5,'rgba(0,0,0,0.28)');grd.addColorStop(1,'rgba(0,0,0,0)');
  g.fillStyle=grd;g.fillRect(0,0,128,128);
  _csTex=new THREE.CanvasTexture(cc);return _csTex;
}
function carbonTex(){return window.F1FX.carbon();}
function carbonNormalTex(){return window.F1FX.carbonNormal();}
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
function nameTex(text,colHex){
  var cw=512,ch=128,cv=document.createElement('canvas');cv.width=cw;cv.height=ch;var x=cv.getContext('2d');
  x.fillStyle='rgba(8,10,14,0.92)';x.fillRect(0,0,cw,ch);
  var r=(colHex>>16)&255,g=(colHex>>8)&255,b=colHex&255;
  x.fillStyle='rgb('+r+','+g+','+b+')';x.fillRect(0,ch-14,cw,14);
  x.fillStyle='#ffffff';x.font='bold 64px Arial,sans-serif';x.textAlign='center';x.textBaseline='middle';
  x.fillText(''+text,cw/2,ch*0.46);
  return new THREE.CanvasTexture(cv);
}
function brakeGlowTex(){
  // 128px clean automotive LED ramp: a tight white-hot core falling off through saturated
  // red to nothing — no colored fringe, so the light reads as an LED, not signage glow.
  if(_bgTex) return _bgTex;
  var c=document.createElement('canvas');c.width=c.height=128;var g=c.getContext('2d');
  var rg=g.createRadialGradient(64,64,0,64,64,64);
  rg.addColorStop(0,'rgba(255,235,232,1)');rg.addColorStop(0.10,'rgba(255,70,60,0.92)');
  rg.addColorStop(0.34,'rgba(225,22,30,0.38)');rg.addColorStop(0.68,'rgba(170,12,22,0.10)');
  rg.addColorStop(1,'rgba(160,10,20,0)');
  g.fillStyle=rg;g.fillRect(0,0,128,128);_bgTex=new THREE.CanvasTexture(c);return _bgTex;
}
function buildCar(bodyColor,accentColor,carNum,envMap,detailed,tyreCol,accent2){
  // Thin wrapper over the shared F1FX.buildChassis: all the realistic bodywork lives there
  // (and is shared with the studio hero render). Here we only layer the game-only FX —
  // additive brake-glow sprites and a contact shadow tied to RIDE_H. The chassis already
  // ran the cast/receive-shadow traverse and exposes the userData contract the loop reads
  // (fw, rw, helmet, haloPost, tailMat, tailMat2, drsMat, bodyMat, decals, setTyre).
  var g=window.F1FX.buildChassis({bodyColor:bodyColor,accentColor:accentColor,accent2:accent2,carNum:carNum,envMap:envMap,detailed:detailed,tyreCol:tyreCol});
  // Brake glow sits on the actual lights (centre rain light + both endplate blades), not the
  // wheels: each light layers a tight hot core sprite under a modest soft halo. Entries are
  // {m,k,rain}: k scales the braking target per layer, rain marks the centre light so it can
  // blink FIA-style in the wet. Halo kept tight so the light stays a point, not a blob.
  var bgTex=brakeGlowTex();g.userData.brakeGlow=[];
  [[-2.10,0.02,0,1.15,true],[-2.13,0.41,-0.80,0.9,false],[-2.13,0.41,0.80,0.9,false]].forEach(function(pt){
    [[0.50,1.0],[1.15,0.14]].forEach(function(ly){
      var bm=new THREE.SpriteMaterial({map:bgTex,transparent:true,opacity:0,blending:THREE.AdditiveBlending,depthWrite:false,fog:false});
      var bs=new THREE.Sprite(bm);bs.scale.set(ly[0]*pt[3],ly[0]*pt[3],1);bs.position.set(pt[0],pt[1],pt[2]);g.add(bs);
      g.userData.brakeGlow.push({m:bm,k:ly[1],rain:pt[4]});
    });
  });
  var cs=new THREE.Mesh(new THREE.PlaneGeometry(5.6,2.7),new THREE.MeshBasicMaterial({map:contactShadowTex(),transparent:true,depthWrite:false,opacity:0.85,fog:true}));
  cs.rotation.x=-Math.PI/2;cs.position.set(0.1,-RIDE_H+0.03,0);cs.renderOrder=2;g.add(cs);
  return g;
}

// Roll the wheels with road speed (1/tyre radius = 1/0.34) and steer the fronts. Steering
// angle is recovered from the car's own heading rate (ackermann-ish: wheelbase * yaw rate /
// speed) so it works for player, AI and safety car alike; below walking pace the player's
// raw steer input turns the wheels so they react on the grid.
function wheelSync(grp,c,dt){
  var ud=grp&&grp.userData;if(!ud||!ud.wheels||dt<=0)return;
  var d=(c.speed||0)*dt*2.94;
  for(var i=0;i<ud.wheels.length;i++)ud.wheels[i].rotation.y-=d;
  if(ud.steerW&&ud.steerW.length){
    var h=c.heading||0,dh=(ud._phead===undefined)?0:h-ud._phead;ud._phead=h;
    if(dh>Math.PI)dh-=Math.PI*2;else if(dh<-Math.PI)dh+=Math.PI*2;
    var tg=(c.speed>6)?(3.24*dh/(dt*Math.max(c.speed,8))):((c.steerIn||0)*0.30);
    tg=Math.max(-0.42,Math.min(0.42,tg));
    ud._svis=(ud._svis||0)+(tg-(ud._svis||0))*Math.min(dt*8,1);
    for(var j=0;j<ud.steerW.length;j++)ud.steerW[j].rotation.y=ud._svis;
  }
}

function buildCockpitWheel(){
  if(cockpitWheel) return;
  var PI=Math.PI,env=gameEnvMap();
  function M(geo,mat,x,y,z,rx,ry,rz){var m=new THREE.Mesh(geo,mat);m.position.set(x||0,y||0,z||0);m.rotation.set(rx||0,ry||0,rz||0);return m;}
  function RB(w,h,d,r){return window.F1FX.roundedBox(w,h,d,r);}
  // Modern (2022+) F1 wheel: rectangular carbon faceplate, suede side grips, titanium bezel,
  // big central LCD, top rev-LED strip, anodised buttons + rotaries, carbon shift/clutch paddles.
  // The whole face lives under `spin` (rotates with steering); only the column/boss stay static.
  // Procedural Alcantara/suede micro-fibre: dark base + fine speckle, used as map+bump so the
  // grips read as real suede instead of flat plastic. color stays white so the map sets the tone.
  function suedeTex(){var c=document.createElement('canvas');c.width=c.height=128;var g=c.getContext('2d');
    g.fillStyle='#0a0a0c';g.fillRect(0,0,128,128);
    for(var i=0;i<4200;i++){var v=16+((Math.random()*30)|0);g.fillStyle='rgba('+v+','+v+','+(v+3)+','+(0.45+Math.random()*0.5).toFixed(2)+')';g.fillRect((Math.random()*128)|0,(Math.random()*128)|0,1,1);}
    var t=new THREE.CanvasTexture(c);t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(2,3);return t;}
  var sTex=suedeTex();
  var carbon=new THREE.MeshStandardMaterial({color:0x101015,map:carbonTex(),metalness:0.42,roughness:0.45,emissive:0x060608,emissiveIntensity:0.5});
  var carbonRecess=new THREE.MeshStandardMaterial({color:0x090a0e,map:carbonTex(),metalness:0.40,roughness:0.52,emissive:0x030305,emissiveIntensity:0.4});
  var suede=new THREE.MeshStandardMaterial({color:0xffffff,map:sTex,bumpMap:sTex,bumpScale:0.004,metalness:0.0,roughness:1.0,emissive:0x040405,emissiveIntensity:0.42});
  var stitchM=new THREE.MeshStandardMaterial({color:0x8c8c96,metalness:0.0,roughness:0.85,emissive:0x1a1a1e,emissiveIntensity:0.4});
  var titan=new THREE.MeshStandardMaterial({color:0x8d939b,metalness:0.9,roughness:0.32,emissive:0x0c0e11,emissiveIntensity:0.5});
  var dark=new THREE.MeshStandardMaterial({color:0x090a0d,metalness:0.4,roughness:0.6,emissive:0x050506,emissiveIntensity:0.5});
  // Thin glossy cover glass that floats over the LCD canvas (composites over the lit screen).
  var glass=new THREE.MeshStandardMaterial({color:0x05070a,metalness:0.1,roughness:0.08,transparent:true,opacity:0.16,depthWrite:false,emissive:0x02040a,emissiveIntensity:0.22});
  if(env){[carbon,carbonRecess,titan,dark,glass].forEach(function(m){m.envMap=env;m.envMapIntensity=0.6;});}
  function anod(c){return new THREE.MeshStandardMaterial({color:c,metalness:0.55,roughness:0.34,emissive:c,emissiveIntensity:0.5});}
  var wheel=new THREE.Group(),spin=new THREE.Group();

  // --- Carbon faceplate + titanium bezel frame (rounded rectangle) ---
  spin.add(M(RB(0.330,0.232,0.014,0.030),titan,0,0,-0.002));   // bezel peeks ~0.01 around the plate
  spin.add(M(RB(0.310,0.212,0.026,0.026),carbonRecess,0,0,0.006));   // recessed carbon plate

  // --- Rim: rectangular flat-top / flat-bottom hoop with rounded corners ---
  var hw=0.190,ht=0.092,hb=0.110;
  var outline=[
    new THREE.Vector3(0,ht,0),new THREE.Vector3(hw*0.52,ht,0),new THREE.Vector3(hw,ht*0.42,0),
    new THREE.Vector3(hw,-hb*0.40,0),new THREE.Vector3(hw*0.62,-hb,0),new THREE.Vector3(hw*0.22,-hb,0),
    new THREE.Vector3(0,-hb,0),new THREE.Vector3(-hw*0.22,-hb,0),new THREE.Vector3(-hw*0.62,-hb,0),
    new THREE.Vector3(-hw,-hb*0.40,0),new THREE.Vector3(-hw,ht*0.42,0),new THREE.Vector3(-hw*0.52,ht,0)
  ];
  var rimCrv=new THREE.CatmullRomCurve3(outline,true,'catmullrom',0.5);
  spin.add(new THREE.Mesh(new THREE.TubeGeometry(rimCrv,180,0.0185,14,true),carbon));

  // --- Suede side grips: thick contoured pads, toed-in ~12deg, with stitch line + thumb scoop ---
  [-1,1].forEach(function(s){
    spin.add(M(new THREE.CylinderGeometry(0.036,0.030,0.176,20),suede,s*(hw-0.006),-0.014,0.028,0,0,s*0.21));
    spin.add(M(RB(0.030,0.150,0.020,0.010),stitchM,s*(hw-0.052),-0.010,0.056,0,0,s*0.21)); // stitch seam strip
    spin.add(M(RB(0.026,0.066,0.014,0.008),dark,s*(hw-0.034),0.030,0.064,0,0,s*0.21));      // thumb scoop recess
  });

  // --- Central LCD display (256x150 canvas) raised PROUD of the carbon faceplate (front face
  // at z=0.019) so the screen module never shares a depth with the plate -> no z-fighting/glitch.
  spin.add(M(RB(0.214,0.130,0.012,0.012),titan,0,0.020,0.020));   // raised screen surround, front ~0.026
  var dcv=document.createElement('canvas');dcv.width=256;dcv.height=150;var dctx=dcv.getContext('2d');
  var dtex=new THREE.CanvasTexture(dcv);
  var dmat=new THREE.MeshBasicMaterial({map:dtex});
  dmat.polygonOffset=true;dmat.polygonOffsetFactor=-1;dmat.polygonOffsetUnits=-1;  // always win the depth tie
  spin.add(M(new THREE.PlaneGeometry(0.200,0.117),dmat,0,0.020,0.0275));            // screen proud of the surround
  spin.add(M(new THREE.PlaneGeometry(0.206,0.122),glass,0,0.020,0.0285));           // glossy cover glass over the screen
  cwDisp={cv:dcv,ctx:dctx,tex:dtex};

  // --- Rev-light strip across the top: green -> red -> violet shift lights ---
  spin.add(M(RB(0.300,0.026,0.008,0.006),dark,0,0.099,0.018));
  cwLeds=[];
  for(var li=0;li<15;li++){var t=li/14;var lc=t<0.47?0x16ff4a:t<0.80?0xff2a16:0x9a35ff;
    var lm=new THREE.MeshStandardMaterial({color:0x07070a,emissive:lc,emissiveIntensity:0.10});
    spin.add(M(new THREE.BoxGeometry(0.0118,0.014,0.006),lm,(li-7)*0.0190,0.099,0.023));cwLeds.push(lm);}

  // --- Anodised buttons + rotaries flanking the display ---
  function btn(x,y,c,r){r=r||0.0108;
    spin.add(M(new THREE.CylinderGeometry(r,r,0.012,16),titan,x,y,0.017,PI/2,0,0));
    spin.add(M(new THREE.CylinderGeometry(r*0.78,r*0.78,0.014,16),anod(c),x,y,0.024,PI/2,0,0));}
  function rotary(x,y,c,r){r=r||0.024;
    spin.add(M(new THREE.CylinderGeometry(r*1.18,r*1.30,0.010,20),dark,x,y,0.016,PI/2,0,0));   // knurled base ring
    spin.add(M(new THREE.CylinderGeometry(r,r*0.94,0.020,20),titan,x,y,0.022,PI/2,0,0));       // knob
    spin.add(M(new THREE.BoxGeometry(0.005,r*0.82,0.004),anod(c),x,y+r*0.42,0.033));}          // pointer marker
  // three buttons each side of the screen
  var lCol=[0xff3b30,0xffcc00,0x0a84ff],rCol=[0x34c759,0xf2f2f2,0xff3b30];
  for(var b=0;b<3;b++){btn(-0.122,0.052-b*0.044,lCol[b]);btn(0.122,0.052-b*0.044,rCol[b]);}
  // upper-corner mode rotaries, lower-corner clutch/bite rotaries
  rotary(-0.122,-0.072,0xffcc00);rotary(0.122,-0.072,0xff3b30);
  // a couple of pit/DRS buttons along the bottom
  btn(-0.034,-0.082,0x34c759,0.012);btn(0.034,-0.082,0x0a84ff,0.012);

  // --- Carbon shift + clutch paddles behind the wheel (rotate with it) ---
  [-1,1].forEach(function(s){
    spin.add(M(RB(0.013,0.118,0.060,0.006),carbon,s*0.150,-0.004,-0.052,0,s*0.42,s*0.14));   // shift paddle
    spin.add(M(RB(0.011,0.070,0.034,0.005),dark,s*0.092,-0.040,-0.044,0,s*0.55,0));           // clutch paddle
  });
  // central quick-release boss
  spin.add(M(new THREE.CylinderGeometry(0.030,0.026,0.030,20),titan,0,-0.054,0.012,PI/2,0,0));
  spin.add(M(new THREE.CylinderGeometry(0.018,0.018,0.034,16),dark,0,-0.054,0.000,PI/2,0,0));

  wheel.add(spin);
  // --- Static steering column + dash housing (do not spin) ---
  wheel.add(M(new THREE.CylinderGeometry(0.024,0.034,0.26,14),dark,0,-0.05,-0.18,PI/2,0,0));
  wheel.add(M(RB(0.74,0.12,0.24,0.05),dark,0,-0.225,-0.02));
  cockpitWheel=wheel;cwSpin=spin;
  wheel.position.set(0,-0.262,-0.43);wheel.rotation.x=-0.33;
  wheel.visible=false;
  wheel.traverse(function(o){if(o.isMesh){o.frustumCulled=false;o.castShadow=false;o.receiveShadow=false;}});
  gameCam.add(wheel);
}

var gameState='IDLE',paused=false,raceTime=0,animRunning=false;
var playerGrp=null,aiGrps=[],aiLabels=[];
var safetyCarGrp=null,recoverQueue=[];
var SC={state:'IDLE',target:-1,x:0,z:0,y:0,heading:0,speed:0,tIdx:0,edge:0,timer:0,rtimer:0};
var PIT_ENTRY_WP=closestWP(PIT_X0+20,PIT_Z);
var RECOVER_MAX=16.0;
var camSmooth=new THREE.Vector3(),camVel=new THREE.Vector3();
var cockpitMode=false;

var P={x:0,z:0,y:0,heading:0,speed:0,yawRate:0,tIdx:0,lap:1,
  sector:0,sectorStart:0,lapStart:0,s1:0,s2:0,s3:0,
  bestS1:Infinity,bestS2:Infinity,bestS3:Infinity,bestLap:Infinity,
  rpmVal:0,gear:1,drs:false,_drsKey:false,
  tireWear:1.0,tireTempF:0.3,tireTempR:0.3,compound:1,_pitCompound:null,_lapGuard:false,
  team:'RBR',pitState:'NONE',pitS:0,pitTimer:0,pitLift:0,pitsDone:0,_pitArmed:false,_pitKey:false,_pitExitGrace:0,
  damage:0,dmgFront:0,dmgRear:0,dnf:false};

var GRID_FRONT=104,GRID_DY=6,PLAYER_GRID=9;
function gridSlot(p){
  var row=Math.floor(p/2),col=p%2;
  return {ti:GRID_FRONT-row*GRID_DY-col*3,lat:(col===0?-1:1)*TW*0.22};
}
var AI=(function(){
  var a=[];
  // Lane fractions widened with the narrower TW so adjacent-lane separation (TW*0.20=3.36)
  // stays above the wheel-to-wheel minSep (2.9) used by the AI space-yield filter.
  var LANES=[-TW*0.40,-TW*0.20,0,TW*0.20,TW*0.40];
  for(var i=0;i<19;i++){
    var bl=LANES[i%5];
    var apxD=0.42+((i*3)%7)*0.090+Math.random()*0.03;
    var initSide=(i%2===0)?1:-1;
    a.push({team:AI_TEAMS[i],tIdx:0,_initTIdx:0,latOff:bl,_gridLat:bl,
            paceFac:1.0,skill:0.7,_aggr:1.0,_randFac:(Math.random()-0.5)*0.01,apexD:apxD,
            _mood:1.0,_moodPhase:0,_deg:0,_mistakeTimer:0,_mistakeKind:0,
            _ovSide:initSide,_initOvSide:initSide,_ovTgt:bl,_ovTimer:0.0,finishTime:Infinity,
            x:0,z:0,y:0,heading:0,speed:0,yawRate:0,lap:1,_inStart:false,_stuckTimer:0,
            _latTarget:bl,_baseLat:bl,_initLat:bl,
            pitState:'NONE',pitS:0,pitTimer:0,pitLift:0,pitsDone:0,pitLap:0,compound:1,_nextCompound:null,
            damage:0,dmgFront:0,dmgRear:0,dnf:false});
  }
  return a;
})();

var keys={};
document.addEventListener('keydown',function(e){
  keys[e.code]=true;
  if(e.code==='Escape'&&gameState!=='IDLE'){
    if(gameState==='FINISHED'){closeGame();return;}
    if(P.dnf&&!P._dnfDone){dnfToClassification();return;}
    paused=!paused;
    if(pauseOvl) pauseOvl.className=paused?'active':'';
  }
  if(e.code==='KeyQ'&&paused) closeGame();
  if(e.code==='KeyC') cockpitMode=!cockpitMode;
  if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space'].indexOf(e.code)>=0&&gameState!=='IDLE') e.preventDefault();
});
document.addEventListener('keyup',function(e){keys[e.code]=false;});

function spawnPos(ti,lat){
  var idx=((ti%N)+N)%N;
  var wp=waypoints[idx],p=wpPerp(idx),d=wpDir(idx);
  return {x:wp.x+p.x*lat,z:wp.z+p.z*lat,y:wp.y,heading:Math.atan2(d.x,d.z)};
}

function makeLabel(driverName,abbr,bodyHex,accentHex){
  var cv=document.createElement('canvas');cv.width=256;cv.height=72;
  var ctx=cv.getContext('2d');
  function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();}
  ctx.fillStyle='rgba(0,0,0,0.78)';rr(0,0,256,72,10);ctx.fill();
  var br=(bodyHex>>16)&255,bg=(bodyHex>>8)&255,bb=bodyHex&255;
  ctx.fillStyle='rgb('+br+','+bg+','+bb+')';rr(5,5,58,62,7);ctx.fill();
  var ar=(accentHex>>16)&255,ag=(accentHex>>8)&255,ab=accentHex&255;
  ctx.fillStyle='rgb('+ar+','+ag+','+ab+')';
  ctx.font='bold 15px Arial';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(abbr,34,36);
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
  if(safetyCarGrp){scene.remove(safetyCarGrp);safetyCarGrp=null;}
  recoverQueue=[];SC.state='IDLE';SC.target=-1;SC.timer=0;SC.rtimer=0;

  PLAYER_GRID=8+Math.floor(Math.random()*5);
  weather=WEATHERS[Math.floor(Math.random()*WEATHERS.length)];
  applyWeatherVisuals(weather);WEATHER.set(weather);
  flagState='none';flagTimer=0;
  AI.forEach(function(ai){
    var noise=((Math.random()+Math.random())-1.0)*0.05;
    ai.paceFac=Math.max(0.94,Math.min(1.11,(TEAM_TIER[ai.team]||0.99)+noise));
    ai.skill=Math.max(0.34,Math.min(0.99,(ai.paceFac-0.92)*4.4+(Math.random()-0.5)*0.26));
    ai._aggr=0.85+Math.random()*0.55;
    ai._mood=1.0;ai._moodPhase=Math.random()*Math.PI*2;ai._deg=0;
    ai._mistakeTimer=0;ai._mistakeKind=0;
    ai.pitState='NONE';ai.pitS=0;ai.pitTimer=0;ai.pitLift=0;ai.pitsDone=0;
    ai.pitLap=(Math.random()<0.7)?(2+Math.floor(Math.random()*2)):0;
    var rc=Math.random();
    ai.compound=rc<0.55?weather.best:(rc<0.8?Math.max(0,weather.best-1):(rc<0.95?Math.min(2,weather.best+1):(Math.random()*3|0)));
    ai._nextCompound=null;
  });
  var qOrder=AI.slice().sort(function(a,b){return b.paceFac-a.paceFac;}),gp=0;
  qOrder.forEach(function(ai){
    if(gp===PLAYER_GRID) gp++;
    var qslot=gridSlot(gp);
    ai._initTIdx=qslot.ti;ai._gridLat=qslot.lat;
    gp++;
  });

  playerGrp=buildCar(0x0A1530,0xD7202E,1,gameEnvMap(),true,null,0xFBC900);scene.add(playerGrp);
  buildCockpitWheel();
  var pslot=gridSlot(PLAYER_GRID),sp=spawnPos(pslot.ti,pslot.lat);
  P.x=sp.x;P.z=sp.z;P.y=sp.y;P.heading=sp.heading;
  P.speed=0;P.yawRate=0;P.tIdx=pslot.ti;P.lap=1;P.rpmVal=0;P.gear=1;
  P.drs=false;P._drsKey=false;P.drsArmed=false;P.tireWear=1;P.tireTempF=0.3;P.tireTempR=0.3;
  P.compound=weather.best;P._pitCompound=null;P._boxWarned=false;
  playerGrp.userData.setTyre(COMPOUNDS[P.compound].col);
  P.sector=0;P.sectorStart=0;P.lapStart=0;P.s1=0;P.s2=0;P.s3=0;
  P.bestS1=Infinity;P.bestS2=Infinity;P.bestS3=Infinity;P.bestLap=Infinity;P._lapGuard=false;P.finishTime=Infinity;
  P.pitState='NONE';P.pitS=0;P.pitTimer=0;P.pitLift=0;P.pitsDone=0;P._pitArmed=false;P._pitExitGrace=0;
  P.damage=0;P.dmgFront=0;P.dmgRear=0;P.dnf=false;P._retired=false;P.dnfReason=null;P._dnfRest=0;P._dnfDone=false;
  for(var _tk in teamCrew){teamCrew[_tk]._busy=false;}
  AI.forEach(function(ai,i){
    var ag=buildCar(AI_COLORS[i],AI_ACCENTS[i],AI_NUMS[i],gameEnvMap(),false,COMPOUNDS[ai.compound].col,AI_TEAMS[i]==='RBR'?0xFBC900:null);scene.add(ag);aiGrps.push(ag);
    var lbl=makeLabel(AI_NAMES[i+1],AI_TEAMS[i],AI_COLORS[i],AI_ACCENTS[i]);scene.add(lbl);aiLabels.push(lbl);
    ai._name=AI_NAMES[i+1];
    ai.tIdx=ai._initTIdx;ai._latTarget=ai._gridLat;ai._baseLat=ai._initLat;
    ai._ovTimer=0;ai._stuckTimer=0;ai._ovSide=ai._initOvSide;ai._ovTgt=ai._initLat;
    var as=spawnPos(ai.tIdx,ai._gridLat);
    ai.x=as.x;ai.z=as.z;ai.y=as.y;ai.heading=as.heading;
    ai.speed=0;ai.yawRate=0;ai.lap=1;ai._inStart=false;ai.finishTime=Infinity;
    ai.damage=0;ai.dmgFront=0;ai.dmgRear=0;ai.dnf=false;ai.dnfReason=null;
    ai._idx=i;ai._awaitRecovery=false;ai._towed=false;ai._recovered=false;
    ag.position.set(ai.x,ai.y+RIDE_H,ai.z);ag.rotation.y=ai.heading-Math.PI/2;
    lbl.position.set(ai.x,ai.y+RIDE_H+3.2,ai.z);
  });
  safetyCarGrp=buildCar(0xEAEAEA,0x18c83c,0,gameEnvMap());
  (function(sc){
    var amber=new THREE.MeshStandardMaterial({color:0xffaa00,emissive:0xffaa00,emissiveIntensity:2.0,roughness:0.5});
    var blue=new THREE.MeshStandardMaterial({color:0x1466ff,emissive:0x1466ff,emissiveIntensity:0.2,roughness:0.5});
    var b1=new THREE.Mesh(new THREE.BoxGeometry(0.5,0.22,0.5),amber);b1.position.set(0,1.35,0.42);
    var b2=new THREE.Mesh(new THREE.BoxGeometry(0.5,0.22,0.5),blue);b2.position.set(0,1.35,-0.42);
    sc.add(b1);sc.add(b2);sc.userData.beacon=[amber,blue];
  })(safetyCarGrp);
  safetyCarGrp.visible=false;scene.add(safetyCarGrp);

  playerGrp.position.set(P.x,P.y+RIDE_H,P.z);playerGrp.rotation.y=P.heading-Math.PI/2;
  var d0=wpDir(P.tIdx);
  gameCam.position.set(P.x-d0.x*28,P.y+13,P.z-d0.z*28);
  camSmooth.copy(gameCam.position);camVel.set(0,0,0);
  gameCam.lookAt(new THREE.Vector3(P.x+d0.x*8,P.y+0.5,P.z+d0.z*8));
  gameCam.fov=80;gameCam.updateProjectionMatrix();
}

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
    gameState='RACING';raceTime=0;P.lapStart=0;P.sectorStart=0;_lastPos=-1;
    setFlag('green',8);
    radio('Lights out and away we go — P'+(getPos()+1)+'. Send it!');
    if(goFlashEl){goFlashEl.style.display='block';setTimeout(function(){goFlashEl.style.display='none';},180);}
  },5*750+400);
}

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

var SPARKS=(function(){
  var POOL=140,parts=[],tex=null,idx=0;
  function mkTex(){
    var c=document.createElement('canvas');c.width=c.height=16;var g=c.getContext('2d');
    var rg=g.createRadialGradient(8,8,0,8,8,8);
    rg.addColorStop(0,'rgba(255,250,235,1)');rg.addColorStop(0.4,'rgba(255,185,70,0.9)');
    rg.addColorStop(1,'rgba(255,120,0,0)');
    g.fillStyle=rg;g.fillRect(0,0,16,16);return new THREE.CanvasTexture(c);
  }
  function init(){
    if(tex) return;tex=mkTex();
    for(var i=0;i<POOL;i++){
      var m=new THREE.SpriteMaterial({map:tex,transparent:true,depthWrite:false,opacity:0,blending:THREE.AdditiveBlending,fog:false});
      var sp=new THREE.Sprite(m);sp.visible=false;sp.scale.set(0.35,0.35,1);scene.add(sp);
      parts.push({sp:sp,life:0,max:1,vx:0,vy:0,vz:0});
    }
  }
  function emit(car,intensity){
    if(!tex) init();
    var fx=Math.sin(car.heading),fz=Math.cos(car.heading);
    var n=2+((Math.random()*intensity*4)|0);
    for(var k=0;k<n;k++){
      var p=parts[idx];idx=(idx+1)%POOL;
      p.life=p.max=0.16+Math.random()*0.20;
      p.vx=-fx*(6+Math.random()*11)+(Math.random()-0.5)*6;
      p.vy=1.2+Math.random()*3.2;
      p.vz=-fz*(6+Math.random()*11)+(Math.random()-0.5)*6;
      p.sp.position.set(car.x-fx*1.7+(Math.random()-0.5)*1.2,(car.y||0)+0.12,car.z-fz*1.7+(Math.random()-0.5)*1.2);
      var s=0.22+Math.random()*0.30*intensity;p.sp.scale.set(s,s,s);
      p.sp.material.opacity=1;p.sp.visible=true;
    }
  }
  function update(dt){
    if(!tex) return;
    for(var i=0;i<parts.length;i++){
      var p=parts[i];if(p.life<=0) continue;
      p.life-=dt;
      if(p.life<=0){p.sp.visible=false;p.sp.material.opacity=0;continue;}
      p.vy-=dt*16;
      p.sp.position.x+=p.vx*dt;p.sp.position.y+=p.vy*dt;p.sp.position.z+=p.vz*dt;
      if(p.sp.position.y<0.05){p.sp.position.y=0.05;p.vy*=-0.3;p.vx*=0.6;p.vz*=0.6;}
      p.sp.material.opacity=Math.max(0,p.life/p.max);
    }
  }
  return {emit:emit,update:update,init:init};
})();

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

var FIREWORKS=(function(){
  var POOL=260,parts=[],tex=null,idx=0,bursts=0,btimer=0;
  function mkTex(){
    var c=document.createElement('canvas');c.width=c.height=32;var g=c.getContext('2d');
    var rg=g.createRadialGradient(16,16,0,16,16,16);
    rg.addColorStop(0,'rgba(255,255,255,1)');rg.addColorStop(0.35,'rgba(255,255,255,0.7)');rg.addColorStop(1,'rgba(255,255,255,0)');
    g.fillStyle=rg;g.fillRect(0,0,32,32);return new THREE.CanvasTexture(c);
  }
  function init(){
    if(tex) return;tex=mkTex();
    for(var i=0;i<POOL;i++){
      var m=new THREE.SpriteMaterial({map:tex,transparent:true,depthWrite:false,opacity:0,blending:THREE.AdditiveBlending,fog:false});
      var sp=new THREE.Sprite(m);sp.scale.set(2.2,2.2,1);sp.visible=false;scene.add(sp);
      parts.push({sp:sp,life:0,max:1,vx:0,vy:0,vz:0});
    }
  }
  var PALETTE=[0xff3b30,0xffd400,0x1e90ff,0x18c83c,0xff7ae0,0xffffff,0xff9500];
  function burst(x,y,z,col){
    if(!tex) init();
    var c=new THREE.Color(col!==undefined?col:PALETTE[(Math.random()*PALETTE.length)|0]);
    var n=22+((Math.random()*12)|0);
    for(var i=0;i<n;i++){
      var p=parts[idx];idx=(idx+1)%POOL;
      var th=Math.random()*Math.PI*2,ph=Math.acos(2*Math.random()-1),spd=8+Math.random()*10;
      p.vx=Math.sin(ph)*Math.cos(th)*spd;p.vy=Math.cos(ph)*spd;p.vz=Math.sin(ph)*Math.sin(th)*spd;
      p.max=0.9+Math.random()*0.7;p.life=p.max;
      p.sp.position.set(x,y,z);p.sp.material.color=c;p.sp.visible=true;p.sp.material.opacity=1;
    }
  }
  function start(){bursts=standAnchors.length?14:6;btimer=0;}
  function update(dt){
    if(!tex) init();
    if(bursts>0){btimer-=dt;if(btimer<=0){btimer=0.18+Math.random()*0.34;bursts--;
      var a=standAnchors.length?standAnchors[(Math.random()*standAnchors.length)|0]:{x:0,y:40,z:0,w:30};
      burst(a.x+(Math.random()-0.5)*a.w*1.5,a.y+34+Math.random()*26,a.z+(Math.random()-0.5)*a.w*1.5);
    }}
    for(var i=0;i<parts.length;i++){var p=parts[i];if(p.life<=0) continue;
      p.life-=dt;if(p.life<=0){p.sp.visible=false;p.sp.material.opacity=0;continue;}
      p.vy-=9.0*dt;p.vx*=(1-dt*0.7);p.vz*=(1-dt*0.7);
      p.sp.position.x+=p.vx*dt;p.sp.position.y+=p.vy*dt;p.sp.position.z+=p.vz*dt;
      p.sp.material.opacity=Math.max(0,p.life/p.max);
    }
  }
  return {update:update,burst:burst,start:start};
})();
var crowdBoost=0;
function crowdSurge(){crowdBoost=1.0;}
function fireworks(){FIREWORKS.start();crowdSurge();}

var WEATHER=(function(){
  var rain=[],debris=[],rtex=null,dtex=null,mode='DRY',built=false;
  var RN=240,DN=14,BOX=70,TOP=42;
  function rainTex(){
    var c=document.createElement('canvas');c.width=8;c.height=64;var g=c.getContext('2d');
    var lg=g.createLinearGradient(0,0,0,64);
    lg.addColorStop(0,'rgba(190,210,255,0)');lg.addColorStop(0.5,'rgba(205,222,255,0.9)');lg.addColorStop(1,'rgba(190,210,255,0)');
    g.fillStyle=lg;g.fillRect(0,0,8,64);return new THREE.CanvasTexture(c);
  }
  function debrisTex(){
    var c=document.createElement('canvas');c.width=c.height=16;var g=c.getContext('2d');
    g.fillStyle='rgba(150,140,110,0.7)';g.beginPath();g.arc(8,8,5,0,6.283);g.fill();return new THREE.CanvasTexture(c);
  }
  function build(){
    if(built) return;built=true;rtex=rainTex();dtex=debrisTex();
    for(var i=0;i<RN;i++){
      var m=new THREE.SpriteMaterial({map:rtex,transparent:true,depthWrite:false,opacity:0,fog:false});
      var sp=new THREE.Sprite(m);sp.scale.set(0.07,1.5,1);sp.visible=false;scene.add(sp);rain.push({sp:sp,vy:0});
    }
    for(var j=0;j<DN;j++){
      var dm=new THREE.SpriteMaterial({map:dtex,transparent:true,depthWrite:false,opacity:0,fog:false});
      var ds=new THREE.Sprite(dm);ds.scale.set(0.5,0.5,1);ds.visible=false;scene.add(ds);debris.push({sp:ds,ang:Math.random()*6.28});
    }
  }
  function seed(p,cam){
    p.sp.position.set(cam.x+(Math.random()-0.5)*BOX*2,cam.y+TOP*Math.random(),cam.z+(Math.random()-0.5)*BOX*2);
    p.vy=46+Math.random()*26;p.sp.material.opacity=0.5+Math.random()*0.4;p.sp.visible=true;
  }
  function set(w){
    build();mode=w.id;
    for(var i=0;i<rain.length;i++){rain[i].sp.visible=false;rain[i].sp.material.opacity=0;}
    for(var j=0;j<debris.length;j++){debris[j].sp.visible=false;debris[j].sp.material.opacity=0;}
  }
  function update(dt,cam){
    if(!built) return;
    if(mode==='WET'){
      for(var i=0;i<rain.length;i++){var p=rain[i];
        if(!p.sp.visible){seed(p,cam);continue;}
        p.sp.position.y-=p.vy*dt;p.sp.position.x+=dt*4;
        if(p.sp.position.y<cam.y-18||Math.abs(p.sp.position.x-cam.x)>BOX||Math.abs(p.sp.position.z-cam.z)>BOX) seed(p,cam);
      }
    } else if(mode==='WINDY'){
      for(var j=0;j<debris.length;j++){var d=debris[j];d.ang+=dt*1.5;
        if(d.sp.material.opacity<=0){d.sp.position.set(cam.x+(Math.random()-0.5)*60,cam.y+2+Math.random()*10,cam.z+(Math.random()-0.5)*60);d.sp.material.opacity=0.6;d.sp.visible=true;}
        d.sp.position.x+=dt*(14+Math.sin(d.ang)*4);d.sp.position.y+=Math.sin(d.ang*2)*dt*3;d.sp.material.opacity-=dt*0.12;
        if(Math.abs(d.sp.position.x-cam.x)>50) d.sp.material.opacity=0;
      }
    }
  }
  return {set:set,update:update};
})();
function applyWeatherVisuals(w){
  if(w.id==='WET'){scene.fog.color.setHex(0x1a2440);scene.fog.density=0.0030;renderer.toneMappingExposure=0.82;}
  else if(w.id==='HOT'){scene.fog.color.setHex(0x2a1a18);scene.fog.density=0.0016;renderer.toneMappingExposure=1.18;}
  else if(w.id==='WINDY'){scene.fog.color.setHex(0x10141f);scene.fog.density=0.0020;renderer.toneMappingExposure=1.02;}
  else {scene.fog.color.setHex(CITY_FOG);scene.fog.density=0.0017;renderer.toneMappingExposure=1.0;}
}

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
  if(P.dnf){
    P.thr=0;P.brk=0;P.drs=false;P.steerIn=0;
    if(hudDrs) hudDrs.className='hud-drs';
    P.speed=Math.max(0,P.speed-dt*6);
    var ddP=wpDir(P.tIdx),ppP=wpPerp(P.tIdx),wwP=waypoints[P.tIdx];
    P.heading=Math.atan2(ddP.x,ddP.z);
    P.x+=Math.sin(P.heading)*P.speed*dt;P.z+=Math.cos(P.heading)*P.speed*dt;
    var edgeP=-(TW*0.5);
    P.x+=((wwP.x+ppP.x*edgeP)-P.x)*Math.min(dt*1.2,1);
    P.z+=((wwP.z+ppP.z*edgeP)-P.z)*Math.min(dt*1.2,1);
    P.tIdx=closestWP(P.x,P.z,P.tIdx);P.y=waypoints[P.tIdx].y;
    if(Math.random()<0.5) SMOKE.emit(P,-1,1,1.3);
    if(P.speed<0.3){P._dnfRest=(P._dnfRest||0)+dt;if(P._dnfRest>2.5&&!P._dnfDone) dnfToClassification();}
    return;
  }
  var thr=(keys['ArrowUp']||keys['KeyW'])?1:0;
  var brk=(keys['ArrowDown']||keys['KeyS'])?1:0;
  P.thr=thr;P.brk=brk;
  var sl=(keys['ArrowLeft']||keys['KeyA'])?1:0;
  var sr=(keys['ArrowRight']||keys['KeyD'])?1:0;
  var si=sl-sr;P.steerIn=si;
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
  if(keys['KeyP']&&!P._pitKey&&P.pitState==='NONE'){P._pitArmed=!P._pitArmed;if(P._pitArmed&&P._pitCompound==null)P._pitCompound=P.compound;radio(P._pitArmed?'Copy, box this lap — pick a compound (1/2/3), pit after the line.':'Stay out, box cancelled.',false);}
  P._pitKey=keys['KeyP'];
  if(P._pitArmed||P.pitState==='LANE'){if(keys['Digit1'])P._pitCompound=0;else if(keys['Digit2'])P._pitCompound=1;else if(keys['Digit3'])P._pitCompound=2;}
  if(P._pitArmed&&P.pitState==='NONE'&&P.tIdx>=2&&P.tIdx<=22&&P.lap<LAPS){
    P.pitState='LANE';P.pitS=pitSFromX(P.x);P._pitArmed=false;
    var _epp=pitPointAt(P.pitS);P._pitErrZ=P.z-_epp.z;P._pitErrH=P.heading-_epp.heading;
  }
  // Drive-in pit entry: commit only when the player has clearly left the track onto the pit entry
  // apron — no P-press required. The zone starts at the apron edge (z=50.5; track edge+curb at
  // x=-280 is ~48.9 with TW=16.8 and shrinks downstream), so running wide on the straight or the
  // final-corner exit never triggers it; you have to deliberately steer across the curb into the
  // entry road.
  if(P.pitState==='NONE'&&P.lap<LAPS&&P.x>-280&&P.x<-160&&P.z>=50.5&&P.z<70){
    P.pitState='LANE';P.pitS=pitSFromX(P.x);P._pitArmed=false;
    if(P._pitCompound==null)P._pitCompound=P.compound;
    var _epp2=pitPointAt(P.pitS);P._pitErrZ=P.z-_epp2.z;P._pitErrH=P.heading-_epp2.heading;
    radio('In the lane — box this lap, crew is ready.',false);
  }
  if(P.pitState!=='NONE'){pitMove(P,dt,true);playerCrossings();return;}
  var towBoost=1.0;
  var _tpp=wpPerp(P.tIdx),_tpw=waypoints[P.tIdx];
  var _myLatT=(P.x-_tpw.x)*_tpp.x+(P.z-_tpw.z)*_tpp.z;
  for(var _ti=0;_ti<AI.length;_ti++){
    var _oc=AI[_ti],_gg=((_oc.tIdx-P.tIdx)+N)%N;
    if(inPit(_oc)||_oc.dnf) continue;
    if(_gg<1||_gg>25) continue;
    var _ocl=_oc._latTarget!==undefined?_oc._latTarget:0;
    if(Math.abs(_myLatT-_ocl)<TW*0.26){towBoost=Math.max(towBoost,1.0+0.04*(1-_gg/25));}
  }
  var gF=COMPOUNDS[P.compound].grip*weather.gripMul*tempGrip(P);
  var maxSpd=(P.drs?MAX_SPD*1.08:MAX_SPD)*(0.93+0.07*P.tireWear)*towBoost*(0.85+0.15*gF)*(1-P.damage*0.10);
  P._tow=towBoost;
  var acc=(thr*ENG-brk*BRK-DRAG*P.speed*P.speed)/CAR_MASS;
  P.speed=Math.max(0,Math.min(P.speed+acc*dt,maxSpd));
  var ms=Math.PI/5*(1-0.6*P.speed/MAX_SPD);
  var ty=si*ms*Math.min(P.speed/5,1);
  P.yawRate+=(ty-P.yawRate)*Math.min(dt*6,1);P.yawRate*=Math.max(0,1-dt*7.5);
  var latLoad=Math.abs(P.yawRate)*P.speed;
  var gripLim=GRIP_LIMIT*gF*(1-P.dmgFront*0.30);
  if(latLoad>gripLim){P.speed=Math.max(0,P.speed-(latLoad-gripLim)*GRIP_SCRUB*dt);P.gripLoss=true;}
  else P.gripLoss=false;
  P.heading+=P.yawRate*P.speed*0.6*dt;
  P.x+=Math.sin(P.heading)*P.speed*dt;
  P.z+=Math.cos(P.heading)*P.speed*dt;
  if(weather.id==='WINDY'&&P.speed>8){var gst=Math.sin(raceTime*1.7)*0.9*dt;P.x+=Math.cos(P.heading)*gst;P.z-=Math.sin(P.heading)*gst;}
  P.tIdx=closestWP(P.x,P.z,P.tIdx);
  P.y=waypoints[P.tIdx].y;
  var _bwp=waypoints[P.tIdx],_bp=wpPerp(P.tIdx);
  var _lat=(P.x-_bwp.x)*_bp.x+(P.z-_bwp.z)*_bp.z;
  var _maxL=TW*0.5+CURB,_absL=Math.abs(_lat);
  if(P._pitExitGrace>0) P._pitExitGrace-=dt;
  // While peeling into the pit-entry corridor, don't shake or snap the car back onto the track —
  // it has to cross the gap from the racing line (Z~40) to the lane (Z>=50) to enter the pit.
  var _pitAppr=(P.pitState==='NONE'&&P.x>-315&&P.x<-150&&P.z>44&&P.z<70);
  if(_absL>TW*0.5&&_absL<_maxL&&P.speed>10&&!(P._pitExitGrace>0)&&!_pitAppr){
    P._shake=Math.max(P._shake||0,0.45);
    if(Math.random()<0.6) SPARKS.emit(P,0.8);
  }
  if(_absL>_maxL&&!_pitAppr){var _sgn=_lat>0?1:-1;
    if(P._pitExitGrace>0){var _ti=_bwp.x+_bp.x*_sgn*(_maxL-0.4),_tj=_bwp.z+_bp.z*_sgn*(_maxL-0.4);P.x+=(_ti-P.x)*Math.min(dt*4,1);P.z+=(_tj-P.z)*Math.min(dt*4,1);}
    else {P.x=_bwp.x+_bp.x*_sgn*_maxL;P.z=_bwp.z+_bp.z*_sgn*_maxL;P.speed*=0.45;P._shake=1.3;}
  }
  var GB=[0,0.12,0.25,0.38,0.54,0.70,0.86,1.0];
  var rpm=P.speed/maxSpd;P.gear=7;
  for(var gi=1;gi<GB.length-1;gi++){if(rpm<GB[gi+1]){P.gear=gi;break;}}
  var gLo=GB[Math.max(P.gear-1,0)],gHi=GB[P.gear];
  P.rpmVal=gHi>gLo?Math.max(0,Math.min(1,(rpm-gLo)/(gHi-gLo))):0;
  var slp=Math.abs(P.yawRate)*P.speed;
  P.tireWear=Math.max(0.02,P.tireWear-slp*dt*0.00004*COMPOUNDS[P.compound].wear*weather.wearMul);
  if(P.tireWear<0.22&&!P._boxWarned&&P.pitState==='NONE'&&P.lap<LAPS){P._boxWarned=true;radio('Tyres are gone — box now: dive into the pit lane (or press P).');}
  if(rollMechFailure(P,dt,0.0006)) return;
  P.tireTempF=Math.min(1,Math.max(0.05,P.tireTempF+(thr*0.3+slp*0.08)*dt*0.08-dt*0.02+weather.tempBias*dt*0.06));
  P.tireTempR=Math.min(1,Math.max(0.05,P.tireTempR+(thr*0.5+brk*0.3)*dt*0.08-dt*0.02+weather.tempBias*dt*0.06));
  if(brk===1&&P.speed>12){SMOKE.emit(P,1,1,1.0);if(P.speed>22){P._shake=Math.max(P._shake||0,0.3);if(Math.random()<0.4) SPARKS.emit(P,0.6);}}
  if((thr===1&&P.speed<7&&P.speed>0.2)||slp>4.5) SMOKE.emit(P,-1,1,slp>4.5?1.2:0.6);
  if(P.gripLoss) P._shake=Math.max(P._shake||0,0.4);
  playerCrossings();
}

// Sector/lap crossing checks for the player. Called from the normal drive path AND from the
// pit path (pitMove keeps tIdx tracking along the lane, which parallels the S/F straight), so
// a lap driven through the pits is still credited and the standings position holds.
function playerCrossings(){
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
    var lapT=P.s1+P.s2+P.s3;
    if(lapT>0&&lapT<P.bestLap){if(P.bestLap<Infinity) radio('Fastest lap! &nbsp;'+fmt(lapT),true);P.bestLap=lapT;}
    P.lap++;P.sector=0;P.sectorStart=raceTime;P.lapStart=raceTime;
    if(P.lap>LAPS){P.lap=LAPS;P.finishTime=raceTime;gameState='FINISHED';setFlag('chequered',999);fireworks();radio('Chequered flag! P'+(getPos()+1)+' — that\\'s the race.',true);showPodium();}
    else if(P.lap===LAPS) radio('White flag — last lap. Leave it all out there.');
    else if(P.lap===2&&P.pitsDone===0) radio('Tyres will fade through the stint — press P to box for fresh rubber.');
  }
  if(P.tIdx>=20) P._lapGuard=false;
}

function setSec(el,cls,t,best){
  if(!el) return;
  el.className=cls+(t>0&&t<best?' purple':' yellow');
  el.textContent=(cls==='hud-s1'?'S1 ':cls==='hud-s2'?'S2 ':'S3 ')+(t>0?t.toFixed(3):'---.---');
}

// AI S/F crossing — also run while in the pits so a lap driven through the lane is credited.
function aiCrossing(ai){
  if(ai.tIdx<12&&!ai._inStart){
    ai._inStart=true;
    ai.lap++;if(ai.lap>LAPS){ai.lap=LAPS;if(ai.finishTime===Infinity)ai.finishTime=raceTime;}
  }
  if(ai.tIdx>=20) ai._inStart=false;
}

function updateAI(ai,dt){
  if(gameState!=='RACING') return;
  if(inPit(ai)){pitMove(ai,dt,false);aiCrossing(ai);return;}
  if(ai.dnf){
    if(ai._recovered) return;
    if(ai._towed) return;
    ai.speed=Math.max(0,ai.speed-dt*7);
    var ddR=wpDir(ai.tIdx),ppR=wpPerp(ai.tIdx),wwR=waypoints[ai.tIdx];
    ai.heading=Math.atan2(ddR.x,ddR.z);
    ai.x+=Math.sin(ai.heading)*ai.speed*dt;ai.z+=Math.cos(ai.heading)*ai.speed*dt;
    var edge=(ai._initOvSide>0?1:-1)*(TW*0.5);
    ai.x+=((wwR.x+ppR.x*edge)-ai.x)*Math.min(dt*1.2,1);
    ai.z+=((wwR.z+ppR.z*edge)-ai.z)*Math.min(dt*1.2,1);
    ai.tIdx=closestWP(ai.x,ai.z,ai.tIdx);ai.y=waypoints[ai.tIdx].y;
    if(Math.random()<0.4) SMOKE.emit(ai,-1,1,1.3);
    if(ai.speed<0.3&&!ai._awaitRecovery){ai._awaitRecovery=true;recoverQueue.push(ai._idx);}
    return;
  }

  ai._moodPhase+=dt*(0.5+ai._randFac);
  var moodAmp=0.03*(1.4-ai.skill);
  ai._mood=1.0+Math.sin(ai._moodPhase)*moodAmp+Math.sin(ai._moodPhase*2.3)*moodAmp*0.4;

  ai._deg+=dt*0.00022*(0.6+ai._aggr*0.7)*COMPOUNDS[ai.compound].wear*weather.wearMul;
  if(rollMechFailure(ai,dt,0.00026)) return;
  var degFac=Math.max(0.93,1.0-ai._deg);
  var aiGrip=0.97+0.03*COMPOUNDS[ai.compound].grip*weather.gripMul;

  var maxSpd=(MAX_SPD*ai.paceFac*ai._mood*degFac*aiGrip+MAX_SPD*ai._randFac)*(1-ai.damage*0.12);

  var curvature=0,turnSign=0;
  for(var k=5;k<=55;k+=5){
    var dA=wpDir((ai.tIdx+k)%N),dB=wpDir((ai.tIdx+k+5)%N);
    var cr=dA.x*dB.z-dA.z*dB.x,absCr=Math.abs(cr);
    if(absCr>curvature) curvature=absCr;
    if(!turnSign&&k<=35&&absCr>0.018) turnSign=cr>0?1:-1;
  }
  curvature=Math.min(1,curvature*4.0);
  var onStraight=curvature<0.12;

  if(ai._mistakeTimer<=0&&raceTime>3.0&&Math.random()<0.012*(1.0-ai.skill)*dt*(1.0+curvature*2.0)){
    ai._mistakeKind=Math.random()<0.55?1:2;
    ai._mistakeTimer=ai._mistakeKind===1?0.7+Math.random()*0.6:0.9+Math.random()*0.8;
    var _ddp=ai.x-P.x,_ddpz=ai.z-P.z;
    if(ai._name&&_ddp*_ddp+_ddpz*_ddpz<6400&&Math.random()<0.7) radio(ai._name+(ai._mistakeKind===1?' has locked up!':' has gone wide!'));
  }

  var brkFac=Math.max(0.16,0.32-ai.skill*0.12-(ai._aggr-1.0)*0.08)+ai.dmgFront*0.12;
  var spdFloor=Math.min(0.66,0.52+ai.skill*0.12);
  var targetSpd=Math.max(maxSpd*spdFloor,maxSpd*(1-curvature*brkFac));
  var launching=raceTime<2.0;

  var bp0=wpPerp(ai.tIdx),bw0=waypoints[ai.tIdx];
  var myLat=ai._latTarget;
  var allCars=[P].concat(AI);
  var fwdCar=null,fwdGap=999;
  if(!launching){
    for(var j=0;j<allCars.length;j++){
      var oth=allCars[j];if(oth===ai||inPit(oth)||oth.dnf) continue;
      var g=((oth.tIdx-ai.tIdx)+N)%N;
      if(g<1||g>40) continue;
      var othLat=oth._latTarget!==undefined?oth._latTarget:((oth.x-bw0.x)*bp0.x+(oth.z-bw0.z)*bp0.z);
      if(Math.abs(myLat-othLat)<TW*0.24&&g<fwdGap){fwdGap=g;fwdCar=oth;}
    }
    if(fwdCar){
      var closingSpd=ai.speed-fwdCar.speed;
      if(closingSpd>0){
        var allowedClose=Math.min(fwdGap*1.1,12.0);
        if(closingSpd>allowedClose) targetSpd=Math.min(targetSpd,fwdCar.speed+allowedClose);
      }
      var desiredGap=ai._ovTimer>0?2.6:3.4;
      if(fwdGap<desiredGap){
        var gapFac=fwdGap/desiredGap;
        targetSpd=Math.min(targetSpd,fwdCar.speed*(0.90+0.08*gapFac));
      }
      if(fwdGap<1.6&&ai.speed>fwdCar.speed) targetSpd=Math.min(targetSpd,fwdCar.speed*0.85);
    }
    if(fwdCar&&fwdGap<40) ai._stuckTimer+=dt;
    else ai._stuckTimer=Math.max(0,ai._stuckTimer-dt*1.4);
  }
  var behindCar=null,behindGap=999,behindLat=0;
  if(!launching){
    for(var jb=0;jb<allCars.length;jb++){
      var ob=allCars[jb];if(ob===ai||inPit(ob)||ob.dnf) continue;
      var gb=((ai.tIdx-ob.tIdx)+N)%N;
      if(gb<1||gb>14) continue;
      if(ob.speed<=ai.speed-0.5) continue;
      if(gb<behindGap){behindGap=gb;behindCar=ob;
        behindLat=ob._latTarget!==undefined?ob._latTarget:((ob.x-bw0.x)*bp0.x+(ob.z-bw0.z)*bp0.z);}
    }
  }
  // Wheel-to-wheel awareness: nearest car longitudinally overlapping this one (within ~half a
  // car length fore/aft). Uses the actual projected lateral position, not _latTarget — the
  // player has no _latTarget, and a mid-move AI can be far from its target lane.
  var alongCar=null,alongLat=0,alongDist=999;
  if(!launching){
    for(var ja=0;ja<allCars.length;ja++){
      var oa=allCars[ja];if(oa===ai||inPit(oa)||oa.dnf) continue;
      var ga=((oa.tIdx-ai.tIdx)+N)%N;
      if(ga>2&&ga<N-2) continue;
      var oaLat=(oa.x-bw0.x)*bp0.x+(oa.z-bw0.z)*bp0.z;
      var oaD=Math.abs(myLat-oaLat);
      if(oaD<4.2&&oaD<alongDist){alongDist=oaD;alongCar=oa;alongLat=oaLat;}
    }
  }
  if(ai._mistakeTimer>0&&ai._mistakeKind===1){
    targetSpd=Math.min(targetSpd,maxSpd*0.45);
    if(ai.speed>14&&Math.random()<0.4){SMOKE.emit(ai,1,1,0.9);SPARKS.emit(ai,0.9);}
  }

  var slipBoost=1.0;
  if(!launching){
    for(var js=0;js<allCars.length;js++){
      var oS=allCars[js];if(oS===ai||inPit(oS)) continue;
      var gS=((oS.tIdx-ai.tIdx)+N)%N;
      if(gS<1||gS>25) continue;
      var oSLat=oS._latTarget!==undefined?oS._latTarget:((oS.x-bw0.x)*bp0.x+(oS.z-bw0.z)*bp0.z);
      if(Math.abs(myLat-oSLat)<TW*0.22){slipBoost=1.10;break;}
    }
  }
  if(slipBoost>1.0&&drsZoneIndex(ai.tIdx)>=0&&gapAheadSec(ai)<1.0) slipBoost=1.16;
  maxSpd*=slipBoost;
  if(ai._ovTimer>0) maxSpd*=1.06;

  var force=targetSpd>ai.speed?ENG:-BRK;
  ai._braking=(force<0);
  ai.speed=Math.max(0,Math.min(ai.speed+(force/CAR_MASS)*dt,maxSpd));
  if(force<0&&ai.speed>15&&curvature>0.35&&Math.random()<0.18){SMOKE.emit(ai,1,1,0.7);if(Math.random()<0.5)SPARKS.emit(ai,0.6);}

  var halfTW=TW*0.44;
  var desiredLat=ai._baseLat;

  if(raceTime<2.5){
    desiredLat=ai._baseLat;ai._ovTimer=0;

  } else if(ai._ovTimer>0){
    ai._ovTimer-=dt;
    // Re-check the target lane every frame during the commit — the car being passed (or a
    // third car) can drift into it. Try flipping to the other side of fwdCar first; if both
    // sides are taken, abort back to the base line rather than press into contact.
    var laneBlocked=function(tgt){
      for(var jr=0;jr<allCars.length;jr++){
        var orC=allCars[jr];if(orC===ai||inPit(orC)||orC.dnf) continue;
        var gr=((orC.tIdx-ai.tIdx)+N)%N;
        if(gr>10&&(N-gr)>2) continue;
        if(Math.abs(tgt-((orC.x-bw0.x)*bp0.x+(orC.z-bw0.z)*bp0.z))<2.9) return true;
      }
      return false;
    };
    if(laneBlocked(ai._ovTgt)){
      var swapped=false;
      if(fwdCar){
        var fLat2=fwdCar._latTarget!==undefined?fwdCar._latTarget:((fwdCar.x-bw0.x)*bp0.x+(fwdCar.z-bw0.z)*bp0.z);
        var mir=Math.max(-halfTW,Math.min(halfTW,fLat2*2-ai._ovTgt));
        if(Math.abs(mir-fLat2)>2.9&&!laneBlocked(mir)){ai._ovTgt=mir;swapped=true;}
      }
      if(!swapped) ai._ovTimer=0;
    }
    if(ai._ovTimer>0){
      desiredLat=ai._ovTgt;
      if(!fwdCar||fwdGap>55){ai._ovTimer=0;ai._baseLat=ai._ovTgt;}
    }

  } else if(curvature>0.22){
    desiredLat=(fwdCar&&fwdGap<25)?ai._baseLat:-turnSign*halfTW*ai.apexD;

  } else {
    if(fwdCar&&fwdGap<40){
      var fwdLat=fwdCar._latTarget!==undefined?fwdCar._latTarget:((fwdCar.x-bw0.x)*bp0.x+(fwdCar.z-bw0.z)*bp0.z);
      var oDir=Math.abs(ai._latTarget)>halfTW*0.65?-Math.sign(ai._latTarget):ai._ovSide;
      var oTgt=Math.max(-halfTW,Math.min(halfTW,fwdLat+oDir*TW*0.35));
      var doOvertake=ai._stuckTimer>0.35||((fwdCar.paceFac!==undefined)&&(ai.paceFac-fwdCar.paceFac)>-0.004)||(slipBoost>1.0&&fwdGap<18)||(ai._aggr>1.05&&fwdGap<20);
      if(doOvertake){
        for(var kk=0;kk<allCars.length;kk++){
          var ocb=allCars[kk];if(ocb===ai||ocb===fwdCar||inPit(ocb)||ocb.dnf) continue;
          var og2=((ocb.tIdx-ai.tIdx)+N)%N;
          if(og2>16||(N-og2)>5) continue;
          var ocbLat=ocb._latTarget!==undefined?ocb._latTarget:((ocb.x-bw0.x)*bp0.x+(ocb.z-bw0.z)*bp0.z);
          if(Math.abs(oTgt-ocbLat)<TW*0.17){doOvertake=false;break;}
        }
      }
      if(doOvertake){ai._ovTgt=oTgt;ai._ovTimer=3.5;ai._ovSide*=-1;desiredLat=oTgt;}
      else desiredLat=ai._baseLat;
    } else {
      desiredLat=ai._baseLat;
    }
    // Defend only while the attacker is still fully behind — once it has a nose alongside
    // (behindGap<3 or an overlap exists) the move would be a swerve into contact, not a block.
    if(ai._ovTimer<=0&&behindCar&&behindGap>=3&&!alongCar){
      var defStr=Math.max(0,(ai.skill-0.3)/0.55)*Math.min(1,0.55+ai._aggr*0.5)*((15-behindGap)/14);
      if(defStr>0){
        var coverLat=Math.max(-halfTW,Math.min(halfTW,behindLat));
        desiredLat=desiredLat+(coverLat-desiredLat)*Math.min(0.6,defStr);
      }
    }
  }
  var hazLat=null,hazGap=999;
  for(var jh=0;jh<allCars.length;jh++){
    var oh=allCars[jh];
    if(oh===ai||inPit(oh)||!oh.dnf||oh._recovered||oh._towed) continue;
    var gh=((oh.tIdx-ai.tIdx)+N)%N;
    if(gh<1||gh>18) continue;
    var ohLat=(oh.x-bw0.x)*bp0.x+(oh.z-bw0.z)*bp0.z;
    if(Math.abs(ai._latTarget-ohLat)<TW*0.34&&gh<hazGap){hazGap=gh;hazLat=ohLat;}
  }
  if(hazLat!==null) desiredLat=hazLat>0?-halfTW:halfTW;
  if(ai._mistakeTimer>0&&ai._mistakeKind===2){
    desiredLat=turnSign?turnSign*halfTW:Math.sign(ai._latTarget||1)*halfTW*0.8;
    ai.speed*=Math.max(0,1-dt*0.45);
  }
  if(ai._mistakeTimer>0) ai._mistakeTimer-=dt;

  // Hard space-yield: whatever the steering decision above (apex line, overtake, defend,
  // even a gone-wide mistake), never aim within a car-width-plus-margin of a car alongside.
  if(alongCar){
    var minSep=2.9,sep=desiredLat-alongLat;
    if(Math.abs(sep)<minSep){
      var sgnY=sep!==0?Math.sign(sep):(myLat>=alongLat?1:-1);
      desiredLat=alongLat+sgnY*minSep;
    }
    if(ai._ovTimer>0&&Math.abs(ai._ovTgt-alongLat)<minSep) ai._ovTimer=0;
  }

  var wallEdge=halfTW*0.78;
  if(Math.abs(desiredLat)>wallEdge) desiredLat-=Math.sign(desiredLat)*(Math.abs(desiredLat)-wallEdge)*1.4;
  desiredLat=Math.max(-halfTW,Math.min(halfTW,desiredLat));
  ai._latTarget+=(desiredLat-ai._latTarget)*Math.min(dt*5.0,1);
  ai._latTarget=Math.max(-halfTW,Math.min(halfTW,ai._latTarget));

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

  var bp=wpPerp(ai.tIdx),bw=waypoints[ai.tIdx];
  var lat=(ai.x-bw.x)*bp.x+(ai.z-bw.z)*bp.z;
  if(Math.abs(lat)>TW*0.5+CURB){
    var s=lat>0?1:-1;
    ai.x=bw.x+bp.x*s*(TW*0.5+CURB);
    ai.z=bw.z+bp.z*s*(TW*0.5+CURB);
    ai.speed*=0.80;
    ai._latTarget=-s*halfTW*0.5;
    ai._ovTimer=0;
    SMOKE.emit(ai,1,2,1.2);
  }

  aiCrossing(ai);
  if(ai.pitState==='NONE'&&ai.pitLap&&ai.lap===ai.pitLap&&ai.tIdx>=2&&ai.tIdx<=22&&ai.lap<LAPS){
    ai.pitState='LANE';ai.pitS=pitSFromX(ai.x);ai.pitLap=0;
    var remain=LAPS-ai.lap;
    var nc=weather.best+(remain>=3?1:0)-(remain<=1?1:0)+(Math.random()<0.25?(Math.random()<0.5?-1:1):0);
    ai._nextCompound=Math.max(0,Math.min(2,nc));
    var _epp=pitPointAt(ai.pitS);ai._pitErrZ=ai.z-_epp.z;ai._pitErrH=ai.heading-_epp.heading;
  }
}

var CAR_HL=2.55,CAR_HW=1.05;
var CRASH_DNF=0.62,DMG_FRONT_K=0.016,DMG_REAR_K=0.010,DMG_SIDE_K=0.09;
function applyDamage(c,amt,region){
  if(c.dnf||amt<=0) return;
  var early=c.lap<2;
  if(early) amt*=0.5;
  if(region==='front') c.dmgFront=Math.min(1,c.dmgFront+amt);
  else if(region==='rear') c.dmgRear=Math.min(1,c.dmgRear+amt);
  else c.dmgFront=Math.min(1,c.dmgFront+amt*0.3);
  c.damage=Math.min(0.95,c.damage+amt);
  var dnfHit=early?0.85:CRASH_DNF;
  if(amt>=dnfHit||(c.damage>=0.9&&amt>=0.10&&!early)){
    var reason=c.dmgFront>=0.8?'front wing torn off in the contact':
               (region==='rear'?'rear-end hit — suspension broken':'heavy contact, too much damage');
    retire(c,reason);
  }
}
var MECH_FAILS=['engine failure','gearbox stuck in gear','hydraulics failure',
  'brake-by-wire failure','power-unit shutdown','oil pressure — engine let go'];
function rollMechFailure(c,dt,baseRate){
  if(c.dnf||gameState!=='RACING'||c.lap<2) return false;
  var r=baseRate;
  if(c.tireWear!==undefined&&c.tireWear<0.12) r*=1.8;
  if(Math.random()<r*dt){
    var reason=(c.tireWear!==undefined&&c.tireWear<0.18&&Math.random()<0.55)
      ?'puncture — tyre let go':MECH_FAILS[Math.floor(Math.random()*MECH_FAILS.length)];
    retire(c,reason);return true;
  }
  return false;
}
function retire(c,reason){
  if(c.dnf) return;
  c.dnf=true;c.dnfReason=reason||'mechanical failure';
  raiseYellow(c);
  if(c===P) retirePlayer(c.dnfReason);
  else { radio(c._name+' is out — '+c.dnfReason+'. Into retirement.',false); }
}
function retirePlayer(reason){
  if(P._retired) return; P._retired=true; P.dnf=true;
  P.dnfReason=reason||'mechanical failure';
  radio('That\\'s our race done — '+P.dnfReason+'. Bring it to a stop, mate.',true);
  showDnf(P.dnfReason);
}
function scDrive(dt,spd,edge){
  SC.speed+=(spd-SC.speed)*Math.min(dt*2,1);
  var dd=wpDir(SC.tIdx);SC.heading=Math.atan2(dd.x,dd.z);
  SC.x+=Math.sin(SC.heading)*SC.speed*dt;SC.z+=Math.cos(SC.heading)*SC.speed*dt;
  SC.tIdx=closestWP(SC.x,SC.z,SC.tIdx);
  var pp=wpPerp(SC.tIdx),wp=waypoints[SC.tIdx];
  SC.x+=((wp.x+pp.x*edge)-SC.x)*Math.min(dt*1.5,1);
  SC.z+=((wp.z+pp.z*edge)-SC.z)*Math.min(dt*1.5,1);
  SC.y=wp.y;
  if(safetyCarGrp){safetyCarGrp.position.set(SC.x,SC.y+RIDE_H,SC.z);safetyCarGrp.rotation.y=SC.heading-Math.PI/2;wheelSync(safetyCarGrp,SC,dt);}
}
function scRecover(ai){
  ai._recovered=true;ai._towed=false;
  if(aiGrps[ai._idx]) aiGrps[ai._idx].visible=false;
  if(aiLabels[ai._idx]) aiLabels[ai._idx].visible=false;
  if(recoverQueue[0]===ai._idx) recoverQueue.shift();
  SC.target=-1;SC.state='RETURN';SC.rtimer=0;
}
function updateSafetyCar(dt){
  if(gameState!=='RACING'){if(safetyCarGrp) safetyCarGrp.visible=false;return;}
  if(!safetyCarGrp) return;
  if(safetyCarGrp.visible&&safetyCarGrp.userData.beacon){
    var on=Math.sin(raceTime*12)>0;
    safetyCarGrp.userData.beacon[0].emissiveIntensity=on?2.2:0.15;
    safetyCarGrp.userData.beacon[1].emissiveIntensity=on?0.15:2.2;
  }
  if(SC.state==='IDLE'){
    while(recoverQueue.length){var q=recoverQueue[0],t=AI[q];if(t&&t.dnf&&!t._recovered) break;recoverQueue.shift();}
    if(!recoverQueue.length) return;
    SC.target=recoverQueue[0];var tgt=AI[SC.target];
    var tp=wpPerp(tgt.tIdx),twp=waypoints[tgt.tIdx];
    SC.edge=(tgt.x-twp.x)*tp.x+(tgt.z-twp.z)*tp.z;
    SC.tIdx=((tgt.tIdx-16)+N)%N;
    var sp=wpPerp(SC.tIdx),swp=waypoints[SC.tIdx],sd=wpDir(SC.tIdx);
    SC.x=swp.x+sp.x*SC.edge;SC.z=swp.z+sp.z*SC.edge;SC.y=swp.y;SC.speed=0;
    SC.heading=Math.atan2(sd.x,sd.z);SC.timer=0;
    safetyCarGrp.position.set(SC.x,SC.y+RIDE_H,SC.z);safetyCarGrp.rotation.y=SC.heading-Math.PI/2;
    safetyCarGrp.visible=true;raiseYellow();SC.state='DISPATCH';
    return;
  }
  if(SC.state==='DISPATCH'){
    var tgt=AI[SC.target];
    if(!tgt||!tgt.dnf||tgt._recovered){SC.state='RETURN';SC.rtimer=0;return;}
    SC.timer+=dt;raiseYellow();
    scDrive(dt,30,SC.edge);
    var gap=((tgt.tIdx-SC.tIdx)+N)%N,dx=tgt.x-SC.x,dz=tgt.z-SC.z;
    if(gap<=3||gap>N-3||(dx*dx+dz*dz)<36){tgt._towed=true;SC.state='TOW';}
    else if(SC.timer>RECOVER_MAX){scRecover(tgt);}
    return;
  }
  if(SC.state==='TOW'){
    var tgt=AI[SC.target];
    if(!tgt){SC.state='RETURN';SC.rtimer=0;return;}
    SC.timer+=dt;raiseYellow();
    scDrive(dt,26,SC.edge);
    var bwp=((SC.tIdx-7)+N)%N,pp=wpPerp(bwp),wp=waypoints[bwp],dd=wpDir(bwp);
    tgt.tIdx=bwp;tgt.x=wp.x+pp.x*SC.edge;tgt.z=wp.z+pp.z*SC.edge;tgt.y=wp.y;
    tgt.heading=Math.atan2(dd.x,dd.z);tgt.speed=SC.speed;
    if(Math.random()<0.25) SMOKE.emit(tgt,-1,1,0.8);
    var fwd=((PIT_ENTRY_WP-tgt.tIdx)+N)%N;
    if(fwd<=4||fwd>N-4||SC.timer>RECOVER_MAX) scRecover(tgt);
    return;
  }
  if(SC.state==='RETURN'){
    SC.rtimer+=dt;scDrive(dt,30,SC.edge);
    var fwd=((PIT_ENTRY_WP-SC.tIdx)+N)%N;
    if(fwd<=4||fwd>N-4||SC.rtimer>4){
      safetyCarGrp.visible=false;SC.state='IDLE';
      if(flagState==='yellow') flagState='none';
    }
    return;
  }
}
function applyDamageVisual(grp,c){
  if(!grp||!grp.userData) return;
  var u=grp.userData;
  if(u.fw){
    if(c.dmgFront>=0.8){ if(u.fw.visible){u.fw.visible=false;SPARKS.emit(c,1.0);} }
    else { u.fw.visible=true; u.fw.rotation.z=-c.dmgFront*0.32; }
  }
  if(u.rw) u.rw.rotation.z=c.dmgRear*0.40;
  if(u.bodyMat){
    var d=c.damage;
    if(u._baseCol===undefined) u._baseCol=u.bodyMat.color.clone();
    u.bodyMat.roughness=0.22+d*0.55;
    u.bodyMat.clearcoat=1.0-d*0.85;
    u.bodyMat.color.copy(u._baseCol).multiplyScalar(1-d*0.45);
  }
}
function resolveCollisions(){
  var cars=[P].concat(AI);
  for(var pass=0;pass<2;pass++){
    for(var i=0;i<cars.length;i++){
      for(var j=i+1;j<cars.length;j++){
        var a=cars[i],b=cars[j];
        if(inPit(a)||inPit(b)||a.dnf||b.dnf) continue;
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
          applyDamage(a,dv*DMG_FRONT_K,'front');
          applyDamage(b,dv*DMG_REAR_K,'rear');
        } else {
          nx=projS>0?sx:-sx;nz=projS>0?sz:-sz;pen=ovS*0.5;
          var bumpY=pen*0.08;
          if(a.yawRate!==undefined) a.yawRate+=(projS>0?bumpY:-bumpY);
          if(b.yawRate!==undefined) b.yawRate+=(projS>0?-bumpY:bumpY);
          var rub=Math.min(pen*0.6,0.45);
          a.speed=Math.max(0,a.speed-rub);b.speed=Math.max(0,b.speed-rub);
          applyDamage(a,rub*DMG_SIDE_K,'side');applyDamage(b,rub*DMG_SIDE_K,'side');
          if(rub>0.2){SPARKS.emit(a,0.5);SPARKS.emit(b,0.5);}
          if(a._ovTimer>0) a._ovTimer=0;
          if(b._ovTimer>0) b._ovTimer=0;
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

var _AC=null,_masterComp=null,_playerEng=null,_aiEngs=[];
var _prevRpm=0,_popCooldown=0,_prevTickT=0,_noiseBuffer=null;
var _audioRpm=0,_prevGear=0,_crackleBurst=0;

function _initAudio(){
  if(_AC) return;
  _AC=new(window.AudioContext||window.webkitAudioContext)();

  _masterComp=_AC.createDynamicsCompressor();
  _masterComp.threshold.value=-18;_masterComp.knee.value=8;
  _masterComp.ratio.value=6;_masterComp.attack.value=0.003;_masterComp.release.value=0.12;
  _masterComp.connect(_AC.destination);

  var eReal=new Float32Array(8),eImag=new Float32Array(8);
  eImag[1]=1.00;eImag[2]=0.74;eImag[3]=0.40;eImag[4]=0.18;
  eImag[5]=0.07;eImag[6]=0.02;eImag[7]=0.0;
  var _exhaustWave=_AC.createPeriodicWave(eReal,eImag);

  _noiseBuffer=_AC.createBuffer(1,_AC.sampleRate,_AC.sampleRate);
  var nd=_noiseBuffer.getChannelData(0);
  for(var i=0;i<nd.length;i++) nd[i]=Math.random()*2-1;

  function _makeDistCurve(k){
    var n=512,c=new Float32Array(n);
    for(var i=0;i<n;i++){var x=(i*2)/n-1;c[i]=Math.tanh(k*x)/Math.tanh(k);}
    return c;
  }

  function buildEng(light){
    var exhaustOsc=_AC.createOscillator();
    exhaustOsc.setPeriodicWave(_exhaustWave);
    exhaustOsc.frequency.value=60;
    // Light engines (the 19 AI cars) get a minimal graph: one oscillator → lowpass → gain.
    // They are quiet and distance-attenuated (vol ≤ 0.10), so the rich layers are inaudible —
    // skipping them cuts ~200 audio nodes at race start and ~5 param updates × 19 cars/frame.
    if(light){
      var lp=_AC.createBiquadFilter();lp.type='lowpass';lp.frequency.value=320;lp.Q.value=0.5;
      var g=_AC.createGain();g.gain.value=0;
      exhaustOsc.connect(lp);lp.connect(g);g.connect(_masterComp);
      exhaustOsc.start();
      return {exhaustOsc:exhaustOsc,exhaustLP:lp,gain:g};
    }
    var exhaustOsc2=_AC.createOscillator();
    exhaustOsc2.setPeriodicWave(_exhaustWave);
    exhaustOsc2.frequency.value=60;
    var dist=_AC.createWaveShaper();
    dist.curve=_makeDistCurve(1.8);dist.oversample='2x';
    var exhaustLP=_AC.createBiquadFilter();
    exhaustLP.type='lowpass';exhaustLP.frequency.value=320;exhaustLP.Q.value=0.5;

    var subOsc=_AC.createOscillator();
    subOsc.type='sine';subOsc.frequency.value=30;
    var subGain=_AC.createGain();subGain.gain.value=0.45;

    var bodyBPF=_AC.createBiquadFilter();
    bodyBPF.type='bandpass';bodyBPF.frequency.value=105;bodyBPF.Q.value=0.9;
    var bodyGain=_AC.createGain();bodyGain.gain.value=0.72;

    var ns=_AC.createBufferSource();ns.buffer=_noiseBuffer;ns.loop=true;ns.start();
    var nBPF=_AC.createBiquadFilter();
    nBPF.type='bandpass';nBPF.frequency.value=200;nBPF.Q.value=0.4;
    var nGain=_AC.createGain();nGain.gain.value=0.035;

    var gain=_AC.createGain();gain.gain.value=0;
    var lfo=_AC.createOscillator();lfo.frequency.value=3.5;
    var lfoDepth=_AC.createGain();lfoDepth.gain.value=0;
    lfo.connect(lfoDepth);lfoDepth.connect(gain.gain);
    lfo.start();
    exhaustOsc.connect(dist);exhaustOsc2.connect(dist);dist.connect(exhaustLP);exhaustLP.connect(gain);
    exhaustOsc.connect(bodyBPF);bodyBPF.connect(bodyGain);bodyGain.connect(gain);
    subOsc.connect(subGain);subGain.connect(gain);
    ns.connect(nBPF);nBPF.connect(nGain);nGain.connect(gain);
    gain.connect(_masterComp);
    exhaustOsc.start();exhaustOsc2.start();subOsc.start();
    return {exhaustOsc:exhaustOsc,exhaustOsc2:exhaustOsc2,subOsc:subOsc,exhaustLP:exhaustLP,gain:gain,lfo:lfo,lfoDepth:lfoDepth};
  }

  _playerEng=buildEng(false);
  _aiEngs=[];
  for(var i=0;i<AI.length;i++) _aiEngs.push(buildEng(true));
}

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

function _fireWheelGun(){
  if(!_AC||!_noiseBuffer) return;
  var now=_AC.currentTime;
  for(var i=0;i<4;i++){
    var t=now+i*0.028;
    var ns=_AC.createBufferSource();ns.buffer=_noiseBuffer;
    var bpf=_AC.createBiquadFilter();bpf.type='bandpass';
    bpf.frequency.value=1800+Math.random()*900;bpf.Q.value=3.0;
    var env=_AC.createGain();
    env.gain.setValueAtTime(0,t);env.gain.setValueAtTime(0.5,t+0.001);
    env.gain.exponentialRampToValueAtTime(0.001,t+0.022);
    ns.connect(bpf);bpf.connect(env);env.connect(_masterComp);
    ns.start(t);ns.stop(t+0.03);
  }
}

function _fireUpshiftCrack(){
  var ns=_AC.createBufferSource();ns.buffer=_noiseBuffer;
  var bpf=_AC.createBiquadFilter();bpf.type='bandpass';
  // Wide, low-Q band + a short fade-in (no instant onset) = a soft natural "blip", not a digital tick.
  bpf.frequency.value=1100+Math.random()*600;bpf.Q.value=1.2;
  var env=_AC.createGain();var t=_AC.currentTime;
  env.gain.setValueAtTime(0,t);
  env.gain.linearRampToValueAtTime(0.18,t+0.006);
  env.gain.exponentialRampToValueAtTime(0.001,t+0.03);
  ns.connect(bpf);bpf.connect(env);env.connect(_masterComp);
  ns.start();ns.stop(t+0.045);
}

function _fireDownshiftCrackle(){
  var now=_AC.currentTime;
  var amps=[0.14,0.11,0.09];var freqs=[1500,1200,1700];
  for(var i=0;i<3;i++){
    var t=now+i*0.075;
    var ns=_AC.createBufferSource();ns.buffer=_noiseBuffer;
    var bpf=_AC.createBiquadFilter();bpf.type='bandpass';
    bpf.frequency.value=freqs[i]+Math.random()*180;bpf.Q.value=1.0;
    var env=_AC.createGain();
    // Fade each pop in over ~5ms so it reads as a soft crackle rather than a stack of clicks.
    env.gain.setValueAtTime(0,t);
    env.gain.linearRampToValueAtTime(amps[i],t+0.005);
    env.gain.exponentialRampToValueAtTime(0.001,t+0.07);
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

  if(_prevGear!==0&&P.gear!==_prevGear&&gameState==='RACING'&&_crackleBurst<=0){
    if(P.gear>_prevGear){
      _audioRpm=Math.max(0.05,_audioRpm-0.14);
      _fireUpshiftCrack();_crackleBurst=0.18;
    } else {
      _audioRpm=Math.min(0.95,_audioRpm+0.10);
      _fireDownshiftCrackle();_crackleBurst=0.30;
    }
  }
  _prevGear=P.gear;

  var rpmT=P.rpmVal;
  var riseR=thr>0?2.2:5.0,fallR=brk>0?6.0:3.5;
  if(rpmT>_audioRpm) _audioRpm=Math.min(rpmT,_audioRpm+riseR*dt);
  else _audioRpm=Math.max(rpmT,_audioRpm-fallR*dt);

  var wobble=(Math.random()-0.5)*(0.08-_audioRpm*0.06);
  var effRpm=Math.max(0,Math.min(1,_audioRpm+wobble));

  var crankHz=38+effRpm*147;
  var lpHz=250+effRpm*850;
  if(brk>0)  lpHz*=0.50;
  else if(!thr) lpHz*=0.72;
  _playerEng.exhaustOsc.frequency.setTargetAtTime(crankHz,now,0.04);
  _playerEng.exhaustOsc2.frequency.setTargetAtTime(crankHz*1.008,now,0.04);
  _playerEng.subOsc.frequency.setTargetAtTime(crankHz*0.5,now,0.04);
  _playerEng.exhaustLP.frequency.setTargetAtTime(lpHz,now,0.04);

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

  if(active&&effRpm>0.95&&thr===1){
    var fl=now%0.067<0.033?0.88:1.0;
    _playerEng.gain.gain.setValueAtTime(targetGain*fl,now);
  }

  _crackleBurst=Math.max(0,_crackleBurst-dt);
  _popCooldown=Math.max(0,_popCooldown-dt);

  if(_prevRpm>0.45&&(_audioRpm-_prevRpm)<-0.04&&_popCooldown<=0&&_crackleBurst<=0&&gameState==='RACING'){
    _fireExhaustCrackle(1+Math.floor(_audioRpm*3));
    _popCooldown=0.10;_crackleBurst=0.35;
  }
  _prevRpm=_audioRpm;

  for(var i=0;i<AI.length;i++){
    var ai=AI[i],e=_aiEngs[i];if(!e) continue;
    var aiNorm=Math.min(1,ai.speed/(MAX_SPD*(ai.paceFac||1.0)));
    var aiCrank=38+aiNorm*147;
    var aiLP=250+aiNorm*850;
    e.exhaustOsc.frequency.setTargetAtTime(aiCrank,now,0.12);
    if(e.exhaustOsc2) e.exhaustOsc2.frequency.setTargetAtTime(aiCrank*1.008,now,0.12);
    if(e.subOsc) e.subOsc.frequency.setTargetAtTime(aiCrank*0.5,now,0.12);
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

function drawWheelDisp(d){
  var x=d.ctx,W=256,H=150;
  x.fillStyle='#0B0A08';x.fillRect(0,0,W,H);
  x.lineWidth=4;x.strokeStyle=P.drs?'#C7A06A':'#2A2E30';x.strokeRect(3,3,W-6,H-6);
  var segs=16,bw=(W-28)/segs,lit=Math.round(P.rpmVal*segs);
  for(var s=0;s<segs;s++){var t=s/(segs-1);
    x.fillStyle=s<lit?(t<0.55?'#9DA3A8':t<0.82?'#C7A06A':'#D6203F'):'#16140F';
    x.fillRect(14+s*bw+1,12,bw-2,10);}
  var cmp=COMPOUNDS[P.compound]||COMPOUNDS[1];
  x.fillStyle='#'+('000000'+cmp.col.toString(16)).slice(-6);
  x.beginPath();x.arc(24,42,11,0,Math.PI*2);x.fill();
  x.fillStyle='#0B0A08';x.font='bold 13px monospace';x.textAlign='center';x.fillText(cmp.id,24,47);
  x.textAlign='right';x.fillStyle='#7E858B';x.font='bold 15px monospace';x.fillText('L'+Math.min(P.lap,LAPS)+'/'+LAPS,W-12,46);
  x.textAlign='center';x.fillStyle=P.drs?'#C7A06A':'#ECE5D5';x.font='900 84px monospace';
  x.fillText(P.speed<0.5?'N':(''+P.gear),W/2,116);
  x.font='bold 18px monospace';x.textAlign='left';x.fillStyle='#C7A06A';x.fillText(((P.speed*3.6)|0)+' KMH',12,H-12);
  if(P.drs){x.textAlign='center';x.fillStyle='#C7A06A';x.font='bold 14px monospace';x.fillText('DRS',W/2,H-12);}
  x.textAlign='right';x.fillStyle='#D6203F';x.font='bold 18px monospace';x.fillText('P'+(getPos()+1),W-12,H-12);
  d.tex.needsUpdate=true;
}
function updateCockpitWheel(dt){
  if(!cockpitWheel) return;
  cwSteer+=((P.steerIn||0)-cwSteer)*Math.min(dt*9,1);
  if(cwSpin) cwSpin.rotation.z=-cwSteer*2.4;
  if(!cockpitMode) return;
  if(cwLeds){
    // Sequential shift-light rev bar driven by a smoothed engine-rev (self-contained so it works
    // with audio off): fills green->red->violet as revs rise, drains as they fall, dips/refills on
    // upshift. At the redline it just sits fully lit and STEADY (no blink). OFF leds keep a dim base
    // glow so the strip stays readable. (No strobe: auto-gears hold max revs on straights.)
    var tgt=P.rpmVal||0;
    if(cwPrevGear&&P.gear>cwPrevGear) cwRev=Math.max(0,cwRev-0.40);  // upshift reset
    cwPrevGear=P.gear;
    var rate=(tgt>cwRev?3.0:4.5)*dt;
    cwRev+=Math.max(-rate,Math.min(rate,tgt-cwRev));
    var lit=Math.floor(cwRev*cwLeds.length+0.0001);
    for(var i=0;i<cwLeds.length;i++)
      cwLeds[i].emissiveIntensity=i<lit?3.0:0.10;}
  if(cwDisp) drawWheelDisp(cwDisp);
}

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
    var fx=Math.sin(P.heading),fz=Math.cos(P.heading);
    var lx=fz,lz=-fx,lean=(P.steerIn||0)*0.05;
    var dip=(P.brk?-0.04:0);
    var ex=P.x+fx*0.42+lx*lean,ey=P.y+RIDE_H+0.78+dip,ez=P.z+fz*0.42+lz*lean;
    gameCam.position.set(ex,ey,ez);
    var pitch=(P.brk?-0.10:0)+(P.thr?0.05:0);
    gameCam.lookAt(new THREE.Vector3(ex+fx*30,ey-0.12+pitch,ez+fz*30));
    gameCam.fov=74;gameCam.updateProjectionMatrix();return;
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
  var shk=P._shake||0;
  if(shk>0.01){
    gameCam.position.x+=(Math.random()-0.5)*shk*0.7;
    gameCam.position.y+=(Math.random()-0.5)*shk*0.45;
    gameCam.position.z+=(Math.random()-0.5)*shk*0.7;
  }
  P._shake=Math.max(0,shk-dt*3.5);
  var pitch=(P.brk?-0.7:0)+(P.thr?0.3:0);
  gameCam.lookAt(new THREE.Vector3(P.x+fx*20,P.y+RIDE_H+0.5+pitch,P.z+fz*20));
  gameCam.fov=(P.drs?84:74)+Math.min(spd/MAX_SPD,1.1)*9+shk*4;gameCam.updateProjectionMatrix();
}

function fmt(t){
  var m=Math.floor(t/60),s=Math.floor(t%60),ms=Math.floor((t%1)*1000);
  return m+':'+(s<10?'0':'')+s+'.'+(ms<100?(ms<10?'00':'0'):'')+ms;
}

var _radioEl=null,_radioT=0;
function radio(msg,big){
  if(!_radioEl){
    _radioEl=document.createElement('div');
    _radioEl.style.cssText='position:fixed;left:50%;top:66px;transform:translateX(-50%);z-index:10002;'+
      'pointer-events:none;font-family:Space Mono,Courier New,monospace;font-size:13px;letter-spacing:.08em;'+
      'color:#ECE5D5;background:rgba(8,7,6,.7);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-left:3px solid #D6203F;padding:7px 16px;border-radius:5px;'+
      'opacity:0;box-shadow:0 8px 26px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.05);max-width:72%;text-align:center;transition:opacity .18s';
    var host=(gameCanvas&&gameCanvas.parentNode)?gameCanvas.parentNode:document.body;
    host.appendChild(_radioEl);
  }
  _radioEl.innerHTML='<span style="color:'+(big?'#C7A06A':'#D6203F')+'">&#9646; RADIO&nbsp;&nbsp;</span>'+msg;
  _radioEl.style.borderLeftColor=big?'#C7A06A':'#D6203F';
  _radioT=big?3.6:3.0;
}
function updateRadio(dt){
  if(!_radioEl) return;
  _radioT=Math.max(0,_radioT-dt);
  _radioEl.style.opacity=_radioT>0?Math.min(1,_radioT*2.2):0;
}

function classifyCmp(a,b){
  if(a.dnf&&!b.dnf) return 1;
  if(b.dnf&&!a.dnf) return -1;
  if(a.dnf&&b.dnf) return b.lap!==a.lap?b.lap-a.lap:b.ti-a.ti;
  if(a.ft<Infinity&&b.ft<Infinity) return a.ft-b.ft;
  if(a.ft<Infinity) return -1;
  if(b.ft<Infinity) return 1;
  return b.lap!==a.lap?b.lap-a.lap:b.ti-a.ti;
}

function getPos(){
  var all=[{n:'P',lap:P.lap,ti:P.tIdx,ft:P.finishTime,dnf:P.dnf}];
  AI.forEach(function(ai,i){all.push({n:'A'+i,lap:ai.lap,ti:ai.tIdx,ft:ai.finishTime,dnf:ai.dnf});});
  all.sort(classifyCmp);
  var pp=0;all.forEach(function(e,i){if(e.n==='P') pp=i;});return pp;
}

var _stLast=0;
function updateStandings(){
  if(!hudStandings) return;
  var now=Date.now();if(now-_stLast<200) return;_stLast=now;
  var arr=[{isP:true,name:AI_NAMES[0],abbr:'YOU',color:'#ECE5D5',lap:P.lap,ti:P.tIdx,sp:P.speed,ft:P.finishTime,pit:inPit(P),dnf:P.dnf}];
  for(var i=0;i<AI.length;i++){
    arr.push({isP:false,name:AI_NAMES[i+1],abbr:AI_TEAMS[i],
      color:'#'+(AI_COLORS[i]||0xAAAAAA).toString(16).padStart(6,'0'),
      lap:AI[i].lap,ti:AI[i].tIdx,sp:AI[i].speed,ft:AI[i].finishTime,pit:inPit(AI[i]),dnf:AI[i].dnf});
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
    var gapCls='st-gap',gapTxt=gap;
    if(e.dnf){gapCls='st-gap st-dnf';gapTxt='DNF';}
    else if(e.pit){gapCls='st-gap st-pit';gapTxt='PIT';}
    html+='<div class="st-row'+(e.isP?' me':'')+'"><span class="st-pos">'+(k+1)+'</span>'+
      '<span class="st-chip" style="background:'+e.color+'"></span>'+
      '<span class="st-name">'+e.abbr+' '+e.name.slice(0,9)+'</span>'+
      '<span class="'+gapCls+'">'+gapTxt+'</span></div>';
  }
  hudStandings.innerHTML=html;
}

var _lastPos=-1,_posFlashUntil=0;
function updateHUD(){
  var _curPos=getPos();
  if(gameState==='RACING'&&_lastPos>=0&&_curPos!==_lastPos){
    if(_curPos<_lastPos){radio('P'+(_curPos+1)+' now — great move, keep pushing!');if(hudPos)hudPos.style.color='#C7A06A';}
    else{radio('Lost a place — we are P'+(_curPos+1)+'. Heads down.');if(hudPos)hudPos.style.color='#D6203F';}
    _posFlashUntil=Date.now()+1000;
  }
  _lastPos=_curPos;
  if(hudPos&&Date.now()>_posFlashUntil) hudPos.style.color='';
  if(hudPos) hudPos.textContent='P'+(_curPos+1);
  if(hudLap) hudLap.textContent='LAP '+P.lap+'/'+LAPS;
  if(hudTimer) hudTimer.textContent=fmt(gameState==='RACING'?raceTime-P.lapStart:0);
  if(hudSpeed) hudSpeed.textContent=((P.speed*3.6)|0)+' km/h';
  if(hudGear) hudGear.textContent=P.speed<0.5?'N':P.gear;
  if(rpmFill) rpmFill.style.width=(P.rpmVal*100)+'%';
  var tw=P.tireWear;
  var tireColor=tw>0.5?'#9DA3A8':tw>0.2?'#C7A06A':'#D6203F';
  if(tireF){tireF.style.width=(tw*100)+'%';tireF.style.background=tireColor;}
  if(tireR){tireR.style.width=(tw*100)+'%';tireR.style.background=tireColor;}
  function tc(t2){return t2>0.7?'#D6203F':t2>0.4?'#C7A06A':'#9DA3A8';}
  if(tempF) tempF.style.background=tc(P.tireTempF);
  if(tempR) tempR.style.background=tc(P.tireTempR);
  var cmp=COMPOUNDS[P.compound];
  if(tireCmp){tireCmp.textContent=cmp.id;tireCmp.style.background='#'+cmp.col.toString(16).padStart(6,'0');}
  if(tirePct){tirePct.textContent=(tw*100|0)+'%';tirePct.style.color=tireColor;}
  var dmg=P.damage,dmgColor=dmg<0.3?'#9DA3A8':dmg<0.6?'#C7A06A':'#D6203F';
  if(dmgBar){dmgBar.style.width=(dmg*100)+'%';dmgBar.style.background=dmgColor;}
  if(dmgPct){dmgPct.textContent=(dmg*100|0)+'%';dmgPct.style.color=dmgColor;}
  if(hudBox){var advise=((tw<0.25||dmg>0.45)&&P.pitState==='NONE'&&P.lap<LAPS);hudBox.className='hud-box'+(advise?' advise':'');}
  if(wheelInd) wheelInd.setAttribute('transform','rotate('+(-P.yawRate*20*180/Math.PI)+')');
  var hudMsgEl=document.querySelector('#hud-main .hud-msg');
  if(hudMsgEl){
    if(P.bestLap<Infinity){hudMsgEl.textContent='BEST '+fmt(P.bestLap);hudMsgEl.style.color='#C7A06A';}
    else{hudMsgEl.textContent='';hudMsgEl.style.color='';}
  }
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
  var pe=document.getElementById('pit-ind');
  if(!pe&&overlay){pe=document.createElement('div');pe.id='pit-ind';pe.style.cssText='position:absolute;top:64px;left:50%;transform:translateX(-50%);font:700 15px Courier New,monospace;letter-spacing:2px;padding:5px 14px;border-radius:4px;background:rgba(0,0,0,0.62);display:none;z-index:30;';overlay.appendChild(pe);}
  if(pe){
    if(P.pitState!=='NONE'){pe.textContent='PIT LIMITER';pe.style.color='#C7A06A';pe.style.display='block';}
    else if(P._pitArmed){pe.textContent='▸ BOX — FIT '+COMPOUNDS[P._pitCompound==null?P.compound:P._pitCompound].name+'  [1]S [2]M [3]H';pe.style.color='#C7A06A';pe.style.display='block';}
    else pe.style.display='none';
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
  minimapCtx.strokeStyle='#C7A06A';minimapCtx.lineWidth=3;
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
  minimapCtx.strokeStyle='#C7A06A';minimapCtx.lineWidth=2;minimapCtx.beginPath();
  for(var pj=0;pj<=PIT_SEGS;pj+=3){var pw2=pitPts[pj];if(pj===0)minimapCtx.moveTo(wx(pw2.x),wz2(pw2.z));else minimapCtx.lineTo(wx(pw2.x),wz2(pw2.z));}
  minimapCtx.stroke();
  minimapCtx.fillStyle='#ECE5D5';minimapCtx.beginPath();
  minimapCtx.arc(wx(P.x),wz2(P.z),5,0,Math.PI*2);minimapCtx.fill();
  AI.forEach(function(ai,i){
    minimapCtx.fillStyle='#'+(AI_COLORS[i]||0xAAAAAA).toString(16).padStart(6,'0');
    minimapCtx.beginPath();
    minimapCtx.arc(wx(ai.x),wz2(ai.z),3,0,Math.PI*2);minimapCtx.fill();
  });
}

function showDnf(reason){
  if(dnfReasonEl) dnfReasonEl.textContent=reason||'mechanical failure';
  if(dnfOvl) dnfOvl.className='active';
}
function dnfToClassification(){
  if(P._dnfDone) return;
  P._dnfDone=true;
  if(dnfOvl) dnfOvl.className='';
  gameState='FINISHED';
  showPodium();
}
function showPodium(){
  var all=[{n:AI_NAMES[0],lap:P.lap,ti:P.tIdx,ft:P.finishTime,dnf:P.dnf,reason:P.dnfReason}];
  AI.forEach(function(ai,i){all.push({n:AI_NAMES[i+1],lap:ai.lap,ti:ai.tIdx,ft:ai.finishTime,dnf:ai.dnf,reason:ai.dnfReason});});
  all.sort(classifyCmp);
  var rows='',medals=['1ST','2ND','3RD','4TH','5TH','6TH','7TH','8TH','9TH','10TH',
    '11TH','12TH','13TH','14TH','15TH','16TH','17TH','18TH','19TH','20TH'];
  all.forEach(function(e,i){
    if(i>=10) return;
    var status=e.dnf
      ?'<span style="color:#D6203F;font-weight:700">DNF</span>'+(e.reason?' <span style="color:#6A7176;font-size:.72em">'+e.reason+'</span>':'')
      :'LAP '+e.lap;
    var pos=e.dnf?'—':medals[i];
    rows+='<tr'+(i===0&&!e.dnf?' class="podium-p1"':'')+'><td>'+pos+'</td><td>'+e.n+'</td><td>'+status+'</td></tr>';
  });
  if(podiumTable) podiumTable.innerHTML=rows;
  if(podiumOvl) podiumOvl.className='active';
}

// Circuit selection — shown before tire select. Tokyo is the only built circuit; San
// Francisco and Monaco are presented as locked placeholders for future maps.
var GAME_MAPS=[
  {name:'TOKYO',sub:'JAPAN \\u00B7 NIGHT STREET CIRCUIT',glyph:'\\u6771\\u4EAC',locked:false,tag:'AVAILABLE'},
  {name:'SAN FRANCISCO',sub:'USA \\u00B7 COMING SOON',glyph:'SF',locked:true,tag:'&#128274; LOCKED'},
  {name:'MONACO',sub:'MONTE CARLO \\u00B7 COMING SOON',glyph:'MC',locked:true,tag:'&#128274; LOCKED'}
];
function showMapSelect(onPick){
  var ov=document.getElementById('map-select'),cards=document.getElementById('ms-cards');
  if(!ov||!cards){onPick();return;}
  cards.innerHTML='';
  function choose(){
    ov.className='';document.removeEventListener('keydown',keyPick,true);
    onPick();
  }
  function keyPick(e){
    if(e.code==='Digit1'||e.code==='Enter'||e.code==='NumpadEnter'){e.preventDefault();e.stopPropagation();choose();}
  }
  GAME_MAPS.forEach(function(mp){
    var card=document.createElement('div');
    card.className='map-card'+(mp.locked?' locked':' sel');
    card.innerHTML=(mp.locked?'':'<span class="mc-key">[1]</span>')+
      '<div class="mc-thumb">'+mp.glyph+'</div>'+
      '<div class="mc-name">'+mp.name+'</div>'+
      '<div class="mc-sub">'+mp.sub+'</div>'+
      '<div class="mc-tag">'+mp.tag+'</div>';
    if(!mp.locked) card.addEventListener('click',choose);
    cards.appendChild(card);
  });
  gameState='SELECT';
  document.addEventListener('keydown',keyPick,true);
  ov.className='active';
}

function showTireSelect(onPick){
  var ov=document.getElementById('tire-select'),cards=document.getElementById('ts-cards');
  var wl=document.getElementById('ts-wlabel'),wn=document.getElementById('ts-wnote');
  if(!ov||!cards){onPick();return;}
  if(wl) wl.textContent=weather.label;
  if(wn) wn.textContent=weather.note;
  cards.innerHTML='';
  function choose(i){
    P.compound=i;P._pitCompound=null;
    if(playerGrp&&playerGrp.userData.setTyre)playerGrp.userData.setTyre(COMPOUNDS[i].col);
    ov.className='';document.removeEventListener('keydown',keyPick,true);
    onPick();
  }
  function keyPick(e){
    var i=e.code==='Digit1'?0:e.code==='Digit2'?1:e.code==='Digit3'?2:-1;
    if(i>=0){e.preventDefault();e.stopPropagation();choose(i);}
  }
  COMPOUNDS.forEach(function(c,i){
    var hex='#'+c.col.toString(16).padStart(6,'0');
    var card=document.createElement('div');card.className='tire-card'+(i===weather.best?' rec':'');
    card.innerHTML='<span class="tc-key">['+(i+1)+']</span>'+
      '<div class="tc-disc" style="border-color:'+hex+';color:'+(i===2?'#222':'#0a0a0a')+';background:'+hex+'">'+c.id+'</div>'+
      '<div class="tc-name" style="color:'+hex+'">'+c.name+'</div>'+
      '<div class="tc-desc">'+c.desc+'</div>'+
      '<div class="tc-rec">RECOMMENDED</div>';
    card.addEventListener('click',function(){choose(i);});
    cards.appendChild(card);
  });
  gameState='SELECT';
  document.addEventListener('keydown',keyPick,true);
  ov.className='active';
}

function openGame(){
  if(!overlay) return;
  ensureWorld();
  overlay.className='active';
  gameCanvas.focus();requestAnimationFrame(resizeCam);
  _initAudio();
  initRace();
  if(!animRunning){animRunning=true;requestAnimationFrame(loop);}
  showMapSelect(function(){showTireSelect(startCountdown);});
}

function closeGame(){
  overlay.className='';
  if(podiumOvl) podiumOvl.className='';
  if(dnfOvl) dnfOvl.className='';
  if(pauseOvl) pauseOvl.className='';
  var tsOv=document.getElementById('tire-select');if(tsOv) tsOv.className='';
  var msOv=document.getElementById('map-select');if(msOv) msOv.className='';
  if(lightsBarEl) lightsBarEl.className='';
  if(gridMsgEl) gridMsgEl.className='';
  _stopAudio();
  _radioT=0;if(_radioEl) _radioEl.style.opacity=0;
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
if(dnfClassify) dnfClassify.addEventListener('click',dnfToClassification);

var prevTs=0;
function loop(ts){
  if(!animRunning) return;
  requestAnimationFrame(loop);
  if(paused) return;
  var dt=Math.min((ts-prevTs)/1000,0.05);prevTs=ts;
  if(dt<=0) return;
  if(gameState==='RACING') raceTime+=dt;
  // Shared light timing: FIA rain-light blink in the wet + a subtle neon hum on brake glow.
  var tNow=ts*0.001;
  // Brake lights are steady when lit (like the real cars); only the centre rain light
  // blinks, FIA-style, in the wet. No global flicker — it read as a strobe.
  var rainBlink=(weather.id==='WET')&&(Math.sin(tNow*22)>0.2);
  updatePlayer(dt);
  AI.forEach(function(ai){updateAI(ai,dt);});
  resolveCollisions();
  updatePits(dt);
  updateSafetyCar(dt);
  if(playerGrp){
    playerGrp.visible=true;
    if(playerGrp.userData.helmet) playerGrp.userData.helmet.visible=!cockpitMode;
    if(playerGrp.userData.halo) playerGrp.userData.halo.visible=!cockpitMode;
    if(cockpitWheel) cockpitWheel.visible=cockpitMode;
    playerGrp.position.set(P.x,P.y+RIDE_H+(P.pitLift||0),P.z);playerGrp.rotation.y=P.heading-Math.PI/2;
    wheelSync(playerGrp,P,dt);
    applyDamageVisual(playerGrp,P);
    if((P.damage>0.55||P.dnf)&&Math.random()<0.6) SMOKE.emit(P,-1,1,Math.min(1.4,P.damage+0.4));
    if(playerGrp.userData.tailMat){var _tm=playerGrp.userData.tailMat,_tt=P.brk?3.4:(rainBlink?3.0:0.7);_tm.emissiveIntensity+=(_tt-_tm.emissiveIntensity)*0.45;}
    if(playerGrp.userData.tailMat2){var _tm2=playerGrp.userData.tailMat2,_tt2=P.brk?3.1:0.7;_tm2.emissiveIntensity+=(_tt2-_tm2.emissiveIntensity)*0.45;}
    if(playerGrp.userData.drsMat) playerGrp.userData.drsMat.emissiveIntensity=P.drs?2.2:0.0;
    if(playerGrp.userData.brakeGlow){var pbg=(P.brk&&P.speed>6)?Math.min(1.0,0.55+0.45*P.speed/MAX_SPD):0;
      playerGrp.userData.brakeGlow.forEach(function(o){var tg=pbg*o.k;if(!pbg&&o.rain&&rainBlink)tg=0.5*o.k;o.m.opacity+=(tg-o.m.opacity)*0.4;});}
  }
  AI.forEach(function(ai,i){
    if(ai._recovered) return;
    if(aiGrps[i]){aiGrps[i].position.set(ai.x,ai.y+RIDE_H+(ai.pitLift||0),ai.z);aiGrps[i].rotation.y=ai.heading-Math.PI/2;
      wheelSync(aiGrps[i],ai,dt);
      applyDamageVisual(aiGrps[i],ai);
      var _dc=aiGrps[i].userData.decals;if(_dc){var _ddx=ai.x-gameCam.position.x,_ddz=ai.z-gameCam.position.z;_dc.visible=(_ddx*_ddx+_ddz*_ddz)<3600;}
      if(ai.damage>0.55&&!ai.dnf&&Math.random()<0.4) SMOKE.emit(ai,-1,1,ai.damage);
      if(aiGrps[i].userData.tailMat){var _am=aiGrps[i].userData.tailMat,_at=ai._braking?3.4:(rainBlink?3.0:0.7);_am.emissiveIntensity+=(_at-_am.emissiveIntensity)*0.45;}
      if(aiGrps[i].userData.tailMat2){var _am2=aiGrps[i].userData.tailMat2,_at2=ai._braking?3.1:0.7;_am2.emissiveIntensity+=(_at2-_am2.emissiveIntensity)*0.45;}
      if(aiGrps[i].userData.brakeGlow){var abg=(ai._braking&&ai.speed>6)?0.95:0;
        aiGrps[i].userData.brakeGlow.forEach(function(o){var tg=abg*o.k;if(!abg&&o.rain&&rainBlink)tg=0.5*o.k;o.m.opacity+=(tg-o.m.opacity)*0.35;});}}
    if(aiLabels[i]){aiLabels[i].position.set(ai.x,ai.y+RIDE_H+3.2,ai.z);}
  });
  SMOKE.update(dt);
  SPARKS.update(dt);
  FLASH.update(dt);
  FIREWORKS.update(dt);
  WEATHER.update(dt,gameCam.position);
  var _t=(typeof performance!=='undefined'&&performance.now)?performance.now()*0.001:Date.now()*0.001;
  for(var _ci=0;_ci<crowdMats.length;_ci++) crowdMats[_ci].emissiveIntensity=0.03+0.03*Math.sin(_t*2.2+_ci*0.8);
  crowdBoost=Math.max(0,crowdBoost-dt*0.4);
  var _cAmp=0.5+crowdBoost*1.4,_cFid=0.05+crowdBoost*0.25;
  for(var _cf=0;_cf<crowdFolk.length;_cf++){var _f=crowdFolk[_cf];
    var _w=Math.sin(_t*2.3-_f.ph);
    _f.m.position.y=_f.by+(_w>0?_w*_w*_cAmp:0)+Math.sin(_t*3.1+_f.ph*2.0)*_cFid;}
  for(var _cl=0;_cl<clouds.length;_cl++){var _c=clouds[_cl];_c.sp.position.x+=_c.vx*dt;if(_c.sp.position.x>1500) _c.sp.position.x=-1500;}
  if(fanBody) _updateFolk(fanWalk,fanBody,fanHead,_t,dt);
  if(staffBody) _updateFolk(staffWalk,staffBody,staffHead,_t,dt);
  for(var _ff=0;_ff<fanFlags.length;_ff++){var _g=fanFlags[_ff];
    _g.flag.rotation.z=Math.sin(_t*3.2+_g.ph)*0.3;
    _g.flag.rotation.y=_g.ry+Math.sin(_t*2.1+_g.ph)*0.25;}
  flagTimer=Math.max(0,flagTimer-dt);
  if(flagTimer<=0&&(flagState==='green'||flagState==='yellow')) flagState='none';
  var _fcol=FLAG_COLORS[flagState]||0x222222,_fshow=(flagState!=='none');
  for(var _mi=0;_mi<marshals.length;_mi++){var _m=marshals[_mi];
    _m.flag.visible=_fshow;
    if(!_fshow) continue;
    if(flagState==='chequered'){if(_m.mat.map!==chequerTexture){_m.mat.map=chequerTexture;_m.mat.color.set(0xffffff);_m.mat.emissive.set(0x000000);_m.mat.needsUpdate=true;}}
    else if(_m.mat.map!==null||_m.mat.color.getHex()!==_fcol){_m.mat.map=null;_m.mat.color.set(_fcol);_m.mat.emissive.set(_fcol);_m.mat.emissiveIntensity=0.35;_m.mat.needsUpdate=true;}
    _m.flag.rotation.z=Math.sin(_t*7+_m.ph)*0.55;
    _m.flag.rotation.y=Math.sin(_t*5+_m.ph)*0.35;
  }
  for(var _si=0;_si<skyObjs.length;_si++){var _s=skyObjs[_si];
    if(_s.kind==='heli'){
      _s.ang+=_s.speed*dt;
      _s.grp.position.set(_s.cx+Math.cos(_s.ang)*_s.radius,_s.height,_s.cz+Math.sin(_s.ang)*_s.radius);
      _s.grp.rotation.y=-_s.ang+Math.PI;_s.grp.rotation.z=0.18;
      if(_s.rotor) _s.rotor.rotation.y+=dt*30;
      if(_s.trotor) _s.trotor.rotation.x+=dt*40;
      if(_s.nav) _s.nav.visible=(Math.sin(_t*4)>0);
    } else if(_s.kind==='blimp'){
      _s.grp.position.x+=_s.vx*dt;if(_s.grp.position.x>_s.xmax) _s.grp.position.x=_s.xmin;
      _s.grp.rotation.y=Math.sin(_t*0.05)*0.1;
    } else if(_s.kind==='bird'){
      _s.ang+=_s.speed*dt;
      _s.grp.position.set(_s.cx+Math.cos(_s.ang)*_s.radius,_s.height+Math.sin(_t*0.6+_s.ph)*6,_s.cz+Math.sin(_s.ang)*_s.radius);
      _s.grp.rotation.y=-_s.ang;
      var _flap=Math.sin(_t*8+_s.ph)*0.5;
      if(_s.wl)_s.wl.rotation.z=0.3+_flap;if(_s.wr)_s.wr.rotation.z=-0.3-_flap;
    } else if(_s.kind==='drone'){
      _s.ang+=_s.speed*dt;
      _s.grp.position.set(_s.cx+Math.cos(_s.ang)*_s.radius,_s.height+Math.sin(_t*1.3+_s.ph)*2.2,_s.cz+Math.sin(_s.ang)*_s.radius);
      _s.grp.rotation.y=-_s.ang+(_s.speed<0?0:Math.PI);
      _s.grp.rotation.z=Math.sin(_t*0.9+_s.ph)*0.08;
      if(_s.nav) _s.nav.material.opacity=(Math.sin(_t*6+_s.ph)>0)?1:0.08;
    }
  }
  updateCamera(dt);
  updateHUD();
  updateRadio(dt);
  updateCockpitWheel(dt);
  _tickAudio();
  if(afterPass){
    var spN=(gameState==='RACING')?P.speed/MAX_SPD:0;
    afterPass.uniforms['damp'].value=Math.min(0.62,Math.max(0,(spN-0.45)/0.55)*0.62);
  }
  if(useComposer) composer.render();else renderer.render(scene,gameCam);
}

// Headless QA hook: the game lives in this strict-mode IIFE, so automated tests (Playwright)
// can't reach its state directly. Read-only snapshot accessor — not used by the game itself.
window._qaGame=function(){
  return {state:gameState,raceTime:raceTime,
    player:{damage:P.damage,dnf:P.dnf,lap:P.lap,x:P.x,z:P.z},
    ai:AI.map(function(a){return {damage:a.damage,dnf:a.dnf,reason:a.dnfReason,lap:a.lap,speed:a.speed};})};
};

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
(function(){
  if(!('IntersectionObserver' in window)) return;
  var els=[].slice.call(document.querySelectorAll('.charts .chart-section'));
  els.forEach(function(el){el.classList.add('reveal');});
  var io=new IntersectionObserver(function(es){
    es.forEach(function(en){
      if(!en.isIntersecting) return;
      io.unobserve(en.target);
      en.target.classList.add('in');
    });
  },{threshold:0.08,rootMargin:'0px 0px -40px 0px'});
  els.forEach(function(el){io.observe(el);});
})();
(function(){
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion:reduce)').matches;
  var nums = [].slice.call(document.querySelectorAll('.g-val[data-count],.kpi-val[data-count]'));
  var arcs = [].slice.call(document.querySelectorAll('.g-arc'));

  function fillArc(a){
    if(reduce) return;
    a.style.transition='none';
    a.style.strokeDashoffset=a.getAttribute('data-c');
    a.getBoundingClientRect();
    a.style.transition='';
    requestAnimationFrame(function(){ a.style.strokeDashoffset=a.getAttribute('data-off'); });
  }
  function countUp(el){
    var to=parseFloat(el.getAttribute('data-count'))||0;
    var dec=parseInt(el.getAttribute('data-dec')||'0',10);
    var pre=el.getAttribute('data-pre')||'';
    var small=el.querySelector('small');
    var suf=small?small.outerHTML:'';
    if(reduce){ el.innerHTML=pre+to.toFixed(dec)+suf; return; }
    var dur=1100,t0=null;
    function step(t){
      if(t0===null)t0=t;
      var k=Math.min(1,(t-t0)/dur), e=1-Math.pow(1-k,3);
      el.innerHTML=pre+(to*e).toFixed(dec)+suf;
      if(k<1) requestAnimationFrame(step); else el.innerHTML=pre+to.toFixed(dec)+suf;
    }
    requestAnimationFrame(step);
  }

  if(!('IntersectionObserver' in window)){ arcs.forEach(fillArc); nums.forEach(countUp); return; }
  var io=new IntersectionObserver(function(es){
    es.forEach(function(en){
      if(!en.isIntersecting) return;
      io.unobserve(en.target);
      if(en.target.classList.contains('g-arc')) fillArc(en.target); else countUp(en.target);
    });
  },{threshold:0.4});
  arcs.forEach(function(a){io.observe(a);});
  nums.forEach(function(n){io.observe(n);});
})();
</script>

</body>
</html>
"""


def _axis_2d(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=_TICK, family=_MONO, size=9)),
        gridcolor=_GRID_SOFT,
        griddash="dot",
        zerolinecolor=_ZERO_LINE,
        tickfont=dict(color=_TICK, family=_MONO, size=9),
        ticks="outside",
        ticklen=4,
        tickcolor=_GRID,
        showline=True,
        linecolor=_GRID,
        linewidth=1,
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
            font=dict(color=_TICK, family=_MONO, size=10),
            x=0,
            xanchor="left",
            pad=dict(t=8, l=0),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=_axis_2d(xaxis_title),
        yaxis=_axis_2d(yaxis_title),
        font=dict(family=_FONT, color=_FONT_COLOR),
        margin=dict(l=52, r=24, t=56, b=48),
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            x=1, xanchor="right",
            y=1.0, yanchor="bottom",
            font=dict(family=_MONO, color=_NEUTRAL, size=10),
            bgcolor="rgba(8,7,6,0)",
            borderwidth=0,
        ),
        hovermode=hovermode,
        hoverdistance=80,
        spikedistance=400,
        hoverlabel=dict(
            bgcolor="rgba(11,10,8,.96)",
            bordercolor=_GRID,
            font=dict(family=_MONO, size=11, color=_FONT_COLOR),
            namelength=-1,
        ),
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _glow_line(x, y, color, *, name=None, width=2.0, shape="spline",
               marker=None, hovertemplate=None) -> list[go.Scatter]:
    """A single crisp data line — no neon halo. Kept as a list so callers can
    keep using add_traces; the minimalist look relies on restraint, not glow."""
    core = go.Scatter(
        x=x, y=y,
        mode="lines+markers" if marker else "lines",
        name=name, line=dict(color=color, width=width, shape=shape),
        marker=marker,
        hovertemplate=hovertemplate,
        showlegend=name is not None,
    )
    return [core]



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
        f"   AND r.grid > 0 AND r.position_order < {DNF_POSITION_ORDER}"
    )
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)



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
            fillgradient=dict(
                type="vertical",
                colorscale=[[0.0, _hex_to_rgba(color, 0.0)],
                            [1.0, _hex_to_rgba(color, 0.30)]],
            ),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_traces(_glow_line(
            g["round"], g["points"], color,
            name=surname,
            marker=dict(size=6, color=color, line=dict(color="#0B0A08", width=1)),
            hovertemplate=f"<b>{surname}</b>  %{{y}} pts<extra></extra>",
        ))

        wins = g[g["position"] == 1]
        if not wins.empty:
            fig.add_trace(go.Scatter(
                x=wins["round"], y=wins["points"],
                mode="markers",
                name=f"{surname} win glow",
                marker=dict(symbol="circle", size=22,
                            color="rgba(255,255,255,0.18)"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=wins["round"], y=wins["points"],
                mode="markers",
                name=f"{surname} win",
                marker=dict(
                    symbol="star-diamond", size=12,
                    color="#ECE5D5",
                    line=dict(color="#0B0A08", width=1),
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
    df = traj_df[(traj_df["year"] == latest) & (traj_df["position"] < DNF_POSITION_ORDER)].sort_values("round")
    layout = _layout_2d(
        f"FINISH POSITIONS · {latest}",
        xaxis_title="ROUND",
        yaxis_title="FINISH",
        height=420,
    )
    layout.yaxis.update(autorange="reversed", dtick=5)
    fig = go.Figure(layout=layout)

    fig.add_hrect(y0=0.5, y1=1.5, fillcolor="rgba(179,18,43,0.10)", layer="below", line_width=0)
    fig.add_hrect(y0=0.5, y1=3.5, fillcolor="rgba(255,255,255,0.04)", layer="below", line_width=0)

    for i, (driver, g) in enumerate(df.groupby("driver")):
        color = _driver_color(driver, i)
        fig.add_traces(_glow_line(
            g["round"], g["position"], color,
            name=driver.split()[-1], shape="linear",
            marker=dict(size=7, color=color, line=dict(color="#0B0A08", width=1.5)),
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
    total_pts = df.groupby("driver")["points"].max()
    sorted_drivers = total_pts.sort_values(ascending=False).index.tolist()
    d_a = next((d for d in sorted_drivers if d in pivot.columns), None)
    remaining = [d for d in sorted_drivers if d in pivot.columns and d != d_a]
    d_b = remaining[0] if remaining else None
    if d_a is None or d_b is None:
        return fig

    rounds = pivot.index.tolist()
    gap = (pivot[d_a] - pivot[d_b]).tolist()
    color_a = _POSITIVE
    color_b = _ACCENT
    surname_a = d_a.split()[-1]
    surname_b = d_b.split()[-1]

    fig.add_hline(y=0, line=dict(color=_ZERO_LINE, width=1.5, dash="dot"))

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

    leader_color = color_a if gap[0] >= 0 else color_b
    seg_x: list = [rounds[0]]
    seg_y: list = [gap[0]]
    for i in range(1, len(gap)):
        cur_color = color_a if gap[i] >= 0 else color_b
        if cur_color != leader_color:
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
    fig.add_trace(go.Scatter(
        x=seg_x, y=seg_y,
        mode="lines",
        line=dict(color=leader_color, width=2.5, shape="hv"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=rounds, y=gap,
        mode="markers",
        name=f"{surname_a} vs {surname_b}",
        marker=dict(
            size=7,
            color=[color_a if g >= 0 else color_b for g in gap],
            line=dict(color="#0B0A08", width=1),
        ),
        customdata=[[surname_a if (g or 0) >= 0 else surname_b, abs(int(g)) if pd.notna(g) else 0] for g in gap],
        hovertemplate=(
            "<b>R%{x}</b>  %{customdata[0]} leads by %{customdata[1]} pts<extra></extra>"
        ),
    ))

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

    df = traj_df[traj_df["position"] < DNF_POSITION_ORDER].copy()
    df["pos_clip"] = df["position"].clip(upper=20)
    df["label"] = "P" + df["position"].astype(int).astype(str)
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
        [0.00, "#C7A06A"],
        [0.22, "#9DA3A8"],
        [0.55, "#B3122B"],
        [0.80, "#45262B"],
        [1.00, "#1A1414"],
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
        delta = g["grid"] - g["finish"]
        point_colors = [
            _hex_to_rgba(_POSITIVE if d > 0 else _ACCENT if d < 0 else _NEUTRAL,
                         0.4 + 0.55 * year_norm[y])
            for d, y in zip(delta, g["year"])
        ]
        fig.add_trace(go.Scatter(
            x=g["grid"], y=g["finish"],
            mode="markers",
            name=driver.split()[-1],
            customdata=list(zip(g["year"], delta)),
            marker=dict(
                size=7, color=point_colors,
                line=dict(color="#0B0A08", width=0.5),
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
                color=_NEUTRAL,
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
    layout.update(showlegend=False)
    fig = go.Figure(layout=layout)
    if df.empty:
        return _no_data_fig("PIT STOP EFFICIENCY · Z-SCORE",
                            "pit stop data not loaded")

    sem = (df["std_z"] / np.sqrt(df["n_stops"])).fillna(0)

    z_min, z_max = df["mean_z"].min(), df["mean_z"].max()
    z_range = max(z_max - z_min, 1e-9)
    norm = ((df["mean_z"] - z_min) / z_range).tolist()
    _gold, _crim = (199, 160, 106), (179, 18, 43)
    bar_colors = [
        f"rgba({int(_gold[0] + (_crim[0] - _gold[0]) * v)},"
        f"{int(_gold[1] + (_crim[1] - _gold[1]) * v)},"
        f"{int(_gold[2] + (_crim[2] - _gold[2]) * v)},0.92)"
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
            color="#ECE5D5", thickness=1.8, width=5,
        ),
        customdata=list(zip(df["n_stops"], df["std_z"].fillna(0))),
        hovertemplate=(
            "<b>%{y}</b>  z = %{x:.3f}σ<br>"
            "SEM ±%{error_x.array:.3f}σ<br>"
            "n = %{customdata[0]} stops<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line=dict(color=_ACCENT_DIM, width=1.5, dash="dot"))

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
    layout.update(showlegend=False)
    fig = go.Figure(layout=layout)
    if df.empty:
        return _no_data_fig("RELIABILITY MODEL · DNF RATE",
                            "no race results loaded")

    surnames = df["driver"].apply(lambda d: d.split()[-1])
    err_upper = (df["ci_upper"] - df["rate"]).clip(lower=0)
    err_lower = (df["rate"] - df["ci_lower"]).clip(lower=0)

    max_rate = max(df["rate"].max(), 1e-9)
    dot_colors = [
        (lambda t: f"rgba({int(179 - 69 * t)},{int(18 + 86 * t)},{int(43 + 49 * t)},0.95)")(r / max_rate)
        for r in df["rate"]
    ]

    fig.add_trace(go.Scatter(
        x=df["rate"],
        y=surnames,
        mode="markers",
        marker=dict(
            size=11,
            color=dot_colors,
            line=dict(color="#0B0A08", width=1.5),
        ),
        error_x=dict(
            type="data", symmetric=False,
            array=err_upper.tolist(),
            arrayminus=err_lower.tolist(),
            visible=True,
            color="rgba(236,229,213,0.55)",
            thickness=2.5, width=5,
        ),
        customdata=list(zip(df["ci_lower"], df["ci_upper"], df["races"], df["dnfs"])),
        hovertemplate=(
            "<b>%{y}</b>  %{x:.1%} DNF rate<br>"
            "95% CI [%{customdata[0]:.1%}, %{customdata[1]:.1%}]<br>"
            "%{customdata[3]:.0f} DNFs / %{customdata[2]:.0f} races<extra></extra>"
        ),
    ))

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
        return _no_data_fig("SECTOR DELTA · Δ FROM BEST (s)",
                            "no green-flag lap telemetry")

    surnames = df["driver"].apply(lambda d: d.split()[-1]).tolist()
    sector_cfg = [
        ("s1_mean", "S1", _ACCENT),
        ("s2_mean", "S2", _FONT_COLOR),
        ("s3_mean", "S3", _NEUTRAL),
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
        best_idx = int(df[col].idxmin())
        best_surname = df.loc[best_idx, "driver"].split()[-1]
        fig.add_annotation(
            x=0, y=best_surname,
            text=f"★ {label} best",
            font=dict(color=_POSITIVE, family=_FONT, size=7),
            showarrow=False, xanchor="right", xshift=-4, yanchor="middle",
        )
    return fig



_COMPOUND_COLORS = {
    "SOFT":         "#B3122B",
    "MEDIUM":       "#C7A06A",
    "HARD":         "#ECE5D5",
    "INTERMEDIATE": "#9DA3A8",
    "WET":          "#6A7176",
}


def _no_data_fig(title: str, note: str = "Run pipeline with --telemetry to load lap data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"NO SIGNAL — {note.upper()}", xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(color=_TICK, family=_MONO, size=10),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(family=_MONO, size=10, color=_TICK), x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(l=12, r=12, t=36, b=12),
        font=dict(family=_FONT, color=_FONT_COLOR),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
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



_GAUGE_R = 50
_GAUGE_C = 2 * np.pi * _GAUGE_R


def _gauge_svg(pct: float, label: str) -> str:
    """A circular progress ring (track + value arc + dotted tick ring) with a centred
    percentage readout. Renders filled by default so it reads correctly without JS;
    the live script resets it to empty and animates up. pct is 0–100."""
    pct = max(0.0, min(100.0, float(pct)))
    target_off = _GAUGE_C * (1 - pct / 100.0)
    return (
        '<div class="gauge">'
        '<div class="g-ring">'
        '<svg width="124" height="124" viewBox="0 0 124 124">'
        '<circle class="g-tick" cx="62" cy="62" r="57" stroke-dasharray="1.3 7.4"/>'
        '<circle class="g-seg2" cx="62" cy="62" r="61" stroke-dasharray="2 8 2 30 110 24 2 8 2 60 50 85"/>'
        '<circle class="g-seg" cx="62" cy="62" r="59" stroke-dasharray="64 10 4 10 24 18 4 18 88 10 4 116"/>'
        '<circle class="g-coil" cx="62" cy="62" r="42" stroke-dasharray="12 4.5"/>'
        f'<circle class="g-track" cx="62" cy="62" r="{_GAUGE_R}" stroke-width="7"/>'
        f'<circle class="g-arc" cx="62" cy="62" r="{_GAUGE_R}" stroke-width="7"'
        f' stroke-dasharray="{_GAUGE_C:.2f}" stroke-dashoffset="{target_off:.2f}"'
        f' data-off="{target_off:.2f}" data-c="{_GAUGE_C:.2f}"/>'
        f'<circle class="g-flow" cx="62" cy="62" r="{_GAUGE_R}"/>'
        '</svg>'
        '<div class="g-core"></div>'
        f'<div class="g-val" data-count="{pct:.0f}" data-dec="0">{pct:.0f}<small>%</small></div>'
        '</div>'
        f'<div class="g-lbl">{label}</div>'
        '</div>'
    )


def _kpi_cell(target: float, dec: int, unit: str, lbl: str, prefix: str = "") -> str:
    """A dense mission-control data cell. Renders the final value (no-JS correct);
    the live script resets to 0 and counts up when scrolled into view."""
    shown = f"{prefix}{target:.{dec}f}"
    return (
        '<div class="kpi"><div class="kpi-top">'
        f'<div class="kpi-val" data-count="{target}" data-dec="{dec}" data-pre="{prefix}">{shown}</div>'
        f'<div class="kpi-unit">{unit}</div></div>'
        f'<div class="kpi-lbl">{lbl}</div></div>'
    )


def _build_ticker(traj: pd.DataFrame, gf: pd.DataFrame) -> str:
    """Scrolling per-driver telemetry line, drivers ordered by total points. The sequence
    is emitted twice so the CSS translateX(-50%) loop is seamless."""
    if traj.empty:
        return ""
    pts = traj.groupby(["driver", "year"])["points"].max().groupby("driver").sum()
    items = []
    for drv in pts.sort_values(ascending=False).index:
        g = gf[gf["driver"] == drv] if not gf.empty else gf
        w  = int((g["finish"] == 1).sum()) if not g.empty else 0
        p  = int((g["finish"] <= 3).sum()) if not g.empty else 0
        pl = int((g["grid"] == 1).sum())   if not g.empty else 0
        av = f"P{g['finish'].mean():.1f}"  if not g.empty else "—"
        name = drv.strip().upper() or "—"
        items.append(
            f'<span class="ticker-item"><b>{name}</b>'
            f'<span class="k">WINS</span>&nbsp;{w}'
            f'<span class="k">PODIUMS</span>&nbsp;{p}'
            f'<span class="k">POLES</span>&nbsp;{pl}'
            f'<span class="k">AVG</span>&nbsp;{av}'
            f'<span class="k">PTS</span>&nbsp;{int(pts.get(drv, 0))}</span>'
        )
    seq = "".join(items)
    return seq + seq


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

    def _df_or_empty(fn):
        try:
            return fn(engine, team_refs)
        except Exception:
            return pd.DataFrame()

    pit_df   = _df_or_empty(pit_stop_efficiency)
    dnf_df   = _df_or_empty(dnf_rate_model)
    sec_df   = _df_or_empty(sector_deltas)
    deg_df   = _df_or_empty(tyre_degradation)
    strat_df = _df_or_empty(pit_strategy)

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

    starts       = int(len(traj))
    total_races  = int(traj[["year", "round"]].drop_duplicates().shape[0]) if not traj.empty else 0
    driver_count = str(traj["driver"].nunique()) if not traj.empty else "—"
    seasons      = len(years)

    wins        = int((gf["finish"] == 1).sum()) if not gf.empty else 0
    podiums     = int((gf["finish"] <= 3).sum()) if not gf.empty else 0
    poles       = int((gf["grid"] == 1).sum())   if not gf.empty else 0
    front_row   = int((gf["grid"] <= 2).sum())   if not gf.empty else 0
    avg_finish  = float(gf["finish"].mean()) if not gf.empty else 0.0
    best_finish = int(gf["finish"].min())    if not gf.empty else 0
    total_points = float(traj.groupby(["driver", "year"])["points"].max().sum()) if not traj.empty else 0.0

    win_rate    = 100.0 * wins / starts    if starts else 0.0
    podium_rate = 100.0 * podiums / starts if starts else 0.0
    if not dnf_df.empty and dnf_df["races"].sum() > 0:
        reliability = 100.0 * (1 - dnf_df["dnfs"].sum() / dnf_df["races"].sum())
    elif starts:
        reliability = 100.0 * (1 - int((traj["position"] >= DNF_POSITION_ORDER).sum()) / starts)
    else:
        reliability = 0.0

    gauges_html = (
        _gauge_svg(win_rate, "Win Rate")
        + _gauge_svg(podium_rate, "Podium Rate")
        + _gauge_svg(reliability, "Reliability")
    )
    kpis_html = (
        _kpi_cell(total_points, 0, "PTS", "Championship Points")
        + _kpi_cell(poles, 0, "POL", "Pole Positions")
        + _kpi_cell(front_row, 0, "FR", "Front-Row Starts")
        + _kpi_cell(avg_finish, 1, "AVG", "Avg Finish Position")
        + _kpi_cell(best_finish, 0, "POS", "Best Finish", prefix="P")
        + _kpi_cell(total_races, 0, "RND", "Races Analyzed")
        + _kpi_cell(4, 0, "WCC", "Constructors Titles")
        + _kpi_cell(seasons, 0, "SSN", "Seasons Covered")
    )
    ticker_html = _build_ticker(traj, gf)
    cluster_tag = f"SYS-CHECK \xb7 {starts} ENTRIES \xb7 OK"
    build_id = f"RBR-{datetime.now(timezone.utc):%Y%m%d}"
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
        .replace("PLACEHOLDER_GAUGES",       gauges_html)
        .replace("PLACEHOLDER_KPIS",         kpis_html)
        .replace("PLACEHOLDER_TICKER",       ticker_html)
        .replace("PLACEHOLDER_CLUSTER_TAG",  cluster_tag)
        .replace("PLACEHOLDER_SEASONS",      str(seasons))
        .replace("PLACEHOLDER_YEAR_RANGE",   year_range)
        .replace("PLACEHOLDER_ROUNDS",       str(total_races))
        .replace("PLACEHOLDER_BUILD",        build_id)
        .replace("PLACEHOLDER_C10",          div10)
        .replace("PLACEHOLDER_C1",           div1)
        .replace("PLACEHOLDER_C2",           div2)
        .replace("PLACEHOLDER_C3",           div3)
        .replace("PLACEHOLDER_C4",           div4)
        .replace("PLACEHOLDER_C5",           div5)
        .replace("PLACEHOLDER_C6",           div6)
        .replace("PLACEHOLDER_C7",           div7)
        .replace("PLACEHOLDER_C8",           div8)
        .replace("PLACEHOLDER_C9",           div9)
        .replace("PLACEHOLDER_DRIVER_COUNT", driver_count)
        .replace("PLACEHOLDER_TS",           ts)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
