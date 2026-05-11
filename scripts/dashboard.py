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

from analytics import _ref_params

logger = logging.getLogger("f1_analytics")

_LOGO_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "redbullracinglogo.jpg")
)

_BG   = "#000000"
_FONT = "Courier New, monospace"
_GRID = "#1e1e1e"
_TICK = "#888888"
_ACCENT = "#FFFFFF"

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
<title>PLACEHOLDER_TITLE · F1 Performance Analytics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;color:#fff;font-family:'Courier New',monospace;min-height:100vh}
header{padding:36px 40px 28px;border-bottom:2px solid #CC0000;background:linear-gradient(180deg,#050505 0%,#000 100%)}
.hd-team{font-size:.55rem;letter-spacing:.30em;color:#1E41FF;text-transform:uppercase;margin-bottom:6px}
h1{font-size:1.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.18em;line-height:1.1}
h1 span.accent{color:#1E41FF}
.sub{color:#555;font-size:.65rem;letter-spacing:.18em;margin-top:10px;text-transform:uppercase}
.stats-row{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid #111}
.stat-card{padding:18px 24px;border-right:1px solid #111;transition:background .2s}
.stat-card:last-child{border-right:none}
.stat-card:hover{background:#0a0a0a}
.stat-val{font-size:1.35rem;font-weight:700;color:#fff;letter-spacing:.06em}
.stat-lbl{font-size:.52rem;color:#555;letter-spacing:.20em;margin-top:4px;text-transform:uppercase}
.car-viewer{padding:32px 0 0;display:flex;justify-content:center;border-bottom:1px solid #0f0f0f;background:radial-gradient(ellipse at 50% 60%,#090912 0%,#000 70%);cursor:pointer}
#f1car{display:block;width:100%;height:440px}
.charts{padding:40px;display:grid;grid-template-columns:1fr;gap:40px}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:32px}
.chart-section{border-top:1px solid #1a1a1a;padding-top:20px}
.chart-label{font-size:.60rem;letter-spacing:.25em;color:#1E41FF;text-transform:uppercase;margin-bottom:12px}
footer{padding:20px 40px;border-top:1px solid #111;display:flex;justify-content:space-between;align-items:center}
.ft-left{font-size:.52rem;color:#333;letter-spacing:.15em;text-transform:uppercase}
.ft-right{font-size:.52rem;color:#333;letter-spacing:.10em}
.ft-right span{color:#1E41FF}
@media(max-width:860px){.chart-row{grid-template-columns:1fr}.stats-row{grid-template-columns:1fr 1fr}}
@media(max-width:480px){h1{font-size:1.3rem}.stats-row{grid-template-columns:1fr}.charts{padding:24px}}
.logo-bar{display:flex;justify-content:center;padding:28px 0 16px}
.logo-img{height:81px;display:block;filter:invert(1) hue-rotate(180deg)}
#game-overlay{display:none;position:fixed;inset:0;z-index:9999;background:#000;flex-direction:column}
#game-overlay.active{display:flex}
#game-canvas{flex:1;width:100%;display:block;outline:none}
#hud{display:flex;flex-direction:column;padding:0;font-family:'Courier New',monospace;font-size:13px;color:#fff;border-top:1px solid #1E41FF;flex-shrink:0;position:relative}
#hud-main{display:flex;align-items:center;gap:10px;padding:6px 14px;background:#000}
#hud-sectors{display:flex;gap:16px;align-items:center;padding:2px 14px 4px;font-size:11px;background:#000;border-top:1px solid #111}
.hud-pos{color:#1E41FF;font-weight:700;font-size:17px;min-width:28px}
.hud-lap{color:#ccc;min-width:72px}
.hud-timer{color:#fff;min-width:80px;font-weight:700}
.hud-speed{color:#888;min-width:70px}
.hud-gear-wrap{display:flex;flex-direction:column;align-items:center;min-width:52px}
.hud-gear{font-size:26px;font-weight:900;color:#fff;line-height:1;transition:color .05s}
.hud-gear.flash{color:#ff4400}
.hud-rpm-bar{width:48px;height:5px;background:#111;border-radius:2px;margin-top:2px}
.hud-rpm-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,#1E41FF 0%,#00ff44 65%,#ff4400 100%);width:0%;transition:width .04s}
.hud-drs{color:#333;border:1px solid #333;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:700;letter-spacing:.12em;transition:color .15s,border-color .15s,text-shadow .15s}
.hud-drs.on{color:#00ff88;border-color:#00ff88;text-shadow:0 0 8px #00ff88}
.hud-tires{display:flex;align-items:center;gap:4px}
.hud-tire-label{color:#555;font-size:10px;min-width:8px}
.hud-tire-wrap{width:52px;height:8px;background:#1a1a1a;border-radius:3px;overflow:hidden;border:1px solid #222}
.hud-tire-bar{height:100%;width:100%;border-radius:3px;transition:width .1s,background .3s}
.hud-tire-temp{width:7px;height:7px;border-radius:50%;background:#555;transition:background .4s}
.hud-s1,.hud-s2,.hud-s3{color:#666;min-width:88px;font-size:11px}
.hud-s1.purple,.hud-s2.purple,.hud-s3.purple{color:#cc00ff;font-weight:700}
.hud-s1.green,.hud-s2.green,.hud-s3.green{color:#00ff44;font-weight:700}
.hud-s1.yellow,.hud-s2.yellow,.hud-s3.yellow{color:#ffdd00}
.hud-delta{color:#888;min-width:80px;font-size:11px}
.hud-delta.green{color:#00ff44}
.hud-delta.red{color:#ff4400}
.hud-msg{flex:1;text-align:center;color:#444;font-size:10px;letter-spacing:.10em}
.hud-close{margin-left:auto;background:none;border:1px solid #CC0000;color:#CC0000;font-family:'Courier New',monospace;cursor:pointer;padding:3px 10px;font-size:12px;letter-spacing:.08em}
.hud-close:hover{background:#CC0000;color:#000}
#hud-minimap{position:absolute;bottom:8px;right:12px;border:1px solid #1E41FF;background:rgba(0,0,0,.75);border-radius:4px;pointer-events:none}
#hud-wheel{position:absolute;bottom:10px;right:182px;pointer-events:none}
#podium-overlay{display:none;position:absolute;inset:0;z-index:10000;background:rgba(0,0,0,.90);flex-direction:column;align-items:center;justify-content:center;font-family:'Courier New',monospace;color:#fff}
#podium-overlay.active{display:flex}
#podium-overlay h2{font-size:1.4rem;letter-spacing:.25em;color:#1E41FF;margin-bottom:24px;text-transform:uppercase}
#podium-table{border-collapse:collapse;font-size:.85rem}
#podium-table td{padding:6px 20px;border-bottom:1px solid #1a1a1a}
#podium-table td:first-child{color:#555;text-align:right}
#podium-table td:nth-child(2){color:#fff;font-weight:700}
#podium-table td:last-child{color:#888;text-align:right}
.podium-p1 td{color:#ffd700 !important}
.podium-btn{margin-top:28px;background:none;border:1px solid #1E41FF;color:#1E41FF;font-family:'Courier New',monospace;cursor:pointer;padding:7px 22px;letter-spacing:.12em;font-size:.8rem}
.podium-btn:hover{background:#1E41FF;color:#000}
#pause-overlay{display:none;position:absolute;inset:0;z-index:10000;background:rgba(0,0,0,.82);flex-direction:column;align-items:center;justify-content:center;font-family:'Courier New',monospace;color:#fff}
#pause-overlay.active{display:flex}
#pause-overlay h2{font-size:1.8rem;letter-spacing:.35em;color:#1E41FF;text-transform:uppercase;margin-bottom:14px}
#pause-overlay p{color:#555;font-size:.72rem;letter-spacing:.22em;text-transform:uppercase}
#lights-bar{display:none;position:absolute;top:14px;left:50%;transform:translateX(-50%);z-index:20001;gap:10px;padding:10px 18px;background:rgba(0,0,0,.9);border:1px solid #330000;border-radius:8px;pointer-events:none}
#lights-bar.active{display:flex}
.light-bulb{width:30px;height:30px;border-radius:50%;background:#1a0000;border:2px solid #440000;transition:background .06s,box-shadow .06s}
.light-bulb.lit{background:#ff2200;border-color:#ff5500;box-shadow:0 0 14px #ff2200,0 0 32px #880000}
#go-flash{display:none;position:absolute;inset:0;background:rgba(255,255,255,.88);z-index:20002;pointer-events:none}
#grid-msg{display:none;position:absolute;bottom:120px;left:50%;transform:translateX(-50%);z-index:20001;color:#fff;font-family:'Courier New',monospace;font-size:.72rem;letter-spacing:.28em;text-transform:uppercase;text-align:center;text-shadow:0 0 8px #1E41FF;pointer-events:none}
#grid-msg.active{display:block}
</style>
</head>
<body>
<div class="logo-bar">PLACEHOLDER_LOGO</div>
<header>
  <div class="hd-team">Oracle Red Bull Racing</div>
  <h1>Red Bull F1 Analytics</h1>
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
    <svg id="hud-wheel" width="52" height="52" viewBox="-26 -26 52 52">
      <circle cx="0" cy="0" r="22" fill="none" stroke="#333" stroke-width="3"/>
      <line x1="-14" y1="0" x2="14" y2="0" stroke="#555" stroke-width="1.5"/>
      <line x1="0" y1="-14" x2="0" y2="14" stroke="#555" stroke-width="1.5"/>
      <g id="hud-wheel-ind">
        <line x1="-18" y1="0" x2="18" y2="0" stroke="#1E41FF" stroke-width="3" stroke-linecap="round"/>
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
  <div class="chart-section">
    <div class="chart-label">Championship · Points Trajectory</div>
    PLACEHOLDER_C1
  </div>
  <div class="chart-row">
    <div class="chart-section">
      <div class="chart-label">Finish Positions · Season</div>
      PLACEHOLDER_C2
    </div>
    <div class="chart-section">
      <div class="chart-label">Driver Delta · Points Gap</div>
      PLACEHOLDER_C3
    </div>
  </div>
  <div class="chart-section">
    <div class="chart-label">Performance Matrix · All Seasons</div>
    PLACEHOLDER_C4
  </div>
  <div class="chart-section">
    <div class="chart-label">Pace · Grid vs Finish</div>
    PLACEHOLDER_C5
  </div>
</div>
<footer>
  <div class="ft-left">Oracle Red Bull Racing · Performance Analytics</div>
  <div class="ft-right">Generated PLACEHOLDER_TS &nbsp;·&nbsp; <span>Oracle Red Bull Racing</span></div>
</footer>
<script>
(function(){
  var c=document.getElementById('f1car');
  if(!c||typeof THREE==='undefined') return;
  var H=440,W=c.parentElement.offsetWidth||900;
  c.width=W; c.height=H;

  // Renderer — PBR pipeline, ACES filmic, PCFSoft shadows
  var renderer=new THREE.WebGLRenderer({canvas:c,antialias:true,alpha:true});
  renderer.setSize(W,H); renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.shadowMap.enabled=true;
  renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.outputEncoding=THREE.sRGBEncoding;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure=1.15;
  renderer.physicallyCorrectLights=true;

  var scene=new THREE.Scene();
  var cam=new THREE.PerspectiveCamera(32,W/H,0.1,100);
  cam.position.set(5.5,2.4,4.8); cam.lookAt(0,0.05,0);

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

  // Lighting
  scene.add(new THREE.HemisphereLight(0xd0e4ff,0x1a1825,0.28));
  var s1=new THREE.DirectionalLight(0xfff8f0,1.70);
  s1.position.set(6,12,5); s1.castShadow=true;
  s1.shadow.mapSize.width=s1.shadow.mapSize.height=2048;
  s1.shadow.camera.near=1; s1.shadow.camera.far=40;
  s1.shadow.camera.left=-5; s1.shadow.camera.right=5;
  s1.shadow.camera.top=4; s1.shadow.camera.bottom=-4;
  s1.shadow.bias=-0.0005; scene.add(s1);
  var s2=new THREE.DirectionalLight(0x8899cc,0.42); s2.position.set(-5,3,-3); scene.add(s2);
  var s3=new THREE.DirectionalLight(0x5577ee,0.32); s3.position.set(-2,2,-8); scene.add(s3);

  // Ground
  var gnd=new THREE.Mesh(new THREE.PlaneGeometry(20,16),
    new THREE.MeshStandardMaterial({color:0x060606,metalness:0.0,roughness:0.52,envMapIntensity:0.5}));
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

  // Expose car builder for the mini-game script
  window._f1Mats={mNav:mNav,mRed:mRed,mGold:mGold,mC:mC,mT:mT,mR:mR,mG:mG,mB:mB};
  window._f1Helpers={mk:mk,bx:bx,cy:cy,bar:bar,el:el,wing:wing,tube3:tube3,addWheel:addWheel,PI:PI};
  // Expose showcase objects so the mini-game can raycast "click the car"
  window._f1Showcase={canvas:c,renderer:renderer,camera:cam,car:car};

  function animate(){requestAnimationFrame(animate);car.rotation.y+=0.004;renderer.render(scene,cam);}
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
var wheelInd=document.getElementById('hud-wheel-ind');
var hudClose=document.querySelector('.hud-close');
var podiumClose=document.getElementById('podium-close');

/* ===================== CONSTANTS ===================== */
var SEGS=600,TW=28,CURB=2.2,BH=3.2;
var MAX_SPD=30,ENG=12000,BRK=18000,DRAG=0.35,CAR_MASS=800;
var RIDE_H=0.59,LAPS=3;
var AI_COLORS=[0xC0C0C0,0xDC0000,0xFF8000];
var AI_NAMES=['VERSTAPPEN','PEREZ','HAMILTON','LECLERC'];

/* ===================== TRACK ===================== */
var trackPts=[
  new THREE.Vector3( 130, 0,   5),
  new THREE.Vector3(  80, 0,   5),
  new THREE.Vector3(  40, 0,   5),
  new THREE.Vector3(  12, 0,  -8),
  new THREE.Vector3(  -2, 0, -35),
  new THREE.Vector3( -10, 0, -58),
  new THREE.Vector3( -45, 0, -68),
  new THREE.Vector3( -85, 0, -72),
  new THREE.Vector3(-115, 0, -65),
  new THREE.Vector3(-138, 0, -40),
  new THREE.Vector3(-145, 0, -10),
  new THREE.Vector3(-138, 0,  18),
  new THREE.Vector3(-120, 1,  38),
  new THREE.Vector3( -95, 2,  58),
  new THREE.Vector3( -65, 3,  72),
  new THREE.Vector3( -30, 3,  80),
  new THREE.Vector3(   5, 3,  80),
  new THREE.Vector3(  35, 3,  72),
  new THREE.Vector3(  60, 2,  60),
  new THREE.Vector3(  72, 2,  42),
  new THREE.Vector3(  65, 1,  22),
  new THREE.Vector3(  88, 0,  12),
  new THREE.Vector3( 108, 0,   8),
  new THREE.Vector3( 122, 0,   5)
];
var trackCurve=new THREE.CatmullRomCurve3(trackPts,true,'catmullrom',0.5);
var wpAll=trackCurve.getSpacedPoints(SEGS);
var waypoints=wpAll.slice(0,SEGS);
var N=waypoints.length;

function wpDir(i){
  var a=waypoints[(i+1)%N],b=waypoints[((i-1)+N)%N];
  var dx=a.x-b.x,dz=a.z-b.z,len=Math.sqrt(dx*dx+dz*dz)||1;
  return {x:dx/len,z:dz/len};
}
function wpPerp(i){var d=wpDir(i);return {x:d.z,z:-d.x};}

function closestWP(x,z,hint){
  var best=hint||0,bd=1e9,start=((best-20)+N*2)%N;
  for(var ii=0;ii<40;ii++){
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
scene.background=new THREE.Color(0x87c8f8);
scene.fog=new THREE.FogExp2(0x87c8f8,0.003);
var gameCam=new THREE.PerspectiveCamera(72,1,0.1,2000);
scene.add(gameCam);

function resizeCam(){
  var W=gameCanvas.clientWidth||900,CH=gameCanvas.clientHeight||600;
  renderer.setSize(W,CH,false);
  gameCam.aspect=W/CH;gameCam.updateProjectionMatrix();
}
window.addEventListener('resize',resizeCam);resizeCam();

/* ===================== LIGHTING ===================== */
scene.add(new THREE.HemisphereLight(0xd8ecff,0x2d4a1e,0.8));
var sun=new THREE.DirectionalLight(0xfff8f0,1.3);
sun.position.set(80,120,60);sun.castShadow=true;
sun.shadow.mapSize.width=sun.shadow.mapSize.height=1024;
sun.shadow.camera.near=1;sun.shadow.camera.far=600;
sun.shadow.camera.left=-250;sun.shadow.camera.right=250;
sun.shadow.camera.top=250;sun.shadow.camera.bottom=-250;
scene.add(sun);

/* ===================== WORLD BUILD ===================== */
(function buildWorld(){
  var i,b,wp,p,hw;
  /* Road */
  var rPos=[],rIdx=[];
  for(i=0;i<N;i++){
    wp=waypoints[i];p=wpPerp(i);hw=TW*0.5;
    rPos.push(wp.x+p.x*hw,wp.y+0.06,wp.z+p.z*hw,
              wp.x-p.x*hw,wp.y+0.06,wp.z-p.z*hw);
    if(i<N-1){b=i*2;rIdx.push(b,b+2,b+1,b+1,b+2,b+3);}
  }
  b=(N-1)*2;rIdx.push(b,0,b+1,b+1,0,1);
  var rGeo=new THREE.BufferGeometry();
  rGeo.setAttribute('position',new THREE.Float32BufferAttribute(rPos,3));
  rGeo.setIndex(rIdx);rGeo.computeVertexNormals();
  var road=new THREE.Mesh(rGeo,new THREE.MeshStandardMaterial({color:0x1a1a1a,roughness:0.9}));
  road.receiveShadow=true;scene.add(road);

  /* Kerbs */
  var kPos=[],kIdx=[];
  for(i=0;i<N;i++){
    wp=waypoints[i];p=wpPerp(i);hw=TW*0.5;
    kPos.push(wp.x+p.x*(hw+CURB),wp.y+0.07,wp.z+p.z*(hw+CURB),
              wp.x+p.x*hw,        wp.y+0.07,wp.z+p.z*hw,
              wp.x-p.x*hw,        wp.y+0.07,wp.z-p.z*hw,
              wp.x-p.x*(hw+CURB), wp.y+0.07,wp.z-p.z*(hw+CURB));
    if(i<N-1){
      b=i*4;
      kIdx.push(b,b+4,b+1,b+1,b+4,b+5,b+2,b+6,b+3,b+3,b+6,b+7);
    }
  }
  var kGeo=new THREE.BufferGeometry();
  kGeo.setAttribute('position',new THREE.Float32BufferAttribute(kPos,3));
  kGeo.setIndex(kIdx);kGeo.computeVertexNormals();
  scene.add(new THREE.Mesh(kGeo,new THREE.MeshStandardMaterial({color:0xcc2200,roughness:0.8})));

  /* Armco barriers */
  var baPos=[],baIdx=[];
  hw=TW*0.5+CURB+0.3;
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

  /* Armco red stripe */
  var strPos=[],strIdx=[];
  hw=TW*0.5+CURB+0.3;
  var sY=BH*0.58,sH=0.28;
  for(i=0;i<N;i++){
    wp=waypoints[i];p=wpPerp(i);
    strPos.push(wp.x+p.x*hw,wp.y+sY,      wp.z+p.z*hw,
                wp.x+p.x*hw,wp.y+sY+sH,   wp.z+p.z*hw,
                wp.x-p.x*hw,wp.y+sY,      wp.z-p.z*hw,
                wp.x-p.x*hw,wp.y+sY+sH,   wp.z-p.z*hw);
    if(i<N-1){b=i*4;strIdx.push(b,b+4,b+1,b+1,b+4,b+5,b+2,b+6,b+3,b+3,b+6,b+7);}
  }
  var strGeo=new THREE.BufferGeometry();
  strGeo.setAttribute('position',new THREE.Float32BufferAttribute(strPos,3));
  strGeo.setIndex(strIdx);strGeo.computeVertexNormals();
  scene.add(new THREE.Mesh(strGeo,new THREE.MeshStandardMaterial({color:0xCC0000,roughness:0.5})));

  /* Terrain */
  var TS=6,TER=430,cx=0,cz=-5;
  var cols=Math.floor(TER/TS)+1,rows=cols;
  var gPos2=new Float32Array(cols*rows*3),gIdx2=[];
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
      gPos2[vi*3]=vx;gPos2[vi*3+1]=(bY-0.8)*(1-t)+(hill-2.5)*t;gPos2[vi*3+2]=vz;
      if(r<rows-1&&c<cols-1){gIdx2.push(vi,vi+cols,vi+1,vi+1,vi+cols,vi+cols+1);}
    }
  }
  var tGeo=new THREE.BufferGeometry();
  tGeo.setAttribute('position',new THREE.Float32BufferAttribute(gPos2,3));
  tGeo.setIndex(gIdx2);tGeo.computeVertexNormals();
  scene.add(new THREE.Mesh(tGeo,new THREE.MeshStandardMaterial({color:0x3d7a2a,roughness:1.0})));

  /* Trees along barriers */
  var mTrunk=new THREE.MeshStandardMaterial({color:0x4a2f12,roughness:1});
  var mLeaf=new THREE.MeshStandardMaterial({color:0x1a5c0a,roughness:1});
  for(var ti=0;ti<N;ti+=10){
    var twp=waypoints[ti],tp=wpPerp(ti);
    var offs=TW*0.5+CURB+5;
    [1,-1].forEach(function(s){
      var jitter=(ti%20===0?3:ti%30===0?-2:1);
      var ox=twp.x+tp.x*s*(offs+jitter),oz=twp.z+tp.z*s*(offs+jitter);
      var trunk=new THREE.Mesh(new THREE.CylinderGeometry(0.4,0.6,3.5,7),mTrunk);
      trunk.position.set(ox,twp.y+1.75,oz);scene.add(trunk);
      var crown=new THREE.Mesh(new THREE.ConeGeometry(3.2,5.5,7),mLeaf);
      crown.position.set(ox,twp.y+6.5,oz);scene.add(crown);
    });
  }

  /* Grandstands on main straight */
  var mConc=new THREE.MeshStandardMaterial({color:0xc8c8b8,roughness:0.85});
  var mSeat=new THREE.MeshStandardMaterial({color:0x1E41FF,roughness:0.8});
  [[80,0,-28],[80,0,36]].forEach(function(s){
    var base=new THREE.Mesh(new THREE.BoxGeometry(55,2,10),mConc);
    base.position.set(s[0],s[1]+1,s[2]);scene.add(base);
    var tier=new THREE.Mesh(new THREE.BoxGeometry(55,3.5,6),mConc);
    tier.position.set(s[0],s[1]+4.25,s[2]+(s[2]<0?3.5:-3.5));scene.add(tier);
    var seats=new THREE.Mesh(new THREE.BoxGeometry(53,0.3,5),mSeat);
    seats.position.set(s[0],s[1]+6.1,s[2]+(s[2]<0?3.5:-3.5));scene.add(seats);
  });

  /* Tire stacks at hairpin */
  var mTireW=new THREE.MeshStandardMaterial({color:0xffffff,roughness:0.9});
  var mTireR=new THREE.MeshStandardMaterial({color:0xcc0000,roughness:0.9});
  [-3,-1.5,0,1.5,3].forEach(function(oz){
    [mTireW,mTireR,mTireW].forEach(function(mt,ti2){
      var tc=new THREE.Mesh(new THREE.CylinderGeometry(0.6,0.6,0.9,12),mt);
      tc.position.set(-148,ti2*0.9+0.45,oz);scene.add(tc);
    });
  });
})();

/* ===================== CAR BUILDER ===================== */
function buildCar(bodyColor){
  var g=new THREE.Group();
  var PI=Math.PI;
  var mNav=new THREE.MeshPhysicalMaterial({color:bodyColor,metalness:0.06,roughness:0.26,clearcoat:1.0,clearcoatRoughness:0.05});
  var mRed=new THREE.MeshPhysicalMaterial({color:0xCC0000,metalness:0.04,roughness:0.23,clearcoat:1.0,clearcoatRoughness:0.04});
  var mGold=new THREE.MeshPhysicalMaterial({color:0xC9A85C,metalness:0.84,roughness:0.12,clearcoat:0.55,clearcoatRoughness:0.14});
  var mC=new THREE.MeshPhysicalMaterial({color:0x0a0a0a,metalness:0.24,roughness:0.55,clearcoat:0.55,clearcoatRoughness:0.25});
  var mT=new THREE.MeshStandardMaterial({color:0x030303,metalness:0.0,roughness:0.98});
  var mR=new THREE.MeshPhysicalMaterial({color:0xBBBBBB,metalness:0.96,roughness:0.03,clearcoat:0.3});
  var mG=new THREE.MeshStandardMaterial({color:0x888888,metalness:0.74,roughness:0.28});
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
  // Suspension
  [[1.72,-0.80],[1.72,0.80]].forEach(function(w){var wx=w[0],wz=w[1],ci=wz>0?0.22:-0.22;add(bar(wx,0.00,wz,1.52,0.08,ci,0.016,mG));add(bar(wx,0.00,wz,1.18,0.06,ci,0.016,mG));add(bar(wx,-0.28,wz,1.50,-0.22,ci,0.016,mG));add(bar(wx,-0.28,wz,1.16,-0.22,ci,0.016,mG));});
  [[-1.52,-0.88],[-1.52,0.88]].forEach(function(w){var wx=w[0],wz=w[1],ci=wz>0?0.24:-0.24;add(bar(wx,0.00,wz,-1.10,0.06,ci,0.016,mG));add(bar(wx,0.00,wz,-1.42,0.04,ci,0.016,mG));add(bar(wx,-0.28,wz,-1.12,-0.20,ci,0.016,mG));add(bar(wx,-0.28,wz,-1.44,-0.18,ci,0.016,mG));});
  // Wheels
  function addWheel(x,z,tw){
    var wg=new THREE.Group();var fs=(z>0)?1:-1,fY=fs*tw*0.46;
    var R=0.340,ri=0.260,hw=tw*0.50;
    var tp=[new THREE.Vector2(ri,hw+0.004),new THREE.Vector2(ri+0.022,hw-0.002),new THREE.Vector2(R-0.030,hw-0.004),new THREE.Vector2(R-0.006,hw-0.026),new THREE.Vector2(R,hw-0.056),new THREE.Vector2(R,0),new THREE.Vector2(R,-(hw-0.056)),new THREE.Vector2(R-0.006,-(hw-0.026)),new THREE.Vector2(R-0.030,-(hw-0.004)),new THREE.Vector2(ri+0.022,-(hw-0.002)),new THREE.Vector2(ri,-(hw+0.004))];
    wg.add(mk(new THREE.LatheGeometry(tp,52),mT,0,0,0));
    wg.add(mk(new THREE.CylinderGeometry(ri-0.002,ri-0.002,tw+0.006,44),mR,0,0,0));
    wg.add(mk(new THREE.CylinderGeometry(ri-0.004,ri-0.004,0.026,44),mR,0,fY,0));
    wg.add(mk(new THREE.CylinderGeometry(0.065,0.065,tw+0.032,16),mG,0,0,0));
    for(var wi=0;wi<5;wi++){var pv=new THREE.Group();pv.rotation.y=wi*2*PI/5;pv.position.y=fY;var sp=new THREE.Mesh(new THREE.BoxGeometry(ri-0.044,0.020,0.020),mR);sp.position.x=(ri-0.044)/2;pv.add(sp);wg.add(pv);}
    wg.rotation.x=PI/2;wg.position.set(x,-0.22,z);g.add(wg);
  }
  addWheel(1.72,-0.80,0.300);addWheel(1.72,0.80,0.300);
  addWheel(-1.52,-0.88,0.405);addWheel(-1.52,0.88,0.405);
  g.traverse(function(o){if(o.isMesh){o.castShadow=true;o.receiveShadow=true;}});
  return g;
}

/* ===================== GAME STATE ===================== */
var gameState='IDLE',paused=false,raceTime=0,animRunning=false;
var playerGrp=null,aiGrps=[];
var camSmooth=new THREE.Vector3(),camVel=new THREE.Vector3();
var cockpitMode=false;

var P={x:0,z:0,y:0,heading:0,speed:0,yawRate:0,tIdx:0,lap:1,
  sector:0,sectorStart:0,lapStart:0,s1:0,s2:0,s3:0,
  bestS1:Infinity,bestS2:Infinity,bestS3:Infinity,
  rpmVal:0,gear:1,drs:false,_drsKey:false,
  tireWear:1.0,tireTempF:0.3,tireTempR:0.3,_lapGuard:false};

var AI=[
  {tIdx:N-4, latOff: 1.8, spdFac:0.94,x:0,z:0,y:0,heading:0,speed:0,yawRate:0,lap:1,_inStart:false,_skipFirst:true},
  {tIdx:N-8, latOff:-1.8, spdFac:0.86,x:0,z:0,y:0,heading:0,speed:0,yawRate:0,lap:1,_inStart:false,_skipFirst:true},
  {tIdx:N-12,latOff: 1.8, spdFac:0.78,x:0,z:0,y:0,heading:0,speed:0,yawRate:0,lap:1,_inStart:false,_skipFirst:true}
];

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

function initRace(){
  if(playerGrp){scene.remove(playerGrp);playerGrp=null;}
  aiGrps.forEach(function(gr){scene.remove(gr);});aiGrps=[];
  playerGrp=buildCar(0x1E41FF);scene.add(playerGrp);
  var sp=spawnPos(0,0);
  P.x=sp.x;P.z=sp.z;P.y=sp.y;P.heading=sp.heading;
  P.speed=0;P.yawRate=0;P.tIdx=0;P.lap=1;P.rpmVal=0;P.gear=1;
  P.drs=false;P._drsKey=false;P.tireWear=1;P.tireTempF=0.3;P.tireTempR=0.3;
  P.sector=0;P.sectorStart=0;P.lapStart=0;P.s1=0;P.s2=0;P.s3=0;
  P.bestS1=Infinity;P.bestS2=Infinity;P.bestS3=Infinity;P._lapGuard=false;
  AI.forEach(function(ai,i){
    var ag=buildCar(AI_COLORS[i]);scene.add(ag);aiGrps.push(ag);
    var as=spawnPos(ai.tIdx,ai.latOff);
    ai.x=as.x;ai.z=as.z;ai.y=as.y;ai.heading=as.heading;
    ai.speed=0;ai.yawRate=0;ai.lap=1;ai._inStart=false;ai._skipFirst=true;
    ag.position.set(ai.x,ai.y+RIDE_H,ai.z);ag.rotation.y=ai.heading-Math.PI/2;
  });
  playerGrp.position.set(P.x,P.y+RIDE_H,P.z);playerGrp.rotation.y=P.heading-Math.PI/2;
  var d0=wpDir(0);
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
function updatePlayer(dt){
  var thr=(keys['ArrowUp']||keys['KeyW'])?1:0;
  var brk=(keys['ArrowDown']||keys['KeyS'])?1:0;
  var sl=(keys['ArrowLeft']||keys['KeyA'])?1:0;
  var sr=(keys['ArrowRight']||keys['KeyD'])?1:0;
  var si=sl-sr;
  if(keys['ShiftLeft']||keys['ShiftRight']){if(!P._drsKey){P.drs=!P.drs;P._drsKey=true;}}
  else{P._drsKey=false;}
  if(hudDrs) hudDrs.className='hud-drs'+(P.drs?' on':'');
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
    P.lap++;P.sector=0;P.sectorStart=raceTime;P.lapStart=raceTime;
    if(P.lap>LAPS){P.lap=LAPS;gameState='FINISHED';showPodium();}
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
  var tgt=waypoints[(ai.tIdx+8)%N];
  var dx=tgt.x-ai.x,dz=tgt.z-ai.z;
  var th=Math.atan2(dx,dz),dh=th-ai.heading;
  while(dh>Math.PI) dh-=2*Math.PI;while(dh<-Math.PI) dh+=2*Math.PI;
  ai.yawRate+=(dh*1.2-ai.yawRate*0.08)*Math.min(dt*5,1);ai.yawRate*=Math.max(0,1-dt*7.5);
  ai.heading+=ai.yawRate*ai.speed*0.6*dt;
  var corner=Math.min(Math.abs(dh)/0.8,1);
  var tspd=MAX_SPD*(0.63+0.37*(1-corner))*ai.spdFac;
  ai.speed=Math.max(0,Math.min(ai.speed+((tspd>ai.speed?8000:-14000)/CAR_MASS)*dt,MAX_SPD*0.92));
  ai.x+=Math.sin(ai.heading)*ai.speed*dt;
  ai.z+=Math.cos(ai.heading)*ai.speed*dt;
  ai.tIdx=closestWP(ai.x,ai.z,ai.tIdx);
  ai.y=waypoints[ai.tIdx].y;
  var _abp=wpPerp(ai.tIdx),_awp=waypoints[ai.tIdx];
  var _alat=(ai.x-_awp.x)*_abp.x+(ai.z-_awp.z)*_abp.z;
  if(Math.abs(_alat)>TW*0.5+CURB){var _as=_alat>0?1:-1;ai.x=_awp.x+_abp.x*_as*(TW*0.5+CURB);ai.z=_awp.z+_abp.z*_as*(TW*0.5+CURB);ai.speed*=0.5;}
  if(ai.tIdx<12&&!ai._inStart){
    ai._inStart=true;
    if(ai._skipFirst){ai._skipFirst=false;}
    else{ai.lap++;if(ai.lap>LAPS) ai.lap=LAPS;}
  }
  if(ai.tIdx>=20) ai._inStart=false;
}

/* ===================== CAMERA ===================== */
function updateCamera(dt){
  var spd=P.speed;
  if(gameState==='COUNTDOWN'){
    var d0=wpDir(0),wy0=waypoints[0].y;
    gameCam.position.set(P.x-d0.x*28,wy0+13,P.z-d0.z*28);
    gameCam.lookAt(new THREE.Vector3(P.x+d0.x*8,wy0+0.5,P.z+d0.z*8));
    gameCam.fov=80;gameCam.updateProjectionMatrix();
    camSmooth.copy(gameCam.position);camVel.set(0,0,0);return;
  }
  if(gameState!=='RACING'&&gameState!=='FINISHED') return;
  if(cockpitMode){
    var dc=wpDir(P.tIdx);
    gameCam.position.set(P.x+dc.x*0.3,P.y+RIDE_H+0.55,P.z+dc.z*0.3);
    gameCam.rotation.set(0,P.heading+Math.PI,0);
    gameCam.fov=92;gameCam.updateProjectionMatrix();return;
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

function getPos(){
  var all=[{n:'P',lap:P.lap,ti:P.tIdx}];
  AI.forEach(function(ai,i){all.push({n:'A'+i,lap:ai.lap,ti:ai.tIdx});});
  all.sort(function(a,b){return b.lap!==a.lap?b.lap-a.lap:b.ti-a.ti;});
  var pp=0;all.forEach(function(e,i){if(e.n==='P') pp=i;});return pp;
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
  if(wheelInd) wheelInd.setAttribute('transform','rotate('+(P.yawRate*20*180/Math.PI)+')');
  drawMinimap();
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
  minimapCtx.strokeStyle='#333';minimapCtx.lineWidth=3;minimapCtx.beginPath();
  for(var j=0;j<N;j+=2){
    var ww=waypoints[j];
    if(j===0) minimapCtx.moveTo(wx(ww.x),wz2(ww.z));
    else minimapCtx.lineTo(wx(ww.x),wz2(ww.z));
  }
  minimapCtx.closePath();minimapCtx.stroke();
  minimapCtx.fillStyle='#1E41FF';minimapCtx.beginPath();
  minimapCtx.arc(wx(P.x),wz2(P.z),5,0,Math.PI*2);minimapCtx.fill();
  var ac=['#C0C0C0','#DC0000','#FF8000'];
  AI.forEach(function(ai,i){
    minimapCtx.fillStyle=ac[i];minimapCtx.beginPath();
    minimapCtx.arc(wx(ai.x),wz2(ai.z),3,0,Math.PI*2);minimapCtx.fill();
  });
}

/* ===================== PODIUM ===================== */
function showPodium(){
  var all=[{n:AI_NAMES[0],lap:P.lap,ti:P.tIdx}];
  AI.forEach(function(ai,i){all.push({n:AI_NAMES[i+1],lap:ai.lap,ti:ai.tIdx});});
  all.sort(function(a,b){return b.lap!==a.lap?b.lap-a.lap:b.ti-a.ti;});
  var rows='',medals=['1ST','2ND','3RD','4TH'];
  all.forEach(function(e,i){
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
  initRace();startCountdown();
  if(!animRunning){animRunning=true;requestAnimationFrame(loop);}
}

function closeGame(){
  overlay.className='';
  if(podiumOvl) podiumOvl.className='';
  if(pauseOvl) pauseOvl.className='';
  if(lightsBarEl) lightsBarEl.className='';
  if(gridMsgEl) gridMsgEl.className='';
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
    var hint=document.createElement('div');
    hint.style.cssText='position:absolute;bottom:10px;left:50%;transform:translateX(-50%);font-size:.55rem;color:#444;letter-spacing:.25em;text-transform:uppercase;pointer-events:none';
    hint.textContent='CLICK TO RACE';
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
  if(playerGrp){playerGrp.position.set(P.x,P.y+RIDE_H,P.z);playerGrp.rotation.y=P.heading-Math.PI/2;}
  AI.forEach(function(ai,i){
    if(aiGrps[i]){aiGrps[i].position.set(ai.x,ai.y+RIDE_H,ai.z);aiGrps[i].rotation.y=ai.heading-Math.PI/2;}
  });
  updateCamera(dt);
  updateHUD();
  renderer.render(scene,gameCam);
}

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
        title=dict(text=title, font=dict(color="#555555", family=_FONT, size=9)),
        gridcolor="#111111",
        zerolinecolor="#222222",
        tickfont=dict(color=_TICK, family=_FONT, size=9),
        showgrid=True,
        showspikes=True,
        spikecolor="#1E41FF",
        spikethickness=1,
        spikedash="solid",
        spikemode="across",
    )


def _layout_2d(title: str, xaxis_title: str = "", yaxis_title: str = "",
               height: int = 420, hovermode: str = "x unified") -> go.Layout:
    return go.Layout(
        title=dict(
            text=title,
            font=dict(color="#ffffff", family=_FONT, size=13),
            x=0,
            xanchor="left",
            pad=dict(t=4, l=0),
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        xaxis=_axis_2d(xaxis_title),
        yaxis=_axis_2d(yaxis_title),
        font=dict(family=_FONT, color="#ffffff"),
        margin=dict(l=52, r=24, t=56, b=48),
        height=height,
        showlegend=True,
        legend=dict(
            font=dict(family=_FONT, color="#888888", size=10),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#222222",
            borderwidth=1,
        ),
        hovermode=hovermode,
        hoverdistance=80,
        spikedistance=400,
        hoverlabel=dict(
            bgcolor="#0a0a0a",
            bordercolor="#1E41FF",
            font=dict(family=_FONT, size=11, color="#ffffff"),
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
    placeholders, params = _ref_params(team_refs)
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
    placeholders, params = _ref_params(team_refs)
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
        fill_color = _hex_to_rgba(color if color != "#FFFFFF" else "#888888", 0.10)
        surname = driver.split()[-1]

        fig.add_trace(go.Scatter(
            x=g["round"], y=g["points"],
            mode="lines+markers",
            name=surname,
            line=dict(color=color, width=3, shape="spline"),
            marker=dict(size=6, color=color, line=dict(color="#000000", width=1)),
            fill="tozeroy",
            fillcolor=fill_color,
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
                    line=dict(color="#000000", width=1),
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

    fig.add_hrect(y0=0.5, y1=1.5, fillcolor="rgba(255,215,0,0.04)", layer="below", line_width=0)
    fig.add_hrect(y0=0.5, y1=3.5, fillcolor="rgba(255,255,255,0.02)", layer="below", line_width=0)

    for i, (driver, g) in enumerate(df.groupby("driver")):
        color = _driver_color(driver, i)
        fig.add_trace(go.Scatter(
            x=g["round"], y=g["position"],
            mode="lines+markers",
            name=driver.split()[-1],
            line=dict(color=color, width=4, shape="spline"),
            marker=dict(size=9, color=color, line=dict(color="#000000", width=1.5)),
            hovertemplate="<b>%{fullData.name}</b>  P%{y}<extra></extra>",
        ))
    return fig


def chart_points_gap_2d(traj_df: pd.DataFrame) -> go.Figure:
    """Points gap between the two drivers — latest season. Filled area above/below zero."""
    layout = _layout_2d(
        "POINTS GAP · DRIVERS",
        xaxis_title="ROUND",
        yaxis_title="GAP (PTS)",
        height=420,
        hovermode="x unified",
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
    d_a, d_b = drivers[0], drivers[1]
    if d_a not in pivot.columns or d_b not in pivot.columns:
        return fig

    rounds = pivot.index.tolist()
    gap = (pivot[d_a] - pivot[d_b]).tolist()
    color_a = _driver_color(d_a, 0)
    color_b = _driver_color(d_b, 1)

    fig.add_hline(y=0, line=dict(color="#333333", width=1, dash="dot"))

    pos_gap = [g if g >= 0 else 0 for g in gap]
    neg_gap = [g if g < 0 else 0 for g in gap]

    fig.add_trace(go.Scatter(
        x=rounds, y=pos_gap,
        mode="none", fill="tozeroy",
        fillcolor=_hex_to_rgba(color_a if color_a != "#FFFFFF" else "#888888", 0.18),
        name=d_a.split()[-1], showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=neg_gap,
        mode="none", fill="tozeroy",
        fillcolor=_hex_to_rgba(color_b if color_b != "#FFFFFF" else "#888888", 0.18),
        name=d_b.split()[-1], showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=gap,
        mode="lines+markers",
        name=f"{d_a.split()[-1]} vs {d_b.split()[-1]}",
        line=dict(color="#ffffff", width=2, shape="spline"),
        marker=dict(size=5, color="#ffffff"),
        hovertemplate="%{y:+d} pts<extra></extra>",
    ))

    fig.add_annotation(
        x=0.01, y=0.96, xref="paper", yref="paper",
        text=d_a.split()[-1], font=dict(color=color_a, family=_FONT, size=9),
        showarrow=False, xanchor="left",
    )
    fig.add_annotation(
        x=0.01, y=0.06, xref="paper", yref="paper",
        text=d_b.split()[-1], font=dict(color=color_b, family=_FONT, size=9),
        showarrow=False, xanchor="left",
    )
    return fig


def chart_heatmap_2d(traj_df: pd.DataFrame) -> go.Figure:
    """Performance matrix — finish position per driver per round, all seasons as color cells."""
    layout = _layout_2d(
        "PERFORMANCE MATRIX · ALL SEASONS",
        xaxis_title="ROUND",
        height=260,
        hovermode="closest",
    )
    fig = go.Figure(layout=layout)
    layout.yaxis.update(showgrid=False)

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
        [0.00, "#FFD700"],
        [0.10, "#1E41FF"],
        [0.40, "#555555"],
        [0.75, "#CC0000"],
        [1.00, "#1a0000"],
    ]

    fig.add_trace(go.Heatmap(
        z=z, x=cols, y=rows,
        text=text, texttemplate="%{text}",
        colorscale=colorscale,
        zmin=1, zmax=20,
        showscale=False,
        xgap=2, ygap=2,
        textfont=dict(family=_FONT, size=8, color="#ffffff"),
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
        line=dict(color="#333333", width=1.5, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))

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
                line=dict(color="#000000", width=0.5),
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

    fig1 = chart_championship_2d(traj)
    fig2 = chart_positions_bump_2d(traj)
    fig3 = chart_points_gap_2d(traj)
    fig4 = chart_heatmap_2d(traj)
    fig5 = chart_grid_finish_2d(gf)

    _cfg = {"displayModeBar": "hover", "scrollZoom": False}
    div1 = fig1.to_html(full_html=False, include_plotlyjs="cdn",  config=_cfg)
    div2 = fig2.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div3 = fig3.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div4 = fig4.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div5 = fig5.to_html(full_html=False, include_plotlyjs=False, config=_cfg)

    years = sorted(traj["year"].unique()) if not traj.empty else []
    year_range = f"{years[0]}–{years[-1]}" if years else ""
    subtitle = f"PERFORMANCE DASHBOARD \xb7 {year_range}" if year_range else "PERFORMANCE DASHBOARD"

    total_races = int(traj["round"].nunique()) if not traj.empty else "—"
    total_wins  = int((gf["finish"] == 1).sum()) if not gf.empty else "—"
    stat_html = (
        f'<div class="stat-card"><div class="stat-val">{year_range or "—"}</div>'
        f'<div class="stat-lbl">Seasons</div></div>'
        f'<div class="stat-card"><div class="stat-val">{total_races}</div>'
        f'<div class="stat-lbl">Race Rounds Analyzed</div></div>'
        f'<div class="stat-card"><div class="stat-val">{total_wins}</div>'
        f'<div class="stat-lbl">Wins in Dataset</div></div>'
        f'<div class="stat-card"><div class="stat-val">4</div>'
        f'<div class="stat-lbl">Constructors Titles</div></div>'
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    logo_html = ""
    if os.path.exists(_LOGO_PATH):
        with open(_LOGO_PATH, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        logo_html = f'<img src="data:image/jpeg;base64,{b64}" class="logo-img" alt="Red Bull Racing">'

    html = (
        _HTML_TEMPLATE
        .replace("PLACEHOLDER_TITLE",    team_name)
        .replace("PLACEHOLDER_SUBTITLE", subtitle)
        .replace("PLACEHOLDER_LOGO",     logo_html)
        .replace("PLACEHOLDER_STATS",    stat_html)
        .replace("PLACEHOLDER_C1",       div1)
        .replace("PLACEHOLDER_C2",       div2)
        .replace("PLACEHOLDER_C3",       div3)
        .replace("PLACEHOLDER_C4",       div4)
        .replace("PLACEHOLDER_C5",       div5)
        .replace("PLACEHOLDER_TS",       ts)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
