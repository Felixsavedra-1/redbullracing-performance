from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import text
from sqlalchemy.engine import Engine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analytics import _ref_params, championship_trajectory

logger = logging.getLogger("f1_analytics")

_BG             = "#000000"
_FONT           = "Courier New, monospace"
_GRID           = "#1e1e1e"
_TICK           = "#888888"
_ACCENT         = "#FFFFFF"   # white
_MAX_COLOR      = "#1E41FF"   # Verstappen blue
_TEAMMATE_COLOR = "#FF1800"   # teammate red


def _driver_color(name: str, idx: int) -> str:
    if "Verstappen" in name:
        return _MAX_COLOR
    return _TEAMMATE_COLOR if idx > 0 else _MAX_COLOR


# --------------------------------------------------------------------------- #
#  HTML template                                                                #
# --------------------------------------------------------------------------- #

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PLACEHOLDER_TITLE</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;color:#fff;font-family:'Courier New',monospace}
header{padding:40px 32px 32px;border-bottom:3px solid #FFFFFF}
h1{font-size:1.5rem;font-weight:700;text-transform:uppercase;letter-spacing:.2em}
.sub{color:#666;font-size:.7rem;letter-spacing:.15em;margin-top:8px;text-transform:uppercase}
.car-viewer{padding:24px 0 0;display:flex;justify-content:center}
#f1car{display:block;width:100%;height:440px}
.charts{padding:32px;display:grid;grid-template-columns:1fr;gap:32px}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:32px}
.chart-section{border-top:1px solid #FFFFFF;padding-top:16px}
@media(max-width:860px){.chart-row{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>PLACEHOLDER_HEADING</h1>
  <p class="sub">PLACEHOLDER_SUBTITLE</p>
</header>
<div class="car-viewer">
  <canvas id="f1car"></canvas>
</div>
<div class="charts">
  <div class="chart-section">PLACEHOLDER_C1</div>
  <div class="chart-row">
    <div class="chart-section">PLACEHOLDER_C2</div>
    <div class="chart-section">PLACEHOLDER_C3</div>
  </div>
</div>
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

  // Lighting — supplements IBL for hard shadows
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

  // Ground — dark reflective surface
  var gnd=new THREE.Mesh(new THREE.PlaneGeometry(20,16),
    new THREE.MeshStandardMaterial({color:0x060606,metalness:0.0,roughness:0.52,envMapIntensity:0.5}));
  gnd.rotation.x=-Math.PI/2; gnd.position.y=-0.57; gnd.receiveShadow=true; scene.add(gnd);

  // Materials — MeshPhysicalMaterial with clearcoat for all painted/metallic surfaces
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

  var car=new THREE.Group();

  // Chassis — 5-section tapered monocoque (Coke-bottle waist visible from above)
  car.add(bx(0.70,0.40,0.62,mNav, 1.25,0.05,0));
  car.add(bx(0.80,0.40,0.70,mNav, 0.50,0.05,0));
  car.add(bx(0.50,0.38,0.52,mNav,-0.15,0.05,0));
  car.add(bx(0.50,0.40,0.62,mNav,-0.65,0.05,0));
  car.add(bx(0.75,0.36,0.55,mNav,-1.28,0.05,0));
  car.add(bx(1.10,0.21,0.54,mNav,-0.30,0.31,0));
  car.add(bx(0.58,0.13,0.60,mNav, 0.53,0.27,0));

  // Nose — smooth LatheGeometry revolution profile (replaces 4 coarse cylinders)
  // rotation.z=-PI/2 maps local Y to world +X; position.x=1.60 places base at monocoque front
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

  // Front wing — 3 carbon cascade elements + red endplates
  car.add(bx(0.30,0.040,2.12,mC,2.86,-0.245,0));
  car.add(bx(0.24,0.036,1.94,mC,2.66,-0.200,0));
  car.add(bx(0.18,0.030,1.74,mC,2.48,-0.158,0));
  [-1.03,1.03].forEach(function(z){
    car.add(bx(0.50,0.32,0.05,mRed,2.66,-0.095,z));
    car.add(bx(0.20,0.10,0.05,mRed,2.84,-0.30,z));
  });
  [-0.20,0.20].forEach(function(z){car.add(bx(0.06,0.24,0.04,mC,2.72,-0.075,z));});

  // Sidepods — navy body with red accent on outer face
  [-0.53,0.53].forEach(function(z){
    var sg=z>0?1:-1;
    car.add(bx(1.62,0.35,0.33,mNav,-0.24,-0.03,z));
    car.add(bx(0.90,0.32,0.012,mRed,-0.16,-0.03,z+sg*0.166));
    car.add(bx(0.09,0.21,0.09,mC,0.37,0.04,z));
    car.add(bx(1.22,0.08,0.26,mC,-0.39,-0.23,z));
  });

  // Bargeboards
  for(var bi=0;bi<3;bi++){
    [-0.34-bi*0.09,0.34+bi*0.09].forEach(function(z){car.add(bx(0.09,0.20,0.03,mC,0.84,-0.04,z));});
  }

  // Engine cover — sculpted 3-piece PU hump + gold stripe repositioned to fin top
  car.add(bx(0.70,0.52,0.054,mNav,-0.20,0.46,0));
  car.add(bx(0.28,0.28,0.054,mNav, 0.09,0.39,0));
  car.add(bx(0.70,0.040,0.054,mGold,-0.20,0.71,0));
  car.add(bx(0.21,0.19,0.32,mNav,0.29,0.39,0));

  // Halo — central post + top arch + angled rear legs
  car.add(bx(0.06,0.34,0.042,mG,0.60,0.52,0));
  car.add(bx(0.50,0.072,0.56,mG,0.57,0.70,0));
  car.add(bar(0.40,0.70,-0.28, 0.18,0.30,-0.26, 0.022,mG));
  car.add(bar(0.40,0.70, 0.28, 0.18,0.30, 0.26, 0.022,mG));

  // Mirrors — stanchion + polished face
  car.add(bx(0.04,0.28,0.04,mC,0.50,0.38,-0.26));
  car.add(bx(0.04,0.28,0.04,mC,0.50,0.38, 0.26));
  car.add(bx(0.11,0.07,0.17,mR,0.47,0.53,-0.26));
  car.add(bx(0.11,0.07,0.17,mR,0.47,0.53, 0.26));

  // Helmet (Verstappen blue)
  var helm=mk(new THREE.SphereGeometry(0.14,24,18),mB,0.41,0.36,0);
  helm.scale.set(1.2,0.92,1.1); car.add(helm);

  // Floor (1.80 m wide) + diffuser strakes + diffuser ramp wedge
  car.add(bx(3.12,0.042,1.80,mC,-0.08,-0.21,0));
  for(var ds=-3;ds<=3;ds++){car.add(bx(0.64,0.13,0.03,mC,-1.83,-0.15,ds*0.245));}
  // Diffuser ramp: rz=2.719 → dir(-0.912,+0.410); rear exit(-2.021,-0.060), floor junction(-1.639,-0.232)
  car.add(bx(0.42,0.030,1.60,mC,-1.830,-0.146,0,0,0,2.719));

  // Rear wing assembly — swan-neck pylons connect crash structure to wing (fixes floating wing)
  car.add(bx(0.33,0.23,0.42,mNav,-1.71,0.07,0));
  car.add(bx(0.54,0.065,0.90,mC,-1.85,0.19,0));
  // Swan-neck pylons: crash top (y=0.185) → wing underside (y=0.661), 18 mm carbon tube
  car.add(bar(-1.62,0.185,-0.13, -2.00,0.661,-0.18, 0.018,mC));
  car.add(bar(-1.62,0.185, 0.13, -2.00,0.661, 0.18, 0.018,mC));
  car.add(bx(0.27,0.058,1.60,mC,-2.03,0.69,0));
  car.add(bx(0.18,0.046,1.48,mC,-1.87,0.63,0));
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

  // Wheels — LatheGeometry tire profile; flat tread, rounded shoulders, bulging sidewalls
  function addWheel(x,z,tw){
    var g=new THREE.Group();
    var fs=(z>0)?1:-1,fY=fs*tw*0.46;
    var R=0.340,ri=0.260,hw=tw*0.50;
    // Tire cross-section profile revolved around the axle (Y axis pre-rotation)
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
  // Enable cast + receive shadows on every mesh in the car group
  car.traverse(function(o){if(o.isMesh){o.castShadow=true;o.receiveShadow=true;}});

  function animate(){requestAnimationFrame(animate);car.rotation.y+=0.004;renderer.render(scene,cam);}
  animate();
  window.addEventListener('resize',function(){
    var nW=c.parentElement.offsetWidth||900;
    cam.aspect=nW/H; cam.updateProjectionMatrix(); renderer.setSize(nW,H);
  });
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
        title=dict(text=title, font=dict(color="#888888", family=_FONT, size=10)),
        gridcolor=_GRID,
        zerolinecolor=_GRID,
        tickfont=dict(color=_TICK, family=_FONT, size=10),
        linecolor=_ACCENT,
        linewidth=2,
        showline=True,
        showgrid=True,
        mirror=False,
    )


def _layout_2d(title: str, xaxis_title: str = "", yaxis_title: str = "",
               height: int = 420, hovermode: str = "x unified") -> go.Layout:
    return go.Layout(
        title=dict(
            text=title,
            font=dict(color="#ffffff", family=_FONT, size=14),
            x=0,
            xanchor="left",
            pad=dict(t=4, l=0),
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        xaxis=_axis_2d(xaxis_title),
        yaxis=_axis_2d(yaxis_title),
        font=dict(family=_FONT, color="#ffffff"),
        margin=dict(l=52, r=16, t=56, b=48),
        height=height,
        showlegend=True,
        legend=dict(
            font=dict(family=_FONT, color="#888888", size=10),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=_ACCENT,
            borderwidth=1,
        ),
        hovermode=hovermode,
        hoverlabel=dict(
            bgcolor="#000000",
            bordercolor=_ACCENT,
            font=dict(family=_FONT, size=11, color="#ffffff"),
            namelength=-1,
        ),
    )


# --------------------------------------------------------------------------- #
#  SQL helpers                                                                  #
# --------------------------------------------------------------------------- #

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
    """Championship points per round — latest season, one line per driver."""
    if traj_df.empty:
        return go.Figure(layout=_layout_2d(
            "CHAMPIONSHIP TRAJECTORY", xaxis_title="ROUND", yaxis_title="POINTS", height=440,
        ))

    latest = int(traj_df["year"].max())
    df = traj_df[traj_df["year"] == latest].sort_values("round")
    layout = _layout_2d(
        f"CHAMPIONSHIP TRAJECTORY · {latest}",
        xaxis_title="ROUND",
        yaxis_title="POINTS",
        height=440,
    )
    fig = go.Figure(layout=layout)

    for i, (driver, g) in enumerate(df.groupby("driver")):
        color = _driver_color(driver, i)
        fig.add_trace(go.Scatter(
            x=g["round"], y=g["points"],
            mode="lines+markers",
            name=driver.split()[-1],
            line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color, line=dict(color="#000000", width=1)),
            hovertemplate="<b>%{fullData.name}</b>  %{y} pts<extra></extra>",
        ))
    return fig


def chart_race_positions_2d(traj_df: pd.DataFrame) -> go.Figure:
    """Race finish positions per round — latest season, one trace per driver.
    Y axis is inverted so P1 sits at the top."""
    if traj_df.empty:
        layout = _layout_2d("RACE POSITIONS", xaxis_title="ROUND", yaxis_title="FINISH", height=400)
        layout.yaxis.update(autorange="reversed", dtick=5)
        return go.Figure(layout=layout)

    latest = int(traj_df["year"].max())
    df = traj_df[traj_df["year"] == latest].sort_values("round")
    layout = _layout_2d(
        f"RACE POSITIONS · {latest}",
        xaxis_title="ROUND",
        yaxis_title="FINISH",
        height=400,
    )
    layout.yaxis.update(autorange="reversed", dtick=5)
    fig = go.Figure(layout=layout)

    for i, (driver, g) in enumerate(df.groupby("driver")):
        color = _driver_color(driver, i)
        fig.add_trace(go.Scatter(
            x=g["round"], y=g["position"],
            mode="lines+markers",
            name=driver.split()[-1],
            line=dict(color=color, width=1.8),
            marker=dict(size=6, color=color, line=dict(color="#000000", width=1)),
            hovertemplate="<b>%{fullData.name}</b>  P%{y}<extra></extra>",
        ))
    return fig


def chart_grid_finish_2d(df: pd.DataFrame) -> go.Figure:
    """Grid position vs finish position scatter — all seasons combined.
    Points below the diagonal gained positions; above lost them."""
    layout = _layout_2d(
        "GRID vs FINISH · ALL SEASONS",
        xaxis_title="GRID",
        yaxis_title="FINISH",
        height=400,
        hovermode="closest",
    )
    fig = go.Figure(layout=layout)
    if df.empty:
        return fig

    # Diagonal reference line (grid = finish, no change)
    mx = max(df["grid"].max(), df["finish"].max()) + 1
    fig.add_trace(go.Scatter(
        x=[1, mx], y=[1, mx],
        mode="lines",
        name="no change",
        line=dict(color="#444444", width=1, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    ))

    for i, (driver, g) in enumerate(df.groupby("driver")):
        color = _driver_color(driver, i)
        delta = g["grid"] - g["finish"]   # positive = gained positions
        fig.add_trace(go.Scatter(
            x=g["grid"], y=g["finish"],
            mode="markers",
            name=driver.split()[-1],
            customdata=list(zip(g["year"], delta)),
            marker=dict(
                size=6, color=color, opacity=0.80,
                line=dict(color="#000000", width=0.5),
            ),
            hovertemplate=(
                "<b>%{fullData.name}</b>  %{customdata[0]}<br>"
                "Grid P%{x} → Finish P%{y}<br>"
                "%{customdata[1]:+d} positions"
                "<extra></extra>"
            ),
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
    traj = championship_trajectory(engine, team_refs)
    if traj.empty:
        logger.warning("championship_trajectory returned no data — dashboard charts will be blank")

    gf = _grid_finish_df(engine, team_refs)

    fig1 = chart_championship_2d(traj)
    fig2 = chart_race_positions_2d(traj)
    fig3 = chart_grid_finish_2d(gf)

    _cfg = {"displayModeBar": "hover", "scrollZoom": False}
    div1 = fig1.to_html(full_html=False, include_plotlyjs="cdn",  config=_cfg)
    div2 = fig2.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
    div3 = fig3.to_html(full_html=False, include_plotlyjs=False, config=_cfg)

    years = sorted(traj["year"].unique()) if not traj.empty else []
    year_range = f"{years[0]}–{years[-1]}" if years else ""
    subtitle = f"PERFORMANCE DASHBOARD \xb7 {year_range}" if year_range else "PERFORMANCE DASHBOARD"

    html = (
        _HTML_TEMPLATE
        .replace("PLACEHOLDER_TITLE",    f"{team_name} — F1 ANALYTICS")
        .replace("PLACEHOLDER_HEADING",  f"{team_name} — F1 ANALYTICS")
        .replace("PLACEHOLDER_SUBTITLE", subtitle)
        .replace("PLACEHOLDER_C1",       div1)
        .replace("PLACEHOLDER_C2",       div2)
        .replace("PLACEHOLDER_C3",       div3)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
