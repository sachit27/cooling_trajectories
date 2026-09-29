"""Build cartographic base layers (LV95) for the revision figures.

Sources (both open, redistributed as npm packages):
* City of Zurich statistical office, sszvis 2.0.0 ``topo/stadt-zurich.json``
  (Stadtkreise, statistical quarters, Lake Zurich within the city), WGS84.
* swiss-maps 4.7.0 (interactivethings; swisstopo generalised boundaries,
  2025 edition): Switzerland, cantons, lakes, municipalities, LV95.
"""
import json, os, sys
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union

SSZ = sys.argv[1]; SWM = sys.argv[2]; OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
topo = json.load(open(SSZ))
sc, tr = topo["transform"]["scale"], topo["transform"]["translate"]
arcs = []
for a in topo["arcs"]:
    xy = np.cumsum(np.array(a, float), axis=0)
    arcs.append(np.column_stack([xy[:, 0] * sc[0] + tr[0], xy[:, 1] * sc[1] + tr[1]]))

def arc(i):
    return arcs[i] if i >= 0 else arcs[~i][::-1]

def ring(idx):
    pts = [arc(i)[(0 if k == 0 else 1):] for k, i in enumerate(idx)]
    return np.vstack(pts)

def geom(g):
    t = g["type"]
    if t == "Polygon":
        r = [ring(x) for x in g["arcs"]]
        return Polygon(r[0], r[1:])
    if t == "MultiPolygon":
        return MultiPolygon([Polygon(ring(p[0]), [ring(x) for x in p[1:]]) for p in g["arcs"]])
    if t == "LineString":
        return LineString(ring(g["arcs"]))
    if t == "MultiLineString":
        return MultiLineString([ring(x) for x in g["arcs"]])
    raise ValueError(t)

for name in ["stadtkreise", "statistische_quartiere", "lakezurich", "stadtkreis_lakebounds"]:
    gs = topo["objects"][name]["geometries"]
    g = gpd.GeoDataFrame([x.get("properties") or {} for x in gs],
                         geometry=[geom(x) for x in gs], crs=4326).to_crs(2056)
    g["geometry"] = g.buffer(0)
    g.to_file(os.path.join(OUT, f"{name}.gpkg"), driver="GPKG")
    print(name, len(g), g.total_bounds.round(0), (g.area.sum()/1e6).round(2))
kreise = gpd.read_file(os.path.join(OUT, "stadtkreise.gpkg"))
city = gpd.GeoDataFrame(geometry=[unary_union(kreise.geometry)], crs=2056)
city.to_file(os.path.join(OUT, "city_boundary.gpkg"), driver="GPKG")
print("city area km2", city.area.iloc[0] / 1e6)
for n in ["country", "cantons", "lakes", "municipalities"]:
    g = gpd.read_file(os.path.join(SWM, f"{n}.shp"))
    if g.crs is None: g = g.set_crs(2056)
    g.to_file(os.path.join(OUT, f"ch_{n}.gpkg"), driver="GPKG")
    print(n, len(g))
