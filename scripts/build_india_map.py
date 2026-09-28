"""One-off generator: fetches a real, public India boundary outline and bakes
it into a static SVG-path asset for the Mission Map page.

Source: datameet/maps' india-composite.geojson — a full-detail MultiPolygon
maintained by India's civic-tech open-data community (chosen specifically
because it's India-curated, not a generic Natural Earth extract whose
international-boundary convention for J&K/Arunachal Pradesh commonly
diverges from India's official government line — a real accuracy concern
for a national-level submission). Fetched once here rather than at request
time so the Mission Map works fully offline during a demo.

Simplification: uses shapely's Douglas-Peucker simplify() (topology-
preserving) rather than naive point-striding, so the coastline/border shape
stays geometrically faithful at a much smaller path size than the raw
survey-grade source.

Projection: a simple linear equirectangular fit (lng -> x, lat -> y, y
flipped) scaled to a fixed viewBox. frontend/mission-map.js reimplements the
exact same formula so markers/trajectory plot in perfect alignment with the
baked path — keep both in sync if this changes.

Usage (from project root):
    python scripts/build_india_map.py
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
from shapely.geometry import shape
from shapely import simplify

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "frontend" / "assets" / "india_outline.json"

SOURCE_URL = "https://raw.githubusercontent.com/datameet/maps/master/Country/india-composite.geojson"

# Douglas-Peucker tolerance in degrees. ~0.008 deg ~= 800m at this latitude —
# preserves coastline/border character while cutting point count drastically.
SIMPLIFY_TOLERANCE = 0.008

# Fixed geographic bounds the projection is fit to (slightly padded around
# India's mainland + island territories). Must match frontend/mission-map.js's PROJECTION.
LNG_MIN, LNG_MAX = 68.0, 97.5
LAT_MIN, LAT_MAX = 6.5, 36.0
VIEW_W, VIEW_H = 1000, 1000

# Minimum ring bounding-box span (in projected px) to keep — drops
# sub-pixel islet fragments left over after simplification.
MIN_RING_SPAN_PX = 4.0


def project(lng: float, lat: float) -> tuple[float, float]:
    x = (lng - LNG_MIN) / (LNG_MAX - LNG_MIN) * VIEW_W
    y = (1.0 - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * VIEW_H
    return round(x, 2), round(y, 2)


def ring_to_path(coords) -> str:
    pts = [project(lng, lat) for lng, lat, *_ in coords]
    dedup = [pts[0]]
    for p in pts[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    if len(dedup) < 3:
        return ""
    xs = [p[0] for p in dedup]
    ys = [p[1] for p in dedup]
    if (max(xs) - min(xs)) + (max(ys) - min(ys)) < MIN_RING_SPAN_PX:
        return ""
    d = f"M{dedup[0][0]},{dedup[0][1]} "
    d += " ".join(f"L{x},{y}" for x, y in dedup[1:])
    d += " Z"
    return d


def polygon_paths(poly) -> list[str]:
    paths = [ring_to_path(list(poly.exterior.coords))]
    for interior in poly.interiors:
        paths.append(ring_to_path(list(interior.coords)))
    return [p for p in paths if p]


def main():
    print(f"[build_india_map] Fetching {SOURCE_URL} ...")
    resp = requests.get(SOURCE_URL, timeout=60)
    resp.raise_for_status()
    geo = resp.json()

    total_points_before = 0
    paths: list[str] = []

    for feature in geo["features"]:
        geom = shape(feature["geometry"])
        total_points_before += sum(len(p.exterior.coords) for p in _iter_polygons(geom))

        simplified = simplify(geom, SIMPLIFY_TOLERANCE, preserve_topology=True)
        for poly in _iter_polygons(simplified):
            paths.extend(polygon_paths(poly))

    print(f"[build_india_map] {len(paths)} path ring(s) kept after simplify+fragment filtering "
          f"(source had {total_points_before} raw exterior points)")

    combined_path = " ".join(paths)

    out = {
        "path": combined_path,
        "viewBox": f"0 0 {VIEW_W} {VIEW_H}",
        "projection_bounds": {
            "lng_min": LNG_MIN, "lng_max": LNG_MAX,
            "lat_min": LAT_MIN, "lat_max": LAT_MAX,
            "view_w": VIEW_W, "view_h": VIEW_H,
        },
        "source": SOURCE_URL,
        "simplify_tolerance_deg": SIMPLIFY_TOLERANCE,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out), encoding="utf-8")
    print(f"[build_india_map] Wrote {OUT_PATH} ({len(combined_path)} chars of path data)")


def _iter_polygons(geom):
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


if __name__ == "__main__":
    main()
