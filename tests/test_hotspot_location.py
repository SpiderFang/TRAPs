"""``traps.hotspot_location`` 之合成資料測試（無需 aggregate 輸出）。"""

from __future__ import annotations

import numpy as np

from traps import hotspot_location as hl


def test_label_components_connectivity():
    m = np.zeros((6, 6), bool)
    m[0, 0] = m[1, 1] = True          # 對角相鄰
    m[4, 4:6] = True
    _, n8 = hl.label_components(m, 8)
    _, n4 = hl.label_components(m, 4)
    assert (n8, n4) == (2, 3)


def test_region_stats_enrichment():
    valid = np.ones((10, 10), bool)
    w = np.zeros((10, 10))
    w[:, :2] = 1.0                    # 全部特徵在左側 2 欄（面積 20%）
    s = hl.region_stats(w, valid, np.arange(10)[None, :] < 2 + np.zeros((10, 1), int))
    assert s["share_feature"] == 1.0 and abs(s["share_valid"] - 0.2) < 1e-12 and abs(s["enrichment"] - 5.0) < 1e-9
    # 區域超出有效格時只計有效格
    valid2 = valid.copy()
    valid2[:, 1] = False
    s2 = hl.region_stats(w, valid2, np.ones((10, 10), bool))
    assert s2["n_valid_in"] == 90 and s2["n_feature_in"] == 10.0


def test_orientation_east_west_and_north_south():
    lon = np.arange(0.0, 3.0, 0.01)
    lat = np.arange(0.0, 3.0, 0.01)
    m = np.zeros((lat.size, lon.size), bool)
    m[100:103, 20:240] = True         # 東西向長條（寬 3 格、長 220 格）
    t = hl.component_table(m, lon, lat)
    assert len(t) == 1 and t[0]["n_cells"] == 3 * 220
    assert hl.is_east_west(t[0]["angle_from_east_deg"]) and t[0]["elongation"] > 10
    mv = np.zeros_like(m)
    mv[20:240, 100:103] = True
    tv = hl.component_table(mv, lon, lat)
    assert abs(tv[0]["angle_from_east_deg"] - 90.0) < 1.0 and not hl.is_east_west(tv[0]["angle_from_east_deg"])
    md = np.zeros_like(m)
    for i in range(100):
        md[20 + i, 20 + i] = True     # 45° 對角線
    td = hl.component_table(md, lon, lat)
    assert abs(td[0]["angle_from_east_deg"] - 45.0) < 5.0 and not hl.is_east_west(td[0]["angle_from_east_deg"])


def test_land_components_island_vs_mainland():
    ocean = np.ones((20, 20), bool)
    ocean[:, :4] = False              # 碰到邊界的陸地
    ocean[10:12, 10:12] = False       # 內部島嶼
    labels, touches, n = hl.land_components(ocean)
    assert n == 2
    main = labels[0, 0]
    isl = labels[10, 10]
    assert touches[main] and not touches[isl]
    lon2d, lat2d = hl.grid_coordinates(np.arange(20) * 0.01, np.arange(20) * 0.01)
    x, y = hl.to_km(lon2d, lat2d, 0.0, 0.0)
    rr, cc = np.array([10, 10]), np.array([13, 8])
    lab, dist = hl.nearest_land_label((rr, cc), labels, x, y)
    assert lab[0] == main or lab[0] == isl
    assert lab[1] == isl and dist[1] < dist[0] + 5.0


def test_verdict_rule():
    assert hl.verdict_by_rule(0.6, 1.4) == "supported"
    assert hl.verdict_by_rule(0.85, 0.96) == "partly"
    assert hl.verdict_by_rule(0.3, 1.8) == "supported"
    assert hl.verdict_by_rule(0.35, 1.05) == "unsupported"
    assert hl.verdict_by_rule(float("nan"), float("nan")) == "unsupported"


def test_within_km_of_point_and_box():
    lon2d, lat2d = hl.grid_coordinates(np.linspace(120, 121, 11), np.linspace(22, 23, 11))
    x, y = hl.to_km(lon2d, lat2d, 120.5, 22.5)
    m = hl.within_km_of_point(x, y, 0.0, 0.0, 15.0)
    assert m[5, 5] and not m[0, 0]
    b = hl.lonlat_box(lon2d, lat2d, lon_min=120.5, lat_max=22.5)
    assert b[5, 5] and not b[6, 5] and not b[5, 4]


def test_present_rule():
    assert hl.present_rule(0.25, 0.9) == "supported"
    assert hl.present_rule(0.12, 1.6) == "supported"
    assert hl.present_rule(0.096, 1.58) == "unsupported"
    assert hl.present_rule(0.12, 0.74) == "unsupported"
