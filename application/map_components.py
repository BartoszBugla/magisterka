from __future__ import annotations

import html
import math
from typing import Any

import pandas as pd
import pydeck as pdk

from application.dataset_upload_validation import REPOSITORY_MAP_SYNTHETIC_ASPECT
from config.global_config import SENTIMENT_LABELS

_LABEL_SET = frozenset(SENTIMENT_LABELS)


def _first_non_empty_str(series: pd.Series) -> str:
    for v in series:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return ""


def _classify_aspect_cell(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1 or value == 1.0:
            return "positive"
        if value == -1 or value == -1.0:
            return "negative"
        if value == 0 or value == 0.0:
            return "neutral"
    s = str(value).strip().lower()
    if not s or s in ("nan", "null", "none"):
        return "notmentioned"
    if s in _LABEL_SET:
        return s
    return None


COLOR_STRONG_NEGATIVE = [198, 40, 40]  # Dark Red
COLOR_MILD_NEGATIVE = [230, 106, 36]  # Orange
COLOR_POSITIVE = [56, 142, 60]  # Green
COLOR_NEUTRAL = [245, 180, 0]  # Yellow
COLOR_NO_DATA = [158, 158, 158, 200]  # Gray (includes fixed alpha)

ALPHA_BASE = 165
ALPHA_MAX_ADDITION = 90
SIGNAL_SATURATION_LIMIT = 10
MAX_ALPHA = 255


def _place_color_from_label_counts(n_pos: int, n_neg: int) -> list[int]:
    signal = n_pos + n_neg

    if signal == 0:
        return COLOR_NO_DATA.copy()

    signal_factor = min(signal, SIGNAL_SATURATION_LIMIT) / SIGNAL_SATURATION_LIMIT
    alpha = min(MAX_ALPHA, ALPHA_BASE + int(ALPHA_MAX_ADDITION * signal_factor))

    if n_neg > n_pos:
        is_strongly_negative = n_neg >= 2 * max(n_pos, 1)
        base_color = (
            COLOR_STRONG_NEGATIVE if is_strongly_negative else COLOR_MILD_NEGATIVE
        )

    elif n_pos > n_neg:
        base_color = COLOR_POSITIVE

    else:
        base_color = COLOR_NEUTRAL

    return [*base_color, alpha]


def _hover_sentiment_label(n_pos: int, n_neg: int, n_neu: int, n_nm: int) -> str:
    signal = n_pos + n_neg
    if signal == 0:
        if n_neu > 0:
            return "Neutral"
        return "No signal"
    if n_pos > n_neg:
        return "Mostly positive"
    if n_neg > n_pos:
        return "Mostly negative"
    return "Mixed"


def place_group_keys(df: pd.DataFrame) -> pd.Series:
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    lat_r = lat.round(6).astype(str)
    lon_r = lon.round(6).astype(str)
    fallback = lat_r + "|" + lon_r
    if "gmap_id" not in df.columns:
        return fallback
    gid = df["gmap_id"]
    has_gid = (
        gid.notna() & (gid.astype(str).str.strip() != "") & (gid.astype(str) != "nan")
    )
    out = fallback.copy()
    out.loc[has_gid] = "gmap:" + gid.loc[has_gid].astype(str)
    return out


def radius_for_review_count(n: int, base: float = 35.0, scale: float = 18.0) -> float:
    return base + scale * math.log2(max(n, 1) + 1)


def aggregate_points_by_place(df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    work["_place_key"] = place_group_keys(work)
    rows: list[dict[str, Any]] = []

    for key, g in work.groupby("_place_key", sort=False):
        g_lat = pd.to_numeric(g["latitude"], errors="coerce")
        g_lon = pd.to_numeric(g["longitude"], errors="coerce")
        n_reviews = len(g)

        place_name = (
            _first_non_empty_str(g["name"]) if "name" in g.columns else ""
        ) or "Unknown place"
        place_category = (
            _first_non_empty_str(g["category"]) if "category" in g.columns else ""
        ) or "—"
        gmap_str = (
            _first_non_empty_str(g["gmap_id"]) if "gmap_id" in g.columns else ""
        ) or "—"

        rec: dict[str, Any] = {
            "_place_key": key,
            "latitude": float(g_lat.mean()),
            "longitude": float(g_lon.mean()),
            "n_reviews": n_reviews,
            "_radius": radius_for_review_count(n_reviews),
            "place_name": place_name,
            "place_category": place_category,
            "gmap_id_display": gmap_str,
        }

        precomputed_color: list[int] | None = None
        hover_sentiment = "—"

        if aspect in work.columns:
            if aspect == REPOSITORY_MAP_SYNTHETIC_ASPECT:
                precomputed_color = [158, 158, 158, 200]
                hover_sentiment = "—"
            else:
                labels = [_classify_aspect_cell(v) for v in g[aspect]]
                n_pos = sum(1 for x in labels if x == "positive")
                n_neg = sum(1 for x in labels if x == "negative")
                n_neu = sum(1 for x in labels if x == "neutral")
                n_nm = sum(1 for x in labels if x == "notmentioned")
                n_labeled = sum(1 for x in labels if x is not None)

                if n_labeled > 0:
                    precomputed_color = _place_color_from_label_counts(n_pos, n_neg)
                    hover_sentiment = _hover_sentiment_label(n_pos, n_neg, n_neu, n_nm)
                else:
                    sub = pd.to_numeric(g[aspect], errors="coerce")
                    mean_a = float(sub.mean()) if sub.notna().any() else float("nan")
                    if pd.isna(mean_a):
                        precomputed_color = [158, 158, 158, 200]
                        hover_sentiment = "No data"
                    else:
                        hover_sentiment = (
                            "Positive"
                            if mean_a > 0.05
                            else ("Negative" if mean_a < -0.05 else "Neutral / mixed")
                        )
                        if mean_a > 0:
                            intensity = min(255, int(100 + 155 * min(abs(mean_a), 1.0)))
                            precomputed_color = [76, 175, 80, intensity]
                        elif mean_a < 0:
                            intensity = min(255, int(100 + 155 * min(abs(mean_a), 1.0)))
                            precomputed_color = [244, 67, 54, intensity]
                        else:
                            precomputed_color = [255, 193, 7, 100]
        else:
            precomputed_color = [158, 158, 158, 200]

        rec["_hover_sentiment"] = hover_sentiment
        if precomputed_color is not None:
            rec["color"] = precomputed_color
            rec["score"] = 0.0

        rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty and "color" in out.columns:
        out.attrs["map_colors_precomputed"] = True
    return out


def compute_aspect_scores(df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    if aspect not in df.columns:
        gray = [128, 128, 128, 100]
        return df.assign(score=0.0, color=[gray] * len(df))
    scores = pd.to_numeric(df[aspect], errors="coerce").fillna(0.0).values
    colors: list[list[int]] = []
    for score in scores:
        if score > 0:
            intensity = min(255, int(100 + 155 * abs(score)))
            colors.append([76, 175, 80, intensity])
        elif score < 0:
            intensity = min(255, int(100 + 155 * abs(score)))
            colors.append([244, 67, 54, intensity])
        else:
            colors.append([255, 193, 7, 100])
    return df.assign(score=scores, color=colors)


def create_scatter_layer(
    df: pd.DataFrame,
    aspect: str,
    radius_scale: float = 1.0,
    layer_id: str = "absa-points",
) -> pdk.Layer:
    if df.attrs.get("map_colors_precomputed") and "color" in df.columns:
        df_scored = df
    else:
        df_scored = compute_aspect_scores(df, aspect)

    use_data_radius = "_radius" in df_scored.columns
    return pdk.Layer(
        "ScatterplotLayer",
        id=layer_id,
        data=df_scored,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius="_radius" if use_data_radius else 50 * radius_scale,
        radius_min_pixels=5,
        radius_max_pixels=40,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 80],
    )


def create_heatmap_layer(
    df: pd.DataFrame, aspect: str, intensity: float = 1.0
) -> pdk.Layer | None:
    if aspect not in df.columns:
        return None
    df_aspect = df[df[aspect].notna()].copy()
    if len(df_aspect) == 0:
        return None
    df_aspect["weight"] = df_aspect[aspect].apply(
        lambda x: 1.0 if x > 0 else (0.5 if x == 0 else 0.2)
    )
    return pdk.Layer(
        "HeatmapLayer",
        id="absa-heatmap",
        data=df_aspect,
        get_position=["longitude", "latitude"],
        get_weight="weight",
        aggregation="SUM",
        radiusPixels=60,
        intensity=intensity,
        threshold=0.1,
    )


def create_hexagon_layer(
    df: pd.DataFrame, aspect: str, elevation_scale: float = 100
) -> pdk.Layer | None:
    if aspect not in df.columns:
        return None
    df_aspect = df[df[aspect].notna()].copy()
    if len(df_aspect) == 0:
        return None
    df_aspect["elevation"] = df_aspect[aspect].abs() * 100
    return pdk.Layer(
        "HexagonLayer",
        id="absa-hex",
        data=df_aspect,
        get_position=["longitude", "latitude"],
        radius=200,
        elevation_scale=elevation_scale,
        elevation_range=[0, 1000],
        extruded=True,
        pickable=True,
    )


def _build_hover_tooltip(aspect: str) -> dict:
    aspect_esc = html.escape(aspect)
    is_synthetic = aspect == REPOSITORY_MAP_SYNTHETIC_ASPECT

    if is_synthetic:
        body = (
            "<b>{place_name}</b><br/>"
            "<span style='opacity:.85'>{place_category}</span><br/>"
            "<small>Reviews: {n_reviews}</small>"
        )
    else:
        body = (
            "<b>{place_name}</b><br/>"
            "<span style='opacity:.85'>{place_category}</span><br/>"
            f"<small>{aspect_esc}: {{_hover_sentiment}}</small><br/>"
            "<small>Reviews: {n_reviews}</small>"
        )

    return {
        "html": (
            f"<div style='max-width:280px;padding:4px;font-size:13px'>"
            f"{body}"
            f"<div style='margin-top:4px;font-size:11px;opacity:.6'>"
            f"Click for details</div></div>"
        ),
        "style": {
            "backgroundColor": "#1e1e1e",
            "color": "white",
            "borderRadius": "6px",
            "padding": "8px 10px",
        },
    }


def build_map(
    df: pd.DataFrame,
    aspect: str,
    viz_type: str,
    view_state: pdk.ViewState,
    point_size: float = 1.0,
    layer_id: str = "absa-points",
    scatter_df: pd.DataFrame | None = None,
) -> tuple[pdk.Deck, pd.DataFrame]:
    layers: list[pdk.Layer] = []
    df_scatter = pd.DataFrame()

    if viz_type in ("Points", "Combined"):
        df_scatter = (
            scatter_df
            if scatter_df is not None
            else aggregate_points_by_place(df, aspect)
        )
        layers.append(create_scatter_layer(df_scatter, aspect, point_size, layer_id))

    if viz_type in ("Heatmap", "Combined"):
        h = create_heatmap_layer(df, aspect)
        if h:
            layers.append(h)

    if viz_type in ("3D Hexagons", "Combined"):
        hx = create_hexagon_layer(df, aspect)
        if hx:
            layers.append(hx)

    tooltip = _build_hover_tooltip(aspect)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="dark",
        map_provider="carto",
        tooltip=tooltip,
    )
    return deck, df_scatter
