import html
from typing import Any

import pandas as pd
import pydeck as pdk

from application.dataset_upload_validation import REPOSITORY_MAP_SYNTHETIC_ASPECT
from config.global_config import SENTIMENT_LABELS


def _first_non_empty_str(series: pd.Series) -> str:
    for v in series:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return ""


def _aspect_mean_sentiment_pl(mean_score: float) -> str:
    if mean_score > 0.05:
        return "Pozytywny (średnia etykieta dla wybranego aspektu)"
    if mean_score < -0.05:
        return "Negatywny (średnia etykieta dla wybranego aspektu)"
    return "Neutralny / mieszany (średnia etykieta dla wybranego aspektu)"


_LABEL_SET = frozenset(SENTIMENT_LABELS)


def _classify_aspect_cell(value: Any) -> str | None:
    """Map a cell to a canonical sentiment label, or None if unlabeled / invalid."""
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


def _place_color_from_label_counts(n_pos: int, n_neg: int) -> list[int]:
    """RGBA for one place from positive vs negative mentions (ABSA labels)."""
    signal = n_pos + n_neg
    if signal == 0:
        return [158, 158, 158, 200]
    alpha = min(255, 165 + int(90 * min(signal, 10) / 10))
    if n_neg > n_pos:
        if n_neg >= 2 * max(n_pos, 1):
            return [198, 40, 40, alpha]
        return [230, 106, 36, alpha]
    if n_pos > n_neg:
        return [56, 142, 60, alpha]
    return [245, 180, 0, alpha]


def _summary_pl_for_aspect_counts(
    n_pos: int, n_neg: int, n_neu: int, n_nm: int
) -> str:
    parts = [
        f"pozytywne: {n_pos}",
        f"negatywne: {n_neg}",
        f"neutralne: {n_neu}",
        f"niewspomniane: {n_nm}",
    ]
    return ", ".join(parts)


def _place_group_keys(df: pd.DataFrame) -> pd.Series:
    """Stable key per venue: gmap_id when present, else rounded lat|lon."""
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


def aggregate_points_by_place(df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    """One scatter point per place; mean `aspect` for color; all reviews in `reviews_html`."""
    if df.empty:
        return df
    work = df.copy()
    work["_place_key"] = _place_group_keys(work)
    rows: list[dict[str, Any]] = []

    for _key, g in work.groupby("_place_key", sort=False):
        g_lat = pd.to_numeric(g["latitude"], errors="coerce")
        g_lon = pd.to_numeric(g["longitude"], errors="coerce")
        lat_m = float(g_lat.mean())
        lon_m = float(g_lon.mean())
        parts: list[str] = []
        for i, (_, row) in enumerate(g.iterrows(), start=1):
            raw = row.get("text", "")
            t = html.escape(_truncate_text(str(raw) if raw is not None else ""))
            if (
                aspect != REPOSITORY_MAP_SYNTHETIC_ASPECT
                and aspect in row.index
                and pd.notna(row[aspect])
            ):
                av = html.escape(str(row[aspect]))
                parts.append(
                    f"<div style='margin-bottom:6px'><b>{i}.</b> {t}<br/><small>{html.escape(aspect)}: {av}</small></div>"
                )
            else:
                parts.append(f"<div style='margin-bottom:6px'><b>{i}.</b> {t}</div>")
        reviews_html = "".join(parts)
        place_name = (
            (_first_non_empty_str(g["name"]) if "name" in g.columns else "")
            or "[Placeholder] Brak nazwy w pliku — po uzupełnieniu zbioru pojawi się tutaj nazwa miejsca."
        )
        place_category = (
            _first_non_empty_str(g["category"]) if "category" in g.columns else ""
        ) or "—"
        gmap_str = (
            _first_non_empty_str(g["gmap_id"]) if "gmap_id" in g.columns else ""
        ) or "—"

        rec: dict[str, Any] = {
            "latitude": lat_m,
            "longitude": lon_m,
            "n_reviews": int(len(g)),
            "reviews_html": reviews_html,
            "place_name": place_name,
            "place_category": place_category,
            "gmap_id_display": gmap_str,
        }
        precomputed_color: list[int] | None = None
        if aspect in work.columns:
            if aspect == REPOSITORY_MAP_SYNTHETIC_ASPECT:
                rec[aspect] = ""
                precomputed_color = [158, 158, 158, 200]
                rec["sentiment_summary"] = (
                    "[Placeholder] Sentyment miejsca — dane pojawią się po podłączeniu "
                    "zbioru z etykietami aspektów."
                )
            else:
                labels = [_classify_aspect_cell(v) for v in g[aspect]]
                n_pos = sum(1 for x in labels if x == "positive")
                n_neg = sum(1 for x in labels if x == "negative")
                n_neu = sum(1 for x in labels if x == "neutral")
                n_nm = sum(1 for x in labels if x == "notmentioned")
                n_labeled = sum(1 for x in labels if x is not None)

                if n_labeled > 0:
                    rec[aspect] = _summary_pl_for_aspect_counts(
                        n_pos, n_neg, n_neu, n_nm
                    )
                    precomputed_color = _place_color_from_label_counts(n_pos, n_neg)
                    sig = n_pos + n_neg
                    if sig == 0:
                        rec["sentiment_summary"] = (
                            "Brak opinii z etykietą poz./neg. dla tego aspektu "
                            "(tylko neutralne / niewspomniane)."
                        )
                    elif n_neg > n_pos:
                        rec["sentiment_summary"] = "Dominują opinie negatywne dla aspektu."
                    elif n_pos > n_neg:
                        rec["sentiment_summary"] = "Dominują opinie pozytywne dla aspektu."
                    else:
                        rec["sentiment_summary"] = (
                            "Równa liczba pozytywnych i negatywnych etykiet."
                        )
                else:
                    sub = pd.to_numeric(g[aspect], errors="coerce")
                    mean_a = float(sub.mean()) if sub.notna().any() else float("nan")
                    rec[aspect] = (
                        f"{mean_a:.3f}"
                        if not pd.isna(mean_a)
                        else "[Placeholder] Brak wartości"
                    )
                    if pd.isna(mean_a):
                        precomputed_color = [158, 158, 158, 200]
                        rec["sentiment_summary"] = (
                            "[Placeholder] Brak wartości sentymentu dla tego aspektu."
                        )
                    else:
                        rec["sentiment_summary"] = _aspect_mean_sentiment_pl(mean_a)
                        if mean_a > 0:
                            intensity = min(
                                255, int(100 + 155 * min(abs(mean_a), 1.0))
                            )
                            precomputed_color = [76, 175, 80, intensity]
                        elif mean_a < 0:
                            intensity = min(
                                255, int(100 + 155 * min(abs(mean_a), 1.0))
                            )
                            precomputed_color = [244, 67, 54, intensity]
                        else:
                            precomputed_color = [255, 193, 7, 100]
        else:
            rec["sentiment_summary"] = (
                "[Placeholder] Sentyment — wymagane kolumny aspektów w zbiorze."
            )
            precomputed_color = [158, 158, 158, 200]

        if precomputed_color is not None:
            rec["color"] = precomputed_color
            rec["score"] = 0.0

        rec["sentiment_profile_placeholder"] = (
            "[Placeholder] Pełny profil sentymentu (wszystkie aspekty, trendy) — "
            "do podpięcia w kolejnej iteracji zbioru."
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty and "color" in out.columns:
        out.attrs["map_colors_precomputed"] = True
    return out


def _truncate_text(s: str, max_len: int = 800) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


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
    return pdk.Layer(
        "ScatterplotLayer",
        id=layer_id,
        data=df_scored,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius=50 * radius_scale,
        pickable=True,
        auto_highlight=True,
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


def build_map(
    df: pd.DataFrame,
    aspect: str,
    viz_type: str,
    view_state: pdk.ViewState,
    point_size: float = 1.0,
    layer_id: str = "absa-points",
    scatter_df: pd.DataFrame | None = None,
) -> pdk.Deck:
    layers: list[pdk.Layer] = []
    df_scatter = df
    grouped_tooltip = False
    if viz_type in ("Points", "Combined"):
        df_scatter = (
            scatter_df
            if scatter_df is not None
            else aggregate_points_by_place(df, aspect)
        )
        grouped_tooltip = not df_scatter.empty and "reviews_html" in df_scatter.columns
        layers.append(create_scatter_layer(df_scatter, aspect, point_size, layer_id))
    if viz_type in ("Heatmap", "Combined"):
        h = create_heatmap_layer(df, aspect)
        if h:
            layers.append(h)
    if viz_type in ("3D Hexagons", "Combined"):
        hx = create_hexagon_layer(df, aspect)
        if hx:
            layers.append(hx)

    if grouped_tooltip:
        if aspect == REPOSITORY_MAP_SYNTHETIC_ASPECT:
            tooltip = {
                "html": (
                    "<div style='max-width:400px;max-height:300px;overflow:auto'>"
                    "<small><b>Opinii:</b> {n_reviews}</small><br/><hr/>"
                    "{reviews_html}</div>"
                ),
                "style": {"backgroundColor": "steelblue", "color": "white"},
            }
        else:
            aspect_esc = html.escape(aspect)
            tooltip = {
                "html": (
                    "<div style='max-width:400px;max-height:300px;overflow:auto'>"
                    "<small><b>Opinii:</b> {n_reviews}</small><br/>"
                    f"<b>Średnia ({aspect_esc}):</b> {{{aspect}}}<br/><hr/>"
                    "{reviews_html}</div>"
                ),
                "style": {"backgroundColor": "steelblue", "color": "white"},
            }
    else:
        tooltip = {
            "html": f"<b>Review:</b> {{text}}<br/><b>{html.escape(aspect)}:</b> {{{aspect}}}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }
    # Carto dark basemap — no API key. Mapbox URLs require MAPBOX_API_KEY or the
    # browser requests ...access_token=no-token and fails (often reported as CORS).
    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="dark",
        map_provider="carto",
        tooltip=tooltip,
    )
