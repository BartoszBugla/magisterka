"""Zbiór scenariuszy B: widok mapy + panel szczegółów miejsca.

Mapa (warstwa, agregacja, tooltip):
    B1. Klucz miejsca ``place_group_keys`` — ten sam ``gmap_id`` grupuje opinie w jeden punkt (niezależnie od drobnych różnic współrzędnych).
    B2. ``aggregate_points_by_place`` — dwa rekordy jednej restauracji dają jeden punkt z ``n_reviews`` oraz promieniem zależnym od liczby opinii.
    B3. ``build_map`` — ciemny styl mapy, warstwa ``ScatterplotLayer``, szablon HTML tooltipu (placeholder ``{place_name}`` + „Kliknij…”) oraz wiersz agregatu z faktyczną nazwą miejsca.
    B4. Kolor punktu dla etykiet sentymentu — przy przewadze pozytywnych etykiet intensywność alf kanału odpowiada sygnałowi (test na zielonym RGB).
Szczegóły miejsca (logika panelu bez UI Streamlit):
    B5. ``find_metadata_for_place`` znajduje rekord dodatkowych metadanych po dopasowaniu ``name``.
    B6. ``reviews_for_place`` + ``sentiment_counts`` — filtrowanie wierszy do jednego `_place_key` i zliczenia etykiet dla aspektu.
    B7. ``filter_reviews`` zawęża listę opinii do wybranego sentymentu przy danym aspekcie.
"""

from __future__ import annotations

import unittest

import pandas as pd
import pydeck as pdk

from application.dataset_upload_validation import REPOSITORY_MAP_SYNTHETIC_ASPECT
from application.map_components import (
    COLOR_POSITIVE,
    aggregate_points_by_place,
    build_map,
    place_group_keys,
    radius_for_review_count,
)
from application.place_details import (
    compute_place_keys,
    filter_reviews,
    find_metadata_for_place,
    reviews_for_place,
    sentiment_counts,
)
from config.global_config import SentimentLabel


class TestMapAggregationAndDeck(unittest.TestCase):
    """Scenariusze wizualnej mapy (pydeck) i agregacji punktów."""

    def test_place_group_prefers_stable_gmap_id(self) -> None:
        df = pd.DataFrame(
            {
                "latitude": [50.1234567, 50.123456],
                "longitude": [19.9, 19.9],
                "gmap_id": ["place-42", "place-42"],
            }
        )
        keys = place_group_keys(df)
        self.assertTrue(keys.str.startswith("gmap:").all())
        self.assertEqual(keys.iloc[0], keys.iloc[1])

    def test_aggregate_merges_rows_by_place(self) -> None:
        df = pd.DataFrame(
            {
                "latitude": [50.0, 50.0000001],
                "longitude": [19.0, 19.0],
                "gmap_id": ["g1", "g1"],
                "name": ["Cafe", "Cafe"],
                "category": ["food", "food"],
                "cleanliness": ["positive", "positive"],
            }
        )
        agg = aggregate_points_by_place(df, "cleanliness")
        self.assertEqual(len(agg), 1)
        row = agg.iloc[0]
        self.assertEqual(int(row["n_reviews"]), 2)
        self.assertEqual(row["place_name"], "Cafe")
        self.assertEqual(row["_radius"], radius_for_review_count(2))

    def test_build_map_scatter_layer_and_tooltip(self) -> None:
        df = pd.DataFrame(
            {
                "latitude": [51.1],
                "longitude": [17.0],
                "name": ["Rynek"],
                "category": ["square"],
                "gmap_id": ["x1"],
                "heritage": ["neutral"],
            }
        )
        view = pdk.ViewState(latitude=51.1, longitude=17.0, zoom=11)
        deck, scatter_df = build_map(df, "heritage", view)

        style = str(deck.map_style).lower()
        self.assertTrue("dark" in style)
        self.assertEqual(len(deck.layers), 1)
        self.assertEqual(deck.layers[0].id, "absa-points")
        tooltip = deck._tooltip
        assert isinstance(tooltip, dict)
        html = tooltip["html"]
        self.assertIn("{place_name}", html)
        self.assertIn("Kliknij dla szczegółów", html)
        self.assertFalse(scatter_df.empty)
        self.assertEqual(scatter_df.iloc[0]["place_name"], "Rynek")

    def test_aggregate_positive_cluster_uses_green_channel(self) -> None:
        df = pd.DataFrame(
            {
                "latitude": [52.0, 52.0],
                "longitude": [21.0, 21.0],
                "gmap_id": ["p1", "p1"],
                "name": ["Park", "Park"],
                "costs": ["positive", "positive"],
            }
        )
        agg = aggregate_points_by_place(df, "costs")
        color = list(agg.iloc[0]["color"])
        self.assertGreaterEqual(color[3], 165)
        self.assertEqual(color[:3], COLOR_POSITIVE)

    def test_synthetic_aspect_produces_gray_marker(self) -> None:
        df = pd.DataFrame(
            {
                "latitude": [50.0],
                "longitude": [19.0],
                "name": ["Spot"],
                "gmap_id": ["s"],
                REPOSITORY_MAP_SYNTHETIC_ASPECT: ["x"],
            }
        )
        agg = aggregate_points_by_place(df, REPOSITORY_MAP_SYNTHETIC_ASPECT)
        self.assertEqual(list(agg.iloc[0]["color"][:3]), [158, 158, 158])


class TestPlaceDetailsLogic(unittest.TestCase):
    """Scenariusze danych używanych w „szczegółach po kliknięciu” na mapę."""

    def test_find_metadata_by_place_name(self) -> None:
        meta_df = pd.DataFrame(
            {
                "name": ["Muzeum", "Park"],
                "category": ["Museum||Art", "Park"],
                "extra": [1, 2],
            }
        )
        found = find_metadata_for_place(meta_df, place_name="Muzeum")
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found["category"], "Museum||Art")
        self.assertIsNone(find_metadata_for_place(meta_df, place_name="Brak"))

    def test_reviews_and_sentiment_counts_for_one_place_key(self) -> None:
        df = pd.DataFrame(
            {
                "latitude": [1.0, 1.0, 2.0],
                "longitude": [1.0, 1.0, 2.0],
                "gmap_id": ["a", "a", "b"],
                "text": ["t1", "t2", "t3"],
                "safety": ["positive", "negative", "neutral"],
            }
        )
        keys = compute_place_keys(df)
        target = keys.iloc[0]
        revs = reviews_for_place(df, keys, target)
        self.assertEqual(len(revs), 2)
        counts = sentiment_counts(revs, "safety")
        self.assertEqual(counts[SentimentLabel.POSITIVE.value], 1)
        self.assertEqual(counts[SentimentLabel.NEGATIVE.value], 1)

    def test_filter_reviews_by_sentiment(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["a", "b", "c"],
                "cleanliness": ["positive", "positive", "negative"],
            }
        )
        only_pos = filter_reviews(df, "cleanliness", "positive")
        self.assertEqual(len(only_pos), 2)
        only_neg = filter_reviews(df, "cleanliness", SentimentLabel.NEGATIVE.value)
        self.assertEqual(len(only_neg), 1)


if __name__ == "__main__":
    unittest.main()
