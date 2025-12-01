"""Spotify playlist sorting module for optimal track transitions."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast

import cloudscraper
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

# Import constants from local constants module
from constants import (
    API_BATCH_SIZE,
    BPM_GOOD_THRESHOLD,
    BPM_MEDIUM_THRESHOLD,
    CAMELOT_MAX_NUMBER,
    CAMELOT_MIN_NUMBER,
    DATA_ROW_THRESHOLD,
    ENERGY_MODERATE_INCREASE_MAX,
    ENERGY_SCALING_FACTOR,
    ENERGY_SMALL_DECREASE_MIN,
    ENERGY_SMALL_INCREASE_MAX,
    HTTP_TIMEOUT,
)

if TYPE_CHECKING:
    import spotipy

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for numpy array or pd.Series operations
T = TypeVar("T")

# Constants for transition analysis
MIN_TRANSITION_COUNT = 2


class SpotifyPlaylistSorter:
    """Class for sorting Spotify playlists based on musical compatibility.

    This class analyzes track features (key, BPM, energy) and creates an optimized
    playlist order that provides smooth transitions between tracks.
    """

    def __init__(self, playlist_id: str, sp: spotipy.Spotify) -> None:
        """Initialize the playlist sorter.

        Args:
            playlist_id: Spotify playlist ID to sort
            sp: Authenticated spotipy.Spotify client
        """
        self.playlist_id = playlist_id
        self.sp = sp
        self.tracks_data: pd.DataFrame | None = None
        self.camelot_map = self._build_camelot_map()
        self.playlist_name: str | None = None
        self.original_track_order: list[str] | None = None

    def _build_camelot_map(self) -> dict[str, list[str]]:
        """Build a map of compatible Camelot keys."""
        camelot_map = {}
        numbers = range(CAMELOT_MIN_NUMBER, CAMELOT_MAX_NUMBER + 1)
        letters = ["A", "B"]

        for num in numbers:
            for letter in letters:
                key = f"{num}{letter}"
                neighbors = []

                # Same number, different letter (switching between minor/major)
                other_letter = "B" if letter == "A" else "A"
                neighbors.append(f"{num}{other_letter}")

                # Same letter, adjacent numbers
                prev_num = CAMELOT_MAX_NUMBER if num == CAMELOT_MIN_NUMBER else num - 1
                next_num = CAMELOT_MIN_NUMBER if num == CAMELOT_MAX_NUMBER else num + 1
                neighbors.extend([f"{prev_num}{letter}", f"{next_num}{letter}"])

                camelot_map[key] = neighbors

        return camelot_map

    def _scrape_songdata_io(self) -> pd.DataFrame | None:
        """Scrape track data from songdata.io for the playlist."""
        from playwright.sync_api import sync_playwright
        
        url = f"https://songdata.io/playlist/{self.playlist_id}"
        logger.info("Attempting to scrape data from: %s", url)

        content = ""
        try:
            with sync_playwright() as p:
                logger.info("Launching System Chrome...")
                # CHANGE 1: Use channel="chrome" to use your real installed browser
                # CHANGE 2: Add argument to start maximized
                browser = p.chromium.launch(
                    headless=False,
                    channel="chrome",  # Uses actual Google Chrome (or try "msedge" if you don't have Chrome)
                    args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
                )
                
                # CHANGE 3: Remove manual User-Agent and Viewport to avoid fingerprint mismatches
                context = browser.new_context(
                    viewport=None,  # Let the browser decide the size
                    no_viewport=True
                )
                
                page = context.new_page()
                
                # Stealth: Hide the webdriver property
                page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """)
                
                logger.info("Navigating to page...")
                page.goto(url, timeout=90000, wait_until="domcontentloaded")
                
                try:
                    logger.info("Waiting for data table...")
                    # Wait for the table to appear
                    page.wait_for_selector("#table_chart", state="visible", timeout=30000)
                except Exception:
                    logger.warning("Table not detected immediately. If you see a CAPTCHA, please solve it manually.")
                    # Wait a bit longer for manual intervention
                    time.sleep(10)
                
                content = page.content()
                browser.close()
                
        except Exception as e:
            logger.exception("Failed to scrape with Playwright")
            return None

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")
        
        # ... (Rest of the function remains exactly the same) ...
        table = cast("Optional[Tag]", soup.find("table", {"id": "table_chart"}))
        if not table:
            logger.error("Could not find the track table (id='table_chart') on the page.")
            # ... continue with the rest of your parsing logic ...
            table = cast("Optional[Tag]", soup.find("table", {"class": "table"}))
            if not table:
                logger.error("Could not find the track table by class either.")
                return None
            logger.warning("Found table using class='table' as fallback.")

        table_body = cast("Optional[Tag]", table.find("tbody", {"id": "table_body"}))
        if not table_body:
            # Fallback if tbody doesn't have the specific ID
            table_body = cast("Optional[Tag]", table.find("tbody"))
            if not table_body:
                logger.error("Could not find the table body (tbody) within the table.")
                return None
            logger.warning("Found tbody without specific ID.")

        tracks = []
        rows = table_body.find_all("tr", {"class": "table_object"})

        if not rows:
            logger.error("Found table body, but no rows with class='table_object'.")
            return None

        logger.info("Found %s potential track rows in the table.", len(rows))

        for row in rows:
            row_tag = cast("Tag", row)
            try:
                # Extract data based on common class names
                track_name_tag = cast("Optional[Tag]", row_tag.find("td", {"class": "table_name"}))
                track_name = None
                if track_name_tag and track_name_tag.find("a"):
                    name_a_tag = cast("Tag", track_name_tag.find("a"))
                    track_name = name_a_tag.text.strip()

                artist_tag = cast("Optional[Tag]", row_tag.find("td", {"class": "table_artist"}))
                artist = artist_tag.text.strip() if artist_tag else None

                key_tag = cast("Optional[Tag]", row_tag.find("td", {"class": "table_key"}))
                key = key_tag.text.strip() if key_tag else None

                camelot_tag = cast("Optional[Tag]", row_tag.find("td", {"class": "table_camelot"}))
                camelot = camelot_tag.text.strip() if camelot_tag else None

                bpm_tag = cast("Optional[Tag]", row_tag.find("td", {"class": "table_bpm"}))
                bpm = bpm_tag.text.strip() if bpm_tag else None

                energy_tag = cast("Optional[Tag]", row_tag.find("td", {"class": "table_energy"}))
                energy = energy_tag.text.strip() if energy_tag else None

                # Popularity is often in 'table_data' but might need specific identification
                all_data_tags = row_tag.find_all("td", {"class": "table_data"})
                popularity = None
                if len(all_data_tags) > DATA_ROW_THRESHOLD:
                    # Use a pattern matching approach instead of a lambda function
                    date_pattern = re.compile(r"[-/]")
                    release_date_tag = cast(
                        "Optional[Tag]",
                        row_tag.find(
                            "td",
                            {"class": "table_data"},
                            string=date_pattern,
                        ),
                    )
                    if release_date_tag:
                        prev_sibling = cast(
                            "Optional[Tag]", release_date_tag.find_previous_sibling("td", {"class": "table_data"})
                        )
                        if prev_sibling:
                            popularity = prev_sibling.text.strip()

                # Spotify ID is usually in a data-src attribute
                spotify_link_cell = cast("Optional[Tag]", row_tag.find("td", {"id": "spotify_obj"}))
                spotify_id = None
                if spotify_link_cell and "data-src" in spotify_link_cell.attrs:
                    spotify_id = str(spotify_link_cell["data-src"])

                if not all([track_name, artist, camelot, bpm, energy, spotify_id]):
                    logger.warning(
                        "Skipping row due to missing essential data (Name, Artist, Camelot, BPM, Energy, ID): %s, %s",
                        track_name,
                        artist,
                    )
                    continue

                tracks.append(
                    {
                        "id": spotify_id,
                        "Track": track_name,
                        "Artist": artist,
                        "Key": key,
                        "Camelot": camelot,
                        "BPM": bpm,
                        "Energy": energy,
                        "Popularity": popularity,
                    }
                )
            except (AttributeError, IndexError, KeyError) as e:
                logger.warning("Error parsing a row: %s. Row content: %s...", e, str(row_tag)[:100])
                continue

        if not tracks:
            logger.error("No tracks successfully parsed from the table.")
            return None

        track_df = pd.DataFrame(tracks)

        # --- Data Cleaning and Type Conversion ---
        try:
            # Convert relevant columns to numeric, coercing errors to NaN
            track_df["BPM"] = pd.to_numeric(track_df["BPM"], errors="coerce")
            # Energy from songdata.io might be 1-10 scale or 0-1. Let's assume 0-1 for now.
            raw_energy = pd.to_numeric(track_df["Energy"], errors="coerce")
            if raw_energy.max() > 1.0:
                logger.warning("Detected Energy values > 1. Assuming 1-10 scale and normalizing to 0-1.")
                track_df["Energy"] = raw_energy / 10.0
            else:
                track_df["Energy"] = raw_energy

            track_df["Popularity"] = pd.to_numeric(track_df["Popularity"], errors="coerce")

            # Validate Camelot format (e.g., '1A', '12B')
            track_df["Camelot"] = track_df["Camelot"].str.upper()
            valid_camelot_mask = track_df["Camelot"].str.match(r"^[1-9]A$|^1[0-2]A$|^[1-9]B$|^1[0-2]B$", na=False)
            invalid_camelot = track_df[~valid_camelot_mask]["Camelot"].unique()
            if len(invalid_camelot) > 0:
                logger.warning("Found potentially invalid Camelot keys: %s. Replacing with NaN.", invalid_camelot)
                track_df.loc[~valid_camelot_mask, "Camelot"] = np.nan

        except Exception:
            logger.exception("Error during data type conversion")

        logger.info("Successfully scraped and parsed %s tracks.", len(track_df))
        return track_df
    
    def load_playlist(self) -> pd.DataFrame | None:
        """Load playlist name from Spotify and track data by scraping songdata.io."""
        logger.info("Loading playlist metadata for: %s", self.playlist_id)
        try:
            # Get playlist name from Spotify (more reliable than scraping)
            playlist_info = self.sp.playlist(self.playlist_id, fields="name")
            self.playlist_name = playlist_info["name"]
            logger.info("Playlist Name (from Spotify): '%s'", self.playlist_name)
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.warning("Failed to get playlist name from Spotify: %s. Will proceed without it.", e)
            self.playlist_name = f"Playlist {self.playlist_id}"

        # Scrape track data from songdata.io
        scraped_data = self._scrape_songdata_io()

        if scraped_data is None or scraped_data.empty:
            logger.error("Failed to scrape data from songdata.io. Cannot proceed.")
            self.tracks_data = pd.DataFrame()
            self.original_track_order = []
            return None
        self.tracks_data = scraped_data
        # Store original order based on scraped table
        self.original_track_order = self.tracks_data["id"].tolist()
        logger.info(
            "Using original track order based on songdata.io table (%s tracks).", len(self.original_track_order)
        )

        # Ensure required columns exist even if scraping missed some
        for col in ["id", "Camelot", "BPM", "Energy"]:
            if col not in self.tracks_data.columns:
                logger.error("Required column '%s' not found in scraped data.", col)
                self.tracks_data[col] = np.nan

        # Remove rows with missing essential data
        initial_count = len(self.tracks_data)
        self.tracks_data = self.tracks_data.dropna(subset=["id", "Camelot", "BPM", "Energy"])

        dropped_count = initial_count - len(self.tracks_data)
        if dropped_count > 0:
            logger.warning(
                "Dropped %s tracks due to missing essential data (ID, Camelot, BPM, or Energy) after scraping.",
                dropped_count,
            )

        if self.tracks_data.empty:
            logger.error("No tracks remaining after dropping those with missing essential data.")
            return None

        return self.tracks_data

    def calculate_transition_score(self, track1: pd.Series, track2: pd.Series) -> float:
        """Calculate a transition score between two tracks based on key, BPM, and energy."""
        # Get track data
        key1 = track1.get("Camelot")
        key2 = track2.get("Camelot")
        bpm1 = track1.get("BPM")
        bpm2 = track2.get("BPM")
        energy1 = track1.get("Energy")
        energy2 = track2.get("Energy")

        # Key compatibility score (highest weight)
        key_score = 0.0
        key_compatible = False
        key_multiplier = 1.0  # Full weight

        if pd.isna(key1) or pd.isna(key2):
            # If either key is missing, reduce weight but don't penalize completely
            key_multiplier = 0.5
            if not pd.isna(key1) and key1 not in self.camelot_map:
                logger.debug("Key %s not in camelot map for score calc.", key1)
        else:
            key1_str = str(key1)
            key2_str = str(key2)
            if key1_str in self.camelot_map:
                key_compatible = key2_str in self.camelot_map[key1_str]

        # Perfect match (same key) is slightly better than compatible keys
        if not pd.isna(key1) and key1 == key2:
            key_score = 1.0
        elif key_compatible:
            key_score = 0.9  # Very good but not perfect
        else:
            key_score = 0.1  # Poor key compatibility

        # BPM score - closer is better, within 5 BPM is great
        bpm_score = 0.0
        if not pd.isna(bpm1) and not pd.isna(bpm2):
            try:
                # Convert BPM values to float
                bpm1_val = float(bpm1)
                bpm2_val = float(bpm2)
                if bpm1_val > 0 and bpm2_val > 0:
                    bpm_diff = abs(bpm1_val - bpm2_val)
                    if bpm_diff <= BPM_GOOD_THRESHOLD:
                        bpm_score = 1.0
                    elif bpm_diff <= BPM_MEDIUM_THRESHOLD:
                        bpm_score = 0.7
                    else:
                        # Gradually scale down as BPM difference increases
                        bpm_score = max(0, 1 - (bpm_diff - BPM_MEDIUM_THRESHOLD) / 20)
            except (ValueError, TypeError):
                # Handle case where BPM cannot be converted to float
                logger.debug("Cannot convert BPM to float for scoring: %s, %s", bpm1, bpm2)

        # Energy flow score - slight increase is good, big jumps are bad
        energy_score = 0.0
        if not pd.isna(energy1) and not pd.isna(energy2):
            try:
                # Convert energy values to float
                energy1_val = float(energy1)
                energy2_val = float(energy2)
                energy_diff = energy2_val - energy1_val  # Positive for increasing energy
                # Small energy increases are ideal
                if 0 <= energy_diff <= ENERGY_SMALL_INCREASE_MAX:
                    energy_score = 1.0
                # Small decreases or moderate increases are ok
                elif (
                    ENERGY_SMALL_DECREASE_MIN <= energy_diff < 0
                    or ENERGY_SMALL_INCREASE_MAX < energy_diff <= ENERGY_MODERATE_INCREASE_MAX
                ):
                    energy_score = 0.7
                # Big jumps are scored lower
                else:
                    energy_score = max(0, 1 - abs(energy_diff) / ENERGY_SCALING_FACTOR)
            except (ValueError, TypeError):
                # Handle case where energy cannot be converted to float
                logger.debug("Cannot convert Energy to float for scoring: %s, %s", energy1, energy2)

        # Weight the scores and apply the key multiplier
        # Key is weighted most heavily since harmonic compatibility is paramount
        return key_score * 0.5 * key_multiplier + bpm_score * 0.3 + energy_score * 0.2

    def sort_playlist(self, start_track_id: str) -> list[str]:
        """Sort the playlist using transition scores, starting from anchor."""
        if self.tracks_data is None or self.tracks_data.empty:
            logger.error("Track data is not loaded or is empty. Cannot sort.")
            return []

        sortable_tracks = self.tracks_data.copy()

        if start_track_id not in sortable_tracks["id"].to_numpy():
            logger.error("Start track ID '%s' not found in the loaded & filtered tracks.", start_track_id)
            if self.original_track_order and start_track_id in self.original_track_order:
                logger.warning(
                    "Anchor track was present initially but filtered out due to missing data. Cannot use as anchor."
                )
            return []

        logger.info("Starting sort with anchor track ID: %s", start_track_id)
        current_id = start_track_id
        sorted_ids = [current_id]
        remaining_ids = set(sortable_tracks["id"].tolist())
        remaining_ids.remove(current_id)
        initial_sortable_ids = remaining_ids.copy()

        while remaining_ids:
            # Get current track data
            current_track_data = sortable_tracks[sortable_tracks["id"] == current_id]
            if current_track_data.empty:
                logger.error("Could not find data for current track ID: %s. Stopping sort.", current_id)
                break
            current_track = current_track_data.iloc[0]

            # Calculate scores for all remaining tracks
            scores = pd.Series(index=sortable_tracks.index, dtype=float)

            for idx, row in sortable_tracks.iterrows():
                if row["id"] in remaining_ids:
                    scores[idx] = self.calculate_transition_score(current_track, row)
                else:
                    scores[idx] = np.nan

            if scores.empty or scores.isna().all():
                logger.warning(
                    "Could not calculate valid scores from %s. Stopping sort.", current_track.get("Track", current_id)
                )
                break

            # Get next track with highest score
            next_track_idx = scores.idxmax()
            next_track = sortable_tracks.loc[next_track_idx]
            next_track_id = str(next_track["id"])

            if next_track_id not in remaining_ids:
                # This should not happen but protect against it
                break

            # Add to sorted list and update for next iteration
            sorted_ids.append(next_track_id)
            remaining_ids.remove(next_track_id)
            current_id = next_track_id
            logger.debug("Added: %s (Score: %.2f)", next_track.get("Track", current_id), scores.loc[next_track_idx])

        original_ids_set = set(self.original_track_order) if self.original_track_order else set()
        missing_from_sort = original_ids_set - set(sorted_ids)

        if missing_from_sort and self.original_track_order:
            logger.warning("Sort finished, but %s tracks that had data were not placed.", len(missing_from_sort))
            missing_tracks_ordered = [tid for tid in self.original_track_order if tid in missing_from_sort]
            logger.info("Appending %s tracks that were not placed during sorting.", len(missing_tracks_ordered))
            sorted_ids.extend(missing_tracks_ordered)
        elif len(sorted_ids) < len(initial_sortable_ids):
            logger.warning(
                "Sorting ended with %s tracks, but started with %s sortable tracks.",
                len(sorted_ids),
                len(initial_sortable_ids),
            )

        logger.info("Playlist sorting complete. Final track count: %s", len(sorted_ids))
        return sorted_ids

    def compare_playlists(self, sorted_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compare original (scraped order) and sorted playlist."""
        if self.tracks_data is None or self.tracks_data.empty or not self.original_track_order:
            logger.error("Cannot compare playlists: Data not loaded or original order missing.")
            return pd.DataFrame(), pd.DataFrame()

        # Check if we have valid IDs to work with
        valid_original_ids = [tid for tid in self.original_track_order if tid in self.tracks_data["id"].to_numpy()]
        valid_sorted_ids = [tid for tid in sorted_ids if tid in self.tracks_data["id"].to_numpy()]

        if not valid_original_ids or not valid_sorted_ids:
            logger.error("No valid track data found for comparison after filtering.")
            return pd.DataFrame(), pd.DataFrame()

        # Create DataFrames for the two orderings
        original_df = pd.DataFrame(index=range(len(valid_original_ids)))
        sorted_df = pd.DataFrame(index=range(len(valid_sorted_ids)))

        # Populate columns with track data
        for df, id_list in [(original_df, valid_original_ids), (sorted_df, valid_sorted_ids)]:
            id_col, track_col, artist_col, camelot_col, bpm_col, energy_col = [], [], [], [], [], []

            for track_id in id_list:
                track_data = self.tracks_data[self.tracks_data["id"] == track_id]
                if track_data.empty:
                    continue

                id_col.append(track_id)
                track_col.append(track_data["Track"].iloc[0])
                artist_col.append(track_data["Artist"].iloc[0])
                camelot_col.append(track_data["Camelot"].iloc[0])
                bpm_col.append(track_data["BPM"].iloc[0])
                energy_col.append(track_data["Energy"].iloc[0])

            df["id"] = id_col
            df["Track"] = track_col
            df["Artist"] = artist_col
            df["Camelot"] = camelot_col
            df["BPM"] = bpm_col
            df["Energy"] = energy_col

        return original_df, sorted_df

    def get_transition_analysis(self, sorted_ids: list[str]) -> list[dict[str, Any]]:
        """Generate analysis of the transitions in the sorted playlist."""
        if self.tracks_data is None or self.tracks_data.empty:
            logger.warning("No track data to analyze transitions.")
            return []

        valid_ids = [tid for tid in sorted_ids if tid in self.tracks_data["id"].to_numpy()]

        if len(valid_ids) < MIN_TRANSITION_COUNT:
            return []

        transitions = []

        for i in range(len(valid_ids) - 1):
            track1_id = valid_ids[i]
            track2_id = valid_ids[i + 1]

            track1_data = self.tracks_data[self.tracks_data["id"] == track1_id].iloc[0]
            track2_data = self.tracks_data[self.tracks_data["id"] == track2_id].iloc[0]

            key1 = track1_data.get("Camelot")
            key2 = track2_data.get("Camelot")
            bpm1 = track1_data.get("BPM")
            bpm2 = track2_data.get("BPM")
            energy1 = track1_data.get("Energy")
            energy2 = track2_data.get("Energy")

            transition = {
                "index": i + 1,
                "track1_id": track1_id,
                "track2_id": track2_id,
                "track1_name": track1_data.get("Track", "Unknown"),
                "track2_name": track2_data.get("Track", "Unknown"),
                "track1_artist": track1_data.get("Artist", "Unknown"),
                "track2_artist": track2_data.get("Artist", "Unknown"),
                "key1": key1,
                "key2": key2,
                "bpm1": bpm1,
                "bpm2": bpm2,
                "energy1": energy1,
                "energy2": energy2,
            }

            # Skip this transition if critical data is missing
            if pd.isna(key1) or pd.isna(key2):
                transition["message"] = "Missing key information, cannot analyze compatibility"
                transitions.append(transition)
                continue

            if pd.isna(bpm1) or pd.isna(bpm2):
                transition["message"] = "Missing BPM information, cannot analyze tempo change"
                transitions.append(transition)
                continue

            # Check key compatibility
            key_compatible = False
            perfect_key_match = key1 == key2

            if not pd.isna(key1) and str(key1) in self.camelot_map:
                key_compatible = str(key2) in self.camelot_map[str(key1)]

            # Calculate BPM difference
            bpm_diff = None
            if not pd.isna(bpm1) and not pd.isna(bpm2):
                try:
                    bpm_diff = abs(float(bpm1) - float(bpm2))
                except (ValueError, TypeError):
                    logger.debug("Cannot convert BPM to float for analysis: %s, %s", bpm1, bpm2)

            # Calculate energy difference
            energy_diff = None
            if not pd.isna(energy1) and not pd.isna(energy2):
                try:
                    energy_diff = float(energy2) - float(energy1)
                except (ValueError, TypeError):
                    logger.debug("Cannot convert Energy to float for analysis: %s, %s", energy1, energy2)

            # Add compatibility details
            transition["key_compatible"] = key_compatible
            transition["perfect_key_match"] = perfect_key_match
            transition["bpm_diff"] = bpm_diff
            transition["energy_diff"] = energy_diff

            # Calculate overall transition score
            transition["score"] = self.calculate_transition_score(track1_data, track2_data)

            transitions.append(transition)

        return transitions

    def _get_track_uris(self, track_ids: list[str]) -> dict[str, str]:
        """Get Spotify URIs for track IDs, using the API to ensure accuracy."""
        uri_map = {}

        # Process in batches of 50 to avoid hitting API rate limits
        for i in range(0, len(track_ids), API_BATCH_SIZE):
            batch_ids = track_ids[i : i + API_BATCH_SIZE]
            try:
                # Fetch full track details to ensure we have correct URIs
                tracks_details = self.sp.tracks(batch_ids)["tracks"]

                for track in tracks_details:
                    if track and "id" in track and "uri" in track:
                        uri_map[track["id"]] = track["uri"]
                    elif track and track["id"]:
                        logger.warning("Could not find URI for track ID: %s", track["id"])
            except Exception:
                logger.exception("Failed to fetch track details batch (starting index %s)", i)
            time.sleep(0.5)

        return uri_map

    def update_spotify_playlist(self, sorted_ids: list[str]) -> tuple[bool, str]:
        """Update the Spotify playlist with the new track order."""
        if not sorted_ids:
            logger.error("No sorted track IDs provided to update playlist.")
            return False, "No sorted track IDs provided"
        if self.tracks_data is None or self.tracks_data.empty:
            logger.error("No track data available to map IDs to URIs.")
            return False, "No track data available"

        logger.info("Fetching URIs for %s sorted tracks...", len(sorted_ids))
        uri_map = self._get_track_uris(sorted_ids)

        track_uris = [uri_map[track_id] for track_id in sorted_ids if track_id in uri_map]

        if not track_uris:
            logger.error("No valid track URIs could be fetched for the sorted IDs. Cannot update playlist.")
            return False, "No valid track URIs could be fetched"

        if len(track_uris) != len(sorted_ids):
            logger.warning(
                "Could only find URIs for %s out of %s tracks. Playlist will be updated with available tracks.",
                len(track_uris),
                len(sorted_ids),
            )

        logger.info("Updating Spotify playlist '%s' with %s tracks.", self.playlist_name, len(track_uris))

        try:
            self.sp.playlist_replace_items(self.playlist_id, track_uris[:100])
            logger.info("Replaced/set first %s tracks.", min(len(track_uris), 100))

            for i in range(100, len(track_uris), 100):
                batch = track_uris[i : i + 100]
                self.sp.playlist_add_items(self.playlist_id, batch)
                logger.info("Added batch of %s tracks (starting index %s).", len(batch), i)
                time.sleep(1)

            logger.info("Successfully updated playlist '%s' order on Spotify!", self.playlist_name)
            return True, f"Successfully updated playlist '{self.playlist_name}' with {len(track_uris)} tracks"

        except Exception as e:
            error_msg = str(e)
            logger.exception("Failed to update Spotify playlist: %s", error_msg)
            logger.exception("Check API permissions (scope), rate limits, and playlist ownership.")
            return False, f"Failed to update playlist: {error_msg}"
