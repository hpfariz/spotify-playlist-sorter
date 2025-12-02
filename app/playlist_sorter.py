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
    BPM_MAX_DIFFERENCE,
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

    def _scrape_chosic(self) -> pd.DataFrame | None:
        """Scrape track data from chosic.com for the playlist."""
        from playwright.sync_api import sync_playwright

        playlist_url = f"https://open.spotify.com/playlist/{self.playlist_id}"
        chosic_url = "https://www.chosic.com/spotify-playlist-analyzer/"
        logger.info("Attempting to scrape data from Chosic: %s for playlist %s", chosic_url, playlist_url)

        content = ""
        try:
            with sync_playwright() as p:
                logger.info("Launching Browser for Chosic...")
                browser = p.chromium.launch(
                    headless=True,  # Can be headless for server environments
                    args=["--disable-blink-features=AutomationControlled", "--start-maximized"],
                )
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                )
                page = context.new_page()

                logger.info("Navigating to Chosic...")
                page.goto(chosic_url, timeout=60000, wait_until="domcontentloaded")

                # Wait for input
                page.wait_for_selector("#search-word", state="visible", timeout=30000)

                # Type playlist URL
                logger.info("Inputting playlist URL...")
                page.fill("#search-word", playlist_url)

                # Click Analyze
                logger.info("Clicking Analyze...")
                page.click("#analyze")

                # Wait for the results table
                try:
                    logger.info("Waiting for results table...")
                    page.wait_for_selector("#tracks-table tbody tr", state="visible", timeout=60000)
                    # Give it a small buffer to ensure rendering completes if it's dynamic
                    time.sleep(2)
                except Exception:
                    logger.warning("Timeout waiting for results table. Attempting to grab content anyway.")

                content = page.content()
                browser.close()

        except Exception as e:
            logger.exception("Failed to scrape Chosic with Playwright")
            return None

        # Parse content
        soup = BeautifulSoup(content, "html.parser")
        table = soup.find("table", {"id": "tracks-table"})

        if not table:
            logger.error("Could not find #tracks-table in Chosic response.")
            return None

        table_body = table.find("tbody")
        if not table_body:
            logger.error("Could not find tbody in Chosic results.")
            return None

        rows = table_body.find_all("tr")
        logger.info("Found %s track rows in Chosic table.", len(rows))

        tracks = []
        for row in rows:
            try:
                cols = row.find_all("td")
                # Based on HTML provided:
                # 0: #, 1: Song, 2: Artist, 3: Popularity, 4: BPM, ..., 11: Energy, ..., 21: Spotify Track Id, ..., 23: Camelot

                if len(cols) < 24:
                    continue

                spotify_id = cols[21].text.strip()

                # Track name is inside td > div > span, or sometimes just text
                track_name_span = cols[1].find("span", class_="td-name-text")
                if track_name_span:
                    track_name = track_name_span.text.strip()
                else:
                    track_name = cols[1].text.strip()

                # Remove checkbox text if caught
                if " " in track_name and len(track_name) > 1:
                     track_name = track_name.replace("Check to delete", "").strip()

                artist = cols[2].text.strip()
                bpm = cols[4].text.strip()
                energy = cols[11].text.strip()  # 0-100
                camelot = cols[23].text.strip()
                popularity = cols[3].text.strip()

                if not all([spotify_id, bpm, energy, camelot]):
                    continue

                tracks.append(
                    {
                        "id": spotify_id,
                        "Track": track_name,
                        "Artist": artist,
                        "Camelot": camelot,
                        "BPM": bpm,
                        "Energy": energy,  # Will normalize later
                        "Popularity": popularity,
                    }
                )

            except Exception as e:
                logger.warning("Error parsing Chosic row: %s", e)
                continue

        if not tracks:
            return None

        track_df = pd.DataFrame(tracks)

        # Data Cleaning
        try:
            track_df["BPM"] = pd.to_numeric(track_df["BPM"], errors="coerce")
            # Normalize Energy (Chosic is 0-100, we need 0-1)
            track_df["Energy"] = pd.to_numeric(track_df["Energy"], errors="coerce") / 100.0
            track_df["Popularity"] = pd.to_numeric(track_df["Popularity"], errors="coerce")

            # Normalize Camelot (ensure uppercase)
            track_df["Camelot"] = track_df["Camelot"].str.upper()

        except Exception:
            logger.exception("Error processing Chosic dataframe")

        return track_df

    def _scrape_songdata_io(self) -> pd.DataFrame | None:
        """Scrape track data from songdata.io for the playlist."""
        from playwright.sync_api import sync_playwright
        
        url = f"https://songdata.io/playlist/{self.playlist_id}"
        logger.info("Attempting to scrape data from: %s", url)

        content = ""
        try:
            with sync_playwright() as p:
                logger.info("Launching System Chrome...")
                browser = p.chromium.launch(
                    headless=False,
                    channel="chrome", 
                    args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
                )
                
                context = browser.new_context(
                    viewport=None,
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
        
        table = cast("Optional[Tag]", soup.find("table", {"id": "table_chart"}))
        if not table:
            logger.error("Could not find the track table (id='table_chart') on the page.")
            # Fallback
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

                # Popularity
                all_data_tags = row_tag.find_all("td", {"class": "table_data"})
                popularity = None
                if len(all_data_tags) > DATA_ROW_THRESHOLD:
                    # Use a pattern matching approach
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

                # Spotify ID
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
            # Convert relevant columns to numeric
            track_df["BPM"] = pd.to_numeric(track_df["BPM"], errors="coerce")
            # Energy
            raw_energy = pd.to_numeric(track_df["Energy"], errors="coerce")
            if raw_energy.max() > 1.0:
                logger.warning("Detected Energy values > 1. Assuming 1-10 scale and normalizing to 0-1.")
                track_df["Energy"] = raw_energy / 10.0
            else:
                track_df["Energy"] = raw_energy

            track_df["Popularity"] = pd.to_numeric(track_df["Popularity"], errors="coerce")

            # Validate Camelot format
            track_df["Camelot"] = track_df["Camelot"].str.upper()
            valid_camelot_mask = track_df["Camelot"].str.match(r"^[1-9]A$|^1[0-2]A$|^[1-9]B$|^1[0-2]B$", na=False)
            track_df.loc[~valid_camelot_mask, "Camelot"] = np.nan

        except Exception:
            logger.exception("Error during data type conversion")

        logger.info("Successfully scraped and parsed %s tracks.", len(track_df))
        return track_df
    
    def load_playlist(self, source: str = "songdata.io") -> pd.DataFrame | None:
        """Load playlist name from Spotify and track data by scraping."""
        logger.info("Loading playlist metadata for: %s using %s", self.playlist_id, source)
        try:
            # Get playlist name from Spotify
            playlist_info = self.sp.playlist(self.playlist_id, fields="name")
            self.playlist_name = playlist_info["name"]
            logger.info("Playlist Name (from Spotify): '%s'", self.playlist_name)
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.warning("Failed to get playlist name from Spotify: %s. Will proceed without it.", e)
            self.playlist_name = f"Playlist {self.playlist_id}"

        # Scrape track data
        scraped_data = None
        if source == "chosic.com":
            scraped_data = self._scrape_chosic()
        else:
            # Default to songdata.io
            scraped_data = self._scrape_songdata_io()

        if scraped_data is None or scraped_data.empty:
            logger.error(f"Failed to scrape data from {source}. Cannot proceed.")
            self.tracks_data = pd.DataFrame()
            self.original_track_order = []
            return None
        self.tracks_data = scraped_data
        # Store original order based on scraped table
        self.original_track_order = self.tracks_data["id"].tolist()
        logger.info(
            "Using original track order based on %s table (%s tracks).", source, len(self.original_track_order)
        )

        # Ensure required columns exist
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

    def calculate_transition_score(self, track1: pd.Series | dict[str, Any], track2: pd.Series | dict[str, Any]) -> float:
        """Calculate a transition score between two tracks based on key, BPM, and energy."""
        # Get track data
        key1 = track1.get("Camelot")
        key2 = track2.get("Camelot")
        bpm1 = track1.get("BPM")
        bpm2 = track2.get("BPM")
        energy1 = track1.get("Energy")
        energy2 = track2.get("Energy")

        # --- Key Compatibility Score ---
        key_score = 0.0
        
        if pd.isna(key1) or pd.isna(key2):
            key_score = 0.5 
        else:
            key1_str = str(key1)
            key2_str = str(key2)
            key_compatible = False
            if key1_str in self.camelot_map:
                key_compatible = key2_str in self.camelot_map[key1_str]

            if key1 == key2:
                key_score = 1.0
            elif key_compatible:
                key_score = 0.8
            else:
                key_score = 0.0

        # --- BPM Score ---
        bpm_score = 0.0
        bpm_diff = 0.0
        
        # New Feature: Double/Half Time Detection
        # This fixes issues where 95 BPM -> 190 BPM scores 0.00 despite being perfectly mixable
        if not pd.isna(bpm1) and not pd.isna(bpm2):
            try:
                b1 = float(bpm1)
                b2 = float(bpm2)
                if b1 > 0 and b2 > 0:
                    # Calculate differences for standard, half-time, and double-time mixing
                    diff_std = abs(b1 - b2)
                    diff_double = abs((b1 * 2) - b2)
                    diff_half = abs((b1 / 2) - b2)
                    
                    # Use the best matching BPM interpretation
                    bpm_diff = min(diff_std, diff_double, diff_half)
                    
                    if bpm_diff <= BPM_GOOD_THRESHOLD:  # <= 5
                        bpm_score = 1.0
                    elif bpm_diff <= BPM_MEDIUM_THRESHOLD:  # <= 10
                        bpm_score = 0.8
                    elif bpm_diff <= 20:  # Allow mixes up to 20 BPM diff but penalize
                        ratio = (bpm_diff - BPM_MEDIUM_THRESHOLD) / (20 - BPM_MEDIUM_THRESHOLD)
                        bpm_score = 0.8 * (1.0 - ratio)
                    else:
                        bpm_score = 0.0
            except (ValueError, TypeError):
                logger.debug("Cannot convert BPM to float for scoring: %s, %s", bpm1, bpm2)

        # --- Energy Flow Score ---
        energy_score = 0.0
        if not pd.isna(energy1) and not pd.isna(energy2):
            try:
                energy1_val = float(energy1)
                energy2_val = float(energy2)
                energy_diff = energy2_val - energy1_val
                
                if 0 <= energy_diff <= ENERGY_SMALL_INCREASE_MAX:
                    energy_score = 1.0
                elif (
                    ENERGY_SMALL_DECREASE_MIN <= energy_diff < 0
                    or ENERGY_SMALL_INCREASE_MAX < energy_diff <= ENERGY_MODERATE_INCREASE_MAX
                ):
                    energy_score = 0.7
                else:
                    energy_score = max(0, 1 - abs(energy_diff) / ENERGY_SCALING_FACTOR)
            except (ValueError, TypeError):
                logger.debug("Cannot convert Energy to float for scoring: %s, %s", energy1, energy2)

        # --- BPM RESCUE: High Score Override ---
        # If the BPM is perfect (score 1.0), we consider this a high-quality mix 
        # regardless of Key. We start with a high base score and apply light penalties.
        if bpm_score == 1.0:
            # Base score for a perfect beatmatch, ignoring key initially
            total_score = 0.85
            
            # Add small bonus for energy match
            total_score += (energy_score * 0.15)
            
            # Apply moderate penalty if keys are incompatible.
            # 0.8 multiplier -> 0.85 * 0.8 = 0.68 (Good "Energy Mix" score)
            if key_score == 0.0:
                total_score *= 0.8
            
            # No further penalties applied here. This ensures perfect BPM matches stay high.
            return total_score

        # --- Standard Weighted Score (for non-perfect BPM) ---
        total_score = (key_score * 0.4) + (bpm_score * 0.4) + (energy_score * 0.2)

        # --- Standard Penalties ---
        # 1. BPM Deal-Breaker: If BPM diff is huge (>20), punish severely.
        if bpm_score == 0.0:
            # Check for "Harmonic Bridge": Compatible Key + Genre Switch (Large BPM jump)
            # If keys are compatible (>=0.8) and it's a big jump, give it a "Bridge Score" (~0.5-0.6)
            if key_score >= 0.8 and bpm_diff <= 40:
                total_score = 0.6  # Fixed score for a harmonic bridge
            else:
                total_score *= 0.01 # Unmixable
                
        # 2. BPM Rough Patch: If BPM diff is 12-20, penalize significantly.
        elif bpm_diff > BPM_MAX_DIFFERENCE: 
            total_score *= 0.5
            
        # 3. Key Mismatch: If keys clash AND BPM is not perfect, apply penalty.
        if key_score == 0.0 and not (pd.isna(key1) or pd.isna(key2)):
            total_score *= 0.5

        return total_score

    def sort_playlist(self, start_track_id: str, method: str = "greedy") -> list[str]:
        """Sort the playlist using transition scores, starting from anchor.
        
        Args:
            start_track_id: The ID of the track to start the playlist with.
            method: The sorting algorithm to use ('greedy' or 'beam').
        """
        if self.tracks_data is None or self.tracks_data.empty:
            logger.error("Track data is not loaded or is empty. Cannot sort.")
            return []

        if start_track_id not in self.tracks_data["id"].to_numpy():
            logger.error("Start track ID '%s' not found in the loaded & filtered tracks.", start_track_id)
            if self.original_track_order and start_track_id in self.original_track_order:
                logger.warning(
                    "Anchor track was present initially but filtered out due to missing data. Cannot use as anchor."
                )
            return []
            
        logger.info("Starting sort with anchor track ID: %s using %s method", start_track_id, method)

        if method == "beam":
             sorted_ids = self._sort_playlist_beam(start_track_id)
        else:
             sorted_ids = self._sort_playlist_greedy(start_track_id)

        # Handle tracks that were filtered out or not placed
        original_ids_set = set(self.original_track_order) if self.original_track_order else set()
        missing_from_sort = original_ids_set - set(sorted_ids)

        if missing_from_sort and self.original_track_order:
            logger.warning("Sort finished, but %s tracks that had data were not placed.", len(missing_from_sort))
            missing_tracks_ordered = [tid for tid in self.original_track_order if tid in missing_from_sort]
            logger.info("Appending %s tracks that were not placed during sorting.", len(missing_tracks_ordered))
            sorted_ids.extend(missing_tracks_ordered)
        
        logger.info("Playlist sorting complete. Final track count: %s", len(sorted_ids))
        return sorted_ids

    def _sort_playlist_greedy(self, start_track_id: str) -> list[str]:
        """Sort using a greedy algorithm: always pick best next track."""
        sortable_tracks = self.tracks_data.copy()
        
        current_id = start_track_id
        sorted_ids = [current_id]
        remaining_ids = set(sortable_tracks["id"].tolist())
        remaining_ids.remove(current_id)

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
                break

            # Get next track with highest score
            next_track_idx = scores.idxmax()
            next_track = sortable_tracks.loc[next_track_idx]
            next_track_id = str(next_track["id"])

            if next_track_id not in remaining_ids:
                break

            # Add to sorted list and update for next iteration
            sorted_ids.append(next_track_id)
            remaining_ids.remove(next_track_id)
            current_id = next_track_id
            logger.debug("Added: %s (Score: %.2f)", next_track.get("Track", current_id), scores.loc[next_track_idx])
            
        return sorted_ids

    def _sort_playlist_beam(self, start_track_id: str, beam_width: int = 100) -> list[str]:
        """Sort playlist using Beam Search algorithm for higher quality transitions."""
        track_lookup = self.tracks_data.set_index('id').to_dict('index')
        for tid, data in track_lookup.items():
            data['id'] = tid
            
        all_ids = set(track_lookup.keys())
        num_tracks = len(all_ids)
        
        # Beam state: (cumulative_score, [path_of_ids], {set_of_visited_ids})
        current_beam = [(0.0, [start_track_id], {start_track_id})]
        
        for step in range(num_tracks - 1):
            all_candidates = []
            
            for score, path, visited in current_beam:
                last_track_id = path[-1]
                last_track_data = track_lookup[last_track_id]
                
                potential_next_tracks = []
                
                for next_id in all_ids:
                    if next_id in visited:
                        continue
                        
                    next_track_data = track_lookup[next_id]
                    trans_score = self.calculate_transition_score(last_track_data, next_track_data)
                    potential_next_tracks.append((trans_score, next_id))
                
                if not potential_next_tracks:
                    continue
                    
                # Sort by transition score descending
                potential_next_tracks.sort(key=lambda x: x[0], reverse=True)
                
                # Keep top 5 best next steps for this specific path
                top_next_steps = potential_next_tracks[:5]
                
                for step_score, step_id in top_next_steps:
                    new_total_score = score + step_score
                    new_path = path + [step_id]
                    new_visited = visited | {step_id}
                    all_candidates.append((new_total_score, new_path, new_visited))
            
            if not all_candidates:
                break
                
            # Prune the beam
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            current_beam = all_candidates[:beam_width]
            
        best_path = current_beam[0][1]
        return best_path

    def compare_playlists(self, sorted_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compare original (scraped order) and sorted playlist."""
        if self.tracks_data is None or self.tracks_data.empty or not self.original_track_order:
            return pd.DataFrame(), pd.DataFrame()

        valid_original_ids = [tid for tid in self.original_track_order if tid in self.tracks_data["id"].to_numpy()]
        valid_sorted_ids = [tid for tid in sorted_ids if tid in self.tracks_data["id"].to_numpy()]

        if not valid_original_ids or not valid_sorted_ids:
            return pd.DataFrame(), pd.DataFrame()

        original_df = pd.DataFrame(index=range(len(valid_original_ids)))
        sorted_df = pd.DataFrame(index=range(len(valid_sorted_ids)))

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

            if pd.isna(key1) or pd.isna(key2):
                transition["message"] = "Missing key information, cannot analyze compatibility"
                transitions.append(transition)
                continue

            if pd.isna(bpm1) or pd.isna(bpm2):
                transition["message"] = "Missing BPM information, cannot analyze tempo change"
                transitions.append(transition)
                continue

            key_compatible = False
            perfect_key_match = key1 == key2

            if not pd.isna(key1) and str(key1) in self.camelot_map:
                key_compatible = str(key2) in self.camelot_map[str(key1)]
            
            if perfect_key_match:
                key_compatible = True

            bpm_diff = None
            if not pd.isna(bpm1) and not pd.isna(bpm2):
                try:
                    bpm_diff = abs(float(bpm1) - float(bpm2))
                except (ValueError, TypeError):
                    pass

            energy_diff = None
            if not pd.isna(energy1) and not pd.isna(energy2):
                try:
                    energy_diff = float(energy2) - float(energy1)
                except (ValueError, TypeError):
                    pass

            transition["key_compatible"] = key_compatible
            transition["perfect_key_match"] = perfect_key_match
            transition["bpm_diff"] = bpm_diff
            transition["energy_diff"] = energy_diff
            transition["score"] = self.calculate_transition_score(track1_data, track2_data)

            transitions.append(transition)

        return transitions

    def _get_track_uris(self, track_ids: list[str]) -> dict[str, str]:
        """Get Spotify URIs for track IDs, using the API to ensure accuracy."""
        uri_map = {}

        for i in range(0, len(track_ids), API_BATCH_SIZE):
            batch_ids = track_ids[i : i + API_BATCH_SIZE]
            try:
                tracks_details = self.sp.tracks(batch_ids)["tracks"]
                for track in tracks_details:
                    if track and "id" in track and "uri" in track:
                        uri_map[track["id"]] = track["uri"]
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
            for i in range(100, len(track_uris), 100):
                batch = track_uris[i : i + 100]
                self.sp.playlist_add_items(self.playlist_id, batch)
                time.sleep(1)

            return True, f"Successfully updated playlist '{self.playlist_name}' with {len(track_uris)} tracks"

        except Exception as e:
            error_msg = str(e)
            logger.exception("Failed to update Spotify playlist: %s", error_msg)
            return False, f"Failed to update playlist: {error_msg}"

    def create_sorted_playlist(self, sorted_ids: list[str], new_name: str) -> tuple[bool, str]:
        """Create a new playlist with the sorted tracks."""
        if not sorted_ids:
            logger.error("No sorted track IDs provided to create playlist.")
            return False, "No sorted track IDs provided"
        if self.tracks_data is None or self.tracks_data.empty:
            logger.error("No track data available to map IDs to URIs.")
            return False, "No track data available"

        logger.info("Fetching URIs for %s sorted tracks...", len(sorted_ids))
        uri_map = self._get_track_uris(sorted_ids)
        track_uris = [uri_map[track_id] for track_id in sorted_ids if track_id in uri_map]

        if not track_uris:
            logger.error("No valid track URIs could be fetched. Cannot create playlist.")
            return False, "No valid track URIs could be fetched"

        try:
            # Get current user ID
            user_id = self.sp.current_user()["id"]
            logger.info("Creating new playlist '%s' for user %s", new_name, user_id)
            
            # Create the new playlist
            new_playlist = self.sp.user_playlist_create(
                user=user_id,
                name=new_name,
                public=False, # Default to private
                description=f"Sorted version of {self.playlist_name} generated by Spotify Playlist Sorter"
            )
            new_playlist_id = new_playlist["id"]
            
            # Add tracks in batches (Spotify API limit is 100 tracks per request)
            for i in range(0, len(track_uris), 100):
                batch = track_uris[i : i + 100]
                self.sp.playlist_add_items(new_playlist_id, batch)
                logger.info("Added batch of %s tracks to new playlist (starting index %s).", len(batch), i)
                time.sleep(0.5) # Slight delay to respect API limits

            logger.info("Successfully created playlist '%s'!", new_name)
            return True, f"Successfully created new playlist '{new_name}' with {len(track_uris)} tracks"

        except Exception as e:
            error_msg = str(e)
            logger.exception("Failed to create new playlist: %s", error_msg)
            return False, f"Failed to create playlist: {error_msg}"