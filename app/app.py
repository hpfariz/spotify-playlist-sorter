"""Streamlit web application for Spotify playlist sorting."""

from __future__ import annotations

import asyncio
import sys

# Fix for Playwright "NotImplementedError" on Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import logging
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from playlist_sorter import SpotifyPlaylistSorter
from spotify_auth import (
    get_all_playlists,
    get_auth_url,
    get_redirect_uri,
    get_spotify_client,
)
from spotify_auth import (
    load_credentials as load_spotify_credentials,
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants
# Path for saving credentials
CREDENTIALS_FILE = Path("./.spotify_credentials")

# Constants for score thresholds
SCORE_HIGH_THRESHOLD = 0.7
SCORE_MEDIUM_THRESHOLD = 0.4
BPM_GOOD_THRESHOLD = 5
BPM_MEDIUM_THRESHOLD = 10


# Function to save credentials to file
def save_credentials(client_id: str, client_secret: str) -> bool:
    """Save credentials to a local file."""
    try:
        # Simple encoding - not fully secure but better than plaintext
        # In a production app, you'd use a proper secure storage
        with CREDENTIALS_FILE.open("w") as f:
            f.write(f"{client_id}\n{client_secret}")
        logger.info("Credentials saved successfully")
        return True
    except Exception:
        logger.exception("Failed to save credentials")
        return False


# Function to load credentials from file
def load_credentials() -> tuple[str | None, str | None]:
    """Load credentials from a local file."""
    # Number of expected lines in credentials file (client_id and client_secret)
    EXPECTED_CREDENTIALS_LINES = 2  # noqa: N806

    try:
        if not CREDENTIALS_FILE.exists():
            return None, None

        with CREDENTIALS_FILE.open() as f:
            lines = f.readlines()

        if len(lines) >= EXPECTED_CREDENTIALS_LINES:
            client_id = lines[0].strip()
            client_secret = lines[1].strip()
            logger.info("Credentials loaded successfully")
            return client_id, client_secret
        return None, None
    except Exception:
        logger.exception("Failed to load credentials")
        return None, None


# Set page config
st.set_page_config(
    page_title="Spotify Playlist Sorter",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1DB954;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #1DB954;
    }
    .info-text {
        font-size: 1rem;
        color: #777777;
    }
    .success-box {
        padding: 1rem;
        background-color: rgba(29, 185, 84, 0.1);
        border-left: 5px solid #1DB954;
        margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem;
        background-color: rgba(255, 173, 51, 0.1);
        border-left: 5px solid #FFAD33;
        margin-bottom: 1rem;
    }
    .error-box {
        padding: 1rem;
        background-color: rgba(255, 82, 82, 0.1);
        border-left: 5px solid #FF5252;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        background-color: rgba(0, 123, 255, 0.1);
        border-left: 5px solid #007BFF;
        margin-bottom: 1rem;
    }
    .transition-card {
        padding: 1rem;
        background-color: #f9f9f9;
        border-radius: 5px;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    .key-compatible {
        color: #1DB954;
        font-weight: bold;
    }
    .key-incompatible {
        color: #FF5252;
        font-weight: bold;
    }
    .perfect-match {
        color: #1DB954;
        font-weight: bold;
    }
    .bpm-good {
        color: #1DB954;
    }
    .bpm-medium {
        color: #FFAD33;
    }
    .bpm-bad {
        color: #FF5252;
    }
    .score-high {
        color: #1DB954;
        font-weight: bold;
    }
    .score-medium {
        color: #FFAD33;
        font-weight: bold;
    }
    .score-low {
        color: #FF5252;
        font-weight: bold;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #777777;
        font-size: 0.8rem;
    }
    .documentation-link {
        color: #1DB954;
        text-decoration: underline;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main() -> None:
    """Main application function that handles the Streamlit UI and app flow."""
    # Header
    st.markdown("<h1 class='main-header'>Spotify Playlist Sorter</h1>", unsafe_allow_html=True)

    # Initialize session state variables if they don't exist
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "playlists" not in st.session_state:
        st.session_state.playlists = None
    if "playlist_id" not in st.session_state:
        st.session_state.playlist_id = None
    if "tracks_data" not in st.session_state:
        st.session_state.tracks_data = None
    if "sorter" not in st.session_state:
        st.session_state.sorter = None
    if "sorted_ids" not in st.session_state:
        st.session_state.sorted_ids = None
    if "anchor_track_id" not in st.session_state:
        st.session_state.anchor_track_id = None
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
    if "sorted_df" not in st.session_state:
        st.session_state.sorted_df = None
    if "transitions" not in st.session_state:
        st.session_state.transitions = None
    if "custom_client_id" not in st.session_state:
        # Try to load from file first
        client_id, client_secret = load_spotify_credentials()
        st.session_state.custom_client_id = client_id or ""
        st.session_state.custom_client_secret = client_secret or ""
    if "credentials_locked" not in st.session_state:
        # Auto-lock credentials if they were loaded from file
        st.session_state.credentials_locked = bool(
            st.session_state.custom_client_id and st.session_state.custom_client_secret
        )
    if "auth_flow_started" not in st.session_state:
        st.session_state.auth_flow_started = False
    if "auth_error" not in st.session_state:
        st.session_state.auth_error = None

    # Function to clear auth state
    def clear_auth_state() -> None:
        st.session_state.authenticated = False
        st.session_state.token_info = None
        st.session_state.playlists = None
        st.session_state.auth_flow_started = False
        st.session_state.auth_error = None

    # Sidebar for authentication and playlist selection
    with st.sidebar:
        st.markdown("<h2 class='sub-header'>Authentication</h2>", unsafe_allow_html=True)

        # Custom API credentials input
        st.markdown(
            """
            <div class='info-box'>
            To use this app, you need to provide your own
            <a href="https://developer.spotify.com/documentation/web-api/concepts/apps" target="_blank" class="documentation-link">
            Spotify API credentials.</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Environment selection
        if "is_local_environment" not in st.session_state:
            st.session_state.is_local_environment = True

        is_local = st.checkbox("I'm running this app locally", value=st.session_state.is_local_environment)
        if is_local != st.session_state.is_local_environment:
            st.session_state.is_local_environment = is_local
            # Clear auth state if environment changes
            if "token_info" in st.session_state:
                clear_auth_state()
                st.rerun()

        # Credential input fields
        if st.session_state.credentials_locked:
            # Display locked credentials with masked values
            st.text_input("Client ID", value="*" * 10, disabled=True)
            st.text_input("Client Secret", value="*" * 10, disabled=True)

            # Reset credentials button
            if st.button("Reset Credentials", type="primary", use_container_width=True):
                st.session_state.custom_client_id = ""
                st.session_state.custom_client_secret = ""
                st.session_state.credentials_locked = False
                clear_auth_state()
                # Delete the credentials file
                if CREDENTIALS_FILE.exists():
                    try:
                        CREDENTIALS_FILE.unlink()
                        logger.info("Credentials file deleted")
                    except Exception:
                        logger.exception("Failed to delete credentials file")
                st.rerun()
        else:
            # Editable credential fields
            custom_client_id = st.text_input("Client ID", value=st.session_state.custom_client_id, type="password")
            custom_client_secret = st.text_input(
                "Client Secret", value=st.session_state.custom_client_secret, type="password"
            )

            # Save credentials button
            if st.button("Save Credentials", type="primary", use_container_width=True):
                if custom_client_id and custom_client_secret:
                    st.session_state.custom_client_id = custom_client_id
                    st.session_state.custom_client_secret = custom_client_secret

                    # Save to file for persistence
                    if save_credentials(custom_client_id, custom_client_secret):
                        st.session_state.credentials_locked = True
                        # Reset authentication when credentials change
                        clear_auth_state()
                        st.rerun()
                    else:
                        st.error("Failed to save credentials. Please try again.")
                else:
                    st.error("Please enter both Client ID and Client Secret")

        # Authentication status
        if st.session_state.authenticated:
            st.markdown("<div class='success-box'>✓ Authenticated with Spotify</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warning-box'>⚠️ Not authenticated with Spotify</div>", unsafe_allow_html=True)

        # Reset authentication if needed
        if st.session_state.authenticated and st.button("Sign Out"):
            clear_auth_state()
            st.rerun()

        # Debug info section (collapsible)
        with st.expander("Debug Information", expanded=False):
            st.write(
                "Authentication State:", "Authenticated" if st.session_state.authenticated else "Not Authenticated"
            )
            st.write(
                "Token Present:", "Yes" if "token_info" in st.session_state and st.session_state.token_info else "No"
            )
            if "query_params" in st.session_state:
                st.write("Query Parameters:", st.session_state.query_params)
            if st.button("Clear All Session State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        # Get Spotify client with custom credentials
        if st.session_state.custom_client_id and st.session_state.custom_client_secret:
            # Temporarily set the credentials for OAuth flow
            os.environ["SPOTIFY_CLIENT_ID"] = st.session_state.custom_client_id
            os.environ["SPOTIFY_CLIENT_SECRET"] = st.session_state.custom_client_secret

            # Check if we're already authenticated
            if not st.session_state.authenticated:
                sp = get_spotify_client()
                if sp:
                    st.session_state.authenticated = True
                    st.session_state.auth_error = None
                    st.rerun()
            else:
                sp = get_spotify_client()

                if not sp:
                    st.session_state.authenticated = False
                    st.session_state.auth_error = "Authentication token expired or invalid"
                    st.error("Authentication token expired or invalid. Please try again.")
                    st.rerun()

            # If authenticated, load playlists
            if st.session_state.authenticated and sp:
                if st.button("Refresh Playlists"):
                    with st.spinner("Loading your playlists..."):
                        st.session_state.playlists = get_all_playlists(sp)
                    st.success(f"Loaded {len(st.session_state.playlists)} playlists")

                # Load playlists if not already loaded
                if st.session_state.playlists is None:
                    with st.spinner("Loading your playlists..."):
                        st.session_state.playlists = get_all_playlists(sp)

                # Playlist selection
                st.markdown("<h2 class='sub-header'>Select Playlist</h2>", unsafe_allow_html=True)

                if st.session_state.playlists:
                    playlist_options = {
                        f"{p['name']} ({p['tracks']['total']} tracks)": p["id"] for p in st.session_state.playlists
                    }
                    selected_playlist = st.selectbox(
                        "Choose a playlist to sort:", options=list(playlist_options.keys())
                    )

                    # Data Source Selection
                    data_source = st.radio(
                        "Select Data Source:",
                        options=["songdata.io", "chosic.com"],
                        index=1,
                        help="Chosic usually provides more accurate data but requires simulating a browser interaction.",
                        horizontal=True,
                    )

                    if selected_playlist:
                        playlist_id = playlist_options[selected_playlist]

                        if st.session_state.playlist_id != playlist_id:
                            st.session_state.playlist_id = playlist_id
                            st.session_state.tracks_data = None
                            st.session_state.sorter = None
                            st.session_state.sorted_ids = None
                            st.session_state.anchor_track_id = None
                            st.session_state.original_df = None
                            st.session_state.sorted_df = None
                            st.session_state.transitions = None

                        if st.button("Load Playlist Data"):
                            with st.spinner(f"Loading playlist data from Spotify and {data_source}..."):
                                sorter = SpotifyPlaylistSorter(playlist_id, sp)
                                tracks_data = sorter.load_playlist(source=data_source)

                                if tracks_data is not None and not tracks_data.empty:
                                    st.session_state.tracks_data = tracks_data
                                    st.session_state.sorter = sorter
                                    st.success(f"Loaded {len(tracks_data)} tracks with key, BPM, and energy data")
                                else:
                                    st.error("Failed to load playlist data. Please check the logs for details.")
                else:
                    st.info("No playlists found. Please refresh your playlists.")
            else:
                # Authentication flow using SpotifyOAuth
                auth_url = get_auth_url()

                if auth_url:
                    st.markdown(
                        "<div class='info-box'>Connect to Spotify to access your playlists.</div>",
                        unsafe_allow_html=True,
                    )

                    # Add authentication instructions
                    st.markdown(
                        """
                        <div class='info-box' style='background-color: #fff8e1; border-left: 5px solid #ffb300;'>
                        <strong>Authentication Instructions:</strong>
                        <ol>
                            <li>Click the button below to open Spotify authentication in a new tab</li>
                            <li>Log in to your Spotify account and authorize the app</li>
                            <li>After authorization, you'll be redirected back to this app</li>
                            <li>If the redirect doesn't work automatically, copy the URL from the Spotify auth page and paste it in this browser window</li>
                        </ol>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Create button that automatically redirects to Spotify auth
                    st.markdown(
                        f'<a href="{auth_url}" target="_blank">'
                        f'<button style="width:100%;padding:0.5em;background-color:#1DB954;color:white;'
                        f'border:none;border-radius:4px;cursor:pointer;font-weight:bold;">'
                        f"Connect Spotify Account</button></a>",
                        unsafe_allow_html=True,
                    )

                    # Show any auth errors
                    if st.session_state.auth_error:
                        st.error(f"Authentication error: {st.session_state.auth_error}")

                    # Handle auth flow debug info
                    if "code" in st.query_params:
                        st.info("Authorization code received. Processing...")
                        st.write("**Debug Information:**")
                        st.write(f"Code parameter length: {len(st.query_params['code'])}")
                        st.write(f"Redirect URI being used: {get_redirect_uri()}")
                        st.write("If authentication keeps failing, try these steps:")
                        st.write("1. Verify your Spotify app's redirect URI exactly matches the one above")
                        st.write("2. Check that your Client ID and Secret are correct")
                        st.write("3. Try clearing your session state and browser cookies")

                        if st.button("Retry Authentication"):
                            clear_auth_state()
                            st.rerun()
                else:
                    st.error("Please enter valid Spotify API credentials above to continue.")
        else:
            st.warning("Please enter your Spotify API credentials to continue.")

    # Main content area
    if st.session_state.authenticated:
        if st.session_state.tracks_data is not None and st.session_state.sorter is not None:
            # Display playlist info
            playlist_name = st.session_state.sorter.playlist_name
            st.markdown(
                f"**Playlist:** {playlist_name}   |     **Tracks with complete data:** {len(st.session_state.tracks_data)}"
            )

            st.divider()

            # Select anchor track
            st.markdown("<h2 class='sub-header'>Select Anchor Track</h2>", unsafe_allow_html=True)
            st.markdown(
                "Choose the first track for your sorted playlist. This track will be the starting point, "
                "and all other tracks will be arranged based on optimal transitions from this track."
            )

            # Add note about preview vs actual sorting
            st.warning(
                "This will not sort your Spotify playlist but it'll generate a preview of the sorted playlist. "
                "To sort your Spotify playlist please scroll to the bottom and click on Sort Playlist."
            )

            # Create a dataframe for display with track name, artist, key, BPM, energy
            display_df = st.session_state.tracks_data[["Track", "Artist", "Camelot", "BPM", "Energy"]].copy()
            display_df["BPM"] = display_df["BPM"].round().astype("Int64")
            display_df["Energy"] = (display_df["Energy"] * 10).round() / 10

            # Add a select button column
            track_options = {
                f"{row['Track']} - {row['Artist']}": row["id"] for _, row in st.session_state.tracks_data.iterrows()
            }
            selected_anchor = st.selectbox("Choose your anchor track:", options=list(track_options.keys()))

            if selected_anchor:
                anchor_track_id = track_options[selected_anchor]
                st.session_state.anchor_track_id = anchor_track_id

                # Optimization Mode Selection
                sort_mode = st.radio(
                    "Optimization Mode:",
                    options=["Fast (Greedy)", "High Quality (Beam Search)"],
                    index=0,
                    help="Fast mode sorts instantly. High Quality mode takes longer but finds better transitions across the entire playlist.",
                    horizontal=True,
                )

                # Sort button
                button_label = "Re-sort Playlist" if st.session_state.get("sorted_ids") else "Sort Playlist"
                if st.button(button_label):
                    method = "greedy" if "Fast" in sort_mode else "beam"
                    with st.spinner(f"Sorting playlist using {sort_mode} algorithm..."):
                        sorted_ids = st.session_state.sorter.sort_playlist(anchor_track_id, method=method)

                        if sorted_ids:
                            st.session_state.sorted_ids = sorted_ids

                            # Compare playlists
                            original_df, sorted_df = st.session_state.sorter.compare_playlists(sorted_ids)
                            st.session_state.original_df = original_df
                            st.session_state.sorted_df = sorted_df

                            # Get transition analysis
                            transitions = st.session_state.sorter.get_transition_analysis(sorted_ids)
                            st.session_state.transitions = transitions

                            st.success("Playlist sorted successfully!")
                        else:
                            st.error("Failed to sort playlist. Please check the logs for details.")

            st.divider()

            # Display sorted results if available
            if (
                st.session_state.sorted_ids
                and st.session_state.original_df is not None
                and st.session_state.sorted_df is not None
            ):
                st.markdown("<h2 class='sub-header'>Sorted Playlist</h2>", unsafe_allow_html=True)

                # Display comparison
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original Order**")
                    st.dataframe(
                        st.session_state.original_df[["Track", "Artist", "Camelot", "BPM", "Energy"]],
                        hide_index=True,
                    )

                with col2:
                    st.markdown("**Sorted Order**")
                    st.dataframe(
                        st.session_state.sorted_df[
                            [
                                "Track",
                                "Artist",
                                "Camelot",
                                "BPM",
                                "Energy",
                            ]
                        ],
                        hide_index=True,
                    )

                st.divider()

                # Transition analysis
                if st.session_state.transitions:
                    st.markdown("<h2 class='sub-header'>Transition Analysis</h2>", unsafe_allow_html=True)

                    # Filter out summary
                    transitions = [t for t in st.session_state.transitions if not t.get("summary", False)]
                    summary = next((t for t in st.session_state.transitions if t.get("summary", False)), None)

                    # Display summary if available
                    if summary:
                        if "average_score" in summary:
                            score_class = (
                                "score-high"
                                if summary["average_score"] > SCORE_HIGH_THRESHOLD
                                else "score-medium"
                                if summary["average_score"] > SCORE_MEDIUM_THRESHOLD
                                else "score-low"
                            )
                            st.markdown(
                                f"<div class='success-box'>"
                                f"Average Transition Score: <span class='{score_class}'>{summary['average_score']:.2f}</span> "
                                f"({summary['valid_transitions']} of {summary['total_transitions']} transitions scored)"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        elif "message" in summary:
                            st.markdown(f"<div class='warning-box'>{summary['message']}</div>", unsafe_allow_html=True)

                    # Create a DataFrame for transitions
                    transition_data = []
                    for transition in transitions:
                        if "message" in transition and "score" not in transition:
                            st.markdown(
                                f"<div class='warning-box'>{transition['message']}</div>", unsafe_allow_html=True
                            )
                            continue

                        transition_row = {
                            "Index": transition["index"],
                            "From Track": transition["track1_name"],
                            "To Track": transition["track2_name"],
                            "From Artist": transition["track1_artist"],
                            "To Artist": transition["track2_artist"],
                            "From Key": transition["key1"],
                            "To Key": transition["key2"],
                            "From BPM": transition["bpm1"],
                            "To BPM": transition["bpm2"],
                        }

                        if "bpm_diff" in transition:
                            transition_row["BPM Diff"] = (
                                transition["bpm_diff"] if transition["bpm_diff"] is not None else "N/A"
                            )

                        if "energy1" in transition and "energy2" in transition:
                            transition_row["From Energy"] = f"{transition['energy1']:.1f}"
                            transition_row["To Energy"] = f"{transition['energy2']:.1f}"

                        if "energy_diff" in transition:
                            transition_row["Energy Diff"] = f"{transition['energy_diff']:.1f}"

                        if "score" in transition:
                            transition_row["Score"] = f"{transition['score']:.2f}"
                            transition_row["Key Compatible"] = "Yes" if transition["key_compatible"] else "No"
                            transition_row["Perfect Key Match"] = (
                                "Yes" if transition.get("perfect_key_match", False) else "No"
                            )

                        transition_data.append(transition_row)

                    # Display transitions as a dataframe
                    if transition_data:
                        st.dataframe(pd.DataFrame(transition_data), hide_index=True, use_container_width=True)

                    st.divider()

                    # Add visual transition analysis with chart
                    st.markdown("### Visual Transition Analysis")
                    st.markdown("""
                    This scatter plot visualizes your playlist's transitions:
                    - Each point represents a track in your playlist with its track number
                    - **X-axis**: Tempo (BPM) - tracks with similar BPM are easier to mix
                    - **Y-axis**: Musical key (Camelot notation) - tracks with compatible keys are closer vertically
                    - **Color**: Represents energy level - brighter colors indicate higher energy
                    - **Lines**: Connect consecutive tracks, showing your playlist's progression

                    This visualization helps you see patterns in your playlist flow and identify any potentially jarring transitions.
                    """)

                    # Create and display the transition chart
                    try:
                        transition_chart = create_transition_chart(transitions)
                        st.plotly_chart(transition_chart, use_container_width=True)
                    except (ValueError, TypeError, KeyError) as e:
                        st.error(f"Failed to create transition chart: {e}")
                        st.warning("Visual chart could not be displayed due to missing or invalid data.")

                # Update playlist button
                st.markdown("<h2 class='sub-header'>Update Spotify Playlist</h2>", unsafe_allow_html=True)
                st.warning(
                    "⚠️ This will replace the current order of your playlist on Spotify. "
                    "Make sure you're happy with the sorted order before proceeding."
                )

                if st.button("Update Playlist on Spotify", type="primary"):
                    with st.spinner("Updating playlist on Spotify..."):
                        success, message = st.session_state.sorter.update_spotify_playlist(st.session_state.sorted_ids)

                        if success:
                            st.markdown(f"<div class='success-box'>✅ {message}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='error-box'>❌ {message}</div>", unsafe_allow_html=True)
        else:
            # Instructions when no playlist is loaded
            st.markdown("<h2 class='sub-header'>How It Works</h2>", unsafe_allow_html=True)
            st.markdown(
                """
                This app helps you sort your Spotify playlists for optimal transitions between tracks.

                **Features:**
                - Loads playlist data directly from Spotify
                - Analyzes tracks using songdata.io
                - Sorts tracks based on:
                  - Harmonic mixing (Camelot wheel)
                  - BPM (tempo) similarity
                  - Energy level transitions
                - Updates playlist order on Spotify
                - Provides detailed transition analysis

                **To get started:**
                1. Enter your Spotify API credentials in the sidebar
                2. Authenticate with Spotify
                3. Select a playlist to sort
                4. Choose an anchor track (the first track in your sorted playlist)
                5. Review the transition analysis
                6. Update your playlist with the optimized order
                """
            )

            st.markdown(
                "<div class='warning-box'>⚠️ Please provide your Spotify API credentials in the sidebar and select a playlist to continue.</div>",
                unsafe_allow_html=True,
            )
    else:
        # Not authenticated
        col1, col2 = st.columns(spec=[1, 1], gap="medium")

        with col1:
            st.markdown(
                """
                <div class='info-box' style='margin-bottom: 20px;'>
                This app helps you create the perfect playlist flow by sorting your tracks based on:

                • <strong>Harmonic Compatibility</strong> (Camelot wheel)
                • <strong>BPM Matching</strong> (tempo transitions)
                • <strong>Energy Flow</strong> (smooth energy level progression)
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class='info-box' style='background-color: rgba(29, 185, 84, 0.1); border-left: 5px solid #1DB954;'>
                <h3 style='color: #1DB954; margin-top: 0;'>Getting Spotify API Credentialss</h3>

                <ol style='padding-left: 20px; margin-bottom: 0;'>
                    <li>Go to the <a href="https://developer.spotify.com/dashboard" target="_blank" style="color: #1DB954; text-decoration: underline;">Spotify Developer Dashboard</a></li>
                    <li>Log in with your Spotify account</li>
                    <li>Click "Create an App"</li>
                    <li>Fill in the app name and description</li>
                    <li>Set the Redirect URI based on your environment:<br>
                    <code style="background-color: #f0f0f0; padding: 2px 5px; border-radius: 3px;">"""
                + (
                    """http://127.0.0.1:8501"""
                    if st.session_state.is_local_environment
                    else """https://spotify-playlist-sorter.streamlit.app"""
                )
                + """</code></li>
                    <li>Copy the Client ID and Client Secret to use in this app</li>
                </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            # How it works
            st.markdown(
                """
                <div class='transition-card' style='background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 15px; border-left: 5px solid #1DB954;'>
                    <h4 style='color: #1DB954; margin-top: 0;'>1. Camelot Wheel</h4>
                    <p style='margin-bottom: 0;'>Tracks are arranged based on musical key compatibility using the Camelot wheel system.
                    Compatible keys create harmonic transitions between tracks.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class='transition-card' style='background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 15px; border-left: 5px solid #1DB954;'>
                    <h4 style='color: #1DB954; margin-top: 0;'>2. BPM Matching</h4>
                    <p style='margin-bottom: 0;'>Tracks with similar tempos are placed together to create smooth transitions.
                    Gradual BPM changes prevent jarring tempo shifts.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class='transition-card' style='background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 15px; border-left: 5px solid #1DB954;'>
                    <h4 style='color: #1DB954; margin-top: 0;'>3. Energy Flow</h4>
                    <p style='margin-bottom: 0;'>The algorithm considers energy levels to create a natural flow.
                    This prevents sudden drops or spikes in energy throughout your playlist.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Footer
    st.markdown("---")
    st.markdown("Created with ❤️ by [@MishraMishry](https://x.com/MishraMishry)")
    st.markdown(
        "Psst! This app is open source. Check out the [GitHub repo](https://github.com/SarthakMishra/spotify-playlist-sorter) if you're into that sort of thing. 😉"
    )


def create_transition_chart(transitions: list[dict]) -> go.Figure:
    """Create a scatter plot visualization of track transitions.

    Args:
        transitions: List of transition dictionaries with track and transition data

    Returns:
        A plotly figure object representing transitions
    """
    # Constants
    MIN_TRACKS_FOR_CHART = 2  # Minimum number of tracks needed for a meaningful chart  # noqa: N806

    # Prepare data for the chart
    chart_data = []

    # For each transition, create data points for the tracks
    for i, transition in enumerate(transitions):
        if "score" not in transition:
            continue

        try:
            # Convert BPM and Energy values safely
            bpm1 = float(transition["bpm1"]) if transition["bpm1"] is not None else 0
            bpm2 = float(transition["bpm2"]) if transition["bpm2"] is not None else 0
            energy1 = float(transition["energy1"]) if transition["energy1"] is not None else 0
            energy2 = float(transition["energy2"]) if transition["energy2"] is not None else 0

            # Extract data for the first track
            track1 = {
                "Track": f"{transition['track1_name']} - {transition['track1_artist']}",
                "Key": transition["key1"],
                "BPM": bpm1,
                "Energy": energy1,
                "Position": i,
                "TrackNum": i + 1,
            }

            # Extract data for the second track (if it's not already in the list as a previous track)
            track2 = {
                "Track": f"{transition['track2_name']} - {transition['track2_artist']}",
                "Key": transition["key2"],
                "BPM": bpm2,
                "Energy": energy2,
                "Position": i + 1,
                "TrackNum": i + 2,
            }

            # Only add track1 if it's the first track or different from previous track2
            if i == 0 or chart_data[-1]["Track"] != track1["Track"]:
                chart_data.append(track1)
            chart_data.append(track2)
        except (ValueError, TypeError, KeyError):
            # Skip this transition if data conversion fails
            continue

    # If we don't have enough data, raise an error
    if len(chart_data) < MIN_TRACKS_FOR_CHART:
        msg = "Not enough valid transition data to create chart"
        raise ValueError(msg)

    # Create a DataFrame for the scatter plot
    df = pd.DataFrame(chart_data)  # noqa: PD901

    # Create a scatter plot with Plotly Express
    fig = px.scatter(
        df,
        x="BPM",
        y="Key",
        color="Energy",
        size="Energy",  # Varying sizes based on energy level
        color_continuous_scale="Viridis",
        hover_name="Track",
        text="TrackNum",
        range_color=[0, 1],  # Energy is 0-1
        title="Playlist Flow: BPM vs Key (color shows Energy)",
    )

    # Improve the layout
    fig.update_layout(
        height=800,
        xaxis_title="Tempo (BPM)",
        yaxis_title="Musical Key (Camelot)",
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
    )

    # Configure text display on points
    fig.update_traces(
        textposition="top center",
        textfont={"size": 10, "color": "gray"},
        marker={"line": {"width": 1, "color": "darkgray"}},
        selector={"mode": "markers+text"},
    )

    return fig


if __name__ == "__main__":
    main()