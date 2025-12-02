"""Constants used throughout the Spotify Playlist Sorter application."""

# Score thresholds for transition quality
SCORE_HIGH_THRESHOLD = 0.7
SCORE_MEDIUM_THRESHOLD = 0.4

# BPM difference thresholds
BPM_GOOD_THRESHOLD = 5
BPM_MEDIUM_THRESHOLD = 10
BPM_MAX_DIFFERENCE = 16  # New constant: BPM differences higher than this incur heavy penalties

# Energy thresholds
ENERGY_SMALL_INCREASE_MAX = 0.15
ENERGY_SMALL_DECREASE_MIN = -0.1
ENERGY_MODERATE_INCREASE_MAX = 0.3
ENERGY_SCALING_FACTOR = 0.6

# Camelot wheel constants
CAMELOT_MAX_NUMBER = 12
CAMELOT_MIN_NUMBER = 1

# API constants
API_BATCH_SIZE = 50
DATA_ROW_THRESHOLD = 5

# HTTP request timeout (seconds)
HTTP_TIMEOUT = 30

# Spotify API endpoint
AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
HTTP_STATUS_UNAUTHORIZED = 401