"""Protocol settings for serving graphs."""

IMPLEMENTATION = "31ba03a874f775372154f0edf3e17d436f2d7462"
COMPILATION = {
    "mode": 0,
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [1, 2, 3, 4],
}
