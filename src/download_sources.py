"""Brick 3: download every sorting guide listed in data/sources.csv into data/raw/.

The guides are copyrighted ("All rights reserved"), so the repo only stores the links.
Anyone can rebuild the same documents locally with:
    python src/download_sources.py
"""
from datetime import date

import pandas as pd
import requests

SOURCES_PATH = "data/sources.csv"
RAW_DIR = "data/raw"
HEADERS = {"User-Agent": "sort-it-right student project (recycling assistant)"}

sources = pd.read_csv(SOURCES_PATH, dtype=str)   # dtype=str: keep empty dates as text, not NaN floats
today = date.today().isoformat()

for i, row in sources.iterrows():
    try:
        response = requests.get(row["url"], headers=HEADERS, timeout=30)
        response.raise_for_status()               # turn 404 / 500 errors into exceptions
    except requests.RequestException as error:
        print(f"FAILED  {row['file']}: {error}")
        continue                                  # keep going with the other sources

    # Write raw bytes ("wb"): works for both PDF and HTML without decoding problems.
    with open(f"{RAW_DIR}/{row['file']}", "wb") as f:
        f.write(response.content)
    sources.loc[i, "downloaded_on"] = today
    print(f"OK      {row['file']} ({len(response.content) // 1024} KB)")

sources.to_csv(SOURCES_PATH, index=False)          # save the download dates back
print(f"\nDownloaded {(sources['downloaded_on'] == today).sum()}/{len(sources)} sources into {RAW_DIR}/")
