"""
RotoWire confirmed lineup scraper.

CONSTRAINT: RotoWire has no official API. This module scrapes HTML from
rotowire.com/baseball/lineups.php. Any change to the page structure will
break Cycle B lineup ingestion and must be monitored.

On scrape failure, the pipeline falls back to MLB Stats API probable pitchers
and marks the prediction as Cycle-A-quality (Section 5.4).
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag

from config import ROTOWIRE_LINEUPS_URL, ROTOWIRE_RETRIES, ROTOWIRE_RETRY_DELAY, HTTP_TIMEOUT
from utils.retry import fixed_delay_retry

logger = logging.getLogger(__name__)

_TRANSIENT = (requests.exceptions.RequestException, ValueError)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.rotowire.com/",
}


@dataclass
class ConfirmedLineup:
    team_abbr: str
    pitcher_name: str
    pitcher_hand: str   # L / R / ?
    batting_order: list[dict] = field(default_factory=list)  # [{name, hand, pos}]
    is_confirmed: bool = False


@dataclass
class GameLineups:
    home: ConfirmedLineup
    away: ConfirmedLineup
    game_time: str = ""


# ---------------------------------------------------------------------------
# Raw HTML fetch
# ---------------------------------------------------------------------------

@fixed_delay_retry(
    retries=ROTOWIRE_RETRIES,
    delay=ROTOWIRE_RETRY_DELAY,
    exceptions=_TRANSIENT,
)
def _fetch_html() -> str:
    resp = requests.get(ROTOWIRE_LINEUPS_URL, headers=_HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    if len(resp.text) < 1000:
        raise ValueError("RotoWire returned unexpectedly short page")
    return resp.text


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_lineups(html: str) -> list[GameLineups]:
    """
    Parse the RotoWire lineups page into a list of GameLineups.

    RotoWire renders a series of '.lineup__main' cards, each containing:
      - two .lineup__team blocks (away / home)
      - a .lineup__list with batters
      - a .lineup__pitcher block with the starting pitcher

    The exact CSS class names are matched loosely to tolerate minor redesigns.
    """
    soup = BeautifulSoup(html, "lxml")
    games: list[GameLineups] = []

    cards = soup.find_all("div", class_=re.compile(r"lineup__main|lineup-card"))
    if not cards:
        # Fallback: look for any div with two team abbreviations
        cards = soup.find_all("div", class_=re.compile(r"lineup"))
        cards = [c for c in cards if c.find(class_=re.compile(r"lineup__list|lineup-list"))]

    for card in cards:
        try:
            game = _parse_card(card)
            if game:
                games.append(game)
        except Exception as exc:
            logger.debug("Failed to parse lineup card: %s", exc)

    return games


def _parse_card(card: Tag) -> GameLineups | None:
    # Extract team abbreviations
    team_tags = card.find_all(class_=re.compile(r"lineup__abbr|team-abbr"))
    if len(team_tags) < 2:
        # Try text content of header elements
        abbr_tags = card.find_all(["span", "div"], string=re.compile(r"^[A-Z]{2,3}$"))
        if len(abbr_tags) < 2:
            return None
        team_tags = abbr_tags[:2]

    away_abbr = team_tags[0].get_text(strip=True)
    home_abbr = team_tags[1].get_text(strip=True)

    # Game time
    time_tag = card.find(class_=re.compile(r"lineup__time|game-time"))
    game_time = time_tag.get_text(strip=True) if time_tag else ""

    # Pitcher blocks
    pitcher_tags = card.find_all(class_=re.compile(r"lineup__pitcher|sp-name"))

    def _extract_pitcher(tag: Tag | None) -> tuple[str, str]:
        if tag is None:
            return ("TBD", "?")
        text = tag.get_text(separator=" ", strip=True)
        # Hand is often in parens: "J. Doe (R)" or as a sub-span
        hand_match = re.search(r"\(([LRS])\)", text)
        hand = hand_match.group(1) if hand_match else "?"
        name = re.sub(r"\([LRS]\)", "", text).strip()
        return (name, hand)

    away_pitcher_name, away_pitcher_hand = _extract_pitcher(
        pitcher_tags[0] if pitcher_tags else None
    )
    home_pitcher_name, home_pitcher_hand = _extract_pitcher(
        pitcher_tags[1] if len(pitcher_tags) > 1 else None
    )

    # Batting order lists
    list_tags = card.find_all(class_=re.compile(r"lineup__list|batter-list"))
    away_batters = _extract_batters(list_tags[0] if list_tags else None)
    home_batters = _extract_batters(list_tags[1] if len(list_tags) > 1 else None)

    confirmed_tag = card.find(class_=re.compile(r"lineup__confirmed|lineup-confirmed"))
    is_confirmed = confirmed_tag is not None or "confirmed" in card.get_text().lower()

    return GameLineups(
        home=ConfirmedLineup(
            team_abbr=home_abbr,
            pitcher_name=home_pitcher_name,
            pitcher_hand=home_pitcher_hand,
            batting_order=home_batters,
            is_confirmed=is_confirmed,
        ),
        away=ConfirmedLineup(
            team_abbr=away_abbr,
            pitcher_name=away_pitcher_name,
            pitcher_hand=away_pitcher_hand,
            batting_order=away_batters,
            is_confirmed=is_confirmed,
        ),
        game_time=game_time,
    )


def _extract_batters(tag: Tag | None) -> list[dict]:
    if tag is None:
        return []
    batters = []
    items = tag.find_all(["li", "div"], class_=re.compile(r"lineup__player|batter"))
    for item in items:
        text = item.get_text(separator=" ", strip=True)
        hand_match = re.search(r"\b([LRS])\b", text)
        hand = hand_match.group(1) if hand_match else "?"
        pos_match = re.search(r"\b([A-Z]{1,2})\b", text)
        pos = pos_match.group(1) if pos_match and len(pos_match.group(1)) <= 2 else "?"
        name = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", text)).strip()
        batters.append({"name": name, "hand": hand, "pos": pos})
    return batters[:9]  # batting order is 9 spots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_confirmed_lineups(game_date: date) -> dict[str, GameLineups]:
    """
    Return a dict keyed by "{away_abbr}@{home_abbr}" -> GameLineups.
    Returns {} on complete scrape failure (caller handles fallback).
    """
    try:
        html = _fetch_html()
    except Exception as exc:
        logger.error("RotoWire scrape failed: %s", exc)
        return {}

    lineups = _parse_lineups(html)

    result: dict[str, GameLineups] = {}
    for gl in lineups:
        key = f"{gl.away.team_abbr}@{gl.home.team_abbr}"
        result[key] = gl

    logger.info("RotoWire: %d game lineup cards parsed", len(result))
    if not result:
        logger.warning(
            "RotoWire parser returned 0 games — page structure may have changed"
        )
    return result
