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
    # --- Team abbreviations ---
    # Current structure: button[data-team] with data-home="0" (away) / "1" (home)
    away_abbr = home_abbr = None
    for btn in card.find_all("button", attrs={"data-team": True}):
        team = btn.get("data-team", "").upper()
        if not team:
            continue
        if btn.get("data-home") == "1" and not home_abbr:
            home_abbr = team
        elif btn.get("data-home") == "0" and not away_abbr:
            away_abbr = team

    # Fallback: legacy .lineup__abbr elements
    if not away_abbr or not home_abbr:
        team_tags = card.find_all(class_=re.compile(r"lineup__abbr|team-abbr"))
        if len(team_tags) >= 2:
            away_abbr = team_tags[0].get_text(strip=True)
            home_abbr = team_tags[1].get_text(strip=True)

    if not away_abbr or not home_abbr:
        return None

    # --- Game time ---
    time_tag = card.find(class_=re.compile(r"lineup__time|game-time"))
    game_time = time_tag.get_text(strip=True) if time_tag else ""

    # --- Batting order lists ---
    # is-visit = away, is-home = home
    visit_list = card.find(class_=lambda c: c and "lineup__list" in c and "is-visit" in c)
    home_list  = card.find(class_=lambda c: c and "lineup__list" in c and "is-home" in c)

    # Positional fallback
    if visit_list is None or home_list is None:
        list_tags = card.find_all(class_=re.compile(r"lineup__list|batter-list"))
        if visit_list is None:
            visit_list = list_tags[0] if list_tags else None
        if home_list is None:
            home_list = list_tags[1] if len(list_tags) > 1 else None

    # --- Pitchers ---
    # Current: first .lineup__player-highlight in each list contains pitcher
    # Legacy: .lineup__pitcher block
    def _pitcher_from_list(lst: Tag | None) -> tuple[str, str]:
        if lst is None:
            return ("TBD", "?")
        highlight = lst.find(class_=re.compile(r"lineup__player-highlight"))
        if highlight:
            name_block = highlight.find(class_="lineup__player-highlight-name")
            if name_block:
                name_link = name_block.find("a")
                name = name_link.get_text(strip=True) if name_link else "TBD"
                throws = name_block.find(class_="lineup__throws")
                hand = throws.get_text(strip=True) if throws else "?"
                return (name or "TBD", hand or "?")
        # Legacy fallback: .lineup__pitcher or parens-hand text
        pitcher_tag = lst.find(class_=re.compile(r"lineup__pitcher|sp-name"))
        if pitcher_tag:
            text = pitcher_tag.get_text(separator=" ", strip=True)
            hand_m = re.search(r"\(([LRS])\)", text)
            hand = hand_m.group(1) if hand_m else "?"
            name = re.sub(r"\([LRS]\)", "", text).strip()
            return (name or "TBD", hand or "?")
        return ("TBD", "?")

    away_pitcher_name, away_pitcher_hand = _pitcher_from_list(visit_list)
    home_pitcher_name, home_pitcher_hand = _pitcher_from_list(home_list)

    away_batters = _extract_batters(visit_list)
    home_batters = _extract_batters(home_list)

    is_confirmed = bool(card.find(class_=re.compile(r"is-confirmed")))

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
    # Exclude pitcher-highlight items; match only plain lineup__player entries
    items = tag.find_all(
        "li",
        class_=lambda c: c and "lineup__player" in c and "lineup__player-highlight" not in " ".join(c),
    )
    # Fallback to div-based selectors used by some older page variants
    if not items:
        items = tag.find_all(["li", "div"], class_=re.compile(r"^lineup__player$|batter"))
    for item in items[:9]:
        pos_tag = item.find(class_="lineup__pos")
        pos = pos_tag.get_text(strip=True) if pos_tag else "?"
        name_link = item.find("a")
        if name_link:
            name = name_link.get("title") or name_link.get_text(strip=True)
        else:
            name = item.get_text(separator=" ", strip=True)
        bats_tag = item.find(class_="lineup__bats")
        hand = bats_tag.get_text(strip=True) if bats_tag else "?"
        batters.append({"name": name, "hand": hand, "pos": pos})
    return batters


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
