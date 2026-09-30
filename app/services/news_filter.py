import httpx
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from app.services.brainstorm_service import brainstorm_service

logger = logging.getLogger("NewsFilterService")

class EconomicNewsFilterService:
    """
    Real-Time Economic Calendar & High-Impact News Filter (Enterprise).
    Monitors High-Impact (Red Folder) Events:
    - FOMC Interest Rate Decisions & Fed Chair Press Conferences
    - US Consumer Price Index (CPI & Core CPI Inflation)
    - US Non-Farm Payrolls (NFP) & Unemployment Rate
    - US Gross Domestic Product (GDP) & Retail Sales
    - ECB, BOE, BOJ, SNB Interest Rate Decisions
    
    Functions:
    1. Fetches official live ForexFactory economic calendar (FairEconomy feed).
    2. Calculates real-time blackout window (+/- pause_minutes) around news releases.
    3. Blocks auto-execution during active windows to prevent slippage & spread spikes.
    """

    # A calendar older than this is treated as unusable rather than as "clear"
    MAX_CALENDAR_AGE_SECONDS = 3 * 3600

    def __init__(self):
        self.cached_events: List[Dict[str, Any]] = []
        self.last_fetch_time: Optional[datetime] = None
        self.high_impact_keywords = [
            "FOMC", "FEDERAL FUNDS RATE", "POWELL", "FED CHAIR", "INTEREST RATE",
            "CPI", "CORE CPI", "NON-FARM", "NFP", "UNEMPLOYMENT", "GDP",
            "RETAIL SALES", "PPI", "PCE", "ECB", "BOE", "BOJ", "SNB", "RATE DECISION"
        ]

    def parse_event_time(self, raw_time_str: str) -> Optional[datetime]:
        """Parses various date/time formats from financial calendar feeds to UTC datetime"""
        if not raw_time_str:
            return None
        
        time_str = raw_time_str.strip()
        
        formats_to_try = [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%m-%d-%Y %I:%M%p",
            "%Y-%m-%d"
        ]
        
        for fmt in formats_to_try:
            try:
                dt = datetime.strptime(time_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt
            except Exception:
                continue
                
        try:
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            return None

    async def fetch_live_economic_calendar(self) -> List[Dict[str, Any]]:
        """Fetches live economic calendar events from global feeds with fallback cache"""
        now = datetime.now(timezone.utc)
        if self.last_fetch_time and (now - self.last_fetch_time).total_seconds() < 600 and self.cached_events:
            return self.cached_events

        # Official live FairEconomy / ForexFactory weekly JSON feed
        endpoints = [
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        ]

        async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}) as client:
            for url in endpoints:
                try:
                    res = await client.get(url)
                    if res.status_code == 200:
                        data = res.json()
                        if isinstance(data, list) and len(data) > 0:
                            parsed = []
                            for item in data:
                                title = str(item.get("title", item.get("name", item.get("event", "")))).upper()
                                currency = str(item.get("country", item.get("currency", "USD"))).upper()
                                impact = str(item.get("impact", "High")).capitalize()
                                
                                is_high_impact = (
                                    impact in ["High", "Red", "Critical"] or
                                    any(k in title for k in self.high_impact_keywords)
                                )

                                if is_high_impact:
                                    event_time_str = item.get("date", item.get("time", item.get("datetime", "")))
                                    event_dt = self.parse_event_time(event_time_str)
                                    
                                    parsed.append({
                                        "title": title,
                                        "currency": currency,
                                        "impact": "HIGH",
                                        "time_str": event_time_str,
                                        "timestamp_utc": event_dt.isoformat() if event_dt else None,
                                        "forecast": item.get("forecast", "N/A"),
                                        "previous": item.get("previous", "N/A"),
                                        "actual": item.get("actual", "N/A")
                                    })

                            if parsed:
                                self.cached_events = parsed
                                self.last_fetch_time = now
                                logger.info(f"Loaded {len(parsed)} Live High-Impact ForexFactory Events.")
                                return self.cached_events
                except Exception as e:
                    logger.debug(f"Calendar endpoint {url} failed: {e}")

        return self.cached_events

    def is_symbol_affected_by_currency(self, symbol: str, currency: str) -> bool:
        """Determines if a trading symbol is affected by an economic currency release"""
        sym = symbol.upper()
        curr = currency.upper()
        
        # Gold is heavily USD denominated
        if "XAU" in sym or "GOLD" in sym:
            return curr in ["USD", "US"]
        
        # Currency pairs (e.g. EURUSD is affected by EUR and USD)
        if len(curr) == 3:
            return curr in sym
        elif curr == "US":
            return "USD" in sym
        elif curr == "EU":
            return "EUR" in sym
        elif curr == "UK" or curr == "GB":
            return "GBP" in sym
        elif curr == "JP":
            return "JPY" in sym

        return False

    def is_trading_allowed(self, symbol: str, pause_minutes: int = 15) -> Dict[str, Any]:
        """
        Evaluates whether high-impact news is currently occurring or imminent.
        Returns safety verdict, upcoming event info, and delay rationale.
        """
        now = datetime.now(timezone.utc)
        pause_delta = timedelta(minutes=pause_minutes)

        # Fail closed on a stale calendar. If the feed has been unreachable for hours
        # the cache silently becomes last week's events, and an "allowed" verdict then
        # means "we are blind", not "the calendar is clear".
        age_seconds = (now - self.last_fetch_time).total_seconds() if self.last_fetch_time else None
        if age_seconds is None or age_seconds > self.MAX_CALENDAR_AGE_SECONDS:
            age_desc = "never fetched" if age_seconds is None else f"{int(age_seconds / 60)} minutes old"
            return {
                "allowed": False,
                "stale": True,
                "reason": f"Economic calendar unavailable ({age_desc}). Blocking auto-execution rather than trading blind.",
                "upcoming_event": None
            }

        for event in self.cached_events:
            curr = event.get("currency", "USD")
            if not self.is_symbol_affected_by_currency(symbol, curr):
                continue

            event_time_iso = event.get("timestamp_utc")
            if not event_time_iso:
                continue

            try:
                event_dt = datetime.fromisoformat(event_time_iso)
            except Exception:
                continue

            # Check if current time is within [event_dt - pause_delta, event_dt + pause_delta]
            time_diff = event_dt - now
            abs_seconds = abs(time_diff.total_seconds())

            if abs_seconds <= pause_minutes * 60:
                title = event.get("title", "High-Impact Economic Release")
                minutes_relative = int(time_diff.total_seconds() / 60)
                timing_desc = f"in {minutes_relative} mins" if minutes_relative > 0 else f"released {abs(minutes_relative)} mins ago"

                brainstorm_service.add_log(
                    level="RISK",
                    category="NEWS",
                    symbol=symbol,
                    message=f"⚠️ HIGH-IMPACT NEWS BLACKOUT: {title} ({curr}) {timing_desc}. Auto-trading paused."
                )
                return {
                    "allowed": False,
                    "stale": False,
                    "event": title,
                    "reason": f"High-Impact News ({title} - {curr}) {timing_desc}. Pausing trades for {pause_minutes}m to avoid slippage.",
                    "upcoming_event": event
                }

        return {
            "allowed": True,
            "stale": False,
            "reason": "Economic calendar clear. No active high-impact blackout window.",
            "upcoming_event": None
        }

news_filter = EconomicNewsFilterService()
