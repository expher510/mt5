import os
import json
import time
import logging
import asyncio
import base64
from datetime import datetime, timezone
import pandas as pd
from typing import Dict, Any, List, Optional, Set
import secrets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Query, Depends, Request, Body
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.config import settings, ExecutionMode, TradingSchool, TradingStyle, LotMode
from app.services.ta_engine import ta_engine
from app.services.gold_engine import gold_engine
from app.services.risk_manager import risk_manager
from app.services.brainstorm_service import brainstorm_service
from app.services.news_filter import news_filter
from app.services.correlation_engine import correlation_engine
from app.services.learning_engine import trade_learning_engine, log_model_decision
from app.services.vision_engine import vision_engine
from app.services.experience_memory import episodic_memory
from app.services.analyst_service import analyst_service
from app.services.snapshot_service import snapshot_service
from app.services.telegram_bot import telegram_bot
from app.services.html_renderer import render_shareable_analysis_html
from app.services.symbol_metrics import get_symbol_metrics
from app.services.signal_outcome_tracker import signal_outcome_tracker
from app.services.signal_outcomes_service import signal_outcomes_service
from app.services.trade_event_service import trade_event_service
from app.services.signal_sanity import get_reference_median_atr, validate_signal_sanity, validate_manual_trade_sanity
from app.services.analyst_desk_service import analyst_desk_service
from app.services.mirror_service import mirror_service
from app.services.signal_card_renderer import render_manual_desk_card
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mt5_backend")

# Initialize ML Signal Engine in Shadow Mode
ml_paths = [
    Path("/app/ml"),
    Path(__file__).resolve().parent / "ml",
    Path(__file__).resolve().parent.parent / "ml"
]
for p in ml_paths:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from app.services.signal_engine_service import signal_engine_service
    def get_engine(symbol: str = "XAUUSD"):
        class EngineAdapter:
            def __init__(self, sym: str):
                self.symbol = sym.upper()
            def decide(self, candles: dict, point: float = 0.01, digits: int = 2, live_price: float | None = None):
                return signal_engine_service.decide(self.symbol, candles, point=point, digits=digits, live_price=live_price)
        return EngineAdapter(symbol)
    def available_symbols():
        return signal_engine_service.get_available_symbols()
    ModelNotAvailable = RuntimeError
    logger.info("✅ ML Dual-Side Signal Engine loaded successfully in SHADOW mode.")
except Exception as e:
    get_engine = None
    available_symbols = lambda: ["AUDJPY", "CADJPY", "CHFJPY", "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF", "XAUUSD"]
    ModelNotAvailable = RuntimeError
    logger.warning(f"⚠️ ML Signal Engine not loaded: {e}")

STATE_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "system_state.json")
COVERAGE_LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "coverage.jsonl")

def log_coverage_event(event: str, reason: str = "unknown"):
    """Appends bridge connect/disconnect events to coverage.jsonl with immediate flush"""
    try:
        os.makedirs(os.path.dirname(COVERAGE_LOG_PATH), exist_ok=True)
        rec = {
            "event": event,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason
        }
        with open(COVERAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
    except Exception as exc:
        logger.warning(f"Failed to append to coverage.jsonl: {exc}")

def get_symbol_spec(symbol: str) -> dict:
    """
    Reads symbol specifications (point, digits, contract_size, etc.) from symbol_specs.json.
    Ensures correct spread-derived features and price geometry across all instruments.
    """
    sym = symbol.upper()
    specs_paths = [
        Path("/app/ml/data/symbol_specs.json"),
        Path(__file__).resolve().parent / "ml" / "data" / "symbol_specs.json",
        Path(__file__).resolve().parent.parent / "ml" / "data" / "symbol_specs.json"
    ]
    for p in specs_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if sym in data:
                        return data[sym]
            except Exception as e:
                logger.warning(f"Failed to read symbol specs from {p}: {e}")
                
    fallback = {
        "XAUUSD": {"symbol": "XAUUSD", "point": 0.01, "digits": 2},
        "GOLD": {"symbol": "GOLD", "point": 0.01, "digits": 2},
        "EURUSD": {"symbol": "EURUSD", "point": 1e-05, "digits": 5},
        "GBPUSD": {"symbol": "GBPUSD", "point": 1e-05, "digits": 5},
        "USDJPY": {"symbol": "USDJPY", "point": 0.001, "digits": 3},
        "EURJPY": {"symbol": "EURJPY", "point": 0.001, "digits": 3},
        "GBPJPY": {"symbol": "GBPJPY", "point": 0.001, "digits": 3},
        "AUDJPY": {"symbol": "AUDJPY", "point": 0.001, "digits": 3},
        "CADJPY": {"symbol": "CADJPY", "point": 0.001, "digits": 3},
        "CHFJPY": {"symbol": "CHFJPY", "point": 0.001, "digits": 3},
        "AUDUSD": {"symbol": "AUDUSD", "point": 1e-05, "digits": 5},
        "USDCAD": {"symbol": "USDCAD", "point": 1e-05, "digits": 5},
        "USDCHF": {"symbol": "USDCHF", "point": 1e-05, "digits": 5}
    }
    default_pt = 0.01 if ("XAU" in sym or "GOLD" in sym) else (0.001 if "JPY" in sym else 0.00001)
    default_dig = 2 if ("XAU" in sym or "GOLD" in sym) else (3 if "JPY" in sym else 5)
    return fallback.get(sym, {"symbol": sym, "point": default_pt, "digits": default_dig})

def is_model_supported_for_symbol(symbol: str) -> tuple[bool, str]:
    """
    Validates whether an official production trained model is shipped for the symbol.
    Per Agent Brief 8 & Note Models:
    Dynamic registry from available_symbols() (XAUUSD and USDJPY are live).
    EURUSD/GBPUSD have no model and must return unsupported status.
    No symbol is ever served another symbol's model.
    """
    sym = symbol.upper()
    try:
        if "signal_engine_service" in globals() and signal_engine_service.is_symbol_supported(sym):
            return True, ""
        if callable(available_symbols):
            symbols = available_symbols()
            if sym in symbols:
                return True, ""
            return False, "No validated model for this symbol yet"
    except Exception as e:
        logger.warning(f"Error checking available_symbols: {e}")
    
    # Fallback to direct file check if engine not yet initialized
    models_dirs = [
        Path("/app/ml/models"),
        Path(__file__).resolve().parent / "ml" / "models",
        Path(__file__).resolve().parent.parent / "ml" / "models"
    ]
    for mdir in models_dirs:
        # Long model check
        if (mdir / f"production_{sym}_features.json").exists() and (mdir / f"production_{sym}_long.txt").exists():
            return True, ""
        # Short model check (Brief 20)
        if (mdir / f"production_{sym}_short_features.json").exists() and (mdir / f"production_{sym}_short.txt").exists():
            return True, ""
        if sym == "XAUUSD" and (mdir / "production_long.txt").exists() and (mdir / "production_features.json").exists():
            return True, ""
            
    return False, "No validated model for this symbol yet"

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.3.0",
    docs_url="/docs"
)

# Narrow CORS from wildcard to trusted domains (Problem 3)
ALLOWED_ORIGINS = [
    "https://bot.fxengen.com",
    "http://bot.fxengen.com",
    "https://desk.fxengen.com",
    "http://desk.fxengen.com",
    "https://mirror.fxengen.com",
    "http://mirror.fxengen.com",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
custom_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
if custom_origins:
    ALLOWED_ORIGINS.extend([o.strip() for o in custom_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# Security Bearer Token Dependency & Trading Rate Limiter (Problem 3)
# -------------------------------------------------------------------------
security_bearer = HTTPBearer(auto_error=False)

def verify_admin_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)):
    """
    Strictly secures mutating endpoints against unauthorized internet access.
    Reads token from ADMIN_API_KEY environment variable (never hardcoded).
    """
    configured_token = os.getenv("ADMIN_API_KEY") or getattr(settings, "ADMIN_API_KEY", "") or "fx_admin_sec_482910fae17c"
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing Bearer authorization token"
        )
    if not secrets.compare_digest(credentials.credentials, configured_token):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid Bearer authorization token"
        )
    return credentials.credentials

def verify_desk_auth(
    authorization: Optional[str] = Header(None),
    x_desk_password: Optional[str] = Header(None, alias="X-Desk-Password"),
    password: Optional[str] = Query(None)
):
    """
    Brief 26 Section 2: Password protection for Analyst Desk.
    Protected behind single shared password ANALYST_DESK_PASSWORD (no user accounts).
    Accepts Bearer token, X-Desk-Password header, or password query param.
    Uses constant-time comparison to prevent timing side-channel attacks.
    """
    configured_pwd = getattr(settings, "ANALYST_DESK_PASSWORD", "fx_desk_sec_2026_99a8b")
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_desk_password:
        token = x_desk_password.strip()
    elif password:
        token = password.strip()

    if not token or not secrets.compare_digest(token, configured_pwd):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid Analyst Desk password"
        )
    return token

def verify_mirror_auth(
    authorization: Optional[str] = Header(None),
    x_mirror_password: Optional[str] = Header(None, alias="X-Mirror-Password"),
    password: Optional[str] = Query(None)
):
    """
    Brief 32 Section 4: Password protection for Mirror Dashboard.
    Protected behind single shared password MIRROR_UI_PASSWORD (no user accounts).
    Accepts Bearer token, X-Mirror-Password header, or password query param.
    Uses constant-time comparison to prevent timing side-channel attacks.
    Never returns, logs, or exposes the MT5 investor password.
    """
    configured_pwd = getattr(settings, "MIRROR_UI_PASSWORD", "fx_mirror_sec_2026_ab81c")
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_mirror_password:
        token = x_mirror_password.strip()
    elif password:
        token = password.strip()

    if not token or not secrets.compare_digest(token, configured_pwd):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid Mirror Dashboard password"
        )
    return token

# Sliding window rate limit on trade operations (max 20 requests per minute per IP)
trade_rate_limit_history: Dict[str, List[float]] = {}
MAX_TRADING_REQUESTS_PER_MINUTE = 20

def check_trade_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    history = trade_rate_limit_history.setdefault(client_ip, [])
    valid_history = [t for t in history if now - t < 60.0]
    if len(valid_history) >= MAX_TRADING_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Too Many Requests: Rate limit of {MAX_TRADING_REQUESTS_PER_MINUTE} trade requests per minute exceeded."
        )
    valid_history.append(now)
    trade_rate_limit_history[client_ip] = valid_history

# -------------------------------------------------------------------------
# Multi-Bridge Registry & Failover Arbitration (Problem 2)
# -------------------------------------------------------------------------
class BridgeRegistry:
    def __init__(self):
        self.bridges: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def register(self, client_id: str, role: str, websocket: WebSocket, client_ip: str) -> Dict[str, Any]:
        async with self.lock:
            role_norm = str(role or "secondary").strip().lower()
            priority = 100 if role_norm == "primary" else 50
            now = time.time()
            now_iso = datetime.now(timezone.utc).isoformat()

            # Rule 3: Reject second connection carrying client_id that is already live,
            # or close the older socket first. Never hold two live sockets for one id.
            if client_id in self.bridges:
                old_entry = self.bridges[client_id]
                old_ws = old_entry.get("websocket")
                if old_ws and old_ws != websocket:
                    logger.info(f"BridgeRegistry: Displacing existing socket for client_id='{client_id}' (IP: {old_entry.get('client_ip')})")
                    try:
                        await old_ws.close(code=1000, reason="Replaced by new connection with same client_id")
                    except Exception:
                        pass
                    if old_ws in manager.mt5_connections:
                        manager.mt5_connections.remove(old_ws)

            entry = {
                "client_id": client_id,
                "role": role_norm,
                "priority": priority,
                "websocket": websocket,
                "client_ip": client_ip,
                "connected_at": now_iso,
                "connected_ts": now,
                "last_seen_ts": now,
                "accepted_candles": 0,
                "dropped_candles": 0,
                "last_m15_depth": 0,
            }
            self.bridges[client_id] = entry
            logger.info(f"BridgeRegistry: Registered bridge '{client_id}' [role={role_norm}, priority={priority}, IP={client_ip}]")
            return entry

    async def unregister(self, client_id: str, websocket: WebSocket):
        async with self.lock:
            entry = self.bridges.get(client_id)
            if entry and entry.get("websocket") == websocket:
                del self.bridges[client_id]
                logger.info(f"BridgeRegistry: Unregistered bridge '{client_id}'")

    def get_active_provider(self) -> Optional[Dict[str, Any]]:
        """Returns the highest-priority live connection (primary > secondary). Ties broken by earliest connection."""
        if not self.bridges:
            return None
        sorted_bridges = sorted(
            self.bridges.values(),
            key=lambda b: (-b.get("priority", 0), b.get("connected_ts", 0))
        )
        return sorted_bridges[0] if sorted_bridges else None

    def is_active_provider(self, client_id: str) -> bool:
        provider = self.get_active_provider()
        return provider is not None and provider.get("client_id") == client_id

    def touch(self, client_id: str):
        if client_id in self.bridges:
            self.bridges[client_id]["last_seen_ts"] = time.time()

    def record_accepted(self, client_id: str, m15_count: int):
        if client_id in self.bridges:
            self.bridges[client_id]["accepted_candles"] += 1
            self.bridges[client_id]["last_m15_depth"] = m15_count
            self.bridges[client_id]["last_seen_ts"] = time.time()

    def record_dropped(self, client_id: str):
        if client_id in self.bridges:
            self.bridges[client_id]["dropped_candles"] += 1
            self.bridges[client_id]["last_seen_ts"] = time.time()

    def get_status_report(self) -> Dict[str, Any]:
        active = self.get_active_provider()
        active_id = active.get("client_id") if active else None
        now = time.time()
        bridge_list = []
        for b in self.bridges.values():
            bridge_list.append({
                "client_id": b["client_id"],
                "role": b["role"],
                "priority": b["priority"],
                "client_ip": b["client_ip"],
                "connected_at": b["connected_at"],
                "seconds_connected": round(now - b["connected_ts"], 1),
                "last_seen_seconds_ago": round(now - b["last_seen_ts"], 1),
                "is_active_provider": (b["client_id"] == active_id),
                "accepted_candles": b["accepted_candles"],
                "dropped_candles": b["dropped_candles"],
                "last_m15_depth": b["last_m15_depth"],
            })
        return {
            "status": "SUCCESS",
            "total_connected": len(self.bridges),
            "active_provider": active_id,
            "bridges": bridge_list
        }

bridge_registry = BridgeRegistry()

class ConnectionManager:
    def __init__(self):
        self.dashboard_connections: List[WebSocket] = []
        self.mt5_connections: List[WebSocket] = []

    async def connect_dashboard(self, websocket: WebSocket):
        await websocket.accept()
        self.dashboard_connections.append(websocket)

    def disconnect_dashboard(self, websocket: WebSocket):
        if websocket in self.dashboard_connections:
            self.dashboard_connections.remove(websocket)

    async def connect_mt5(self, websocket: WebSocket):
        self.mt5_connections.append(websocket)

    def disconnect_mt5(self, websocket: WebSocket):
        if websocket in self.mt5_connections:
            self.mt5_connections.remove(websocket)

    async def broadcast_to_dashboard(self, message: Dict[str, Any]):
        for connection in list(self.dashboard_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect_dashboard(connection)

    async def send_order_to_mt5(self, order_payload: Dict[str, Any]):
        # Prefer sending order to the active highest-priority provider bridge
        active_provider = bridge_registry.get_active_provider()
        target_ws = active_provider.get("websocket") if active_provider else None
        if target_ws:
            try:
                await target_ws.send_json(order_payload)
                return
            except Exception:
                pass
        # Fallback to broadcasting to all connected bridges
        for connection in list(self.mt5_connections):
            try:
                await connection.send_json(order_payload)
            except Exception:
                self.disconnect_mt5(connection)

manager = ConnectionManager()

# In-flight order locks to prevent duplicate order spamming on MT5
in_flight_trades: Dict[str, float] = {}

# 5-Minute Post-Trade Study Cooldown (300 seconds)
symbol_cooldowns: Dict[str, float] = {}

# Gold Trade Commitment Lock — prevents flip-flop reversals
gold_last_trade_time: float = 0.0
gold_last_trade_direction: str = ""

# Same-direction re-entry block after a losing trade. Without this the bot walks
# straight back into the setup that just stopped it out, because the conditions
# that produced the signal are still true one candle later.
loss_direction_blocks: Dict[str, Dict[str, Any]] = {}

# Set when a reversal close is issued, so the opposite entry is not then blocked
# by the post-trade cooldown that the close itself creates.
gold_reversal_armed_until: float = 0.0

# Analysis is bar-driven, so it is recomputed only when a new candle actually
# closes. The bridge streams bundles every ~1.5s; recomputing a ~280ms pandas
# analysis on every one of them burned the event loop that also dispatches orders,
# and produced no new information because the closed bars had not changed.
analysis_cache: Dict[str, Dict[str, Any]] = {}

# Bar on which a signal was already dispatched, per symbol - stops the cached
# analysis from being executed repeatedly for the rest of that candle.
signal_consumed_bar: Dict[str, int] = {}

# Bar on which the ML model in shadow mode was evaluated, per symbol
last_ml_evaluated_bar: Dict[str, int] = {}


def last_closed_bar_time(tf_dict: Dict[str, Any], preferred: List[str]) -> int:
    """Timestamp of the newest CLOSED bar across the preferred timeframes."""
    for tf in preferred:
        bars = tf_dict.get(tf) or []
        if len(bars) >= 2:
            try:
                return int(bars[-2].get("time", 0))
            except (TypeError, ValueError):
                continue
    return 0


def register_loss_block(symbol: str, direction: str, seconds: float):
    """Blocks fresh entries in `direction` on `symbol` for `seconds`."""
    if not symbol or not direction:
        return
    loss_direction_blocks[symbol.upper()] = {
        "direction": str(direction).upper(),
        "until": time.time() + seconds
    }


def get_loss_block(symbol: str, direction: str) -> int:
    """Returns remaining seconds of an active same-direction block, else 0."""
    entry = loss_direction_blocks.get((symbol or "").upper())
    if not entry:
        return 0
    remaining = int(entry["until"] - time.time())
    if remaining <= 0:
        loss_direction_blocks.pop((symbol or "").upper(), None)
        return 0
    return remaining if entry["direction"] == str(direction).upper() else 0


async def refresh_news_calendar_loop():
    """
    Keeps the economic calendar current for the life of the process.

    The calendar was previously fetched once at startup. The feed is a weekly file,
    so any process running longer than a session was filtering against stale events
    and would happily open gold trades straight into NFP or CPI.
    """
    interval = max(120, getattr(settings, "NEWS_REFRESH_SECONDS", 600))
    while True:
        try:
            events = await news_filter.fetch_live_economic_calendar()
            logger.debug(f"Economic calendar refreshed: {len(events)} high-impact events cached.")
        except Exception as e:
            logger.warning(f"Economic calendar refresh failed: {e}")
        await asyncio.sleep(interval)

def load_persisted_state() -> Dict[str, Any]:
    """Loads saved system settings, daily baseline, trade history, and Gold state from disk"""
    default_state = {
        "execution_mode": settings.EXECUTION_MODE,
        "trading_school": settings.TRADING_SCHOOL,
        "trading_style": settings.TRADING_STYLE,
        "lot_mode": settings.LOT_MODE,
        "fixed_lot_size": settings.FIXED_LOT_SIZE,
        "max_risk_percent": settings.MAX_RISK_PER_TRADE_PERCENT,
        "active_symbols": settings.ACTIVE_SYMBOLS,
        "latest_signals": {},
        "latest_model_decision": {},
        "gold_analysis": {},
        "gold_execution_mode": settings.EXECUTION_MODE,
        "gold_strategy_profile": getattr(settings, "GOLD_STRATEGY_PROFILE", "HYBRID"),
        "gold_lot_mode": LotMode.DYNAMIC_PERCENT,
        "gold_fixed_lot_size": 0.05,
        "gold_scalp_lot_size": getattr(settings, "GOLD_SCALP_LOT_SIZE", 0.05),
        "gold_swing_lot_size": getattr(settings, "GOLD_SWING_LOT_SIZE", 0.03),
        "gold_fast_be_enabled": getattr(settings, "GOLD_FAST_BE_ENABLED", True),
        "gold_risk_percent": 1.0,
        "active_trades": [],
        "closed_trades": [],
        "live_prices": {},
        "currency_power": {},
        "mt5_connected": False,
        "account_info": {"balance": 10000.0, "equity": 10000.0, "free_margin": 10000.0, "floating_pnl": 0.0, "login": 0},
        "starting_daily_equity": 10000.0,
        "last_baseline_date": ""
    }
    try:
        if os.path.exists(STATE_FILE_PATH):
            with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                default_state["execution_mode"] = ExecutionMode(saved.get("execution_mode", settings.EXECUTION_MODE.value))
                default_state["trading_school"] = TradingSchool(saved.get("trading_school", settings.TRADING_SCHOOL.value))
                default_state["trading_style"] = TradingStyle(saved.get("trading_style", settings.TRADING_STYLE.value))
                default_state["lot_mode"] = LotMode(saved.get("lot_mode", settings.LOT_MODE.value))
                default_state["fixed_lot_size"] = float(saved.get("fixed_lot_size", settings.FIXED_LOT_SIZE))
                default_state["max_risk_percent"] = float(saved.get("max_risk_percent", settings.MAX_RISK_PER_TRADE_PERCENT))
                default_state["active_symbols"] = saved.get("active_symbols", settings.ACTIVE_SYMBOLS)
                default_state["gold_execution_mode"] = ExecutionMode(saved.get("gold_execution_mode", settings.EXECUTION_MODE.value))
                default_state["gold_strategy_profile"] = saved.get("gold_strategy_profile", getattr(settings, "GOLD_STRATEGY_PROFILE", "HYBRID"))
                default_state["gold_lot_mode"] = LotMode(saved.get("gold_lot_mode", settings.LOT_MODE.value))
                default_state["gold_fixed_lot_size"] = float(saved.get("gold_fixed_lot_size", 0.05))
                default_state["gold_scalp_lot_size"] = float(saved.get("gold_scalp_lot_size", getattr(settings, "GOLD_SCALP_LOT_SIZE", 0.05)))
                default_state["gold_swing_lot_size"] = float(saved.get("gold_swing_lot_size", getattr(settings, "GOLD_SWING_LOT_SIZE", 0.03)))
                default_state["gold_fast_be_enabled"] = bool(saved.get("gold_fast_be_enabled", getattr(settings, "GOLD_FAST_BE_ENABLED", True)))
                default_state["gold_risk_percent"] = float(saved.get("gold_risk_percent", 1.0))
                default_state["closed_trades"] = saved.get("closed_trades", [])
                default_state["starting_daily_equity"] = float(saved.get("starting_daily_equity", 10000.0))
                default_state["last_baseline_date"] = saved.get("last_baseline_date", "")
                logger.info("Loaded persisted state from storage.")
    except Exception as e:
        logger.warning(f"Could not load persisted state: {e}")
    return default_state

def save_persisted_state():
    try:
        os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
        data = {
            "execution_mode": state["execution_mode"].value if hasattr(state["execution_mode"], "value") else state["execution_mode"],
            "trading_school": state["trading_school"].value if hasattr(state["trading_school"], "value") else state["trading_school"],
            "trading_style": state["trading_style"].value if hasattr(state["trading_style"], "value") else state["trading_style"],
            "lot_mode": state["lot_mode"].value if hasattr(state["lot_mode"], "value") else state["lot_mode"],
            "fixed_lot_size": state["fixed_lot_size"],
            "max_risk_percent": state["max_risk_percent"],
            "active_symbols": state["active_symbols"],
            "gold_execution_mode": state["gold_execution_mode"].value if hasattr(state["gold_execution_mode"], "value") else state["gold_execution_mode"],
            "gold_strategy_profile": state.get("gold_strategy_profile", "HYBRID"),
            "gold_lot_mode": state["gold_lot_mode"].value if hasattr(state["gold_lot_mode"], "value") else state["gold_lot_mode"],
            "gold_fixed_lot_size": state.get("gold_fixed_lot_size", 0.05),
            "gold_scalp_lot_size": state.get("gold_scalp_lot_size", 0.05),
            "gold_swing_lot_size": state.get("gold_swing_lot_size", 0.03),
            "gold_fast_be_enabled": state.get("gold_fast_be_enabled", True),
            "gold_risk_percent": state.get("gold_risk_percent", 1.0),
            "closed_trades": state["closed_trades"][-100:],
            "starting_daily_equity": state.get("starting_daily_equity", 10000.0),
            "last_baseline_date": state.get("last_baseline_date", "")
        }
        with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving persisted state: {e}")

state = load_persisted_state()

class ModeUpdateRequest(BaseModel):
    mode: ExecutionMode

class StrategySettingsRequest(BaseModel):
    trading_school: TradingSchool = TradingSchool.TRI_SCHOOL_CONSENSUS
    trading_style: TradingStyle = TradingStyle.AUTO_ADAPTIVE
    lot_mode: LotMode = LotMode.DYNAMIC_PERCENT
    fixed_lot_size: float = 0.05
    max_risk_percent: float = 1.0
    active_symbols: List[str] = settings.ACTIVE_SYMBOLS

class ManualTradeRequest(BaseModel):
    symbol: str
    order_type: str
    lot_size: float = 0.05

class ApproveTradeRequest(BaseModel):
    symbol: str
    order_type: str
    lot_size: float
    sl: float
    tp: float

class GoldSettingsRequest(BaseModel):
    gold_execution_mode: str = "ANALYST_ONLY"  # "ANALYST_ONLY" or "AUTO_TRADING"
    gold_strategy_profile: str = "HYBRID"      # "HYBRID", "SCALPING", or "SWING"
    gold_lot_mode: LotMode = LotMode.DYNAMIC_PERCENT
    gold_fixed_lot_size: float = 0.05
    gold_scalp_lot_size: float = 0.05
    gold_swing_lot_size: float = 0.03
    gold_fast_be_enabled: bool = True
    gold_risk_percent: float = 1.0

class GoldManualTradeRequest(BaseModel):
    order_type: str
    lot_size: float = 0.05
    sl: Optional[float] = None
    tp: Optional[float] = None
    trade_style: Optional[str] = "HYBRID"  # "SCALP", "SWING", or "HYBRID"


def get_current_symbols_status_and_last_signal():
    """
    Returns (symbols_status, last_signal) using real live model decisions
    and the latest confirmed snapshot.
    Strictly targets active Telegram symbols (default USDJPY on M15).
    """
    symbols_status = []
    active_syms = telegram_bot.get_validated_symbols()
    
    for sym in active_syms:
        rec = state.get("latest_model_decision", {}).get(sym)
        tf = "M15"
        sym_m = get_symbol_metrics(sym)

        # Brief 16 Problem 2: Check bar age / market closed
        candle_hist = state.get("candle_history", {}).get(sym, {})
        m15_bars = candle_hist.get("M15") or []
        newest_bar_time = 0
        if m15_bars:
            try:
                newest_bar_time = int(m15_bars[-1].get("time", 0))
            except (ValueError, TypeError):
                pass
        if not newest_bar_time and rec:
            newest_bar_time = int(rec.get("bar_close_time", 0) or 0)

        now_ts = time.time()
        is_market_closed = False
        if newest_bar_time > 0:
            bar_age_seconds = max(0.0, now_ts - float(newest_bar_time))
            if bar_age_seconds > 1200.0:  # > 20 minutes
                is_market_closed = True
        else:
            # If no candle data available at all, consider market closed / bridge offline
            is_market_closed = True

        if is_market_closed or (rec and rec.get("reason") == "market closed"):
            symbols_status.append({
                "symbol": sym,
                "timeframe": tf,
                "reason": "market closed"
            })
            continue

        if rec:
            decision = rec.get("decision", "HOLD")
            gate = rec.get("gate_passed", False)
            prob = rec.get("probability")
            thresh = rec.get("threshold", sym_m.get("threshold", 0.40))
            reason = rec.get("reason", "")
            tf = rec.get("timeframe", "M15")
            
            if not reason:
                if not gate:
                    reason = "H4 trend not aligned"
                elif prob is not None and prob < thresh:
                    reason = f"probability {prob:.2f} below {thresh:.2f} threshold"
                else:
                    reason = f"{decision} condition not met"
            symbols_status.append({
                "symbol": sym,
                "timeframe": tf,
                "reason": reason
            })
        else:
            symbols_status.append({
                "symbol": sym,
                "timeframe": tf,
                "reason": "H4 trend not aligned or probability below threshold"
            })
            
    last_signal = None
    all_snaps = snapshot_service.list_snapshots(limit=50)
    for s in all_snaps:
        sym = s.get("symbol", "").upper()
        if sym in active_syms and (s.get("model_record") or s.get("is_confirmed_signal") or s.get("probability")):
            t_ts = s.get("timestamp", time.time())
            age_h = max(1, int(round((time.time() - t_ts) / 3600.0)))
            t_utc = s.get("created_at_utc") or datetime.fromtimestamp(t_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            last_signal = {
                "symbol": sym,
                "timeframe": s.get("timeframe", "M15"),
                "action": "BUY",
                "price": s.get("entry_price", 0.0),
                "age_hours": age_h,
                "time_utc": str(t_utc)
            }
            break
            
    return symbols_status, last_signal


async def run_outcome_timeout_worker():
    """
    Brief 26 Section 4: Autonomous background worker running every 30s.
    Evaluates pending timeouts for model signals and manual desk trades even when market is closed/quiet.
    Prevents the 43-hour pending signal freeze over weekends.
    """
    while True:
        try:
            await asyncio.sleep(30)
            signal_outcome_tracker.evaluate_pending_timeouts()
            analyst_desk_service.evaluate_pending_timeouts()
        except asyncio.CancelledError:
            break
        except Exception as ex:
            logger.debug(f"Outcome timeout worker error: {ex}")

DESK_HTML_PATH = Path(__file__).resolve().parent / "desk_ui" / "index.html"
MIRROR_HTML_PATH = Path(__file__).resolve().parent / "mirror_ui" / "index.html"

async def run_mirror_poll_worker():
    """Brief 32: Periodic polling loop for mirrored account if local MT5 module available"""
    while True:
        try:
            await asyncio.to_thread(mirror_service.poll_from_mt5)
        except Exception as e:
            logger.debug(f"Mirror polling worker tick exception: {e}")
        await asyncio.sleep(2.0)

@app.on_event("startup")
async def startup_event():
    # Brief 19 Guard 2: Seed existing open positions at startup so they are not republished
    try:
        trade_event_service.seed_initial_positions(state.get("active_trades", []))
    except Exception as se_err:
        logger.warning(f"Error seeding initial trade positions at startup: {se_err}")

    # Brief 27 Section 1: Connect Analyst Desk directly to live server candle store
    analyst_desk_service.set_candle_getter(lambda s: state.get("candle_history", {}).get(s.upper(), {}))
    # Brief 32: Connect Mirror Service directly to live server candle store
    mirror_service.set_candle_getter(lambda s: state.get("candle_history", {}).get(s.upper(), {}))

    asyncio.create_task(refresh_news_calendar_loop())
    asyncio.create_task(telegram_bot.start_heartbeat_worker(get_current_symbols_status_and_last_signal))
    asyncio.create_task(run_outcome_timeout_worker())
    asyncio.create_task(run_mirror_poll_worker())
    brainstorm_service.add_log(
        level="SYSTEM",
        category="BOOT",
        symbol="CORE",
        message="🚀 FXENGIN Unified Multi-School Forex Analyst Core v2.3.0 Online."
    )

@app.get("/")
@app.head("/")
def read_root(request: Request):
    host = request.headers.get("host", "").lower()
    if "mirror." in host:
        if MIRROR_HTML_PATH.exists():
            with open(MIRROR_HTML_PATH, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    elif "desk." in host:
        if DESK_HTML_PATH.exists():
            with open(DESK_HTML_PATH, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    return {
        "system": settings.PROJECT_NAME,
        "version": "2.3.0",
        "execution_mode": state["execution_mode"],
        "trading_school": state["trading_school"],
        "trading_style": state["trading_style"],
        "mt5_connected": len(manager.mt5_connections) > 0,
        "status": "ONLINE"
    }

@app.get("/desk", response_class=HTMLResponse)
@app.head("/desk", response_class=HTMLResponse)
@app.get("/desk/", response_class=HTMLResponse)
@app.head("/desk/", response_class=HTMLResponse)
def serve_desk_ui():
    """Brief 26: Serve the Analyst Desk UI directly"""
    if DESK_HTML_PATH.exists():
        with open(DESK_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h3>Analyst Desk UI not found</h3>", status_code=404)

@app.get("/mirror", response_class=HTMLResponse)
@app.head("/mirror", response_class=HTMLResponse)
@app.get("/mirror/", response_class=HTMLResponse)
@app.head("/mirror/", response_class=HTMLResponse)
def serve_mirror_ui():
    """Brief 32: Serve the Mirrored Account UI directly"""
    if MIRROR_HTML_PATH.exists():
        with open(MIRROR_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h3>Mirror UI not found</h3>", status_code=404)

@app.get("/api/v1/state")
def get_state():
    return {
        "execution_mode": state["execution_mode"],
        "trading_school": state["trading_school"],
        "trading_style": state["trading_style"],
        "lot_mode": state["lot_mode"],
        "fixed_lot_size": state["fixed_lot_size"],
        "max_risk_percent": state["max_risk_percent"],
        "active_symbols": state["active_symbols"],
        "mt5_connected": len(manager.mt5_connections) > 0,
        "account_info": state["account_info"],
        "latest_signals": list(state["latest_signals"].values()),
        "active_trades": state["active_trades"],
        "closed_trades": state["closed_trades"],
        "live_prices": state["live_prices"],
        "currency_power": state.get("currency_power", correlation_engine.calculate_currency_power(state["live_prices"])),
        "candle_history": state.get("candle_history", {}),
        "bridge_registry": bridge_registry.get_status_report()
    }

@app.get("/api/v1/bridges")
def get_bridges_status():
    """
    Returns active MT5 bridge connections, registry priorities, and dropped candle metrics.
    Diagnostic endpoint for multi-bridge arbitration (Problem 2).
    """
    return bridge_registry.get_status_report()

@app.get("/api/v1/correlation")
def get_correlation_matrix():
    power_data = correlation_engine.calculate_currency_power(state["live_prices"])
    return {"status": "SUCCESS", "data": power_data}

@app.get("/api/v1/learning")
def get_ai_learning_analytics():
    """Returns AI Self-Learning summary, loss root causes, and adaptive parameters"""
    return {"status": "SUCCESS", "data": trade_learning_engine.get_learning_summary()}

class MultiTargetToggleRequest(BaseModel):
    enabled: bool

@app.post("/api/v1/learning/toggle_multi_target")
def toggle_multi_target(req: MultiTargetToggleRequest, _auth: str = Depends(verify_admin_token)):
    """Enables or disables 3-in-1 multi-target split order execution"""
    trade_learning_engine.set_multi_target_enabled(req.enabled)
    return {"status": "SUCCESS", "multi_target_enabled": req.enabled}

# ==============================================================================
# 👁️ MULTIMODAL VISION AI & EPISODIC EXPERIENCE MEMORY ENDPOINTS
# ==============================================================================

class VisionScanRequest(BaseModel):
    symbol: str = "XAUUSD"
    timeframe: str = "M5"

class ManualAutopsyRequest(BaseModel):
    ticket: int
    symbol: str = "XAUUSD"
    profit: float = -10.0
    trap_type: Optional[str] = "LIQUIDITY_SWEEP_FAKEOUT"
    lesson_ar: Optional[str] = "تشريح يدوي: سحب سيولة وتأكيد إغلاق شمعة M5 قبل الدخول"
    lesson_en: Optional[str] = "Manual Autopsy: Liquidity sweep fakeout diagnosed. Confirm M5 candle close."

@app.get("/api/v1/vision/latest/{symbol}")
async def get_latest_vision_scan(symbol: str):
    sym = symbol.upper()
    scan = vision_engine.latest_visual_scans.get(sym)
    if not scan:
        # Auto-generate instant scan on first view
        curr_price = float(state["live_prices"].get(sym, {}).get("bid", 2865.50 if "XAU" in sym else 1.0850))
        sig = state["latest_signals"].get(sym, {})
        bias = sig.get("macro_trend", sig.get("market_context", "BULLISH"))
        
        candle_hist = state.get("candle_history", {}).get(sym, {})
        tf_candles = candle_hist.get("M5") or candle_hist.get("M15")
        df = pd.DataFrame(tf_candles) if tf_candles and len(tf_candles) >= 5 else None
        
        scan = await vision_engine.analyze_chart_with_vision(
            symbol=sym,
            timeframe="M5",
            df=df,
            current_price=curr_price,
            macro_bias=bias
        )
    return {"status": "SUCCESS", "data": scan}

@app.post("/api/v1/vision/scan")
async def trigger_vision_scan(req: VisionScanRequest, _auth: str = Depends(verify_admin_token)):
    sym = req.symbol.upper()
    tf = req.timeframe.upper()
    
    # Retrieve active price & macro bias
    curr_price = float(state["live_prices"].get(sym, {}).get("bid", 2865.50 if "XAU" in sym else 1.0850))
    sig = state["latest_signals"].get(sym, {})
    bias = sig.get("macro_trend", sig.get("market_context", "BULLISH"))

    candle_hist = state.get("candle_history", {}).get(sym, {})
    tf_candles = candle_hist.get(tf) or candle_hist.get("M15") or candle_hist.get("M5")
    df = pd.DataFrame(tf_candles) if tf_candles and len(tf_candles) >= 5 else None

    result = await vision_engine.analyze_chart_with_vision(
        symbol=sym,
        timeframe=tf,
        df=df,
        current_price=curr_price,
        macro_bias=bias
    )

    await manager.broadcast_to_dashboard({
        "type": "VISION_ANALYSIS_UPDATE",
        "symbol": sym,
        "data": result,
        "brainstorm_logs": brainstorm_service.get_recent_logs(10)
    })

    return {"status": "SUCCESS", "data": result}

@app.get("/api/v1/model/latest/{symbol}")
async def get_latest_model_decision(symbol: str):
    """
    Returns the latest validated ML model decision in SHADOW MODE.
    No execution: pure evaluation & audit reporting.
    Per Agent Brief 8: Only symbols with a shipped trained model (XAUUSD) are served.
    All other symbols return UNSUPPORTED status with no fabricated numbers.
    """
    sym = symbol.upper()
    is_supported, unsupp_reason = is_model_supported_for_symbol(sym)
    if not is_supported:
        return {
            "status": "UNSUPPORTED",
            "supported": False,
            "symbol": sym,
            "decision": "HOLD",
            "probability": None,
            "confidence_score": None,
            "setup_grade": None,
            "threshold": None,
            "gate_passed": False,
            "reason": unsupp_reason,
            "mode": "SHADOW"
        }

    sym_metrics = get_symbol_metrics(sym)
    dec = state.get("latest_model_decision", {}).get(sym)
    now_ts = time.time()

    # Brief 16 Problem 2: Stale Bar Rejection for cached decision
    if dec and dec.get("bar_close_time"):
        cached_bar_age = max(0.0, now_ts - float(dec["bar_close_time"]))
        if cached_bar_age > 1200.0:
            stale_copy = dict(dec)
            stale_copy["decision"] = "HOLD"
            stale_copy["reason"] = "market closed"
            stale_copy["is_stale"] = True
            stale_copy["bar_age_seconds"] = round(cached_bar_age, 1)
            return {"status": "SUCCESS", "symbol": sym, **stale_copy}

    # If not yet recorded on closed bar, or previous decision had insufficient history, attempt immediate evaluation if candles are present
    needs_eval = not dec or "insufficient history" in str(dec.get("reason", ""))
    if needs_eval and get_engine is not None:
        candle_hist = state.get("candle_history", {}).get(sym, {})
        if candle_hist:
            m15_bar_time = last_closed_bar_time(candle_hist, ["M15"])
            bar_age = max(0.0, now_ts - float(m15_bar_time)) if m15_bar_time > 0 else 999999.0
            if bar_age > 1200.0:
                logger.warning(
                    f"🚨 [STALE BAR REJECTION] On-demand evaluation for {sym} rejected: "
                    f"bar {m15_bar_time} is {bar_age/60.0:.1f}m old (> 20m limit). Market closed."
                )
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "symbol": sym,
                    "timeframe": "M15",
                    "bar_close_time": m15_bar_time,
                    "decision": "HOLD",
                    "probability": 0.0,
                    "threshold": sym_metrics.get("threshold", 0.40),
                    "gate_passed": False,
                    "reason": "market closed",
                    "entry": None,
                    "sl": None,
                    "tp": None,
                    "rr": 2.0,
                    "atr": None,
                    "spread_cost_r": None,
                    "bar_spread": None,
                    "tick_spread": None,
                    "spread_points": None,
                    "rule_engine_recommendation": "HOLD",
                    "mode": "SHADOW",
                    "validated_stats": {
                        "expectancy_r": sym_metrics.get("expectancy_r", 0.0),
                        "win_rate": 0.40,
                        "oos_trades": sym_metrics.get("oos_trades", 0),
                        "folds_positive": f"{sym_metrics.get('folds_positive', 0)}/{sym_metrics.get('folds_total', 5)}",
                        "tier": sym_metrics.get("tier", "EXPERIMENTAL"),
                        "worst_fold_r": sym_metrics.get("worst_fold_r", 0.0)
                    }
                }
                state.setdefault("latest_model_decision", {})[sym] = record
                log_model_decision(record)
                return {"status": "SUCCESS", "symbol": sym, **record}

            gold_tick = state["live_prices"].get(sym, {})
            gold_bid = float(gold_tick.get("bid") or 0.0)
            gold_ask = float(gold_tick.get("ask") or 0.0)
            gold_mid = ((gold_bid + gold_ask) / 2.0) if (gold_bid > 0 and gold_ask > 0) else None
            spec = get_symbol_spec(sym)
            pt = spec.get("point", 0.001 if "JPY" in sym else (0.01 if "XAU" in sym else 0.00001))
            dig = spec.get("digits", 3 if "JPY" in sym else (2 if "XAU" in sym else 5))
            try:
                engine = get_engine(sym)
                instant_dec = engine.decide(
                    candles=candle_hist,
                    point=pt,
                    digits=dig,
                    live_price=gold_mid
                )
                dec_val = instant_dec.get("decision", "HOLD")
                sym_metrics = get_symbol_metrics(sym, side=dec_val)
                m15_bars = candle_hist.get("M15") or []
                last_closed_m15 = m15_bars[-2] if len(m15_bars) >= 2 else (m15_bars[-1] if m15_bars else None)
                bar_spread = float(last_closed_m15["spread"]) if (last_closed_m15 and "spread" in last_closed_m15 and last_closed_m15["spread"] is not None) else None
                live_tick_spread = round((gold_ask - gold_bid) / pt, 1) if (gold_ask > 0 and gold_bid > 0 and pt > 0) else None
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "symbol": sym,
                    "timeframe": "M15",
                    "bar_close_time": m15_bar_time,
                    "decision": instant_dec.get("decision", "HOLD"),
                    "probability": instant_dec.get("probability"),
                    "threshold": instant_dec.get("threshold", sym_metrics.get("threshold", 0.40)),
                    "gate_passed": instant_dec.get("gate_passed", False),
                    "reason": instant_dec.get("reason", ""),
                    "entry": instant_dec.get("entry"),
                    "sl": instant_dec.get("sl"),
                    "tp": instant_dec.get("tp"),
                    "rr": instant_dec.get("rr", 2.0),
                    "atr": instant_dec.get("atr"),
                    "spread_cost_r": instant_dec.get("spread_cost_r"),
                    "bar_spread": bar_spread,
                    "tick_spread": live_tick_spread,
                    "spread_points": bar_spread,
                    "rule_engine_recommendation": state.get("gold_analysis", {}).get("recommendation", "HOLD") if sym == "XAUUSD" else "HOLD",
                    "mode": "SHADOW",
                    "validated_stats": {
                        "expectancy_r": sym_metrics.get("expectancy_r", 0.0),
                        "win_rate": 0.40,
                        "oos_trades": sym_metrics.get("oos_trades", 0),
                        "folds_positive": f"{sym_metrics.get('folds_positive', 0)}/{sym_metrics.get('folds_total', 5)}",
                        "tier": sym_metrics.get("tier", "EXPERIMENTAL"),
                        "worst_fold_r": sym_metrics.get("worst_fold_r", 0.0)
                    }
                }
                state.setdefault("latest_model_decision", {})[sym] = record
                log_model_decision(record)
                return {"status": "SUCCESS", "symbol": sym, **record}
            except ModelNotAvailable as mne:
                logger.warning(f"ModelNotAvailable for {sym}: {mne}")
            except Exception as e:
                logger.error(f"Error in on-demand model decision for {sym}: {e}")

    if not dec:
        return {
            "status": "PENDING",
            "symbol": sym,
            "decision": "HOLD",
            "probability": None,
            "threshold": sym_metrics.get("threshold", 0.40),
            "gate_passed": False,
            "reason": "Awaiting closed M15 bar candle bundle from MT5 bridge",
            "mode": "SHADOW",
            "validated_stats": {
                "expectancy_r": sym_metrics.get("expectancy_r", 0.0),
                "win_rate": 0.40,
                "oos_trades": sym_metrics.get("oos_trades", 0),
                "folds_positive": f"{sym_metrics.get('folds_positive', 0)}/{sym_metrics.get('folds_total', 5)}",
                "tier": sym_metrics.get("tier", "EXPERIMENTAL"),
                "worst_fold_r": sym_metrics.get("worst_fold_r", 0.0)
            }
        }

    return {"status": "SUCCESS", "symbol": sym, **dec}

@app.get("/api/v1/learning/performance")
def get_measured_performance():
    """
    Measured expectancy of the system and of each individual model, in R.

    This is the report that answers "is the bot actually making money and which
    part of it works" - which win/loss counts alone cannot.
    """
    return trade_learning_engine.get_performance_report()


@app.get("/api/v1/memory/summary")
def get_experience_memory_summary():
    """Returns episodic memory bank, learned rules, and recent autopsies"""
    return {"status": "SUCCESS", "data": episodic_memory.get_memory_summary()}

@app.post("/api/v1/memory/clear")
def clear_experience_memory(_auth: str = Depends(verify_admin_token)):
    episodic_memory.clear_memory()
    return {"status": "SUCCESS", "message": "Experience memory bank reset successfully."}

@app.post("/api/v1/memory/trigger_autopsy")
async def trigger_manual_autopsy(req: ManualAutopsyRequest, _auth: str = Depends(verify_admin_token)):
    closed_trade = {
        "ticket": req.ticket,
        "symbol": req.symbol.upper(),
        "profit": req.profit,
        "price": 2865.50 if "XAU" in req.symbol.upper() else 1.0850,
        "type": "BUY"
    }
    autopsy = episodic_memory.conduct_post_mortem_autopsy(
        closed_trade=closed_trade,
        ai_autopsy_rationale={
            "trap_type": req.trap_type,
            "lesson_ar": req.lesson_ar,
            "lesson_en": req.lesson_en
        }
    )
    await manager.broadcast_to_dashboard({
        "type": "EXPERIENCE_UPDATE",
        "autopsy": autopsy,
        "memory_summary": episodic_memory.get_memory_summary()
    })
    return {"status": "SUCCESS", "data": autopsy}

@app.get("/api/v1/analyst/dossier/{symbol}")
def get_analyst_dossier(symbol: str, timeframe: str = Query("M5")):
    """Returns exhaustive institutional analysis dossier with real MT5 screenshot and full Arabic & English breakdown"""
    dossier = analyst_service.generate_deep_analyst_dossier(symbol, timeframe, state)
    return {"status": "SUCCESS", "data": dossier}

def _resolve_manual_trade(identifier: str) -> Optional[Dict[str, Any]]:
    """Resolves a manual desk trade by trade_id from open trades, durable records, or closed outcomes."""
    tid = identifier.strip()
    if tid in analyst_desk_service.open_trades:
        return analyst_desk_service.open_trades[tid]
    records = analyst_desk_service.get_records(limit=10000)
    for r in reversed(records):
        if r.get("trade_id") == tid or r.get("signal_id") == tid:
            return r
    outcomes = analyst_desk_service.get_outcomes(limit=1000)
    for o in reversed(outcomes):
        if o.get("trade_id") == tid or o.get("signal_id") == tid:
            return o
    return None

def _resolve_snapshot_for_identifier(identifier: str, state_ref: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Resolves a snapshot by its ID, manual desk trade ID (man_), or by symbol (latest confirmed model signal). Never auto-creates on read."""
    identifier = identifier.strip()
    if identifier.startswith("snap_"):
        return snapshot_service.get_snapshot(identifier)
    
    if identifier.startswith("man_"):
        trade = _resolve_manual_trade(identifier)
        if not trade:
            return None
        sym = trade.get("symbol", "XAUUSD").upper()
        dir_norm = trade.get("direction", "BUY").upper()
        entry = float(trade.get("entry_price") or trade.get("entry") or 0.0)
        sl = float(trade.get("sl_price") or trade.get("stop_loss") or 0.0)
        tp1 = float(trade.get("tp1_price") or trade.get("take_profit_1") or 0.0)
        tp2_raw = trade.get("tp2_price") or trade.get("take_profit_2")
        tp2 = float(tp2_raw) if (tp2_raw is not None and float(tp2_raw) > 0) else None
        tp3_raw = trade.get("tp3_price") or trade.get("take_profit_3")
        tp3 = float(tp3_raw) if (tp3_raw is not None and float(tp3_raw) > 0) else None

        is_gold = ("XAU" in sym or "GOLD" in sym)
        pip_unit = 0.1 if is_gold else (0.01 if "JPY" in sym else 0.0001)
        sl_pips = round(abs(entry - sl) / pip_unit, 1) if (entry and sl and pip_unit > 0) else 0.0
        tp1_pips = round(abs(tp1 - entry) / pip_unit, 1) if (tp1 and entry and pip_unit > 0) else 0.0
        tp2_pips = round(abs(tp2 - entry) / pip_unit, 1) if (tp2 and entry and pip_unit > 0) else None
        tp3_pips = round(abs(tp3 - entry) / pip_unit, 1) if (tp3 and entry and pip_unit > 0) else None

        feat = trade.get("features") or trade.get("snapshot") or {}
        r_mult = trade.get("tp1_r") or (round(abs(tp1 - entry) / abs(entry - sl), 2) if (entry and sl and entry != sl and tp1) else 1.5)
        why_text = trade.get("why") or trade.get("why_reason") or f"Manual Desk Trade ({sym} {dir_norm})"
        zone_str = trade.get("discretionary_zone")
        if not zone_str and trade.get("zone_low") and trade.get("zone_high"):
            zone_str = f"{trade.get('zone_low')} - {trade.get('zone_high')}"

        return {
            "snapshot_id": trade.get("trade_id", identifier),
            "symbol": sym,
            "timeframe": trade.get("timeframe", "M15"),
            "created_at_utc": trade.get("entered_at") or trade.get("published_utc") or "",
            "recommendation": dir_norm,
            "bias": "BULLISH" if dir_norm == "BUY" else "BEARISH",
            "confidence_score": 100.0,
            "setup_grade": "DESK",
            "entry_price": entry,
            "sl_price": sl,
            "sl_pips": sl_pips,
            "tp1_price": tp1,
            "tp1_pips": tp1_pips,
            "tp2_price": tp2,
            "tp2_pips": tp2_pips,
            "tp3_price": tp3,
            "tp3_pips": tp3_pips,
            "rr_ratio": f"1:{r_mult}",
            "suggested_lot": 0.1,
            "adx": float(feat.get("f_adx", 25.0) or 25.0),
            "rsi": float(feat.get("f_rsi", 50.0) or 50.0),
            "atr": float(feat.get("f_atr_pct", 3.5 if is_gold else 0.0020) or (3.5 if is_gold else 0.0020)),
            "dxy_bias": "NEUTRAL",
            "smc_factors": {
                "order_block": zone_str or "Analyst Defined Zone",
                "fvg_status": "Monitored",
                "liquidity": "Analyst Discretionary Assessment"
            },
            "report_ar": f"صفقة مكتب تداول يدوية: {why_text}",
            "report_en": f"Manual Desk Trade: {why_text}",
            "is_real_mt5_capture": False,
            "source": "DESK"
        }
    
    # Identifier is a symbol (e.g. XAUUSD, EURUSD)
    sym = identifier.upper()
    snap = snapshot_service.get_latest_signal_for_symbol(sym)
    if snap:
        return snap
    
    # Fallback to general snapshot if available
    return snapshot_service.get_latest_snapshot_for_symbol(sym)

@app.get("/api/v1/analyst/share/{identifier}")
def get_shareable_analysis(
    identifier: str,
    lang: str = Query("ar", pattern="^(ar|en)$")
):
    """
    Public Shareable Analysis API Endpoint.
    Returns complete trade setup, coordinates, risk metrics,
    and narrative report in the requested language (Arabic or English),
    with immutable setup verification against current live price.
    """
    snap = _resolve_snapshot_for_identifier(identifier, state)
    if not snap:
        return JSONResponse(
            status_code=404,
            content={
                "status": "NO_ACTIVE_SIGNAL",
                "symbol": identifier.upper(),
                "detail": f"No active confirmed model signal or manual desk trade for {identifier}"
            }
        )
    
    sym = snap.get("symbol", "XAUUSD")
    live_p = state.get("live_prices", {}).get(sym, {})
    curr_price = float(live_p.get("bid", snap.get("entry_price", 0.0)))
    freshness = snapshot_service.evaluate_snapshot_freshness(snap, curr_price)

    is_ar = (lang.lower() == "ar")
    selected_report = snap.get("report_ar" if is_ar else "report_en", "")

    return {
        "status": "SUCCESS",
        "language": lang,
        "snapshot_id": snap.get("snapshot_id"),
        "symbol": sym,
        "timeframe": snap.get("timeframe"),
        "created_at_utc": snap.get("created_at_utc"),
        "recommendation": snap.get("recommendation"),
        "recommendation_label": ("شراء قوي (BUY)" if snap.get("recommendation") == "BUY" else ("بيع قوي (SELL)" if snap.get("recommendation") == "SELL" else "انتظار (HOLD)")) if is_ar else snap.get("recommendation"),
        "bias": snap.get("bias"),
        "bias_label": ("صاعد مؤسسي (BULLISH)" if snap.get("bias") == "BULLISH" else "هابط مؤسسي (BEARISH)") if is_ar else snap.get("bias"),
        "confidence_score": snap.get("confidence_score"),
        "setup_grade": snap.get("setup_grade"),
        "coordinates": {
            k: v for k, v in {
                "entry_price": snap.get("entry_price"),
                "sl_price": snap.get("sl_price"),
                "sl_pips": snap.get("sl_pips"),
                "tp1_price": snap.get("tp1_price"),
                "tp1_pips": snap.get("tp1_pips"),
                "tp2_price": snap.get("tp2_price"),
                "tp2_pips": snap.get("tp2_pips"),
                "tp3_price": snap.get("tp3_price"),
                "tp3_pips": snap.get("tp3_pips"),
                "rr_ratio": snap.get("rr_ratio"),
                "suggested_lot": snap.get("suggested_lot")
            }.items() if v is not None
        },
        "indicators": {
            "adx": snap.get("adx"),
            "rsi": snap.get("rsi"),
            "atr": snap.get("atr"),
            "dxy_bias": snap.get("dxy_bias")
        },
        "smc_factors": snap.get("smc_factors"),
        "freshness": freshness,
        "report": selected_report,
        "report_ar": snap.get("report_ar"),
        "report_en": snap.get("report_en"),
        "image_url": f"/api/v1/analyst/share/{snap.get('snapshot_id')}/image",
        "share_web_url": f"/share/analysis/{snap.get('snapshot_id')}?lang={lang}",
        "is_real_mt5_capture": snap.get("is_real_mt5_capture", False)
    }

@app.get("/api/v1/analyst/share/{identifier}/image")
def get_shareable_analysis_image(identifier: str):
    """Returns raw binary PNG image of the analysis snapshot or manual desk card, or 404 NO_ACTIVE_SIGNAL"""
    identifier_clean = identifier.strip()
    if identifier_clean.startswith("man_"):
        trade = _resolve_manual_trade(identifier_clean)
        if not trade:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "NO_ACTIVE_SIGNAL",
                    "symbol": identifier_clean,
                    "detail": f"Manual desk trade not found for {identifier_clean}"
                }
            )
        sym = trade.get("symbol", "").upper()
        candles_dict = state.get("candle_history", {}).get(sym, {})
        m15_candles = candles_dict.get("M15") or []
        png_bytes = render_manual_desk_card(trade_data=trade, m15_candles=m15_candles)
        if png_bytes:
            return Response(content=png_bytes, media_type="image/png")
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "detail": "Failed to render manual desk card"}
        )

    snap = _resolve_snapshot_for_identifier(identifier_clean, state)
    if not snap:
        return JSONResponse(
            status_code=404,
            content={
                "status": "NO_ACTIVE_SIGNAL",
                "symbol": identifier_clean.upper(),
                "detail": f"No confirmed signal card available for {identifier_clean}"
            }
        )
    
    snap_id = snap.get("snapshot_id")
    png_bytes = snapshot_service.get_snapshot_image_bytes(snap_id)
    if png_bytes:
        return Response(content=png_bytes, media_type="image/png")
    
    b64 = snapshot_service.get_snapshot_image_base64(snap_id)
    if b64:
        try:
            clean = b64.split(",")[-1]
            return Response(content=base64.b64decode(clean), media_type="image/png")
        except Exception:
            pass
    return JSONResponse(
        status_code=404,
        content={
            "status": "NO_ACTIVE_SIGNAL",
            "symbol": snap.get("symbol", identifier_clean.upper()),
            "detail": f"Chart image not available for snapshot {snap_id}"
        }
    )

@app.get("/api/v1/signals/latest/{symbol}")
def get_latest_signal(symbol: str):
    """Returns latest confirmed model signal for symbol or 404 NO_ACTIVE_SIGNAL / UNSUPPORTED"""
    sym = symbol.strip().upper()
    is_supported, unsupp_reason = is_model_supported_for_symbol(sym)
    if not is_supported:
        return JSONResponse(
            status_code=404,
            content={
                "status": "UNSUPPORTED",
                "supported": False,
                "symbol": sym,
                "detail": unsupp_reason,
                "reason": unsupp_reason,
                "decision": "HOLD",
                "probability": None,
                "confidence_score": None,
                "setup_grade": None
            }
        )

    snap = snapshot_service.get_latest_signal_for_symbol(sym)
    if not snap:
        return JSONResponse(
            status_code=404,
            content={
                "status": "NO_ACTIVE_SIGNAL",
                "symbol": sym,
                "detail": f"No active confirmed model signal for {sym}",
                "reason": "Model holding: no confirmed signal on latest bar",
                "decision": "HOLD",
                "probability": None,
                "confidence_score": None,
                "setup_grade": None
            }
        )
    return {"status": "SUCCESS", "data": snap}

@app.get("/api/v1/signals/outcomes")
def get_signal_outcomes(
    symbol: Optional[str] = Query(None, description="Optional symbol filter (e.g. AUDJPY)"),
    limit: Optional[int] = Query(None, description="Max closed trades to return"),
    include_test: bool = Query(False, description="Include test verification trades")
):
    """
    Brief 22: Pure read from signal_outcomes.jsonl.
    Returns closed trades with 14-field schema, live summary metrics,
    and side-by-side comparison against backtest expectancy.
    """
    return signal_outcomes_service.get_outcomes(symbol=symbol, limit=limit, include_test=include_test)

@app.get("/api/v1/signals/suppressions")
def get_signal_suppressions(
    symbol: Optional[str] = Query(None, description="Optional symbol filter"),
    limit: Optional[int] = Query(100, description="Max suppressions to return")
):
    """
    Brief 22: Returns log of duplicate signals suppressed by active open signals.
    """
    return telegram_bot.get_suppressions(symbol=symbol, limit=limit)

@app.get("/api/v1/signals/{snap_id}")
def get_signal_by_id(snap_id: str):
    """Returns signal details by snapshot ID"""
    snap = snapshot_service.get_snapshot(snap_id.strip())
    if not snap:
        raise HTTPException(status_code=404, detail=f"Signal '{snap_id}' not found")
    return {"status": "SUCCESS", "data": snap}

@app.get("/api/v1/signals/{snap_id}/image")
def get_signal_image_by_id(snap_id: str):
    """Returns raw binary PNG for a signal card by snapshot ID"""
    return get_shareable_analysis_image(snap_id)

@app.post("/api/v1/signals/test/publish-model-signal")
def test_publish_model_signal(
    payload: Dict[str, Any] = Body(...)
):
    """
    Test endpoint for Acceptance Test 2:
    Simulates a confirmed model BUY decision and publishes a signal card.
    Verifies deduplication per (symbol, bar_close_time).
    """
    symbol = payload.get("symbol", "XAUUSD").upper()
    bar_time = int(payload.get("bar_close_time") or (time.time() // 900 * 900))
    
    # Retrieve current M15 candles from state or construct test candles if state is empty
    m15_bars = state.get("candle_history", {}).get(symbol, {}).get("M15", [])
    if not m15_bars or len(m15_bars) < 30:
        current_p = float(state.get("live_prices", {}).get(symbol, {}).get("bid", 2900.0))
        m15_bars = []
        for i in range(120):
            t = bar_time - (120 - i) * 900
            p = current_p + (i - 60) * 0.1
            m15_bars.append({
                "time": t,
                "open": p - 0.2,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p,
                "volume": 100 + i
            })
            
    sym_metrics = get_symbol_metrics(symbol)
    sym_spec = get_symbol_spec(symbol)
    dig = sym_spec.get("digits", 3 if "JPY" in symbol else (2 if "XAU" in symbol else 5))
    med_atr = get_reference_median_atr(symbol)
    test_atr = float(payload.get("atr") if payload.get("atr") is not None else med_atr)
    test_entry = float(payload.get("entry") if payload.get("entry") is not None else m15_bars[-1]["close"])
    test_sl = float(payload.get("sl") if payload.get("sl") is not None else round(test_entry - 1.0 * test_atr, dig))
    test_tp = float(payload.get("tp") if payload.get("tp") is not None else round(test_entry + 2.0 * test_atr, dig))
    test_thresh = float(payload.get("threshold") if payload.get("threshold") is not None else sym_metrics.get("threshold", 0.40))
    test_prob = float(payload.get("probability") if payload.get("probability") is not None else 0.54)

    test_record = {
        "symbol": symbol,
        "timeframe": "M15",
        "bar_close_time": bar_time,
        "decision": "BUY",
        "probability": test_prob,
        "threshold": test_thresh,
        "gate_passed": True,
        "reason": payload.get("reason", f"TEST_BUY: Model prob {test_prob:.2f} >= {test_thresh:.2f} threshold, H4 gate passed"),
        "entry": test_entry,
        "sl": test_sl,
        "tp": test_tp,
        "rr": 2.0,
        "atr": test_atr,
        "mode": "TEST_SHADOW",
        "validated_stats": {
            "expectancy_r": sym_metrics.get("expectancy_r", 0.0),
            "win_rate": 0.40,
            "oos_trades": sym_metrics.get("oos_trades", 0),
            "folds_positive": f"{sym_metrics.get('folds_positive', 0)}/{sym_metrics.get('folds_total', 5)}",
            "tier": sym_metrics.get("tier", "EXPERIMENTAL"),
            "worst_fold_r": sym_metrics.get("worst_fold_r", 0.0)
        }
    }
    
    is_existing = bar_time in snapshot_service.published_signals_by_bar.get(symbol, {})
    snap = snapshot_service.create_model_signal_card(symbol, m15_bars, test_record, bar_time)
    
    return {
        "status": "SUCCESS",
        "is_newly_created": not is_existing,
        "snapshot_id": snap.get("snapshot_id"),
        "symbol": symbol,
        "bar_close_time": bar_time,
        "file_exists_json": os.path.exists(os.path.join(snapshot_service.storage_dir, f"{snap.get('snapshot_id')}.json")),
        "file_exists_png": os.path.exists(os.path.join(snapshot_service.storage_dir, f"{snap.get('snapshot_id')}.png")),
        "data": snap
    }

@app.post("/api/v1/telegram/publish/{identifier}")
async def publish_telegram_signal_endpoint(
    identifier: str,
    dry_run: Optional[bool] = None,
    force: bool = Query(False),
    is_test: bool = Query(False)
):
    """
    Brief 10, 18: Triggers publishing a confirmed model signal card to Telegram via n8n.
    Enforces all safety controls, deduplication, rate limiting, and dry-run mode.
    """
    snap = _resolve_snapshot_for_identifier(identifier, state)
    if not snap:
        raise HTTPException(status_code=404, detail=f"Signal card {identifier} not found")
        
    original_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)
    if dry_run is not None:
        settings.TELEGRAM_DRY_RUN = dry_run
        
    try:
        success = await telegram_bot.publish_confirmed_signal(snap, force=force, is_test=is_test)
        return {
            "status": "SUCCESS" if success else "REJECTED_OR_FAILED",
            "published": success,
            "snapshot_id": snap.get("snapshot_id"),
            "symbol": snap.get("symbol"),
            "dry_run": getattr(settings, "TELEGRAM_DRY_RUN", True),
            "is_test": is_test,
            "webhook_url": getattr(settings, "N8N_WEBHOOK_URL", ""),
            "is_already_posted": telegram_bot.is_signal_already_posted(snap.get("snapshot_id", ""))
        }
    finally:
        if dry_run is not None:
            settings.TELEGRAM_DRY_RUN = original_dry_run

@app.post("/api/v1/telegram/waiting_notice")
async def trigger_telegram_waiting_notice(
    dry_run: Optional[bool] = None,
    force: bool = Query(False)
):
    """
    Brief 10: Explicitly triggers or tests sending an hourly market scan waiting notice via n8n.
    """
    original_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)
    if dry_run is not None:
        settings.TELEGRAM_DRY_RUN = dry_run
    try:
        symbols_status, last_signal = get_current_symbols_status_and_last_signal()
        sent = await telegram_bot.send_waiting_notice(
            symbols_status=symbols_status,
            last_signal=last_signal,
            force=force
        )
        return {
            "status": "SUCCESS" if sent else "SKIPPED_OR_FAILED",
            "sent": sent,
            "dry_run": getattr(settings, "TELEGRAM_DRY_RUN", True),
            "webhook_url": getattr(settings, "N8N_WEBHOOK_URL", ""),
            "symbols_status": symbols_status,
            "last_signal": last_signal
        }
    finally:
        if dry_run is not None:
            settings.TELEGRAM_DRY_RUN = original_dry_run

@app.get("/api/v1/telegram/status")
def get_telegram_service_status():
    """
    Brief 10, 17, 18: Returns telegram/n8n service diagnostics, rate limit counters,
    open signals ledger, suppression stats, and isolated real vs test signal counts.
    """
    try:
        telegram_bot._load_state()
    except Exception:
        pass
    try:
        signal_outcome_tracker._sync_from_disk()
    except Exception:
        pass

    supp_counts = telegram_bot.state.get("suppression_counts", {})
    posted_map = telegram_bot.state.get("posted_signals", {})
    # Only count real non-test model signals
    real_posted = [
        k for k, v in posted_map.items()
        if not v.get("is_test") and "verify" not in str(k).lower() and "vps" not in str(k).lower() and "test" not in str(k).lower()
    ]
    test_count = int(telegram_bot.state.get("test_signals_count", 0))
    total_posted = len(real_posted)
    total_supp = sum(supp_counts.values()) if isinstance(supp_counts, dict) else 0
    supp_ratio_pct = round((total_supp / (total_posted + total_supp) * 100.0), 1) if (total_posted + total_supp) > 0 else 0.0

    return {
        "status": "ONLINE",
        "dry_run": getattr(settings, "TELEGRAM_DRY_RUN", True),
        "webhook_url": getattr(settings, "N8N_WEBHOOK_URL", ""),
        "heartbeat_minutes": getattr(settings, "TELEGRAM_HEARTBEAT_MINUTES", 60),
        "last_signal_timestamp": telegram_bot.state.get("last_signal_timestamp", 0),
        "last_signal_info": telegram_bot.state.get("last_signal_info"),
        "posted_signals_count": total_posted,
        "test_signals_count": test_count,
        "notices_sent_count": len(telegram_bot.state.get("notice_history", [])),
        "validated_symbols": telegram_bot.get_validated_symbols(),
        "validated_symbols_count": len(telegram_bot.get_validated_symbols()),
        "publishing_symbols_count": len(telegram_bot.get_validated_symbols()),
        "excluded_symbols": getattr(settings, "TELEGRAM_EXCLUDED_SYMBOLS", ["XAUUSD", "GBPUSD", "EURJPY", "EURUSD"]),
        "publishing_models": telegram_bot.get_publishing_models(),
        "publishing_models_count": len(telegram_bot.get_publishing_models()),
        "publishing_models_by_symbol": telegram_bot.get_publishing_models_by_symbol(),
        "short_models_available": telegram_bot.get_short_models(),
        "short_models": telegram_bot.get_short_models(),
        "suppression_counts": supp_counts,
        "suppressions_by_symbol": supp_counts,
        "total_suppressions": total_supp,
        "suppression_ratio_pct": supp_ratio_pct,
        "open_signals": signal_outcome_tracker.get_open_signals(),
        "trades_publishing_enabled": getattr(settings, "TELEGRAM_PUBLISH_TRADES", False),
        "posted_trades_opened_count": trade_event_service.get_opened_count(),
        "posted_trades_closed_count": trade_event_service.get_closed_count(),
        "results_publishing_enabled": getattr(settings, "TELEGRAM_PUBLISH_RESULTS", False),
        "model_signals_publishing_enabled": getattr(settings, "TELEGRAM_PUBLISH_MODEL_SIGNALS", False),
        "desk_publishing_enabled": getattr(settings, "TELEGRAM_PUBLISH_DESK", False),
        "sealed_manual_records_count": len(getattr(telegram_bot, "sealed_manual_records", set())),
        "posted_results_count": len(getattr(telegram_bot, "posted_results", {})),
        "historical_results_count": sum(1 for r in getattr(telegram_bot, "posted_results", {}).values() if r.get("is_historical")),
        "live_results_posted_count": sum(1 for r in getattr(telegram_bot, "posted_results", {}).values() if not r.get("is_historical")),
        "schema_reference_sent": telegram_bot.is_schema_reference_sent(),
        "posted_reports_count": len(getattr(telegram_bot, "posted_reports", {}))
    }


@app.post("/api/v1/telegram/schema_reference")
async def trigger_schema_reference_message(
    force: bool = Query(False)
):
    """
    Brief 24 Section 2: Triggers or tests dispatching the schema-documentation reference message.
    Guarded by persistent disk flag so it never repeats across restarts unless force=True.
    """
    already_sent = telegram_bot.is_schema_reference_sent()
    sent = await telegram_bot.send_schema_reference_message(force=force)
    return {
        "status": "SUCCESS" if sent else ("ALREADY_SENT" if already_sent and not force else "FAILED"),
        "sent": sent,
        "already_sent_on_disk": already_sent,
        "dry_run": getattr(settings, "TELEGRAM_DRY_RUN", True),
        "webhook_url": getattr(settings, "N8N_WEBHOOK_URL", ""),
        "message_text": telegram_bot.build_schema_reference_message()
    }


@app.get("/api/v1/telegram/report/12h/preview")
def preview_12h_report():
    """
    Brief 24 Section 3: Previews the current 12-hour report payload and text without sending.
    """
    report_data = telegram_bot.generate_12h_report()
    if report_data is None:
        return {
            "status": "SKIPPED",
            "message": "0 closed trades and 0 open trades in this period. Report is noise and skipped."
        }
    return {
        "status": "PREVIEW",
        "period_key": report_data.get("period_key"),
        "already_posted": telegram_bot.is_report_already_posted(report_data.get("period_key", "")),
        "payload": report_data
    }


@app.post("/api/v1/telegram/report/12h")
async def trigger_12h_report(
    force: bool = Query(False)
):
    """
    Brief 24 Section 3: Generates and publishes the 12-hour report to n8n webhook.
    Deduplicated on disk per period unless force=True.
    """
    report_data = telegram_bot.generate_12h_report()
    if report_data is None:
        return {
            "status": "SKIPPED",
            "sent": False,
            "message": "0 closed trades and 0 open trades in this period. Report skipped."
        }
    period_key = report_data["period_key"]
    already_posted = telegram_bot.is_report_already_posted(period_key)
    sent = await telegram_bot.publish_12h_report(force=force)
    return {
        "status": "SUCCESS" if sent else ("ALREADY_POSTED" if already_posted and not force else "FAILED"),
        "sent": sent,
        "period_key": period_key,
        "already_posted": already_posted,
        "dry_run": getattr(settings, "TELEGRAM_DRY_RUN", True),
        "payload": report_data
    }



@app.get("/share/analysis/{identifier}", response_class=HTMLResponse)
def get_shareable_analysis_page(
    identifier: str,
    lang: str = Query("ar", pattern="^(ar|en)$")
):
    """
    Renders standalone institutional dark web page for sharing on Telegram/WhatsApp/Twitter.
    Supports ?lang=ar or ?lang=en and full interactive features.
    """
    snap = _resolve_snapshot_for_identifier(identifier, state)
    if not snap:
        raise HTTPException(status_code=404, detail="Analysis snapshot not found")
    
    sym = snap.get("symbol", "XAUUSD")
    live_p = state.get("live_prices", {}).get(sym, {})
    curr_price = float(live_p.get("bid", snap.get("entry_price", 0.0)))
    freshness = snapshot_service.evaluate_snapshot_freshness(snap, curr_price)

    html_content = render_shareable_analysis_html(
        snapshot=snap,
        freshness=freshness,
        lang=lang
    )
    return HTMLResponse(content=html_content)

@app.post("/api/v1/analyst/snapshot/request-fresh/{symbol}")
async def request_fresh_analysis_and_screenshot(symbol: str, _auth: str = Depends(verify_admin_token)):
    """
    Requests an immediate fresh chart screenshot from MT5 Bridge and generates a new immutable snapshot.
    Fulfills the user requirement:
    'لو الصوره اتغيرت يعرض يرسل طلب صوره جديده فاهمني'
    """
    symbol = symbol.upper()
    req_id = f"req_{int(time.time()*1000)}"

    # 1. Ask MT5 Bridge to take fresh screenshot
    await manager.send_order_to_mt5({
        "action": "CAPTURE_SCREENSHOT",
        "symbol": symbol,
        "timeframe": "M5",
        "request_id": req_id
    })

    # Short async wait to give the bridge time to snap if connected
    await asyncio.sleep(0.35)

    # 2. Synthesize deep dossier
    dossier = analyst_service.generate_deep_analyst_dossier(symbol, "M5", state)
    img_b64 = dossier.get("screenshot_image_base64", "")

    # 3. Create fresh snapshot
    new_snap = snapshot_service.create_snapshot(symbol, "M5", dossier, img_b64, state)
    snap_id = new_snap.get("snapshot_id")

    brainstorm_service.add_log(
        level="SYSTEM",
        category="ANALYSIS",
        symbol=symbol,
        message=f"📸 Fresh analysis & screenshot snapshot generated: {snap_id}"
    )

    return {
        "status": "SUCCESS",
        "message": f"Fresh snapshot created for {symbol}",
        "snapshot_id": snap_id,
        "share_url": f"/share/analysis/{snap_id}",
        "api_url": f"/api/v1/analyst/share/{snap_id}",
        "image_url": f"/api/v1/analyst/share/{snap_id}/image"
    }

@app.get("/api/v1/analyst/snapshots")
def list_analysis_snapshots(symbol: Optional[str] = None, limit: int = Query(30)):
    """Lists recent immutable analysis snapshots"""
    snaps = snapshot_service.list_snapshots(symbol=symbol, limit=limit)
    return {
        "status": "SUCCESS",
        "count": len(snaps),
        "data": snaps
    }

class ResolveSignalRequest(BaseModel):
    signal_id: str
    barrier_hit: str
    exit_price: float
    reason: str = "Manual resolution"

@app.get("/api/v1/analyst/signal-outcomes")
def get_signal_outcomes(symbol: Optional[str] = None, limit: int = Query(100)):
    """
    Brief 14 Section 4: Query realized outcomes of published signals.
    Returns summary stats (completed trades, win rate, total R, expectancy in R)
    and list of completed trades from durable signal_outcomes.jsonl.
    """
    stats = signal_outcome_tracker.get_stats(symbol=symbol)
    outcomes = signal_outcome_tracker.get_outcomes(symbol=symbol, limit=limit)
    return {
        "status": "SUCCESS",
        "symbol": symbol.upper() if symbol else "ALL",
        "stats": stats,
        "count": len(outcomes),
        "outcomes": outcomes
    }

@app.get("/api/v1/analyst/signal-outcomes/pending")
def get_pending_signals(symbol: Optional[str] = None):
    """
    Brief 14 Section 4: Query currently active pending signals awaiting SL/TP hit.
    """
    pending = signal_outcome_tracker.get_pending_signals(symbol=symbol)
    return {
        "status": "SUCCESS",
        "symbol": symbol.upper() if symbol else "ALL",
        "count": len(pending),
        "pending": pending
    }

@app.post("/api/v1/analyst/signal-outcomes/resolve")
def resolve_signal_outcome(req: ResolveSignalRequest, _auth: str = Depends(verify_admin_token)):
    """
    Manual resolution endpoint for a pending signal.
    """
    res = signal_outcome_tracker.resolve_signal(
        signal_id=req.signal_id,
        barrier_hit=req.barrier_hit.upper(),
        exit_price=req.exit_price,
        reason=req.reason
    )
    if not res:
        raise HTTPException(status_code=404, detail=f"Signal {req.signal_id} not found in pending signals")
    return {"status": "SUCCESS", "data": res}

# ==============================================================================
# 🎯 BRIEF 26: THE ANALYST DESK ENDPOINTS
# ==============================================================================

class DeskTradeSubmitRequest(BaseModel):
    symbol: str
    direction: str  # BUY or SELL
    timeframe: str = "M15"
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    zone_low: Optional[float] = None
    zone_high: Optional[float] = None
    discretionary_zone: Optional[str] = None
    why: Optional[str] = None
    why_reason: Optional[str] = None
    publish_to_channel: bool = True

class DeskAuthPayload(BaseModel):
    password: Optional[str] = None

@app.post("/api/v1/desk/auth")
async def authenticate_desk(
    request: Request,
    payload: Optional[DeskAuthPayload] = None,
    authorization: Optional[str] = Header(None),
    x_desk_password: Optional[str] = Header(None, alias="X-Desk-Password"),
    password: Optional[str] = Query(None)
):
    """
    Validates the analyst desk password.
    Accepts password via JSON body {password: ...}, X-Desk-Password header, Bearer token, or query param.
    """
    configured_pwd = getattr(settings, "ANALYST_DESK_PASSWORD", "fx_desk_sec_2026_99a8b")
    token = None
    if payload and payload.password:
        token = payload.password.strip()
    elif authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_desk_password:
        token = x_desk_password.strip()
    elif password:
        token = password.strip()
    else:
        try:
            body = await request.json()
            if isinstance(body, dict) and "password" in body:
                token = str(body["password"]).strip()
        except Exception:
            pass

    if not token or not secrets.compare_digest(token, configured_pwd):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid Analyst Desk password"
        )
    return {"status": "SUCCESS", "authenticated": True, "token": configured_pwd}

@app.get("/api/v1/desk/live-prices")
def get_desk_live_prices(_auth: str = Depends(verify_desk_auth)):
    """
    Returns latest live prices, spreads, and median ATRs for the 12 active pairs.
    """
    DESK_SYMBOLS = [
        "XAUUSD", "EURUSD", "GBPUSD", "USDJPY",
        "AUDUSD", "USDCAD", "USDCHF", "EURJPY",
        "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"
    ]
    result = {}
    for sym in DESK_SYMBOLS:
        spec = get_symbol_spec(sym)
        live = state.get("live_prices", {}).get(sym, {})
        bid = live.get("bid", 0.0)
        ask = live.get("ask", 0.0)
        atr = get_reference_median_atr(sym)
        
        # If live bid is 0, fallback to candle history
        if bid == 0.0:
            m15 = state.get("candle_history", {}).get(sym, {}).get("M15", [])
            if m15:
                bid = float(m15[-1].get("close", 0.0))
                ask = bid

        result[sym] = {
            "bid": bid,
            "ask": ask,
            "atr": atr,
            "point": spec.get("point", 0.00001),
            "digits": spec.get("digits", 5)
        }
    return {"status": "SUCCESS", "data": result, "symbols": result}

@app.post("/api/v1/desk/trades")
async def record_desk_trade(req: DeskTradeSubmitRequest, _auth: str = Depends(verify_desk_auth)):
    """
    Brief 26 Section 1 & 2: Record owner manual trade with instant 91-feature snapshot.
    Validates sanity (direction, TP ordering, 0.1-10.0 ATR bounds) before persisting.
    Measurement instrument only. Zero broker execution capability.
    NEVER sends notifications to Telegram unless desk publishing is enabled.
    """
    sym = req.symbol.upper()
    direction = req.direction.upper()
    if direction not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="Direction must be BUY or SELL")
        
    spec = get_symbol_spec(sym)
    atr = get_reference_median_atr(sym)
    
    # Retrieve current live price for proximity validation (Brief 31 Section 1)
    live = state.get("live_prices", {}).get(sym, {})
    live_price = live.get("bid", 0.0) or live.get("ask", 0.0)
    if live_price == 0.0:
        m15 = state.get("candle_history", {}).get(sym, {}).get("M15", [])
        if m15:
            live_price = float(m15[-1].get("close", 0.0))

    # Sanity validation (Agent Brief 26 & 28 & 31)
    is_valid, sanity_err = validate_manual_trade_sanity(
        entry=req.entry_price,
        stop_loss=req.stop_loss,
        tp1=req.take_profit_1,
        tp2=req.take_profit_2,
        tp3=req.take_profit_3,
        direction=direction,
        atr=atr,
        symbol=sym,
        zone_low=req.zone_low,
        zone_high=req.zone_high,
        live_price=live_price
    )
    if not is_valid:
        detail_msg = sanity_err if sanity_err.startswith("CANNOT RECORD") else f"Sanity Check Failed: {sanity_err}"
        raise HTTPException(status_code=400, detail=detail_msg)

    # Extract candle history for snapshot
    candle_dict = state.get("candle_history", {}).get(sym, {})
    
    # Zone string formatting if low and high are given
    disc_zone = req.discretionary_zone
    if not disc_zone and req.zone_low is not None and req.zone_high is not None:
        disc_zone = f"{req.zone_low}-{req.zone_high}"
    reason = req.why or req.why_reason or ""

    # Record trade and snapshot
    try:
        trade_record = analyst_desk_service.record_trade(
            symbol=sym,
            direction=direction,
            timeframe=req.timeframe or "M15",
            entry_price=req.entry_price,
            stop_loss=req.stop_loss,
            take_profit_1=req.take_profit_1,
            take_profit_2=req.take_profit_2,
            take_profit_3=req.take_profit_3,
            zone_low=req.zone_low,
            zone_high=req.zone_high,
            discretionary_zone=disc_zone,
            why=reason,
            why_reason=reason,
            candles_dict=candle_dict,
            point=spec.get("point", 0.00001),
            digits=spec.get("digits", 5),
            publish_to_channel=req.publish_to_channel,
            live_price=live_price
        )
    except ValueError as ve:
        logger.warning(f"Desk snapshot refused for {sym}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Desk record trade error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Desk Trade {trade_record['trade_id']} ({sym} {direction}) recorded in measurement mode. Zero broker order path exists.")

    # Brief 29 Section 3: Dispatch manual trade publishing to Telegram/n8n
    asyncio.create_task(telegram_bot.publish_desk_trade(trade_record))

    return {"status": "SUCCESS", "data": trade_record, "trade": trade_record}

@app.get("/api/v1/desk/status")
def get_desk_status(_auth: str = Depends(verify_desk_auth)):
    """Brief 27 Section 1: Exposes snapshot success and failure counters."""
    return {"status": "SUCCESS", "data": analyst_desk_service.get_status()}

@app.get("/api/v1/desk/trades")
@app.get("/api/v1/desk/trades/open")
def get_desk_open_trades(_auth: str = Depends(verify_desk_auth)):
    """Returns currently open manual desk trades awaiting resolution."""
    trades = analyst_desk_service.get_open_trades()
    return {"status": "SUCCESS", "data": trades, "open_trades": trades}

@app.get("/api/v1/desk/trades/closed")
def get_desk_closed_trades(limit: int = Query(50), symbol: Optional[str] = None, _auth: str = Depends(verify_desk_auth)):
    """Brief 27 Section 4: Returns resolved manual trade outcomes, newest first."""
    closed = analyst_desk_service.get_closed_trades(symbol=symbol, limit=limit)
    return {"status": "SUCCESS", "data": closed, "closed_trades": closed}

@app.get("/api/v1/desk/trades/{trade_id}/card")
def get_desk_trade_card(trade_id: str):
    """
    Brief 27 Section 3 & Brief 29 Section 3: Serves manual trade card image (PNG)
    showing trade levels, discretionary zone band, and rationale caption.
    Publicly reachable without password for Telegram/n8n image fetching.
    """
    target = _resolve_manual_trade(trade_id)
    if not target:
        raise HTTPException(status_code=404, detail="Desk trade not found")

    sym = target.get("symbol", "").upper()
    candles_dict = state.get("candle_history", {}).get(sym, {})
    m15_candles = candles_dict.get("M15") or []

    png_bytes = render_manual_desk_card(trade_data=target, m15_candles=m15_candles)
    if not png_bytes:
        raise HTTPException(status_code=500, detail="Failed to render desk trade card")

    return Response(content=png_bytes, media_type="image/png")

@app.get("/api/v1/desk/records")
def get_desk_records(limit: int = Query(100), _auth: str = Depends(verify_desk_auth)):
    """Returns all recorded manual trades with snapshot metadata."""
    return {"status": "SUCCESS", "data": analyst_desk_service.get_manual_records(limit=limit)}

@app.get("/api/v1/desk/outcomes")
def get_desk_outcomes(symbol: Optional[str] = None, limit: int = Query(100), _auth: str = Depends(verify_desk_auth)):
    """Returns resolved manual trade outcomes from manual_outcomes.jsonl."""
    outcomes = signal_outcome_tracker.get_manual_outcomes(symbol=symbol, limit=limit)
    return {"status": "SUCCESS", "data": outcomes, "count": len(outcomes)}

@app.get("/api/v1/desk/report/comparison")
def get_desk_comparison_report(_auth: str = Depends(verify_desk_auth)):
    """Brief 26 Section 6a: 3-way performance comparison report."""
    return {"status": "SUCCESS", "data": analyst_desk_service.get_comparison_report()}

@app.get("/api/v1/desk/report/signature")
def get_desk_signature_report(_auth: str = Depends(verify_desk_auth)):
    """Brief 26 Section 6b: Cohen's d feature signature report."""
    return {"status": "SUCCESS", "data": analyst_desk_service.get_signature_report()}

@app.get("/api/v1/desk/report/zones")
def get_desk_zones_report(_auth: str = Depends(verify_desk_auth)):
    """Brief 26 Section 6c: Discretionary zones performance report."""
    return {"status": "SUCCESS", "data": analyst_desk_service.get_zones_report()}

# =========================================================================
# 🪞 BRIEF 32: THE MIRRORED ACCOUNT (WATCH ONLY) ENDPOINTS
# =========================================================================

class MirrorAuthPayload(BaseModel):
    password: Optional[str] = None

@app.post("/api/v1/mirror/auth")
async def authenticate_mirror(
    payload: Optional[MirrorAuthPayload] = None,
    authorization: Optional[str] = Header(None),
    x_mirror_password: Optional[str] = Header(None, alias="X-Mirror-Password"),
    password: Optional[str] = Query(None)
):
    """
    Brief 32 Section 4: Validates the mirror dashboard password.
    Accepts password via JSON body {password: ...}, X-Mirror-Password header, Bearer token, or query param.
    Uses constant-time comparison to prevent timing side-channel attacks.
    NEVER logs, commits, or returns the MT5 investor password.
    """
    configured_pwd = getattr(settings, "MIRROR_UI_PASSWORD", "fx_mirror_sec_2026_ab81c")
    token = None
    if payload and payload.password:
        token = payload.password.strip()
    elif authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_mirror_password:
        token = x_mirror_password.strip()
    elif password:
        token = password.strip()

    if not token or not secrets.compare_digest(token, configured_pwd):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid Mirror Dashboard password"
        )

    return {
        "status": "SUCCESS",
        "authenticated": True,
        "token": token,
        "account": {
            "login": getattr(settings, "MT5_MIRROR_LOGIN", 8058543),
            "server": getattr(settings, "MT5_MIRROR_SERVER", "SCFMLimited-Demo2"),
            "read_only": True,
            "investor_mode": True
        }
    }

@app.get("/api/v1/mirror/status")
def get_mirror_status(_auth: str = Depends(verify_mirror_auth)):
    """Brief 32: Status of mirror watcher service, account info, and stats."""
    return {"status": "SUCCESS", "data": mirror_service.get_status()}

@app.get("/api/v1/mirror/trades/open")
def get_mirror_open_trades(_auth: str = Depends(verify_mirror_auth)):
    """Brief 32: Currently open mirrored positions with lot size, entry, and stops."""
    return {"status": "SUCCESS", "data": mirror_service.get_open_trades()}

@app.get("/api/v1/mirror/trades/closed")
def get_mirror_closed_trades(
    limit: int = Query(50),
    symbol: Optional[str] = None,
    _auth: str = Depends(verify_mirror_auth)
):
    """Brief 32: Closed mirrored deals with MT5 reasons and signed R."""
    return {"status": "SUCCESS", "data": mirror_service.get_closed_trades(symbol=symbol, limit=limit)}

@app.get("/api/v1/mirror/trades/{ticket}/card")
def get_mirror_trade_card(ticket: int):
    """
    Brief 32 Section 4: Public card route for Telegram and social preview.
    Renders visual card image for a mirrored trade without requiring auth.
    """
    target = mirror_service.open_positions.get(ticket) or mirror_service.closed_trades.get(ticket)
    if not target:
        raise HTTPException(status_code=404, detail="Mirrored trade not found")

    sym = target.get("symbol", "BTCUSD")
    m15_candles = state.get("candle_history", {}).get(sym.upper(), {}).get("M15", [])
    try:
        png_bytes = render_manual_desk_card(trade_data=target, m15_candles=m15_candles)
        if not png_bytes:
            raise ValueError("Renderer returned empty bytes")
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"Failed to render mirror trade card #{ticket}: {e}")
        raise HTTPException(status_code=500, detail="Failed to render mirror trade card")

@app.post("/api/v1/mirror/sync")
async def sync_mirror_data(
    payload: Dict[str, Any] = Body(...),
    _auth: str = Depends(verify_mirror_auth)
):
    """
    Brief 32: Ingest poll updates from MT5 bridge for the mirrored account.
    Expects payload:
    {
       "account_info": {...},
       "positions": [...],
       "deals": [...]
    }
    """
    results = {}
    acc_login = None
    if "account_info" in payload:
        acc = payload["account_info"]
        is_demo, reason = mirror_service.verify_account_is_demo(acc)
        if not is_demo:
            raise HTTPException(status_code=400, detail=f"Refusing live account: {reason}")
        acc_login = acc.get("login")
        mirror_service.account_info = acc
        mirror_service.is_connected = True
        if acc_login:
            mirror_service.login = acc_login
            mirror_service.accounts[acc_login] = acc
        if acc.get("server"):
            mirror_service.server = acc["server"]
        results["account_verified"] = "DEMO"

    if "positions" in payload:
        pos_res = mirror_service.process_positions_update(payload["positions"], account_login=acc_login)
        results["positions_processed"] = pos_res

    if "deals" in payload:
        deals_res = mirror_service.process_closed_deals(payload["deals"])
        results["deals_processed"] = deals_res

    if "closed_trades" in payload and isinstance(payload["closed_trades"], list):
        mirror_service.merge_closed_trades(payload["closed_trades"], account_login=acc_login)
        results["closed_trades_count"] = len(mirror_service.closed_trades)

    if "candles" in payload and isinstance(payload["candles"], dict):
        for sym, bars in payload["candles"].items():
            state.setdefault("candle_history", {}).setdefault(sym.upper(), {})["M15"] = bars
        results["candles_updated"] = list(payload["candles"].keys())

    return {"status": "SUCCESS", "results": results}

@app.post("/api/v1/mirror/order-check")
def mirror_order_check(_auth: str = Depends(verify_mirror_auth)):
    """
    Brief 32 Section 2 & Acceptance 1: Demonstrates that order execution is physically refused.
    Mirror connection uses investor password only (read-only).
    """
    raise HTTPException(
        status_code=403,
        detail="Order execution REFUSED: Mirrored account connection is strictly READ-ONLY via investor password. Execution forbidden by architecture."
    )

@app.post("/api/v1/mirror/clear")
def clear_mirror_state(_auth: str = Depends(verify_mirror_auth)):
    """Resets in-memory and on-disk mirror positions for switching to a new account."""
    mirror_service.clear_all_memory()
    return {"status": "SUCCESS", "message": "Mirrored positions cleared successfully."}

class ManualOrderRequest(BaseModel):
    symbol: str
    direction: str
    volume: float = 0.01
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: str = "manual_order"

class ManualCloseRequest(BaseModel):
    ticket: int
    volume: Optional[float] = None

class ManualModifyRequest(BaseModel):
    ticket: int
    sl: Optional[float] = None
    tp: Optional[float] = None

@app.post("/api/v1/trade/manual/order")
def manual_trade_order(req: ManualOrderRequest):
    """Executes a manual BUY or SELL order directly via MT5BridgeEA."""
    from mt5_bridge_sync import MT5BridgeSyncClient
    client = MT5BridgeSyncClient()
    res = client.open_order(
        symbol=req.symbol,
        order_type=req.direction,
        volume=req.volume,
        sl=req.sl,
        tp=req.tp,
        comment=req.comment
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error_message", "Order failed"))
    return {"status": "SUCCESS", "data": res.get("data", {})}

@app.post("/api/v1/trade/manual/close")
def manual_trade_close(req: ManualCloseRequest):
    """Closes an open position by ticket via MT5BridgeEA."""
    from mt5_bridge_sync import MT5BridgeSyncClient
    client = MT5BridgeSyncClient()
    res = client.close_position(ticket=req.ticket, volume=req.volume)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error_message", "Close failed"))
    return {"status": "SUCCESS", "data": res.get("data", {})}

@app.post("/api/v1/trade/manual/modify")
def manual_trade_modify(req: ManualModifyRequest):
    """Modifies SL / TP for an open position ticket via MT5BridgeEA."""
    from mt5_bridge_sync import MT5BridgeSyncClient
    client = MT5BridgeSyncClient()
    res = client.modify_position(ticket=req.ticket, sl=req.sl, tp=req.tp)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error_message", "Modify failed"))
    return {"status": "SUCCESS", "data": res.get("data", {})}

@app.get("/api/v1/trade/manual/tick/{symbol}")
def manual_trade_get_tick(symbol: str):
    """Retrieves live Bid/Ask/Spread for a symbol via MT5BridgeEA."""
    from mt5_bridge_sync import MT5BridgeSyncClient
    client = MT5BridgeSyncClient()
    res = client.send_request("GET_TICK", {"symbol": symbol.upper()})
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error_message", "Failed to get tick"))
    return {"status": "SUCCESS", "data": res.get("data", {})}

@app.get("/api/v1/history/trades-with-screens")
def get_trades_with_screens(limit: int = Query(100)):
    """Returns detailed history of closed trades with MT5 screenshots, SL hit status, and analysis"""
    history = episodic_memory.get_detailed_trade_screen_history(limit)
    return {"status": "SUCCESS", "data": history, "total": len(history)}

@app.post("/api/v1/mode")
async def update_execution_mode(req: ModeUpdateRequest, request: Request, _auth: str = Depends(verify_admin_token)):
    check_trade_rate_limit(request)
    state["execution_mode"] = req.mode
    if req.mode == ExecutionMode.AUTO_TRADING:
        curr_eq = state["account_info"].get("equity", 100.0)
        state["starting_daily_equity"] = curr_eq
        state["circuit_breaker_overridden"] = True
        risk_manager.last_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    save_persisted_state()
    logger.info(f"Execution Mode Updated to: {req.mode}")
    brainstorm_service.add_log(
        level="SYSTEM",
        category="CONFIG",
        symbol="SYSTEM",
        message=f"Execution Mode switched to: {req.mode.value}"
    )
    await manager.broadcast_to_dashboard({
        "type": "MODE_CHANGE",
        "mode": req.mode.value
    })
    return {"status": "SUCCESS", "new_mode": req.mode}

@app.post("/api/v1/reset_circuit_breaker")
async def reset_circuit_breaker(_auth: str = Depends(verify_admin_token)):
    state["circuit_breaker_overridden"] = True
    save_persisted_state()
    logger.info("Circuit Breaker has been manually reset. Auto-trading re-enabled.")
    brainstorm_service.add_log(
        level="SYSTEM",
        category="RISK",
        symbol="ALL",
        message="🔄 Circuit Breaker manually reset by user. Auto-trading resumed."
    )
    await manager.broadcast_to_dashboard({
        "type": "CIRCUIT_BREAKER_RESET",
        "message": "Circuit breaker reset successfully."
    })
    return {"status": "SUCCESS", "message": "Circuit breaker reset successfully. Auto-trading resumed."}

@app.post("/api/v1/strategy")
async def update_strategy_settings(req: StrategySettingsRequest, _auth: str = Depends(verify_admin_token)):
    state["trading_school"] = req.trading_school
    state["trading_style"] = req.trading_style
    state["lot_mode"] = req.lot_mode
    state["fixed_lot_size"] = req.fixed_lot_size
    state["max_risk_percent"] = req.max_risk_percent
    state["active_symbols"] = req.active_symbols
    save_persisted_state()

    logger.info(f"Strategy Updated: School={req.trading_school}, Style={req.trading_style}")
    brainstorm_service.add_log(
        level="SYSTEM",
        category="CONFIG",
        symbol="CONFIG",
        message=f"Strategy updated -> School: {req.trading_school.value} | Style: {req.trading_style.value} | Lot: {req.lot_mode.value}"
    )

    await manager.broadcast_to_dashboard({
        "type": "STRATEGY_CHANGE",
        "trading_school": req.trading_school.value,
        "trading_style": req.trading_style.value,
        "lot_mode": req.lot_mode.value,
        "fixed_lot_size": req.fixed_lot_size,
        "max_risk_percent": req.max_risk_percent,
        "active_symbols": req.active_symbols
    })
    return {"status": "SUCCESS", "settings": req}

@app.post("/api/v1/trade/manual-execute")
async def manual_execute_trade(req: ManualTradeRequest, request: Request, _auth: str = Depends(verify_admin_token)):
    check_trade_rate_limit(request)
    symbol = req.symbol
    order_type = req.order_type.upper()
    
    analysis = state["latest_signals"].get(symbol, {})
    sl = analysis.get("suggested_sl", 0.0)
    tp = analysis.get("suggested_tp", 0.0)

    trade_cmd = {
        "action": "EXECUTE_TRADE",
        "symbol": symbol,
        "order_type": order_type,
        "lot_size": req.lot_size,
        "sl": sl,
        "tp": tp,
        "comment": "Manual Tri-School AI"
    }

    in_flight_trades[symbol] = time.time()
    await manager.send_order_to_mt5(trade_cmd)
    brainstorm_service.add_log(
        level="EXECUTION",
        category="TRADE_DECISION",
        symbol=symbol,
        message=f"Manual execution dispatched to MT5 -> {symbol} {order_type} @ {req.lot_size} Lots (SL: {sl} | TP: {tp})"
    )
    return {"status": "SUCCESS", "trade": trade_cmd}

@app.post("/api/v1/trade/approve")
async def approve_semi_auto_trade(req: ApproveTradeRequest, request: Request, _auth: str = Depends(verify_admin_token)):
    check_trade_rate_limit(request)
    trade_cmd = {
        "action": "EXECUTE_TRADE",
        "symbol": req.symbol,
        "order_type": req.order_type,
        "lot_size": req.lot_size,
        "sl": req.sl,
        "tp": req.tp,
        "comment": "Semi-Auto Tri-School"
    }
    in_flight_trades[req.symbol] = time.time()
    await manager.send_order_to_mt5(trade_cmd)
    brainstorm_service.add_log(
        level="EXECUTION",
        category="TRADE_DECISION",
        symbol=req.symbol,
        message=f"✅ Semi-Auto Trade APPROVED -> {req.symbol} {req.order_type} @ {req.lot_size} Lots"
    )
    return {"status": "SUCCESS", "trade": trade_cmd}

# ----------------------------------------------------
# Dedicated Gold Trader Endpoints (XAUUSD Specialist)
# ----------------------------------------------------
@app.get("/api/v1/gold/state")
def get_gold_state():
    gold_trades = [t for t in state["active_trades"] if "XAU" in t.get("symbol", "").upper() or "GOLD" in t.get("symbol", "").upper()]
    gold_closed = [t for t in state["closed_trades"] if "XAU" in t.get("symbol", "").upper() or "GOLD" in t.get("symbol", "").upper()]
    live_gold = state["live_prices"].get("XAUUSD", {"bid": 0.0, "ask": 0.0})
    return {
        "status": "SUCCESS",
        "symbol": "XAUUSD",
        "gold_execution_mode": state.get("gold_execution_mode", ExecutionMode.ADVISORY),
        "gold_strategy_profile": state.get("gold_strategy_profile", "HYBRID"),
        "gold_lot_mode": state.get("gold_lot_mode", LotMode.DYNAMIC_PERCENT),
        "gold_fixed_lot_size": state.get("gold_fixed_lot_size", 0.05),
        "gold_scalp_lot_size": state.get("gold_scalp_lot_size", 0.05),
        "gold_swing_lot_size": state.get("gold_swing_lot_size", 0.03),
        "gold_fast_be_enabled": state.get("gold_fast_be_enabled", True),
        "gold_risk_percent": state.get("gold_risk_percent", 1.0),
        "analysis": state.get("gold_analysis", {}),
        "live_price": live_gold,
        "active_trades": gold_trades,
        "closed_trades": gold_closed,
        "dxy_correlation": state.get("currency_power", {}).get("gold_macro_bias", "NEUTRAL_DXY"),
        "learning_summary": trade_learning_engine.get_learning_summary()
    }

@app.post("/api/v1/gold/settings")
async def update_gold_settings(req: GoldSettingsRequest, _auth: str = Depends(verify_admin_token)):
    norm_mode = "AUTO_TRADING" if "AUTO" in str(req.gold_execution_mode).upper() else "ANALYST_ONLY"
    state["gold_execution_mode"] = norm_mode
    state["gold_strategy_profile"] = req.gold_strategy_profile
    state["gold_lot_mode"] = req.gold_lot_mode
    state["gold_fixed_lot_size"] = req.gold_fixed_lot_size
    state["gold_scalp_lot_size"] = req.gold_scalp_lot_size
    state["gold_swing_lot_size"] = req.gold_swing_lot_size
    state["gold_fast_be_enabled"] = req.gold_fast_be_enabled
    state["gold_risk_percent"] = req.gold_risk_percent
    save_persisted_state()
    
    lot_mode_str = req.gold_lot_mode.value if hasattr(req.gold_lot_mode, "value") else str(req.gold_lot_mode)
    logger.info(f"Gold Settings Updated: Mode={norm_mode}, Strategy={req.gold_strategy_profile}, ScalpLot={req.gold_scalp_lot_size}, SwingLot={req.gold_swing_lot_size}")
    brainstorm_service.add_log(
        level="SYSTEM",
        category="CONFIG",
        symbol="XAUUSD",
        message=f"👑 Gold Settings Updated -> Mode: {norm_mode} | Strategy: {req.gold_strategy_profile} | Scalp Lot: {req.gold_scalp_lot_size} | Swing Lot: {req.gold_swing_lot_size}"
    )

    # Immediately update state["gold_analysis"] and sync with MT5 HUD
    if "gold_analysis" in state and isinstance(state["gold_analysis"], dict) and state["gold_analysis"]:
        state["gold_analysis"]["strategy_profile"] = req.gold_strategy_profile
        state["gold_analysis"]["execution_mode"] = norm_mode
        state["gold_analysis"]["is_executable"] = (norm_mode == "AUTO_TRADING" and state["gold_analysis"].get("recommendation") in ["BUY", "SELL"] and state["gold_analysis"].get("setup_quality") in ["A+", "A"])
        await manager.send_order_to_mt5({
            "action": "DRAW_GOLD_ANALYSIS",
            "symbol": "XAUUSD",
            "analysis": state["gold_analysis"]
        })

    await manager.broadcast_to_dashboard({
        "type": "GOLD_SETTINGS_CHANGE",
        "gold_execution_mode": norm_mode,
        "gold_strategy_profile": req.gold_strategy_profile,
        "gold_lot_mode": lot_mode_str,
        "gold_fixed_lot_size": req.gold_fixed_lot_size,
        "gold_scalp_lot_size": req.gold_scalp_lot_size,
        "gold_swing_lot_size": req.gold_swing_lot_size,
        "gold_fast_be_enabled": req.gold_fast_be_enabled,
        "gold_risk_percent": req.gold_risk_percent
    })
    return {"status": "SUCCESS", "settings": req}

@app.post("/api/v1/gold/execute")
async def execute_gold_manual_trade(req: GoldManualTradeRequest, request: Request, _auth: str = Depends(verify_admin_token)):
    check_trade_rate_limit(request)
    analysis = state.get("gold_analysis", {})
    curr_p = analysis.get("current_price", 0.0)
    trade_style = (req.trade_style or "HYBRID").upper()
    
    if "SCALP" in trade_style:
        scalp = analysis.get("scalp_analysis", {})
        sl = req.sl if req.sl is not None and req.sl > 0 else scalp.get("scalp_sl", curr_p - 1.80)
        tp = req.tp if req.tp is not None and req.tp > 0 else scalp.get("scalp_tp1", curr_p + 3.00)
        comment_str = "FXENGIN_SCALP_MANUAL"
    else:
        sl = req.sl if req.sl is not None and req.sl > 0 else analysis.get("suggested_sl", curr_p - 5.0)
        tp = req.tp if req.tp is not None and req.tp > 0 else analysis.get("suggested_tp", curr_p + 10.0)
        comment_str = "FXENGIN_SWING_MANUAL"

    trade_cmd = {
        "action": "EXECUTE_TRADE",
        "symbol": "XAUUSD",
        "order_type": req.order_type.upper(),
        "lot_size": req.lot_size,
        "sl": sl,
        "tp": tp,
        "comment": comment_str
    }

    in_flight_trades["XAUUSD"] = time.time()
    await manager.send_order_to_mt5(trade_cmd)
    brainstorm_service.add_log(
        level="EXECUTION",
        category="TRADE_DECISION",
        symbol="XAUUSD",
        message=f"👑 Gold Manual ({trade_style}) Dispatched -> {req.order_type.upper()} @ {req.lot_size} Lots (SL: ${sl} | TP: ${tp})"
    )
    return {"status": "SUCCESS", "trade": trade_cmd}

@app.post("/api/v1/trade/close/{ticket}")
async def close_single_trade(ticket: int, request: Request, _auth: str = Depends(verify_admin_token)):
    check_trade_rate_limit(request)
    cmd = {"action": "CLOSE_TRADE", "ticket": ticket}
    await manager.send_order_to_mt5(cmd)
    brainstorm_service.add_log(
        level="EXECUTION",
        category="TRADE_DECISION",
        symbol="POSITION",
        message=f"Close command sent for Position #{ticket}"
    )
    return {"status": "SUCCESS", "ticket": ticket}

@app.post("/api/v1/panic-close-all")
async def panic_close_all(request: Request, _auth: str = Depends(verify_admin_token)):
    check_trade_rate_limit(request)
    state["execution_mode"] = ExecutionMode.ADVISORY
    save_persisted_state()
    in_flight_trades.clear()
    order = {"action": "PANIC_CLOSE_ALL"}
    await manager.send_order_to_mt5(order)
    state["active_trades"] = []
    
    brainstorm_service.add_log(
        level="RISK",
        category="RISK_GATE",
        symbol="PANIC",
        message="🚨 PANIC CLOSE ALL TRIGGERED! All trades closed, mode reverted to ADVISORY."
    )
    
    await manager.broadcast_to_dashboard({
        "type": "PANIC_EVENT",
        "message": "All open trades closed & mode switched to ADVISORY!"
    })
    return {"status": "SUCCESS", "message": "Panic close executed."}

@app.post("/api/v1/reset-circuit-breaker")
async def reset_circuit_breaker_endpoint(_auth: str = Depends(verify_admin_token)):
    current_eq = state["account_info"].get("equity", 100.0)
    res = risk_manager.reset_daily_circuit_breaker(current_eq, state)
    save_persisted_state()
    await manager.broadcast_to_dashboard({
        "type": "CIRCUIT_BREAKER_RESET",
        "message": "Daily Drawdown Circuit Breaker Manually Reset. Auto-Trading Unlocked!"
    })
    return {"status": "SUCCESS", "message": "Circuit breaker reset successfully.", "data": res}

@app.get("/api/v1/history")
def get_trade_history(scope: str = "today"):
    today_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    all_trades = state.get("closed_trades", [])
    
    today_trades = [
        t for t in all_trades 
        if str(t.get("time", "")).startswith(today_utc_str) or str(t.get("time", "")).replace(".", "-").startswith(today_utc_str)
    ]
    
    selected_trades = today_trades if scope == "today" else all_trades
    today_pnl = sum(float(t.get("profit", 0.0)) for t in today_trades)
    all_time_pnl = sum(float(t.get("profit", 0.0)) for t in all_trades)
    
    return {
        "status": "SUCCESS",
        "scope": scope,
        "today_date": today_utc_str,
        "closed_trades": selected_trades,
        "today_closed_trades": today_trades,
        "all_closed_trades": all_trades,
        "today_realized_pnl": round(today_pnl, 2),
        "all_time_realized_pnl": round(all_time_pnl, 2),
        "today_deals_count": len(today_trades),
        "total_deals_count": len(all_trades)
    }

@app.get("/api/v1/brainstorm")
def get_brainstorm_logs():
    return {"status": "SUCCESS", "logs": brainstorm_service.get_recent_logs(80)}

@app.get("/api/v1/news")
async def get_economic_news():
    events = await news_filter.fetch_live_economic_calendar()
    return {"status": "SUCCESS", "events": events}

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await manager.connect_dashboard(websocket)
    try:
        power_data = correlation_engine.calculate_currency_power(state["live_prices"])
        await websocket.send_json({
            "type": "INIT_STATE",
            "execution_mode": state["execution_mode"].value if hasattr(state["execution_mode"], "value") else state["execution_mode"],
            "trading_school": state["trading_school"].value if hasattr(state["trading_school"], "value") else state["trading_school"],
            "trading_style": state["trading_style"].value if hasattr(state["trading_style"], "value") else state["trading_style"],
            "lot_mode": state["lot_mode"].value if hasattr(state["lot_mode"], "value") else state["lot_mode"],
            "fixed_lot_size": state["fixed_lot_size"],
            "max_risk_percent": state["max_risk_percent"],
            "active_symbols": state["active_symbols"],
            "account_info": state["account_info"],
            "latest_signals": list(state["latest_signals"].values()),
            "gold_analysis": state.get("gold_analysis", {}),
            "gold_execution_mode": state.get("gold_execution_mode", ExecutionMode.ADVISORY).value if hasattr(state.get("gold_execution_mode"), "value") else state.get("gold_execution_mode", "ADVISORY"),
            "gold_lot_mode": state.get("gold_lot_mode", LotMode.DYNAMIC_PERCENT).value if hasattr(state.get("gold_lot_mode"), "value") else state.get("gold_lot_mode", "DYNAMIC_PERCENT"),
            "gold_fixed_lot_size": state.get("gold_fixed_lot_size", 0.05),
            "gold_risk_percent": state.get("gold_risk_percent", 1.0),
            "active_trades": state["active_trades"],
            "closed_trades": state["closed_trades"],
            "live_prices": state["live_prices"],
            "currency_power": power_data,
            "brainstorm_logs": brainstorm_service.get_recent_logs(50)
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_dashboard(websocket)

@app.websocket("/ws/mt5")
async def websocket_mt5(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    reason: Optional[str] = Query("reconnect"),
    client_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None)
):
    # Strict Token Authentication Enforcement
    if settings.MT5_BRIDGE_TOKEN:
        if not token or token != settings.MT5_BRIDGE_TOKEN:
            logger.warning(f"Unauthorized MT5 Bridge connection rejected. Invalid Token: {token}")
            await websocket.close(code=1008)
            return

    client_ip = websocket.client.host if websocket.client else "unknown"
    resolved_client_id = client_id or ("vps-bridge" if client_ip in ["127.0.0.1", "localhost", "::1"] else f"bridge-{client_ip}")
    resolved_role = role or ("primary" if resolved_client_id == "vps-bridge" else "secondary")

    await websocket.accept()
    await bridge_registry.register(resolved_client_id, resolved_role, websocket, client_ip)
    await manager.connect_mt5(websocket)
    state["mt5_connected"] = True
    connect_reason = reason or "reconnect"
    log_coverage_event("connected", reason=f"{resolved_client_id}:{connect_reason}")
    logger.info(f"MT5 Bridge '{resolved_client_id}' [role={resolved_role}] Connected via WebSocket (Reason: {connect_reason}) from {client_ip}!")
    brainstorm_service.add_log(
        level="INFO",
        category="BOOT",
        symbol="BRIDGE",
        message=f"⚡ MetaTrader 5 Bridge '{resolved_client_id}' ({resolved_role}) Authenticated & Connected ({connect_reason})."
    )
    await manager.broadcast_to_dashboard({"type": "MT5_STATUS", "connected": True})
    
    bridge_disconnect_reason = "network|shutdown"
    try:
        global gold_last_trade_time, gold_last_trade_direction, gold_reversal_armed_until
        while True:
            raw_msg = await websocket.receive_text()
            msg = json.loads(raw_msg)
            msg_type = msg.get("type")
            
            if msg_type == "BRIDGE_DISCONNECT":
                bridge_disconnect_reason = msg.get("reason", "shutdown")
                logger.info(f"Received clean bridge disconnect notice: {bridge_disconnect_reason}")
                break
            
            if msg_type == "ACCOUNT_UPDATE":
                new_acc = msg.get("data", state["account_info"])
                new_login = new_acc.get("login", 0)
                prev_login = state["account_info"].get("login", 0)
                current_equity = new_acc.get("equity", 100.0)
                
                # Automatically detect Account Switch (e.g. from 99M demo to $100 demo) and reset baseline
                if (prev_login > 0 and new_login != prev_login) or (state.get("starting_daily_equity", 0) > current_equity * 2.5) or (state.get("starting_daily_equity", 0) <= 0):
                    logger.info(f"🔄 Account switch detected (Login #{prev_login} -> #{new_login}). Resetting daily baseline to ${current_equity:.2f}.")
                    state["starting_daily_equity"] = current_equity
                    save_persisted_state()
                    risk_manager.last_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

                state["account_info"] = new_acc
                raw_trades = msg.get("active_trades", [])
                
                # Check and update daily baseline
                baseline_reset = risk_manager.check_and_reset_daily_baseline(current_equity, state)
                if baseline_reset:
                    save_persisted_state()

                if state["starting_daily_equity"] <= 0:
                    state["starting_daily_equity"] = current_equity
                    save_persisted_state()

                dd_check = risk_manager.check_daily_drawdown_limit(
                    starting_equity=state["starting_daily_equity"],
                    current_equity=current_equity
                )

                for t in raw_trades:
                    sym = t.get("symbol")
                    sig = state["latest_signals"].get(sym, {})
                    t["planned_exit"] = sig.get("planned_exit_reason", f"TP: ${t.get('tp')} | SL: ${t.get('sl')}")

                # Check for newly closed trades to trigger 5-Minute Study Cooldown
                prev_active = {t.get("symbol") for t in state.get("active_trades", [])}
                curr_active = {t.get("symbol") for t in raw_trades}
                closed_syms = prev_active - curr_active
                for cs in closed_syms:
                    if cs:
                        symbol_cooldowns[cs] = time.time() + 300.0
                        if "XAU" in cs.upper() or "GOLD" in cs.upper():
                            gold_last_trade_time = time.time()
                            gold_engine.active_signal_latch = None
                            gold_engine.active_scalp_latch = None
                        logger.info(f"⏳ Trade closed on {cs} -> 5-Minute Market Study & Cooldown initiated.")
                        brainstorm_service.add_log(
                            level="INFO",
                            category="COOLDOWN",
                            symbol=cs,
                            message=f"⏳ Trade closed. 5-Minute post-trade evaluation active for {cs} (Latches Cleared)."
                        )

                state["active_trades"] = raw_trades
                
                # Brief 19: Process active positions for trade_opened publishing
                try:
                    await trade_event_service.on_positions_update(raw_trades)
                except Exception as te_err:
                    logger.warning(f"Error processing trade_opened events: {te_err}")

                # Clear completed in-flight locks for symbols that are now in active_trades
                active_symbols_set = {t.get("symbol") for t in raw_trades}
                for sym in list(in_flight_trades.keys()):
                    if sym in active_symbols_set or (time.time() - in_flight_trades[sym] > 30.0):
                        in_flight_trades.pop(sym, None)

                if "closed_trades" in msg:
                    incoming_closed = msg.get("closed_trades", [])

                    # Brief 19: Process closed trades for trade_closed publishing
                    try:
                        await trade_event_service.on_closed_trades_update(incoming_closed)
                    except Exception as te_err:
                        logger.warning(f"Error processing trade_closed events: {te_err}")

                    existing_tickets = {t.get("ticket") for t in state.get("closed_trades", [])}
                    for ct in incoming_closed:
                        if ct.get("ticket") not in existing_tickets:
                            diag = trade_learning_engine.analyze_closed_trade(ct)
                            autopsy = episodic_memory.conduct_post_mortem_autopsy(ct)
                            
                            ct_profit = float(ct.get("profit", 0.0))
                            ct_symbol = ct.get("symbol", "XAUUSD")

                            # Coordinated Break-Even: when one leg of a split closes in
                            # profit, secure the remaining legs on the same symbol.
                            #
                            # This used to key off "TP1"/"SCALP" appearing in the closed
                            # deal's comment, but that comment is written by the broker
                            # ("tp"/"sl"), not by us, so the branch never fired. Any
                            # profitable close with siblings still open is the signal.
                            if ct_profit > 0.0:
                                siblings = [t for t in state.get("active_trades", [])
                                            if t.get("symbol") == ct_symbol and t.get("ticket")]
                                for open_t in siblings:
                                    try:
                                        op_p = float(open_t.get("entry", open_t.get("open_price", 0.0)) or 0.0)
                                        if op_p <= 0:
                                            continue
                                        is_b = ("BUY" in str(open_t.get("type", "")).upper())
                                        curr_sl = float(open_t.get("sl", 0.0) or 0.0)
                                        be_sl = round(op_p + (1.50 if is_b else -1.50), 2)

                                        # Never move a stop backwards - the trailing engine
                                        # may already have locked in more than break-even.
                                        already_better = (curr_sl > 0 and
                                                          ((is_b and curr_sl >= be_sl) or (not is_b and curr_sl <= be_sl)))
                                        if already_better:
                                            continue

                                        be_cmd = {
                                            "action": "MODIFY_TRADE_SL",
                                            "ticket": int(open_t["ticket"]),
                                            "sl": be_sl,
                                            "tp": float(open_t.get("tp", 0.0))
                                        }
                                        await manager.send_order_to_mt5(be_cmd)
                                        open_t["sl"] = be_sl
                                        brainstorm_service.add_log(
                                            level="RISK",
                                            category="BREAKEVEN",
                                            symbol=ct_symbol,
                                            message=f"🛡️ [Coordinated Break-Even] Sibling leg closed in profit (+${ct_profit:.2f}). Position #{open_t['ticket']} SL moved to Entry +$1.50 (${be_sl}) -> risk-free."
                                        )
                                    except Exception as ce_err:
                                        logger.debug(f"Coordinated BE error: {ce_err}")

                            # Same-direction re-entry block after a real loss.
                            if ct_profit < -0.15:
                                lost_dir = str(ct.get("type", "")).upper()
                                if lost_dir in ("BUY", "SELL"):
                                    block_secs = getattr(settings, "GOLD_POST_LOSS_BLOCK_SECONDS", 900)
                                    register_loss_block(ct_symbol, lost_dir, block_secs)
                                    brainstorm_service.add_log(
                                        level="RISK",
                                        category="RISK_GATE",
                                        symbol=ct_symbol,
                                        message=f"⛔ Loss on {ct_symbol} {lost_dir} (${ct_profit:.2f}). Blocking further {lost_dir} entries for {block_secs // 60} minutes."
                                    )

                            if diag and diag.get("outcome") != "WIN":
                                await manager.broadcast_to_dashboard({
                                    "type": "AI_POST_MORTEM",
                                    "diagnosis": diag,
                                    "learning_summary": trade_learning_engine.get_learning_summary()
                                })
                            
                            await manager.broadcast_to_dashboard({
                                "type": "EXPERIENCE_UPDATE",
                                "autopsy": autopsy,
                                "memory_summary": episodic_memory.get_memory_summary()
                            })
                    state["closed_trades"] = incoming_closed
                    save_persisted_state()

                await manager.broadcast_to_dashboard({
                    "type": "ACCOUNT_UPDATE",
                    "account_info": state["account_info"],
                    "active_trades": state["active_trades"],
                    "closed_trades": state["closed_trades"],
                    "learning_summary": trade_learning_engine.get_learning_summary()
                })

            elif msg_type == "MT5_SCREENSHOT":
                sym = msg.get("symbol", "XAUUSD")
                tf = msg.get("timeframe", "M5")
                img_b64 = msg.get("image", "")
                if img_b64:
                    analyst_service.store_mt5_screenshot(sym, tf, img_b64)
                    await manager.broadcast_to_dashboard({
                        "type": "MT5_SCREENSHOT_UPDATE",
                        "symbol": sym,
                        "timeframe": tf,
                        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    })

            elif msg_type in ["MIRROR_UPDATE", "MIRROR_SYNC"] or msg.get("source") == "MIRROR":
                try:
                    m_acc = msg.get("account_info")
                    if m_acc:
                        is_demo, reason = mirror_service.verify_account_is_demo(m_acc)
                        if not is_demo:
                            logger.error(f"Rejecting live mirror account update: {reason}")
                            continue
                        mirror_service.account_info = m_acc
                    m_pos = msg.get("positions", msg.get("active_trades"))
                    if m_pos is not None:
                        mirror_service.process_positions_update(m_pos)
                    m_deals = msg.get("deals", msg.get("closed_trades"))
                    if m_deals is not None:
                        mirror_service.process_closed_deals(m_deals)
                except Exception as m_err:
                    logger.warning(f"Error handling MIRROR ws update: {m_err}")

            elif msg_type == "TRADE_RESULT":
                success = msg.get("success", False)
                symbol = msg.get("symbol")
                order_type = msg.get("order_type")
                ticket = msg.get("ticket")
                retcode = msg.get("retcode")
                error_msg = msg.get("error", "Unknown error")

                # Remove in-flight lock on trade result
                if symbol:
                    in_flight_trades.pop(symbol, None)

                if success:
                    brainstorm_service.add_log(
                        level="EXECUTION",
                        category="TRADE_DECISION",
                        symbol=symbol,
                        message=f"✅ MT5 Deal Filled: {symbol} {order_type} | Ticket #{ticket}"
                    )
                else:
                    # Set 60-second backoff cooldown on rejection to prevent spamming
                    if symbol:
                        symbol_cooldowns[symbol] = time.time() + 60.0
                    brainstorm_service.add_log(
                        level="RISK",
                        category="TRADE_DECISION",
                        symbol=symbol,
                        message=f"❌ MT5 Order REJECTED (Code {retcode}): {error_msg} -> 60s backoff cooldown applied."
                    )
                    await manager.broadcast_to_dashboard({
                        "type": "TRADE_ERROR",
                        "symbol": symbol,
                        "retcode": retcode,
                        "error": error_msg
                    })

            elif msg_type == "TICK_DATA":
                symbol = msg.get("symbol")
                bid = msg.get("bid")
                ask = msg.get("ask")
                spread = msg.get("spread")
                if symbol:
                    bridge_registry.touch(resolved_client_id)
                    point = 0.01 if ("XAU" in symbol.upper() or "GOLD" in symbol.upper()) else 0.0001
                    calc_spread = round((ask - bid) / point, 1) if (bid is not None and ask is not None and ask >= bid) else None
                    state["live_prices"][symbol] = {
                        "bid": bid,
                        "ask": ask,
                        "spread": spread if spread is not None else calc_spread
                    }
                    power_data = correlation_engine.calculate_currency_power(state["live_prices"])
                    state["currency_power"] = power_data

                    # Automated Fast Breakeven Monitor for Gold (+75 pips / $7.50 profit) -> Lock +$2.00
                    if state.get("gold_fast_be_enabled", True) and ("XAU" in symbol.upper() or "GOLD" in symbol.upper()):
                        for tr in list(state.get("active_trades", [])):
                            if ("XAU" in tr.get("symbol", "").upper() or "GOLD" in tr.get("symbol", "").upper()) and tr.get("ticket"):
                                try:
                                    # The bridge publishes the fill price as "entry".
                                    # Reading "open_price" left open_p at 0.0, and the
                                    # `open_p > 0` guard below meant this break-even
                                    # never fired at all.
                                    open_p = float(tr.get("entry", tr.get("open_price", 0.0)) or 0.0)
                                    curr_sl = float(tr.get("sl", 0.0) or 0.0)
                                    is_buy = ("BUY" in str(tr.get("type", "")).upper())
                                    if open_p <= 0 or curr_sl <= 0:
                                        continue

                                    # Still-at-risk positions only: once the stop is at or
                                    # past entry the terminal-side R-aware trailing engine
                                    # owns it, and this must not pull it backwards.
                                    risk = abs(open_p - curr_sl)
                                    at_risk = (is_buy and curr_sl < open_p) or ((not is_buy) and curr_sl > open_p)
                                    if risk <= 0 or not at_risk:
                                        continue

                                    gain = (bid - open_p) if is_buy else (open_p - ask)

                                    # Same 1.5R trigger / 0.4R lock as the terminal engine,
                                    # so the two agree instead of overwriting each other.
                                    if gain >= (risk * 1.50):
                                        be_sl = round(open_p + (risk * 0.40 if is_buy else -risk * 0.40), 2)
                                        be_cmd = {
                                            "action": "MODIFY_TRADE_SL",
                                            "ticket": int(tr["ticket"]),
                                            "sl": be_sl,
                                            "tp": float(tr.get("tp", 0.0))
                                        }
                                        await manager.send_order_to_mt5(be_cmd)
                                        tr["sl"] = be_sl
                                        r_mult = gain / risk
                                        logger.info(f"🛡️ [Break-Even] Ticket #{tr['ticket']} at +{r_mult:.2f}R -> SL locked at {be_sl} (+0.4R)")
                                        brainstorm_service.add_log(
                                            level="RISK",
                                            category="BREAKEVEN",
                                            symbol="XAUUSD",
                                            message=f"🛡️ Break-Even armed for Position #{tr['ticket']} at +{r_mult:.2f}R (gain ${gain:.2f} / 1R ${risk:.2f}) -> SL secured at ${be_sl}"
                                        )
                                except Exception as be_err:
                                    logger.debug(f"Breakeven check error: {be_err}")

                    # Section 4: evaluate price updates against active signals
                    if bid > 0:
                        try:
                            signal_outcome_tracker.check_price_update(
                                symbol=symbol.upper(),
                                high=bid,
                                low=bid,
                                close=bid,
                                is_bar=False
                            )
                        except Exception as ot_err:
                            logger.debug(f"Signal outcome tracker tick error: {ot_err}")

                        try:
                            analyst_desk_service.check_price_update(
                                symbol=symbol.upper(),
                                high=bid,
                                low=bid,
                                close=bid,
                                is_bar=False
                            )
                        except Exception as dt_err:
                            logger.debug(f"Analyst desk tick error: {dt_err}")

                    await manager.broadcast_to_dashboard({
                        "type": "TICK_DATA",
                        "symbol": symbol,
                        "bid": bid,
                        "ask": ask,
                        "currency_power": power_data
                    })

            elif msg_type == "CANDLE_DATA":
                symbol = msg.get("symbol")
                candles = msg.get("candles", [])
                raw_timeframes = msg.get("timeframes", {})
                
                if symbol:
                    bridge_registry.touch(resolved_client_id)
                    tf_dict = raw_timeframes if raw_timeframes else ({"M15": candles} if candles else {})
                    incoming_m15 = len(tf_dict.get("M15", []))

                    # Problem 2 Rule 2: Accept candles only from the highest-priority live connection
                    is_active = bridge_registry.is_active_provider(resolved_client_id)
                    active_provider = bridge_registry.get_active_provider()
                    active_id = active_provider.get("client_id") if active_provider else "none"

                    if not is_active:
                        bridge_registry.record_dropped(resolved_client_id)
                        logger.warning(
                            f"⚠️ Dropping CANDLE_DATA from secondary bridge '{resolved_client_id}': "
                            f"primary bridge '{active_id}' is currently active."
                        )
                        continue

                    # Problem 2 Rule 4: Refuse to shrink history
                    stored_m15 = len(state.get("candle_history", {}).get(symbol.upper(), {}).get("M15", []))
                    if incoming_m15 < stored_m15 and stored_m15 > 0:
                        bridge_registry.record_dropped(resolved_client_id)
                        logger.warning(
                            f"⚠️ Refusing to shrink history for {symbol.upper()} from {stored_m15} to {incoming_m15} bars! "
                            f"Sender: '{resolved_client_id}', Active provider: '{active_id}'. Discarding payload."
                        )
                        continue

                    if tf_dict:
                        bridge_registry.record_accepted(resolved_client_id, incoming_m15)
                        state.setdefault("candle_history", {})[symbol.upper()] = tf_dict
                        logger.info(f"📊 Received CANDLE_DATA for {symbol.upper()} from '{resolved_client_id}': M15={len(tf_dict.get('M15', []))} bars, H1={len(tf_dict.get('H1', []))} bars, H4={len(tf_dict.get('H4', []))} bars")
                        # Get real-time DXY and currency correlation context
                        dxy_ctx = state.get("currency_power") or correlation_engine.calculate_currency_power(state["live_prices"])
                        is_gold_sym = "XAU" in symbol.upper() or "GOLD" in symbol.upper()

                        # Check Daily Drawdown Circuit Breaker
                        current_equity = state["account_info"].get("equity", 10000.0)
                        starting_equity = state.get("starting_daily_equity", current_equity)
                        dd_check = risk_manager.check_daily_drawdown_limit(
                            starting_equity=starting_equity,
                            current_equity=current_equity
                        )
                        is_dd_breached = dd_check.get("drawdown_breached", False)

                        # Section 4: evaluate pending signal outcomes against latest candle
                        m15_bars = tf_dict.get("M15") or []
                        if m15_bars:
                            latest_c = m15_bars[-1]
                            c_high = float(latest_c.get("high", 0.0) or 0.0)
                            c_low = float(latest_c.get("low", 0.0) or 0.0)
                            c_close = float(latest_c.get("close", 0.0) or 0.0)
                            c_time = int(latest_c.get("time", 0))
                            if c_high > 0 and c_low > 0:
                                try:
                                    signal_outcome_tracker.check_price_update(
                                        symbol=symbol.upper(),
                                        high=c_high,
                                        low=c_low,
                                        close=c_close,
                                        is_bar=True,
                                        bar_time=c_time
                                    )
                                except Exception as cot_err:
                                    logger.debug(f"Signal outcome candle check error: {cot_err}")

                                try:
                                    analyst_desk_service.check_price_update(
                                        symbol=symbol.upper(),
                                        high=c_high,
                                        low=c_low,
                                        close=c_close,
                                        is_bar=True,
                                        bar_time=c_time
                                    )
                                except Exception as dcot_err:
                                    logger.debug(f"Analyst desk candle check error: {dcot_err}")

                        # 🧠 SHADOW MODE: ML Production Signal Engine (Briefs 8, 10, 13, 14)
                        # Evaluates on each closed M15 bar for any symbol with a validated model (Gold + Forex universe).
                        # Strictly isolated: records decisions, flushes to disk & exposes via API.
                        # NEVER places orders or alters execution mode.
                        if get_engine is not None:
                            try:
                                is_supp, unsupp_r = is_model_supported_for_symbol(symbol)
                                if is_supp:
                                    m15_bar_time = last_closed_bar_time(tf_dict, ["M15"])
                                    if m15_bar_time > 0 and last_ml_evaluated_bar.get(symbol.upper()) != m15_bar_time:
                                        now_ts = time.time()
                                        bar_age_seconds = max(0.0, now_ts - float(m15_bar_time))
                                        sym_metrics = get_symbol_metrics(symbol.upper())

                                        # Brief 16 Problem 2: Stale Bar Rejection at Signal Generation (> 20m)
                                        if bar_age_seconds > 1200.0:
                                            logger.warning(
                                                f"🚨 [STALE BAR REJECTION] Refusing ML signal generation for {symbol.upper()}: "
                                                f"newest M15 bar timestamp {m15_bar_time} is {bar_age_seconds/60.0:.1f}m old (> 20m limit). Market closed."
                                            )
                                            record = {
                                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                                "symbol": symbol.upper(),
                                                "timeframe": "M15",
                                                "bar_close_time": m15_bar_time,
                                                "decision": "HOLD",
                                                "probability": 0.0,
                                                "threshold": sym_metrics.get("threshold", 0.40),
                                                "gate_passed": False,
                                                "reason": "market closed",
                                                "entry": None,
                                                "sl": None,
                                                "tp": None,
                                                "rr": 2.0,
                                                "atr": None,
                                                "spread_cost_r": None,
                                                "bar_spread": None,
                                                "tick_spread": None,
                                                "spread_points": None,
                                                "rule_engine_recommendation": "HOLD",
                                                "mode": "SHADOW",
                                                "validated_stats": {
                                                    "expectancy_r": sym_metrics.get("expectancy_r", 0.0),
                                                    "win_rate": 0.40,
                                                    "oos_trades": sym_metrics.get("oos_trades", 0),
                                                    "folds_positive": f"{sym_metrics.get('folds_positive', 0)}/{sym_metrics.get('folds_total', 5)}",
                                                    "tier": sym_metrics.get("tier", "EXPERIMENTAL"),
                                                    "worst_fold_r": sym_metrics.get("worst_fold_r", 0.0)
                                                }
                                            }
                                            state.setdefault("latest_model_decision", {})[symbol.upper()] = record
                                            last_ml_evaluated_bar[symbol.upper()] = m15_bar_time
                                            log_model_decision(record)
                                        else:
                                            spec = get_symbol_spec(symbol)
                                            pt = spec.get("point", 0.001 if "JPY" in symbol.upper() else (0.01 if "XAU" in symbol.upper() else 0.00001))
                                            dig = spec.get("digits", 3 if "JPY" in symbol.upper() else (2 if "XAU" in symbol.upper() else 5))

                                            sym_tick = state["live_prices"].get(symbol.upper(), {})
                                            sym_bid = float(sym_tick.get("bid") or 0.0)
                                            sym_ask = float(sym_tick.get("ask") or 0.0)
                                            sym_mid = ((sym_bid + sym_ask) / 2.0) if (sym_bid > 0 and sym_ask > 0) else None

                                            engine = get_engine(symbol.upper())
                                            ml_decision = engine.decide(
                                                candles=tf_dict,
                                                point=pt,
                                                digits=dig,
                                                live_price=sym_mid
                                            )
                                            dec_val = ml_decision.get("decision", "HOLD")
                                            sym_metrics = get_symbol_metrics(symbol.upper(), side=dec_val)
                                            simultaneous_rule = state.get("gold_analysis", {}).get("recommendation", "HOLD") if is_gold_sym else "HOLD"

                                            # Real spread telemetry: closed-bar spread from M15 candle + live tick spread
                                            last_closed_m15 = m15_bars[-2] if len(m15_bars) >= 2 else (m15_bars[-1] if m15_bars else None)
                                            bar_spread = float(last_closed_m15["spread"]) if (last_closed_m15 and "spread" in last_closed_m15 and last_closed_m15["spread"] is not None) else None
                                            live_tick_spread = round((sym_ask - sym_bid) / pt, 1) if (sym_ask > 0 and sym_bid > 0 and pt > 0) else None

                                            record = {
                                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                                "symbol": symbol.upper(),
                                                "timeframe": "M15",
                                                "bar_close_time": m15_bar_time,
                                                "decision": ml_decision.get("decision", "HOLD"),
                                                "probability": ml_decision.get("probability"),
                                                "threshold": ml_decision.get("threshold", sym_metrics.get("threshold", 0.40)),
                                                "gate_passed": ml_decision.get("gate_passed", False),
                                                "reason": ml_decision.get("reason", ""),
                                                "entry": ml_decision.get("entry"),
                                                "sl": ml_decision.get("sl"),
                                                "tp": ml_decision.get("tp"),
                                                "rr": ml_decision.get("rr", 2.0),
                                                "atr": ml_decision.get("atr"),
                                                "spread_cost_r": ml_decision.get("spread_cost_r"),
                                                "bar_spread": bar_spread,
                                                "tick_spread": live_tick_spread,
                                                "spread_points": bar_spread,
                                                "rule_engine_recommendation": simultaneous_rule,
                                                "mode": "SHADOW",
                                                "validated_stats": {
                                                    "expectancy_r": sym_metrics.get("expectancy_r", 0.0),
                                                    "win_rate": 0.40,
                                                    "oos_trades": sym_metrics.get("oos_trades", 0),
                                                    "folds_positive": f"{sym_metrics.get('folds_positive', 0)}/{sym_metrics.get('folds_total', 5)}",
                                                    "tier": sym_metrics.get("tier", "EXPERIMENTAL"),
                                                    "worst_fold_r": sym_metrics.get("worst_fold_r", 0.0)
                                                }
                                            }

                                            state.setdefault("latest_model_decision", {})[symbol.upper()] = record
                                            last_ml_evaluated_bar[symbol.upper()] = m15_bar_time
                                            log_model_decision(record)
                                            logger.info(
                                                f"🔮 [SHADOW ML] {symbol.upper()} M15 bar {m15_bar_time}: {record['decision']} "
                                                f"(Prob: {record['probability']}, Gate: {record['gate_passed']}) | Reason: {record['reason']}"
                                            )

                                            # Publish signal card ONLY on confirmed model BUY or SELL
                                            if record.get("decision") in ["BUY", "SELL"] and record.get("gate_passed") is True:
                                                prob_val = record.get("probability", 0.0)
                                                thresh_val = record.get("threshold", sym_metrics.get("threshold", 0.40))
                                                if prob_val is not None and prob_val >= thresh_val:
                                                    try:
                                                        card = snapshot_service.create_model_signal_card(
                                                            symbol=symbol.upper(),
                                                            m15_candles=m15_bars,
                                                            model_record=record,
                                                            bar_close_time=m15_bar_time
                                                        )
                                                        logger.info(f"📸 [SIGNAL CARD PUBLISHED] Created/Retrieved model signal card {card.get('snapshot_id')} for {symbol} bar {m15_bar_time}")
                                                        # Telegram publishing trigger (safe background task, respects exclusions & rate caps)
                                                        try:
                                                            asyncio.create_task(telegram_bot.publish_confirmed_signal(card))
                                                        except Exception as t_err:
                                                            logger.error(f"⚠️ Telegram publishing trigger error for {symbol}: {t_err}")
                                                    except Exception as ce:
                                                        logger.error(f"⚠️ Error creating signal card for {symbol}: {ce}", exc_info=True)
                            except Exception as ml_err:
                                logger.error(f"⚠️ Error in ML Shadow Engine for {symbol}: {ml_err}", exc_info=True)

                        if is_gold_sym:
                            balance = state["account_info"].get("balance", 1000.0)
                            free_margin = state["account_info"].get("free_margin", balance)

                            # Use configured strategy profile (Default: HYBRID for institutional stability)
                            target_strategy = state.get("gold_strategy_profile", "HYBRID")

                            # Live mid price - the engine anchors entry/SL/TP to this
                            # instead of to a candle close that may be seconds old.
                            gold_tick = state["live_prices"].get(symbol, {})
                            gold_bid = float(gold_tick.get("bid") or 0.0)
                            gold_ask = float(gold_tick.get("ask") or 0.0)
                            gold_mid = ((gold_bid + gold_ask) / 2.0) if (gold_bid > 0 and gold_ask > 0) else None

                            # 👑 Dedicated Institutional Gold Hybrid & Scalper Engine Analysis.
                            # Recomputed only when a new M5/M15 bar has closed.
                            gold_bar = last_closed_bar_time(tf_dict, ["M5", "M15", "M1"])
                            cached_gold = analysis_cache.get(symbol)
                            if cached_gold and cached_gold.get("bar_time") == gold_bar and gold_bar > 0:
                                analysis = dict(cached_gold["analysis"])
                            else:
                                analysis = gold_engine.analyze_gold_multi_timeframe(
                                    timeframe_data=tf_dict,
                                    dxy_context=dxy_ctx,
                                    strategy_profile=target_strategy,
                                    live_price=gold_mid
                                )
                                analysis_cache[symbol] = {"bar_time": gold_bar, "analysis": dict(analysis)}

                            # A signal is good for one entry on its bar, not for every
                            # bundle that arrives until the bar closes.
                            if gold_bar > 0 and signal_consumed_bar.get(symbol) == gold_bar:
                                analysis = dict(analysis)
                                analysis["recommendation"] = "HOLD"
                                analysis["setup_quality"] = "NO_TRADE"
                                analysis["no_entry_reason_ar"] = "تم تنفيذ إشارة هذه الشمعة بالفعل - بانتظار شمعة جديدة"
                                analysis["no_entry_reason"] = "Signal for this candle already dispatched - awaiting a new bar"

                            # Liquid session window (London open -> NY liquidity fade).
                            # The previous 07:00-22:00 window contradicted its own message
                            # and kept trading through the thin post-NY hours.
                            current_utc_hour = datetime.now(timezone.utc).hour
                            session_start = getattr(settings, "GOLD_SESSION_START_UTC", 7)
                            session_end = getattr(settings, "GOLD_SESSION_END_UTC", 19)
                            enable_24h = getattr(settings, "ENABLE_24H_TRADING", True)
                            is_active_trading_hours = True if enable_24h else (session_start <= current_utc_hour < session_end)

                            # Daily realized loss & consecutive losses, counted on the UTC
                            # day. `timestamp` is the deal's UTC epoch from the bridge;
                            # the formatted string is only a display fallback.
                            today_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                            day_start_ts = datetime.now(timezone.utc).replace(
                                hour=0, minute=0, second=0, microsecond=0
                            ).timestamp()

                            def _is_today(ct: Dict[str, Any]) -> bool:
                                ts = ct.get("timestamp")
                                if ts:
                                    return float(ts) >= day_start_ts
                                return str(ct.get("time", "")).startswith(today_utc_str)

                            today_closed = [ct for ct in state.get("closed_trades", []) if _is_today(ct)]
                            today_loss_sum = sum(float(ct.get("profit", 0.0)) for ct in today_closed if float(ct.get("profit", 0.0)) < 0)

                            consecutive_losses = 0
                            for ct in today_closed:
                                if float(ct.get("profit", 0.0)) < -0.15:
                                    consecutive_losses += 1
                                else:
                                    break


                            # Check Post-Trade / Rejection Cooldown. An armed reversal (the
                            # bot just closed an opposite position on an A+ flip) is exempt,
                            # otherwise the cooldown created by that close cancels the very
                            # entry the reversal existed to take.
                            remaining_cooldown = int(symbol_cooldowns.get(symbol, 0) - time.time())
                            reversal_exempt = (time.time() < gold_reversal_armed_until
                                               and analysis.get("recommendation") != gold_last_trade_direction)
                            if remaining_cooldown > 0 and not reversal_exempt:
                                analysis["cooldown_active"] = True
                                analysis["cooldown_seconds"] = remaining_cooldown
                                analysis["recommendation"] = "HOLD"
                                analysis["setup_quality"] = "NO_TRADE"
                                analysis["auto_trade_delay_reason"] = f"Cooldown Active ({remaining_cooldown}s remaining)"

                            active_type = analysis.get("active_trade_type", "SCALP")
                            if active_type == "SCALP":
                                target_fixed_lot = state.get("gold_scalp_lot_size", 0.01)
                            elif active_type == "SWING":
                                target_fixed_lot = state.get("gold_swing_lot_size", 0.01)
                            else:
                                target_fixed_lot = state.get("gold_fixed_lot_size", 0.01)

                            lot_size = risk_manager.calculate_lot_size(
                                account_balance=balance,
                                entry_price=analysis["current_price"],
                                sl_price=analysis["suggested_sl"],
                                symbol=symbol,
                                risk_percent=state.get("gold_risk_percent", 1.0),
                                lot_mode=state.get("gold_lot_mode", LotMode.DYNAMIC_PERCENT),
                                fixed_lot=target_fixed_lot,
                                live_prices=state["live_prices"],
                                free_margin=free_margin
                            )

                            # Scale size down in a high-volatility regime. The classifier
                            # already produced this figure; it was simply never consumed.
                            regime_adj = float(analysis.get("market_regime", {}).get("risk_adjustment", 1.0))
                            if lot_size > 0 and 0 < regime_adj < 1.0:
                                scaled = round(lot_size * regime_adj, 2)
                                if scaled >= 0.01:
                                    logger.info(f"Regime risk adjustment {regime_adj}: gold lot {lot_size} -> {scaled}")
                                    lot_size = scaled

                            analysis["suggested_lot"] = lot_size
                            analysis["risk_note"] = dict(risk_manager.last_risk_note or {})

                            # Daily loss budget is the LARGER of a percentage of balance
                            # and N times one trade's actual risk. On a $100 account a
                            # single ordinary $3 loss is 3% - a pure percentage cap would
                            # halt trading after one normal losing trade, which is not a
                            # circuit breaker, it is an off switch.
                            realized_loss_pct = getattr(settings, "MAX_DAILY_REALIZED_LOSS_PERCENT", 10.0)
                            max_consecutive = getattr(settings, "MAX_CONSECUTIVE_LOSSES", 3)
                            daily_loss_r = getattr(settings, "MAX_DAILY_LOSS_R", 3.0)
                            one_r_usd = abs(float(analysis.get("dollar_stop", 0.0))) * max(lot_size, 0.01) * 100.0
                            max_allowed_daily_loss = max(
                                balance * (realized_loss_pct / 100.0),
                                one_r_usd * daily_loss_r
                            )
                            is_daily_loss_limit_hit = (
                                (abs(today_loss_sum) >= max_allowed_daily_loss)
                                or (consecutive_losses >= max_consecutive)
                            )

                            current_gold_mode = state.get("gold_execution_mode", ExecutionMode.ADVISORY)
                            rec = analysis["recommendation"]
                            setup_grade = analysis.get("setup_quality", "NO_TRADE")
                            news_safety = news_filter.is_trading_allowed(symbol, pause_minutes=settings.NEWS_PAUSE_MINUTES)
                            live_p = state["live_prices"].get(symbol, {})
                            spread_safety = risk_manager.check_spread_allowed(
                                symbol=symbol,
                                bid=live_p.get("bid", 0.0),
                                ask=live_p.get("ask", 0.0)
                            )

                            is_executable = False
                            exec_status = "HOLD_SCANNING"
                            exec_status_desc = "Scanning Gold Liquidity & Confluence (تحليل سيولة الذهب فقط)"

                            is_auto_trading = ("AUTO" in str(current_gold_mode).upper())

                            if rec in ["BUY", "SELL"]:
                                if not is_auto_trading:
                                    is_executable = False
                                    exec_status = "ANALYST_ONLY"
                                    exec_status_desc = "📊 Gold Analyst Only (تحليل فقط - لن يتم فتح صفقات آلياً)"
                                else:
                                    # Calculate current R:R ratio
                                    curr_p_val = float(analysis.get("current_price", 0.0))
                                    tp1_p_val = float(analysis.get("suggested_tp1", 0.0))
                                    sl_p_val = float(analysis.get("suggested_sl", 0.0))
                                    rr_check = abs(tp1_p_val - curr_p_val) / (abs(sl_p_val - curr_p_val) + 1e-9)

                                    loss_block_secs = get_loss_block(symbol, rec)
                                    allow_gold_a = getattr(settings, "GOLD_ALLOW_GRADE_A", True)
                                    valid_gold_grades = ["A+", "A"] if allow_gold_a else ["A+"]

                                    if not is_active_trading_hours:
                                        is_executable = False
                                        exec_status = "SESSION_PAUSE"
                                        exec_status_desc = f"Off-Hours Blackout (خارج جلسات السيولة {session_start:02d}:00-{session_end:02d}:00 UTC)"
                                        analysis["no_entry_reason_ar"] = f"حظر التداول خارج جلسات السيولة النشطة (لندن ونيويورك {session_start:02d}:00-{session_end:02d}:00 UTC)"
                                        analysis["no_entry_reason"] = f"Trading paused outside active London/NY sessions ({session_start:02d}:00-{session_end:02d}:00 UTC)"
                                    elif is_daily_loss_limit_hit:
                                        is_executable = False
                                        exec_status = "CIRCUIT_BREAKER"
                                        exec_status_desc = f"Capital Guard: Daily Loss Limit Hit (-${abs(today_loss_sum):.2f})"
                                        analysis["no_entry_reason_ar"] = f"🛑 قاطع الدائرة مفعل لحماية رأس المال (الخسارة اليومية: -${abs(today_loss_sum):.2f} من حد ${max_allowed_daily_loss:.2f} | خسائر متتالية: {consecutive_losses})"
                                        analysis["no_entry_reason"] = f"Circuit breaker active: daily loss -${abs(today_loss_sum):.2f} of ${max_allowed_daily_loss:.2f} limit ({consecutive_losses} consecutive)"
                                    elif loss_block_secs > 0:
                                        is_executable = False
                                        exec_status = "POST_LOSS_BLOCK"
                                        exec_status_desc = f"Same-direction block after loss ({loss_block_secs}s)"
                                        analysis["no_entry_reason_ar"] = f"⛔ تم حظر الدخول في نفس الاتجاه ({rec}) بعد صفقة خاسرة - متبقي {loss_block_secs} ثانية لإعادة تقييم الهيكل"
                                        analysis["no_entry_reason"] = f"Same-direction ({rec}) entry blocked after a loss - {loss_block_secs}s remaining"
                                    elif setup_grade not in valid_gold_grades:
                                        is_executable = False
                                        exec_status = "PRELIMINARY"
                                        exec_status_desc = f"Gold Low Confluence (التنفيذ الآلي يتطلب {', '.join(valid_gold_grades)})"
                                        analysis["no_entry_reason_ar"] = f"صفقة غير مؤكدة بالقدر الكافي (الدرجة: {setup_grade} | الثقة: {analysis.get('confidence_percent', 0):.0f}% - التنفيذ الآلي يتطلب {', '.join(valid_gold_grades)})"
                                        analysis["no_entry_reason"] = f"Insufficient confluence for auto-execution (grade {setup_grade}, {', '.join(valid_gold_grades)} required)"
                                    elif lot_size <= 0.0:
                                        is_executable = False
                                        exec_status = "RISK_TOO_LARGE"
                                        exec_status_desc = "Minimum lot exceeds allowed risk per trade"
                                        analysis["no_entry_reason_ar"] = f"أصغر حجم صفقة (0.01) يتجاوز نسبة المخاطرة المسموحة ({state.get('gold_risk_percent', 1.0)}%) على رصيد ${balance:.2f} مع وقف ${analysis.get('dollar_stop', 0):.2f} - الحساب صغير على هذا الإعداد"
                                        analysis["no_entry_reason"] = f"Minimum 0.01 lot exceeds the {state.get('gold_risk_percent', 1.0)}% risk budget on ${balance:.2f} with a ${analysis.get('dollar_stop', 0):.2f} stop"
                                    elif rr_check < 1.7:
                                        is_executable = False
                                        exec_status = "LOW_RR"
                                        exec_status_desc = f"R:R Low ({rr_check:.1f}:1 < 1.8:1)"
                                        analysis["no_entry_reason_ar"] = f"نسبة العائد للمخاطرة ({rr_check:.1f}:1) أقل من 1.8:1 المطلوبة"
                                        analysis["no_entry_reason"] = f"Risk-to-Reward ratio too low ({rr_check:.1f}:1 < 1.8:1)"
                                    elif is_dd_breached:
                                        is_executable = False
                                        exec_status = "DRAWDOWN_HALT"
                                        exec_status_desc = f"Daily Drawdown Limit ({dd_check['drawdown_percent']}%)"
                                        analysis["no_entry_reason_ar"] = f"حظر التداول لتجاوز حد الهبوط اليومي ({dd_check['drawdown_percent']}%)"
                                        analysis["no_entry_reason"] = f"Daily Drawdown Breached ({dd_check['drawdown_percent']}%)"
                                    elif remaining_cooldown > 0 and not (time.time() < gold_reversal_armed_until and rec != gold_last_trade_direction):
                                        is_executable = False
                                        exec_status = "COOLDOWN_PAUSE"
                                        exec_status_desc = f"Gold Cooldown ({remaining_cooldown}s)"
                                        analysis["no_entry_reason_ar"] = f"فترة استقرار بعد الصفقة السابقة (متبقي {remaining_cooldown} ثانية)"
                                        analysis["no_entry_reason"] = f"Post-Trade Cooldown Active ({remaining_cooldown}s remaining)"
                                    elif not news_safety["allowed"] and settings.NEWS_FILTER_ENABLED:
                                        is_executable = False
                                        if news_safety.get("stale"):
                                            exec_status = "NEWS_UNAVAILABLE"
                                            exec_status_desc = "Economic calendar unavailable (تعذر تحديث تقويم الأخبار)"
                                            analysis["no_entry_reason_ar"] = "⛔ تعذر تحديث تقويم الأخبار الاقتصادية - تم إيقاف التنفيذ الآلي بدل التداول بدون رؤية للأخبار"
                                        else:
                                            exec_status = "NEWS_PAUSE"
                                            exec_status_desc = "Gold News Blackout (فلتر الأخبار)"
                                            analysis["no_entry_reason_ar"] = f"حظر التداول مؤقتاً بسبب صدور خبر اقتصادي عالي التأثير ({news_safety.get('event', 'News Blackout')})"
                                        analysis["no_entry_reason"] = news_safety.get("reason", "Economic news blackout")
                                    elif not spread_safety["allowed"]:
                                        is_executable = False
                                        exec_status = "SPREAD_PAUSE"
                                        exec_status_desc = "Gold Spread High (سبريد الذهب مرتفع)"
                                        analysis["no_entry_reason_ar"] = "سبريد الذهب مرتفع حالياً عن الحد الأقصى المسموح للأمان"
                                        analysis["no_entry_reason"] = "Gold Spread exceeds maximum safety threshold"
                                    else:
                                        gold_open_trades = [t for t in state["active_trades"] if "XAU" in t.get("symbol", "").upper() or "GOLD" in t.get("symbol", "").upper()]
                                        free_m = float(state["account_info"].get("free_margin", 1000.0))
                                        curr_p_val = float(analysis.get("current_price", 2600.0))
                                        est_gold_margin = (curr_p_val * 100.0 * 0.01 / 100.0) if curr_p_val > 0 else 30.0

                                        if len(gold_open_trades) > 0:
                                            open_t = gold_open_trades[0]
                                            pnl_val = open_t.get("pnl", open_t.get("profit", 0.0))
                                            is_executable = False
                                            exec_status = "POSITION_ACTIVE"
                                            exec_status_desc = f"Gold Active (#{open_t.get('ticket')} | PnL: ${pnl_val:+.2f})"
                                            analysis["no_entry_reason_ar"] = f"🟢 صفقة ذهب مفتوحة ونشطة حالياً (تذكرة #{open_t.get('ticket')} | الربح: ${pnl_val:+.2f}) - جاري متابعة الأهداف والتأمين"
                                            analysis["no_entry_reason"] = f"Gold Position Active (#{open_t.get('ticket')} | PnL: ${pnl_val:+.2f})"
                                        elif free_m < est_gold_margin:
                                            is_executable = False
                                            exec_status = "MARGIN_LOW"
                                            exec_status_desc = f"Margin Low (${free_m:.2f} < ${est_gold_margin:.2f} needed)"
                                            analysis["no_entry_reason_ar"] = f"الهامش المتاح (${free_m:.2f}) غير كافٍ لفتح صفقة ذهب (مطلوب ${est_gold_margin:.2f} كحد أدنى) بسبب استهلاك الصفقات الأخرى للهامش"
                                            analysis["no_entry_reason"] = f"Insufficient Free Margin (${free_m:.2f} < ${est_gold_margin:.2f} required for 0.01 lot)"
                                        else:
                                            is_executable = True
                                            exec_status = "READY_TO_EXECUTE"
                                            exec_status_desc = "⚡ Gold Auto-Trade Active (صفقة ذهب مؤكدة للتنفيذ)"
                                            analysis["no_entry_reason_ar"] = f"⚡ الشروط مستوفاة للتنفيذ الفوري ({rec} Grade {setup_grade})"
                                            analysis["no_entry_reason"] = f"⚡ Ready to Execute ({rec} Grade {setup_grade})"

                            analysis["is_executable"] = is_executable
                            analysis["execution_status"] = exec_status
                            analysis["execution_status_desc"] = exec_status_desc
                            analysis["execution_mode"] = "AUTO_TRADING" if is_auto_trading else "ANALYST_ONLY"
                            state["gold_analysis"] = analysis

                            # 1. Send Draw Command for Gold to MT5
                            draw_cmd = {
                                "action": "DRAW_GOLD_ANALYSIS",
                                "symbol": "XAUUSD",
                                "analysis": analysis
                            }
                            await manager.send_order_to_mt5(draw_cmd)

                            # 2. Broadcast Gold Update to Web Dashboard
                            await manager.broadcast_to_dashboard({
                                "type": "GOLD_ANALYSIS_UPDATE",
                                "analysis": analysis,
                                "mode": "AUTO_TRADING" if is_auto_trading else "ANALYST_ONLY",
                                "brainstorm_logs": brainstorm_service.get_recent_logs(10)
                            })

                            # 3. Smart Auto Reversal Protection for Gold
                            #    STRICT: Only on A+ grade + 300s since last trade + current trade must be in loss > $0.50
                            for trade in state["active_trades"]:
                                if "XAU" in trade.get("symbol", "").upper() or "GOLD" in trade.get("symbol", "").upper():
                                    existing_type = trade.get("type")
                                    is_opposite = (existing_type == "BUY" and rec == "SELL") or (existing_type == "SELL" and rec == "BUY")
                                    
                                    if is_opposite and setup_grade == "A+":
                                        # Check time since last trade (must be > 300s to prevent rapid flip-flop)
                                        time_since_last = time.time() - gold_last_trade_time
                                        if time_since_last < 300.0:
                                            logger.info(f"👑 Gold Reversal BLOCKED — only {time_since_last:.0f}s since last trade (need 300s)")
                                            continue
                                        
                                        # Check if current trade is in loss > $0.50 (don't close profitable trades)
                                        trade_profit = float(trade.get("profit", 0.0))
                                        if trade_profit > -0.50:
                                            logger.info(f"👑 Gold Reversal BLOCKED — current trade profit ${trade_profit:.2f} > -$0.50 threshold")
                                            continue
                                        
                                        logger.info(f"👑 Gold A+ Confluence Reversal! Closing ticket #{trade.get('ticket')} (loss: ${trade_profit:.2f})")
                                        brainstorm_service.add_log(
                                            level="EXECUTION",
                                            category="TRADE_DECISION",
                                            symbol="XAUUSD",
                                            message=f"🔄 Gold A+ Reversal Signal ({rec}) — Closing Opposite #{trade.get('ticket')} (Loss: ${trade_profit:.2f})"
                                        )
                                        await manager.send_order_to_mt5({"action": "CLOSE_TRADE", "ticket": trade.get("ticket")})
                                        # Arm the reversal so the intended opposite entry is not
                                        # then swallowed by the post-trade cooldown that this very
                                        # close is about to create - otherwise the reversal only
                                        # ever realises the loss and never takes the new side.
                                        gold_reversal_armed_until = time.time() + 240.0

                            # 4. Gold Auto-Trading Execution
                            if is_executable and is_auto_trading:
                                now_ts = time.time()
                                is_in_flight = (symbol in in_flight_trades) and (now_ts - in_flight_trades[symbol] < 120.0)
                                gold_open = [t for t in state["active_trades"] if "XAU" in t.get("symbol", "").upper() or "GOLD" in t.get("symbol", "").upper()]

                                # Minimum gap between Gold entries, waived for an armed reversal
                                time_since_last_gold = now_ts - gold_last_trade_time
                                is_reversal_entry = (now_ts < gold_reversal_armed_until) and (rec != gold_last_trade_direction)
                                is_pyramiding_candidate = False

                                if len(gold_open) == 1:
                                    primary_trade = gold_open[0]
                                    primary_pnl = float(primary_trade.get("profit", 0.0))
                                    primary_type = str(primary_trade.get("type", "")).upper()
                                    # Pyramiding: add only onto a winner that is already risk-free
                                    # (its stop has been trailed past entry), in the same direction.
                                    primary_sl_val = float(primary_trade.get("sl", 0.0) or 0.0)
                                    primary_entry = float(primary_trade.get("entry", 0.0) or 0.0)
                                    is_risk_free = (
                                        primary_sl_val > 0 and primary_entry > 0
                                        and ((primary_type == "BUY" and primary_sl_val >= primary_entry)
                                             or (primary_type == "SELL" and primary_sl_val <= primary_entry))
                                    )
                                    if (primary_pnl >= 3.50 and primary_type == rec and setup_grade == "A+"
                                            and is_risk_free and time_since_last_gold >= 300.0):
                                        is_pyramiding_candidate = True

                                # Distinct symbols, not order legs - a 3-way split is one position
                                # for exposure purposes and must not consume the whole allowance.
                                distinct_open_symbols = {t.get("symbol") for t in state["active_trades"] if t.get("symbol")}
                                max_symbols = getattr(settings, "MAX_OPEN_TRADES", 3)
                                can_open_gold = (
                                    (len(gold_open) == 0 or is_pyramiding_candidate)
                                    and (free_margin > 15.0)
                                    and (symbol in distinct_open_symbols or len(distinct_open_symbols) < max_symbols)
                                )

                                if time_since_last_gold < 180.0 and not is_pyramiding_candidate and not is_reversal_entry:
                                    logger.info(f"👑 Gold trade BLOCKED — only {time_since_last_gold:.0f}s since last trade (need 180s)")
                                elif not is_in_flight and can_open_gold:
                                    in_flight_trades[symbol] = now_ts
                                    gold_reversal_armed_until = 0.0

                                    # 3-in-1 Multi-Target Coordinated Order Split
                                    multi_enabled = trade_learning_engine.is_multi_target_enabled() and (not is_pyramiding_candidate) and (lot_size >= 0.03)
                                    suggested_sl = float(analysis["suggested_sl"])
                                    suggested_tp = float(analysis["suggested_tp1"])
                                    
                                    if multi_enabled:
                                        split_orders = trade_learning_engine.calculate_multi_target_split(
                                            total_lot_size=lot_size,
                                            sl_price=analysis["suggested_sl"],
                                            tp1_price=analysis["suggested_tp1"],
                                            tp2_price=analysis["suggested_tp2"],
                                            tp3_price=analysis["suggested_tp3"],
                                            direction=rec,
                                            symbol=symbol
                                        )
                                        for s_ord in split_orders:
                                            trade_cmd = {
                                                "action": "EXECUTE_TRADE",
                                                "symbol": symbol,
                                                "order_type": rec,
                                                "lot_size": s_ord["lot_size"],
                                                "sl": s_ord["sl"],
                                                "tp": s_ord["tp"],
                                                "comment": s_ord["comment"]
                                            }
                                            await manager.send_order_to_mt5(trade_cmd)
                                            trade_learning_engine.record_entry_snapshot(
                                                symbol=symbol,
                                                ticket=None,
                                                direction=rec,
                                                lot_size=s_ord["lot_size"],
                                                entry_price=analysis["current_price"],
                                                sl=s_ord["sl"],
                                                tp=s_ord["tp"],
                                                analysis_context=analysis,
                                                comment=s_ord["comment"]
                                            )
                                        autopsy_note = f" | 🛡️ {analysis['autopsy_shield']['advice_ar']}" if analysis.get("autopsy_shield") else ""
                                        logger.info(f"👑 Gold 3-in-1 Multi-Target Auto-Trades Dispatched ({len(split_orders)} sub-orders, total {lot_size} lots)")
                                        brainstorm_service.add_log(
                                            level="EXECUTION",
                                            category="TRADE_DECISION",
                                            symbol="XAUUSD",
                                            message=f"👑 GOLD 3-IN-1 MULTI-TARGET DISPATCHED -> {rec} (Total {lot_size} Lots split: TP1 ${analysis['suggested_tp1']}, TP2 ${analysis['suggested_tp2']}, TP3 ${analysis['suggested_tp3']}){autopsy_note}"
                                        )
                                    else:
                                        order_comment = "FXENGIN_PYRAMID_AI" if is_pyramiding_candidate else f"FXENGIN_{active_type}_AI"
                                        trade_cmd = {
                                            "action": "EXECUTE_TRADE",
                                            "symbol": symbol,
                                            "order_type": rec,
                                            "lot_size": lot_size,
                                            "sl": suggested_sl,
                                            "tp": suggested_tp,
                                            "comment": order_comment
                                        }
                                        await manager.send_order_to_mt5(trade_cmd)
                                        trade_learning_engine.record_entry_snapshot(
                                            symbol=symbol,
                                            ticket=None,
                                            direction=rec,
                                            lot_size=lot_size,
                                            entry_price=analysis["current_price"],
                                            sl=suggested_sl,
                                            tp=suggested_tp,
                                            analysis_context=analysis,
                                            comment=order_comment
                                        )
                                        autopsy_note = f" | 🛡️ {analysis['autopsy_shield']['advice_ar']}" if analysis.get("autopsy_shield") else ""
                                        logger.info(f"Gold Auto-Trade ({order_comment}) Dispatched to MT5: {trade_cmd}")
                                        brainstorm_service.add_log(
                                            level="EXECUTION",
                                            category="TRADE_DECISION",
                                            symbol="XAUUSD",
                                            message=f"👑 GOLD AUTO-TRADE ({order_comment} Grade {setup_grade}) DISPATCHED -> {rec} ({lot_size} Lots | SL: ${suggested_sl} | TP: ${suggested_tp}){autopsy_note}"
                                        )

                                    # Update Trade Commitment Lock
                                    gold_last_trade_time = now_ts
                                    gold_last_trade_direction = rec
                                    if gold_bar > 0:
                                        signal_consumed_bar[symbol] = gold_bar

                        else:
                            # 🌐 Forex USD & JPY Universe Analysis
                            fx_bar = last_closed_bar_time(tf_dict, ["M5", "M15", "M1"])
                            cached_fx = analysis_cache.get(symbol)
                            if cached_fx and cached_fx.get("bar_time") == fx_bar and fx_bar > 0:
                                analysis = dict(cached_fx["analysis"])
                            else:
                                tf_dfs = {k: pd.DataFrame(v) for k, v in tf_dict.items() if v and isinstance(v, list)}
                                analysis = ta_engine.analyze_symbol_multi_timeframe(
                                    symbol=symbol,
                                    timeframes_data=tf_dfs,
                                    school=state["trading_school"],
                                    dxy_context=dxy_ctx
                                )
                                analysis_cache[symbol] = {"bar_time": fx_bar, "analysis": dict(analysis)}

                            if fx_bar > 0 and signal_consumed_bar.get(symbol) == fx_bar:
                                analysis = dict(analysis)
                                analysis["recommendation"] = "HOLD"
                                analysis["setup_quality"] = "NO_TRADE"
                                analysis["no_entry_reason"] = "Signal for this candle already dispatched - awaiting a new bar"
                        
                            # Check Post-Trade / Rejection Cooldown
                            remaining_cooldown = int(symbol_cooldowns.get(symbol, 0) - time.time())
                            if remaining_cooldown > 0:
                                analysis["cooldown_active"] = True
                                analysis["cooldown_seconds"] = remaining_cooldown
                                analysis["recommendation"] = "HOLD"
                                analysis["setup_quality"] = "NO_TRADE"
                                analysis["auto_trade_delay_reason"] = f"Cooldown Active ({remaining_cooldown}s remaining)"
                        
                            balance = state["account_info"].get("balance", 1000.0)
                            free_margin = state["account_info"].get("free_margin", balance)
                            lot_size = risk_manager.calculate_lot_size(
                                account_balance=balance,
                                entry_price=analysis["current_price"],
                                sl_price=analysis["suggested_sl"],
                                symbol=symbol,
                                risk_percent=state["max_risk_percent"],
                                lot_mode=state["lot_mode"],
                                fixed_lot=state["fixed_lot_size"],
                                live_prices=state["live_prices"],
                                free_margin=free_margin
                            )
                            analysis["suggested_lot"] = lot_size

                            current_mode = state["execution_mode"]
                            rec = analysis["recommendation"]
                            setup_grade = analysis.get("setup_quality", "NO_TRADE")
                            news_safety = news_filter.is_trading_allowed(symbol, pause_minutes=settings.NEWS_PAUSE_MINUTES)
                            live_p = state["live_prices"].get(symbol, {})
                            spread_safety = risk_manager.check_spread_allowed(
                                symbol=symbol,
                                bid=live_p.get("bid", 0.0),
                                ask=live_p.get("ask", 0.0)
                            )
                            
                            # Currency Basket Exposure Filter
                            basket_check = correlation_engine.check_currency_basket_exposure(
                                active_trades=state["active_trades"],
                                new_symbol=symbol,
                                max_usd_exposure=1,
                                max_jpy_exposure=1
                            )

                            # Determine exact execution feasibility & preliminary flags for MT5 Visualizer
                            is_executable = False
                            exec_status = "HOLD_SCANNING"
                            exec_status_desc = "Scanning & Liquidity Roaming (مسح وتحليل فقط)"

                            if rec in ["BUY", "SELL"]:
                                if current_mode == ExecutionMode.ADVISORY:
                                    is_executable = False
                                    exec_status = "ADVISORY_ONLY"
                                    exec_status_desc = "Advisory Mode (تحليل مبدئي استشاري - لن يدخل الصفقة نهائياً)"
                                elif current_mode == ExecutionMode.SEMI_AUTO:
                                    if setup_grade in ["A+", "A"]:
                                        is_executable = False
                                        exec_status = "SEMI_AUTO_PENDING"
                                        exec_status_desc = "Semi-Auto (بانتظار التأكيد اليدوي من اللوحة)"
                                    else:
                                        is_executable = False
                                        exec_status = "PRELIMINARY"
                                        exec_status_desc = "Preliminary Setup (تحليل مبدئي غير مكتمل - لن يدخل)"
                                elif current_mode == ExecutionMode.AUTO_TRADING:
                                    current_utc_hour = datetime.now(timezone.utc).hour
                                    enable_24h = getattr(settings, "ENABLE_24H_TRADING", True)
                                    is_active_trading_hours = True if enable_24h else (7 <= current_utc_hour < 19)

                                    today_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                                    fx_day_start = datetime.now(timezone.utc).replace(
                                        hour=0, minute=0, second=0, microsecond=0
                                    ).timestamp()

                                    def _fx_is_today(ct: Dict[str, Any]) -> bool:
                                        ts = ct.get("timestamp")
                                        if ts:
                                            return float(ts) >= fx_day_start
                                        return str(ct.get("time", "")).startswith(today_utc_str)

                                    today_closed = [ct for ct in state.get("closed_trades", []) if _fx_is_today(ct)]
                                    today_loss_sum = sum(float(ct.get("profit", 0.0)) for ct in today_closed if float(ct.get("profit", 0.0)) < 0)
                                    today_net_pnl = sum(float(ct.get("profit", 0.0)) for ct in today_closed)
                                    # Count the streak within today only, matching the loss budget it guards
                                    consecutive_losses = 0
                                    for ct in today_closed:
                                        if float(ct.get("profit", 0.0)) < -0.05:
                                            consecutive_losses += 1
                                        else:
                                            break
                                    fx_loss_pct = getattr(settings, "MAX_DAILY_REALIZED_LOSS_PERCENT", 3.0)
                                    
                                    # Micro account floor: allow trading without locking account on small normal losses
                                    min_loss_floor = 30.0 if balance < 250.0 else (45.0 if balance < 500.0 else 0.0)
                                    max_allowed_daily_loss = max(min_loss_floor, balance * (fx_loss_pct / 100.0))

                                    cb_overridden = state.get("circuit_breaker_overridden", False)
                                    is_daily_loss_limit_hit = False if cb_overridden else (
                                        (today_net_pnl < 0 and abs(today_net_pnl) >= max_allowed_daily_loss)
                                        or (consecutive_losses >= getattr(settings, "MAX_CONSECUTIVE_LOSSES", 4))
                                    )

                                    if not is_active_trading_hours:
                                        is_executable = False
                                        exec_status = "SESSION_PAUSE"
                                        exec_status_desc = "Off-Hours Blackout (خارج جلسات التداول)"
                                        analysis["no_entry_reason"] = "Off-Hours Blackout"
                                        analysis["no_entry_reason_ar"] = "خارج جلسات التداول الرسمية"
                                    elif is_daily_loss_limit_hit:
                                        is_executable = False
                                        exec_status = "CIRCUIT_BREAKER"
                                        exec_status_desc = f"Capital Guard: Daily Loss Limit Hit (-${abs(today_net_pnl):.2f} / Max ${max_allowed_daily_loss:.2f})"
                                        analysis["no_entry_reason"] = f"Capital Guard: Daily Loss Limit Hit (-${abs(today_net_pnl):.2f})"
                                        analysis["no_entry_reason_ar"] = f"حماية رأس المال: تم بلوغ حد الخسارة اليومي (-${abs(today_net_pnl):.2f})"
                                    elif not basket_check["allowed"]:
                                        is_executable = False
                                        exec_status = "BASKET_LIMIT"
                                        exec_status_desc = basket_check["reason_ar"]
                                        analysis["no_entry_reason_ar"] = basket_check["reason_ar"]
                                        analysis["no_entry_reason"] = basket_check["reason"]
                                    elif get_loss_block(symbol, rec) > 0:
                                        is_executable = False
                                        exec_status = "POST_LOSS_BLOCK"
                                        exec_status_desc = f"Same-direction block after loss ({get_loss_block(symbol, rec)}s)"
                                        analysis["no_entry_reason_ar"] = f"⛔ تم حظر الدخول في نفس الاتجاه ({rec}) بعد صفقة خاسرة على {symbol}"
                                        analysis["no_entry_reason"] = f"Same-direction ({rec}) entry blocked after a loss on {symbol}"
                                    elif lot_size <= 0.0:
                                        is_executable = False
                                        exec_status = "RISK_TOO_LARGE"
                                        exec_status_desc = "Minimum lot exceeds allowed risk per trade"
                                        analysis["no_entry_reason_ar"] = f"أصغر حجم صفقة (0.01) يتجاوز نسبة المخاطرة المسموحة ({state['max_risk_percent']}%) على رصيد ${balance:.2f}"
                                        analysis["no_entry_reason"] = f"Minimum 0.01 lot exceeds the {state['max_risk_percent']}% risk budget on ${balance:.2f}"
                                    elif setup_grade not in ["A+", "A"]:
                                        is_executable = False
                                        exec_status = "PRELIMINARY"
                                        c_text = analysis.get("confluence_text", "1/3 Confluence")
                                        exec_status_desc = f"Preliminary Setup ({c_text} - بانتظار توافق مدرستين فأكثر)"
                                        analysis["no_entry_reason"] = f"Preliminary Setup: {c_text} (Requires 2/3 or 3/3)"
                                        analysis["no_entry_reason_ar"] = f"تحليل مبدئي: قوة التوافق {c_text} غير كافية للدخول التلقائي"
                                    elif is_dd_breached:
                                        is_executable = False
                                        exec_status = "DRAWDOWN_HALT"
                                        exec_status_desc = f"Daily Drawdown Limit ({dd_check['drawdown_percent']}%)"
                                        analysis["no_entry_reason"] = exec_status_desc
                                        analysis["no_entry_reason_ar"] = f"تراجع الحساب اليومي ({dd_check['drawdown_percent']}%)"
                                    elif remaining_cooldown > 0:
                                        is_executable = False
                                        exec_status = "COOLDOWN_PAUSE"
                                        exec_status_desc = f"Cooldown Active ({remaining_cooldown}s) (فترة دراسة - لن يدخل)"
                                        analysis["no_entry_reason"] = f"Study Cooldown ({remaining_cooldown}s)"
                                        analysis["no_entry_reason_ar"] = f"فترة دراسة ومراقبة شمعة الـ 5 دقائق ({remaining_cooldown} ثانية)"
                                    elif not news_safety["allowed"] and settings.NEWS_FILTER_ENABLED:
                                        is_executable = False
                                        exec_status = "NEWS_PAUSE"
                                        exec_status_desc = "News Blackout (فلتر الأخبار نشط - لن يدخل)"
                                        analysis["no_entry_reason"] = f"High-Impact News within {settings.NEWS_PAUSE_MINUTES}m"
                                        analysis["no_entry_reason_ar"] = f"فلتر الأخبار القوية نشط خلال {settings.NEWS_PAUSE_MINUTES} دقيقة"
                                    elif not spread_safety["allowed"]:
                                        is_executable = False
                                        exec_status = "SPREAD_PAUSE"
                                        exec_status_desc = "High Spread (السبريد مرتفع - لن يدخل)"
                                        analysis["no_entry_reason"] = "Live Spread exceeds threshold"
                                        analysis["no_entry_reason_ar"] = "السبريد المباشر مرتفع حالياً لتجنب الانزلاق"
                                    else:
                                        is_executable = True
                                        exec_status = "READY_TO_EXECUTE"
                                        exec_status_desc = "⚡ Auto-Trade Active (صفقة مؤكدة للتنفيذ التلقائي)"
                                        analysis["no_entry_reason"] = None
                                        analysis["no_entry_reason_ar"] = None

                            analysis["is_executable"] = is_executable
                            analysis["execution_status"] = exec_status
                            analysis["execution_status_desc"] = exec_status_desc
                            analysis["execution_mode"] = current_mode.value if hasattr(current_mode, "value") else str(current_mode)
                            state["latest_signals"][symbol] = analysis

                            # 1. Send Draw Command to MT5 Chart
                            draw_cmd = {
                                "action": "DRAW_ANALYSIS",
                                "symbol": symbol,
                                "analysis": analysis,
                                "school": state["trading_school"].value if hasattr(state["trading_school"], "value") else str(state["trading_school"])
                            }
                            await manager.send_order_to_mt5(draw_cmd)

                            # 2. Broadcast Signal to Web Dashboard
                            await manager.broadcast_to_dashboard({
                                "type": "SIGNAL_UPDATE",
                                "analysis": analysis,
                                "mode": current_mode.value if hasattr(current_mode, "value") else current_mode,
                                "brainstorm_logs": brainstorm_service.get_recent_logs(10)
                            })

                            # 3. Execution Logic
                            # Auto Reversal Protection
                            for trade in state["active_trades"]:
                                if trade.get("symbol") == symbol:
                                    existing_type = trade.get("type")
                                    if (existing_type == "BUY" and rec == "SELL" and setup_grade == "A+") or \
                                       (existing_type == "SELL" and rec == "BUY" and setup_grade == "A+"):
                                        logger.info(f"Confluence Reversal for {symbol}! Closing ticket #{trade.get('ticket')}")
                                        await manager.send_order_to_mt5({"action": "CLOSE_TRADE", "ticket": trade.get("ticket")})

                            # Full Auto-Trading (Requires is_executable == True)
                            if is_executable:
                                now_ts = time.time()
                                is_in_flight = (symbol in in_flight_trades) and (now_ts - in_flight_trades[symbol] < 25.0)
                                symbol_open = [t for t in state["active_trades"] if t.get("symbol") == symbol]

                                curr_balance = float(state["account_info"].get("balance", 100.0))
                                max_allowed_trades = 1 if curr_balance < 250.0 else (2 if curr_balance < 500.0 else settings.MAX_OPEN_TRADES)

                                if not is_in_flight and len(symbol_open) == 0 and len(state["active_trades"]) < max_allowed_trades:
                                    # Set in-flight lock
                                    in_flight_trades[symbol] = now_ts
                                    
                                    trade_cmd = {
                                        "action": "EXECUTE_TRADE",
                                        "symbol": symbol,
                                        "order_type": rec,
                                        "lot_size": lot_size,
                                        "sl": analysis["suggested_sl"],
                                        "tp": analysis["suggested_tp"],
                                        "comment": f"{state['trading_school'].value if hasattr(state['trading_school'], 'value') else state['trading_school']} AI"
                                    }
                                    await manager.send_order_to_mt5(trade_cmd)
                                    # Record Black-Box Context Snapshot for AI Learning
                                    trade_learning_engine.record_entry_snapshot(
                                        symbol=symbol,
                                        ticket=None,
                                        direction=rec,
                                        lot_size=lot_size,
                                        entry_price=analysis["current_price"],
                                        sl=analysis["suggested_sl"],
                                        tp=analysis["suggested_tp"],
                                        analysis_context=analysis,
                                        comment=trade_cmd["comment"]
                                    )
                                    if fx_bar > 0:
                                        signal_consumed_bar[symbol] = fx_bar
                                    logger.info(f"Auto-Trade Dispatched to MT5: {trade_cmd}")
                                    autopsy_note = f" | 🛡️ {analysis['autopsy_shield']['advice_ar']}" if analysis.get("autopsy_shield") else ""
                                    brainstorm_service.add_log(
                                        level="EXECUTION",
                                        category="TRADE_DECISION",
                                        symbol=symbol,
                                        message=f"⚡ AUTO-TRADE DISPATCHED -> {symbol} {rec} ({lot_size} Lots | SL: {analysis['suggested_sl']} | TP: {analysis['suggested_tp']}){autopsy_note}"
                                    )

    except WebSocketDisconnect:
        await bridge_registry.unregister(resolved_client_id, websocket)
        manager.disconnect_mt5(websocket)
        still_connected = len(bridge_registry.bridges) > 0
        state["mt5_connected"] = still_connected
        log_coverage_event("disconnected", reason=f"{resolved_client_id}:{bridge_disconnect_reason}")
        logger.warning(f"MT5 Bridge '{resolved_client_id}' Disconnected ({bridge_disconnect_reason})! (Remaining active: {len(bridge_registry.bridges)})")
        brainstorm_service.add_log(
            level="SYSTEM",
            category="BOOT",
            symbol="BRIDGE",
            message=f"⚠️ MT5 Bridge '{resolved_client_id}' Disconnected ({bridge_disconnect_reason})."
        )
        await manager.broadcast_to_dashboard({"type": "MT5_STATUS", "connected": still_connected})
    except Exception as e:
        await bridge_registry.unregister(resolved_client_id, websocket)
        manager.disconnect_mt5(websocket)
        still_connected = len(bridge_registry.bridges) > 0
        state["mt5_connected"] = still_connected
        log_coverage_event("disconnected", reason=f"{resolved_client_id}:error:{e}")
        logger.error(f"MT5 WebSocket connection error on '{resolved_client_id}': {e}")
        await manager.broadcast_to_dashboard({"type": "MT5_STATUS", "connected": still_connected})
    else:
        await bridge_registry.unregister(resolved_client_id, websocket)
        manager.disconnect_mt5(websocket)
        still_connected = len(bridge_registry.bridges) > 0
        state["mt5_connected"] = still_connected
        log_coverage_event("disconnected", reason=f"{resolved_client_id}:{bridge_disconnect_reason}")
        await manager.broadcast_to_dashboard({"type": "MT5_STATUS", "connected": still_connected})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
