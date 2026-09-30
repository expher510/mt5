import os
from enum import Enum
from pydantic import field_validator
from pydantic_settings import BaseSettings

class ExecutionMode(str, Enum):
    ADVISORY = "ADVISORY"        # Analysis & Signals only, no auto execution
    SEMI_AUTO = "SEMI_AUTO"      # Requires user confirmation via Dashboard
    AUTO_TRADING = "AUTO_TRADING"  # Executes trades directly on MT5

class TradingSchool(str, Enum):
    TRI_SCHOOL_CONSENSUS = "TRI_SCHOOL_CONSENSUS" # Unified Classical + Fibonacci + SMC Confluence
    CLASSICAL = "CLASSICAL"      # Support/Resistance, Trendlines, Double Top/Bottom
    FIBONACCI = "FIBONACCI"      # Golden Ratio (0.618 / 0.50), Extensions (1.272, 1.618), OTE
    SMC = "SMC"                  # Smart Money Concepts, Order Blocks, Fair Value Gaps, BOS / Sweeps

class TradingStyle(str, Enum):
    AUTO_ADAPTIVE = "AUTO_ADAPTIVE"  # Autonomous AI: Selects Scalping vs Swing dynamically per market state
    SCALPING = "SCALPING"            # Fast M5/M15 Scalp setups with quick target TP & tight SL
    SWING = "SWING"                  # H1/H4 Day/Swing trading setups with multi-day target TP

class LotMode(str, Enum):
    DYNAMIC_PERCENT = "DYNAMIC_PERCENT"  # Dynamic lot calculation from account equity %
    FIXED_LOT = "FIXED_LOT"              # Fixed manual lot size (e.g. 0.05, 0.10)

class Settings(BaseSettings):
    PROJECT_NAME: str = "FXENGIN Multi-School Forex Analyst Core"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-mt5-vps-key-2026")
    
    # System Execution Mode (Default: ADVISORY for safety)
    EXECUTION_MODE: ExecutionMode = ExecutionMode.ADVISORY
    
    # Strategy & Trading School Defaults (Default: Unified 3-School + Autonomous Multi-Timeframe)
    TRADING_SCHOOL: TradingSchool = TradingSchool.TRI_SCHOOL_CONSENSUS
    TRADING_STYLE: TradingStyle = TradingStyle.AUTO_ADAPTIVE
    LOT_MODE: LotMode = LotMode.DYNAMIC_PERCENT
    FIXED_LOT_SIZE: float = 0.01
    
    # Risk Management Defaults (Strict Capital Preservation with Active Trading Room)
    MAX_RISK_PER_TRADE_PERCENT: float = 1.0
    DEFAULT_STOP_LOSS_PIPS: float = 15.0
    DEFAULT_TAKE_PROFIT_PIPS: float = 35.0
    MAX_OPEN_TRADES: int = 3
    MAX_DAILY_DRAWDOWN_PERCENT: float = 12.0  # Daily equity circuit breaker
    MAX_DAILY_DRAWDOWN_DOLLARS: float = 0.0   # Optional absolute cap (0 = percentage only)
    MAX_DAILY_REALIZED_LOSS_PERCENT: float = 10.0  # Realized loss cap before halting
    # Hard ceiling on the TRUE risk of a single trade. Small accounts cannot always
    # hit MAX_RISK_PER_TRADE_PERCENT because 0.01 lot is the broker's floor; this is
    # the line that must never be crossed regardless.
    MAX_ABSOLUTE_RISK_PERCENT: float = 6.0
    # The daily loss cap is also expressed in R (multiples of one trade's risk), so
    # a small account is not halted by a single ordinary losing trade.
    MAX_DAILY_LOSS_R: float = 3.0
    MAX_CONSECUTIVE_LOSSES: int = 3
    NEWS_FILTER_ENABLED: bool = True
    NEWS_PAUSE_MINUTES: int = 20
    NEWS_REFRESH_SECONDS: int = 600   # Background economic-calendar refresh interval
    
    # 24/7 Smart Continuous Trading & Session Architecture
    ENABLE_24H_TRADING: bool = True   # Trades 24 hours across Asian, London & NY sessions with live spread protection
    OFF_PEAK_MAX_SPREAD_POINTS: float = 40.0 # Extra strict spread threshold during off-peak hours (e.g. rollover 21:00-23:00 UTC)
    GOLD_ALLOW_GRADE_A: bool = True   # Allows high-confluence Grade A setups in addition to Grade A+ for Gold
    
    GOLD_SESSION_START_UTC: int = 7   # London open (reference)
    GOLD_SESSION_END_UTC: int = 19    # NY liquidity fade (reference)
    GOLD_POST_LOSS_BLOCK_SECONDS: int = 900  # Same-direction re-entry block after a loss
    
    # Telegram Integration via n8n Webhook, Direct Bot API & Safety Controls
    N8N_ENABLED: bool = os.getenv("N8N_ENABLED", "false").lower() in ("true", "1", "yes")
    N8N_WEBHOOK_URL: str = os.getenv("N8N_WEBHOOK_URL", "")
    N8N_WEBHOOK_URL_FOREX: str = os.getenv("N8N_WEBHOOK_URL_FOREX", "")
    N8N_WEBHOOK_URL_GOLD: str = os.getenv("N8N_WEBHOOK_URL_GOLD", "")
    N8N_WEBHOOK_URL_INDEX: str = os.getenv("N8N_WEBHOOK_URL_INDEX", "")

    # Direct Telegram Bot API (Optional alternative to n8n)
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID_GOLD: str = os.getenv("TELEGRAM_CHAT_ID_GOLD", "")
    TELEGRAM_CHAT_ID_FOREX: str = os.getenv("TELEGRAM_CHAT_ID_FOREX", "")
    TELEGRAM_CHAT_ID_INDEX: str = os.getenv("TELEGRAM_CHAT_ID_INDEX", "")

    # Dual MT5 Accounts Configuration (Account 1: Gold, Account 2: Forex)
    MT5_MIRROR_LOGIN_GOLD: int = int(os.getenv("MT5_MIRROR_LOGIN_GOLD", "8058543"))
    MT5_MIRROR_SERVER_GOLD: str = os.getenv("MT5_MIRROR_SERVER_GOLD", "SCFMLimited-Demo2")
    MT5_MIRROR_INVESTOR_PASSWORD_GOLD: str = os.getenv("MT5_MIRROR_INVESTOR_PASSWORD_GOLD", "")

    MT5_MIRROR_LOGIN_FOREX: int = int(os.getenv("MT5_MIRROR_LOGIN_FOREX", "160767360"))
    MT5_MIRROR_SERVER_FOREX: str = os.getenv("MT5_MIRROR_SERVER_FOREX", "SCFMLimited-Demo2")
    MT5_MIRROR_INVESTOR_PASSWORD_FOREX: str = os.getenv("MT5_MIRROR_INVESTOR_PASSWORD_FOREX", "")

    MT5_MIRROR_LOGIN_SPLIT: int = int(os.getenv("MT5_MIRROR_LOGIN_SPLIT", "160767360"))
    TELEGRAM_DRY_RUN: bool = os.getenv("TELEGRAM_DRY_RUN", "false").lower() in ("true", "1", "yes")
    TELEGRAM_GLOBAL_MAX_HOURLY_SIGNALS: int = int(os.getenv("TELEGRAM_GLOBAL_MAX_HOURLY_SIGNALS", "25"))
    TELEGRAM_MAX_HOURLY_SIGNALS: int = int(os.getenv("TELEGRAM_MAX_HOURLY_SIGNALS", "10"))
    TELEGRAM_MAX_DAILY_SIGNALS: int = int(os.getenv("TELEGRAM_MAX_DAILY_SIGNALS", "60"))
    TELEGRAM_HEARTBEAT_MINUTES: int = int(os.getenv("TELEGRAM_HEARTBEAT_MINUTES", "60"))
    TELEGRAM_MAX_HOURLY_NOTICES: int = 1
    TELEGRAM_ENABLE_WAITING_NOTICES: bool = os.getenv("TELEGRAM_ENABLE_WAITING_NOTICES", "false").lower() in ("true", "1", "yes")
    # Trade Events Publishing Settings
    TELEGRAM_PUBLISH_TRADES: bool = os.getenv("TELEGRAM_PUBLISH_TRADES", "true").lower() in ("true", "1", "yes")
    TELEGRAM_MAX_HOURLY_TRADES: int = int(os.getenv("TELEGRAM_MAX_HOURLY_TRADES", "10"))
    TELEGRAM_MAX_DAILY_TRADES: int = int(os.getenv("TELEGRAM_MAX_DAILY_TRADES", "60"))
    # Model Signals Publishing
    TELEGRAM_PUBLISH_MODEL_SIGNALS: bool = os.getenv("TELEGRAM_PUBLISH_MODEL_SIGNALS", "true").lower() in ("true", "1", "yes")
    # Analyst Desk Manual Trades Publishing
    TELEGRAM_PUBLISH_DESK: bool = os.getenv("TELEGRAM_PUBLISH_DESK", "true").lower() in ("true", "1", "yes")
    # Signal Result Publishing Settings
    TELEGRAM_PUBLISH_RESULTS: bool = os.getenv("TELEGRAM_PUBLISH_RESULTS", "true").lower() in ("true", "1", "yes")
    TELEGRAM_REPORT_INTERVAL_HOURS: int = int(os.getenv("TELEGRAM_REPORT_INTERVAL_HOURS", "12"))

    TELEGRAM_ALLOWED_SYMBOLS: list[str] = [
        "AUDJPY", "AUDUSD", "CADJPY", "CHFJPY", "EURJPY", "EURUSD",
        "GBPJPY", "GBPUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD"
    ]
    TELEGRAM_EXCLUDED_SYMBOLS: list[str] = []

    @field_validator("TELEGRAM_ALLOWED_SYMBOLS", mode="before")
    @classmethod
    def parse_allowed_symbols(cls, v):
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    import json
                    return json.loads(v)
                except Exception:
                    pass
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(s).strip().upper() for s in v]
        return [
            "AUDJPY", "AUDUSD", "CADJPY", "CHFJPY", "EURJPY", "EURUSD",
            "GBPJPY", "GBPUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD"
        ]

    @field_validator("TELEGRAM_EXCLUDED_SYMBOLS", mode="before")
    @classmethod
    def parse_excluded_symbols(cls, v):
        if not v:
            return []
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    import json
                    return json.loads(v)
                except Exception:
                    pass
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(s).strip().upper() for s in v]
        return []
    # MT5 Communication & Security Tokens (Rotated & Hardened)
    MT5_BRIDGE_TOKEN: str = os.getenv("MT5_BRIDGE_TOKEN", "fx_bridge_sec_993427f1c84b")
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "fx_admin_sec_482910fae17c")
    
    # Active Forex Universe (USD Basket & JPY Crosses - Strictly NO Gold)
    USD_BASKET_SYMBOLS: list[str] = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD"]
    JPY_BASKET_SYMBOLS: list[str] = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]
    ACTIVE_SYMBOLS: list[str] = [
        "EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY",
        "AUDUSD", "USDCAD", "USDCHF", "AUDJPY", "CADJPY", "CHFJPY"
    ]
    
    # Dedicated Gold Trader Universe & Engine Settings
    GOLD_SYMBOL: str = "XAUUSD"
    GOLD_ENABLED: bool = True
    GOLD_DEFAULT_EXECUTION_MODE: str = "ANALYST_ONLY"  # "ANALYST_ONLY" or "AUTO_TRADING"
    GOLD_STRATEGY_PROFILE: str = "HYBRID"              # "HYBRID", "SCALPING", or "SWING"
    GOLD_MAX_RISK_PERCENT: float = 1.0
    GOLD_SCALP_LOT_SIZE: float = 0.01
    GOLD_SWING_LOT_SIZE: float = 0.01
    GOLD_MIN_ADX_TREND: float = 22.0                   # Minimum ADX to allow trend trading (blocks flat chop)
    GOLD_SCALP_TP1_PIPS: float = 80.0                  # $8.00 (80 pips)
    GOLD_SCALP_TP2_PIPS: float = 160.0                 # $16.00 (160 pips)
    GOLD_SCALP_SL_PIPS: float = 55.0                   # $5.50 (55 pips)
    GOLD_SCALP_FAST_BE_PIPS: float = 75.0              # Move SL to BE only after reaching +75 pips ($7.50)
    GOLD_FAST_BE_ENABLED: bool = True
    GOLD_DEFAULT_SL_POINTS: float = 700.0              # $7.00
    GOLD_DEFAULT_TP_POINTS: float = 1800.0             # $18.00
    GOLD_MAX_SPREAD_POINTS: float = 50.0               # $0.50 max spread
    GOLD_BOLLINGER_PERIOD: int = 20
    GOLD_BOLLINGER_STD: float = 2.0
    GOLD_BOLLINGER_SCALP_STD: float = 2.2
    GOLD_EMA_FAST: int = 20
    GOLD_EMA_MID: int = 50
    GOLD_EMA_SLOW: int = 200
    
    # Multimodal Vision AI & Autonomous Learning Lab Settings
    VISION_ENABLED: bool = True
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", "gemini-2.0-flash")
    AUTONOMOUS_AUTOPSY_ENABLED: bool = True
    EXPERIENCE_MEMORY_FILE: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "episodic_experience_memory.json")
    
    # Analyst Desk Settings (Agent Brief 26) - Measurement instrument only (zero broker execution capability)
    ANALYST_DESK_PASSWORD: str = os.getenv("ANALYST_DESK_PASSWORD", "fx_desk_sec_2026_99a8b")

    # Mirrored Account Settings (Agent Brief 32) - Read-Only Investor Watcher
    MT5_MIRROR_LOGIN: int = int(os.getenv("MT5_MIRROR_LOGIN", "0"))
    MT5_MIRROR_SERVER: str = os.getenv("MT5_MIRROR_SERVER", "")
    MT5_MIRROR_INVESTOR_PASSWORD: str = os.getenv("MT5_MIRROR_INVESTOR_PASSWORD", "")
    MIRROR_ALLOW_LIVE: bool = os.getenv("MIRROR_ALLOW_LIVE", "true").lower() in ("true", "1", "yes")
    MIRROR_UI_PASSWORD: str = os.getenv("MIRROR_UI_PASSWORD", "fx_mirror_sec_2026_ab81c")
    MIRROR_PUBLISH_TELEGRAM: bool = os.getenv("MIRROR_PUBLISH_TELEGRAM", "false").lower() in ("true", "1", "yes")
    MIRROR_PUBLISH_GOLD: bool = os.getenv("MIRROR_PUBLISH_GOLD", "false").lower() in ("true", "1", "yes")
    
    class Config:
        case_sensitive = True

settings = Settings()


