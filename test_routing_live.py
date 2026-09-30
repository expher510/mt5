
from app.config import settings
from app.services.telegram_bot import telegram_bot, classify_market_category

print("=== VPS LIVE SETTINGS ===")
print("FOREX Webhook:", settings.N8N_WEBHOOK_URL_FOREX)
print("INDEX Webhook:", settings.N8N_WEBHOOK_URL_INDEX)
print("GOLD Webhook: ", settings.N8N_WEBHOOK_URL_GOLD)
print("MT5_MIRROR_LOGIN_GOLD (Engineer):", settings.MT5_MIRROR_LOGIN_GOLD)

tests = [
    # 1. Engineer Account (8058543) -> ALL trades go to GOLD
    {"symbol": "XAUUSD", "account": 8058543, "type": "signal"},
    {"symbol": "GOLD", "account": 8058543, "type": "update"},
    
    # 2. Second Account (160767360) -> FOREX
    {"symbol": "EURUSD", "account": 160767360, "type": "signal"},
    {"symbol": "GBPUSD", "account": 160767360, "type": "update"},
    
    # 3. Second Account (160767360) -> INDICES / COMMODITIES / CRYPTO
    {"symbol": "US30", "account": 160767360, "type": "signal"},
    {"symbol": "NAS100", "account": 160767360, "type": "signal"},
    {"symbol": "USOIL", "account": 160767360, "type": "signal"},
    {"symbol": "BTCUSD", "account": 160767360, "type": "signal"},
    {"symbol": "XAGUSD", "account": 160767360, "type": "signal"},
]

print("\n=== LIVE ROUTING RESULTS ===")
for t in tests:
    cat = classify_market_category(t["symbol"])
    url = telegram_bot.resolve_target_webhook(t)
    endpoint = url.split("/")[-1]
    print(f"Sym: {t['symbol']:<8} | Acc: {t['account']} | Cat: {cat:<6} -> {endpoint}")
