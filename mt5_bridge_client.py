"""
MT5 Live Sync Bridge for Real MetaTrader 5 Terminal.
Run this script on the computer where MT5 is installed and logged in.
It reads open positions and closed deals directly from MT5 and syncs them to your Local Dashboard.
"""
import sys
import time
import httpx

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

SERVER_URL = "http://localhost:8000/api/v1/mirror/sync"
MIRROR_PASSWORD = "fx_mirror_sec_2026_ab81c"

def sync_live_mt5():
    if mt5 is None:
        print("❌ مكتبة MetaTrader5 غير مثبتة. إذا كنت تشغل هذا على ويندوز مع MT5 نفذ:")
        print("   pip install MetaTrader5")
        return

    if not mt5.initialize():
        print(f"❌ فشل الاتصال بتطبيق MT5: {mt5.last_error()}")
        return

    acc = mt5.account_info()
    if acc is None:
        print("❌ لم يتم تسجيل الدخول في MT5.")
        return

    print(f"✅ متصل بحساب MT5 #{acc.login} ({acc.server}) - الرصيد: ${acc.balance:,.2f}")
    print("🚀 بدء مزامنة الصفقات الحية مع الداشبورد...")

    headers = {
        "Content-Type": "application/json",
        "X-Mirror-Password": MIRROR_PASSWORD,
        "Authorization": f"Bearer {MIRROR_PASSWORD}"
    }

    last_tickets = set()

    while True:
        try:
            positions = mt5.positions_get()
            pos_list = []
            if positions:
                for p in positions:
                    pos_list.append({
                        "ticket": int(p.ticket),
                        "symbol": str(p.symbol),
                        "direction": "BUY" if p.type == 0 else "SELL",
                        "lots": float(p.volume),
                        "entry": float(p.price_open),
                        "sl": float(p.sl) if p.sl > 0 else None,
                        "tp": float(p.tp) if p.tp > 0 else None,
                        "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(p.time))
                    })

            payload = {
                "account_info": {
                    "login": int(acc.login),
                    "server": str(acc.server),
                    "balance": float(acc.balance),
                    "equity": float(acc.equity),
                    "trade_mode": int(acc.trade_mode)
                },
                "positions": pos_list
            }

            resp = httpx.post(SERVER_URL, json=payload, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                current_tickets = {p["ticket"] for p in pos_list}
                new_tickets = current_tickets - last_tickets
                if new_tickets:
                    print(f"🔔 رصد صفقة جديدة من MT5 تظهر على الداشبورد: {new_tickets}")
                last_tickets = current_tickets
            else:
                print(f"⚠️ خطأ مزامنة: {resp.status_code}")

        except Exception as e:
            print(f"خطأ في دورة المزامنة: {e}")

        time.sleep(1.5)

if __name__ == "__main__":
    sync_live_mt5()
