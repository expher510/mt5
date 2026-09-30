"""
Interactive & CLI Test Tool: Simulate MetaTrader 5 (MT5) Trades on Local Dashboard.
Sends real-time positions directly into the local server and verifies the mirror dashboard.
Supports any custom MT5 account and server.
"""
import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
import httpx

BACKEND_URL = "http://localhost:8000"
MIRROR_PASSWORD = "fx_mirror_sec_2026_ab81c"

DATA_DIR = Path(__file__).resolve().parent / "data"

def get_headers():
    return {
        "Content-Type": "application/json",
        "X-Mirror-Password": MIRROR_PASSWORD,
        "Authorization": f"Bearer {MIRROR_PASSWORD}"
    }

def get_default_specs(symbol: str):
    sym = symbol.upper()
    if "XAU" in sym or "GOLD" in sym:
        return 2, 0.01
    if "JPY" in sym:
        return 3, 0.001
    return 5, 0.00001

def clear_all_previous_trades():
    """Resets all stored mirrored trades to start clean for a new MT5 account."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "mirrored_positions.json", "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(DATA_DIR / "mirrored_published_signals.json", "w", encoding="utf-8") as f:
        json.dump([], f)
    with open(DATA_DIR / "mirrored_published_results.json", "w", encoding="utf-8") as f:
        json.dump([], f)
    with open(DATA_DIR / "mirrored_trades.jsonl", "w", encoding="utf-8") as f:
        f.write("")
    with open(DATA_DIR / "mirrored_outcomes.jsonl", "w", encoding="utf-8") as f:
        f.write("")
        
    # Notify server to reset in-memory state
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(f"{BACKEND_URL}/api/v1/mirror/clear", headers=get_headers())
    except Exception:
        pass
    print("🧹 تم تصفير جميع الصفقات القديمة بنجاح! الداشبورد أصبح فارغاً وجاهزاً للحساب الجديد.")

def send_open_trade(ticket: int, symbol: str, direction: str, lot: float, entry: float, sl: float, tp: float, account: int, server: str):
    """Sends a new open position as if MT5 terminal opened it."""
    digits, point = get_default_specs(symbol)
    payload = {
        "account_info": {
            "login": account,
            "server": server,
            "balance": 10000.0,
            "equity": 10000.0,
            "trade_mode": 0
        },
        "positions": [
            {
                "ticket": ticket,
                "symbol": symbol.upper(),
                "direction": direction.upper(),
                "lots": lot,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "digits": digits,
                "point": point,
                "account": account
            }
        ]
    }
    
    print("\n" + "="*60)
    print(f"🚀 [MT5 -> السيرفر] إرسال صفقة جديدة للحساب #{account} ({server}):")
    print(f"   التذكرة (Ticket):   #{ticket}")
    print(f"   الرمز (Symbol):      {symbol.upper()}")
    print(f"   النوع (Side):        {direction.upper()}")
    print(f"   حجم اللوت (Lot):     {lot}")
    print(f"   نقطة البداية (Entry): {entry}")
    print(f"   نقطة الوقف (SL):     {sl}")
    print(f"   الهدف (TP):          {tp}")
    print("="*60)
    
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{BACKEND_URL}/api/v1/mirror/sync", json=payload, headers=get_headers())
            if resp.status_code == 200:
                print(f"✅ تم استلام الصفقة بنجاح في السيرفر!")
            else:
                print(f"❌ خطأ من السيرفر ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"❌ تعذر الاتصال بالسيرفر على {BACKEND_URL}. هل قمت بتشغيل السيرفر؟\n   الخطأ: {e}")

def modify_trade_sl(ticket: int, symbol: str, direction: str, lot: float, entry: float, new_sl: float, tp: float, account: int, server: str):
    """Modifies an existing position (e.g. moves SL to Breakeven)."""
    digits, point = get_default_specs(symbol)
    payload = {
        "account_info": {
            "login": account,
            "server": server
        },
        "positions": [
            {
                "ticket": ticket,
                "symbol": symbol.upper(),
                "direction": direction.upper(),
                "lots": lot,
                "entry": entry,
                "sl": new_sl,
                "tp": tp,
                "digits": digits,
                "point": point,
                "account": account
            }
        ]
    }
    print(f"\n🔄 [MT5 -> السيرفر] تعديل نقطة الوقف (Stop Loss) للصفقة #{ticket}:")
    print(f"   الوقف الجديد: {new_sl} (تأمين / Breakeven)")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{BACKEND_URL}/api/v1/mirror/sync", json=payload, headers=get_headers())
            if resp.status_code == 200:
                print(f"✅ تم تحديث الوقف بنجاح!")
            else:
                print(f"❌ خطأ: {resp.text}")
    except Exception as e:
        print(f"❌ تعذر الاتصال: {e}")

def close_trade_deal(ticket: int, symbol: str, direction: str, lot: float, entry: float, close_price: float, profit: float, account: int, server: str, reason: int = 5):
    """Sends a closed deal outcome as if closed on MT5 by TP, SL, or manual."""
    reason_str = "TARGET HIT (TP)" if reason == 5 else ("STOP HIT (SL)" if reason == 4 else "CLOSED MANUALLY")
    payload = {
        "account_info": {
            "login": account,
            "server": server
        },
        "deals": [
            {
                "ticket": ticket + 1000,
                "order": ticket,
                "position_id": ticket,
                "symbol": symbol.upper(),
                "type": 1 if direction.upper() == "BUY" else 0,
                "entry": 1,
                "volume": lot,
                "price": close_price,
                "profit": profit,
                "reason": reason,
                "comment": f"Close #{ticket} by {reason_str}"
            }
        ],
        "positions": []
    }
    print(f"\n🏁 [MT5 -> السيرفر] إغلاق الصفقة #{ticket} للحساب #{account}:")
    print(f"   سعر الإغلاق: {close_price} | الربح/الخسارة: ${profit:+.2f} | السبب: {reason_str}")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{BACKEND_URL}/api/v1/mirror/sync", json=payload, headers=get_headers())
            if resp.status_code == 200:
                print(f"✅ تم إغلاق الصفقة ونقلها إلى سجل الصفقات المغلقة (Closed Trades)!")
            else:
                print(f"❌ خطأ: {resp.text}")
    except Exception as e:
        print(f"❌ تعذر الاتصال: {e}")

def fetch_dashboard_state():
    """Fetches currently open trades from the dashboard endpoint."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{BACKEND_URL}/api/v1/mirror/trades/open", headers=get_headers())
            if resp.status_code == 200:
                trades = resp.json().get("data", [])
                print("\n" + "="*60)
                print(f"📊 [الداشبورد المحلي] الصفقات المفتوحة المعروضة حالياً ({len(trades)} صفقة):")
                print("="*60)
                if not trades:
                    print("   (لا توجد صفقات مفتوحة حالياً)")
                for t in trades:
                    print(f"   🔹 صفقة #{t.get('ticket')} (الحساب: {t.get('account', 'N/A')}):")
                    print(f"      الرمز:        {t.get('symbol')}")
                    print(f"      النوع:        {t.get('direction')}")
                    print(f"      اللوت:        {t.get('lot')}")
                    print(f"      سعر الدخول:   {t.get('entry')}")
                    print(f"      وقف الخسارة:  {t.get('stop_loss')}")
                    print(f"      جني الأرباح:  {t.get('take_profit')}")
                    print(f"      نسبة R:R:     {t.get('rr')}")
                    print(f"      الوقت:        {t.get('opened_utc')}")
                    print("-" * 40)
                return trades
            else:
                print(f"❌ تعذر القراءة: {resp.text}")
    except Exception as e:
        print(f"❌ تعذر الاتصال بالسيرفر: {e}")
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="محاكي تداول MT5 للاختبار المحلي مع دعم أي حساب جديد")
    parser.add_argument("--account", type=int, default=77889900, help="رقم حساب MT5 الجديد (مثال: 77889900)")
    parser.add_argument("--server", default="CustomBroker-Demo", help="اسم سيرفر البروكر (مثال: Exness-Demo, ICMarkets-Live)")
    parser.add_argument("--symbol", default="XAUUSD", help="رمز العملة (مثال: XAUUSD, EURUSD)")
    parser.add_argument("--side", default="BUY", choices=["BUY", "SELL", "buy", "sell"], help="نوع الصفقة: BUY أو SELL")
    parser.add_argument("--lot", type=float, default=0.10, help="حجم اللوت (مثال: 0.10)")
    parser.add_argument("--entry", type=float, default=None, help="سعر الدخول / نقطة البداية")
    parser.add_argument("--sl", type=float, default=None, help="نقطة الوقف (Stop Loss)")
    parser.add_argument("--tp", type=float, default=None, help="نقطة الهدف (Take Profit)")
    parser.add_argument("--ticket", type=int, default=None, help="رقم التذكرة (اختياري)")
    parser.add_argument("--list", action="store_true", help="عرض الصفقات المفتوحة الحالية في الداشبورد")
    parser.add_argument("--close", type=int, help="إغلاق تذكرة معينة")
    parser.add_argument("--profit", type=float, default=150.0, help="الربح عند الإغلاق بالدولار")
    parser.add_argument("--clear", action="store_true", help="تصفير ومسح جميع الصفقات القديمة للبدء بحساب جديد نظيف")
    
    args = parser.parse_args()
    
    if args.clear:
        clear_all_previous_trades()
        sys.exit(0)
        
    if args.list:
        fetch_dashboard_state()
        sys.exit(0)
        
    if args.close:
        close_trade_deal(ticket=args.close, symbol=args.symbol, direction=args.side, lot=args.lot, entry=args.entry or 2650.0, close_price=args.entry or 2665.0, profit=args.profit, account=args.account, server=args.server)
        fetch_dashboard_state()
        sys.exit(0)

    sym = args.symbol.upper()
    side = args.side.upper()
    lot = args.lot
    ticket = args.ticket or random.randint(10000000, 99999999)
    
    if "XAU" in sym or "GOLD" in sym:
        entry = args.entry or (2652.50 if side == "BUY" else 2652.50)
        sl = args.sl or (2644.00 if side == "BUY" else 2661.00)
        tp = args.tp or (2668.00 if side == "BUY" else 2635.00)
    elif "EURUSD" in sym:
        entry = args.entry or (1.08500 if side == "BUY" else 1.08500)
        sl = args.sl or (1.08200 if side == "BUY" else 1.08800)
        tp = args.tp or (1.09100 if side == "BUY" else 1.07900)
    else:
        entry = args.entry or 100.0
        sl = args.sl or (95.0 if side == "BUY" else 105.0)
        tp = args.tp or (110.0 if side == "BUY" else 90.0)

    # 1. إرسال الصفقة
    send_open_trade(ticket, sym, side, lot, entry, sl, tp, args.account, args.server)
    
    # 2. عرض الداشبورد
    fetch_dashboard_state()
    
    print("\n🌐 افتح المتصفح على:")
    print("   👉 http://localhost:8000/mirror")
    print(f"   لتشاهد صفقات الحساب الجديد #{args.account} مباشرة على الداشبورد!")
