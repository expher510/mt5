"""
MT5 Bridge Sync & Manual Trading Service.
Connects to MT5BridgeEA via ZeroMQ (port 5556) and synchronizes
live manual trades with the Local Mirror Dashboard (http://localhost:8000/mirror).
Also provides manual trade execution (Buy, Sell, Close, Modify) via MT5BridgeEA.
"""
import sys
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import httpx
import zmq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MT5BridgeSync")

BRIDGE_ENDPOINT = "tcp://127.0.0.1:5556"
DASHBOARD_SYNC_URL = "http://localhost:8000/api/v1/mirror/sync"
MIRROR_PASSWORD = "fx_mirror_sec_2026_ab81c"


class MT5BridgeSyncClient:
    def __init__(self, endpoint: str = BRIDGE_ENDPOINT, timeout_ms: int = 3000):
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket: Optional[zmq.Socket] = None
        self._connect()

    def _connect(self):
        if self.socket:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.endpoint)

    def send_request(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        payload = {
            "request_id": req_id,
            "action": action,
            "params": params or {},
            "timestamp": time.time(),
        }
        try:
            self.socket.send_string(json.dumps(payload))
            raw = self.socket.recv_string()
            return json.loads(raw)
        except zmq.Again:
            logger.warning(f"Bridge request {action} timed out after {self.timeout_ms}ms")
            self._connect()
            return {"success": False, "error_message": f"Bridge request timed out ({self.timeout_ms}ms)"}
        except Exception as e:
            logger.error(f"Error communicating with bridge: {e}")
            self._connect()
            return {"success": False, "error_message": str(e)}

    def ping(self) -> Dict[str, Any]:
        return self.send_request("PING")

    def get_account_info(self) -> Dict[str, Any]:
        return self.send_request("GET_ACCOUNT_INFO")

    def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        params = {"symbol": symbol} if symbol else {}
        return self.send_request("GET_POSITIONS", params)

    def get_deals(self, days: int = 7) -> Dict[str, Any]:
        return self.send_request("GET_DEALS", {"days": days})

    def open_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "manual_trade",
    ) -> Dict[str, Any]:
        params = {
            "symbol": symbol.upper(),
            "order_type": order_type.upper(),
            "volume": float(volume),
            "sl": float(sl) if sl is not None else 0.0,
            "tp": float(tp) if tp is not None else 0.0,
            "comment": comment,
        }
        return self.send_request("OPEN_ORDER", params)

    def close_position(self, ticket: int, volume: Optional[float] = None, comment: str = "manual_close") -> Dict[str, Any]:
        params = {
            "ticket": int(ticket),
            "volume": float(volume) if volume is not None else 0.0,
            "comment": comment,
        }
        return self.send_request("CLOSE_POSITION", params)

    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> Dict[str, Any]:
        params = {
            "ticket": int(ticket),
            "sl": float(sl) if sl is not None else 0.0,
            "tp": float(tp) if tp is not None else 0.0,
        }
        return self.send_request("MODIFY_POSITION", params)


def extract_closed_trades_from_deals(deals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    positions = {}
    for d in deals:
        pos_id = d.get('position_id')
        if not pos_id:
            continue
        if pos_id not in positions:
            positions[pos_id] = {'in': None, 'out': []}
        if d.get('entry') == 'IN':
            positions[pos_id]['in'] = d
        elif d.get('entry') == 'OUT':
            positions[pos_id]['out'].append(d)

    closed_list = []
    for pos_id, data in positions.items():
        in_deal = data['in']
        out_deals = data['out']
        if not out_deals:
            continue
        sym = (in_deal.get('symbol') if in_deal else out_deals[0].get('symbol') or '').upper()
        direction = in_deal.get('type') if in_deal else (out_deals[0].get('type') or 'BUY')
        entry_price = float(in_deal.get('price', 0.0)) if in_deal else 0.0

        for out_deal in out_deals:
            exit_price = float(out_deal.get('price', 0.0))
            profit = float(out_deal.get('profit', 0.0))
            vol = float(out_deal.get('volume', 0.01))
            t_epoch = int(out_deal.get('time', 0))
            closed_time = datetime.fromtimestamp(t_epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

            closed_list.append({
                'ticket': pos_id,
                'deal_ticket': out_deal.get('ticket'),
                'symbol': sym,
                'direction': direction,
                'lot': vol,
                'entry': entry_price,
                'exit_price': exit_price,
                'profit': round(profit, 2),
                'closed_utc': closed_time,
                'closed_at_epoch': t_epoch,
                'reason_name': 'CLOSED MANUALLY' if profit >= 0 else 'STOP / EXIT'
            })

    closed_list.sort(key=lambda x: x.get('closed_at_epoch', 0), reverse=True)
    return closed_list


def run_sync_loop(client: Optional[MT5BridgeSyncClient] = None, endpoint: Optional[str] = None, poll_interval: float = 1.0):
    """
    Main background sync loop.
    Fetches positions, account info & closed deals from MT5BridgeEA and pushes them to the dashboard.
    """
    ep = endpoint or BRIDGE_ENDPOINT
    if client is None:
        client = MT5BridgeSyncClient(endpoint=ep)
    logger.info("=" * 60)
    logger.info("🚀 MT5BridgeEA Live Sync Started")
    logger.info(f"Bridge ZeroMQ Endpoint: {client.endpoint}")
    logger.info(f"Dashboard Endpoint:     {DASHBOARD_SYNC_URL}")
    logger.info("=" * 60)

    headers = {
        "Content-Type": "application/json",
        "X-Mirror-Password": MIRROR_PASSWORD,
        "Authorization": f"Bearer {MIRROR_PASSWORD}",
    }

    last_connected_state = None
    last_known_tickets = set()
    cached_closed_trades = []
    cycle_counter = 0

    with httpx.Client(timeout=4.0) as http_client:
        while True:
            try:
                acc_resp = client.get_account_info()
                pos_resp = client.get_positions()
                cycle_counter += 1

                is_connected = acc_resp.get("success", False) and pos_resp.get("success", False)

                if is_connected != last_connected_state:
                    if is_connected:
                        acc_data = acc_resp.get("data", {})
                        logger.info(
                            f"✅ Connected to MT5! Account #{acc_data.get('login')} "
                            f"({acc_data.get('company', 'Demo')}) - Balance: ${acc_data.get('balance', 0):,.2f}"
                        )
                    else:
                        err = acc_resp.get("error_message") or pos_resp.get("error_message")
                        logger.warning(f"⏳ Waiting for MT5BridgeEA on MT5 chart... ({err})")
                    last_connected_state = is_connected

                if is_connected:
                    acc_data = acc_resp.get("data", {})
                    positions_raw = pos_resp.get("data", {}).get("positions", [])

                    # Refresh closed deals from MT5 every 4 cycles (or on first sync)
                    if cycle_counter % 4 == 1 or not cached_closed_trades:
                        deals_resp = client.get_deals(days=90)
                        if deals_resp.get("success"):
                            deals_raw = deals_resp.get("data", {}).get("deals", [])
                            cached_closed_trades = extract_closed_trades_from_deals(deals_raw)
                            logger.info(f"📚 Synced {len(cached_closed_trades)} historical closed deals from MT5")

                    pos_list = []
                    current_tickets = set()
                    for p in positions_raw:
                        ticket = int(p.get("ticket", 0))
                        if ticket <= 0:
                            continue
                        current_tickets.add(ticket)
                        pos_list.append({
                            "ticket": ticket,
                            "symbol": str(p.get("symbol", "")).upper(),
                            "direction": str(p.get("type", "BUY")).upper(),
                            "lots": float(p.get("volume", 0.01)),
                            "entry": float(p.get("price_open", 0.0)),
                            "sl": float(p.get("sl", 0.0)) if float(p.get("sl", 0.0)) > 0 else None,
                            "tp": float(p.get("tp", 0.0)) if float(p.get("tp", 0.0)) > 0 else None,
                            "profit": float(p.get("profit", 0.0)),
                            "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                        })

                    # Check for new manual trades
                    new_trades = current_tickets - last_known_tickets
                    if new_trades:
                        for p in pos_list:
                            if p["ticket"] in new_trades:
                                logger.info(
                                    f"🔔 [NEW MANUAL TRADE DETECTED] #{p['ticket']} {p['direction']} {p['lots']} lots "
                                    f"{p['symbol']} @ {p['entry']} (SL: {p['sl']}, TP: {p['tp']}) -> Synced to Dashboard!"
                                )
                    closed_trades = last_known_tickets - current_tickets
                    if closed_trades:
                        logger.info(f"ℹ️ [TRADE CLOSED] Tickets closed: {closed_trades} -> Synced to Dashboard!")

                    last_known_tickets = current_tickets

                    payload = {
                        "account_info": {
                            "login": int(acc_data.get("login", 0)),
                            "server": str(acc_data.get("company", "MetaQuotes-Demo")),
                            "balance": float(acc_data.get("balance", 0.0)),
                            "equity": float(acc_data.get("equity", 0.0)),
                            "trade_mode": 0
                        },
                        "positions": pos_list,
                        "closed_trades": cached_closed_trades
                    }

                    try:
                        sync_res = http_client.post(DASHBOARD_SYNC_URL, json=payload, headers=headers)
                        if sync_res.status_code != 200:
                            logger.error(f"Dashboard sync returned HTTP {sync_res.status_code}: {sync_res.text}")
                    except Exception as e:
                        logger.error(f"Failed to post sync to dashboard: {e}")

            except Exception as e:
                logger.error(f"Error in sync loop iteration: {e}")

            time.sleep(poll_interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MT5BridgeEA Sync & Manual Trade Execution Tool")
    parser.add_argument("--endpoint", type=str, default=None, help="ZeroMQ Bridge endpoint (e.g. tcp://127.0.0.1:5556)")
    parser.add_argument("--port", type=int, default=None, help="ZeroMQ Bridge port (e.g. 5556 for Gold account, 5558 for Forex account)")
    parser.add_argument("--sync", action="store_true", help="Run background live sync loop with dashboard")
    parser.add_argument("--status", action="store_true", help="Get MT5 Account info and current open positions")
    parser.add_argument("--buy", nargs=2, metavar=("SYMBOL", "LOTS"), help="Execute manual BUY order (e.g. --buy XAUUSD 0.01)")
    parser.add_argument("--sell", nargs=2, metavar=("SYMBOL", "LOTS"), help="Execute manual SELL order (e.g. --sell EURUSD 0.02)")
    parser.add_argument("--sl", type=float, default=0.0, help="Stop Loss price for manual order")
    parser.add_argument("--tp", type=float, default=0.0, help="Take Profit price for manual order")
    parser.add_argument("--close", type=int, metavar="TICKET", help="Close open position by ticket")
    parser.add_argument("--modify", type=int, metavar="TICKET", help="Modify SL/TP for open position ticket")

    args = parser.parse_args()

    # Determine bridge endpoint
    endpoint = args.endpoint
    if not endpoint and args.port:
        endpoint = f"tcp://127.0.0.1:{args.port}"
    if not endpoint:
        endpoint = BRIDGE_ENDPOINT

    client = MT5BridgeSyncClient(endpoint=endpoint)

    if args.status:
        print(f"\n--- MT5 ACCOUNT INFO ({endpoint}) ---")
        acc = client.get_account_info()
        print(json.dumps(acc, indent=2))
        print(f"\n--- MT5 OPEN POSITIONS ({endpoint}) ---")
        pos = client.get_positions()
        print(json.dumps(pos, indent=2))

    elif args.buy:
        sym, vol = args.buy[0], float(args.buy[1])
        print(f"Executing manual BUY on {endpoint}: {vol} lots {sym} (SL: {args.sl}, TP: {args.tp})...")
        res = client.open_order(symbol=sym, order_type="BUY", volume=vol, sl=args.sl, tp=args.tp, comment="manual_buy")
        print("Result:", json.dumps(res, indent=2))

    elif args.sell:
        sym, vol = args.sell[0], float(args.sell[1])
        print(f"Executing manual SELL on {endpoint}: {vol} lots {sym} (SL: {args.sl}, TP: {args.tp})...")
        res = client.open_order(symbol=sym, order_type="SELL", volume=vol, sl=args.sl, tp=args.tp, comment="manual_sell")
        print("Result:", json.dumps(res, indent=2))

    elif args.close:
        print(f"Closing position #{args.close} on {endpoint}...")
        res = client.close_position(ticket=args.close)
        print("Result:", json.dumps(res, indent=2))

    elif args.modify:
        print(f"Modifying position #{args.modify} on {endpoint} (SL: {args.sl}, TP: {args.tp})...")
        res = client.modify_position(ticket=args.modify, sl=args.sl, tp=args.tp)
        print("Result:", json.dumps(res, indent=2))

    elif args.sync:
        run_sync_loop(client=client, endpoint=endpoint)

    else:
        # Default action: run sync
        run_sync_loop(client=client, endpoint=endpoint)

