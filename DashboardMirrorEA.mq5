//+------------------------------------------------------------------+
//|                                            DashboardMirrorEA.mq5 |
//|                                  Copyright 2026, Forex Engineer  |
//|                     Syncs Live MT5 Trades to the Local Dashboard |
//+------------------------------------------------------------------+
#property copyright "Forex Engineer"
#property link      "http://localhost:8000"
#property version   "1.00"
#property strict

//--- Input Parameters
input string InpServerUrl       = "http://localhost:8000/api/v1/mirror/sync"; // Local Dashboard API URL
input string InpMirrorPassword  = "fx_mirror_sec_2026_ab81c";                 // Mirror Password
input int    InpPollIntervalSec = 1;                                          // Sync Interval (Seconds)

//--- Global Tracking
datetime g_last_sync = 0;

//+------------------------------------------------------------------+
//| Helper: Escape JSON string                                        |
//+------------------------------------------------------------------+
string JsonEscape(string text)
{
   StringReplace(text, "\\", "\\\\");
   StringReplace(text, "\"", "\\\"");
   StringReplace(text, "\r", "");
   StringReplace(text, "\n", " ");
   return text;
}

//+------------------------------------------------------------------+
//| Core Function: Gather & Post All Open Positions to Dashboard     |
//+------------------------------------------------------------------+
void SyncPositionsToDashboard()
{
   long login = AccountInfoInteger(ACCOUNT_LOGIN);
   string server = AccountInfoString(ACCOUNT_SERVER);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   long trade_mode = AccountInfoInteger(ACCOUNT_TRADE_MODE);

   string json = "{\n";
   
   // 1. Account Info
   json += "  \"account_info\": {\n";
   json += "    \"login\": " + IntegerToString(login) + ",\n";
   json += "    \"server\": \"" + JsonEscape(server) + "\",\n";
   json += "    \"balance\": " + DoubleToString(balance, 2) + ",\n";
   json += "    \"equity\": " + DoubleToString(equity, 2) + ",\n";
   json += "    \"trade_mode\": " + IntegerToString(trade_mode) + "\n";
   json += "  },\n";

   // 2. Open Positions
   json += "  \"positions\": [\n";
   int total = PositionsTotal();
   int count = 0;

   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      long ptype = PositionGetInteger(POSITION_TYPE);
      double volume = PositionGetDouble(POSITION_VOLUME);
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      datetime time_open = (datetime)PositionGetInteger(POSITION_TIME);
      int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
      double point = SymbolInfoDouble(sym, SYMBOL_POINT);

      if(count > 0) json += ",\n";
      
      json += "    {\n";
      json += "      \"ticket\": " + IntegerToString(ticket) + ",\n";
      json += "      \"symbol\": \"" + JsonEscape(sym) + "\",\n";
      json += "      \"direction\": \"" + (ptype == POSITION_TYPE_BUY ? "BUY" : "SELL") + "\",\n";
      json += "      \"lots\": " + DoubleToString(volume, 2) + ",\n";
      json += "      \"entry\": " + DoubleToString(open_price, digits) + ",\n";
      if(sl > 0)
         json += "      \"sl\": " + DoubleToString(sl, digits) + ",\n";
      if(tp > 0)
         json += "      \"tp\": " + DoubleToString(tp, digits) + ",\n";
      json += "      \"digits\": " + IntegerToString(digits) + ",\n";
      json += "      \"point\": " + DoubleToString(point, 5) + ",\n";
      json += "      \"time_utc\": \"" + TimeToString(time_open, TIME_DATE|TIME_SECONDS) + "\",\n";
      json += "      \"account\": " + IntegerToString(login) + "\n";
      json += "    }";
      count++;
   }

   json += "\n  ]\n}";

   // 3. Send HTTP WebRequest to Dashboard
   char postData[];
   StringToCharArray(json, postData, 0, WHOLE_ARRAY, CP_UTF8);
   int dataSize = ArraySize(postData);
   if(dataSize > 0 && postData[dataSize - 1] == 0)
      ArrayResize(postData, dataSize - 1);

   string headers = "Content-Type: application/json\r\n";
   headers += "X-Mirror-Password: " + InpMirrorPassword + "\r\n";
   headers += "Authorization: Bearer " + InpMirrorPassword + "\r\n";

   char result[];
   string resultHeaders;
   ResetLastError();
   int res = WebRequest("POST", InpServerUrl, headers, 3000, postData, result, resultHeaders);

   if(res == -1)
   {
      int err = GetLastError();
      // Error 4014: URL not allowed in WebRequest settings
      if(err == 4014)
      {
         Print("⚠️ [DashboardMirror] URL not allowed! Please add " + InpServerUrl + " in Tools -> Options -> Experts -> Allow WebRequest.");
      }
      else
      {
         Print("⚠️ [DashboardMirror] WebRequest failed, Error: ", err);
      }
   }
   else if(res >= 200 && res < 300)
   {
      // Synced cleanly
   }
   else
   {
      Print("⚠️ [DashboardMirror] Server returned HTTP ", res);
   }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("🚀 [DashboardMirrorEA] Initialized on ", Symbol(), ". Streaming to: ", InpServerUrl);
   EventSetTimer(MathMax(1, InpPollIntervalSec));
   SyncPositionsToDashboard();
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("🛑 [DashboardMirrorEA] Stopped.");
}

//+------------------------------------------------------------------+
//| Timer function (Triggered every InpPollIntervalSec seconds)       |
//+------------------------------------------------------------------+
void OnTimer()
{
   SyncPositionsToDashboard();
}

//+------------------------------------------------------------------+
//| Trade Event (Triggered instantly when order is opened/modified)  |
//+------------------------------------------------------------------+
void OnTrade()
{
   SyncPositionsToDashboard();
}

//+------------------------------------------------------------------+
//| Tick Event (Live market price tick)                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Throttle to sync once per 2 seconds on ticks
   datetime now = TimeCurrent();
   if(now - g_last_sync >= InpPollIntervalSec)
   {
      g_last_sync = now;
      SyncPositionsToDashboard();
   }
}
//+------------------------------------------------------------------+
