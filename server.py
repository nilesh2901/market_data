import os
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/etf-prices')
def get_etf_prices():
    try:
        ticker_str = request.args.get('tickers', '')
        if not ticker_str:
            return jsonify({})

        raw_tickers = [t.strip().upper() for t in ticker_str.split(',') if t.strip()]
        search_tickers = []
        us_tickers = ['VOO', 'IVV', 'QQQ', 'SPY', 'VTI', 'DIA', 'SCHD']

        for t in raw_tickers:
            clean = t.replace('.', '-')
            if clean in us_tickers:
                search_name = clean
            elif '-' not in clean and '.TO' not in clean:
                search_name = f"{clean}.TO"
            else:
                search_name = clean if ('.TO' in clean or '-' in clean) else f"{clean}.TO"
            search_tickers.append(search_name)
        
        # Added group_by='column' to handle yfinance multi-index reliability
        data = yf.download(search_tickers, period="1d", interval="1m", progress=False, group_by='column')
        
        results = {}
        for i, original in enumerate(raw_tickers):
            search_name = search_tickers[i]
            try:
                if len(search_tickers) > 1:
                    val = data['Close'][search_name].dropna().iloc[-1]
                else:
                    val = data['Close'].dropna().iloc[-1]
                results[original] = round(float(val), 2)
            except:
                results[original] = 0.0
                
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------
# FIXED MULTIPLIER LOGIC
# ---------------------------------------------------------
def get_mult(score, ma50_dist, rsi):
    # 1. Extreme Overheat (Sell/Trim)
    if ma50_dist > 4.2 and rsi > 62: 
        return 0.0
    
    # 2. Tactical Buy Levels
    if score >= 75: return 4.0   # Terminal Capitulation
    if score >= 65: return 3.0   # Aggressive Buy
    if score >= 55: return 2.0   # Market Fear (This was 65 before, now 55)
    
    # 3. Momentum Decay (Halve)
    if score <= 35: return 0.5
    
    # 4. Standard DCA
    return 1.0

def calculate_rsi(series, period=14):
    if len(series) < period + 1:
        return pd.Series([50] * len(series))
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@app.route('/api/market-data')
def get_data():
    try:
        target_date_str = request.args.get('date')
        budget = float(request.args.get('budget', 1000))
        base_daily = budget / 5

        tickers = ["^GSPC", "^VIX", "^TNX", "^IRX"]
        raw_data = yf.download(tickers, period="5y", interval="1d", auto_adjust=True, group_by='column')
        
        # Explicitly select Close to avoid Multi-Index errors
        close_prices = raw_data['Close']['^GSPC'].dropna().tz_localize(None)
        vix_prices = raw_data['Close']['^VIX'].dropna().tz_localize(None)
        ten_year = raw_data['Close']['^TNX'].dropna().tz_localize(None)
        three_month = raw_data['Close']['^IRX'].dropna().tz_localize(None)

        if target_date_str and target_date_str.strip():
            target_dt = pd.to_datetime(target_date_str)
            idx_pos = close_prices.index.get_indexer([target_dt], method='pad')[0]
        else:
            idx_pos = len(close_prices) - 1

        ma50_all = close_prices.rolling(window=50).mean()
        ma200_all = close_prices.rolling(window=200).mean()

        def compute_daily_stats(pos):
            h_slice = close_prices.iloc[:pos+1]
            c_price = float(close_prices.iloc[pos])
            v_val = float(vix_prices.iloc[min(pos, len(vix_prices)-1)])
            d_rsi = float(calculate_rsi(h_slice).iloc[-1])
            d_ma50 = float(ma50_all.iloc[pos])
            d_ma50_dist = ((c_price - d_ma50) / d_ma50) * 100
            d_ma200 = float(ma200_all.iloc[pos])
            d_ma200_dist = ((c_price - d_ma200) / d_ma200) * 100
            d_y10 = float(ten_year.iloc[min(pos, len(ten_year)-1)])
            d_y3m = float(three_month.iloc[min(pos, len(three_month)-1)])
            
            # Simplified Yield Spread calculation
            d_yield = (d_y10 - d_y3m) 
            
            # Robust Breadth: Check if we have enough data
            tail_data = h_slice.tail(51)
            if len(tail_data) > 1:
                d_breadth = ((tail_data.diff() > 0).sum() / 50) * 100
            else:
                d_breadth = 0
            
            # Scoring Logic
            v_pts = max(0, min(30, (v_val - 15) * 1.5))
            r_pts = max(0, min(25, (70 - d_rsi) * 0.625))
            m200_pts = max(0, min(15, (5 - d_ma200_dist) * 1.5))
            y_pts = max(0, min(15, (d_yield + 1) * 7.5))
            b_pts = max(0, min(15, (d_breadth - 20) * 0.25))
            
            score = round(float(v_pts + r_pts + m200_pts + y_pts + b_pts), 1)
            return score, d_ma50_dist, d_rsi, v_val, m200_pts, y_pts, b_pts, v_pts, r_pts, d_ma200_dist

        res = compute_daily_stats(idx_pos)
        curr_dt = close_prices.index[idx_pos]
        start_of_week = curr_dt - timedelta(days=curr_dt.weekday())
        
        weekly_allocs = []
        days_map = ["MON", "TUE", "WED", "THU", "FRI"]
        for i in range(5):
            day_dt = start_of_week + timedelta(days=i)
            if day_dt > curr_dt:
                # Use current day's multiplier for future days in the same week
                current_mult = get_mult(res[0], res[1], res[2])
                weekly_allocs.append({"day": days_map[i], "multiplier": float(current_mult)})
            else:
                try:
                    d_idx = close_prices.index.get_indexer([day_dt], method='pad')[0]
                    d_res = compute_daily_stats(d_idx)
                    weekly_allocs.append({"day": days_map[i], "multiplier": float(get_mult(d_res[0], d_res[1], d_res[2]))})
                except:
                    weekly_allocs.append({"day": days_map[i], "multiplier": 1.0})
        
        # Calculate final total
        weekly_total = sum(float(item['multiplier']) * base_daily for item in weekly_allocs)

        chart_slice_start = max(0, idx_pos-180)
        chart_data = []
        for i in range(chart_slice_start, idx_pos + 1):
            chart_data.append({
                "time": close_prices.index[i].strftime('%Y-%m-%d'),
                "value": float(close_prices.iloc[i]),
                "ma50": float(ma50_all.iloc[i]) if not pd.isna(ma50_all.iloc[i]) else None,
                "ma200": float(ma200_all.iloc[i]) if not pd.isna(ma200_all.iloc[i]) else None
            })

        return jsonify({
            "rsi": round(float(res[2]), 2),
            "vix": round(float(res[3]), 2),
            "ma200": round(float(res[9]), 2),
            "ma50Dist": round(float(res[1]), 2),
            "momentum": round(float(close_prices.iloc[:idx_pos+1].diff(10).iloc[-1]), 2),
            "yieldcurve": round(float(res[5]/7.5 - 1), 2),
            "breadth": round(float(res[6]*4 + 20), 1),
            "riskScore": float(res[0]),
            "drawdown": round(float(((close_prices.iloc[idx_pos] - close_prices.iloc[:idx_pos+1].max()) / close_prices.iloc[:idx_pos+1].max()) * 100), 2),
            "maxDrawdown": round(float(((close_prices - close_prices.cummax())/close_prices.cummax()).min() * 100), 2),
            "mddDate": close_prices.index[((close_prices - close_prices.cummax())/close_prices.cummax()).argmin()].strftime('%Y-%m-%d'),
            "chartData": chart_data,
            "actualDate": curr_dt.strftime('%Y-%m-%d'),
            "currentPrice": float(close_prices.iloc[idx_pos]),
            "vixPoints": round(float(res[7]), 1),
            "rsiPoints": round(float(res[8]), 1),
            "macroPoints": round(float(res[4] + res[5]), 1),
            "breadthPoints": round(float(res[6]), 1),
            "weeklyAllocations": weekly_allocs,
            "weeklyTotal": float(weekly_total)
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)