import os

from flask import Flask, jsonify, request, render_template # Added render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# NEW: Route to serve the HTML Dashboard
# ---------------------------------------------------------
@app.route('/')
def index():
    # Flask looks for this file in a folder named 'templates'
    return render_template('dashboard.html')

# ---------------------------------------------------------
# NEW: Route to fetch Real-Time ETF Prices
# ---------------------------------------------------------
@app.route('/api/etf-prices')
def get_etf_prices():
    try:
        ticker_str = request.args.get('tickers', '')
        if not ticker_str:
            return jsonify({})

        raw_tickers = [t.strip().upper() for t in ticker_str.split(',') if t.strip()]
        search_tickers = []
        
        # List of common US ETFs to bypass the .TO suffix
        us_tickers = ['VOO', 'IVV', 'QQQ', 'SPY', 'VTI', 'DIA', 'SCHD']

        for t in raw_tickers:
            clean = t.replace('.', '-') # Yahoo uses hyphens (e.g., CGL-C)
            
            if clean in us_tickers:
                search_name = clean # Use US ticker as-is
            elif '-' not in clean and '.TO' not in clean:
                search_name = f"{clean}.TO" # Default to TSX for CAD ETFs
            else:
                # Ensure it ends with .TO if it has a hyphen/dot but no exchange suffix
                search_name = clean if ('.TO' in clean or '-' in clean) else f"{clean}.TO"
            
            search_tickers.append(search_name)
        
        data = yf.download(search_tickers, period="1d", interval="1m", progress=False)
        
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
                results[original] = 0.0 # Returns 0.0 to trigger red UI alert
                
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_mult(score, ma50_dist, rsi):
    if ma50_dist > 5.5 and rsi > 68: return 0.0
    if score >= 80: return 4.0
    if score >= 65: return 2.0
    if score <= 35: return 0.5
    return 1.0

def calculate_rsi(series, period=14):
    if len(series) < period:
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
        raw_data = yf.download(tickers, period="5y", interval="1d", auto_adjust=True)
        
        close_prices = raw_data['Close']['^GSPC'].dropna().tz_localize(None)
        vix_prices = raw_data['Close']['^VIX'].dropna().tz_localize(None)
        ten_year = raw_data['Close']['^TNX'].dropna().tz_localize(None)
        three_month = raw_data['Close']['^IRX'].dropna().tz_localize(None)

        if target_date_str and target_date_str.strip():
            target_dt = pd.to_datetime(target_date_str)
            idx_pos = close_prices.index.get_indexer([target_dt], method='pad')[0]
        else:
            idx_pos = len(close_prices) - 1

        def compute_daily_stats(pos):
            h_slice = close_prices.iloc[:pos+1]
            c_price = close_prices.iloc[pos]
            v_val = vix_prices.iloc[min(pos, len(vix_prices)-1)]
            d_rsi = calculate_rsi(h_slice).iloc[-1]
            d_ma50 = h_slice.rolling(window=50).mean().iloc[-1]
            d_ma50_dist = ((c_price - d_ma50) / d_ma50) * 100
            d_ma200 = h_slice.rolling(window=200).mean().iloc[-1]
            d_ma200_dist = ((c_price - d_ma200) / d_ma200) * 100
            
            d_y10 = ten_year.iloc[min(pos, len(ten_year)-1)]
            d_y3m = three_month.iloc[min(pos, len(three_month)-1)]
            d_yield = (d_y10 - d_y3m) / 10
            d_breadth = ((h_slice.tail(50).diff() > 0).sum() / 50) * 100
            
            v_pts = max(0, min(30, (v_val - 15) * 1.5))
            r_pts = max(0, min(25, (70 - d_rsi) * 0.625))
            m200_pts = max(0, min(15, (5 - d_ma200_dist) * 1.5))
            y_pts = max(0, min(15, (d_yield + 1) * 7.5))
            b_pts = max(0, min(15, (d_breadth - 20) * 0.25))
            
            score = round(v_pts + r_pts + m200_pts + y_pts + b_pts, 1)
            return score, d_ma50_dist, d_rsi, v_val, m200_pts, y_pts, b_pts, v_pts, r_pts, d_ma200_dist

        res = compute_daily_stats(idx_pos)
        risk_score, ma50_dist, rsi, vix_val, m200_pts, yield_pts, breadth_pts, vix_pts, rsi_pts, raw_ma200_dist = res

        curr_dt = close_prices.index[idx_pos]
        start_of_week = curr_dt - timedelta(days=curr_dt.weekday())
        
        weekly_allocs = []
        days_map = ["MON", "TUE", "WED", "THU", "FRI"]
        for i in range(5):
            day_dt = start_of_week + timedelta(days=i)
            if day_dt > curr_dt:
                weekly_allocs.append({"day": days_map[i], "multiplier": 1.0})
            else:
                try:
                    d_idx = close_prices.index.get_indexer([day_dt], method='pad')[0]
                    d_res = compute_daily_stats(d_idx)
                    weekly_allocs.append({"day": days_map[i], "multiplier": get_mult(d_res[0], d_res[1], d_res[2])})
                except:
                    weekly_allocs.append({"day": days_map[i], "multiplier": 1.0})
        
        weekly_total = sum(item['multiplier'] * base_daily for item in weekly_allocs)

        return jsonify({
            "rsi": round(float(rsi), 2),
            "vix": round(float(vix_val), 2),
            "ma200": round(float(raw_ma200_dist), 2),
            "ma50Dist": round(float(ma50_dist), 2),
            "momentum": round(float(close_prices.iloc[:idx_pos+1].diff(10).iloc[-1]), 2),
            "yieldcurve": round(float(yield_pts/7.5 - 1), 2),
            "breadth": round(float(breadth_pts*4 + 20), 1),
            "riskScore": risk_score,
            "drawdown": round(float(((close_prices.iloc[idx_pos] - close_prices.iloc[:idx_pos+1].max()) / close_prices.iloc[:idx_pos+1].max()) * 100), 2),
            "maxDrawdown": round(float(((close_prices - close_prices.cummax())/close_prices.cummax()).min() * 100), 2),
            "mddDate": close_prices.index[((close_prices - close_prices.cummax())/close_prices.cummax()).argmin()].strftime('%Y-%m-%d'),
            "chartData": [{"time": d.strftime('%Y-%m-%d'), "value": float(v)} for d, v in close_prices.iloc[max(0, idx_pos-180):idx_pos+1].items()],
            "actualDate": curr_dt.strftime('%Y-%m-%d'),
            "currentPrice": float(close_prices.iloc[idx_pos]),
            "vixPoints": round(float(vix_pts), 1),
            "rsiPoints": round(float(rsi_pts), 1),
            "macroPoints": round(float(m200_pts + yield_pts), 1),
            "breadthPoints": round(float(breadth_pts), 1),
            "weeklyAllocations": weekly_allocs,
            "weeklyTotal": float(weekly_total)
        })
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    #app.run(host='127.0.0.1', port=5000, debug=True)
    # Use the port assigned by Render, or default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

