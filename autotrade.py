"""
AI 비트코인 트레이딩 봇 (GPT 판단형) - 교육용 코드
--------------------------------------------------------
기능:
- 멀티 타임프레임 분석 (15분, 1시간, 4시간 차트)
- 뉴스 감성 분석
- AI 기반 포지션 사이징 및 레버리지 최적화
- 동적 스탑로스/테이크프로핏 설정
- 거래 내역 데이터베이스 기록 및 성과 분석
- 과거 거래 데이터 기반 학습
--------------------------------------------------------
실행
  python autotrade.py                    # 실거래 (GPT 판단)
  python autotrade.py --dry              # GPT 판단만 기록, 주문 X
  python autotrade.py --backtest 180     # 규칙 기반 엔진 백테스트 (bitcoin_final.py 사용)
  python autotrade.py --compare 180      # 5가지 설정 비교 백테스트 (bitcoin_final.py 사용)

※ --backtest / --compare 는 같은 폴더의 bitcoin_final.py 를 불러와 실행합니다.
   (GPT 판단은 과거 시점으로 재현할 수 없어서, 백테스트는 규칙 기반 엔진만 검증합니다.)
"""
import ccxt
import os
import math
import time
import argparse
import pandas as pd
import requests
import json
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime

# ===== 설정 및 초기화 =====
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'portfolioMargin': True,
        'fetchCurrencies': False,
    },
})
symbol = "BTC/USDT:USDT"

# OpenAI 클라이언트는 실제로 필요할 때만 생성 (백테스트 시 OPENAI_API_KEY 없이도 실행되도록)
_client = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


serp_api_key = os.getenv("SERP_API_KEY")
DB_FILE = "bitcoin_trading.db"
BACKTEST_MODULE = os.getenv("BACKTEST_MODULE", "bitcoin_final")


# ===== 데이터베이스 관련 함수 =====
def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        action TEXT NOT NULL,
        entry_price REAL NOT NULL,
        amount REAL NOT NULL,
        leverage INTEGER NOT NULL,
        sl_price REAL NOT NULL,
        tp_price REAL NOT NULL,
        sl_percentage REAL NOT NULL,
        tp_percentage REAL NOT NULL,
        position_size_percentage REAL NOT NULL,
        investment_amount REAL NOT NULL,
        status TEXT DEFAULT 'OPEN',
        exit_price REAL,
        exit_timestamp TEXT,
        profit_loss REAL,
        profit_loss_percentage REAL
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ai_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        current_price REAL NOT NULL,
        direction TEXT NOT NULL,
        recommended_position_size REAL NOT NULL,
        recommended_leverage INTEGER NOT NULL,
        stop_loss_percentage REAL NOT NULL,
        take_profit_percentage REAL NOT NULL,
        reasoning TEXT NOT NULL,
        trade_id INTEGER,
        FOREIGN KEY (trade_id) REFERENCES trades (id)
    )
    ''')
    conn.commit()
    conn.close()
    print("데이터베이스 설정 완료")


def save_ai_analysis(analysis_data, trade_id=None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO ai_analysis (
        timestamp, current_price, direction, recommended_position_size,
        recommended_leverage, stop_loss_percentage, take_profit_percentage,
        reasoning, trade_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        analysis_data.get('current_price', 0),
        analysis_data.get('direction', 'NO_POSITION'),
        analysis_data.get('recommended_position_size', 0),
        analysis_data.get('recommended_leverage', 0),
        analysis_data.get('stop_loss_percentage', 0),
        analysis_data.get('take_profit_percentage', 0),
        analysis_data.get('reasoning', ''),
        trade_id
    ))
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id


def save_trade(trade_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trades (
        timestamp, action, entry_price, amount, leverage, sl_price, tp_price,
        sl_percentage, tp_percentage, position_size_percentage, investment_amount
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        trade_data.get('action', ''),
        trade_data.get('entry_price', 0),
        trade_data.get('amount', 0),
        trade_data.get('leverage', 0),
        trade_data.get('sl_price', 0),
        trade_data.get('tp_price', 0),
        trade_data.get('sl_percentage', 0),
        trade_data.get('tp_percentage', 0),
        trade_data.get('position_size_percentage', 0),
        trade_data.get('investment_amount', 0)
    ))
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def update_trade_status(trade_id, status, exit_price=None, exit_timestamp=None,
                        profit_loss=None, profit_loss_percentage=None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    update_fields = ["status = ?"]
    update_values = [status]
    if exit_price is not None:
        update_fields.append("exit_price = ?")
        update_values.append(exit_price)
    if exit_timestamp is not None:
        update_fields.append("exit_timestamp = ?")
        update_values.append(exit_timestamp)
    if profit_loss is not None:
        update_fields.append("profit_loss = ?")
        update_values.append(profit_loss)
    if profit_loss_percentage is not None:
        update_fields.append("profit_loss_percentage = ?")
        update_values.append(profit_loss_percentage)
    update_sql = f"UPDATE trades SET {', '.join(update_fields)} WHERE id = ?"
    update_values.append(trade_id)
    cursor.execute(update_sql, update_values)
    conn.commit()
    conn.close()


def get_latest_open_trade():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, action, entry_price, amount, leverage, sl_price, tp_price
    FROM trades
    WHERE status = 'OPEN'
    ORDER BY timestamp DESC
    LIMIT 1
    ''')
    result = cursor.fetchone()
    conn.close()
    if result:
        return {
            'id': result[0], 'action': result[1], 'entry_price': result[2],
            'amount': result[3], 'leverage': result[4],
            'sl_price': result[5], 'tp_price': result[6]
        }
    return None


def get_trade_summary(days=7):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
        SUM(profit_loss) as total_profit_loss,
        AVG(profit_loss_percentage) as avg_profit_loss_percentage
    FROM trades
    WHERE exit_timestamp IS NOT NULL
    AND timestamp >= datetime('now', ?)
    ''', (f'-{days} days',))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {
            'total_trades': result[0] or 0,
            'winning_trades': result[1] or 0,
            'losing_trades': result[2] or 0,
            'total_profit_loss': result[3] or 0,
            'avg_profit_loss_percentage': result[4] or 0
        }
    return None


def get_historical_trading_data(limit=10):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT
        t.id as trade_id, t.timestamp as trade_timestamp, t.action, t.entry_price,
        t.exit_price, t.amount, t.leverage, t.sl_price, t.tp_price, t.sl_percentage,
        t.tp_percentage, t.position_size_percentage, t.status, t.profit_loss,
        t.profit_loss_percentage, a.id as analysis_id, a.reasoning, a.direction,
        a.recommended_leverage, a.recommended_position_size, a.stop_loss_percentage,
        a.take_profit_percentage
    FROM trades t
    LEFT JOIN ai_analysis a ON t.id = a.trade_id
    WHERE t.status = 'CLOSED'
    ORDER BY t.timestamp DESC
    LIMIT ?
    ''', (limit,))
    results = cursor.fetchall()
    historical_data = [{k: row[k] for k in row.keys()} for row in results]
    conn.close()
    return historical_data


def get_performance_metrics():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
        SUM(profit_loss) as total_profit_loss,
        AVG(profit_loss_percentage) as avg_profit_loss_percentage,
        MAX(profit_loss_percentage) as max_profit_percentage,
        MIN(profit_loss_percentage) as max_loss_percentage,
        AVG(CASE WHEN profit_loss > 0 THEN profit_loss_percentage ELSE NULL END) as avg_win_percentage,
        AVG(CASE WHEN profit_loss < 0 THEN profit_loss_percentage ELSE NULL END) as avg_loss_percentage
    FROM trades
    WHERE status = 'CLOSED'
    ''')
    overall_metrics = cursor.fetchone()
    cursor.execute('''
    SELECT
        action,
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
        SUM(profit_loss) as total_profit_loss,
        AVG(profit_loss_percentage) as avg_profit_loss_percentage
    FROM trades
    WHERE status = 'CLOSED'
    GROUP BY action
    ''')
    directional_metrics = cursor.fetchall()
    conn.close()

    metrics = {
        "overall": {
            "total_trades": overall_metrics[0] or 0,
            "winning_trades": overall_metrics[1] or 0,
            "losing_trades": overall_metrics[2] or 0,
            "total_profit_loss": overall_metrics[3] or 0,
            "avg_profit_loss_percentage": overall_metrics[4] or 0,
            "max_profit_percentage": overall_metrics[5] or 0,
            "max_loss_percentage": overall_metrics[6] or 0,
            "avg_win_percentage": overall_metrics[7] or 0,
            "avg_loss_percentage": overall_metrics[8] or 0
        },
        "directional": {}
    }
    if metrics["overall"]["total_trades"] > 0:
        metrics["overall"]["win_rate"] = (metrics["overall"]["winning_trades"]
                                          / metrics["overall"]["total_trades"]) * 100
    else:
        metrics["overall"]["win_rate"] = 0
    for row in directional_metrics:
        action, total, winning = row[0], row[1] or 0, row[2] or 0
        metrics["directional"][action] = {
            "total_trades": total, "winning_trades": winning, "losing_trades": row[3] or 0,
            "total_profit_loss": row[4] or 0, "avg_profit_loss_percentage": row[5] or 0,
            "win_rate": (winning / total * 100) if total > 0 else 0
        }
    return metrics


# ===== 데이터 수집 함수 =====
def fetch_multi_timeframe_data():
    timeframes = {
        "15m": {"timeframe": "15m", "limit": 96},
        "1h": {"timeframe": "1h", "limit": 48},
        "4h": {"timeframe": "4h", "limit": 30}
    }
    multi_tf_data = {}
    for tf_name, tf_params in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf_params["timeframe"], limit=tf_params["limit"])
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            multi_tf_data[tf_name] = df
            print(f"Collected {tf_name} data: {len(df)} candles")
        except Exception as e:
            print(f"Error fetching {tf_name} data: {e}")
    return multi_tf_data


def fetch_bitcoin_news():
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine": "google_news", "q": "bitcoin", "gl": "us", "hl": "en", "api_key": serp_api_key}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            news_results = response.json().get("news_results", [])
            recent_news = [{"title": n.get("title", ""), "date": n.get("date", "")} for n in news_results[:10]]
            print(f"Collected {len(recent_news)} recent news articles (title and date only)")
            return recent_news
        print(f"Error fetching news: Status code {response.status_code}")
        return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


# ===== 포지션 관리 함수 =====
def handle_position_closure(current_price, side, amount, current_trade_id=None):
    if current_trade_id is None:
        latest_trade = get_latest_open_trade()
        if latest_trade:
            current_trade_id = latest_trade['id']

    if current_trade_id:
        latest_trade = get_latest_open_trade()
        if latest_trade:
            entry_price = latest_trade['entry_price']
            action = latest_trade['action']
            if action == 'long':
                profit_loss = (current_price - entry_price) * amount
                profit_loss_percentage = (current_price / entry_price - 1) * 100
            else:
                profit_loss = (entry_price - current_price) * amount
                profit_loss_percentage = (1 - current_price / entry_price) * 100

            update_trade_status(
                current_trade_id, 'CLOSED',
                exit_price=current_price,
                exit_timestamp=datetime.now().isoformat(),
                profit_loss=profit_loss,
                profit_loss_percentage=profit_loss_percentage
            )
            print(f"\n=== Position Closed ===")
            print(f"Entry: ${entry_price:,.2f}")
            print(f"Exit: ${current_price:,.2f}")
            print(f"P/L: ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
            print("=======================")

            summary = get_trade_summary(days=7)
            if summary:
                print("\n=== 7-Day Trading Summary ===")
                print(f"Total Trades: {summary['total_trades']}")
                print(f"Win/Loss: {summary['winning_trades']}/{summary['losing_trades']}")
                if summary['total_trades'] > 0:
                    win_rate = (summary['winning_trades'] / summary['total_trades']) * 100
                    print(f"Win Rate: {win_rate:.2f}%")
                print(f"Total P/L: ${summary['total_profit_loss']:,.2f}")
                print(f"Avg P/L %: {summary['avg_profit_loss_percentage']:.2f}%")
                print("=============================")


class ProtectionError(Exception):
    pass


def open_with_protection(side, amount, entry_hint, sl_pct, tp_pct):
    """시장가 진입 후 SL/TP 설정. SL/TP가 실패하면 즉시 청산하고 예외를 발생시킴."""
    close_side = 'sell' if side == 'buy' else 'buy'
    order = exchange.create_order(symbol, 'market', side, amount)
    entry = float(order.get('average') or order.get('price') or entry_hint)

    if side == 'buy':
        sl = round(entry * (1 - sl_pct), 2)
        tp = round(entry * (1 + tp_pct), 2)
    else:
        sl = round(entry * (1 + sl_pct), 2)
        tp = round(entry * (1 - tp_pct), 2)

    try:
        exchange.create_order(symbol, 'STOP_MARKET', close_side, amount, None,
                              {'stopPrice': sl, 'reduceOnly': True})
        exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', close_side, amount, None,
                              {'stopPrice': tp, 'reduceOnly': True})
    except Exception as e:
        print("SL/TP 주문 실패, 포지션 즉시 청산:", e)
        try:
            exchange.create_order(symbol, 'market', close_side, amount, None, {'reduceOnly': True})
        except Exception as e2:
            print("!!! 청산도 실패. 바이낸스 앱에서 직접 확인하세요:", e2)
        raise ProtectionError("보호 주문 실패로 진입 취소")

    return order, entry, sl, tp


SYSTEM_PROMPT = """
You are a crypto trading expert specializing in multi-timeframe analysis and news sentiment analysis applying Kelly criterion to determine optimal position sizing, leverage, and risk management.
You adhere strictly to Warren Buffett's investment principles:

**Rule No.1: Never lose money.**
**Rule No.2: Never forget rule No.1.**

Analyze the market data across different timeframes (15m, 1h, 4h), recent news headlines, and historical trading performance to provide your trading decision.

Follow this process:
1. Review historical trading performance:
   - Examine the outcomes of recent trades (profit/loss)
   - Review your previous analysis and trading decisions
   - Identify what worked well and what didn't
   - Learn from past mistakes and successful patterns
   - Compare the performance of LONG vs SHORT positions
   - Evaluate the effectiveness of your stop-loss and take-profit levels
   - Assess which leverage settings performed best

2. Assess the current market condition across all timeframes:
   - Short-term trend (15m): Recent price action and momentum
   - Medium-term trend (1h): Intermediate market direction
   - Long-term trend (4h): Overall market bias
   - Volatility across timeframes
   - Key support/resistance levels
   - News sentiment: Analyze recent news article titles for bullish or bearish sentiment

3. Based on your analysis, determine:
   - Direction: Whether to go LONG or SHORT
   - Conviction: Probability of success (as a percentage between 51-95%)

4. Calculate Kelly position sizing:
   - Use the Kelly formula: f* = (p - q/b)
   - Where:
     * f* = fraction of capital to risk
     * p = probability of success (your conviction level)
     * q = probability of failure (1 - p)
     * b = win/loss ratio (based on stop loss and take profit distances)
   - Adjust based on historical win rates and profit/loss ratios

5. Determine optimal leverage:
   - Based on market volatility across timeframes
   - Consider higher leverage (up to 20x) in low volatility trending markets
   - Use lower leverage (1-3x) in high volatility or uncertain markets
   - Never exceed what is prudent based on your conviction level
   - Learn from past leverage decisions and their outcomes
   - Be more conservative if recent high-leverage trades resulted in losses

6. Set optimal Stop Loss (SL) and Take Profit (TP) levels:
   - Analyze recent price action, support/resistance levels
   - Consider volatility to prevent premature stop-outs
   - Set SL at a technical level that would invalidate your trade thesis
   - Set TP at a realistic target based on technical analysis
   - Both levels should be expressed as percentages from entry price
   - Adapt based on historical SL/TP performance and premature stop-outs
   - Learn from trades that hit SL vs TP and adjust accordingly

7. Apply risk management:
   - Never recommend betting more than 50% of the Kelly criterion (half-Kelly) to reduce volatility
   - If expected direction has less than 55% conviction, recommend not taking the trade (use "NO_POSITION")
   - Adjust leverage to prevent high risk exposure
   - Be more conservative if recent trades showed losses
   - If overall win rate is below 50%, be more selective with your entries

8. Provide reasoning:
   - Explain the rationale behind your trading direction, leverage, and SL/TP recommendations
   - Highlight key factors from your analysis that influenced your decision
   - Discuss how historical performance informed your current decision
   - If applicable, explain how you're adapting based on recent trade outcomes
   - Mention specific patterns you've observed in successful vs unsuccessful trades

Your response must contain ONLY a valid JSON object with exactly these 6 fields:
{
  "direction": "LONG" or "SHORT" or "NO_POSITION",
  "recommended_position_size": [final recommended position size as decimal between 0.1-1.0],
  "recommended_leverage": [an integer between 1-20],
  "stop_loss_percentage": [percentage distance from entry as decimal, e.g., 0.005 for 0.5%],
  "take_profit_percentage": [percentage distance from entry as decimal, e.g., 0.005 for 0.5%],
  "reasoning": "Your detailed explanation for all recommendations"
}

IMPORTANT: Do not format your response as a code block. Do not include ```json, ```, or any other markdown formatting. Return ONLY the raw JSON object.
"""


# ===== 메인 트레이딩 루프 =====
def main_live(dry=False):
    print("\n=== Bitcoin Trading Bot Started ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Mode:", "DRY-RUN (주문 없음)" if dry else "LIVE")
    print("Trading Pair:", symbol)
    print("Dynamic Leverage: AI Optimized")
    print("Dynamic SL/TP: AI Optimized")
    print("Multi Timeframe Analysis: 15m, 1h, 4h")
    print("News Sentiment Analysis: Enabled")
    print("Historical Performance Learning: Enabled")
    print("Database Logging: Enabled")
    print("===================================\n")

    setup_database()

    while True:
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            current_price = exchange.fetch_ticker(symbol)['last']
            print(f"\n[{current_time}] Current BTC Price: ${current_price:,.2f}")

            # ===== 1. 현재 포지션 확인 =====
            current_side = None
            amount = 0
            positions = exchange.fetch_positions([symbol])
            for position in positions:
                if position['symbol'] == 'BTC/USDT:USDT':
                    amt = float(position['info']['positionAmt'])
                    if amt > 0:
                        current_side = 'long'
                        amount = amt
                    elif amt < 0:
                        current_side = 'short'
                        amount = abs(amt)

            current_trade = get_latest_open_trade()
            current_trade_id = current_trade['id'] if current_trade else None

            # ===== 2. 포지션이 있는 경우 처리 =====
            if current_side:
                print(f"Current Position: {current_side.upper()} {amount} BTC")
                if not current_trade:
                    temp_trade_data = {
                        'action': current_side, 'entry_price': current_price, 'amount': amount,
                        'leverage': 1, 'sl_price': 0, 'tp_price': 0, 'sl_percentage': 0,
                        'tp_percentage': 0, 'position_size_percentage': 0, 'investment_amount': 0
                    }
                    current_trade_id = save_trade(temp_trade_data)
                    print("새로운 거래 기록 생성 (기존 포지션)")

            # ===== 3. 포지션이 없는 경우 처리 =====
            else:
                if current_trade:
                    handle_position_closure(current_price, current_trade['action'],
                                            current_trade['amount'], current_trade_id)

                try:
                    normal = exchange.fetch_open_orders(symbol)
                    conditional = exchange.fetch_open_orders(symbol, params={'stop': True})
                    for o in normal:
                        exchange.cancel_order(o['id'], symbol)
                    for o in conditional:
                        exchange.cancel_order(o['id'], symbol, params={'stop': True})
                    if normal or conditional:
                        print("Cancelled remaining open orders for", symbol)
                    else:
                        print("No remaining open orders to cancel.")
                except Exception as e:
                    print("Error cancelling orders:", e)

                time.sleep(5)
                print("No position. Analyzing market...")

                # ===== 4. 시장 데이터 수집 =====
                multi_tf_data = fetch_multi_timeframe_data()
                recent_news = fetch_bitcoin_news()
                historical_trading_data = get_historical_trading_data(limit=10)
                performance_metrics = get_performance_metrics()

                # ===== 5. AI 분석을 위한 데이터 준비 =====
                market_analysis = {
                    "timestamp": datetime.now().isoformat(),
                    "current_price": current_price,
                    "timeframes": {},
                    "recent_news": recent_news,
                    "historical_trading_data": historical_trading_data,
                    "performance_metrics": performance_metrics
                }
                for tf_name, df in multi_tf_data.items():
                    market_analysis["timeframes"][tf_name] = df.to_dict(orient="records")

                # ===== 6. AI 트레이딩 결정 요청 =====
                response = get_client().chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": str(market_analysis)}
                    ]
                )

                # ===== 7. AI 응답 처리 및 거래 실행 =====
                try:
                    response_content = response.choices[0].message.content.strip()
                    print(f"Raw AI response: {response_content}")

                    if response_content.startswith("```"):
                        content_parts = response_content.split("\n", 1)
                        if len(content_parts) > 1:
                            response_content = content_parts[1]
                        if "```" in response_content:
                            response_content = response_content.rsplit("```", 1)[0]
                        response_content = response_content.strip()

                    trading_decision = json.loads(response_content)

                    print(f"AI 거래 결정:")
                    print(f"방향: {trading_decision['direction']}")
                    print(f"추천 포지션 크기: {trading_decision['recommended_position_size'] * 100:.1f}%")
                    print(f"추천 레버리지: {trading_decision['recommended_leverage']}x")
                    print(f"스탑로스 레벨: {trading_decision['stop_loss_percentage'] * 100:.2f}%")
                    print(f"테이크프로핏 레벨: {trading_decision['take_profit_percentage'] * 100:.2f}%")
                    print(f"근거: {trading_decision['reasoning']}")

                    analysis_data = {
                        'current_price': current_price,
                        'direction': trading_decision['direction'],
                        'recommended_position_size': trading_decision['recommended_position_size'],
                        'recommended_leverage': trading_decision['recommended_leverage'],
                        'stop_loss_percentage': trading_decision['stop_loss_percentage'],
                        'take_profit_percentage': trading_decision['take_profit_percentage'],
                        'reasoning': trading_decision['reasoning']
                    }
                    analysis_id = save_ai_analysis(analysis_data)

                    action = trading_decision['direction'].lower()

                    # ===== 8. 트레이딩 결정에 따른 액션 실행 =====
                    if action == "no_position":
                        print("현재 시장 상황에서는 포지션을 열지 않는 것이 좋습니다.")
                        print(f"이유: {trading_decision['reasoning']}")
                        time.sleep(60)
                        continue

                    if dry:
                        print("[DRY-RUN] 주문을 실행하지 않고 판단만 기록했습니다.")
                        time.sleep(900)
                        continue

                    # ===== 9. 투자 금액 계산 =====
                    balance = exchange.fetch_balance()
                    free = balance.get('free', {})
                    available_capital = free.get('USDT') or free.get('USDC') or 0
                    if available_capital <= 0:
                        raise RuntimeError(f"사용 가능한 마진 잔고 없음: {free}")

                    position_size_percentage = trading_decision['recommended_position_size']
                    investment_amount = available_capital * position_size_percentage

                    MIN_NOTIONAL = 100
                    if investment_amount < MIN_NOTIONAL:
                        if available_capital < MIN_NOTIONAL * 2:
                            raise RuntimeError("잔고가 부족해 진입하지 않음")
                        investment_amount = MIN_NOTIONAL
                        print(f"최소 주문 금액({MIN_NOTIONAL})으로 조정됨")

                    # 시험용 상한 (정상 동작 확인 후 이 줄을 지우세요)
                    investment_amount = min(investment_amount, 200)
                    print(f"투자 금액: {investment_amount:.2f} USD")

                    # ===== 10. 주문 수량 계산 =====
                    amount = math.ceil((investment_amount / current_price) * 1000) / 1000
                    print(f"주문 수량: {amount} BTC")

                    # ===== 11. 레버리지 설정 =====
                    recommended_leverage = trading_decision['recommended_leverage']
                    try:
                        exchange.papiPostUmLeverage({
                            'symbol': exchange.market(symbol)['id'],
                            'leverage': recommended_leverage,
                        })
                    except Exception as e:
                        raise RuntimeError(f"레버리지 설정 실패, 진입 취소: {e}")
                    print(f"레버리지 설정: {recommended_leverage}x")

                    # ===== 12. 스탑로스/테이크프로핏 비율 =====
                    sl_percentage = trading_decision['stop_loss_percentage']
                    tp_percentage = trading_decision['take_profit_percentage']

                    # ===== 13. 포지션 진입 및 SL/TP 주문 실행 =====
                    if action in ("long", "short"):
                        side = 'buy' if action == 'long' else 'sell'
                        order, entry_price, sl_price, tp_price = open_with_protection(
                            side, amount, current_price, sl_percentage, tp_percentage)

                        trade_data = {
                            'action': action, 'entry_price': entry_price, 'amount': amount,
                            'leverage': recommended_leverage, 'sl_price': sl_price, 'tp_price': tp_price,
                            'sl_percentage': sl_percentage, 'tp_percentage': tp_percentage,
                            'position_size_percentage': position_size_percentage,
                            'investment_amount': investment_amount
                        }
                        trade_id = save_trade(trade_data)

                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute("UPDATE ai_analysis SET trade_id = ? WHERE id = ?", (trade_id, analysis_id))
                        conn.commit()
                        conn.close()

                        sl_sign, tp_sign = ("-", "+") if action == 'long' else ("+", "-")
                        print(f"\n=== {action.upper()} Position Opened ===")
                        print(f"Entry: ${entry_price:,.2f}")
                        print(f"Stop Loss: ${sl_price:,.2f} ({sl_sign}{sl_percentage * 100:.2f}%)")
                        print(f"Take Profit: ${tp_price:,.2f} ({tp_sign}{tp_percentage * 100:.2f}%)")
                        print(f"Leverage: {recommended_leverage}x")
                        print(f"분석 근거: {trading_decision['reasoning']}")
                        print("===========================")
                    else:
                        print("Action이 'long' 또는 'short'가 아니므로 주문을 실행하지 않습니다.")

                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류: {e}")
                    print(f"AI 응답: {response.choices[0].message.content}")
                    time.sleep(30)
                    continue
                except ProtectionError as e:
                    print(f"치명적: {e} - 봇을 중지합니다. 바이낸스 앱에서 포지션을 확인하세요.")
                    raise SystemExit(1)
                except Exception as e:
                    print(f"기타 오류: {e}")
                    time.sleep(10)
                    continue

            # ===== 14. 일정 시간 대기 후 다음 루프 실행 =====
            time.sleep(60)

        except KeyboardInterrupt:
            print("종료")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            time.sleep(5)


# ===== 백테스트 (bitcoin_final.py 의 규칙 기반 엔진 사용) =====
def run_backtest_cmd(days, compare=False):
    import importlib
    try:
        bot = importlib.import_module(BACKTEST_MODULE)
    except ModuleNotFoundError:
        print(f"'{BACKTEST_MODULE}.py' 를 찾을 수 없습니다.")
        print("같은 폴더에 bitcoin_final.py 를 두거나, 환경변수 BACKTEST_MODULE 로 파일 이름(.py 제외)을 지정하세요.")
        return
    if compare:
        bot.run_compare(days)
    else:
        bot.run_backtest(days)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, metavar="DAYS", help="규칙 기반 엔진 백테스트 (180 이상 권장)")
    ap.add_argument("--compare", type=int, metavar="DAYS", help="5가지 설정 비교 백테스트")
    ap.add_argument("--dry", action="store_true", help="GPT 판단만 기록(주문 없음)")
    args = ap.parse_args()
    if args.compare:
        run_backtest_cmd(args.compare, compare=True)
    elif args.backtest:
        run_backtest_cmd(args.backtest)
    else:
        main_live(dry=args.dry)