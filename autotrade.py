import ccxt
import os
import math
import time
import pandas as pd
import requests
import json
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ==================================================================
# 바이낸스 세팅
# ==================================================================
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True
    }
})
symbol = "BTC/USDT"
futures_symbol = "BTC/USDT:USDT"
client = OpenAI()

# SERP API 설정
serp_api_key = os.getenv("SERP_API_KEY")  # .env 파일에 SERP_API_KEY 추가 필요

# SQLite 데이터베이스 설정
DB_FILE = "bitcoin_trading.db"

# ==================================================================
# 시간대 스케줄 설정 (미국 동부시간, 서머타임 자동 반영)
# ==================================================================
ET = ZoneInfo("America/New_York")

CLOSE_HOUR, CLOSE_MIN = 8, 0          # 평일 08:00 ET: 전량 시장가 청산 + 매매 중단
ENTRY_BLOCK_HOUR, ENTRY_BLOCK_MIN = 7, 30   # 평일 07:30 ET부터 신규 진입 금지
RESUME_HOUR, RESUME_MIN = 16, 0       # 평일 16:00 ET: 매매 재개
# 토/일은 24시간 매매 (금요일 16:00 재개 -> 월요일 08:00 청산까지 연속 운영)


def now_et():
    return datetime.now(ET)


def _minutes(dt):
    return dt.hour * 60 + dt.minute


def is_trading_allowed(dt=None):
    """매매 허용 여부: 토/일은 항상 허용, 평일은 08:00~16:00(ET)만 금지"""
    dt = dt or now_et()
    if dt.weekday() >= 5:  # 5=토, 6=일
        return True
    t = _minutes(dt)
    return not (CLOSE_HOUR * 60 + CLOSE_MIN <= t < RESUME_HOUR * 60 + RESUME_MIN)


def can_open_new_entry(dt=None):
    """신규 진입 허용 여부: 매매 허용 시간대 + 평일 07:30~08:00 제외"""
    dt = dt or now_et()
    if not is_trading_allowed(dt):
        return False
    if dt.weekday() < 5:
        t = _minutes(dt)
        if ENTRY_BLOCK_HOUR * 60 + ENTRY_BLOCK_MIN <= t < CLOSE_HOUR * 60 + CLOSE_MIN:
            return False
    return True


def seconds_until_next_event(dt=None):
    """다음 상태 전환(평일 08:00 청산 / 평일 16:00 재개)까지 남은 초"""
    dt = dt or now_et()
    now_utc = dt.astimezone(timezone.utc)
    best = None
    for day_offset in range(0, 9):
        base = dt + timedelta(days=day_offset)  # 벽시계(wall-clock) 기준 날짜 이동
        if base.weekday() >= 5:
            continue
        for h, m in ((CLOSE_HOUR, CLOSE_MIN), (RESUME_HOUR, RESUME_MIN)):
            c = base.replace(hour=h, minute=m, second=0, microsecond=0)
            delta = (c.astimezone(timezone.utc) - now_utc).total_seconds()
            if delta > 0 and (best is None or delta < best):
                best = delta
    return best


def smart_sleep(seconds):
    """일반 sleep이되, 08:00/16:00 전환 시각을 넘기지 않고 정확히 그 시각에 깨어남"""
    remaining = seconds_until_next_event()
    if remaining is not None and remaining < seconds:
        time.sleep(remaining + 0.3)  # 전환 시각 직후 즉시 실행
    else:
        time.sleep(seconds)


def extract_fill_price(order, fallback):
    """시장가 주문의 체결 평균가 추출 (없으면 fetch_order 재조회, 그래도 없으면 fallback)"""
    try:
        avg = order.get('average') or order.get('price')
        if not avg and order.get('id'):
            time.sleep(0.5)
            fetched = exchange.fetch_order(order['id'], symbol)
            avg = fetched.get('average') or fetched.get('price')
        return float(avg) if avg else fallback
    except Exception as e:
        print("체결가 조회 오류:", e)
        return fallback


def get_open_position():
    """현재 BTC 선물 포지션 반환: (side, amount) / 없으면 (None, 0)"""
    for position in exchange.fetch_positions([symbol]):
        if position['symbol'] == futures_symbol:
            amt = float(position['info']['positionAmt'])
            if amt > 0:
                return 'long', amt
            elif amt < 0:
                return 'short', abs(amt)
    return None, 0


def cancel_all_open_orders():
    """남아있는 SL/TP 등 미체결 주문 전부 취소"""
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        for order in open_orders:
            exchange.cancel_order(order['id'], symbol)
        if open_orders:
            print(f"미체결 주문 {len(open_orders)}건 취소 완료")
    except Exception as e:
        print("주문 취소 오류:", e)


def close_position_market(max_retry=3):
    """
    포지션을 reduceOnly 시장가로 청산하고 체결가를 반환.
    청산을 먼저 하고, 성공 후에 SL/TP를 취소합니다.
    (청산 실패 시 SL/TP 보호가 그대로 남아있도록)
    반환: 체결가(float) 또는 None(포지션 없음/실패)
    """
    for attempt in range(1, max_retry + 1):
        try:
            side, amt = get_open_position()
            if side is None:
                return None
            order_side = 'sell' if side == 'long' else 'buy'
            order = exchange.create_order(
                symbol, 'market', order_side, amt, None, {'reduceOnly': True}
            )
            fallback = exchange.fetch_ticker(symbol)['last']
            fill_price = extract_fill_price(order, fallback)
            print(f"[스케줄 청산] {order_side.upper()} {amt} BTC 시장가 체결가: ${fill_price:,.2f}")

            # 청산 직후 잔여 SL/TP 주문 정리 (신규 포지션 오작동 방지)
            cancel_all_open_orders()

            # 포지션이 실제로 0인지 확인
            time.sleep(1)
            side_after, _ = get_open_position()
            if side_after is None:
                return fill_price
            print("청산 후에도 포지션이 남아있음 - 재시도")
        except Exception as e:
            print(f"청산 시도 {attempt}/{max_retry} 실패: {e}")
            time.sleep(1)
    print("!!! 청산 실패: 수동 확인이 필요합니다 !!!")
    return None


def handle_off_hours():
    """매매 금지 시간대 처리: 포지션 청산 + DB 정리 + 미체결 주문 취소"""
    side, amt = get_open_position()
    trade = get_latest_open_trade()

    if side is not None:
        print(f"[{now_et().strftime('%a %H:%M:%S ET')}] 매매 금지 시간대 - 포지션 청산 시작 ({side.upper()} {amt} BTC)")
        exit_price = close_position_market()
        if exit_price is not None:
            if trade:
                handle_position_closure(exit_price, trade['action'], amt, trade['id'])
            return
    else:
        # 포지션은 없는데 DB에 열린 거래가 남아있는 경우 (SL/TP 체결 등) 정리
        if trade:
            try:
                price = exchange.fetch_ticker(symbol)['last']
                handle_position_closure(price, trade['action'], trade['amount'], trade['id'])
            except Exception as e:
                print("DB 정리 오류:", e)

    cancel_all_open_orders()


# ==================================================================
# SQLite 데이터베이스
# ==================================================================
def setup_database():
    """데이터베이스 및 필요한 테이블 생성"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 거래 기록 테이블
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

    # AI 분석 결과 테이블
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
    """AI 분석 결과를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO ai_analysis (
        timestamp,
        current_price,
        direction,
        recommended_position_size,
        recommended_leverage,
        stop_loss_percentage,
        take_profit_percentage,
        reasoning,
        trade_id
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
    """거래 정보를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO trades (
        timestamp,
        action,
        entry_price,
        amount,
        leverage,
        sl_price,
        tp_price,
        sl_percentage,
        tp_percentage,
        position_size_percentage,
        investment_amount
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


def update_trade_status(trade_id, status, exit_price=None, exit_timestamp=None, profit_loss=None, profit_loss_percentage=None):
    """거래 상태 업데이트"""
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
    """가장 최근의 열린 거래 정보 가져오기"""
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
            'id': result[0],
            'action': result[1],
            'entry_price': result[2],
            'amount': result[3],
            'leverage': result[4],
            'sl_price': result[5],
            'tp_price': result[6]
        }
    return None


def get_trade_summary(days=7):
    """최근 거래 요약 정보 가져오기"""
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
    """과거 거래 내역과 관련 AI 분석 결과를 가져오기"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # 컬럼명으로 접근 가능하게 설정
    cursor = conn.cursor()

    cursor.execute('''
    SELECT
        t.id as trade_id,
        t.timestamp as trade_timestamp,
        t.action,
        t.entry_price,
        t.exit_price,
        t.amount,
        t.leverage,
        t.sl_price,
        t.tp_price,
        t.sl_percentage,
        t.tp_percentage,
        t.position_size_percentage,
        t.status,
        t.profit_loss,
        t.profit_loss_percentage,
        a.id as analysis_id,
        a.reasoning,
        a.direction,
        a.recommended_leverage,
        a.recommended_position_size,
        a.stop_loss_percentage,
        a.take_profit_percentage
    FROM
        trades t
    LEFT JOIN
        ai_analysis a ON t.id = a.trade_id
    WHERE
        t.status = 'CLOSED'
    ORDER BY
        t.timestamp DESC
    LIMIT ?
    ''', (limit,))

    results = cursor.fetchall()

    historical_data = []
    for row in results:
        historical_data.append({k: row[k] for k in row.keys()})

    conn.close()
    return historical_data


def get_performance_metrics():
    """거래 성과 메트릭스 계산"""
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
        metrics["overall"]["win_rate"] = (metrics["overall"]["winning_trades"] / metrics["overall"]["total_trades"]) * 100
    else:
        metrics["overall"]["win_rate"] = 0

    for row in directional_metrics:
        action = row[0]
        total = row[1] or 0
        winning = row[2] or 0

        direction_metrics = {
            "total_trades": total,
            "winning_trades": winning,
            "losing_trades": row[3] or 0,
            "total_profit_loss": row[4] or 0,
            "avg_profit_loss_percentage": row[5] or 0,
            "win_rate": (winning / total * 100) if total > 0 else 0
        }

        metrics["directional"][action] = direction_metrics

    return metrics


print("\n=== Bitcoin Trading Bot Started ===")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  ET: {now_et().strftime('%Y-%m-%d %H:%M:%S %Z')}")
print("Trading Pair:", symbol)
print("Dynamic Leverage: AI Optimized")
print("Dynamic SL/TP: AI Optimized")
print("Multi Timeframe Analysis: 15m, 1h, 4h")
print("News Sentiment Analysis: Enabled")
print("Historical Performance Learning: Enabled")
print("Database Logging: Enabled")
print("Schedule (ET): Weekdays 07:30 no new entry / 08:00 market close / 16:00 resume, Weekends 24h")
print("===================================\n")

# 데이터베이스 설정
setup_database()


# ==================================================================
# 데이터 수집
# ==================================================================
def fetch_multi_timeframe_data():
    timeframes = {
        "15m": {"timeframe": "15m", "limit": 96},  # 24시간
        "1h": {"timeframe": "1h", "limit": 48},    # 48시간
        "4h": {"timeframe": "4h", "limit": 30}     # 5일
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
        params = {
            "engine": "google_news",
            "q": "bitcoin",
            "gl": "us",
            "hl": "en",
            "api_key": serp_api_key
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            news_results = data.get("news_results", [])

            recent_news = []
            for i, news in enumerate(news_results[:10]):
                news_item = {
                    "title": news.get("title", ""),
                    "date": news.get("date", "")
                }
                recent_news.append(news_item)

            print(f"Collected {len(recent_news)} recent news articles (title and date only)")
            return recent_news
        else:
            print(f"Error fetching news: Status code {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


# ==================================================================
# 포지션 종료 처리 (DB 업데이트)
# ==================================================================
def handle_position_closure(current_price, side, amount, current_trade_id=None):
    """포지션 종료 시 데이터베이스 업데이트"""
    latest_trade = get_latest_open_trade()
    if current_trade_id is None and latest_trade:
        current_trade_id = latest_trade['id']

    if current_trade_id and latest_trade:
        entry_price = latest_trade['entry_price']
        action = latest_trade['action']

        # 손익 계산
        if action == 'long':
            profit_loss = (current_price - entry_price) * amount
            profit_loss_percentage = (current_price / entry_price - 1) * 100
        else:  # 'short'
            profit_loss = (entry_price - current_price) * amount
            profit_loss_percentage = (1 - current_price / entry_price) * 100

        update_trade_status(
            current_trade_id,
            'CLOSED',
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


# ==================================================================
# AI 시스템 프롬프트
# ==================================================================
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


# ==================================================================
# 메인 루프
# ==================================================================
while True:
    try:
        # ------------------------------------------------------------
        # [1] 매매 금지 시간대(평일 08:00~16:00 ET): 즉시 청산 후 대기
        #     (가격 조회 등 다른 작업보다 먼저 실행해서 청산 지연 최소화)
        # ------------------------------------------------------------
        if not is_trading_allowed():
            handle_off_hours()
            print(f"[{now_et().strftime('%a %H:%M:%S ET')}] 매매 금지 시간대 - 16:00 ET 재개 대기")
            smart_sleep(60)  # 16:00에 맞춰 정확히 깨어남
            continue

        # ------------------------------------------------------------
        # [2] 정상 매매 시간대
        # ------------------------------------------------------------
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = exchange.fetch_ticker(symbol)['last']
        print(f"\n[{current_time}] [{now_et().strftime('%a %H:%M ET')}] Current BTC Price: ${current_price:,.2f}")

        # 포지션 확인
        current_side, amount = get_open_position()

        # 열린 거래 확인
        current_trade = get_latest_open_trade()
        current_trade_id = current_trade['id'] if current_trade else None

        if current_side:
            print(f"Current Position: {current_side.upper()} {amount} BTC")

            # 포지션이 있지만 DB에 기록이 없는 경우 (프로그램 재시작 등)
            if not current_trade:
                temp_trade_data = {
                    'action': current_side,
                    'entry_price': current_price,
                    'amount': amount,
                    'leverage': 1,
                    'sl_price': 0,
                    'tp_price': 0,
                    'sl_percentage': 0,
                    'tp_percentage': 0,
                    'position_size_percentage': 0,
                    'investment_amount': 0
                }
                current_trade_id = save_trade(temp_trade_data)
                print("새로운 거래 기록 생성 (기존 포지션)")
        else:
            # 이전에 포지션이 있었고 DB에 열린 거래가 있는 경우 (SL/TP로 종료됨)
            if current_trade:
                handle_position_closure(current_price, current_trade['action'], current_trade['amount'], current_trade_id)

            # 포지션이 없을 경우, 남아있는 미체결 주문 취소
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                if open_orders:
                    for order in open_orders:
                        exchange.cancel_order(order['id'], symbol)
                    print("Cancelled remaining open orders for", symbol)
                else:
                    print("No remaining open orders to cancel.")
            except Exception as e:
                print("Error cancelling orders:", e)

            # ---- 신규 진입 차단 구간(평일 07:30~08:00 ET) ----
            if not can_open_new_entry():
                print(f"[{now_et().strftime('%a %H:%M ET')}] 신규 진입 차단 구간(07:30~08:00 ET) - 분석/진입 생략")
                smart_sleep(60)
                continue

            time.sleep(5)
            print("No position. Analyzing market...")

            multi_tf_data = fetch_multi_timeframe_data()
            recent_news = fetch_bitcoin_news()
            historical_trading_data = get_historical_trading_data(limit=10)
            performance_metrics = get_performance_metrics()

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

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(market_analysis)}
                ]
            )

            # AI 응답 파싱
            try:
                response_content = response.choices[0].message.content.strip()
                print(f"Raw AI response: {response_content}")

                # 코드 블록 제거
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
                print(f"추천 포지션 크기: {trading_decision['recommended_position_size']*100:.1f}%")
                print(f"추천 레버리지: {trading_decision['recommended_leverage']}x")
                print(f"스탑로스 레벨: {trading_decision['stop_loss_percentage']*100:.2f}%")
                print(f"테이크프로핏 레벨: {trading_decision['take_profit_percentage']*100:.2f}%")
                print(f"근거: {trading_decision['reasoning']}")

                # AI 분석 결과를 데이터베이스에 저장
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

                # 포지션을 열지 말아야 하는 경우
                if action == "no_position":
                    print("현재 시장 상황에서는 포지션을 열지 않는 것이 좋습니다.")
                    print(f"이유: {trading_decision['reasoning']}")
                    smart_sleep(60)
                    continue

                # ---- 주문 직전 재확인: AI 응답 대기 중 차단 구간에 진입했는지 ----
                if not can_open_new_entry():
                    print("분석 중 신규 진입 차단 구간에 진입 - 주문 취소")
                    smart_sleep(60)
                    continue

                # 투자 금액 계산
                balance = exchange.fetch_balance()
                available_capital = balance['USDT']['free']
                position_size_percentage = trading_decision['recommended_position_size']
                investment_amount = available_capital * position_size_percentage

                # 최소 주문 금액 확인 (최소 100 USDT)
                if investment_amount < 100:
                    investment_amount = 100
                    print(f"최소 주문 금액(100 USDT)으로 조정됨")

                print(f"투자 금액: {investment_amount:.2f} USDT")

                # 주문 수량 계산
                amount = math.ceil((investment_amount / current_price) * 1000) / 1000
                print(f"주문 수량: {amount} BTC")

                # AI 추천 레버리지 설정
                recommended_leverage = trading_decision['recommended_leverage']
                exchange.set_leverage(recommended_leverage, symbol)
                print(f"레버리지 설정: {recommended_leverage}x")

                sl_percentage = trading_decision['stop_loss_percentage']
                tp_percentage = trading_decision['take_profit_percentage']

                # 포지션 진입 및 SL/TP 주문
                if action == "long":
                    order = exchange.create_market_buy_order(symbol, amount)
                    entry_price = current_price
                    sl_price = round(entry_price * (1 - sl_percentage), 2)
                    tp_price = round(entry_price * (1 + tp_percentage), 2)

                    # SL/TP 주문 (reduceOnly: 청산 후 잔여 주문이 새 포지션을 여는 것을 방지)
                    exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price, 'reduceOnly': True})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price, 'reduceOnly': True})

                    trade_data = {
                        'action': 'long',
                        'entry_price': entry_price,
                        'amount': amount,
                        'leverage': recommended_leverage,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'sl_percentage': sl_percentage,
                        'tp_percentage': tp_percentage,
                        'position_size_percentage': position_size_percentage,
                        'investment_amount': investment_amount
                    }
                    trade_id = save_trade(trade_data)

                    update_analysis_sql = "UPDATE ai_analysis SET trade_id = ? WHERE id = ?"
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute(update_analysis_sql, (trade_id, analysis_id))
                    conn.commit()
                    conn.close()

                    print(f"\n=== LONG Position Opened ===")
                    print(f"Entry: ${entry_price:,.2f}")
                    print(f"Stop Loss: ${sl_price:,.2f} (-{sl_percentage*100:.2f}%)")
                    print(f"Take Profit: ${tp_price:,.2f} (+{tp_percentage*100:.2f}%)")
                    print(f"Leverage: {recommended_leverage}x")
                    print(f"분석 근거: {trading_decision['reasoning']}")
                    print("===========================")

                elif action == "short":
                    order = exchange.create_market_sell_order(symbol, amount)
                    entry_price = current_price
                    sl_price = round(entry_price * (1 + sl_percentage), 2)
                    tp_price = round(entry_price * (1 - tp_percentage), 2)

                    exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price, 'reduceOnly': True})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price, 'reduceOnly': True})

                    trade_data = {
                        'action': 'short',
                        'entry_price': entry_price,
                        'amount': amount,
                        'leverage': recommended_leverage,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'sl_percentage': sl_percentage,
                        'tp_percentage': tp_percentage,
                        'position_size_percentage': position_size_percentage,
                        'investment_amount': investment_amount
                    }
                    trade_id = save_trade(trade_data)

                    update_analysis_sql = "UPDATE ai_analysis SET trade_id = ? WHERE id = ?"
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute(update_analysis_sql, (trade_id, analysis_id))
                    conn.commit()
                    conn.close()

                    print(f"\n=== SHORT Position Opened ===")
                    print(f"Entry: ${entry_price:,.2f}")
                    print(f"Stop Loss: ${sl_price:,.2f} (+{sl_percentage*100:.2f}%)")
                    print(f"Take Profit: ${tp_price:,.2f} (-{tp_percentage*100:.2f}%)")
                    print(f"Leverage: {recommended_leverage}x")
                    print(f"분석 근거: {trading_decision['reasoning']}")
                    print("============================")
                else:
                    print("Action이 'long' 또는 'short'가 아니므로 주문을 실행하지 않습니다.")

            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print(f"AI 응답: {response.choices[0].message.content}")
                smart_sleep(30)
                continue
            except Exception as e:
                print(f"기타 오류: {e}")
                smart_sleep(10)
                continue

        smart_sleep(60)  # 메인 루프 1분 주기 (08:00 / 16:00 전환 시각에는 정확히 깨어남)

    except Exception as e:
        print(f"\n Error: {e}")
        time.sleep(5)