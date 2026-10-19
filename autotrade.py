"""
AI 비트코인 트레이딩 봇 - 교육용 코드
--------------------------------------------------------
기능:
- 멀티 타임프레임 분석 (15분, 1시간, 4시간 차트)
- 뉴스 감성 분석
- AI 기반 포지션 사이징 및 레버리지 최적화
- 동적 스탑로스/테이크프로핏 설정
- 거래 내역 데이터베이스 기록 및 성과 분석
- 과거 거래 데이터 기반 학습
--------------------------------------------------------
"""
# ===== 필요한 라이브러리 임포트 =====
import ccxt  # 암호화폐 거래소 API 라이브러리
import os  # 환경 변수 및 파일 시스템 접근
import math  # 수학 연산
import time  # 시간 지연 및 타임스탬프
import pandas as pd  # 데이터 분석 및 조작
import requests  # HTTP 요청
import json  # JSON 데이터 처리
import sqlite3  # 로컬 데이터베이스
from dotenv import load_dotenv  # 환경 변수 로드
load_dotenv()  # .env 파일에서 환경 변수 로드
from openai import OpenAI  # OpenAI API 접근
from datetime import datetime  # 날짜 및 시간 처리

# ===== 설정 및 초기화 =====
# 바이낸스 API 설정
api_key = os.getenv("BINANCE_API_KEY")  # 바이낸스 API 키
secret = os.getenv("BINANCE_SECRET_KEY")  # 바이낸스 시크릿 키
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,  # API 호출 제한 준수
    'options': {
        'defaultType': 'future',  # 선물 거래 설정
        'adjustForTimeDifference': True  # 시간대 차이 조정
    }
})
symbol = "BTC/USDT"  # 거래 페어 설정

# OpenAI API 클라이언트 초기화
client = OpenAI()

# SERP API 설정 (뉴스 데이터 수집용)
serp_api_key = os.getenv("SERP_API_KEY")  # 서프 API 키

# SQLite 데이터베이스 설정
DB_FILE = "bitcoin_trading.db"  # 데이터베이스 파일명

# ===== 데이터베이스 관련 함수 =====
def setup_database():
    """
    데이터베이스 및 필요한 테이블 생성
    
    거래 기록과 AI 분석 결과를 저장하기 위한 테이블을 생성합니다.
    - trades: 모든 거래 정보 (진입가, 청산가, 손익 등)
    - ai_analysis: AI의 분석 결과 및 추천 사항
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 거래 기록 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,           -- 거래 시작 시간
        action TEXT NOT NULL,              -- long 또는 short
        entry_price REAL NOT NULL,         -- 진입 가격
        amount REAL NOT NULL,              -- 거래량 (BTC)
        leverage INTEGER NOT NULL,         -- 레버리지 배수
        sl_price REAL NOT NULL,            -- 스탑로스 가격
        tp_price REAL NOT NULL,            -- 테이크프로핏 가격
        sl_percentage REAL NOT NULL,       -- 스탑로스 백분율
        tp_percentage REAL NOT NULL,       -- 테이크프로핏 백분율
        position_size_percentage REAL NOT NULL,  -- 자본 대비 포지션 크기
        investment_amount REAL NOT NULL,   -- 투자 금액 (USDT)
        status TEXT DEFAULT 'OPEN',        -- 거래 상태 (OPEN/CLOSED)
        exit_price REAL,                   -- 청산 가격
        exit_timestamp TEXT,               -- 청산 시간
        profit_loss REAL,                  -- 손익 (USDT)
        profit_loss_percentage REAL        -- 손익 백분율
    )
    ''')
    
    # AI 분석 결과 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ai_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,               -- 분석 시간
        current_price REAL NOT NULL,           -- 분석 시점 가격
        direction TEXT NOT NULL,               -- 방향 추천 (LONG/SHORT/NO_POSITION)
        recommended_position_size REAL NOT NULL,  -- 추천 포지션 크기
        recommended_leverage INTEGER NOT NULL,    -- 추천 레버리지
        stop_loss_percentage REAL NOT NULL,       -- 추천 스탑로스 비율
        take_profit_percentage REAL NOT NULL,     -- 추천 테이크프로핏 비율
        reasoning TEXT NOT NULL,                  -- 분석 근거 설명
        trade_id INTEGER,                         -- 연결된 거래 ID
        FOREIGN KEY (trade_id) REFERENCES trades (id)  -- 외래 키 설정
    )
    ''')
    
    conn.commit()
    conn.close()
    print("데이터베이스 설정 완료")

def save_ai_analysis(analysis_data, trade_id=None):
    """
    AI 분석 결과를 데이터베이스에 저장
    
    매개변수:
        analysis_data (dict): AI 분석 결과 데이터
        trade_id (int, optional): 연결된 거래 ID
        
    반환값:
        int: 생성된 분석 기록의 ID
    """
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
        datetime.now().isoformat(),  # 현재 시간
        analysis_data.get('current_price', 0),  # 현재 가격
        analysis_data.get('direction', 'NO_POSITION'),  # 추천 방향
        analysis_data.get('recommended_position_size', 0),  # 추천 포지션 크기
        analysis_data.get('recommended_leverage', 0),  # 추천 레버리지
        analysis_data.get('stop_loss_percentage', 0),  # 스탑로스 비율
        analysis_data.get('take_profit_percentage', 0),  # 테이크프로핏 비율
        analysis_data.get('reasoning', ''),  # 분석 근거
        trade_id  # 연결된 거래 ID
    ))
    
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id

def save_trade(trade_data):
    """
    거래 정보를 데이터베이스에 저장
    
    매개변수:
        trade_data (dict): 거래 정보 데이터
        
    반환값:
        int: 생성된 거래 기록의 ID
    """
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
        datetime.now().isoformat(),  # 진입 시간
        trade_data.get('action', ''),  # 포지션 방향
        trade_data.get('entry_price', 0),  # 진입 가격
        trade_data.get('amount', 0),  # 거래량
        trade_data.get('leverage', 0),  # 레버리지
        trade_data.get('sl_price', 0),  # 스탑로스 가격
        trade_data.get('tp_price', 0),  # 테이크프로핏 가격
        trade_data.get('sl_percentage', 0),  # 스탑로스 비율
        trade_data.get('tp_percentage', 0),  # 테이크프로핏 비율
        trade_data.get('position_size_percentage', 0),  # 자본 대비 포지션 크기
        trade_data.get('investment_amount', 0)  # 투자 금액
    ))
    
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id

def update_trade_status(trade_id, status, exit_price=None, exit_timestamp=None, profit_loss=None, profit_loss_percentage=None):
    """
    거래 상태를 업데이트합니다
    
    매개변수:
        trade_id (int): 업데이트할 거래의 ID
        status (str): 새 상태 ('OPEN' 또는 'CLOSED')
        exit_price (float, optional): 청산 가격
        exit_timestamp (str, optional): 청산 시간
        profit_loss (float, optional): 손익 금액
        profit_loss_percentage (float, optional): 손익 비율
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 동적으로 SQL 업데이트 쿼리 구성
    update_fields = ["status = ?"]
    update_values = [status]
    
    # 제공된 필드만 업데이트에 포함
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
    """
    가장 최근의 열린 거래 정보를 가져옵니다
    
    반환값:
        dict: 거래 정보 또는 None (열린 거래가 없는 경우)
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, action, entry_price, amount, leverage, sl_price, tp_price
    FROM trades
    WHERE status = 'OPEN'
    ORDER BY timestamp DESC  -- 가장 최근 거래 먼저
    LIMIT 1
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    # 결과가 있을 경우 사전 형태로 변환하여 반환
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
    return None  # 열린 거래가 없음

def get_trade_summary(days=7):
    """
    지정된 일수 동안의 거래 요약 정보를 가져옵니다
    
    매개변수:
        days (int): 요약할 기간(일)
        
    반환값:
        dict: 거래 요약 정보 또는 None
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        COUNT(*) as total_trades,                            -- 총 거래 수
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,  -- 이익 거래 수
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,   -- 손실 거래 수
        SUM(profit_loss) as total_profit_loss,               -- 총 손익
        AVG(profit_loss_percentage) as avg_profit_loss_percentage  -- 평균 손익률
    FROM trades
    WHERE exit_timestamp IS NOT NULL  -- 청산된 거래만
    AND timestamp >= datetime('now', ?)  -- 지정된 일수 내 거래만
    ''', (f'-{days} days',))
    
    result = cursor.fetchone()
    conn.close()
    
    # 결과가 있을 경우 사전 형태로 변환하여 반환
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
    """
    과거 거래 내역과 관련 AI 분석 결과를 가져옵니다
    
    매개변수:
        limit (int): 가져올 최대 거래 기록 수
        
    반환값:
        list: 거래 및 분석 데이터 사전 목록
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # 컬럼명으로 결과에 접근 가능하게 설정
    cursor = conn.cursor()
    
    # 완료된 거래 내역과 관련 AI 분석 함께 조회 (LEFT JOIN 사용)
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
        t.status = 'CLOSED'  -- 완료된 거래만
    ORDER BY 
        t.timestamp DESC  -- 최신 거래 먼저
    LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    
    # 결과를 사전 목록으로 변환
    historical_data = []
    for row in results:
        historical_data.append({k: row[k] for k in row.keys()})
    
    conn.close()
    return historical_data

def get_performance_metrics():
    """
    거래 성과 메트릭스를 계산합니다
    
    이 함수는 다음을 포함한 전체 및 방향별(롱/숏) 성과 지표를 계산합니다:
    - 총 거래 수
    - 승률
    - 평균 수익률
    - 최대 이익/손실
    - 방향별 성과
    
    반환값:
        dict: 성과 메트릭스 데이터
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 전체 거래 성과 쿼리
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
    
    # 방향별(롱/숏) 성과 쿼리
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
    
    # 결과 구성
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
    
    # 승률 계산
    if metrics["overall"]["total_trades"] > 0:
        metrics["overall"]["win_rate"] = (metrics["overall"]["winning_trades"] / metrics["overall"]["total_trades"]) * 100
    else:
        metrics["overall"]["win_rate"] = 0
    
    # 방향별 메트릭스 추가
    for row in directional_metrics:
        action = row[0]  # 'long' 또는 'short'
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

# ===== 데이터 수집 함수 =====
def fetch_multi_timeframe_data():
    """
    여러 타임프레임의 가격 데이터를 수집합니다
    
    각 타임프레임(15분, 1시간, 4시간)에 대해 다음 데이터를 가져옵니다:
    - 날짜/시간
    - 시가
    - 고가
    - 저가
    - 종가
    - 거래량
    
    반환값:
        dict: 타임프레임별 DataFrame 데이터
    """
    # 타임프레임별 데이터 수집 설정
    timeframes = {
        "15m": {"timeframe": "15m", "limit": 96},  # 24시간 (15분 * 96)
        "1h": {"timeframe": "1h", "limit": 48},    # 48시간 (1시간 * 48)
        "4h": {"timeframe": "4h", "limit": 30}     # 5일 (4시간 * 30)
    }
    
    multi_tf_data = {}
    
    # 각 타임프레임별로 데이터 수집
    for tf_name, tf_params in timeframes.items():
        try:
            # OHLCV 데이터 가져오기 (시가, 고가, 저가, 종가, 거래량)
            ohlcv = exchange.fetch_ohlcv(
                symbol, 
                timeframe=tf_params["timeframe"], 
                limit=tf_params["limit"]
            )
            
            # 데이터프레임으로 변환
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 타임스탬프를 날짜/시간 형식으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 결과 딕셔너리에 저장
            multi_tf_data[tf_name] = df
            print(f"Collected {tf_name} data: {len(df)} candles")
        except Exception as e:
            print(f"Error fetching {tf_name} data: {e}")
    
    return multi_tf_data

def fetch_bitcoin_news():
    """
    비트코인 관련 최신 뉴스를 가져옵니다
    
    SERP API를 사용해 Google 뉴스에서 비트코인 관련 최신 뉴스 10개를 가져옵니다.
    
    반환값:
        list: 최신 뉴스 기사 정보 (제목과 날짜만 포함)
    """
    try:
        # SERP API 요청 설정
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_news",  # Google 뉴스 검색
            "q": "bitcoin",           # 검색어: 비트코인
            "gl": "us",               # 국가: 미국
            "hl": "en",               # 언어: 영어
            "api_key": serp_api_key   # API 키
        }
        
        # API 요청 보내기
        response = requests.get(url, params=params)
        
        # 응답 확인 및 처리
        if response.status_code == 200:
            data = response.json()
            news_results = data.get("news_results", [])
            
            # 최신 뉴스 10개만 추출하고 제목과 날짜만 포함
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

# ===== 포지션 관리 함수 =====
def handle_position_closure(current_price, side, amount, current_trade_id=None):
    """
    포지션 종료 시 데이터베이스를 업데이트하고 결과를 표시합니다
    
    매개변수:
        current_price (float): 현재 가격(청산 가격)
        side (str): 포지션 방향 ('long' 또는 'short')
        amount (float): 포지션 수량
        current_trade_id (int, optional): 현재 거래 ID
    """
    # 거래 ID가 제공되지 않은 경우 최신 열린 거래 정보 조회
    if current_trade_id is None:
        latest_trade = get_latest_open_trade()
        if latest_trade:
            current_trade_id = latest_trade['id']
    
    if current_trade_id:
        # 가장 최근의 열린 거래 가져오기
        latest_trade = get_latest_open_trade()
        if latest_trade:
            entry_price = latest_trade['entry_price']
            action = latest_trade['action']
            
            # 손익 계산 (방향에 따라 다름)
            if action == 'long':
                # 롱 포지션의 경우: (청산가 - 진입가) * 수량
                profit_loss = (current_price - entry_price) * amount
                profit_loss_percentage = (current_price / entry_price - 1) * 100
            else:  # 'short'
                # 숏 포지션의 경우: (진입가 - 청산가) * 수량
                profit_loss = (entry_price - current_price) * amount
                profit_loss_percentage = (1 - current_price / entry_price) * 100
                
            # 거래 상태 업데이트
            update_trade_status(
                current_trade_id,
                'CLOSED',  # 상태를 '종료됨'으로 변경
                exit_price=current_price,
                exit_timestamp=datetime.now().isoformat(),
                profit_loss=profit_loss,
                profit_loss_percentage=profit_loss_percentage
            )
            
            # 결과 출력
            print(f"\n=== Position Closed ===")
            print(f"Entry: ${entry_price:,.2f}")
            print(f"Exit: ${current_price:,.2f}")
            print(f"P/L: ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
            print("=======================")
            
            # 최근 거래 요약 표시
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

# ===== 메인 프로그램 시작 =====
print("\n=== Bitcoin Trading Bot Started ===")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Trading Pair:", symbol)
print("Dynamic Leverage: AI Optimized")
print("Dynamic SL/TP: AI Optimized")
print("Multi Timeframe Analysis: 15m, 1h, 4h")
print("News Sentiment Analysis: Enabled")
print("Historical Performance Learning: Enabled")
print("Database Logging: Enabled")
print("===================================\n")

# 데이터베이스 설정
setup_database()

# ===== 메인 트레이딩 루프 =====
while True:
    try:
        # 현재 시간 및 가격 조회
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = exchange.fetch_ticker(symbol)['last']
        print(f"\n[{current_time}] Current BTC Price: ${current_price:,.2f}")

        # ===== 1. 현재 포지션 확인 =====
        current_side = None  # 현재 포지션 방향 (long/short/None)
        amount = 0  # 포지션 수량
        
        # 바이낸스에서 현재 포지션 조회
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
        
        # 데이터베이스에서 현재 거래 정보 조회
        current_trade = get_latest_open_trade()
        current_trade_id = current_trade['id'] if current_trade else None
        
        # ===== 2. 포지션이 있는 경우 처리 =====
        if current_side:
            print(f"Current Position: {current_side.upper()} {amount} BTC")
            
            # 포지션이 있지만 DB에 기록이 없는 경우 (프로그램 재시작 등)
            if not current_trade:
                # 임시 거래 정보 생성하여 DB에 저장
                temp_trade_data = {
                    'action': current_side,
                    'entry_price': current_price,  # 현재 가격으로 임시 설정
                    'amount': amount,
                    'leverage': 1,  # 기본값
                    'sl_price': 0,
                    'tp_price': 0,
                    'sl_percentage': 0,
                    'tp_percentage': 0,
                    'position_size_percentage': 0,
                    'investment_amount': 0
                }
                current_trade_id = save_trade(temp_trade_data)
                print("새로운 거래 기록 생성 (기존 포지션)")
        
        # ===== 3. 포지션이 없는 경우 처리 =====
        else:
            # 이전에 포지션이 있었고 DB에 열린 거래가 있는 경우 (포지션 종료됨)
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
                
            # 잠시 대기 후 시장 분석 시작
            time.sleep(5)
            print("No position. Analyzing market...")

            # ===== 4. 시장 데이터 수집 =====
            # 멀티 타임프레임 차트 데이터 수집
            multi_tf_data = fetch_multi_timeframe_data()
            
            # 최신 비트코인 뉴스 수집
            recent_news = fetch_bitcoin_news()
            
            # 과거 거래 내역 및 AI 분석 결과 가져오기
            historical_trading_data = get_historical_trading_data(limit=10)  # 최근 10개 거래
            
            # 전체 거래 성과 메트릭스 계산
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
            
            # 각 타임프레임 데이터를 dict로 변환하여 저장
            for tf_name, df in multi_tf_data.items():
                market_analysis["timeframes"][tf_name] = df.to_dict(orient="records")
            
            # ===== 6. AI 트레이딩 결정 요청 =====
            # AI 분석을 위한 시스템 프롬프트 설정
            system_prompt = """
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
            
            # OpenAI API 호출하여 트레이딩 결정 요청
            response = client.chat.completions.create(
                model="gpt-4o",  # GPT-4o 모델 사용
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(market_analysis)}
                ]
            )

            # ===== 7. AI 응답 처리 및 거래 실행 =====
            try:
                # API 응답에서 내용 추출
                response_content = response.choices[0].message.content.strip()
                print(f"Raw AI response: {response_content}")  # 디버깅용 출력
                
                # JSON 형식 정리 (코드 블록 제거)
                if response_content.startswith("```"):
                    # 첫 번째 줄바꿈 이후부터 마지막 ``` 이전까지의 내용만 추출
                    content_parts = response_content.split("\n", 1)
                    if len(content_parts) > 1:
                        response_content = content_parts[1]
                    # 마지막 ``` 제거
                    if "```" in response_content:
                        response_content = response_content.rsplit("```", 1)[0]
                    response_content = response_content.strip()
                
                # JSON 파싱
                trading_decision = json.loads(response_content)
                
                # 결정 내용 출력
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
                
                # AI 추천 방향 가져오기
                action = trading_decision['direction'].lower()
                
                # ===== 8. 트레이딩 결정에 따른 액션 실행 =====
                # 포지션을 열지 말아야 하는 경우
                if action == "no_position":
                    print("현재 시장 상황에서는 포지션을 열지 않는 것이 좋습니다.")
                    print(f"이유: {trading_decision['reasoning']}")
                    time.sleep(60)  # 포지션 없을 때 1분 대기
                    continue
                    
                # ===== 9. 투자 금액 계산 =====
                # 현재 잔액 확인
                balance = exchange.fetch_balance()
                available_capital = balance['USDT']['free']  # 가용 USDT 잔액
                
                # AI 추천 포지션 크기 비율 적용
                position_size_percentage = trading_decision['recommended_position_size']
                investment_amount = available_capital * position_size_percentage
                
                # 최소 주문 금액 확인 (최소 100 USDT)
                if investment_amount < 100:
                    investment_amount = 100
                    print(f"최소 주문 금액(100 USDT)으로 조정됨")
                
                print(f"투자 금액: {investment_amount:.2f} USDT")
                
                # ===== 10. 주문 수량 계산 =====
                # BTC 수량 = 투자금액 / 현재가격, 소수점 3자리까지 반올림
                amount = math.ceil((investment_amount / current_price) * 1000) / 1000
                print(f"주문 수량: {amount} BTC")

                # ===== 11. 레버리지 설정 =====
                # AI 추천 레버리지 설정
                recommended_leverage = trading_decision['recommended_leverage']
                exchange.set_leverage(recommended_leverage, symbol)
                print(f"레버리지 설정: {recommended_leverage}x")

                # ===== 12. 스탑로스/테이크프로핏 설정 =====
                # AI 추천 SL/TP 비율 가져오기
                sl_percentage = trading_decision['stop_loss_percentage']
                tp_percentage = trading_decision['take_profit_percentage']

                # ===== 13. 포지션 진입 및 SL/TP 주문 실행 =====
                if action == "long":  # 롱 포지션
                    # 시장가 매수 주문
                    order = exchange.create_market_buy_order(symbol, amount)
                    entry_price = current_price
                    
                    # 스탑로스/테이크프로핏 가격 계산
                    sl_price = round(entry_price * (1 - sl_percentage), 2)   # AI 추천 비율만큼 하락
                    tp_price = round(entry_price * (1 + tp_percentage), 2)   # AI 추천 비율만큼 상승
                    
                    # SL/TP 주문 생성
                    exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price})
                    
                    # 거래 데이터 저장
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
                    
                    # AI 분석 결과와 거래 연결
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

                elif action == "short":  # 숏 포지션
                    # 시장가 매도 주문
                    order = exchange.create_market_sell_order(symbol, amount)
                    entry_price = current_price
                    
                    # 스탑로스/테이크프로핏 가격 계산
                    sl_price = round(entry_price * (1 + sl_percentage), 2)   # AI 추천 비율만큼 상승
                    tp_price = round(entry_price * (1 - tp_percentage), 2)   # AI 추천 비율만큼 하락
                    
                    # SL/TP 주문 생성
                    exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price})
                    
                    # 거래 데이터 저장
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
                    
                    # AI 분석 결과와 거래 연결
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
                time.sleep(30)  # 대기 후 다시 시도
                continue
            except Exception as e:
                print(f"기타 오류: {e}")
                time.sleep(10)
                continue

        # ===== 14. 일정 시간 대기 후 다음 루프 실행 =====
        time.sleep(60)  # 메인 루프는 1분마다 실행

    except Exception as e:
        print(f"\n Error: {e}")
        time.sleep(5)