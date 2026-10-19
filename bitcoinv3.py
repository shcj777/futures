
bitcoinv3.PY
"""
비트코인 선물 트레이딩 봇 v3  (규칙 기반 엔진 + 저비용 AI 필터)
=====================================================================
v2 대비 변경점
  [빈도]  진입 타임프레임 5분봉(기본) / 신호 4종(눌림목·돌파·평균회귀·모멘텀) / 기준 완화
          + 장시간 무거래 시 점수 하한을 최대 6점까지만 단계적으로 완화(품질 바닥선 유지)
          + --backtest N --sweep 으로 '하루 거래수 vs 수수료 후 기대값'을 직접 측정
  [비용]  LLM은 '경계선 점수(MIN~LLM_SKIP_SCORE)' 신호에서만 호출, 하루 호출 상한,
          같은 전략/방향 30분 캐시, 캔들 원본 미전송(요약 지표만), 뉴스 60분 캐시
          -> 예상 비용: 하루 약 0.001~0.02 USD (v1: 약 1 USD)
  [안정]  진입가 재조회, 청산 손익 집계 지연 대기, PM 계정 자본 조회, 시간동기화,
          스탑 이동 실패 시 안전 폴백
 
실행
  python bitcoinv3.py --backtest 90 --sweep   # 점수 하한별 하루 거래수/수익성 비교 (먼저!)
  python bitcoinv3.py --backtest 90           # 현재 설정으로 백테스트
  python bitcoinv3.py --dry                   # 신호만 기록, 주문 X
  python bitcoinv3.py                         # 실거래
 
주의: 하루 5건은 '목표'일 뿐 수익 보장이 아닙니다. 거래를 억지로 늘리면 수수료(왕복 약 0.1%)가
기대값을 깎습니다. 반드시 --sweep 결과에서 수수료 후 profit_factor 가 1.2 이상인 설정만 쓰세요.
"""
import os
import math
import time
import json
import sqlite3
import argparse
import ccxt
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
 
load_dotenv()
 
# ===================== 설정 =====================
SYMBOL = "BTC/USDT:USDT"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.getenv("DB_FILE", os.path.join(BASE_DIR, "bitcoin_trading_v3.db"))
 
ENTRY_TF = os.getenv("ENTRY_TF", "5m")          # '5m'(거래 많음) 또는 '15m'(거래 적음)
TF_MIN = {"5m": 5, "15m": 15}[ENTRY_TF]
 
TARGET_TRADES_PER_DAY = 5    # 목표(참고용). 강제 체결 아님
RISK_PER_TRADE = 0.005       # 거래당 최대 손실 = 자본의 0.5%
MAX_LEVERAGE = 5
MAX_MARGIN_USAGE = 0.30
DAILY_LOSS_LIMIT = 0.02      # 하루 -2% 도달 시 당일 중단
MAX_TRADES_PER_DAY = 15
MAX_CONSEC_LOSSES = 3
COOLDOWN_MIN = 45
 
MIN_SCORE = 55               # v2: 62 -> 55 (거래 증가)
RELAX_FLOOR = 6              # 무거래가 길어질 때 완화 가능한 최대 점수 (하한 49)
MIN_RR = 1.4
MIN_TP_PCT = 0.003           # TP 최소 0.3% (왕복 수수료+슬리피지 ~0.12% 대비 2.5배)
FEE_RATE = 0.0005
SLIPPAGE = 0.0001
MAX_HOLD_MIN = 180
MIN_NOTIONAL = 100
BE_BUFFER = 0.0012
MAX_NOTIONAL_USD = float(os.getenv("MAX_NOTIONAL_USD", "200"))  # 0 이면 해제 (검증 후 상향)
 
# ---- LLM 비용 통제 ----
USE_LLM = os.getenv("USE_LLM", "1") == "1"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_SKIP_SCORE = 70          # 이 점수 이상은 LLM 없이 바로 진입 (비용 0)
LLM_MAX_CALLS_PER_DAY = 25   # 하루 호출 상한
LLM_CACHE_SEC = 1800
NEWS_TTL_SEC = 3600
 
# ===================== 거래소 =====================
exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY"),
    'secret': os.getenv("BINANCE_SECRET_KEY"),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'portfolioMargin': True,
        'fetchCurrencies': False,
        'adjustForTimeDifference': True,
        'recvWindow': 10000,
    },
})
serp_api_key = os.getenv("SERP_API_KEY")
 
 
class ProtectionError(Exception):
    pass
 
 
# ===================== DB =====================
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn
 
 
def setup_database():
    conn = db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL, action TEXT NOT NULL,
        entry_price REAL NOT NULL, amount REAL NOT NULL, leverage INTEGER NOT NULL,
        sl_price REAL NOT NULL, tp_price REAL NOT NULL,
        sl_percentage REAL NOT NULL, tp_percentage REAL NOT NULL,
        position_size_percentage REAL NOT NULL, investment_amount REAL NOT NULL,
        status TEXT DEFAULT 'OPEN', exit_price REAL, exit_timestamp TEXT,
        profit_loss REAL, profit_loss_percentage REAL)''')
    for col, typ in [("strategy", "TEXT"), ("score", "REAL"), ("sl_order_id", "TEXT"),
                     ("tp_order_id", "TEXT"), ("risk_amount", "REAL"), ("fees", "REAL"),
                     ("r_multiple", "REAL")]:
        try:
            c.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    c.execute('''CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
        strategy TEXT, side TEXT, score REAL, price REAL, taken INTEGER DEFAULT 0, note TEXT)''')
    c.execute("CREATE TABLE IF NOT EXISTS llm_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL)")
    conn.commit()
    conn.close()
    print("데이터베이스 설정 완료:", DB_FILE)
 
 
def save_signal(sig, taken, note=""):
    conn = db()
    conn.execute("INSERT INTO signals (timestamp,strategy,side,score,price,taken,note) VALUES (?,?,?,?,?,?,?)",
                 (datetime.now().isoformat(), sig['strategy'], sig['side'], sig['score'],
                  sig['entry'], int(taken), note[:500]))
    conn.commit()
    conn.close()
 
 
def save_trade(t):
    conn = db()
    cur = conn.execute('''INSERT INTO trades (timestamp,action,entry_price,amount,leverage,sl_price,tp_price,
        sl_percentage,tp_percentage,position_size_percentage,investment_amount,strategy,score,
        sl_order_id,tp_order_id,risk_amount) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                       (datetime.now().isoformat(), t['action'], t['entry_price'], t['amount'], t['leverage'],
                        t['sl_price'], t['tp_price'], t['sl_percentage'], t['tp_percentage'],
                        t.get('position_size_percentage', 0), t.get('investment_amount', 0),
                        t.get('strategy', 'adopted'), t.get('score', 0),
                        t.get('sl_order_id'), t.get('tp_order_id'), t.get('risk_amount', 0)))
    tid = cur.lastrowid
    conn.commit()
    conn.close()
    return tid
 
 
def db_update(trade_id, fields):
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn = db()
    conn.execute(f"UPDATE trades SET {sets} WHERE id = ?", list(fields.values()) + [trade_id])
    conn.commit()
    conn.close()
 
 
def get_latest_open_trade():
    conn = db()
    r = conn.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY timestamp DESC LIMIT 1").fetchone()
    conn.close()
    return dict(r) if r else None
 
 
def risk_gate(equity):
    today = datetime.now().strftime('%Y-%m-%d')
    conn = db()
    row = conn.execute("SELECT COALESCE(SUM(profit_loss),0) s FROM trades WHERE status='CLOSED' "
                       "AND substr(exit_timestamp,1,10)=?", (today,)).fetchone()
    n_today = conn.execute("SELECT COUNT(*) n FROM trades WHERE substr(timestamp,1,10)=?", (today,)).fetchone()['n']
    last = conn.execute("SELECT profit_loss, exit_timestamp FROM trades WHERE status='CLOSED' "
                        "ORDER BY exit_timestamp DESC LIMIT ?", (MAX_CONSEC_LOSSES,)).fetchall()
    conn.close()
    day_pnl = row['s']
    base = max(equity - day_pnl, 1e-9)
    if day_pnl / base <= -DAILY_LOSS_LIMIT:
        return False, f"일일 손실 한도 도달 ({day_pnl / base * 100:.2f}%) - 오늘은 거래 중단"
    if n_today >= MAX_TRADES_PER_DAY:
        return False, f"일일 최대 거래수({MAX_TRADES_PER_DAY}) 도달"
    if len(last) == MAX_CONSEC_LOSSES and all((x['profit_loss'] or 0) < 0 for x in last):
        until = datetime.fromisoformat(last[0]['exit_timestamp']) + timedelta(minutes=COOLDOWN_MIN)
        if datetime.now() < until:
            return False, f"{MAX_CONSEC_LOSSES}연패 쿨다운 ~{until.strftime('%H:%M')}"
    return True, ""
 
 
def strategy_stats(last_n=20):
    conn = db()
    rows = conn.execute("SELECT strategy, r_multiple, profit_loss FROM trades WHERE status='CLOSED' "
                        "AND strategy IS NOT NULL ORDER BY exit_timestamp DESC LIMIT 200").fetchall()
    conn.close()
    stats = {}
    for r in rows:
        s = stats.setdefault(r['strategy'], [])
        if len(s) < last_n and r['r_multiple'] is not None:
            s.append((r['r_multiple'], (r['profit_loss'] or 0) > 0))
    out = {}
    for k, v in stats.items():
        if v:
            out[k] = {"n": len(v), "avg_R": round(sum(x[0] for x in v) / len(v), 2),
                      "win_rate": round(sum(1 for x in v if x[1]) / len(v) * 100, 1)}
    return out
 
 
def print_summary(days=7):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = db()
    r = conn.execute("""SELECT COUNT(*) n, SUM(CASE WHEN profit_loss>0 THEN 1 ELSE 0 END) w,
        COALESCE(SUM(profit_loss),0) pnl, COALESCE(AVG(r_multiple),0) r FROM trades
        WHERE status='CLOSED' AND exit_timestamp>=?""", (cutoff,)).fetchone()
    conn.close()
    if r['n']:
        print(f"[{days}일 요약] 거래 {r['n']}건 (하루 평균 {r['n'] / days:.1f}건) | 승률 {r['w'] / r['n'] * 100:.1f}% | "
              f"손익 {r['pnl']:+.2f} USDT | 평균 {r['r']:+.2f}R")
 
 
def hours_since_last_entry():
    conn = db()
    r = conn.execute("SELECT MAX(timestamp) t FROM trades WHERE strategy != 'adopted'").fetchone()
    conn.close()
    if not r or not r['t']:
        return 999.0
    return (datetime.now() - datetime.fromisoformat(r['t'])).total_seconds() / 3600
 
 
def llm_calls_today():
    today = datetime.now().strftime('%Y-%m-%d')
    conn = db()
    n = conn.execute("SELECT COUNT(*) n FROM llm_calls WHERE substr(ts,1,10)=?", (today,)).fetchone()['n']
    conn.close()
    return n
 
 
# ===================== 지표 =====================
def add_indicators(df):
    d = df.copy()
    c, h, l, v = d['close'], d['high'], d['low'], d['volume']
    for n in (20, 50, 200):
        d[f'ema{n}'] = c.ewm(span=n, adjust=False).mean()
 
    delta = c.diff()
    up, dn = delta.clip(lower=0), (-delta.clip(upper=0))
    rs = up.ewm(alpha=1 / 14, adjust=False).mean() / dn.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    d['rsi'] = (100 - 100 / (1 + rs)).fillna(50)
 
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d['atr'] = atr
 
    um, dm = h.diff(), -l.diff()
    plus = pd.Series(np.where((um > dm) & (um > 0), um, 0.0), index=d.index)
    minus = pd.Series(np.where((dm > um) & (dm > 0), dm, 0.0), index=d.index)
    d['pdi'] = 100 * plus.ewm(alpha=1 / 14, adjust=False).mean() / atr
    d['mdi'] = 100 * minus.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (d['pdi'] - d['mdi']).abs() / (d['pdi'] + d['mdi']).replace(0, np.nan)
    d['adx'] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
 
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d['macd_h'] = macd - macd.ewm(span=9, adjust=False).mean()
 
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d['bb_mid'], d['bb_up'], d['bb_lo'] = mid, mid + 2 * sd, mid - 2 * sd
    bbw = (d['bb_up'] - d['bb_lo']) / mid
    d['bb_squeeze'] = bbw <= bbw.rolling(100).quantile(0.25)
 
    d['don_hi'] = h.rolling(20).max().shift(1)
    d['don_lo'] = l.rolling(20).min().shift(1)
    d['vol_z'] = ((v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)).fillna(0)
    return d
 
 
def bias(row, fast, slow):
    if row['close'] > row[slow] and row[fast] > row[slow]:
        return 1
    if row['close'] < row[slow] and row[fast] < row[slow]:
        return -1
    return 0
 
 
# ===================== 신호 엔진 =====================
def _mk(strategy, side, score, price, sl_dist, atr, rr=None, tp_dist=None, min_rr=MIN_RR,
        funding=None, reasons=None):
    sl_dist = min(max(sl_dist, 0.8 * atr), 3.0 * atr)
    if tp_dist is None:
        tp_dist = rr * sl_dist
    real_rr = tp_dist / sl_dist
    if real_rr < min_rr or tp_dist / price < MIN_TP_PCT:
        return None
    if funding is not None:
        if side == 'long' and funding > 0.0005:
            score -= 6
        if side == 'short' and funding < -0.0005:
            score -= 6
    return {"strategy": strategy, "side": side, "score": int(min(score, 95)), "entry": price,
            "sl_dist": sl_dist, "tp_dist": tp_dist, "rr": round(real_rr, 2), "reasons": reasons or []}
 
 
def evaluate(d, h1, h4, funding=None):
    """d: 진입TF(지표 포함, 마지막=직전 마감봉), h1/h4: 상위 TF 마지막 마감봉(Series)"""
    if len(d) < 12:
        return []
    c, p = d.iloc[-1], d.iloc[-2]
    if pd.isna(c['atr']) or pd.isna(c['bb_lo']) or pd.isna(c['don_hi']) or pd.isna(h4['ema200']):
        return []
    price, atr = c['close'], c['atr']
    b1, b4 = bias(h1, 'ema20', 'ema50'), bias(h4, 'ema50', 'ema200')
    adx1 = h1['adx']
    rec5, rec8 = d.iloc[-5:], d.iloc[-8:]
    sigs = []
 
    for side, sgn in (('long', 1), ('short', -1)):
        # ---- A. 추세 눌림목 ----
        if b1 == sgn and b4 in (0, sgn):
            if side == 'long':
                touched = (rec5['low'] <= rec5['ema20'] * 1.0005).any()
                ok = (touched and c['close'] > c['ema20'] and c['close'] > c['open'] and c['close'] > c['ema50']
                      and 40 <= c['rsi'] <= 68 and c['macd_h'] > p['macd_h'])
                sl_dist = price - (rec8['low'].min() - 0.2 * atr)
            else:
                touched = (rec5['high'] >= rec5['ema20'] * 0.9995).any()
                ok = (touched and c['close'] < c['ema20'] and c['close'] < c['open'] and c['close'] < c['ema50']
                      and 32 <= c['rsi'] <= 60 and c['macd_h'] < p['macd_h'])
                sl_dist = (rec8['high'].max() + 0.2 * atr) - price
            if ok:
                score = (58 + (10 if b4 == sgn else 0) + (8 if adx1 > 25 else 0) + (6 if c['vol_z'] > 0.3 else 0)
                         + (5 if c['adx'] > 20 else 0))
                s = _mk("pullback", side, score, price, max(sl_dist, 1.2 * atr), atr, rr=2.0, funding=funding,
                        reasons=[f"1h/4h 추세 {b1}/{b4}", "EMA20 눌림 후 재상승" if side == 'long' else "EMA20 반등 후 재하락"])
                if s:
                    sigs.append(s)
 
        # ---- B. 돌파 ----
        if b1 in (0, sgn) and b4 in (0, sgn) and c['vol_z'] > 0.6 and c['adx'] > 16:
            if side == 'long':
                brk = c['close'] > c['don_hi'] and c['close'] > c['open'] and (c['close'] - c['ema20']) < 2.5 * atr
            else:
                brk = c['close'] < c['don_lo'] and c['close'] < c['open'] and (c['ema20'] - c['close']) < 2.5 * atr
            if brk:
                score = (56 + (10 if d['bb_squeeze'].iloc[-10:].any() else 0) + (8 if (b1 == sgn and b4 == sgn) else 0)
                         + (6 if c['vol_z'] > 2 else 0) + (5 if c['adx'] > p['adx'] else 0))
                s = _mk("breakout", side, score, price, 1.5 * atr, atr, rr=2.0, funding=funding,
                        reasons=["20봉 돌파", f"거래량 z={c['vol_z']:.1f}"])
                if s:
                    sigs.append(s)
 
        # ---- C. 횡보 평균회귀 ----
        if adx1 < 24 and (b4 in (0, sgn) or adx1 < 18):
            if side == 'long':
                ok = (p['low'] < p['bb_lo'] and c['close'] > c['bb_lo'] and c['close'] > c['open']
                      and p['rsi'] < 38 and c['rsi'] > p['rsi'])
                sl_dist = price - (min(p['low'], c['low']) - 0.3 * atr)
                tp_dist = c['bb_mid'] - price
                extreme = p['rsi'] < 28
            else:
                ok = (p['high'] > p['bb_up'] and c['close'] < c['bb_up'] and c['close'] < c['open']
                      and p['rsi'] > 62 and c['rsi'] < p['rsi'])
                sl_dist = (max(p['high'], c['high']) + 0.3 * atr) - price
                tp_dist = price - c['bb_mid']
                extreme = p['rsi'] > 72
            if ok and tp_dist > 0:
                score = 55 + (8 if extreme else 0) + (8 if c['vol_z'] > 0.5 else 0) + (6 if adx1 < 18 else 0) \
                        + (5 if b4 == sgn else 0)
                s = _mk("mean_reversion", side, score, price, max(sl_dist, 1.0 * atr), atr, tp_dist=tp_dist,
                        min_rr=1.2, funding=funding, reasons=["볼린저 밴드 이탈 후 복귀", f"1h ADX={adx1:.0f}(횡보)"])
                if s:
                    sigs.append(s)
 
        # ---- D. 모멘텀 지속 (신규: 추세 중 직전봉 고/저 돌파) ----
        if b1 == sgn and b4 in (0, sgn) and c['adx'] > 20:
            if side == 'long':
                ok = (c['close'] > c['ema20'] > c['ema50'] and c['macd_h'] > 0 and c['macd_h'] > p['macd_h']
                      and 50 <= c['rsi'] <= 70 and c['close'] > p['high'] and (c['close'] - c['ema20']) < 1.5 * atr)
                sl_dist = price - d.iloc[-4:]['low'].min()
            else:
                ok = (c['close'] < c['ema20'] < c['ema50'] and c['macd_h'] < 0 and c['macd_h'] < p['macd_h']
                      and 30 <= c['rsi'] <= 50 and c['close'] < p['low'] and (c['ema20'] - c['close']) < 1.5 * atr)
                sl_dist = d.iloc[-4:]['high'].max() - price
            if ok:
                score = 55 + (10 if b4 == sgn else 0) + (6 if adx1 > 25 else 0) + (5 if c['vol_z'] > 0.5 else 0) \
                        + (4 if c['adx'] > p['adx'] else 0)
                s = _mk("momentum", side, score, price, max(sl_dist, 1.2 * atr), atr, rr=1.8, funding=funding,
                        reasons=[f"추세 {b1}/{b4}", "직전봉 고/저 돌파 모멘텀"])
                if s:
                    sigs.append(s)
 
    sigs.sort(key=lambda x: x['score'], reverse=True)
    return sigs
 
 
# ===================== 포지션 사이징 =====================
def calc_position(equity, entry, sl_dist, size_mult=1.0, notional_cap=None):
    risk_amt = equity * RISK_PER_TRADE * size_mult
    notional = (risk_amt / sl_dist) * entry
    cap = equity * MAX_LEVERAGE * 0.9
    if notional_cap:
        cap = min(cap, notional_cap)
    notional = min(notional, cap)
    if notional < MIN_NOTIONAL:
        if MIN_NOTIONAL <= cap and MIN_NOTIONAL * sl_dist / entry <= risk_amt * 2:
            notional = MIN_NOTIONAL
        else:
            return None
    qty = math.floor(notional / entry * 1000) / 1000
    if qty * entry < MIN_NOTIONAL:
        qty = round(qty + 0.001, 3)
    if qty <= 0:
        return None
    notional = qty * entry
    lev = int(min(MAX_LEVERAGE, max(1, math.ceil(notional / (equity * MAX_MARGIN_USAGE)))))
    return {"qty": qty, "notional": notional, "leverage": lev, "risk_amt": qty * sl_dist}
 
 
# ===================== 거래소 헬퍼 =====================
def get_equity():
    """PM(포트폴리오 마진) 계정은 papi 계정 정보의 equity가 가장 정확함. 실패 시 잔고 조회로 폴백"""
    try:
        acc = exchange.papiGetAccount()
        v = float(acc.get('actualEquity') or acc.get('accountEquity') or 0)
        if v > 0:
            return v
    except Exception as e:
        print("papi 자본 조회 실패, 잔고로 대체:", e)
    bal = exchange.fetch_balance()
    for key in ('total', 'free'):
        part = bal.get(key, {}) or {}
        v = part.get('USDT') or part.get('USDC')
        if v and v > 0:
            return float(v)
    raise RuntimeError(f"잔고 조회 실패: {bal.get('free')}")
 
 
def get_position():
    for pos in exchange.fetch_positions([SYMBOL]):
        if pos['symbol'] == SYMBOL:
            amt = float(pos['info']['positionAmt'])
            if amt > 0:
                return 'long', amt, float(pos['info'].get('entryPrice') or 0)
            if amt < 0:
                return 'short', abs(amt), float(pos['info'].get('entryPrice') or 0)
    return None, 0.0, 0.0
 
 
def px(v):
    return float(exchange.price_to_precision(SYMBOL, v))
 
 
def cancel_all_orders():
    try:
        normal = exchange.fetch_open_orders(SYMBOL)
        cond = exchange.fetch_open_orders(SYMBOL, params={'stop': True})
        for o in normal:
            exchange.cancel_order(o['id'], SYMBOL)
        for o in cond:
            exchange.cancel_order(o['id'], SYMBOL, params={'stop': True})
    except Exception as e:
        print("주문 취소 오류:", e)
 
 
def place_protection(side, qty, entry, sl_dist, tp_dist):
    close_side = 'sell' if side == 'long' else 'buy'
    dirn = 1 if side == 'long' else -1
    sl, tp = px(entry - dirn * sl_dist), px(entry + dirn * tp_dist)
    o1 = exchange.create_order(SYMBOL, 'STOP_MARKET', close_side, qty, None, {'stopPrice': sl, 'reduceOnly': True})
    o2 = exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', close_side, qty, None,
                               {'stopPrice': tp, 'reduceOnly': True})
    return sl, tp, o1.get('id'), o2.get('id')
 
 
def close_market(side, qty):
    close_side = 'sell' if side == 'long' else 'buy'
    exchange.create_order(SYMBOL, 'market', close_side, qty, None, {'reduceOnly': True})
 
 
def open_with_protection(side, qty, entry_hint, sl_dist, tp_dist):
    """시장가 진입 -> 실제 평단 재조회 -> SL/TP 설정. 실패 시 즉시 청산 + ProtectionError."""
    order = exchange.create_order(SYMBOL, 'market', 'buy' if side == 'long' else 'sell', qty)
    entry = float(order.get('average') or 0)
    if entry <= 0:
        time.sleep(1)
        try:
            _, _, ep = get_position()
            entry = ep
        except Exception:
            entry = 0
    if entry <= 0:
        entry = float(order.get('price') or entry_hint)
    try:
        sl, tp, sl_id, tp_id = place_protection(side, qty, entry, sl_dist, tp_dist)
    except Exception as e:
        print("SL/TP 주문 실패, 포지션 즉시 청산:", e)
        try:
            close_market(side, qty)
        except Exception as e2:
            print("!!! 청산도 실패. 바이낸스 앱에서 직접 확인하세요:", e2)
        raise ProtectionError("보호 주문 실패로 진입 취소")
    return entry, sl, tp, sl_id, tp_id
 
 
def get_funding():
    try:
        return float(exchange.fetch_funding_rate(SYMBOL)['fundingRate'])
    except Exception:
        return None
 
 
def load_features():
    def get(tf, limit):
        o = exchange.fetch_ohlcv(SYMBOL, timeframe=tf, limit=limit)
        df = pd.DataFrame(o, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.iloc[:-1].reset_index(drop=True)
 
    d = add_indicators(get(ENTRY_TF, 300))
    h1 = add_indicators(get('1h', 300))
    h4 = add_indicators(get('4h', 300))
    return d, h1.iloc[-1], h4.iloc[-1]
 
 
# ===================== AI 필터 (저비용) =====================
_client = None
_news_cache = (0, [])
_llm_cache = {}
 
 
def fetch_news():
    global _news_cache
    if time.time() - _news_cache[0] < NEWS_TTL_SEC:
        return _news_cache[1]
    items = []
    try:
        r = requests.get("https://serpapi.com/search.json", timeout=10, params={
            "engine": "google_news", "q": "bitcoin", "gl": "us", "hl": "en", "api_key": serp_api_key})
        if r.status_code == 200:
            items = [n.get("title", "")[:100] for n in r.json().get("news_results", [])[:5]]
    except Exception as e:
        print("뉴스 수집 오류:", e)
    _news_cache = (time.time(), items)
    return items
 
 
def llm_review(sig, d, h1, h4):
    """승인/거부 + 사이즈 배수. 실패 시 None. 입력은 요약 지표만(약 400토큰)."""
    global _client
    try:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(timeout=20)
        c = d.iloc[-1]
        payload = {
            "signal": {k: sig[k] for k in ("strategy", "side", "score", "rr", "reasons")},
            "price": round(float(c['close']), 1),
            f"{ENTRY_TF}": {k: round(float(c[k]), 2) for k in ("rsi", "adx", "vol_z")},
            "1h": {"rsi": round(float(h1['rsi']), 1), "adx": round(float(h1['adx']), 1),
                   "trend": bias(h1, 'ema20', 'ema50')},
            "4h": {"rsi": round(float(h4['rsi']), 1), "trend": bias(h4, 'ema50', 'ema200')},
            "news": fetch_news(),
            "strategy_recent_perf": strategy_stats().get(sig['strategy']),
        }
        sys_prompt = (
            "Risk reviewer for a BTC futures bot. A rule engine found a setup. DEFAULT APPROVE. "
            "VETO only for concrete evidence: major news opposing the side, clear higher-timeframe conflict, "
            "or this strategy's recent performance clearly poor. size_multiplier 0.5-1.0. "
            'Reply ONLY JSON: {"verdict":"APPROVE"|"VETO","size_multiplier":0.5-1.0,"reason":"<=120 chars"}')
        r = _client.chat.completions.create(
            model=LLM_MODEL, max_tokens=100, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        conn = db()
        conn.execute("INSERT INTO llm_calls (ts) VALUES (?)", (datetime.now().isoformat(),))
        conn.commit()
        conn.close()
        out = json.loads(r.choices[0].message.content)
        out["size_multiplier"] = min(1.0, max(0.5, float(out.get("size_multiplier", 1.0))))
        out["verdict"] = "VETO" if str(out.get("verdict", "")).upper() == "VETO" else "APPROVE"
        return out
    except Exception as e:
        print("LLM 검토 실패:", e)
        return None
 
 
def ai_decision(sig, d, h1, h4):
    """(진입여부, 사이즈배수, 메모). 호출 최소화 규칙 적용"""
    if not USE_LLM or sig['score'] >= LLM_SKIP_SCORE:
        return True, 1.0, "LLM 생략(고점수/비활성)"
    key = (sig['strategy'], sig['side'])
    cached = _llm_cache.get(key)
    if cached and time.time() - cached[0] < LLM_CACHE_SEC:
        v = cached[1]
        return v['verdict'] == 'APPROVE', v['size_multiplier'], "캐시: " + str(v.get('reason', ''))
    if llm_calls_today() >= LLM_MAX_CALLS_PER_DAY:
        return True, 0.7, "LLM 일일 상한 도달 -> 0.7배 사이즈"
    rv = llm_review(sig, d, h1, h4)
    if rv is None:
        return True, 0.5, "LLM 실패 -> 절반 사이즈"
    _llm_cache[key] = (time.time(), rv)
    return rv['verdict'] == 'APPROVE', rv['size_multiplier'], str(rv.get('reason', ''))
 
 
# ===================== 포지션 관리 =====================
def finalize_closed_trade(trade):
    time.sleep(4)  # 청산 체결 내역 반영 대기
    entry, amt = trade['entry_price'], trade['amount']
    dirn = 1 if trade['action'] == 'long' else -1
    since = int(datetime.fromisoformat(trade['timestamp']).timestamp() * 1000) - 5000
    pnl, exit_price, fees = None, None, 0.0
    try:
        ts = exchange.fetch_my_trades(SYMBOL, since=since)
        if ts and any('realizedPnl' in (t.get('info') or {}) for t in ts):
            pnl = sum(float(t['info'].get('realizedPnl') or 0) for t in ts)
            fees = sum(float((t.get('fee') or {}).get('cost') or 0) for t in ts)
            pnl -= fees
            closing = [t for t in ts if t['side'] == ('sell' if dirn == 1 else 'buy')]
            if closing:
                exit_price = sum(t['price'] * t['amount'] for t in closing) / sum(t['amount'] for t in closing)
    except Exception as e:
        print("체결내역 조회 실패(추정치 사용):", e)
    if exit_price is None:
        exit_price = exchange.fetch_ticker(SYMBOL)['last']
    if pnl is None:
        fees = (entry + exit_price) * amt * FEE_RATE
        pnl = dirn * (exit_price - entry) * amt - fees
    risk = trade.get('risk_amount') or 0
    r_mult = pnl / risk if risk > 0 else None
    db_update(trade['id'], {
        'status': 'CLOSED', 'exit_price': exit_price, 'exit_timestamp': datetime.now().isoformat(),
        'profit_loss': pnl, 'profit_loss_percentage': dirn * (exit_price / entry - 1) * 100,
        'fees': fees, 'r_multiple': r_mult})
    print(f"\n=== 포지션 종료 [{trade.get('strategy')}] ===")
    print(f"진입 {entry:,.1f} -> 청산 {exit_price:,.1f} | 손익 {pnl:+.2f} USDT"
          + (f" ({r_mult:+.2f}R)" if r_mult is not None else ""))
    print_summary(7)
 
 
def replace_stop(trade, side, amt, new_sl):
    """1차: 새 스탑 먼저 생성 후 기존 취소. 거래소가 거부하면 2차: 기존 취소 후 생성(실패 시 원복/시장가 청산)"""
    close_side = 'sell' if side == 'long' else 'buy'
    new_sl = px(new_sl)
    old_id, old_sl = trade.get('sl_order_id'), trade['sl_price']
    new_order = None
    try:
        new_order = exchange.create_order(SYMBOL, 'STOP_MARKET', close_side, amt, None,
                                          {'stopPrice': new_sl, 'reduceOnly': True})
        if old_id:
            try:
                exchange.cancel_order(old_id, SYMBOL, params={'stop': True})
            except Exception as e:
                print("기존 스탑 취소 오류(무시):", e)
    except Exception as e1:
        print("스탑 선생성 실패, 취소 후 재생성 시도:", e1)
        try:
            if old_id:
                exchange.cancel_order(old_id, SYMBOL, params={'stop': True})
            new_order = exchange.create_order(SYMBOL, 'STOP_MARKET', close_side, amt, None,
                                              {'stopPrice': new_sl, 'reduceOnly': True})
        except Exception as e2:
            print("스탑 이동 최종 실패, 기존 스탑 원복 시도:", e2)
            try:
                new_order = exchange.create_order(SYMBOL, 'STOP_MARKET', close_side, amt, None,
                                                  {'stopPrice': px(old_sl), 'reduceOnly': True})
                new_sl = px(old_sl)
            except Exception as e3:
                print("!!! 스탑 원복도 실패 -> 보호 공백 방지를 위해 시장가 청산:", e3)
                try:
                    close_market(side, amt)
                except Exception as e4:
                    print("!!! 청산 실패. 바이낸스 앱에서 직접 확인하세요:", e4)
                return
    if new_order:
        db_update(trade['id'], {'sl_price': new_sl, 'sl_order_id': new_order.get('id')})
        print(f"스탑 이동 -> {new_sl:,.1f}")
 
 
def manage_position(trade, side, amt):
    price = exchange.fetch_ticker(SYMBOL)['last']
    entry = trade['entry_price']
    r_dist = entry * (trade['sl_percentage'] or 0)
    if r_dist <= 0:
        return
    dirn = 1 if side == 'long' else -1
    r_now = dirn * (price - entry) / r_dist
    age_min = (datetime.now() - datetime.fromisoformat(trade['timestamp'])).total_seconds() / 60
 
    if age_min >= MAX_HOLD_MIN and r_now < 0.5:
        print(f"시간 청산 ({age_min:.0f}분 경과, {r_now:+.2f}R)")
        close_market(side, amt)
        cancel_all_orders()
        return
 
    new_sl = None
    if r_now >= 1.0:
        new_sl = entry + dirn * entry * BE_BUFFER
    if r_now >= 1.5:
        trail = price - dirn * 0.8 * r_dist
        if new_sl is None or (trail - new_sl) * dirn > 0:
            new_sl = trail
    if new_sl is not None and (new_sl - trade['sl_price']) * dirn > 0.1 * r_dist:
        replace_stop(trade, side, amt, new_sl)
 
 
def adopt_orphan(side, amt, entry):
    d, _, _ = load_features()
    atr = float(d.iloc[-1]['atr'])
    sl_dist = max(1.5 * atr, entry * 0.004)
    tp_dist = 2 * sl_dist
    cond = []
    try:
        cond = exchange.fetch_open_orders(SYMBOL, params={'stop': True})
    except Exception:
        pass
    sl, tp, sl_id, tp_id = (None,) * 4
    if not cond:
        sl, tp, sl_id, tp_id = place_protection(side, amt, entry, sl_dist, tp_dist)
        print("보호주문이 없는 기존 포지션 -> SL/TP 설정")
    dirn = 1 if side == 'long' else -1
    save_trade({'action': side, 'entry_price': entry, 'amount': amt, 'leverage': 1,
                'sl_price': sl or entry - dirn * sl_dist, 'tp_price': tp or entry + dirn * tp_dist,
                'sl_percentage': sl_dist / entry, 'tp_percentage': tp_dist / entry,
                'strategy': 'adopted', 'sl_order_id': sl_id, 'tp_order_id': tp_id,
                'risk_amount': amt * sl_dist})
    print("기존 포지션을 DB에 편입")
 
 
# ===================== 신호 사이클 =====================
def effective_min_score():
    """무거래가 길어지면 최대 RELAX_FLOOR점까지만 단계 완화 (4시간마다 3점씩)"""
    h = hours_since_last_entry()
    relax = 0
    if h >= 4:
        relax = 3
    if h >= 8:
        relax = RELAX_FLOOR
    return MIN_SCORE - relax, relax
 
 
def run_signal_cycle(dry=False):
    equity = get_equity()
    ok, reason = risk_gate(equity)
    if not ok:
        print("[게이트]", reason)
        return
    d, h1, h4 = load_features()
    sigs = evaluate(d, h1, h4, get_funding())
 
    stats = strategy_stats()
    for s in sigs:
        st = stats.get(s['strategy'])
        if st and st['n'] >= 8:
            s['score'] += int(max(-10, min(5, st['avg_R'] * 10)))
    min_score, relax = effective_min_score()
    sigs = sorted([s for s in sigs if s['score'] >= min_score], key=lambda x: x['score'], reverse=True)
 
    price = float(d.iloc[-1]['close'])
    if not sigs:
        print(f"[{datetime.now():%H:%M}] {price:,.1f} | 신호 없음 (하한 {min_score}{f', 완화 -{relax}' if relax else ''})")
        return
    sig = sigs[0]
    print(f"[{datetime.now():%H:%M}] 신호: {sig['strategy']} {sig['side'].upper()} 점수 {sig['score']} "
          f"RR {sig['rr']} | {', '.join(sig['reasons'])}")
 
    if dry:
        save_signal(sig, False, "dry-run")
        return
 
    go, size_mult, note = ai_decision(sig, d, h1, h4)
    if not go:
        print("AI 거부:", note)
        save_signal(sig, False, "VETO: " + note)
        return
    print(f"진입 승인 (x{size_mult:.2f}): {note}")
 
    pos = calc_position(equity, sig['entry'], sig['sl_dist'], size_mult, MAX_NOTIONAL_USD or None)
    if not pos:
        save_signal(sig, False, "사이징 불가(잔고/최소주문)")
        print("사이징 불가 - 건너뜀")
        return
 
    try:
        exchange.papiPostUmLeverage({'symbol': exchange.market(SYMBOL)['id'], 'leverage': pos['leverage']})
    except Exception as e:
        raise RuntimeError(f"레버리지 설정 실패, 진입 취소: {e}")
 
    entry, sl, tp, sl_id, tp_id = open_with_protection(
        sig['side'], pos['qty'], sig['entry'], sig['sl_dist'], sig['tp_dist'])
    tid = save_trade({
        'action': sig['side'], 'entry_price': entry, 'amount': pos['qty'], 'leverage': pos['leverage'],
        'sl_price': sl, 'tp_price': tp, 'sl_percentage': sig['sl_dist'] / entry,
        'tp_percentage': sig['tp_dist'] / entry, 'position_size_percentage': pos['notional'] / equity,
        'investment_amount': pos['notional'], 'strategy': sig['strategy'], 'score': sig['score'],
        'sl_order_id': sl_id, 'tp_order_id': tp_id, 'risk_amount': pos['risk_amt']})
    save_signal(sig, True, note)
    print(f"\n=== {sig['side'].upper()} 진입 #{tid} [{sig['strategy']}] ===")
    print(f"진입 {entry:,.1f} | SL {sl:,.1f} | TP {tp:,.1f} | RR {sig['rr']} | {pos['leverage']}x | "
          f"수량 {pos['qty']} BTC (명목 {pos['notional']:.0f} USDT, 위험 {pos['risk_amt']:.2f} USDT)")
 
 
# ===================== 메인 루프 =====================
def main_live(dry=False):
    print("\n=== Bitcoin Bot v3 시작 ===", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"모드: {'DRY-RUN(주문 없음)' if dry else '실거래'} | 진입TF {ENTRY_TF} | "
          f"AI필터: {LLM_MODEL if USE_LLM else 'OFF'}(경계 점수만, 일 {LLM_MAX_CALLS_PER_DAY}회 상한) | "
          f"위험/거래 {RISK_PER_TRADE * 100:.2f}% | 최대 {MAX_LEVERAGE}x | 최소점수 {MIN_SCORE}")
    setup_database()
    period = TF_MIN * 60
    last_bucket = None
    while True:
        try:
            side, amt, pos_entry = (None, 0, 0) if dry else get_position()
            trade = None if dry else get_latest_open_trade()
 
            if side:
                if not trade:
                    adopt_orphan(side, amt, pos_entry or exchange.fetch_ticker(SYMBOL)['last'])
                else:
                    manage_position(trade, side, amt)
                time.sleep(10)
                continue
 
            if trade:
                age = (datetime.now() - datetime.fromisoformat(trade['timestamp'])).total_seconds()
                if age < 20:
                    time.sleep(3)
                    continue
                finalize_closed_trade(trade)
                cancel_all_orders()
 
            now = time.time()
            bucket = int(now // period)
            if bucket == last_bucket or now % period < 5:   # 봉 마감 5초 후 1회만 평가
                time.sleep(3)
                continue
            last_bucket = bucket
            run_signal_cycle(dry)
            time.sleep(3)
        except KeyboardInterrupt:
            print("종료")
            break
        except ProtectionError as e:
            print(f"치명적: {e} - 봇 중지. 바이낸스 앱에서 포지션을 확인하세요.")
            raise SystemExit(1)
        except Exception as e:
            print(f"오류: {e}")
            time.sleep(10)
 
 
# ===================== 백테스트 =====================
def fetch_history(tf, since_ms):
    out, step = [], exchange.parse_timeframe(tf) * 1000
    while True:
        batch = exchange.fetch_ohlcv(SYMBOL, tf, since=since_ms, limit=1000)
        if not batch:
            break
        out += batch
        since_ms = batch[-1][0] + step
        if len(batch) < 1000:
            break
    df = pd.DataFrame(out, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.iloc[:-1]
 
 
def simulate(df, h1, h4, equity0=1000.0, warmup=120, verbose=True, min_score=None, tf_min=None):
    """진입TF 봉 마감 신호 -> 다음 봉 시가 진입. SL/TP 동시 터치 시 SL 우선(보수적)."""
    min_score = MIN_SCORE if min_score is None else min_score
    tf_min = TF_MIN if tf_min is None else tf_min
    ct = (df['timestamp'] + pd.Timedelta(minutes=tf_min)).values
    ct1 = (h1['timestamp'] + pd.Timedelta(hours=1)).values
    ct4 = (h4['timestamp'] + pd.Timedelta(hours=4)).values
    eq, peak, mdd = equity0, equity0, 0.0
    trades, pos, pending = [], None, None
    cur_day, day_start, day_n, halted, consec, cool_until = None, eq, 0, False, 0, -1
    hold_bars = MAX_HOLD_MIN // tf_min
    cool_bars = COOLDOWN_MIN // tf_min
 
    for i in range(warmup, len(df)):
        bar = df.iloc[i]
        day = bar['timestamp'].date()
        if day != cur_day:
            cur_day, day_start, day_n, halted = day, eq, 0, False
 
        if pending and pos is None:
            s = pending
            e = bar['open'] * (1 + (SLIPPAGE if s['side'] == 'long' else -SLIPPAGE))
            dirn = 1 if s['side'] == 'long' else -1
            q = calc_position(eq, e, s['sl_dist'], 1.0)
            if q:
                pos = {"side": s['side'], "dirn": dirn, "entry": e, "qty": q['qty'], "r": s['sl_dist'],
                       "sl": e - dirn * s['sl_dist'], "tp": e + dirn * s['tp_dist'], "open_i": i,
                       "ext": e, "risk": q['risk_amt'], "strategy": s['strategy']}
                day_n += 1
            pending = None
 
        if pos:
            dn = pos['dirn']
            hit_sl = bar['low'] <= pos['sl'] if dn == 1 else bar['high'] >= pos['sl']
            hit_tp = bar['high'] >= pos['tp'] if dn == 1 else bar['low'] <= pos['tp']
            exit_p = None
            if hit_sl:
                exit_p = pos['sl'] * (1 - dn * SLIPPAGE)
            elif hit_tp:
                exit_p = pos['tp']
            else:
                pos['ext'] = max(pos['ext'], bar['high']) if dn == 1 else min(pos['ext'], bar['low'])
                r_ext = dn * (pos['ext'] - pos['entry']) / pos['r']
                if r_ext >= 1.0:
                    pos['sl'] = max(pos['sl'] * dn, (pos['entry'] + dn * pos['entry'] * BE_BUFFER) * dn) * dn
                if r_ext >= 1.5:
                    pos['sl'] = max(pos['sl'] * dn, (pos['ext'] - dn * 0.8 * pos['r']) * dn) * dn
                r_now = dn * (bar['close'] - pos['entry']) / pos['r']
                if i - pos['open_i'] >= hold_bars and r_now < 0.5:
                    exit_p = bar['close']
            if exit_p is not None:
                fees = (pos['entry'] + exit_p) * pos['qty'] * FEE_RATE
                pnl = dn * (exit_p - pos['entry']) * pos['qty'] - fees
                eq += pnl
                trades.append({"strategy": pos['strategy'], "pnl": pnl, "R": pnl / pos['risk'],
                               "bars": i - pos['open_i'], "day": day})
                consec = consec + 1 if pnl < 0 else 0
                if consec >= MAX_CONSEC_LOSSES:
                    cool_until, consec = i + cool_bars, 0
                pos = None
                peak = max(peak, eq)
                mdd = max(mdd, (peak - eq) / peak)
                if eq - day_start <= -DAILY_LOSS_LIMIT * day_start:
                    halted = True
 
        if pos is None and not halted and day_n < MAX_TRADES_PER_DAY and i >= cool_until and i < len(df) - 1:
            j1 = np.searchsorted(ct1, ct[i], side='right') - 1
            j4 = np.searchsorted(ct4, ct[i], side='right') - 1
            if j1 >= 0 and j4 >= 0:
                sigs = [s for s in evaluate(df.iloc[i - 29:i + 1], h1.iloc[j1], h4.iloc[j4])
                        if s['score'] >= min_score]
                if sigs:
                    pending = sigs[0]
 
    n = len(trades)
    span = max(1, (df['timestamp'].iloc[-1] - df['timestamp'].iloc[warmup]).days)
    res = {"min_score": min_score, "trades": n, "trades_per_day": n / span, "final_equity": eq,
           "return_pct": (eq / equity0 - 1) * 100, "max_dd_pct": mdd * 100,
           "win_rate": 0.0, "avg_R": 0.0, "profit_factor": 0.0}
    if n:
        wins = [t for t in trades if t['pnl'] > 0]
        gp, gl = sum(t['pnl'] for t in wins), -sum(t['pnl'] for t in trades if t['pnl'] <= 0)
        res.update({"win_rate": len(wins) / n * 100, "avg_R": float(np.mean([t['R'] for t in trades])),
                    "profit_factor": gp / gl if gl > 0 else float('inf')})
        if verbose:
            by = {}
            for t in trades:
                by.setdefault(t['strategy'], []).append(t['R'])
            res["by_strategy(건수,평균R)"] = {k: (len(v), round(float(np.mean(v)), 2)) for k, v in by.items()}
    if verbose:
        print("\n===== 백테스트 결과 (수수료·슬리피지 포함) =====")
        for k, v in res.items():
            print(f"{k:>22}: {v:.2f}" if isinstance(v, float) else f"{k:>22}: {v}")
        print("※ 과거 성과는 미래를 보장하지 않습니다. 파라미터를 과최적화하지 마세요.")
    return res
 
 
def run_backtest(days, sweep=False):
    now_ms = exchange.milliseconds()
    day_ms = 86400000
    print(f"데이터 수집 중... ({days}일, 진입TF {ENTRY_TF})")
    d = add_indicators(fetch_history(ENTRY_TF, now_ms - (days + 3) * day_ms)).reset_index(drop=True)
    h1 = add_indicators(fetch_history('1h', now_ms - (days + 15) * day_ms)).reset_index(drop=True)
    h4 = add_indicators(fetch_history('4h', now_ms - (days + 60) * day_ms)).reset_index(drop=True)
    if not sweep:
        simulate(d, h1, h4)
        return
    print("\n점수 하한별 비교 (하루 거래수 / 수수료 후 성과)")
    print(f"{'점수':>5} {'거래':>5} {'일평균':>6} {'승률%':>6} {'평균R':>6} {'PF':>5} {'수익%':>7} {'MDD%':>6}")
    for ms in (50, 52, 55, 58, 62, 66, 70):
        r = simulate(d, h1, h4, verbose=False, min_score=ms)
        print(f"{ms:>5} {r['trades']:>5} {r['trades_per_day']:>6.1f} {r['win_rate']:>6.1f} {r['avg_R']:>6.2f} "
              f"{r['profit_factor']:>5.2f} {r['return_pct']:>7.2f} {r['max_dd_pct']:>6.2f}")
    print(f"\n선택 기준: 일평균 거래 >= {TARGET_TRADES_PER_DAY} 이면서 PF >= 1.2, 평균R > 0 인 가장 높은 점수를 "
          f"MIN_SCORE 로 설정하세요. 그런 행이 없으면 거래 빈도보다 수익성이 우선입니다.")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, metavar="DAYS", help="최근 N일 백테스트")
    ap.add_argument("--sweep", action="store_true", help="백테스트 시 점수 하한별 비교")
    ap.add_argument("--dry", action="store_true", help="신호만 기록(주문 없음)")
    args = ap.parse_args()
    if args.backtest:
        run_backtest(args.backtest, args.sweep)
    else:
        main_live(dry=args.dry)
