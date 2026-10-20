"""
비트코인 선물 트레이딩 봇 v2  (하이브리드: 규칙 기반 엔진 + AI 필터)
=====================================================================
v1의 문제
  - 15분마다가 아니라 '포지션이 없을 때 60초마다' GPT에 raw 캔들 전체를 보내 비용이 큼 (~$1/일)
  - LLM이 "확률 p"를 지어내서 Kelly를 계산 -> 근거 없는 수치로 베팅/레버리지 결정
  - Rule No.1(잃지 마라) 프롬프트 때문에 NO_POSITION 이 대부분 -> 1일 1건
  - 최소주문금액 강제 상향, ceil 수량, 현재가로 손익 계산 등 리스크/정합성 버그
 
v2 구조
  1) 지표/신호는 전부 로컬 계산(무료): EMA·RSI·MACD·ADX·ATR·볼린저·돈치안·거래량
     - A. 추세 눌림목  B. 돌파(스퀴즈 포함)  C. 횡보 평균회귀
     - 4h/1h 상위 타임프레임 방향 필터 + 펀딩비 필터
  2) 신호가 나온 순간에만 GPT(mini)에 요약본 전송 -> 승인/거부 + 사이즈 조절 (호출당 약 0.1센트)
  3) 리스크: 거래당 고정 위험(자본의 0.5%), ATR 기반 SL/TP, 수수료 게이트,
     일일 손실 한도, 연속손실 쿨다운, 본전 스탑 + 트레일링, 시간 청산
  4) 전략별 성과를 DB에 기록하고 최근 기대값이 나쁜 전략은 점수를 자동으로 깎음(적응형)
 
실행
  python bitcoin_bot_v2.py --backtest 30     # 최근 30일 백테스트 (먼저 이걸로 검증!)
  python bitcoin_bot_v2.py --dry             # 신호만 기록, 주문 X
  python bitcoin_bot_v2.py                   # 실거래
주의: 어떤 전략도 수익을 보장하지 않습니다. 백테스트 -> dry -> 소액 순으로 검증하세요.
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
DB_FILE = "bitcoin_trading.db"
 
RISK_PER_TRADE = 0.005       # 거래당 최대 손실 = 자본의 0.5% (SL 맞으면 이만큼만 잃음)
MAX_LEVERAGE = 5             # 레버리지 상한 (v1은 20x)
MAX_MARGIN_USAGE = 0.30      # 증거금은 자본의 30% 이내
DAILY_LOSS_LIMIT = 0.02      # 하루 -2% 넘으면 당일 거래 중단
MAX_TRADES_PER_DAY = 12
MAX_CONSEC_LOSSES = 3        # 연속 3패 -> 쿨다운
COOLDOWN_MIN = 60
 
MIN_SCORE = 62               # 신호 점수 하한 (낮추면 거래 증가, 높이면 감소) <- 백테스트로 튜닝
MIN_RR = 1.5                 # 최소 손익비
MIN_TP_PCT = 0.0025          # TP 최소 0.25% (왕복 수수료 ~0.1% 대비 최소 2.5배)
FEE_RATE = 0.0005            # 바이낸스 선물 테이커 0.05% (편도)
SLIPPAGE = 0.0001
MAX_HOLD_MIN = 240           # 4시간 지나도 +0.5R 미만이면 시간 청산
MIN_NOTIONAL = 100
BE_BUFFER = 0.0012           # 본전 스탑 = 진입가 + 0.12% (수수료 커버)
 
# 시험용 상한: 한 번에 최대 이 금액(USDT)까지만 진입. 0 이면 해제. (v1의 200 상한 유지)
MAX_NOTIONAL_USD = float(os.getenv("MAX_NOTIONAL_USD", "200"))
 
USE_LLM = os.getenv("USE_LLM", "1") == "1"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
NEWS_TTL_SEC = 1800
 
# ===================== 거래소 =====================
exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY"),
    'secret': os.getenv("BINANCE_SECRET_KEY"),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'portfolioMargin': True,
        'fetchCurrencies': False,
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
    # v2 확장 컬럼 (기존 DB 호환 마이그레이션)
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
    conn.commit()
    conn.close()
    print("데이터베이스 설정 완료")
 
 
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
    """일일 손실/거래수/연속손실 게이트. (통과여부, 사유)"""
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
    """전략별 최근 성과 (적응형 점수 보정 + LLM 컨텍스트용)"""
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
        print(f"[{days}일 요약] 거래 {r['n']}건 | 승률 {r['w'] / r['n'] * 100:.1f}% | "
              f"손익 {r['pnl']:+.2f} USDT | 평균 {r['r']:+.2f}R")
 
 
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
    if funding is not None:  # 군중 쏠림 반대 방향 가점/감점
        if side == 'long' and funding > 0.0005:
            score -= 8
        if side == 'short' and funding < -0.0005:
            score -= 8
    return {"strategy": strategy, "side": side, "score": int(min(score, 95)), "entry": price,
            "sl_dist": sl_dist, "tp_dist": tp_dist, "rr": round(real_rr, 2), "reasons": reasons or []}
 
 
def evaluate(d, h1, h4, funding=None):
    """d: 15분봉(지표 포함, 마지막=직전 마감봉), h1/h4: 상위 TF 마지막 마감봉(Series)"""
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
                      and 42 <= c['rsi'] <= 65 and c['macd_h'] > p['macd_h'])
                sl_dist = price - (rec8['low'].min() - 0.2 * atr)
            else:
                touched = (rec5['high'] >= rec5['ema20'] * 0.9995).any()
                ok = (touched and c['close'] < c['ema20'] and c['close'] < c['open'] and c['close'] < c['ema50']
                      and 35 <= c['rsi'] <= 58 and c['macd_h'] < p['macd_h'])
                sl_dist = (rec8['high'].max() + 0.2 * atr) - price
            if ok:
                score = (58 + (10 if b4 == sgn else 0) + (8 if adx1 > 25 else 0) + (6 if c['vol_z'] > 0.3 else 0)
                         + (5 if c['adx'] > 20 else 0))
                s = _mk("pullback", side, score, price, max(sl_dist, 1.2 * atr), atr, rr=2.0, funding=funding,
                        reasons=[f"1h/4h 추세 {b1}/{b4}", "EMA20 눌림 후 재상승" if side == 'long' else "EMA20 반등 후 재하락"])
                if s:
                    sigs.append(s)
 
        # ---- B. 돌파 (볼린저 스퀴즈 가점) ----
        if b1 in (0, sgn) and b4 in (0, sgn) and c['vol_z'] > 1.0 and c['adx'] > 18:
            if side == 'long':
                brk = c['close'] > c['don_hi'] and c['close'] > c['open'] and (c['close'] - c['ema20']) < 2.5 * atr
            else:
                brk = c['close'] < c['don_lo'] and c['close'] < c['open'] and (c['ema20'] - c['close']) < 2.5 * atr
            if brk:
                score = (57 + (10 if d['bb_squeeze'].iloc[-10:].any() else 0) + (8 if (b1 == sgn and b4 == sgn) else 0)
                         + (6 if c['vol_z'] > 2 else 0) + (5 if c['adx'] > p['adx'] else 0))
                s = _mk("breakout", side, score, price, 1.5 * atr, atr, rr=2.0, funding=funding,
                        reasons=["20봉 돌파", f"거래량 z={c['vol_z']:.1f}"])
                if s:
                    sigs.append(s)
 
        # ---- C. 횡보 평균회귀 ----
        if adx1 < 22 and (b4 in (0, sgn) or adx1 < 18):
            if side == 'long':
                ok = (p['low'] < p['bb_lo'] and c['close'] > c['bb_lo'] and c['close'] > c['open']
                      and p['rsi'] < 35 and c['rsi'] > p['rsi'])
                sl_dist = price - (min(p['low'], c['low']) - 0.3 * atr)
                tp_dist = c['bb_mid'] - price
                extreme = p['rsi'] < 28
            else:
                ok = (p['high'] > p['bb_up'] and c['close'] < c['bb_up'] and c['close'] < c['open']
                      and p['rsi'] > 65 and c['rsi'] < p['rsi'])
                sl_dist = (max(p['high'], c['high']) + 0.3 * atr) - price
                tp_dist = price - c['bb_mid']
                extreme = p['rsi'] > 72
            if ok and tp_dist > 0:
                score = 56 + (8 if extreme else 0) + (8 if c['vol_z'] > 0.5 else 0) + (6 if adx1 < 18 else 0) \
                        + (5 if b4 == sgn else 0)
                s = _mk("mean_reversion", side, score, price, max(sl_dist, 1.0 * atr), atr, tp_dist=tp_dist,
                        min_rr=1.3, funding=funding, reasons=["볼린저 밴드 이탈 후 복귀", f"1h ADX={adx1:.0f}(횡보)"])
                if s:
                    sigs.append(s)
 
    sigs.sort(key=lambda x: x['score'], reverse=True)
    return sigs
 
 
# ===================== 포지션 사이징 =====================
def calc_position(equity, entry, sl_dist, size_mult=1.0, notional_cap=None):
    """고정 위험 사이징: SL 맞으면 (자본 x RISK_PER_TRADE x size_mult) 만 손실."""
    risk_amt = equity * RISK_PER_TRADE * size_mult
    notional = (risk_amt / sl_dist) * entry
    cap = equity * MAX_LEVERAGE * 0.9
    if notional_cap:
        cap = min(cap, notional_cap)
    notional = min(notional, cap)
    if notional < MIN_NOTIONAL:
        # 최소주문금액으로 올려도 위험이 목표의 2배 이내일 때만 허용 (v1처럼 무조건 상향 X)
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
    """SL/TP 주문 설정. side = 포지션 방향('long'/'short')"""
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
    """시장가 진입 후 SL/TP 설정. 실패 시 즉시 청산 + ProtectionError."""
    order = exchange.create_order(SYMBOL, 'market', 'buy' if side == 'long' else 'sell', qty)
    entry = float(order.get('average') or order.get('price') or entry_hint)
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
        return df.iloc[:-1].reset_index(drop=True)  # 형성 중인 마지막 캔들 제외
 
    d15 = add_indicators(get('15m', 300))
    h1 = add_indicators(get('1h', 300))
    h4 = add_indicators(get('4h', 300))
    return d15, h1.iloc[-1], h4.iloc[-1]
 
 
# ===================== AI 필터 =====================
_client = None
_news_cache = (0, [])
 
 
def fetch_news():
    global _news_cache
    if time.time() - _news_cache[0] < NEWS_TTL_SEC:
        return _news_cache[1]
    items = []
    try:
        r = requests.get("https://serpapi.com/search.json", timeout=10, params={
            "engine": "google_news", "q": "bitcoin", "gl": "us", "hl": "en", "api_key": serp_api_key})
        if r.status_code == 200:
            items = [{"title": n.get("title", ""), "date": n.get("date", "")}
                     for n in r.json().get("news_results", [])[:8]]
    except Exception as e:
        print("뉴스 수집 오류:", e)
    _news_cache = (time.time(), items)
    return items
 
 
def llm_review(sig, d, h1, h4):
    """신호 승인/거부 + 사이즈 배수. 실패 시 None."""
    global _client
    try:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(timeout=20)
        c = d.iloc[-1]
        payload = {
            "signal": {k: sig[k] for k in ("strategy", "side", "score", "rr", "reasons")},
            "price": round(float(c['close']), 1),
            "15m": {k: round(float(c[k]), 2) for k in ("rsi", "adx", "atr", "vol_z", "macd_h")},
            "1h": {k: round(float(h1[k]), 2) for k in ("rsi", "adx")} | {"trend": bias(h1, 'ema20', 'ema50')},
            "4h": {k: round(float(h4[k]), 2) for k in ("rsi", "adx")} | {"trend": bias(h4, 'ema50', 'ema200')},
            "last_6_candles_15m": d.tail(6)[['open', 'high', 'low', 'close']].round(1).values.tolist(),
            "news": fetch_news(),
            "strategy_recent_perf": strategy_stats(),
        }
        sys_prompt = (
            "You are a risk reviewer for an automated BTC futures bot. A rule-based engine already found a valid "
            "setup. DEFAULT TO APPROVE. VETO only with concrete evidence: a major negative/positive news event "
            "opposing the side, or strong conflict with higher-timeframe structure, or this strategy's recent "
            "performance is clearly poor. Do not veto out of generic caution. size_multiplier in [0.5,1.0]: "
            "1.0 for clean setups, lower when uncertain. Reply ONLY JSON: "
            '{"verdict":"APPROVE"|"VETO","size_multiplier":0.5-1.0,"reason":"<=200 chars"}')
        r = _client.chat.completions.create(
            model=LLM_MODEL, max_tokens=150, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        out = json.loads(r.choices[0].message.content)
        out["size_multiplier"] = min(1.0, max(0.5, float(out.get("size_multiplier", 1.0))))
        out["verdict"] = "VETO" if str(out.get("verdict", "")).upper() == "VETO" else "APPROVE"
        return out
    except Exception as e:
        print("LLM 검토 실패:", e)
        return None
 
 
# ===================== 포지션 관리 =====================
def finalize_closed_trade(trade):
    """포지션이 사라졌을 때 실제 체결 내역으로 손익 확정 (불가하면 추정)"""
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
    """새 스탑을 먼저 만들고 성공하면 기존 스탑 취소 (보호 공백 방지)"""
    close_side = 'sell' if side == 'long' else 'buy'
    new_sl = px(new_sl)
    try:
        o = exchange.create_order(SYMBOL, 'STOP_MARKET', close_side, amt, None,
                                  {'stopPrice': new_sl, 'reduceOnly': True})
    except Exception as e:
        print("스탑 이동 실패(기존 스탑 유지):", e)
        return
    if trade.get('sl_order_id'):
        try:
            exchange.cancel_order(trade['sl_order_id'], SYMBOL, params={'stop': True})
        except Exception as e:
            print("기존 스탑 취소 오류(무시):", e)
    db_update(trade['id'], {'sl_price': new_sl, 'sl_order_id': o.get('id')})
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
        new_sl = entry + dirn * entry * BE_BUFFER          # 본전 스탑
    if r_now >= 1.5:
        trail = price - dirn * 0.8 * r_dist                  # 트레일링
        if new_sl is None or (trail - new_sl) * dirn > 0:
            new_sl = trail
    if new_sl is not None and (new_sl - trade['sl_price']) * dirn > 0.1 * r_dist:
        replace_stop(trade, side, amt, new_sl)
 
 
def adopt_orphan(side, amt, entry):
    """DB에 없는 기존 포지션 편입 + 보호주문 없으면 설정"""
    d15, _, _ = load_features()
    atr = float(d15.iloc[-1]['atr'])
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
def run_signal_cycle(dry=False):
    equity = get_equity()
    ok, reason = risk_gate(equity)
    if not ok:
        print("[게이트]", reason)
        return
    d15, h1, h4 = load_features()
    sigs = evaluate(d15, h1, h4, get_funding())
 
    # 적응형 점수 보정: 최근 기대값이 나쁜 전략은 감점, 좋은 전략은 가점
    stats = strategy_stats()
    for s in sigs:
        st = stats.get(s['strategy'])
        if st and st['n'] >= 8:
            s['score'] += int(max(-10, min(5, st['avg_R'] * 10)))
    sigs = sorted([s for s in sigs if s['score'] >= MIN_SCORE], key=lambda x: x['score'], reverse=True)
 
    price = float(d15.iloc[-1]['close'])
    if not sigs:
        print(f"[{datetime.now():%H:%M}] {price:,.1f} | 신호 없음")
        return
    sig = sigs[0]
    print(f"[{datetime.now():%H:%M}] 신호: {sig['strategy']} {sig['side'].upper()} 점수 {sig['score']} "
          f"RR {sig['rr']} | {', '.join(sig['reasons'])}")
 
    if dry:
        save_signal(sig, False, "dry-run")
        return
 
    size_mult, note = 1.0, ""
    if USE_LLM:
        rv = llm_review(sig, d15, h1, h4)
        if rv is None:
            if sig['score'] < 75:
                save_signal(sig, False, "LLM 실패 + 점수 미달")
                return
            size_mult, note = 0.5, "LLM 실패, 절반 사이즈"
        elif rv['verdict'] == 'VETO':
            print("AI 거부:", rv.get('reason'))
            save_signal(sig, False, "VETO: " + str(rv.get('reason')))
            return
        else:
            size_mult, note = rv['size_multiplier'], str(rv.get('reason', ''))
            print(f"AI 승인 (x{size_mult:.2f}): {note}")
 
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
    print("\n=== Bitcoin Bot v2 시작 ===", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"모드: {'DRY-RUN(주문 없음)' if dry else '실거래'} | AI필터: {LLM_MODEL if USE_LLM else 'OFF'} "
          f"| 위험/거래 {RISK_PER_TRADE * 100:.2f}% | 최대 {MAX_LEVERAGE}x | 최소점수 {MIN_SCORE}")
    setup_database()
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
 
            if trade:  # 포지션 없음 + DB엔 열림 -> 종료됨 (진입 직후 반영 지연 방지)
                age = (datetime.now() - datetime.fromisoformat(trade['timestamp'])).total_seconds()
                if age < 20:
                    time.sleep(3)
                    continue
                finalize_closed_trade(trade)
                cancel_all_orders()
 
            now = time.time()
            bucket = int(now // 900)
            if bucket == last_bucket or now % 900 < 5:   # 15분봉 마감 5초 후 1회만 평가
                time.sleep(5)
                continue
            last_bucket = bucket
            run_signal_cycle(dry)
            time.sleep(5)
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
 
 
def simulate(df15, h1, h4, equity0=1000.0, warmup=120, verbose=True):
    """15분봉 마감 신호 -> 다음 봉 시가 진입. SL/TP 동시 터치 시 SL 우선(보수적)."""
    ct15 = (df15['timestamp'] + pd.Timedelta(minutes=15)).values
    ct1 = (h1['timestamp'] + pd.Timedelta(hours=1)).values
    ct4 = (h4['timestamp'] + pd.Timedelta(hours=4)).values
    eq, peak, mdd = equity0, equity0, 0.0
    trades, pos, pending = [], None, None
    cur_day, day_start, day_n, halted, consec, cool_until = None, eq, 0, False, 0, -1
    hold_bars = MAX_HOLD_MIN // 15
 
    for i in range(warmup, len(df15)):
        bar = df15.iloc[i]
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
                    cool_until, consec = i + COOLDOWN_MIN // 15, 0
                pos = None
                peak = max(peak, eq)
                mdd = max(mdd, (peak - eq) / peak)
                if eq - day_start <= -DAILY_LOSS_LIMIT * day_start:
                    halted = True
 
        if pos is None and not halted and day_n < MAX_TRADES_PER_DAY and i >= cool_until and i < len(df15) - 1:
            j1 = np.searchsorted(ct1, ct15[i], side='right') - 1
            j4 = np.searchsorted(ct4, ct15[i], side='right') - 1
            if j1 >= 0 and j4 >= 0:
                sigs = [s for s in evaluate(df15.iloc[i - 29:i + 1], h1.iloc[j1], h4.iloc[j4])
                        if s['score'] >= MIN_SCORE]
                if sigs:
                    pending = sigs[0]
 
    n = len(trades)
    res = {"trades": n, "final_equity": eq, "return_pct": (eq / equity0 - 1) * 100, "max_dd_pct": mdd * 100}
    if n:
        wins = [t for t in trades if t['pnl'] > 0]
        gp, gl = sum(t['pnl'] for t in wins), -sum(t['pnl'] for t in trades if t['pnl'] <= 0)
        days = max(1, len({t['day'] for t in trades}))
        span = max(1, (df15['timestamp'].iloc[-1] - df15['timestamp'].iloc[warmup]).days)
        res.update({"trades_per_day": n / span, "win_rate": len(wins) / n * 100,
                    "avg_R": float(np.mean([t['R'] for t in trades])),
                    "profit_factor": gp / gl if gl > 0 else float('inf')})
        if verbose:
            by = {}
            for t in trades:
                by.setdefault(t['strategy'], []).append(t['R'])
            res["by_strategy"] = {k: (len(v), round(float(np.mean(v)), 2)) for k, v in by.items()}
    if verbose:
        print("\n===== 백테스트 결과 (수수료·슬리피지 포함) =====")
        for k, v in res.items():
            print(f"{k:>16}: {v:.2f}" if isinstance(v, float) else f"{k:>16}: {v}")
        print("※ 과거 성과는 미래를 보장하지 않습니다. 파라미터를 과최적화하지 마세요.")
    return res
 
 
def run_backtest(days):
    now_ms = exchange.milliseconds()
    day_ms = 86400000
    print(f"데이터 수집 중... ({days}일)")
    d15 = add_indicators(fetch_history('15m', now_ms - (days + 3) * day_ms))
    h1 = add_indicators(fetch_history('1h', now_ms - (days + 15) * day_ms))
    h4 = add_indicators(fetch_history('4h', now_ms - (days + 60) * day_ms))
    simulate(d15.reset_index(drop=True), h1.reset_index(drop=True), h4.reset_index(drop=True))
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, metavar="DAYS", help="최근 N일 백테스트")
    ap.add_argument("--dry", action="store_true", help="신호만 기록(주문 없음)")
    args = ap.parse_args()
    if args.backtest:
        run_backtest(args.backtest)
    else:
        main_live(dry=args.dry)