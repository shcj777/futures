"""
비트코인 선물 트레이딩 봇 FINAL  (역방향 매매 + 손절 가깝게/익절 멀게 + AI 레버리지 1~5배)
=====================================================================
신호 엔진은 v4 (추세 눌림목 + 돌파) 를 그대로 사용하고, 체결 방향을 반대로 뒤집습니다.
  원신호 LONG -> SHORT 체결 / 원신호 SHORT -> LONG 체결

INVERT_MODE (손익 거리 설정 방식)
  "far_tp" : 방향 반대 + 손절은 가깝게(ATR x FAR_SL_ATR), 익절은 멀게(손절 x FAR_TP_R)   <- 기본
  "same"   : 방향만 반대, SL/TP 거리는 원신호 그대로 (RR 3:1)
  "mirror" : 방향 반대 + 원신호의 SL <-> TP 거리를 맞바꿈 (익절 가깝게/손절 멀게)

투자금 / 레버리지
  - 증거금 = 원금 x MARGIN_FRACTION x (AI 가 정한 비중 0.2~1.0)
  - AI 가 레버리지 1~5배를 판단. AI 실패/미사용 시 기본값(DEFAULT_LEV, DEFAULT_FRAC)
  - SL 이 청산가에 가까우면 레버리지를 자동으로 낮춤

포지션 관리
  - far_tp 기본: SL/TP 브래킷 + 시간청산만 사용 (MANAGE_FAR=True 로 부분익절/트레일링 켤 수 있음)
  - same: 1R 본전스탑, 1.5R 부분익절, 2R 이상 트레일링, 조기청산 사용

!! 경고 !!
  - 지금까지의 백테스트(180일)에서 원본/same/mirror 모두 평균 R 이 음수였습니다.
    신호를 뒤집는다고 수익이 되는 것이 아니며, 수수료·슬리피지 때문에 양쪽 모두 손실일 수 있습니다.
  - 반드시 --compare 로 여러 모드를 비교하고, 기간(90/180/365일)을 바꿔 일관성을 확인하세요.
  - 실거래 전 --dry 로 충분히 관찰하고, 처음에는 MAX_NOTIONAL_USD 로 상한을 걸고 소액으로 시작하세요.

실행
  python bitcoinv5.py --backtest 180     # 현재 설정으로 백테스트
  python bitcoinv5.py --compare 180      # 5가지 설정 한 번에 비교 (표 출력 + CSV 저장)
  python bitcoinv5.py --dry              # 신호만 기록, 주문 X
  python bitcoinv5.py                    # 실거래
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
DB_FILE = "bitcoin_trading_final.db"

# --- 역방향 ---
INVERT = True
INVERT_MODE = "far_tp"        # "far_tp" | "same" | "mirror"
MANAGE_FAR = False            # far_tp 에서 부분익절/본전/트레일링/조기청산 사용 여부
MANAGE = not (INVERT and (INVERT_MODE == "mirror" or (INVERT_MODE == "far_tp" and not MANAGE_FAR)))
INV_MAX_HOLD_MIN = 480        # 관리 OFF 모드의 시간청산 (분)

# --- far_tp 모드: 손절 가깝게, 익절 멀게 ---
FAR_SL_ATR = 1.0              # 손절 거리 = ATR x 1.0
FAR_SL_MIN_PCT = 0.003        # 손절 거리 최소 0.3% (너무 좁으면 비용이 손절폭을 잠식)
FAR_TP_R = 3.0                # 익절 거리 = 손절 거리 x 3.0

# --- 투자금 / 레버리지 ---
MARGIN_FRACTION = 0.50        # 증거금 최대 = 원금의 50% (원하면 0.90 까지)
MIN_LEV, MAX_LEVERAGE = 1, 5
DEFAULT_LEV = 2               # AI 미사용/실패 시
DEFAULT_FRAC = 0.5
MIN_FRAC = 0.2
LIQ_SAFETY = 0.6              # SL 거리가 (청산거리 x 0.6) 이내가 되도록 레버리지 제한
BT_LEVERAGE = 1               # 백테스트 고정 레버리지
BT_FRAC = 1.0

# --- 위험 관리 ---
DAILY_LOSS_LIMIT = 0.10
DAILY_PROFIT_LOCK = 0.10
MAX_TRADES_PER_DAY = 10
MAX_CONSEC_LOSSES = 2
COOLDOWN_MIN = 60

# --- 신호 / 손익비 (원신호 생성용) ---
MIN_SCORE = 62
MIN_RR = 2.5
MIN_TP_PCT = 0.004
SL_MIN_ATR = 0.8
SL_MAX_ATR = 2.5
RR_PULLBACK = 3.0
RR_BREAKOUT = 3.5
ENABLE = {"pullback": True, "breakout": True, "mean_reversion": False}

# --- 비용 ---
FEE_RATE = 0.0005
SLIPPAGE = 0.0002
MIN_NOTIONAL = 100

# --- 포지션 관리 (MANAGE=True 일 때만) ---
BE_R = 1.0
BE_BUFFER = 0.0012
PARTIAL_R = 1.5
PARTIAL_FRAC = 0.5
TRAIL_START_R = 2.0
TRAIL_DIST_R = 1.0
EARLY_CUT_MIN = 90
EARLY_CUT_R = 0.3
MAX_HOLD_MIN = 300

# 안전 상한(USDT). 0 이면 해제
MAX_NOTIONAL_USD = float(os.getenv("MAX_NOTIONAL_USD", "0"))

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
    for col, typ in [("strategy", "TEXT"), ("score", "REAL"), ("sl_order_id", "TEXT"),
                     ("tp_order_id", "TEXT"), ("risk_amount", "REAL"), ("fees", "REAL"),
                     ("r_multiple", "REAL"), ("partial_done", "INTEGER")]:
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
        sl_order_id,tp_order_id,risk_amount,partial_done) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)''',
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
    if day_pnl / base >= DAILY_PROFIT_LOCK:
        return False, f"일일 이익 잠금 도달 ({day_pnl / base * 100:.2f}%) - 오늘은 거래 중단"
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


# ===================== 신호 엔진 (원본 v4 로직) =====================
def _mk(strategy, side, score, price, sl_dist, atr, rr=None, tp_dist=None, min_rr=MIN_RR,
        funding=None, reasons=None):
    sl_dist = min(max(sl_dist, SL_MIN_ATR * atr), SL_MAX_ATR * atr)
    if tp_dist is None:
        tp_dist = rr * sl_dist
    real_rr = tp_dist / sl_dist
    if real_rr < min_rr or tp_dist / price < MIN_TP_PCT:
        return None
    if funding is not None:
        if side == 'long' and funding > 0.0005:
            score -= 8
        if side == 'short' and funding < -0.0005:
            score -= 8
    return {"strategy": strategy, "side": side, "score": int(min(score, 95)), "entry": price,
            "sl_dist": sl_dist, "tp_dist": tp_dist, "rr": round(real_rr, 2), "reasons": reasons or [],
            "atr": atr}


def evaluate(d, h1, h4, funding=None):
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
        if ENABLE["pullback"] and b1 == sgn and b4 in (0, sgn):
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
                s = _mk("pullback", side, score, price, max(sl_dist, 1.2 * atr), atr, rr=RR_PULLBACK,
                        funding=funding,
                        reasons=[f"1h/4h 추세 {b1}/{b4}",
                                 "EMA20 눌림 후 재상승" if side == 'long' else "EMA20 반등 후 재하락"])
                if s:
                    sigs.append(s)

        # ---- B. 돌파 ----
        if ENABLE["breakout"] and b1 in (0, sgn) and b4 in (0, sgn) and c['vol_z'] > 1.0 and c['adx'] > 18:
            if side == 'long':
                brk = c['close'] > c['don_hi'] and c['close'] > c['open'] and (c['close'] - c['ema20']) < 2.5 * atr
            else:
                brk = c['close'] < c['don_lo'] and c['close'] < c['open'] and (c['ema20'] - c['close']) < 2.5 * atr
            if brk:
                score = (57 + (10 if d['bb_squeeze'].iloc[-10:].any() else 0) + (8 if (b1 == sgn and b4 == sgn) else 0)
                         + (6 if c['vol_z'] > 2 else 0) + (5 if c['adx'] > p['adx'] else 0))
                s = _mk("breakout", side, score, price, 1.5 * atr, atr, rr=RR_BREAKOUT, funding=funding,
                        reasons=["20봉 돌파", f"거래량 z={c['vol_z']:.1f}"])
                if s:
                    sigs.append(s)

        # ---- C. 평균회귀 (기본 OFF) ----
        if ENABLE["mean_reversion"] and adx1 < 22 and (b4 in (0, sgn) or adx1 < 18):
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
                        funding=funding, reasons=["볼린저 밴드 이탈 후 복귀", f"1h ADX={adx1:.0f}(횡보)"])
                if s:
                    sigs.append(s)

    sigs.sort(key=lambda x: x['score'], reverse=True)
    return sigs


def invert_signal(sig):
    """원신호를 반대 방향으로 변환하고 모드에 따라 SL/TP 거리를 재설정."""
    if not INVERT:
        return sig
    s = dict(sig)
    s['orig_side'] = sig['side']
    s['side'] = 'short' if sig['side'] == 'long' else 'long'
    s['strategy'] = "inv_" + sig['strategy']
    if INVERT_MODE == "mirror":
        s['sl_dist'], s['tp_dist'] = sig['tp_dist'], sig['sl_dist']
        s['rr'] = round(s['tp_dist'] / s['sl_dist'], 2)
    elif INVERT_MODE == "far_tp":
        s['sl_dist'] = max(FAR_SL_ATR * sig['atr'], sig['entry'] * FAR_SL_MIN_PCT)
        s['tp_dist'] = FAR_TP_R * s['sl_dist']
        s['rr'] = round(FAR_TP_R, 2)
    s['reasons'] = [f"[역방향/{INVERT_MODE}] 원신호 {sig['side'].upper()} -> {s['side'].upper()}"] \
                   + list(sig['reasons'])
    return s


# ===================== 포지션 사이징 =====================
def calc_position(equity, entry, sl_dist, leverage=DEFAULT_LEV, frac=DEFAULT_FRAC):
    """증거금 = 원금 x MARGIN_FRACTION x frac, 명목 = 증거금 x 레버리지.
    SL 거리가 청산가에 너무 가까우면 레버리지를 자동으로 낮춘다."""
    leverage = int(min(MAX_LEVERAGE, max(MIN_LEV, leverage)))
    frac = min(1.0, max(MIN_FRAC, frac))
    sl_pct = sl_dist / entry
    max_lev_liq = max(MIN_LEV, int(LIQ_SAFETY / sl_pct)) if sl_pct > 0 else MAX_LEVERAGE
    leverage = min(leverage, max_lev_liq)

    margin = equity * MARGIN_FRACTION * frac
    notional = margin * leverage
    if MAX_NOTIONAL_USD:
        notional = min(notional, MAX_NOTIONAL_USD)
    if notional < MIN_NOTIONAL:
        return None
    qty = math.floor(notional / entry * 1000) / 1000
    if qty * entry < MIN_NOTIONAL:
        qty = round(qty + 0.001, 3)
    if qty <= 0:
        return None
    notional = qty * entry
    return {"qty": qty, "notional": notional, "leverage": leverage, "margin": notional / leverage,
            "risk_amt": qty * sl_dist}


# ===================== 거래소 헬퍼 =====================
def get_equity():
    bal = exchange.fetch_balance()
    for key in ('total', 'free'):
        part = bal.get(key, {}) or {}
        v = sum(float(part.get(c) or 0) for c in ('USDT', 'USDC'))
        if v > 0:
            return v
    raise RuntimeError(f"잔고 조회 실패: total={bal.get('total')}, free={bal.get('free')}")


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
        return df.iloc[:-1].reset_index(drop=True)

    d15 = add_indicators(get('15m', 300))
    h1 = add_indicators(get('1h', 300))
    h4 = add_indicators(get('4h', 300))
    return d15, h1.iloc[-1], h4.iloc[-1]


# ===================== AI 판단 (승인/거부 + 레버리지 + 투자 비중) =====================
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
    """반환: {verdict, leverage(1~5), size_fraction(0.2~1.0), reason} 또는 None"""
    global _client
    try:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(timeout=20)
        c = d.iloc[-1]
        price = float(c['close'])
        payload = {
            "trade": {"side_to_execute": sig['side'], "original_signal_side": sig.get('orig_side'),
                      "strategy": sig['strategy'], "score": sig['score'], "reasons": sig['reasons'],
                      "sl_distance_pct": round(sig['sl_dist'] / price * 100, 3),
                      "tp_distance_pct": round(sig['tp_dist'] / price * 100, 3), "rr": sig['rr']},
            "price": round(price, 1),
            "15m": {k: round(float(c[k]), 2) for k in ("rsi", "adx", "atr", "vol_z", "macd_h")},
            "atr_pct": round(float(c['atr']) / price * 100, 3),
            "1h": {k: round(float(h1[k]), 2) for k in ("rsi", "adx")} | {"trend": bias(h1, 'ema20', 'ema50')},
            "4h": {k: round(float(h4[k]), 2) for k in ("rsi", "adx")} | {"trend": bias(h4, 'ema50', 'ema200')},
            "last_6_candles_15m": d.tail(6)[['open', 'high', 'low', 'close']].round(1).values.tolist(),
            "news": fetch_news(),
            "strategy_recent_perf": strategy_stats(),
            "limits": {"leverage": "1-5", "size_fraction": "0.2-1.0 of the margin budget"},
        }
        sys_prompt = (
            "You are a risk manager for an automated BTC futures bot that deliberately FADES (trades opposite to) "
            "a trend/breakout signal. 'side_to_execute' is the real order direction; 'original_signal_side' is the "
            "signal being faded. The trade uses a tight stop and a far take-profit, so a low win rate is expected. "
            "Decide: (1) verdict APPROVE or VETO. VETO only with concrete evidence, e.g. a strong "
            "trend/news move that is clearly running against side_to_execute with high ADX and momentum, or this "
            "strategy's recent performance is clearly poor. (2) leverage, integer 1-5: use higher leverage ONLY when "
            "the stop distance is small relative to price, volatility (atr_pct) is low, and the setup is clean; "
            "use 1-2 when the stop is wide, volatility is high, or higher timeframes oppose side_to_execute. "
            "A stop-loss hit at leverage L loses roughly sl_distance_pct * L percent of the margin, so keep that "
            "tolerable. (3) size_fraction 0.2-1.0: how much of the margin budget to commit. "
            'Reply ONLY JSON: {"verdict":"APPROVE"|"VETO","leverage":1-5,"size_fraction":0.2-1.0,'
            '"reason":"<=200 chars"}')
        r = _client.chat.completions.create(
            model=LLM_MODEL, max_tokens=200, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        out = json.loads(r.choices[0].message.content)
        out["leverage"] = int(min(MAX_LEVERAGE, max(MIN_LEV, round(float(out.get("leverage", DEFAULT_LEV))))))
        out["size_fraction"] = min(1.0, max(MIN_FRAC, float(out.get("size_fraction", DEFAULT_FRAC))))
        out["verdict"] = "VETO" if str(out.get("verdict", "")).upper() == "VETO" else "APPROVE"
        return out
    except Exception as e:
        print("LLM 검토 실패:", e)
        return None


# ===================== 포지션 관리 =====================
def finalize_closed_trade(trade):
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
    close_side = 'sell' if side == 'long' else 'buy'
    new_sl = px(new_sl)
    try:
        o = exchange.create_order(SYMBOL, 'STOP_MARKET', close_side, amt, None,
                                  {'stopPrice': new_sl, 'reduceOnly': True})
    except Exception as e:
        print("스탑 이동 실패(기존 스탑 유지):", e)
        return False
    if trade.get('sl_order_id'):
        try:
            exchange.cancel_order(trade['sl_order_id'], SYMBOL, params={'stop': True})
        except Exception as e:
            print("기존 스탑 취소 오류(무시):", e)
    db_update(trade['id'], {'sl_price': new_sl, 'sl_order_id': o.get('id')})
    trade['sl_price'], trade['sl_order_id'] = new_sl, o.get('id')
    print(f"스탑 이동 -> {new_sl:,.1f}")
    return True


def do_partial_take_profit(trade, side, amt, price):
    entry = trade['entry_price']
    dirn = 1 if side == 'long' else -1
    half = math.floor(amt * PARTIAL_FRAC * 1000) / 1000
    remain = round(amt - half, 3)
    if half < 0.001 or remain < 0.001 or remain * price < MIN_NOTIONAL:
        db_update(trade['id'], {'partial_done': 1})
        trade['partial_done'] = 1
        return
    try:
        close_market(side, half)
    except Exception as e:
        print("부분 익절 실패:", e)
        return
    print(f"부분 익절: {half} BTC @ ~{price:,.1f}")
    db_update(trade['id'], {'amount': remain, 'partial_done': 1})
    trade['amount'], trade['partial_done'] = remain, 1

    be = entry + dirn * entry * BE_BUFFER
    replace_stop(trade, side, remain, be)

    close_side = 'sell' if side == 'long' else 'buy'
    if trade.get('tp_order_id'):
        try:
            exchange.cancel_order(trade['tp_order_id'], SYMBOL, params={'stop': True})
        except Exception as e:
            print("기존 TP 취소 오류(무시):", e)
    try:
        o = exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', close_side, remain, None,
                                  {'stopPrice': px(trade['tp_price']), 'reduceOnly': True})
        db_update(trade['id'], {'tp_order_id': o.get('id')})
        trade['tp_order_id'] = o.get('id')
    except Exception as e:
        print("TP 재설정 실패(트레일링/SL 로 관리):", e)


def manage_position(trade, side, amt):
    price = exchange.fetch_ticker(SYMBOL)['last']
    entry = trade['entry_price']
    r_dist = entry * (trade['sl_percentage'] or 0)
    if r_dist <= 0:
        return
    dirn = 1 if side == 'long' else -1
    r_now = dirn * (price - entry) / r_dist
    age_min = (datetime.now() - datetime.fromisoformat(trade['timestamp'])).total_seconds() / 60

    # 관리 OFF 모드: SL/TP 브래킷 + 시간청산만 사용
    if not MANAGE:
        if age_min >= INV_MAX_HOLD_MIN:
            print(f"시간 청산 ({age_min:.0f}분 경과, {r_now:+.2f}R)")
            close_market(side, amt)
            cancel_all_orders()
        return

    partial_done = bool(trade.get('partial_done'))
    if not partial_done and age_min >= EARLY_CUT_MIN and r_now < EARLY_CUT_R:
        print(f"조기 청산 ({age_min:.0f}분 경과, {r_now:+.2f}R)")
        close_market(side, amt)
        cancel_all_orders()
        return
    if age_min >= MAX_HOLD_MIN and r_now < 0.5:
        print(f"시간 청산 ({age_min:.0f}분 경과, {r_now:+.2f}R)")
        close_market(side, amt)
        cancel_all_orders()
        return
    if not partial_done and r_now >= PARTIAL_R:
        do_partial_take_profit(trade, side, amt, price)
        return

    new_sl = None
    if r_now >= BE_R:
        new_sl = entry + dirn * entry * BE_BUFFER
    if r_now >= TRAIL_START_R:
        trail = price - dirn * TRAIL_DIST_R * r_dist
        if new_sl is None or (trail - new_sl) * dirn > 0:
            new_sl = trail
    if new_sl is not None and (new_sl - trade['sl_price']) * dirn > 0.1 * r_dist:
        replace_stop(trade, side, amt, new_sl)


def adopt_orphan(side, amt, entry):
    d15, _, _ = load_features()
    atr = float(d15.iloc[-1]['atr'])
    sl_dist = max(1.5 * atr, entry * 0.004)
    tp_dist = 2.5 * sl_dist
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
    sigs = [s for s in evaluate(d15, h1, h4, get_funding()) if s['score'] >= MIN_SCORE]
    sigs.sort(key=lambda x: x['score'], reverse=True)

    price = float(d15.iloc[-1]['close'])
    if not sigs:
        print(f"[{datetime.now():%H:%M}] {price:,.1f} | 신호 없음")
        return

    orig = sigs[0]
    sig = invert_signal(orig)      # <<< 방향 반전 + SL/TP 재설정
    print(f"[{datetime.now():%H:%M}] 원신호 {orig['strategy']} {orig['side'].upper()} (점수 {orig['score']}) "
          f"-> 실행 {sig['side'].upper()} | SL {sig['sl_dist'] / price * 100:.2f}% / "
          f"TP {sig['tp_dist'] / price * 100:.2f}% (RR {sig['rr']})")

    if dry:
        save_signal(sig, False, "dry-run")
        return

    lev, frac, note = DEFAULT_LEV, DEFAULT_FRAC, ""
    if USE_LLM:
        rv = llm_review(sig, d15, h1, h4)
        if rv is None:
            note = f"LLM 실패 -> 기본값 {DEFAULT_LEV}x / {DEFAULT_FRAC:.0%}"
            print(note)
        elif rv['verdict'] == 'VETO':
            print("AI 거부:", rv.get('reason'))
            save_signal(sig, False, "VETO: " + str(rv.get('reason')))
            return
        else:
            lev, frac, note = rv['leverage'], rv['size_fraction'], str(rv.get('reason', ''))
            print(f"AI 승인: 레버리지 {lev}x, 증거금 비중 {frac:.0%} | {note}")

    pos = calc_position(equity, sig['entry'], sig['sl_dist'], lev, frac)
    if not pos:
        save_signal(sig, False, "사이징 불가(잔고/최소주문)")
        print("사이징 불가 - 건너뜀")
        return
    if pos['leverage'] != lev:
        print(f"청산 안전 규칙으로 레버리지 {lev}x -> {pos['leverage']}x 조정")

    try:
        exchange.papiPostUmLeverage({'symbol': exchange.market(SYMBOL)['id'], 'leverage': pos['leverage']})
    except Exception as e:
        raise RuntimeError(f"레버리지 설정 실패, 진입 취소: {e}")

    entry, sl, tp, sl_id, tp_id = open_with_protection(
        sig['side'], pos['qty'], sig['entry'], sig['sl_dist'], sig['tp_dist'])
    tid = save_trade({
        'action': sig['side'], 'entry_price': entry, 'amount': pos['qty'], 'leverage': pos['leverage'],
        'sl_price': sl, 'tp_price': tp, 'sl_percentage': sig['sl_dist'] / entry,
        'tp_percentage': sig['tp_dist'] / entry, 'position_size_percentage': pos['margin'] / equity,
        'investment_amount': pos['margin'], 'strategy': sig['strategy'], 'score': sig['score'],
        'sl_order_id': sl_id, 'tp_order_id': tp_id, 'risk_amount': pos['risk_amt']})
    save_signal(sig, True, note)
    print(f"\n=== {sig['side'].upper()} 진입 #{tid} [{sig['strategy']}] ===")
    print(f"진입 {entry:,.1f} | SL {sl:,.1f} | TP {tp:,.1f} | {pos['leverage']}x | "
          f"수량 {pos['qty']} BTC | 증거금 {pos['margin']:.2f} USDT ({pos['margin'] / equity * 100:.0f}%) | "
          f"SL 시 예상손실 {pos['risk_amt']:.2f} USDT ({pos['risk_amt'] / equity * 100:.1f}%)")


# ===================== 메인 루프 =====================
def main_live(dry=False):
    print("\n=== Bitcoin Bot FINAL 시작 ===", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"모드: {'DRY-RUN(주문 없음)' if dry else '실거래'} | 반전: {INVERT} ({INVERT_MODE}) | "
          f"AI: {LLM_MODEL if USE_LLM else 'OFF'} | 증거금 최대 {MARGIN_FRACTION:.0%} | "
          f"레버리지 {MIN_LEV}~{MAX_LEVERAGE}x | 최소점수 {MIN_SCORE} | 관리기능 {MANAGE}")
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

            if trade:
                age = (datetime.now() - datetime.fromisoformat(trade['timestamp'])).total_seconds()
                if age < 20:
                    time.sleep(3)
                    continue
                finalize_closed_trade(trade)
                cancel_all_orders()

            now = time.time()
            bucket = int(now // 900)
            if bucket == last_bucket or now % 900 < 5:
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


def _better(a, b, dn):
    return max(a, b) if dn == 1 else min(a, b)


def simulate(df15, h1, h4, equity0=1000.0, warmup=120, verbose=True):
    """원신호 -> (반전/SL·TP 재설정) -> 다음 봉 시가 진입. SL/TP 동시 터치 시 SL 우선(보수적).
    증거금 MARGIN_FRACTION x 고정 레버리지(BT_LEVERAGE) 로 사이징. (AI 판단은 백테스트에서 제외)"""
    ct15 = (df15['timestamp'] + pd.Timedelta(minutes=15)).values
    ct1 = (h1['timestamp'] + pd.Timedelta(hours=1)).values
    ct4 = (h4['timestamp'] + pd.Timedelta(hours=4)).values
    eq, peak, mdd = equity0, equity0, 0.0
    trades, pos, pending = [], None, None
    cur_day, day_start, day_n, halted, consec, cool_until = None, eq, 0, False, 0, -1
    hold_bars = MAX_HOLD_MIN // 15
    inv_hold_bars = INV_MAX_HOLD_MIN // 15
    cut_bars = max(1, EARLY_CUT_MIN // 15)

    for i in range(warmup, len(df15)):
        bar = df15.iloc[i]
        day = bar['timestamp'].date()
        if day != cur_day:
            cur_day, day_start, day_n, halted = day, eq, 0, False

        # --- 진입 ---
        if pending and pos is None:
            s = pending
            e = bar['open'] * (1 + (SLIPPAGE if s['side'] == 'long' else -SLIPPAGE))
            dirn = 1 if s['side'] == 'long' else -1
            q = calc_position(eq, e, s['sl_dist'], BT_LEVERAGE, BT_FRAC)
            if q:
                pos = {"side": s['side'], "dirn": dirn, "entry": e, "qty": q['qty'], "r": s['sl_dist'],
                       "sl": e - dirn * s['sl_dist'], "tp": e + dirn * s['tp_dist'], "open_i": i,
                       "ext": e, "risk": q['risk_amt'], "strategy": s['strategy'], "score": s['score'],
                       "partial": False, "banked": 0.0}
                day_n += 1
            pending = None

        # --- 포지션 관리 ---
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
                age = i - pos['open_i']
                if not MANAGE:
                    if age >= inv_hold_bars:
                        exit_p = bar['close'] * (1 - dn * SLIPPAGE)
                else:
                    pos['ext'] = max(pos['ext'], bar['high']) if dn == 1 else min(pos['ext'], bar['low'])
                    r_ext = dn * (pos['ext'] - pos['entry']) / pos['r']
                    if r_ext >= BE_R:
                        pos['sl'] = _better(pos['sl'], pos['entry'] + dn * pos['entry'] * BE_BUFFER, dn)
                    if r_ext >= TRAIL_START_R:
                        pos['sl'] = _better(pos['sl'], pos['ext'] - dn * TRAIL_DIST_R * pos['r'], dn)
                    if not pos['partial'] and r_ext >= PARTIAL_R:
                        q_half = pos['qty'] * PARTIAL_FRAC
                        p_px = pos['entry'] + dn * PARTIAL_R * pos['r']
                        fee = (pos['entry'] + p_px) * q_half * FEE_RATE
                        part = dn * (p_px - pos['entry']) * q_half - fee
                        eq += part
                        pos['banked'] += part
                        pos['qty'] -= q_half
                        pos['partial'] = True
                        pos['sl'] = _better(pos['sl'], pos['entry'] + dn * pos['entry'] * BE_BUFFER, dn)
                    r_now = dn * (bar['close'] - pos['entry']) / pos['r']
                    if (not pos['partial'] and age >= cut_bars and r_now < EARLY_CUT_R) or \
                            (age >= hold_bars and r_now < 0.5):
                        exit_p = bar['close'] * (1 - dn * SLIPPAGE)

            if exit_p is not None:
                fees = (pos['entry'] + exit_p) * pos['qty'] * FEE_RATE
                pnl_last = dn * (exit_p - pos['entry']) * pos['qty'] - fees
                eq += pnl_last
                pnl = pnl_last + pos['banked']
                trades.append({"strategy": pos['strategy'], "pnl": pnl, "R": pnl / pos['risk'],
                               "bars": i - pos['open_i'], "day": day, "score": pos['score']})
                consec = consec + 1 if pnl < 0 else 0
                if consec >= MAX_CONSEC_LOSSES:
                    cool_until, consec = i + COOLDOWN_MIN // 15, 0
                pos = None
                peak = max(peak, eq)
                mdd = max(mdd, (peak - eq) / peak)
                if eq - day_start <= -DAILY_LOSS_LIMIT * day_start:
                    halted = True
                if eq - day_start >= DAILY_PROFIT_LOCK * day_start:
                    halted = True
                if eq <= 0:
                    break

        # --- 신호 평가 ---
        if pos is None and not halted and day_n < MAX_TRADES_PER_DAY and i >= cool_until and i < len(df15) - 1:
            j1 = np.searchsorted(ct1, ct15[i], side='right') - 1
            j4 = np.searchsorted(ct4, ct15[i], side='right') - 1
            if j1 >= 0 and j4 >= 0:
                sigs = [s for s in evaluate(df15.iloc[i - 29:i + 1], h1.iloc[j1], h4.iloc[j4])
                        if s['score'] >= MIN_SCORE]
                if sigs:
                    pending = invert_signal(sigs[0])

    n = len(trades)
    res = {"trades": n, "final_equity": eq, "return_pct": (eq / equity0 - 1) * 100, "max_dd_pct": mdd * 100}
    if n:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        gp, gl = sum(t['pnl'] for t in wins), -sum(t['pnl'] for t in losses)
        span = max(1, (df15['timestamp'].iloc[-1] - df15['timestamp'].iloc[warmup]).days)
        res.update({"trades_per_day": n / span, "win_rate": len(wins) / n * 100,
                    "avg_R": float(np.mean([t['R'] for t in trades])),
                    "avg_win_R": float(np.mean([t['R'] for t in wins])) if wins else 0.0,
                    "avg_loss_R": float(np.mean([t['R'] for t in losses])) if losses else 0.0,
                    "profit_factor": gp / gl if gl > 0 else float('inf')})
    if verbose:
        print(f"\n===== 백테스트 결과 (반전={INVERT}/{INVERT_MODE}, 관리={MANAGE}, "
              f"{BT_LEVERAGE}x, 증거금 {MARGIN_FRACTION:.0%}) =====")
        for k, v in res.items():
            print(f"{k:>16}: {v:.2f}" if isinstance(v, float) else f"{k:>16}: {v}")
        if n:
            print("\n-- 전략별 (거래수, 평균R, 승률%) --")
            by = {}
            for t in trades:
                by.setdefault(t['strategy'], []).append(t)
            for k, v in by.items():
                print(f"{k:>16}: {len(v)}건 | {np.mean([x['R'] for x in v]):+.2f}R | "
                      f"{sum(1 for x in v if x['pnl'] > 0) / len(v) * 100:.0f}%")
        print("\n※ 백테스트에는 AI 판단이 포함되지 않으며, 과거 성과는 미래를 보장하지 않습니다.")
        print("※ 평균 R > 0, 프로핏 팩터 > 1.3, MDD 가 감당 가능한 수준, 거래 100건 이상인지 확인하세요.")
    res["_trades"] = trades
    return res


def load_backtest_data(days):
    now_ms = exchange.milliseconds()
    day_ms = 86400000
    print(f"데이터 수집 중... ({days}일)")
    d15 = add_indicators(fetch_history('15m', now_ms - (days + 3) * day_ms)).reset_index(drop=True)
    h1 = add_indicators(fetch_history('1h', now_ms - (days + 15) * day_ms)).reset_index(drop=True)
    h4 = add_indicators(fetch_history('4h', now_ms - (days + 60) * day_ms)).reset_index(drop=True)
    return d15, h1, h4


def run_backtest(days):
    d15, h1, h4 = load_backtest_data(days)
    simulate(d15, h1, h4)


def run_compare(days):
    """5가지 설정을 같은 데이터로 한 번에 백테스트해서 표로 비교."""
    global INVERT, INVERT_MODE, MANAGE
    saved = (INVERT, INVERT_MODE, MANAGE)
    d15, h1, h4 = load_backtest_data(days)

    configs = [
        ("A. 원본 v4 (반전X)", False, "same", True),
        ("B. 반전 same", True, "same", True),
        ("C. 반전 mirror", True, "mirror", False),
        ("D. 반전 far_tp", True, "far_tp", False),
        ("E. 반전 far_tp+관리", True, "far_tp", True),
    ]
    rows, detail = [], {}
    try:
        for name, inv, mode, manage in configs:
            INVERT, INVERT_MODE, MANAGE = inv, mode, manage
            print(f">>> 실행 중: {name}")
            res = simulate(d15, h1, h4, verbose=False)
            detail[name] = res.pop("_trades")
            rows.append({"설정": name, **res})
    finally:
        INVERT, INVERT_MODE, MANAGE = saved

    table = pd.DataFrame(rows).set_index("설정")
    cols = ["trades", "win_rate", "avg_R", "avg_win_R", "avg_loss_R", "profit_factor",
            "return_pct", "max_dd_pct", "final_equity"]
    table = table[[c for c in cols if c in table.columns]]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(f"\n===== 비교 결과 ({days}일, {BT_LEVERAGE}x, 증거금 {MARGIN_FRACTION:.0%}, "
          f"far_tp: SL {FAR_SL_ATR}ATR / TP {FAR_TP_R}R) =====")
    print(table.round(2).to_string())

    print("\n-- 전략별 평균 R --")
    for name, trades in detail.items():
        by = {}
        for t in trades:
            by.setdefault(t['strategy'], []).append(t['R'])
        line = " | ".join(f"{k}: {sum(v) / len(v):+.2f}R ({len(v)}건)" for k, v in by.items()) or "거래 없음"
        print(f"{name:<22} {line}")

    print("\n-- 판단 기준 --")
    print("  avg_R > 0, profit_factor > 1.3, 거래 100건 이상 -> 검토할 가치가 있음")
    print("  모든 설정이 avg_R < 0 이면 신호 엔진 자체에 우위가 없는 것입니다.")
    print("  90/180/365일 모두에서 일관되게 좋은지 확인하세요. 한 기간만 좋으면 우연일 수 있습니다.")
    table.round(4).to_csv("compare_backtest_result.csv", encoding="utf-8-sig")
    print("\n결과 저장: compare_backtest_result.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, metavar="DAYS", help="현재 설정으로 최근 N일 백테스트 (180 이상 권장)")
    ap.add_argument("--compare", type=int, metavar="DAYS", help="5가지 설정을 한 번에 비교 백테스트")
    ap.add_argument("--dry", action="store_true", help="신호만 기록(주문 없음)")
    args = ap.parse_args()
    if args.compare:
        run_compare(args.compare)
    elif args.backtest:
        run_backtest(args.backtest)
    else:
        main_live(dry=args.dry)