"""
1시간봉 추세 눌림목 봇  (워크포워드 검증에서 반복 선택된 규칙 반영)
=====================================================================
규칙 (optimize_walkforward.py 결과에서 4개 폴드 중 3~4번 선택된 설정)
  - 1시간봉 종가 기준, 다음 봉 시가에 시장가 진입
  - LONG : 종가 > EMA50 > EMA200 (상승추세) 이고 ADX >= 25 이고 RSI < 45 (눌림)
  - SHORT: 종가 < EMA50 < EMA200 (하락추세) 이고 ADX >= 25 이고 RSI > 55 (되돌림)
  - 손절 = 1.5 x ATR, 익절 = 4.0 x ATR (손익비 약 2.7), 최대 보유 48시간 후 시간청산
  - 거래당 손실 한도 = 자본의 0.5% (고정 위험 사이징), 레버리지 최대 3배

!! 중요 !!
  이 규칙은 아직 '입증'되지 않았습니다. 검증 구간 합산이 32건, 평균 R +0.07, PF 1.10 으로
  판정 기준(100건 이상, 평균 R > 0.1, PF > 1.3)에 못 미쳤습니다. 그래서 기본 모드는 '모의매매'입니다.
  모의매매는 실제 주문 없이 실시간 신호를 기록하고, 이후 캔들로 SL/TP 결과를 계산해 누적합니다.
  이것이 과거 데이터로는 만들 수 없는 '진짜 처음 보는 데이터' 검증입니다.

실행
  python bitcoinv5.py --backtest 1095     # 3년 백테스트 + 전반/후반 분할 + 민감도 표 (API 키 불필요)
  python bitcoinv5.py                     # 모의매매 (기본, 주문 없음, API 키 불필요)
  python bitcoinv5.py --stats             # 누적 모의/실거래 성적 보기
  python bitcoinv5.py --live              # 실거래 (확인 문구 입력 필요)
  python bitcoinv5.py --backtest 365 --synthetic   # 무작위 가짜 데이터로 동작 점검

실거래 전 권장 기준
  모의매매 30건 이상 누적 + 평균 R > 0 + 3년 백테스트에서 전반/후반 모두 플러스 -> 그때 소액 실거래
"""
import os
import math
import time
import json
import sqlite3
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ===================== 설정 =====================
SYMBOL = "BTC/USDT:USDT"
TF = "1h"
DB_FILE = "pullback_bot.db"

# --- 전략 파라미터 (워크포워드 결과) ---
ADX_MIN = 25
RSI_LO = 45                 # 롱: RSI < 45 / 숏: RSI > 100 - 45 = 55
SL_ATR = 1.5
TP_ATR = 4.0
HOLD_BARS = 48              # 최대 보유 봉 수 (1시간봉 -> 48시간)

# --- 위험 관리 ---
RISK_PER_TRADE = 0.005      # 거래당 최대 손실 = 자본의 0.5%
MAX_LEVERAGE = 3
MAX_NOTIONAL_USD = float(os.getenv("MAX_NOTIONAL_USD", "0"))   # 0 이면 해제. 처음엔 300 같은 상한 권장
DAILY_LOSS_LIMIT = 0.03     # 하루 -3% 도달 시 당일 중단
MAX_CONSEC_LOSSES = 3
COOLDOWN_HOURS = 24
MIN_NOTIONAL = 100
DRY_EQUITY = float(os.getenv("DRY_EQUITY", "1000"))   # 백테스트/모의매매 기준 자본

# --- 비용 ---
FEE = 0.0005                # 편도 수수료
SLIP = 0.0002               # 편도 슬리피지

_ex = None


def get_ex(auth=False):
    """거래소 객체 (지연 생성). 데이터 조회만 할 때는 API 키가 필요 없습니다."""
    global _ex
    import ccxt
    if _ex is None or (auth and not _ex.apiKey):
        cfg = {'enableRateLimit': True, 'options': {'defaultType': 'future', 'fetchCurrencies': False}}
        if auth:
            cfg['apiKey'] = os.getenv("BINANCE_API_KEY")
            cfg['secret'] = os.getenv("BINANCE_SECRET_KEY")
            cfg['options']['portfolioMargin'] = True
        _ex = ccxt.binance(cfg)
    return _ex


class ProtectionError(Exception):
    pass


# ===================== 지표 / 신호 =====================
def indicators(df):
    d = df.copy()
    c, h, l = d['close'], d['high'], d['low']
    d['ema50'] = c.ewm(span=50, adjust=False).mean()
    d['ema200'] = c.ewm(span=200, adjust=False).mean()
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
    pdi = 100 * plus.ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * minus.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d['adx'] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
    return d


def signal_array(d, adx_min=ADX_MIN, rsi_lo=RSI_LO):
    """각 봉 종가 기준 신호: +1 롱, -1 숏, 0 없음."""
    c, rsi, adx = d['close'].values, d['rsi'].values, d['adx'].values
    e50, e200, atr = d['ema50'].values, d['ema200'].values, d['atr'].values
    valid = np.isfinite(atr) & np.isfinite(e200) & (np.arange(len(d)) >= 210)
    up = (c > e50) & (e50 > e200)
    dn = (c < e50) & (e50 < e200)
    lg = up & (adx >= adx_min) & (rsi < rsi_lo)
    st = dn & (adx >= adx_min) & (rsi > 100 - rsi_lo)
    sig = np.where(lg, 1, np.where(st, -1, 0))
    return np.where(valid, sig, 0)


# ===================== 거래 결과 계산 (백테스트/모의매매 공용) =====================
def r_multiple(dr, entry, exit_p, sl_d):
    """수수료(왕복) 포함 R 배수."""
    return (dr * (exit_p - entry) - 2 * FEE * entry) / sl_d


def simulate_R(o, h, l, c, atr, sig, sl_atr, tp_atr, hold, lo=0, hi=None):
    """겹치는 포지션 없이 한 번에 하나씩. 같은 봉에서 SL/TP 동시 터치 시 SL 우선(보수적).
    반환: [(진입봉 인덱스, 방향, R), ...]"""
    hi = hi or len(o)
    out, nxt = [], lo
    for i in np.nonzero(sig)[0]:
        if i < nxt or i < lo or i + 1 >= hi:
            continue
        dr = int(sig[i])
        e = i + 1
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = o[e] * (1 + dr * SLIP)
        sl_d, tp_d = sl_atr * a, tp_atr * a
        sl, tp = entry - dr * sl_d, entry + dr * tp_d
        end = min(e + hold, hi - 1)
        seg_h, seg_l = h[e:end + 1], l[e:end + 1]
        if dr == 1:
            sl_hit, tp_hit = seg_l <= sl, seg_h >= tp
        else:
            sl_hit, tp_hit = seg_h >= sl, seg_l <= tp
        s = int(sl_hit.argmax()) if sl_hit.any() else 10 ** 9
        t = int(tp_hit.argmax()) if tp_hit.any() else 10 ** 9
        if s <= t and s < 10 ** 9:
            exit_p, k = sl * (1 - dr * SLIP), s
        elif t < 10 ** 9:
            exit_p, k = tp, t
        else:
            k = end - e
            exit_p = c[end] * (1 - dr * SLIP)
        out.append((e, dr, r_multiple(dr, entry, exit_p, sl_d)))
        nxt = e + k
    return out


def stats(rs):
    n = len(rs)
    if n == 0:
        return dict(n=0, wr=0.0, avg_R=0.0, pf=0.0, tot_R=0.0)
    a = np.array(rs)
    w, ls = a[a > 0].sum(), -a[a <= 0].sum()
    return dict(n=n, wr=float((a > 0).mean() * 100), avg_R=float(a.mean()),
                pf=float(w / ls) if ls > 0 else float('inf'), tot_R=float(a.sum()))


def fmt(m):
    pf = f"{m['pf']:.2f}" if np.isfinite(m['pf']) else "inf"
    return f"n={m['n']:>3} 승률={m['wr']:5.1f}% 평균R={m['avg_R']:+.3f} PF={pf:>5} 총R={m['tot_R']:+7.1f}"


# ===================== 데이터 =====================
def fetch_history(days):
    ex = get_ex()
    since = ex.milliseconds() - (days + 12) * 86400000     # 지표 워밍업 약 210봉 포함
    step = ex.parse_timeframe(TF) * 1000
    out = []
    print(f"데이터 수집 중... ({days}일, {TF})")
    while True:
        b = ex.fetch_ohlcv(SYMBOL, TF, since=since, limit=1000)
        if not b:
            break
        out += b
        since = b[-1][0] + step
        if len(b) < 1000:
            break
    df = pd.DataFrame(out, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.iloc[:-1].reset_index(drop=True)


def synthetic(days, seed=11):
    n = (days + 12) * 24
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.0035, n)
    c = 30000 * np.exp(np.cumsum(r))
    o = np.concatenate([[30000], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    ts = pd.date_range('2023-01-01', periods=n, freq='1h')
    return pd.DataFrame({'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': c,
                         'volume': rng.lognormal(5, 0.5, n)})


# ===================== 백테스트 =====================
def backtest(days, use_synthetic=False):
    df = synthetic(days) if use_synthetic else fetch_history(days)
    d = indicators(df)
    arr = tuple(d[k].values for k in ('open', 'high', 'low', 'close', 'atr'))
    n = len(d)
    sig = signal_array(d)
    tr = simulate_R(*arr, sig, SL_ATR, TP_ATR, HOLD_BARS)
    rs = [x[2] for x in tr]

    print(f"\n===== 백테스트: {d['timestamp'].iloc[210]:%Y-%m-%d} ~ {d['timestamp'].iloc[-1]:%Y-%m-%d} "
          f"({TF}, ADX>={ADX_MIN}, RSI<{RSI_LO}, SL {SL_ATR}ATR, TP {TP_ATR}ATR, 보유 {HOLD_BARS}봉) =====")
    print("전체       ", fmt(stats(rs)))

    eq, peak, mdd = DRY_EQUITY, DRY_EQUITY, 0.0
    for r in rs:
        eq *= 1 + RISK_PER_TRADE * r
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    print(f"자본곡선    {DRY_EQUITY:,.0f} -> {eq:,.0f} USDT ({(eq / DRY_EQUITY - 1) * 100:+.1f}%), "
          f"최대낙폭 {mdd * 100:.1f}%  (거래당 위험 {RISK_PER_TRADE * 100:.1f}% 가정, 레버리지 제한 미반영)")

    print("\n-- 방향별 --")
    print("롱         ", fmt(stats([x[2] for x in tr if x[1] == 1])))
    print("숏         ", fmt(stats([x[2] for x in tr if x[1] == -1])))

    print("\n-- 시기별 일관성 (4등분) --")
    edges = np.linspace(210, n, 5).astype(int)
    for q in range(4):
        part = [x[2] for x in tr if edges[q] <= x[0] < edges[q + 1]]
        print(f"{d['timestamp'].iloc[edges[q]]:%Y-%m-%d}~{d['timestamp'].iloc[edges[q + 1] - 1]:%Y-%m-%d} ", fmt(stats(part)))

    print("\n-- 민감도: 파라미터를 조금 바꿔도 결과가 유지되는가 (현재 설정은 *) --")
    print(f"{'ADX':>4} {'RSI':>4} {'SL':>4} {'TP':>4} | 결과")
    pos = tot = 0
    for adx_min in (20, 25):
        for rsi_lo in (45, 50):
            sg = signal_array(d, adx_min, rsi_lo)
            for sl in (1.0, 1.5, 2.0):
                for tp in (3.0, 4.0):
                    m = stats([x[2] for x in simulate_R(*arr, sg, sl, tp, HOLD_BARS)])
                    mark = "*" if (adx_min, rsi_lo, sl, tp) == (ADX_MIN, RSI_LO, SL_ATR, TP_ATR) else " "
                    print(f"{adx_min:>4} {rsi_lo:>4} {sl:>4} {tp:>4} |{mark}{fmt(m)}")
                    tot += 1
                    pos += m['avg_R'] > 0
    print(f"\n평균 R > 0 인 조합: {pos}/{tot}")

    m_all = stats(rs)
    q_pos = sum(1 for q in range(4) if stats([x[2] for x in tr if edges[q] <= x[0] < edges[q + 1]])['avg_R'] > 0)
    ok = m_all['n'] >= 100 and m_all['avg_R'] > 0.1 and m_all['pf'] > 1.3 and q_pos >= 3 and pos / tot >= 0.7
    print("\n===== 판정 =====")
    if ok:
        print("기준 통과 (100건 이상, 평균 R > 0.1, PF > 1.3, 4구간 중 3구간 이상 플러스, 민감도 70% 이상).")
        print("그래도 다음 단계는 모의매매 -> 소액 실거래 순서입니다.")
    else:
        print("기준 미통과: 100건 이상 / 평균 R > 0.1 / PF > 1.3 / 4구간 중 3구간 이상 플러스 / 민감도 조합 70% 이상.")
        print("이 경우 실거래하지 말고 모의매매로 계속 관찰하거나 규칙을 폐기하세요.")
    print("※ 이 백테스트는 같은 데이터로 규칙을 골랐을 가능성이 있어 낙관적일 수 있습니다.")


# ===================== DB =====================
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    conn = db()
    conn.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, mode TEXT NOT NULL, side TEXT NOT NULL,
        entry_bar_ts TEXT NOT NULL, entry_price REAL NOT NULL, sl_price REAL NOT NULL, tp_price REAL NOT NULL,
        sl_dist REAL NOT NULL, qty REAL, leverage INTEGER, risk_amt REAL,
        status TEXT DEFAULT 'OPEN', exit_time TEXT, exit_price REAL, exit_reason TEXT,
        r_multiple REAL, pnl REAL, sl_order_id TEXT, tp_order_id TEXT, opened_at TEXT)''')
    conn.commit()
    conn.close()


def open_trade(mode):
    conn = db()
    r = conn.execute("SELECT * FROM trades WHERE mode=? AND status='OPEN' ORDER BY id DESC LIMIT 1", (mode,)).fetchone()
    conn.close()
    return dict(r) if r else None


def print_stats(mode):
    conn = db()
    rows = conn.execute("SELECT r_multiple, pnl, side FROM trades WHERE mode=? AND status='CLOSED' "
                        "AND r_multiple IS NOT NULL", (mode,)).fetchall()
    conn.close()
    if not rows:
        print(f"[{mode}] 완료된 거래 없음")
        return
    rs = [r['r_multiple'] for r in rows]
    print(f"[{mode} 누적] {fmt(stats(rs))}")
    if len(rs) < 30:
        print(f"   (표본 {len(rs)}건: 30건 미만이라 우위 여부를 판단하기엔 이릅니다)")


# ===================== 모의매매 =====================
def open_paper(sig, forming_open, forming_ts):
    dr = sig['dir']
    entry = forming_open * (1 + dr * SLIP)
    sl = entry - dr * sig['sl_dist']
    tp = entry + dr * sig['tp_dist']
    conn = db()
    conn.execute('''INSERT INTO trades (mode, side, entry_bar_ts, entry_price, sl_price, tp_price, sl_dist, opened_at)
                    VALUES ('paper', ?, ?, ?, ?, ?, ?, ?)''',
                 ('long' if dr == 1 else 'short', forming_ts.isoformat(), entry, sl, tp, sig['sl_dist'],
                  datetime.now().isoformat()))
    conn.commit()
    conn.close()
    print(f"[모의] {'LONG' if dr == 1 else 'SHORT'} 진입 {entry:,.1f} | SL {sl:,.1f} | TP {tp:,.1f}")


def resolve_paper(closed_df):
    """열려 있는 모의 거래를 이후 마감 캔들로 판정."""
    tr = open_trade('paper')
    if not tr:
        return
    dr = 1 if tr['side'] == 'long' else -1
    entry_ts = pd.Timestamp(tr['entry_bar_ts'])
    bars = closed_df[closed_df['timestamp'] >= entry_ts].reset_index(drop=True)
    for k in range(len(bars)):
        b = bars.iloc[k]
        sl_hit = b['low'] <= tr['sl_price'] if dr == 1 else b['high'] >= tr['sl_price']
        tp_hit = b['high'] >= tr['tp_price'] if dr == 1 else b['low'] <= tr['tp_price']
        exit_p = reason = None
        if sl_hit:
            exit_p, reason = tr['sl_price'] * (1 - dr * SLIP), 'SL'
        elif tp_hit:
            exit_p, reason = tr['tp_price'], 'TP'
        elif k >= HOLD_BARS:
            exit_p, reason = b['close'] * (1 - dr * SLIP), 'TIME'
        if exit_p is not None:
            r = r_multiple(dr, tr['entry_price'], exit_p, tr['sl_dist'])
            conn = db()
            conn.execute("UPDATE trades SET status='CLOSED', exit_time=?, exit_price=?, exit_reason=?, r_multiple=? "
                         "WHERE id=?", (b['timestamp'].isoformat(), exit_p, reason, r, tr['id']))
            conn.commit()
            conn.close()
            print(f"[모의] 종료 #{tr['id']} {reason} | {r:+.2f}R")
            print_stats('paper')
            return


# ===================== 실거래 =====================
def get_equity():
    bal = get_ex(True).fetch_balance()
    for key in ('total', 'free'):
        part = bal.get(key, {}) or {}
        v = sum(float(part.get(c) or 0) for c in ('USDT', 'USDC'))
        if v > 0:
            return v
    raise RuntimeError("잔고 조회 실패")


def get_position():
    for pos in get_ex(True).fetch_positions([SYMBOL]):
        if pos['symbol'] == SYMBOL:
            amt = float(pos['info']['positionAmt'])
            if amt > 0:
                return 'long', amt
            if amt < 0:
                return 'short', abs(amt)
    return None, 0.0


def px(v):
    return float(get_ex(True).price_to_precision(SYMBOL, v))


def cancel_all_orders():
    ex = get_ex(True)
    try:
        for o in ex.fetch_open_orders(SYMBOL):
            ex.cancel_order(o['id'], SYMBOL)
        for o in ex.fetch_open_orders(SYMBOL, params={'stop': True}):
            ex.cancel_order(o['id'], SYMBOL, params={'stop': True})
    except Exception as e:
        print("주문 취소 오류:", e)


def close_market(side, qty):
    get_ex(True).create_order(SYMBOL, 'market', 'sell' if side == 'long' else 'buy', qty, None, {'reduceOnly': True})


def size_position(equity, entry, sl_dist):
    risk = equity * RISK_PER_TRADE
    notional = risk / sl_dist * entry
    cap = equity * MAX_LEVERAGE * 0.9
    if MAX_NOTIONAL_USD:
        cap = min(cap, MAX_NOTIONAL_USD)
    notional = min(notional, cap)
    if notional < MIN_NOTIONAL:
        return None
    qty = math.floor(notional / entry * 1000) / 1000
    if qty * entry < MIN_NOTIONAL:
        qty = round(qty + 0.001, 3)
    if qty <= 0:
        return None
    lev = int(min(MAX_LEVERAGE, max(1, math.ceil(qty * entry / equity))))
    return {"qty": qty, "leverage": lev, "risk_amt": qty * sl_dist}


def risk_gate(equity):
    today = datetime.now().strftime('%Y-%m-%d')
    conn = db()
    day = conn.execute("SELECT COALESCE(SUM(pnl),0) s FROM trades WHERE mode='live' AND status='CLOSED' "
                       "AND substr(exit_time,1,10)=?", (today,)).fetchone()['s']
    last = conn.execute("SELECT pnl, exit_time FROM trades WHERE mode='live' AND status='CLOSED' "
                        "ORDER BY id DESC LIMIT ?", (MAX_CONSEC_LOSSES,)).fetchall()
    conn.close()
    if day / max(equity - day, 1e-9) <= -DAILY_LOSS_LIMIT:
        return False, f"일일 손실 한도 도달 ({day:+.2f} USDT)"
    if len(last) == MAX_CONSEC_LOSSES and all((x['pnl'] or 0) < 0 for x in last):
        until = datetime.fromisoformat(last[0]['exit_time']) + timedelta(hours=COOLDOWN_HOURS)
        if datetime.now() < until:
            return False, f"{MAX_CONSEC_LOSSES}연패 쿨다운 ~{until:%m-%d %H:%M}"
    return True, ""


def open_live(sig, forming_open, forming_ts):
    ex = get_ex(True)
    equity = get_equity()
    ok, why = risk_gate(equity)
    if not ok:
        print("[게이트]", why)
        return
    dr = sig['dir']
    side = 'long' if dr == 1 else 'short'
    pos = size_position(equity, forming_open, sig['sl_dist'])
    if not pos:
        print("사이징 불가(잔고/최소주문) - 건너뜀")
        return
    try:
        ex.papiPostUmLeverage({'symbol': ex.market(SYMBOL)['id'], 'leverage': pos['leverage']})
    except Exception as e:
        raise RuntimeError(f"레버리지 설정 실패, 진입 취소: {e}")

    order = ex.create_order(SYMBOL, 'market', 'buy' if dr == 1 else 'sell', pos['qty'])
    entry = float(order.get('average') or order.get('price') or forming_open)
    sl, tp = px(entry - dr * sig['sl_dist']), px(entry + dr * sig['tp_dist'])
    close_side = 'sell' if dr == 1 else 'buy'
    try:
        o1 = ex.create_order(SYMBOL, 'STOP_MARKET', close_side, pos['qty'], None, {'stopPrice': sl, 'reduceOnly': True})
        o2 = ex.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', close_side, pos['qty'], None,
                             {'stopPrice': tp, 'reduceOnly': True})
    except Exception as e:
        print("SL/TP 주문 실패, 포지션 즉시 청산:", e)
        try:
            close_market(side, pos['qty'])
        except Exception as e2:
            print("!!! 청산도 실패. 바이낸스 앱에서 직접 확인하세요:", e2)
        raise ProtectionError("보호 주문 실패로 진입 취소")

    conn = db()
    conn.execute('''INSERT INTO trades (mode, side, entry_bar_ts, entry_price, sl_price, tp_price, sl_dist, qty,
                    leverage, risk_amt, sl_order_id, tp_order_id, opened_at)
                    VALUES ('live', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (side, forming_ts.isoformat(), entry, sl, tp, sig['sl_dist'], pos['qty'], pos['leverage'],
                  pos['risk_amt'], o1.get('id'), o2.get('id'), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    print(f"\n=== {side.upper()} 진입 | {entry:,.1f} | SL {sl:,.1f} | TP {tp:,.1f} | {pos['leverage']}x | "
          f"{pos['qty']} BTC | 예상 최대손실 {pos['risk_amt']:.2f} USDT ({pos['risk_amt'] / equity * 100:.2f}%) ===")


def finalize_live(tr):
    ex = get_ex(True)
    dr = 1 if tr['side'] == 'long' else -1
    since = int(datetime.fromisoformat(tr['opened_at']).timestamp() * 1000) - 5000
    pnl, exit_price = None, None
    try:
        ts = ex.fetch_my_trades(SYMBOL, since=since)
        if ts and any('realizedPnl' in (t.get('info') or {}) for t in ts):
            pnl = sum(float(t['info'].get('realizedPnl') or 0) for t in ts)
            pnl -= sum(float((t.get('fee') or {}).get('cost') or 0) for t in ts)
            closing = [t for t in ts if t['side'] == ('sell' if dr == 1 else 'buy')]
            if closing:
                exit_price = sum(t['price'] * t['amount'] for t in closing) / sum(t['amount'] for t in closing)
    except Exception as e:
        print("체결내역 조회 실패(추정치 사용):", e)
    if exit_price is None:
        exit_price = ex.fetch_ticker(SYMBOL)['last']
    if pnl is None:
        pnl = dr * (exit_price - tr['entry_price']) * tr['qty'] - 2 * FEE * tr['entry_price'] * tr['qty']
    r = pnl / tr['risk_amt'] if tr['risk_amt'] else None
    conn = db()
    conn.execute("UPDATE trades SET status='CLOSED', exit_time=?, exit_price=?, exit_reason='closed', "
                 "r_multiple=?, pnl=? WHERE id=?",
                 (datetime.now().isoformat(), exit_price, r, pnl, tr['id']))
    conn.commit()
    conn.close()
    print(f"\n=== 포지션 종료 #{tr['id']} | {tr['entry_price']:,.1f} -> {exit_price:,.1f} | "
          f"{pnl:+.2f} USDT" + (f" ({r:+.2f}R)" if r is not None else "") + " ===")
    print_stats('live')


def manage_live():
    tr = open_trade('live')
    side, amt = get_position()
    if tr and not side:
        if (datetime.now() - datetime.fromisoformat(tr['opened_at'])).total_seconds() < 20:
            return
        finalize_live(tr)
        cancel_all_orders()
    elif tr and side:
        age_h = (datetime.now() - datetime.fromisoformat(tr['opened_at'])).total_seconds() / 3600
        if age_h >= HOLD_BARS:
            print(f"시간 청산 ({age_h:.0f}시간 경과)")
            close_market(side, amt)
            cancel_all_orders()
    elif side and not tr:
        print("경고: DB에 없는 포지션이 있습니다. 이 봇은 자동 편입하지 않습니다. 바이낸스 앱에서 확인하세요.")


# ===================== 메인 루프 =====================
def latest_signal(d):
    sig = signal_array(d)
    i = len(d) - 1
    if sig[i] == 0:
        return None
    a = float(d['atr'].iloc[i])
    return {"dir": int(sig[i]), "atr": a, "sl_dist": SL_ATR * a, "tp_dist": TP_ATR * a,
            "rsi": float(d['rsi'].iloc[i]), "adx": float(d['adx'].iloc[i])}


def cycle(live):
    ex = get_ex(live)
    raw = ex.fetch_ohlcv(SYMBOL, TF, limit=500)
    forming_ts = pd.to_datetime(raw[-1][0], unit='ms')
    forming_open = float(raw[-1][1])
    closed = pd.DataFrame(raw[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    closed['timestamp'] = pd.to_datetime(closed['timestamp'], unit='ms')
    d = indicators(closed)
    price = float(closed['close'].iloc[-1])

    if not live:
        resolve_paper(closed)
    mode = 'live' if live else 'paper'
    if open_trade(mode):
        print(f"[{datetime.now():%m-%d %H:%M}] {price:,.1f} | 진행 중인 {mode} 거래 있음")
        return
    sig = latest_signal(d)
    if not sig:
        print(f"[{datetime.now():%m-%d %H:%M}] {price:,.1f} | 신호 없음 (ADX {d['adx'].iloc[-1]:.0f}, RSI {d['rsi'].iloc[-1]:.0f})")
        return
    print(f"[{datetime.now():%m-%d %H:%M}] 신호: {'LONG' if sig['dir'] == 1 else 'SHORT'} | "
          f"ADX {sig['adx']:.0f} RSI {sig['rsi']:.0f} ATR {sig['atr']:.1f}")
    if live:
        open_live(sig, forming_open, forming_ts)
    else:
        open_paper(sig, forming_open, forming_ts)


def main(live):
    setup_db()
    print("\n=== 1시간봉 눌림목 봇 ===", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"모드: {'실거래' if live else '모의매매(주문 없음)'} | ADX>={ADX_MIN}, RSI<{RSI_LO}/>{100 - RSI_LO}, "
          f"SL {SL_ATR}ATR, TP {TP_ATR}ATR, 보유 {HOLD_BARS}h | 거래당 위험 {RISK_PER_TRADE * 100:.1f}%, 최대 {MAX_LEVERAGE}x")
    if live:
        print("!! 이 규칙은 검증이 끝나지 않았습니다 (검증 구간 32건, PF 1.10). 잃을 수 있는 금액만 사용하세요.")
        if input("실거래를 시작하려면 YES 를 입력하세요: ").strip() != "YES":
            print("취소되었습니다.")
            return
    now = time.time()
    last_bucket = None if now % 3600 < 300 else int(now // 3600)   # 시작이 정각 직후가 아니면 이번 시간은 건너뜀
    while True:
        try:
            if live:
                manage_live()
            now = time.time()
            bucket = int(now // 3600)
            if bucket != last_bucket and now % 3600 >= 15:
                last_bucket = bucket
                cycle(live)
            time.sleep(30)
        except KeyboardInterrupt:
            print("종료")
            break
        except ProtectionError as e:
            print(f"치명적: {e} - 봇 중지. 바이낸스 앱에서 포지션을 확인하세요.")
            raise SystemExit(1)
        except Exception as e:
            print(f"오류: {e}")
            time.sleep(30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, metavar="DAYS", help="백테스트 (1095 = 3년 권장)")
    ap.add_argument("--synthetic", action="store_true", help="백테스트를 무작위 가짜 데이터로 실행")
    ap.add_argument("--live", action="store_true", help="실거래 (기본은 모의매매)")
    ap.add_argument("--stats", action="store_true", help="누적 성적 보기")
    args = ap.parse_args()
    if args.backtest:
        backtest(args.backtest, args.synthetic)
    elif args.stats:
        setup_db()
        print_stats('paper')
        print_stats('live')
    else:
        main(args.live)