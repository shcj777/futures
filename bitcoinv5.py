"""
멀티 코인 추세 추종 프로그램  (손실은 짧게, 이익은 크게 / 하루 여러 건 체결 목표)
=====================================================================
설계 원리
  - 한 종목만 보면 신호가 드뭅니다. 거래소 거래량 상위 N개 코인(기본 30개)을 동시에 감시해서
    '하루 5건 이상' 체결을 노립니다.
  - 진입: 20봉 고가/저가 돌파 + 추세 필터(EMA200) + ADX + 거래량  -> 다음 봉 시가에 진입
  - 손절: 진입가 - 1.5 ATR (고정, 짧게)
  - 익절: 고정 목표가 없음. 가격이 유리하게 움직이면 '고점 - 3 ATR' 추적 손절이 따라 올라감 (크게 이기기)
  - 최대 보유 72봉. 포지션 동시 보유 최대 6개, 거래당 위험 = 자본의 0.3%
  - 기대 구조: 승률 30~40%, 평균 이익 > 평균 손실 x 2

!! 중요 !!
  - 이 프로그램이 수익을 낸다는 보장은 없습니다. 지금까지 같은 계열(15분봉 BTC 단일 종목)은 비용 때문에
    손실이었고, 1시간봉 눌림목 규칙은 약한 가능성만 보였습니다. 그래서 백테스트에서
    '거래 수/일, 평균 R, 프로핏 팩터, 전반/후반 일관성, 비용 민감도, 파라미터 민감도'를 모두 보여주고,
    기준을 통과했는지 정직하게 판정합니다.
  - 백테스트 종목 목록은 '현재 거래량 상위'라서 과거 시점에는 없던 인기 코인이 섞일 수 있습니다(선택 편향).
  - 실거래 기능은 넣지 않았습니다. 백테스트가 기준을 통과하고 모의매매로 확인된 뒤에 붙이는 것이 순서입니다.

실행
  python bitcoinv5.py --backtest 180                 # 180일 백테스트 (API 키 불필요)
  python mbitcoinv5.py --backtest 180 --symbols 50    # 감시 종목을 50개로 늘려 거래 수 늘리기
  python bitcoinv5.py --backtest 180 --tf 30m        # 30분봉 (거래 수 증가, 비용 부담도 증가)
  python bitcoinv5.py                                # 모의매매 (주문 없음, 매시 정각 직후 점검)
  python bitcoinv5.py --stats                        # 모의매매 누적 성적
  python bitcoinv5.py --backtest 180 --synthetic     # 무작위 가짜 데이터로 동작 점검
"""
import os
import time
import argparse
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

# ===================== 설정 =====================
TF = "1h"
DB_FILE = "multi_trend_paper.db"

P = dict(don=20, adx=20, vz=0.5, sl_atr=1.5, trail_atr=3.0, hold=72)   # 전략 파라미터 (고정, 튜닝 X)

RISK_PER_TRADE = 0.003      # 거래당 최대 손실 = 자본의 0.3%
MAX_POS = 6                 # 동시 보유 최대 개수
N_SYMBOLS = 30
FEE = 0.0005                # 편도 수수료 (시장가)
SLIP = 0.0002               # 편도 슬리피지
EQ0 = 1000.0

FALLBACK = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC", "DOT", "TRX", "BCH", "ATOM",
            "NEAR", "APT", "ARB", "OP", "SUI", "INJ", "FIL", "ETC", "AAVE", "UNI", "TON", "SEI", "TIA", "WLD",
            "FET", "RUNE"]
TF_MIN = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}

_ex = None


def get_ex():
    global _ex
    if _ex is None:
        import ccxt
        _ex = ccxt.binance({'enableRateLimit': True,
                            'options': {'defaultType': 'future', 'fetchCurrencies': False}})
    return _ex


# ===================== 지표 / 신호 =====================
def indicators(df, don=20):
    d = df.copy()
    c, h, l, v = d['close'], d['high'], d['low'], d['volume']
    d['ema200'] = c.ewm(span=200, adjust=False).mean()
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
    d['don_hi'] = h.rolling(don).max().shift(1)
    d['don_lo'] = l.rolling(don).min().shift(1)
    d['vol_z'] = ((v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)).fillna(0)
    return d


def signal_array(d, p):
    """봉 종가 기준 신호: +1 롱, -1 숏, 0 없음 (진입은 다음 봉 시가)."""
    c = d['close'].values
    ok = (d['adx'].values >= p['adx']) & (d['vol_z'].values >= p['vz'])
    lg = (c > d['don_hi'].values) & (c > d['ema200'].values) & ok
    st = (c < d['don_lo'].values) & (c < d['ema200'].values) & ok
    sig = np.where(lg, 1, np.where(st, -1, 0))
    valid = np.isfinite(d['atr'].values) & np.isfinite(d['ema200'].values) & np.isfinite(d['don_hi'].values) \
        & (np.arange(len(d)) >= 210)
    return np.where(valid, sig, 0)


# ===================== 한 건의 거래 진행 (백테스트/모의매매 공용) =====================
def run_trade(o, h, l, c, e, dr, a, p, entry=None, force_close=False):
    """e: 진입 봉 인덱스, dr: +1/-1, a: 신호 시점 ATR.
    손절은 고정 거리로 시작해서, 유리한 방향의 극값(고점/저점)에서 trail_atr x ATR 만큼 떨어진 곳으로만 이동.
    같은 봉 안에서는 '이전 봉까지의 정보로 정한 손절가'를 먼저 검사 (미래 정보 사용 방지).
    반환: (종료 봉 인덱스, 종료가, 사유, R) 또는 아직 진행 중이면 None."""
    if entry is None:
        entry = o[e] * (1 + dr * SLIP)
    sl_d = p['sl_atr'] * a
    stop = entry - dr * sl_d
    ext = entry
    last = len(o) - 1
    end = e + p['hold']
    for j in range(e, min(end, last) + 1):
        px_ = None
        if dr == 1:
            if l[j] <= stop:
                px_, why = min(stop, o[j]), 'stop'
        else:
            if h[j] >= stop:
                px_, why = max(stop, o[j]), 'stop'
        if px_ is None and j == end:
            px_, why = c[j], 'time'
        if px_ is not None:
            px_ = px_ * (1 - dr * SLIP)
            return j, px_, why, (dr * (px_ - entry) - 2 * FEE * entry) / sl_d
        ext = max(ext, h[j]) if dr == 1 else min(ext, l[j])
        trail = ext - dr * p['trail_atr'] * a
        stop = max(stop, trail) if dr == 1 else min(stop, trail)
    if force_close:
        px_ = c[last] * (1 - dr * SLIP)
        return last, px_, 'end', (dr * (px_ - entry) - 2 * FEE * entry) / sl_d
    return None


def sim_symbol(sym, d, p):
    o, h, l, c, atr = (d[k].values for k in ('open', 'high', 'low', 'close', 'atr'))
    ts = d['timestamp']
    sig = signal_array(d, p)
    out, nxt = [], 0
    for i in np.nonzero(sig)[0]:
        if i < nxt or i + 1 >= len(o):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        dr = int(sig[i])
        res = run_trade(o, h, l, c, i + 1, dr, a, p, force_close=True)
        j, _, why, r = res
        out.append(dict(sym=sym, dir=dr, entry_ts=ts.iloc[i + 1], exit_ts=ts.iloc[j], R=r, why=why,
                        sl_pct=p['sl_atr'] * a / o[i + 1]))
        nxt = j
    return out


# ===================== 포트폴리오 (동시 보유 제한, 고정 위험 사이징) =====================
def portfolio(trades, eq0=EQ0, risk=RISK_PER_TRADE, max_pos=MAX_POS):
    trades = sorted(trades, key=lambda t: t['entry_ts'])
    eq = peak = eq0
    mdd = 0.0
    open_pos, acc = [], []
    skipped = 0

    def settle(upto):
        nonlocal eq, peak, mdd, open_pos
        done = sorted([x for x in open_pos if x['exit_ts'] <= upto], key=lambda x: x['exit_ts'])
        for x in done:
            eq += x['risk_amt'] * x['R']
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak)
        open_pos = [x for x in open_pos if x['exit_ts'] > upto]

    for t in trades:
        settle(t['entry_ts'])
        if len(open_pos) >= max_pos:
            skipped += 1
            continue
        x = dict(t)
        x['risk_amt'] = eq * risk
        open_pos.append(x)
        acc.append(x)
    settle(pd.Timestamp.max)
    return eq, mdd, acc, skipped


def stats(rs):
    n = len(rs)
    if n == 0:
        return dict(n=0, wr=0.0, avg_R=0.0, pf=0.0, tot_R=0.0, win_R=0.0, loss_R=0.0)
    a = np.array(rs)
    w, ls = a[a > 0], a[a <= 0]
    return dict(n=n, wr=float((a > 0).mean() * 100), avg_R=float(a.mean()),
                pf=float(w.sum() / -ls.sum()) if len(ls) and ls.sum() < 0 else float('inf'),
                tot_R=float(a.sum()), win_R=float(w.mean()) if len(w) else 0.0,
                loss_R=float(ls.mean()) if len(ls) else 0.0)


def fmt(m):
    pf = f"{m['pf']:.2f}" if np.isfinite(m['pf']) else "inf"
    return f"n={m['n']:>4} 승률={m['wr']:5.1f}% 평균R={m['avg_R']:+.3f} PF={pf:>5} 총R={m['tot_R']:+7.1f}"


# ===================== 데이터 =====================
def get_universe(n):
    try:
        ex = get_ex()
        ex.load_markets()
        tk = ex.fetch_tickers()
        cands = []
        for s, t in tk.items():
            m = ex.markets.get(s)
            if not m or not m.get('swap') or not m.get('linear') or m.get('quote') != 'USDT' or m.get('active') is False:
                continue
            cands.append((t.get('quoteVolume') or 0, s))
        cands.sort(reverse=True)
        syms = [s for _, s in cands[:n]]
        if syms:
            return syms
    except Exception as e:
        print("종목 목록 조회 실패, 기본 목록 사용:", e)
    return [f"{b}/USDT:USDT" for b in FALLBACK[:n]]


def fetch_history(sym, days, tf):
    ex = get_ex()
    step = ex.parse_timeframe(tf) * 1000
    since = ex.milliseconds() - int((days + 12) * 86400000)
    out = []
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
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


def synthetic_universe(n, days, tf, seed=5):
    mins = TF_MIN[tf]
    nb = int((days + 12) * 24 * 60 / mins)
    rng = np.random.default_rng(seed)
    ts = pd.date_range('2025-01-01', periods=nb, freq=f'{mins}min')
    sigma = 0.0035 * np.sqrt(mins / 60)
    common = rng.normal(0, sigma * 0.5, nb)
    data = {}
    for k in range(n):
        r = common + rng.normal(0, sigma, nb)
        c = 100 * np.exp(np.cumsum(r))
        o = np.concatenate([[100], c[:-1]])
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, sigma * 0.5, nb)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, sigma * 0.5, nb)))
        data[f"SYN{k}/USDT:USDT"] = pd.DataFrame({'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': c,
                                                  'volume': rng.lognormal(5, 0.5, nb)})
    return data


def load_data(days, tf, n_sym, use_synth):
    if use_synth:
        raw = synthetic_universe(n_sym, days, tf)
    else:
        syms = get_universe(n_sym)
        print(f"감시 종목 {len(syms)}개 데이터 수집 중... ({days}일, {tf})")
        raw = {}
        need = (days + 12) * 24 * 60 / TF_MIN[tf]
        for k, s in enumerate(syms, 1):
            try:
                df = fetch_history(s, days, tf)
                if len(df) < need * 0.5:
                    print(f"  [{k}/{len(syms)}] {s}: 데이터 부족({len(df)}봉), 제외")
                    continue
                raw[s] = df
            except Exception as e:
                print(f"  [{k}/{len(syms)}] {s}: 수집 실패 ({e})")
    return {s: indicators(df, P['don']) for s, df in raw.items()}


# ===================== 백테스트 =====================
def run_all(data, p):
    out = []
    for s, d in data.items():
        out += sim_symbol(s, d, p)
    return out


def backtest(days, tf, n_sym, use_synth):
    data = load_data(days, tf, n_sym, use_synth)
    if not data:
        print("사용 가능한 데이터가 없습니다.")
        return
    trades = run_all(data, P)
    if not trades:
        print("신호가 없습니다.")
        return
    t0 = min(t['entry_ts'] for t in trades)
    t1 = max(t['exit_ts'] for t in trades)
    span_days = max(1.0, (t1 - t0).total_seconds() / 86400)

    eq, mdd, acc, skipped = portfolio(trades)
    m = stats([t['R'] for t in acc])
    per_day = len(acc) / span_days

    print(f"\n===== 멀티 코인 추세 추종 백테스트: {t0:%Y-%m-%d} ~ {t1:%Y-%m-%d} ({span_days:.0f}일), "
          f"{len(data)}종목, {tf} =====")
    print(f"전략: {P['don']}봉 돌파 + EMA200 + ADX>={P['adx']} + 거래량 z>={P['vz']} | 손절 {P['sl_atr']}ATR, "
          f"추적 {P['trail_atr']}ATR, 최대 {P['hold']}봉 | 거래당 위험 {RISK_PER_TRADE * 100:.1f}%, 동시 {MAX_POS}개")
    print(f"\n신호 발생 {len(trades)}건 -> 체결 {len(acc)}건 (동시 보유 제한으로 {skipped}건 제외)")
    print(f"하루 평균 체결: {per_day:.2f}건")
    print("전체       ", fmt(m))
    payoff = abs(m['win_R'] / m['loss_R']) if m['loss_R'] else 0
    print(f"평균 이익 {m['win_R']:+.2f}R | 평균 손실 {m['loss_R']:+.2f}R | 손익비 {payoff:.2f}")
    print(f"자본곡선    {EQ0:,.0f} -> {eq:,.0f} USDT ({(eq / EQ0 - 1) * 100:+.1f}%), 최대낙폭 {mdd * 100:.1f}%")

    print("\n-- 청산 사유별 --")
    for why, label in (("stop", "손절/추적손절"), ("time", "시간청산"), ("end", "데이터 끝")):
        part = [t['R'] for t in acc if t['why'] == why]
        if part:
            print(f"{label:<10}", fmt(stats(part)))

    print("\n-- 방향별 --")
    print("롱         ", fmt(stats([t['R'] for t in acc if t['dir'] == 1])))
    print("숏         ", fmt(stats([t['R'] for t in acc if t['dir'] == -1])))

    mid = t0 + (t1 - t0) / 2
    h1 = [t['R'] for t in acc if t['entry_ts'] < mid]
    h2 = [t['R'] for t in acc if t['entry_ts'] >= mid]
    print("\n-- 전반/후반 일관성 --")
    print(f"전반 {t0:%m-%d}~{mid:%m-%d}", fmt(stats(h1)))
    print(f"후반 {mid:%m-%d}~{t1:%m-%d}", fmt(stats(h2)))

    by = {}
    for t in acc:
        by.setdefault(t['sym'], []).append(t['R'])
    tot = sorted(((sum(v), len(v), k) for k, v in by.items()), reverse=True)
    pos_syms = sum(1 for s, _, _ in tot if s > 0)
    print(f"\n-- 종목별: 플러스 {pos_syms}/{len(tot)}종목 --")
    print("상위:", ", ".join(f"{k.split('/')[0]}({s:+.1f}R/{n}건)" for s, n, k in tot[:5]))
    print("하위:", ", ".join(f"{k.split('/')[0]}({s:+.1f}R/{n}건)" for s, n, k in tot[-5:]))

    global FEE, SLIP
    saved = (FEE, SLIP)
    print("\n-- 비용 민감도 (같은 신호, 비용만 변경) --")
    cost_ok = 0
    for label, fee, slip in (("현재: 수수료 0.05% + 슬리피지 0.02%", 0.0005, 0.0002),
                             ("슬리피지 0", 0.0005, 0.0),
                             ("수수료 0.02% + 슬리피지 0.01% (메이커 가정, 낙관적)", 0.0002, 0.0001),
                             ("비용 0 (이론상)", 0.0, 0.0)):
        FEE, SLIP = fee, slip
        e2, md2, acc2, _ = portfolio(run_all(data, P))
        mm = stats([t['R'] for t in acc2])
        print(f"{label:<48} 평균R={mm['avg_R']:+.3f} PF={mm['pf']:.2f} 수익률={(e2 / EQ0 - 1) * 100:+.1f}%")
    FEE, SLIP = saved

    print("\n-- 파라미터 민감도 (손절 x 추적폭, 현재 설정 *) --")
    print(f"{'손절ATR':>7} {'추적ATR':>7} | 신호 수/일 | 평균R | PF")
    pos = tot_c = 0
    for sl in (1.0, 1.5, 2.0):
        for tr_ in (2.0, 3.0, 4.0):
            pp = dict(P, sl_atr=sl, trail_atr=tr_)
            ts_ = run_all(data, pp)
            mm = stats([t['R'] for t in ts_])
            mark = "*" if (sl, tr_) == (P['sl_atr'], P['trail_atr']) else " "
            print(f"{sl:>7} {tr_:>7} |{mark}{len(ts_) / span_days:6.2f}     | {mm['avg_R']:+.3f} | {mm['pf']:.2f}")
            tot_c += 1
            pos += mm['avg_R'] > 0
    print(f"평균 R > 0 인 조합: {pos}/{tot_c}")

    m1, m2 = stats(h1), stats(h2)
    checks = [
        (f"하루 체결 5건 이상 ({per_day:.2f}건)", per_day >= 5),
        (f"평균 R > 0.1 ({m['avg_R']:+.3f})", m['avg_R'] > 0.1),
        (f"프로핏 팩터 > 1.2 ({m['pf']:.2f})", m['pf'] > 1.2),
        (f"전반/후반 모두 평균 R > 0 ({m1['avg_R']:+.2f} / {m2['avg_R']:+.2f})", m1['avg_R'] > 0 and m2['avg_R'] > 0),
        (f"파라미터 민감도 70% 이상 플러스 ({pos}/{tot_c})", pos / tot_c >= 0.7),
        (f"플러스 종목 비율 55% 이상 ({pos_syms}/{len(tot)})", pos_syms / max(1, len(tot)) >= 0.55),
    ]
    print("\n===== 판정 =====")
    for txt, ok in checks:
        print(f"[{'통과' if ok else '미달'}] {txt}")
    if all(ok for _, ok in checks):
        print("\n모든 기준을 통과했습니다. 그래도 다음 단계는 모의매매(실시간 처음 보는 데이터)입니다.")
    else:
        print("\n기준 미달 항목이 있습니다. 이 상태로는 실거래를 권하지 않습니다.")
        if per_day < 5:
            print(f" - 하루 {per_day:.1f}건: --symbols 를 50 정도로 늘리거나 --tf 30m 로 바꾸면 거래 수가 늘어납니다.")
            print("   (단, 거래가 늘수록 비용이 쌓이므로 평균 R 이 함께 좋아진다는 보장은 없습니다.)")
    print("\n※ 종목 목록이 '현재 거래량 상위'라 선택 편향이 있고, 파라미터는 튜닝하지 않았습니다(고정).")
    print("※ 과거 성과는 미래를 보장하지 않습니다.")


# ===================== 모의매매 (실시간 처음 보는 데이터로 검증) =====================
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    conn = db()
    conn.execute('''CREATE TABLE IF NOT EXISTS paper (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, side TEXT NOT NULL,
        entry_bar_ts TEXT NOT NULL, entry_price REAL NOT NULL, atr REAL NOT NULL,
        status TEXT DEFAULT 'OPEN', exit_time TEXT, exit_price REAL, exit_reason TEXT, r_multiple REAL)''')
    conn.commit()
    conn.close()


def paper_stats():
    conn = db()
    rows = conn.execute("SELECT r_multiple, entry_bar_ts FROM paper WHERE status='CLOSED'").fetchall()
    first = conn.execute("SELECT MIN(entry_bar_ts) t FROM paper").fetchone()['t']
    conn.close()
    if not rows:
        print("[모의 누적] 완료된 거래 없음")
        return
    rs = [r['r_multiple'] for r in rows]
    days = max(1.0, (datetime.now() - datetime.fromisoformat(first)).total_seconds() / 86400)
    print(f"[모의 누적] {fmt(stats(rs))} | 하루 {len(rs) / days:.2f}건")
    if len(rs) < 100:
        print(f"   (표본 {len(rs)}건: 100건 미만이라 우위 판단은 이릅니다)")


def paper_cycle(syms):
    ex = get_ex()
    conn = db()
    opened = {r['symbol'] for r in conn.execute("SELECT symbol FROM paper WHERE status='OPEN'").fetchall()}
    n_open = len(opened)
    conn.close()
    for s in syms:
        try:
            raw = ex.fetch_ohlcv(s, TF, limit=400)
        except Exception as e:
            print(f"{s} 조회 실패: {e}")
            continue
        forming_ts, forming_open = pd.to_datetime(raw[-1][0], unit='ms'), float(raw[-1][1])
        closed = pd.DataFrame(raw[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        closed['timestamp'] = pd.to_datetime(closed['timestamp'], unit='ms')
        d = indicators(closed, P['don'])

        conn = db()
        tr = conn.execute("SELECT * FROM paper WHERE symbol=? AND status='OPEN'", (s,)).fetchone()
        conn.close()
        if tr:
            dr = 1 if tr['side'] == 'long' else -1
            bars = d[d['timestamp'] >= pd.Timestamp(tr['entry_bar_ts'])].reset_index(drop=True)
            if len(bars):
                o, h, l, c = (bars[k].values for k in ('open', 'high', 'low', 'close'))
                res = run_trade(o, h, l, c, 0, dr, tr['atr'], P, entry=tr['entry_price'])
                if res:
                    j, px_, why, r = res
                    conn = db()
                    conn.execute("UPDATE paper SET status='CLOSED', exit_time=?, exit_price=?, exit_reason=?, "
                                 "r_multiple=? WHERE id=?", (bars['timestamp'].iloc[j].isoformat(), px_, why, r, tr['id']))
                    conn.commit()
                    conn.close()
                    n_open -= 1
                    opened.discard(s)
                    print(f"[모의] 종료 {s} {tr['side']} {why} {r:+.2f}R")
            continue

        if n_open >= MAX_POS or s in opened:
            continue
        sig = signal_array(d, P)
        if sig[-1] == 0:
            continue
        dr = int(sig[-1])
        a = float(d['atr'].iloc[-1])
        entry = forming_open * (1 + dr * SLIP)
        conn = db()
        conn.execute("INSERT INTO paper (symbol, side, entry_bar_ts, entry_price, atr) VALUES (?, ?, ?, ?, ?)",
                     (s, 'long' if dr == 1 else 'short', forming_ts.isoformat(), entry, a))
        conn.commit()
        conn.close()
        n_open += 1
        opened.add(s)
        print(f"[모의] 진입 {s} {'LONG' if dr == 1 else 'SHORT'} {entry:,.4f} | 손절 {entry - dr * P['sl_atr'] * a:,.4f}")
    paper_stats()


def paper_main(n_sym):
    setup_db()
    syms = get_universe(n_sym)
    print(f"\n=== 멀티 코인 추세 추종 모의매매 === {datetime.now():%Y-%m-%d %H:%M}")
    print(f"감시 {len(syms)}종목 | {TF} | 거래당 위험 {RISK_PER_TRADE * 100:.1f}% | 동시 최대 {MAX_POS}개 | 주문 없음")
    now = time.time()
    last_bucket = None if now % 3600 < 300 else int(now // 3600)
    step = TF_MIN[TF] * 60
    last_bucket = None if now % step < 300 else int(now // step)
    while True:
        try:
            now = time.time()
            bucket = int(now // step)
            if bucket != last_bucket and now % step >= 15:
                last_bucket = bucket
                print(f"\n[{datetime.now():%m-%d %H:%M}] 점검")
                paper_cycle(syms)
            time.sleep(30)
        except KeyboardInterrupt:
            print("종료")
            break
        except Exception as e:
            print("오류:", e)
            time.sleep(30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, metavar="DAYS", help="백테스트 (180 권장)")
    ap.add_argument("--tf", default="1h", choices=list(TF_MIN.keys()))
    ap.add_argument("--symbols", type=int, default=N_SYMBOLS, help="감시 종목 수 (거래량 상위)")
    ap.add_argument("--synthetic", action="store_true", help="무작위 가짜 데이터로 실행 (동작 점검)")
    ap.add_argument("--stats", action="store_true", help="모의매매 누적 성적")
    args = ap.parse_args()
    TF = args.tf
    if args.backtest:
        backtest(args.backtest, args.tf, args.symbols, args.synthetic)
    elif args.stats:
        setup_db()
        paper_stats()
    else:
        paper_main(args.symbols)