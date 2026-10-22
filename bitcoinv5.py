"""
멀티 코인 '반대 매매' 프로그램 (고정 % 익절/손절, 자본의 70% 진입)
=====================================================================
변경점 (v5 -> v6)
  - 신호 반전: 기존 전략이 LONG 이면 SHORT, SHORT 이면 LONG 으로 진입 (P['invert'])
  - 진입 금액: 현재 자본의 70% (POS_FRAC)
  - 익절: 진입가 대비 +1.0% (가격 기준)  /  손절: 진입가 대비 -0.25% (가격 기준)
  - 한 봉 안에서 익절/손절이 모두 닿으면 '손절 먼저'로 계산 (보수적)
  - 70%씩 여러 종목에 동시에 넣으면 레버리지가 되므로 동시 보유는 기본 1개 (MAX_POS)

!! 중요 !!
  - 반대로 매매한다고 손실이 이익으로 바뀌지 않습니다. 수수료/슬리피지는 양방향 모두 손해이고,
    손절 0.25%는 코인 1시간봉의 평균 변동폭(보통 0.5~1%)보다 좁아서 노이즈에 자주 걸립니다.
  - 그래서 백테스트에서 '정방향 vs 반대방향', 손익분기 승률, 비용 민감도를 함께 보여줍니다.
  - 실거래 기능은 없습니다 (백테스트 + 모의매매만).

실행
  python bitcoinv6.py --backtest 180
  python bitcoinv6.py --backtest 180 --symbols 50 --tf 30m
  python bitcoinv6.py --backtest 180 --synthetic     # 가짜 데이터로 동작 점검
  python bitcoinv6.py                                # 모의매매 (주문 없음)
  python bitcoinv6.py --stats                        # 모의매매 누적 성적
"""
import time
import argparse
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

# ===================== 설정 =====================
TF = "1h"
DB_FILE = "reverse_paper.db"

# 신호 파라미터 + 반전/익절/손절
P = dict(don=20, adx=20, vz=0.5, hold=72,
         invert=True,      # True: 신호 반대로 매매
         tp=0.01,          # 익절 +1.0%
         sl=0.0025)        # 손절 -0.25%

POS_FRAC = 0.70             # 거래당 투입 금액 = 현재 자본의 70%
MAX_POS = 1                 # 동시 보유 최대 개수 (70% x N 이 총 노출. 1 권장)
N_SYMBOLS = 30
FEE = 0.0005                # 편도 수수료
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
    """원래 전략 신호: +1 롱, -1 숏, 0 없음. p['invert']=True 면 부호를 뒤집어 반환 (진입은 다음 봉 시가)."""
    c = d['close'].values
    ok = (d['adx'].values >= p['adx']) & (d['vol_z'].values >= p['vz'])
    lg = (c > d['don_hi'].values) & (c > d['ema200'].values) & ok
    st = (c < d['don_lo'].values) & (c < d['ema200'].values) & ok
    sig = np.where(lg, 1, np.where(st, -1, 0))
    valid = np.isfinite(d['atr'].values) & np.isfinite(d['ema200'].values) & np.isfinite(d['don_hi'].values) \
        & (np.arange(len(d)) >= 210)
    sig = np.where(valid, sig, 0)
    return -sig if p.get('invert') else sig


# ===================== 한 건의 거래 진행 (백테스트/모의매매 공용) =====================
def _net_ret(px, entry, dr):
    """슬리피지 반영한 청산가와 수수료 차감 후 수익률(진입 금액 대비)."""
    px2 = px * (1 - dr * SLIP)
    return px2, dr * (px2 - entry) / entry - 2 * FEE


def run_trade(o, h, l, c, e, dr, p, entry=None, force_close=False):
    """e: 진입 봉 인덱스, dr: +1(롱)/-1(숏).
    손절 = 진입가 -/+ sl%, 익절 = 진입가 +/- tp%. 같은 봉에서 둘 다 닿으면 손절 우선.
    반환: (종료 봉 인덱스, 종료가, 사유, 순수익률) 또는 진행 중이면 None."""
    if entry is None:
        entry = o[e] * (1 + dr * SLIP)
    stop = entry * (1 - dr * p['sl'])
    tp = entry * (1 + dr * p['tp'])
    last = len(o) - 1
    end = e + p['hold']
    for j in range(e, min(end, last) + 1):
        px_ = None
        if dr == 1:
            if l[j] <= stop:
                px_, why = min(stop, o[j]), 'sl'
            elif h[j] >= tp:
                px_, why = max(tp, o[j]), 'tp'
        else:
            if h[j] >= stop:
                px_, why = max(stop, o[j]), 'sl'
            elif l[j] <= tp:
                px_, why = min(tp, o[j]), 'tp'
        if px_ is None and j == end:
            px_, why = c[j], 'time'
        if px_ is not None:
            px2, r = _net_ret(px_, entry, dr)
            return j, px2, why, r
    if force_close:
        px2, r = _net_ret(c[last], entry, dr)
        return last, px2, 'end', r
    return None


def sim_symbol(sym, d, p):
    o, h, l, c = (d[k].values for k in ('open', 'high', 'low', 'close'))
    ts = d['timestamp']
    sig = signal_array(d, p)
    out, nxt = [], 0
    for i in np.nonzero(sig)[0]:
        if i < nxt or i + 1 >= len(o):
            continue
        dr = int(sig[i])
        j, _, why, r = run_trade(o, h, l, c, i + 1, dr, p, force_close=True)
        out.append(dict(sym=sym, dir=dr, entry_ts=ts.iloc[i + 1], exit_ts=ts.iloc[j], ret=r, why=why))
        nxt = j
    return out


# ===================== 포트폴리오 (투입 70%, 동시 보유 제한) =====================
def portfolio(trades, eq0=EQ0, frac=POS_FRAC, max_pos=MAX_POS):
    trades = sorted(trades, key=lambda t: t['entry_ts'])
    eq = peak = eq0
    mdd = 0.0
    open_pos, acc = [], []
    skipped = 0

    def settle(upto):
        nonlocal eq, peak, mdd, open_pos
        done = sorted([x for x in open_pos if x['exit_ts'] <= upto], key=lambda x: x['exit_ts'])
        for x in done:
            eq += x['alloc'] * x['ret']
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak)
        open_pos = [x for x in open_pos if x['exit_ts'] > upto]

    for t in trades:
        settle(t['entry_ts'])
        if len(open_pos) >= max_pos:
            skipped += 1
            continue
        x = dict(t)
        x['alloc'] = eq * frac
        open_pos.append(x)
        acc.append(x)
    settle(pd.Timestamp.max)
    return eq, mdd, acc, skipped


def stats(rets):
    n = len(rets)
    if n == 0:
        return dict(n=0, wr=0.0, avg=0.0, pf=0.0, tot=0.0)
    a = np.array(rets) * 100  # % 단위
    w, ls = a[a > 0], a[a <= 0]
    return dict(n=n, wr=float((a > 0).mean() * 100), avg=float(a.mean()),
                pf=float(w.sum() / -ls.sum()) if len(ls) and ls.sum() < 0 else float('inf'),
                tot=float(a.sum()))


def fmt(m):
    pf = f"{m['pf']:.2f}" if np.isfinite(m['pf']) else "inf"
    return f"n={m['n']:>4} 승률={m['wr']:5.1f}% 평균={m['avg']:+.3f}% PF={pf:>5} 합계={m['tot']:+7.1f}%"


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


def summarize(data, p):
    eq, mdd, acc, skipped = portfolio(run_all(data, p))
    return eq, mdd, acc, skipped, stats([t['ret'] for t in acc])


def backtest(days, tf, n_sym, use_synth):
    global FEE, SLIP
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
    span = max(1.0, (t1 - t0).total_seconds() / 86400)

    eq, mdd, acc, skipped = portfolio(trades)
    m = stats([t['ret'] for t in acc])
    mode = "반대 매매" if P['invert'] else "정방향"
    print(f"\n===== {mode} 백테스트: {t0:%Y-%m-%d} ~ {t1:%Y-%m-%d} ({span:.0f}일), {len(data)}종목, {tf} =====")
    print(f"익절 +{P['tp'] * 100:.2f}% / 손절 -{P['sl'] * 100:.2f}% | 투입 자본의 {POS_FRAC * 100:.0f}% | "
          f"동시 {MAX_POS}개 | 최대 {P['hold']}봉")
    print(f"\n신호 {len(trades)}건 -> 체결 {len(acc)}건 (동시 보유 제한으로 {skipped}건 제외), 하루 {len(acc) / span:.2f}건")
    print("전체      ", fmt(m))

    cost = 2 * (FEE + SLIP) * 100
    be = (P['sl'] * 100 + cost) / ((P['tp'] + P['sl']) * 100)
    print(f"왕복 비용 약 {cost:.2f}% (진입금액 대비) -> 손익분기 승률 약 {be * 100:.1f}% (현재 승률 {m['wr']:.1f}%)")
    print(f"자본곡선   {EQ0:,.0f} -> {eq:,.0f} USDT ({(eq / EQ0 - 1) * 100:+.1f}%), 최대낙폭 {mdd * 100:.1f}%")

    print("\n-- 청산 사유별 --")
    for why, label in (("tp", "익절"), ("sl", "손절"), ("time", "시간청산"), ("end", "데이터 끝")):
        part = [t['ret'] for t in acc if t['why'] == why]
        if part:
            print(f"{label:<8}", fmt(stats(part)))

    print("\n-- 방향별 --")
    print("롱       ", fmt(stats([t['ret'] for t in acc if t['dir'] == 1])))
    print("숏       ", fmt(stats([t['ret'] for t in acc if t['dir'] == -1])))

    print("\n-- 정방향 vs 반대방향 (같은 익절/손절/70%) --")
    for inv, label in ((False, "정방향 (신호 그대로)"), (True, "반대방향 (신호 반전)")):
        e2, md2, _, _, mm = summarize(data, dict(P, invert=inv))
        print(f"{label:<22}", fmt(mm), f"| 수익률 {(e2 / EQ0 - 1) * 100:+.1f}% 낙폭 {md2 * 100:.1f}%")

    mid = t0 + (t1 - t0) / 2
    h1 = [t['ret'] for t in acc if t['entry_ts'] < mid]
    h2 = [t['ret'] for t in acc if t['entry_ts'] >= mid]
    print("\n-- 전반/후반 일관성 --")
    print(f"전반 {t0:%m-%d}~{mid:%m-%d}", fmt(stats(h1)))
    print(f"후반 {mid:%m-%d}~{t1:%m-%d}", fmt(stats(h2)))

    saved = (FEE, SLIP)
    print("\n-- 비용 민감도 --")
    for label, fee, slip in (("현재: 수수료 0.05% + 슬리피지 0.02%", 0.0005, 0.0002),
                             ("메이커 가정 0.02% + 0.01% (낙관적)", 0.0002, 0.0001),
                             ("비용 0 (이론상)", 0.0, 0.0)):
        FEE, SLIP = fee, slip
        e2, _, _, _, mm = summarize(data, P)
        print(f"{label:<40} 평균={mm['avg']:+.3f}% PF={mm['pf']:.2f} 수익률={(e2 / EQ0 - 1) * 100:+.1f}%")
    FEE, SLIP = saved

    print("\n-- 손절 x 익절 민감도 (현재 설정 *) --")
    print(f"{'손절%':>6} {'익절%':>6} | 체결/일 | 승률  | 평균%   | PF")
    pos = tot_c = 0
    for sl in (0.25, 0.5, 1.0):
        for tp in (1.0, 2.0):
            pp = dict(P, sl=sl / 100, tp=tp / 100)
            _, _, acc2, _, mm = summarize(data, pp)
            mark = "*" if (abs(sl / 100 - P['sl']) < 1e-9 and abs(tp / 100 - P['tp']) < 1e-9) else " "
            print(f"{sl:>6} {tp:>6} |{mark}{len(acc2) / span:6.2f} | {mm['wr']:4.1f}% | {mm['avg']:+.3f} | {mm['pf']:.2f}")
            tot_c += 1
            pos += mm['avg'] > 0
    print(f"평균 수익률 > 0 인 조합: {pos}/{tot_c}")

    m1, m2 = stats(h1), stats(h2)
    checks = [
        (f"평균 거래 수익률 > 0.05% ({m['avg']:+.3f}%)", m['avg'] > 0.05),
        (f"프로핏 팩터 > 1.2 ({m['pf']:.2f})", m['pf'] > 1.2),
        (f"승률이 손익분기({be * 100:.1f}%) 초과 ({m['wr']:.1f}%)", m['wr'] > be * 100),
        (f"전반/후반 모두 평균 > 0 ({m1['avg']:+.2f}% / {m2['avg']:+.2f}%)", m1['avg'] > 0 and m2['avg'] > 0),
        (f"민감도 조합 70% 이상 플러스 ({pos}/{tot_c})", pos / tot_c >= 0.7),
    ]
    print("\n===== 판정 =====")
    for txt, ok in checks:
        print(f"[{'통과' if ok else '미달'}] {txt}")
    if all(ok for _, ok in checks):
        print("\n기준을 모두 통과했습니다. 그래도 다음 단계는 모의매매(처음 보는 실시간 데이터)입니다.")
    else:
        print("\n기준 미달 항목이 있습니다. 이 상태로는 실거래를 권하지 않습니다.")
    print("\n※ 자본의 70%를 한 번에 넣으면 손절 -0.25%도 자본의 약 -0.175%(+비용)입니다. "
          "연속 손절이 쌓이면 빠르게 줄어듭니다.")
    print("※ 종목 목록이 '현재 거래량 상위'라 선택 편향이 있습니다. 과거 성과는 미래를 보장하지 않습니다.")


# ===================== 모의매매 =====================
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    conn = db()
    conn.execute('''CREATE TABLE IF NOT EXISTS paper (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, side TEXT NOT NULL,
        entry_bar_ts TEXT NOT NULL, entry_price REAL NOT NULL,
        status TEXT DEFAULT 'OPEN', exit_time TEXT, exit_price REAL, exit_reason TEXT, ret REAL)''')
    conn.commit()
    conn.close()


def paper_stats():
    conn = db()
    rows = conn.execute("SELECT ret FROM paper WHERE status='CLOSED'").fetchall()
    conn.close()
    if not rows:
        print("[모의 누적] 완료된 거래 없음")
        return
    rs = [r['ret'] for r in rows]
    eq = EQ0
    for r in rs:
        eq += eq * POS_FRAC * r
    print(f"[모의 누적] {fmt(stats(rs))} | 가상 자본 {EQ0:,.0f} -> {eq:,.0f}")
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
                res = run_trade(o, h, l, c, 0, dr, P, entry=tr['entry_price'])
                if res:
                    j, px_, why, r = res
                    conn = db()
                    conn.execute("UPDATE paper SET status='CLOSED', exit_time=?, exit_price=?, exit_reason=?, "
                                 "ret=? WHERE id=?", (bars['timestamp'].iloc[j].isoformat(), px_, why, r, tr['id']))
                    conn.commit()
                    conn.close()
                    n_open -= 1
                    opened.discard(s)
                    print(f"[모의] 종료 {s} {tr['side']} {why} {r * 100:+.2f}%")
            continue

        if n_open >= MAX_POS or s in opened:
            continue
        sig = signal_array(d, P)          # 이미 반전 적용됨
        if sig[-1] == 0:
            continue
        dr = int(sig[-1])
        entry = forming_open * (1 + dr * SLIP)
        conn = db()
        conn.execute("INSERT INTO paper (symbol, side, entry_bar_ts, entry_price) VALUES (?, ?, ?, ?)",
                     (s, 'long' if dr == 1 else 'short', forming_ts.isoformat(), entry))
        conn.commit()
        conn.close()
        n_open += 1
        opened.add(s)
        print(f"[모의] 진입 {s} {'LONG' if dr == 1 else 'SHORT'} {entry:,.4f} | "
              f"손절 {entry * (1 - dr * P['sl']):,.4f} | 익절 {entry * (1 + dr * P['tp']):,.4f}")
    paper_stats()


def paper_main(n_sym):
    setup_db()
    syms = get_universe(n_sym)
    print(f"\n=== 반대 매매 모의매매 === {datetime.now():%Y-%m-%d %H:%M}")
    print(f"감시 {len(syms)}종목 | {TF} | 투입 {POS_FRAC * 100:.0f}% | 익절 +{P['tp'] * 100:.2f}% "
          f"손절 -{P['sl'] * 100:.2f}% | 동시 최대 {MAX_POS}개 | 주문 없음")
    step = TF_MIN[TF] * 60
    now = time.time()
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
    ap.add_argument("--synthetic", action="store_true", help="가짜 데이터로 동작 점검")
    ap.add_argument("--stats", action="store_true", help="모의매매 누적 성적")
    ap.add_argument("--no-invert", action="store_true", help="반대 매매 끄기 (정방향)")
    args = ap.parse_args()
    TF = args.tf
    if args.no_invert:
        P['invert'] = False
    if args.backtest:
        backtest(args.backtest, args.tf, args.symbols, args.synthetic)
    elif args.stats:
        setup_db()
        paper_stats()
    else:
        paper_main(args.symbols)