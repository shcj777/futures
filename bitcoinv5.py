"""
5일/20일 이동평균 교차 전략 (롱 <-> 숏 계속 전환, 자본의 50% 진입)
=====================================================================
규칙
  - 5일선이 20일선을 위로 돌파(골든크로스): 숏 청산 + 롱 진입
  - 5일선이 20일선을 아래로 돌파(데드크로스): 롱 청산 + 숏 진입
  - 진입 금액 = 현재 자본의 50% (레버리지 1배), 항상 포지션 보유
  - 신호는 봉 종가 확정 후 판단, 체결은 다음 봉 시가 (미래 참조 방지)
  - 익절/손절/반대매매 없음. 오직 교차 신호로만 전환
  - 실거래 기능 없음 (백테스트 + 모의매매)

실행
  pip install ccxt pandas numpy
  python bitcoinv5.py --backtest 1000                  # 일봉 1000일, 거래량 상위 10종목
  python bitcoinv5.py --backtest 1500 --symbols BTC,ETH
  python bitcoinv5.py --backtest 500 --synthetic       # 가짜 데이터로 동작 점검
  python bitcoinv5.py                                  # 모의매매 (하루 1회 점검, 주문 없음)
  python bitcoinv5.py --stats                          # 모의매매 누적 성적
"""
import time
import argparse
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

# ===================== 설정 =====================
TF = "1d"
FAST, SLOW = 5, 20
POS_FRAC = 0.50             # 거래당 투입 금액 = 현재 자본의 50%
FEE = 0.0005                # 편도 수수료
SLIP = 0.0002               # 편도 슬리피지
EQ0 = 1000.0
N_SYMBOLS = 10
DB_FILE = "ma_cross_paper.db"

_ex = None


def get_ex():
    global _ex
    if _ex is None:
        import ccxt
        _ex = ccxt.binance({'enableRateLimit': True,
                            'options': {'defaultType': 'future', 'fetchCurrencies': False}})
    return _ex


# ===================== 신호 =====================
def add_ma(df):
    d = df.copy()
    d['fast'] = d['close'].rolling(FAST).mean()
    d['slow'] = d['close'].rolling(SLOW).mean()
    return d


def cross_array(d):
    """+1: 골든크로스(5선이 20선 위로), -1: 데드크로스(아래로), 0: 없음. 봉 종가 기준."""
    f, s = d['fast'].values, d['slow'].values
    ok = np.isfinite(f) & np.isfinite(s)
    above = np.where(ok, f > s, False)
    prev_ok = np.concatenate([[False], ok[:-1]])
    prev_above = np.concatenate([[False], above[:-1]])
    up = ok & prev_ok & above & ~prev_above
    dn = ok & prev_ok & ~above & prev_above
    return np.where(up, 1, np.where(dn, -1, 0))


# ===================== 백테스트 =====================
def sim_symbol(d):
    """한 종목 시뮬레이션. 반환: (거래 목록, 봉별 평가자본 배열)"""
    o, c = d['open'].values, d['close'].values
    ts = d['timestamp']
    cross = cross_array(d)
    cost = FEE + SLIP
    eq = EQ0
    pos, qty, entry, entry_ts = 0, 0.0, 0.0, None
    trades, curve = [], []

    for i in range(len(o)):
        sig = cross[i - 1] if i > 0 else 0          # 전 봉 종가에서 확정된 신호
        if sig != 0 and sig != pos:
            if pos != 0:                              # 기존 포지션 청산
                px = o[i] * (1 - pos * cost)          # 롱 매도는 불리하게 낮게, 숏 환매는 높게
                pnl = (px - entry) * qty * pos
                eq += pnl
                trades.append(dict(dir=pos, entry_ts=entry_ts, exit_ts=ts.iloc[i],
                                   ret=pnl / (entry * qty), pnl=pnl))
            entry = o[i] * (1 + sig * cost)           # 롱 매수는 높게, 숏 매도는 낮게
            qty = eq * POS_FRAC / entry
            pos, entry_ts = int(sig), ts.iloc[i]
        curve.append(eq + ((c[i] - entry) * qty * pos if pos else 0.0))
    # 마지막 보유분은 종가 평가(미청산)
    open_pos = dict(dir=pos, entry_ts=entry_ts, price=entry) if pos else None
    return trades, np.array(curve), open_pos


def stats(rets):
    n = len(rets)
    if n == 0:
        return dict(n=0, wr=0.0, avg=0.0, pf=0.0)
    a = np.array(rets) * 100
    w, ls = a[a > 0], a[a <= 0]
    return dict(n=n, wr=float((a > 0).mean() * 100), avg=float(a.mean()),
                pf=float(w.sum() / -ls.sum()) if len(ls) and ls.sum() < 0 else float('inf'))


def fmt(m):
    pf = f"{m['pf']:.2f}" if np.isfinite(m['pf']) else "inf"
    return f"n={m['n']:>3} 승률={m['wr']:5.1f}% 평균={m['avg']:+.2f}% PF={pf:>5}"


def backtest(days, symbols, use_synth):
    data = load_data(days, symbols, use_synth)
    if not data:
        print("사용 가능한 데이터가 없습니다.")
        return
    print(f"\n===== {FAST}/{SLOW} 이동평균 교차 백테스트 ({TF}, 투입 {POS_FRAC * 100:.0f}%) =====")
    print(f"종목별로 각각 {EQ0:,.0f} USDT로 독립 시뮬레이션 | 수수료 {FEE * 100:.2f}% + 슬리피지 {SLIP * 100:.2f}% (편도)\n")
    print(f"{'종목':<14}{'기간':>7} | {'수익률':>8} {'낙폭':>7} {'단순보유':>9} | 거래")

    all_rets, rows = [], []
    for s, d in data.items():
        trades, curve, open_pos = sim_symbol(d)
        ret = curve[-1] / EQ0 - 1
        mdd = (curve / np.maximum.accumulate(curve) - 1).min()
        bh = d['close'].iloc[-1] / d['close'].iloc[0] - 1
        span = (d['timestamp'].iloc[-1] - d['timestamp'].iloc[0]).days
        rets = [t['ret'] for t in trades]
        all_rets += rets
        rows.append((ret, mdd, bh))
        print(f"{s:<14}{span:>6}일 | {ret * 100:>+7.1f}% {mdd * 100:>6.1f}% {bh * 100:>+8.1f}% | {fmt(stats(rets))}")

    r = np.array(rows) * 100
    print(f"\n평균 (종목 동일가중)  수익률 {r[:, 0].mean():+.1f}% | 낙폭 {r[:, 1].mean():.1f}% | 단순보유 {r[:, 2].mean():+.1f}%")
    print(f"플러스 종목 {int((r[:, 0] > 0).sum())}/{len(r)}개 | 전체 거래 {fmt(stats(all_rets))}")

    for dr, name in ((1, '롱'), (-1, '숏')):
        rr = []
        for d in data.values():
            rr += [t['ret'] for t in sim_symbol(d)[0] if t['dir'] == dr]
        print(f"  {name}: {fmt(stats(rr))}")

    print("\n※ 펀딩비, 청산 위험, 실제 호가 체결은 반영되지 않았습니다. 과거 성과는 미래를 보장하지 않습니다.")
    print("※ 종목 목록이 '현재 거래량 상위'라 선택 편향이 있습니다.")


# ===================== 데이터 =====================
def get_universe(symbols, n):
    if symbols:
        return [f"{b.strip().upper()}/USDT:USDT" for b in symbols.split(",")]
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
        return [s for _, s in cands[:n]]
    except Exception as e:
        print("종목 목록 조회 실패, 기본 목록 사용:", e)
        return [f"{b}/USDT:USDT" for b in ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"][:n]]


def fetch_history(sym, days, tf):
    ex = get_ex()
    step = ex.parse_timeframe(tf) * 1000
    since = ex.milliseconds() - int((days + 5) * 86400000)
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
    return df.iloc[:-1].reset_index(drop=True)          # 형성 중인 마지막 봉 제외


def synthetic_universe(n, days, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range('2023-01-01', periods=days, freq='D')
    data = {}
    for k in range(n):
        r = rng.normal(0.0004, 0.03, days)
        c = 100 * np.exp(np.cumsum(r))
        o = np.concatenate([[100], c[:-1]])
        data[f"SYN{k}/USDT:USDT"] = pd.DataFrame({'timestamp': ts, 'open': o, 'high': np.maximum(o, c),
                                                  'low': np.minimum(o, c), 'close': c, 'volume': 1.0})
    return data


def load_data(days, symbols, use_synth):
    n = len(symbols.split(",")) if symbols else N_SYMBOLS
    if use_synth:
        raw = synthetic_universe(n, days)
    else:
        syms = get_universe(symbols, N_SYMBOLS)
        print(f"{len(syms)}종목 데이터 수집 중... ({days}일, {TF})")
        raw = {}
        for k, s in enumerate(syms, 1):
            try:
                df = fetch_history(s, days, TF)
                if len(df) < SLOW + 5:
                    print(f"  [{k}/{len(syms)}] {s}: 데이터 부족({len(df)}봉), 제외")
                    continue
                raw[s] = df
            except Exception as e:
                print(f"  [{k}/{len(syms)}] {s}: 수집 실패 ({e})")
    return {s: add_ma(df) for s, df in raw.items()}


# ===================== 모의매매 (주문 없음) =====================
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    conn = db()
    conn.execute('''CREATE TABLE IF NOT EXISTS paper (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, side TEXT NOT NULL,
        entry_ts TEXT NOT NULL, entry_price REAL NOT NULL,
        status TEXT DEFAULT 'OPEN', exit_ts TEXT, exit_price REAL, ret REAL)''')
    conn.commit()
    conn.close()


def paper_stats():
    conn = db()
    rows = conn.execute("SELECT ret FROM paper WHERE status='CLOSED' ORDER BY id").fetchall()
    conn.close()
    if not rows:
        print("[모의 누적] 완료된 거래 없음")
        return
    rs = [r['ret'] for r in rows]
    eq = EQ0
    for r in rs:
        eq += eq * POS_FRAC * r
    print(f"[모의 누적] {fmt(stats(rs))} | 종목 합산 가정 가상 자본 {EQ0:,.0f} -> {eq:,.0f}")


def paper_cycle(syms):
    ex = get_ex()
    for s in syms:
        try:
            raw = ex.fetch_ohlcv(s, TF, limit=60)
        except Exception as e:
            print(f"{s} 조회 실패: {e}")
            continue
        forming_ts = pd.to_datetime(raw[-1][0], unit='ms')
        forming_open = float(raw[-1][1])
        closed = pd.DataFrame(raw[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        closed['timestamp'] = pd.to_datetime(closed['timestamp'], unit='ms')
        sig = int(cross_array(add_ma(closed))[-1])          # 방금 마감된 봉에서 교차가 났는가
        if sig == 0:
            continue

        conn = db()
        cur = conn.execute("SELECT * FROM paper WHERE symbol=? AND status='OPEN'", (s,)).fetchone()
        cur_dir = 0 if cur is None else (1 if cur['side'] == 'long' else -1)
        if cur_dir == sig:
            conn.close()
            continue
        cost = FEE + SLIP
        if cur:                                              # 기존 포지션 청산
            px = forming_open * (1 - cur_dir * cost)
            r = cur_dir * (px - cur['entry_price']) / cur['entry_price']
            conn.execute("UPDATE paper SET status='CLOSED', exit_ts=?, exit_price=?, ret=? WHERE id=?",
                         (forming_ts.isoformat(), px, r, cur['id']))
            print(f"[모의] 청산 {s} {cur['side']} {r * 100:+.2f}%")
        entry = forming_open * (1 + sig * cost)
        conn.execute("INSERT INTO paper (symbol, side, entry_ts, entry_price) VALUES (?, ?, ?, ?)",
                     (s, 'long' if sig == 1 else 'short', forming_ts.isoformat(), entry))
        conn.commit()
        conn.close()
        print(f"[모의] 진입 {s} {'LONG' if sig == 1 else 'SHORT'} {entry:,.4f} (자본의 {POS_FRAC * 100:.0f}%)")
    paper_stats()


def paper_main(symbols):
    setup_db()
    syms = get_universe(symbols, N_SYMBOLS)
    print(f"\n=== {FAST}/{SLOW} 이평선 교차 모의매매 === {datetime.now():%Y-%m-%d %H:%M}")
    print(f"감시 {len(syms)}종목 | {TF} | 투입 {POS_FRAC * 100:.0f}% | 주문 없음 (Ctrl+C 종료)")
    step = 24 * 3600
    now = time.time()
    last_bucket = None if now % step < 300 else int(now // step)   # 일봉 마감(UTC 00:00) 직후 1회 점검
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
    ap.add_argument("--backtest", type=int, metavar="DAYS", help="백테스트 (1000 권장)")
    ap.add_argument("--symbols", default="", help="종목 지정 예: BTC,ETH (비우면 거래량 상위 10개)")
    ap.add_argument("--synthetic", action="store_true", help="가짜 데이터로 동작 점검")
    ap.add_argument("--stats", action="store_true", help="모의매매 누적 성적")
    args = ap.parse_args()
    if args.backtest:
        backtest(args.backtest, args.symbols, args.synthetic)
    elif args.stats:
        setup_db()
        paper_stats()
    else:
        paper_main(args.symbols)