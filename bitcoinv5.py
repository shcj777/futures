"""
한국시간 22:00~24:00 비트코인 역방향 단타 전략 (백테스트 + 모의매매)
=====================================================================
규칙
  - 매일 한국시간(KST) 22:00~24:00 에만 운영 (UTC 13:00~15:00)
  - 22:00 시점에 직전 LOOKBACK분(기본 60분) 동안 가격이 올랐으면 SHORT, 내렸으면 LONG (역방향)
  - 진입 금액 1000 USDT 고정, 레버리지 1배
  - 익절 +1.0% / 손절 -0.25% (진입가 기준, 수수료/슬리피지 별도 반영)
  - 24:00 까지 익절/손절이 안 나면 그 시점 가격으로 강제 청산 (하루 1회 거래)
  - 같은 1분봉 안에서 익절/손절이 모두 닿으면 손절 먼저 난 것으로 계산 (보수적)
  - 실거래 기능 없음 (백테스트 + 모의매매)

실행
  pip install ccxt pandas numpy
  python bitcoinv5.py --backtest 365                # 최근 365일 백테스트
  python bitcoinv5.py.py --backtest 365 --lookback 30  # 신호 판단 구간을 30분으로
  python bitcoinv5.py --backtest 200 --synthetic    # 가짜 데이터로 동작 점검
  python bitcoinv5.py                               # 모의매매 (주문 없음)
  python bitcoinv5.py --stats                       # 모의매매 누적 성적
"""
import time
import argparse
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# ===================== 설정 =====================
SYMBOL = "BTC/USDT:USDT"
NOTIONAL = 1000.0            # 진입 금액 (USDT), 레버리지 1배
TP = 0.010                   # 익절 +1.0%
SL = 0.0025                  # 손절 -0.25%
START_H, END_H = 22, 24      # 운영 시간 (한국시간)
LOOKBACK = 60                # 22시 기준 직전 몇 분의 등락으로 방향을 정할지
FEE = 0.0005                 # 편도 수수료
SLIP = 0.0002                # 편도 슬리피지
COST = FEE + SLIP
WINDOW_MIN = (END_H - START_H) * 60      # 운영 시간(분) = 120
DB_FILE = "night_contrarian_paper.db"

KST = timezone(timedelta(hours=9))
UTC_START_H = START_H - 9                # 한국 22시 = UTC 13시

_ex = None


def get_ex():
    global _ex
    if _ex is None:
        import ccxt
        _ex = ccxt.binance({'enableRateLimit': True,
                            'options': {'defaultType': 'future', 'fetchCurrencies': False}})
    return _ex


# ===================== 하루치 시뮬레이션 =====================
def run_day(df, lb):
    """
    df: 1분봉 (lb + 120개). 인덱스 0 = 22시 lb분 전, 인덱스 lb = 22:00 봉.
    신호: 22:00 봉 시가 vs lb분 전 봉 시가의 반대 방향. 체결: 22:00 봉 시가.
    """
    o, h, l, c = (df[k].values for k in ('open', 'high', 'low', 'close'))
    ref, ent_open = o[0], o[lb]
    if ent_open == ref:
        return None
    sig = -1 if ent_open > ref else 1               # 올랐으면 SHORT, 내렸으면 LONG
    entry = ent_open * (1 + sig * COST)              # 롱은 높게, 숏은 낮게 체결
    tp_px = entry * (1 + sig * TP)
    sl_px = entry * (1 - sig * SL)

    reason, level = 'TIME', c[-1]
    for j in range(lb, len(o)):
        if sig == 1:
            hit_sl, hit_tp = l[j] <= sl_px, h[j] >= tp_px
        else:
            hit_sl, hit_tp = h[j] >= sl_px, l[j] <= tp_px
        if hit_sl:                                    # 같은 봉에서 둘 다 닿으면 손절 우선
            reason, level = 'SL', sl_px
            break
        if hit_tp:
            reason, level = 'TP', tp_px
            break
    px = level * (1 - sig * COST)                     # 청산 비용 반영
    ret = sig * (px - entry) / entry
    return dict(date=df['timestamp'].iloc[lb], dir=sig, reason=reason,
                ret=ret, pnl=ret * NOTIONAL)


# ===================== 데이터 =====================
def fetch_day(day_utc, lb):
    ex = get_ex()
    start = day_utc + timedelta(hours=UTC_START_H) - timedelta(minutes=lb)
    since = int(start.timestamp() * 1000)
    need = lb + WINDOW_MIN
    b = ex.fetch_ohlcv(SYMBOL, '1m', since=since, limit=need)
    if len(b) < need or b[0][0] != since:
        return None
    df = pd.DataFrame(b, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def synth_day(day_utc, lb):
    rng = np.random.default_rng(int(day_utc.timestamp()))
    n = lb + WINDOW_MIN
    c = 60000 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    o = np.concatenate([[60000], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0003, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0003, n)))
    ts = pd.date_range(day_utc.replace(tzinfo=None) + timedelta(hours=UTC_START_H) - timedelta(minutes=lb),
                       periods=n, freq='min')
    return pd.DataFrame({'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': 1.0})


# ===================== 통계 =====================
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
    return f"n={m['n']:>4} 승률={m['wr']:5.1f}% 평균={m['avg']:+.3f}% PF={pf:>5}"


# ===================== 백테스트 =====================
def backtest(days, lb, use_synth):
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    trades = []
    print(f"\n===== 한국시간 {START_H}~{END_H}시 역방향 단타 백테스트 (최근 {days}일, 신호 구간 {lb}분) =====")
    print(f"진입 {NOTIONAL:,.0f} USDT x 1배 | 익절 +{TP * 100:.2f}% / 손절 -{SL * 100:.2f}% | "
          f"수수료 {FEE * 100:.2f}% + 슬리피지 {SLIP * 100:.2f}% (편도)\n")
    for k in range(days, 0, -1):                      # 오늘은 아직 끝나지 않았을 수 있어 제외
        day = today - timedelta(days=k)
        try:
            df = synth_day(day, lb) if use_synth else fetch_day(day, lb)
        except Exception as e:
            print(f"  {day:%Y-%m-%d} 수집 실패: {e}")
            continue
        if df is None:
            continue
        t = run_day(df, lb)
        if t:
            trades.append(t)
        if not use_synth and (days - k + 1) % 100 == 0:
            print(f"  ... {days - k + 1}/{days}일 처리")

    if not trades:
        print("거래가 없습니다 (데이터 없음).")
        return
    pnl = np.array([t['pnl'] for t in trades])
    curve = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:]
    mdd = (curve - peak).min()

    print(f"\n전체: {fmt(stats([t['ret'] for t in trades]))}")
    print(f"누적 손익 {pnl.sum():+,.2f} USDT | 최대 낙폭 {mdd:,.2f} USDT | "
          f"1회 평균 {pnl.mean():+.2f} USDT")
    for dr, name in ((1, '롱'), (-1, '숏')):
        print(f"  {name}: {fmt(stats([t['ret'] for t in trades if t['dir'] == dr]))}")
    cnt = {r: sum(1 for t in trades if t['reason'] == r) for r in ('TP', 'SL', 'TIME')}
    print(f"청산 사유: 익절 {cnt['TP']}회 | 손절 {cnt['SL']}회 | 24시 강제청산 {cnt['TIME']}회")
    wins_net = np.mean([t['ret'] for t in trades if t['reason'] == 'TP']) * 100 if cnt['TP'] else TP * 100
    print(f"\n※ 비용 반영 시 익절 순수익은 약 {wins_net:.2f}%, 손절 순손실은 약 {(SL + 2 * COST) * 100:.2f}% 입니다.")
    print("※ 펀딩비(한국시간 01시/09시/17시 정산이라 이 시간대엔 없음), 호가 체결 오차는 반영되지 않았습니다.")
    print("※ 과거 성과는 미래를 보장하지 않습니다.")


# ===================== 모의매매 (주문 없음) =====================
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    conn = db()
    conn.execute('''CREATE TABLE IF NOT EXISTS night (
        id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT NOT NULL, side TEXT NOT NULL,
        entry_ts TEXT NOT NULL, entry_price REAL NOT NULL, tp_px REAL NOT NULL, sl_px REAL NOT NULL,
        status TEXT DEFAULT 'OPEN', exit_ts TEXT, exit_price REAL, reason TEXT, ret REAL, pnl REAL)''')
    conn.commit()
    conn.close()


def paper_stats():
    conn = db()
    rows = conn.execute("SELECT ret, pnl FROM night WHERE status='CLOSED' ORDER BY id").fetchall()
    conn.close()
    if not rows:
        print("[모의 누적] 완료된 거래 없음")
        return
    print(f"[모의 누적] {fmt(stats([r['ret'] for r in rows]))} | 누적 손익 {sum(r['pnl'] for r in rows):+,.2f} USDT")


def last_price():
    return float(get_ex().fetch_ticker(SYMBOL)['last'])


def enter(now, lb):
    raw = get_ex().fetch_ohlcv(SYMBOL, '1m', limit=lb + 5)
    closed = raw[:-1]                                  # 형성 중인 봉 제외
    ref, last = float(closed[-lb][1]), float(closed[-1][4])
    sig = -1 if last > ref else 1                      # 올랐으면 SHORT, 내렸으면 LONG
    entry = last_price() * (1 + sig * COST)
    tp_px, sl_px = entry * (1 + sig * TP), entry * (1 - sig * SL)
    conn = db()
    conn.execute("INSERT INTO night (day, side, entry_ts, entry_price, tp_px, sl_px) VALUES (?,?,?,?,?,?)",
                 (now.strftime('%Y-%m-%d'), 'long' if sig == 1 else 'short', now.isoformat(), entry, tp_px, sl_px))
    conn.commit()
    conn.close()
    print(f"[모의] 진입 {'LONG' if sig == 1 else 'SHORT'} {entry:,.2f} | 익절 {tp_px:,.2f} | 손절 {sl_px:,.2f} "
          f"(직전 {lb}분 {ref:,.2f} -> {last:,.2f}, {NOTIONAL:,.0f} USDT)")


def close_pos(pos, price, reason, now):
    d = 1 if pos['side'] == 'long' else -1
    px = price * (1 - d * COST)
    ret = d * (px - pos['entry_price']) / pos['entry_price']
    conn = db()
    conn.execute("UPDATE night SET status='CLOSED', exit_ts=?, exit_price=?, reason=?, ret=?, pnl=? WHERE id=?",
                 (now.isoformat(), px, reason, ret, ret * NOTIONAL, pos['id']))
    conn.commit()
    conn.close()
    label = {'TP': '익절', 'SL': '손절', 'TIME': '24시 강제청산'}[reason]
    print(f"[모의] {label} {pos['side']} {ret * 100:+.3f}% ({ret * NOTIONAL:+.2f} USDT)")
    paper_stats()


def paper_main(lb):
    setup_db()
    print(f"\n=== 한국시간 {START_H}~{END_H}시 역방향 단타 모의매매 === {datetime.now(KST):%Y-%m-%d %H:%M}")
    print(f"{SYMBOL} | {NOTIONAL:,.0f} USDT x 1배 | 익절 +{TP * 100:.2f}% / 손절 -{SL * 100:.2f}% | 주문 없음 (Ctrl+C 종료)")
    while True:
        try:
            now = datetime.now(KST)
            in_window = START_H <= now.hour < END_H
            conn = db()
            pos = conn.execute("SELECT * FROM night WHERE status='OPEN'").fetchone()
            done_today = conn.execute("SELECT 1 FROM night WHERE day=?", (now.strftime('%Y-%m-%d'),)).fetchone()
            conn.close()

            if pos:
                price = last_price()
                d = 1 if pos['side'] == 'long' else -1
                if (d == 1 and price <= pos['sl_px']) or (d == -1 and price >= pos['sl_px']):
                    close_pos(pos, price, 'SL', now)
                elif (d == 1 and price >= pos['tp_px']) or (d == -1 and price <= pos['tp_px']):
                    close_pos(pos, price, 'TP', now)
                elif not in_window:
                    close_pos(pos, price, 'TIME', now)
            elif now.hour == START_H and now.minute < 5 and not done_today:
                enter(now, lb)                         # 22:00~22:05 사이에만 진입, 늦게 켜면 그날은 건너뜀
            time.sleep(10)
        except KeyboardInterrupt:
            print("종료")
            break
        except Exception as e:
            print("오류:", e)
            time.sleep(30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, metavar="DAYS", help="백테스트 일수 (예: 365)")
    ap.add_argument("--lookback", type=int, default=LOOKBACK, help="22시 기준 방향 판단 구간(분), 기본 60")
    ap.add_argument("--synthetic", action="store_true", help="가짜 데이터로 동작 점검")
    ap.add_argument("--stats", action="store_true", help="모의매매 누적 성적")
    args = ap.parse_args()
    if args.backtest:
        backtest(args.backtest, args.lookback, args.synthetic)
    elif args.stats:
        setup_db()
        paper_stats()
    else:
        paper_main(args.lookback)