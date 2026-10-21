"""
승률 / 기대값 최적화 + 워크포워드 검증 도구
=====================================================================
목적
  "과거 데이터에서 가장 잘 맞는 설정"을 찾는 것은 쉽습니다. 하지만 그 설정이 '미래'에서도 통하는지는
  별개입니다. 이 도구는 설정을 찾는 과정(최적화)과 그 설정을 처음 보는 구간에서 시험하는 과정(검증)을
  분리해서, 최적화 결과가 진짜인지 우연인지(과최적화) 숫자로 보여줍니다.

동작
  1) 여러 진입 규칙 x 손절/익절 배수 x 보유시간 조합(수천 개)을 전부 시험
  2) 앞 구간(학습)에서 '승률 1등'과 '기대값(평균 R) 1등' 설정을 각각 선택
  3) 뒤 구간(검증, 학습에 쓰지 않은 데이터)에서 그 설정을 그대로 실행해 성적 확인
  4) 이것을 구간을 옮겨가며 반복(워크포워드) -> 검증 구간 성적만 모아서 최종 판정

실행
  python optimize_walkforward.py --days 365              # 실제 바이낸스 데이터 (API 키 불필요)
  python optimize_walkforward.py --days 365 --tf 15m     # 15분봉 (거래 많음, 비용 부담 큼)
  python optimize_walkforward.py --synthetic             # 완전 무작위 가짜 데이터로 과최적화 시연

판정 기준 (검증 구간 합산)
  평균 R > 0.1, 프로핏 팩터 > 1.3, 거래 100건 이상, 그리고 학습 성적과 검증 성적의 차이가 크지 않을 것
"""
import argparse
import itertools
import json
import numpy as np
import pandas as pd

FEE = 0.0005        # 편도 수수료 (시장가 taker)
SLIP = 0.0002       # 편도 슬리피지


# ===================== 데이터 =====================
def fetch(days, tf):
    import ccxt
    ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    sym = 'BTC/USDT:USDT'
    since = ex.milliseconds() - days * 86400000
    step = ex.parse_timeframe(tf) * 1000
    out = []
    print(f"데이터 수집 중... ({days}일, {tf})")
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


def synthetic(days, tf, seed=7):
    """방향 정보가 전혀 없는 무작위 가격 (랜덤워크). 최적화가 우연을 얼마나 '발견'하는지 보여주기 위함."""
    mins = {'15m': 15, '30m': 30, '1h': 60, '4h': 240}[tf]
    n = days * 24 * 60 // mins
    rng = np.random.default_rng(seed)
    sigma = 0.0035 * math_sqrt(mins / 60)
    r = rng.normal(0, sigma, n)
    c = 30000 * np.exp(np.cumsum(r))
    o = np.concatenate([[30000], c[:-1]])
    wick = np.abs(rng.normal(0, sigma * 0.6, n))
    h = np.maximum(o, c) * (1 + wick)
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, sigma * 0.6, n)))
    v = rng.lognormal(5, 0.5, n)
    ts = pd.date_range('2024-01-01', periods=n, freq=f'{mins}min')
    return pd.DataFrame({'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})


def math_sqrt(x):
    return float(np.sqrt(x))


# ===================== 지표 =====================
def indicators(df):
    d = df.copy()
    c, h, l, v = d['close'], d['high'], d['low'], d['volume']
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

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d['bb_up'], d['bb_lo'] = mid + 2 * sd, mid - 2 * sd
    d['don_hi'] = h.rolling(20).max().shift(1)
    d['don_lo'] = l.rolling(20).min().shift(1)
    d['vol_z'] = ((v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)).fillna(0)
    return d


# ===================== 신호 후보 만들기 =====================
def build_signal_sets(d):
    """진입 규칙 후보들. 각 후보는 (신호 발생 봉 인덱스, 방향) 배열.
    신호는 해당 봉의 '종가'에서 계산하고, 진입은 다음 봉 시가에 합니다 (미래 정보 사용 방지)."""
    n = len(d)
    c, rsi, adx = d['close'].values, d['rsi'].values, d['adx'].values
    ema50, ema200 = d['ema50'].values, d['ema200'].values
    bb_up, bb_lo = d['bb_up'].values, d['bb_lo'].values
    don_hi, don_lo, vz = d['don_hi'].values, d['don_lo'].values, d['vol_z'].values
    atr = d['atr'].values
    valid = np.isfinite(atr) & np.isfinite(ema200) & np.isfinite(bb_up) & np.isfinite(don_hi) & (np.arange(n) >= 210)

    up = (c > ema50) & (ema50 > ema200)
    dn = (c < ema50) & (ema50 < ema200)
    sets = {}

    def add(key, sig):
        sig = np.where(valid, sig, 0)
        idx = np.nonzero(sig)[0]
        if len(idx) >= 10:
            sets[key] = (idx, sig[idx].astype(int))

    for flip in (1, -1):
        tag = "역방향" if flip == -1 else "정방향"
        # A. 추세 눌림목: 상승추세에서 RSI 하락 -> 매수 / 하락추세에서 RSI 상승 -> 매도
        for adx_min in (0, 20, 25):
            for rl in (35, 40, 45, 50):
                lg = up & (adx >= adx_min) & (rsi < rl)
                st = dn & (adx >= adx_min) & (rsi > 100 - rl)
                add(f"눌림목|{tag}|adx>={adx_min}|rsi<{rl}", (lg.astype(int) - st.astype(int)) * flip)
        # B. 돌파: 20봉 고가/저가 돌파 + 거래량
        for adx_min in (0, 20, 25):
            for vmin in (0.5, 1.0, 1.5):
                lg = (c > don_hi) & (vz > vmin) & (adx >= adx_min)
                st = (c < don_lo) & (vz > vmin) & (adx >= adx_min)
                add(f"돌파|{tag}|adx>={adx_min}|vol_z>{vmin}", (lg.astype(int) - st.astype(int)) * flip)
        # C. 볼린저 이탈: 하단 이탈+RSI 낮음 -> 매수 / 상단 이탈+RSI 높음 -> 매도
        for rl in (20, 25, 30, 35):
            lg = (c < bb_lo) & (rsi < rl)
            st = (c > bb_up) & (rsi > 100 - rl)
            add(f"볼린저|{tag}|rsi<{rl}", (lg.astype(int) - st.astype(int)) * flip)
    return sets


SL_ATR = (1.0, 1.5, 2.0, 3.0)
TP_ATR = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
HOLD = (12, 24, 48)
EXITS = list(itertools.product(SL_ATR, TP_ATR, HOLD))


# ===================== 시뮬레이션 (겹치는 포지션 없음, 한 번에 하나) =====================
def run(arr, idxs, dirs, lo, hi, sl_atr, tp_atr, hold):
    o, h, l, c, atr = arr
    rs = []
    nxt = lo
    for i, dr in zip(idxs, dirs):
        if i < nxt or i + 1 >= hi:
            continue
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
        if s <= t and s < 10 ** 9:          # 같은 봉에서 둘 다 닿으면 손절 우선 (보수적)
            exit_p, k = sl * (1 - dr * SLIP), s
        elif t < 10 ** 9:
            exit_p, k = tp, t
        else:
            k = end - e
            exit_p = c[end] * (1 - dr * SLIP)
        r = (dr * (exit_p - entry) - 2 * FEE * entry) / sl_d     # 수수료 포함 R 배수
        rs.append(r)
        nxt = e + k
    return rs


def metrics(rs):
    n = len(rs)
    if n == 0:
        return dict(n=0, wr=0.0, avg_R=0.0, pf=0.0, tot_R=0.0, mdd_R=0.0)
    a = np.array(rs)
    w, ls = a[a > 0].sum(), -a[a <= 0].sum()
    cum = np.concatenate([[0], np.cumsum(a)])
    mdd = (np.maximum.accumulate(cum) - cum).max()
    return dict(n=n, wr=float((a > 0).mean() * 100), avg_R=float(a.mean()),
                pf=float(w / ls) if ls > 0 else float('inf'), tot_R=float(a.sum()), mdd_R=float(mdd))


def search(sets, arr, lo, hi, min_trades):
    res = []
    for key, (idx, dirs) in sets.items():
        m = (idx >= lo) & (idx < hi - 1)
        ii, dd = idx[m], dirs[m]
        if len(ii) < min_trades:
            continue
        for sl, tp, hold in EXITS:
            rs = run(arr, ii, dd, lo, hi, sl, tp, hold)
            if len(rs) >= min_trades:
                res.append((key, (sl, tp, hold), metrics(rs)))
    return res


def evaluate(sets, arr, key, ex, lo, hi):
    idx, dirs = sets[key]
    m = (idx >= lo) & (idx < hi - 1)
    return run(arr, idx[m], dirs[m], lo, hi, *ex)


def fmt(m):
    pf = f"{m['pf']:.2f}" if np.isfinite(m['pf']) else "inf"
    return f"n={m['n']:>3} 승률={m['wr']:5.1f}% 평균R={m['avg_R']:+.3f} PF={pf:>5} 총R={m['tot_R']:+7.1f}"


def pick(res, objective):
    if objective == "wr":
        return max(res, key=lambda x: (x[2]['wr'], x[2]['avg_R']))
    return max(res, key=lambda x: x[2]['avg_R'])


# ===================== 메인 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--tf", default="1h", choices=["15m", "30m", "1h", "4h"])
    ap.add_argument("--segments", type=int, default=6, help="데이터를 몇 구간으로 나눌지 (워크포워드)")
    ap.add_argument("--min-trades", type=int, default=25, help="학습 구간에서 인정할 최소 거래 수")
    ap.add_argument("--synthetic", action="store_true", help="무작위 가짜 데이터로 실행 (과최적화 시연)")
    args = ap.parse_args()

    df = synthetic(args.days, args.tf) if args.synthetic else fetch(args.days, args.tf)
    if args.synthetic:
        print("※ 방향 정보가 전혀 없는 무작위 가격으로 실행합니다. 여기서 '좋은 설정'이 나온다면 전부 우연입니다.")
    d = indicators(df)
    arr = tuple(d[k].values for k in ('open', 'high', 'low', 'close', 'atr'))
    n = len(d)
    sets = build_signal_sets(d)
    n_combo = len(sets) * len(EXITS)
    print(f"봉 {n}개 | 진입 규칙 후보 {len(sets)}개 x 청산 조합 {len(EXITS)}개 = {n_combo}개 설정 시험")

    start = 210
    edges = np.linspace(start, n, args.segments + 1).astype(int)

    # ---------- 워크포워드 ----------
    oos = {"wr": [], "exp": []}
    print("\n===== 워크포워드 (학습 구간에서 고른 설정을, 처음 보는 검증 구간에서 시험) =====")
    for k in range(2, args.segments):
        tr_lo, tr_hi, te_lo, te_hi = edges[0], edges[k], edges[k], edges[k + 1]
        res = search(sets, arr, tr_lo, tr_hi, args.min_trades)
        if not res:
            print(f"[폴드 {k - 1}] 학습 구간에서 조건을 만족하는 설정이 없습니다.")
            continue
        print(f"\n[폴드 {k - 1}] 학습 {df['timestamp'].iloc[tr_lo]:%Y-%m-%d}~{df['timestamp'].iloc[tr_hi - 1]:%Y-%m-%d}"
              f" | 검증 {df['timestamp'].iloc[te_lo]:%Y-%m-%d}~{df['timestamp'].iloc[te_hi - 1]:%Y-%m-%d}")
        for name, obj in (("승률 최대화 선택", "wr"), ("기대값 최대화 선택", "exp")):
            key, ex, m_tr = pick(res, obj)
            rs = evaluate(sets, arr, key, ex, te_lo, te_hi)
            m_te = metrics(rs)
            oos[obj] += rs
            print(f"  {name}: {key} | SL {ex[0]}ATR TP {ex[1]}ATR 보유 {ex[2]}봉")
            print(f"     학습  {fmt(m_tr)}")
            print(f"     검증  {fmt(m_te)}")

    print("\n===== 검증 구간 합산 (이 숫자가 '실전에서 기대할 수 있는 성적'에 가장 가깝습니다) =====")
    verdicts = {}
    for name, obj in (("승률 최대화 방식", "wr"), ("기대값 최대화 방식", "exp")):
        m = metrics(oos[obj])
        verdicts[obj] = m
        print(f"  {name:<14} {fmt(m)}")

    # ---------- 전체 구간 최적화 (참고용) ----------
    print("\n===== 전체 구간 최적화 결과 (참고용: 이 숫자는 이미 본 데이터에 맞춘 것이라 과장되어 있습니다) =====")
    res_all = search(sets, arr, edges[0], n, args.min_trades)
    if res_all:
        print("\n[승률 상위 5개]")
        for key, ex, m in sorted(res_all, key=lambda x: (-x[2]['wr'], -x[2]['avg_R']))[:5]:
            print(f"  {key} | SL {ex[0]} TP {ex[1]} 보유 {ex[2]} | {fmt(m)}")
        print("\n[기대값(평균 R) 상위 5개]")
        top = sorted(res_all, key=lambda x: -x[2]['avg_R'])[:5]
        for key, ex, m in top:
            print(f"  {key} | SL {ex[0]} TP {ex[1]} 보유 {ex[2]} | {fmt(m)}")
        pos = sum(1 for _, _, m in res_all if m['avg_R'] > 0)
        print(f"\n전체 {len(res_all)}개 설정 중 평균 R > 0 인 설정: {pos}개 ({pos / len(res_all) * 100:.1f}%)")
        print("-> 수천 개를 시험하면 우연히 좋아 보이는 설정이 반드시 나옵니다. 위 '검증 구간 합산'이 진짜 판정입니다.")

        best = top[0]
        m_exp = verdicts["exp"]
        validated = bool(m_exp['n'] >= 100 and m_exp['avg_R'] > 0.1 and m_exp['pf'] > 1.3)
        with open("best_params.json", "w", encoding="utf-8") as f:
            json.dump({"rule": best[0], "sl_atr": best[1][0], "tp_atr": best[1][1], "hold_bars": best[1][2],
                       "timeframe": args.tf, "in_sample": best[2], "out_of_sample_validated": validated,
                       "oos_summary": m_exp}, f, ensure_ascii=False, indent=2, default=float)
        print("\n결과 저장: best_params.json")

    # ---------- 최종 판정 ----------
    m = verdicts["exp"]
    ok = m['n'] >= 100 and m['avg_R'] > 0.1 and m['pf'] > 1.3
    print("\n===== 최종 판정 =====")
    if ok:
        print("검증 구간에서도 기준을 통과했습니다. 그래도 곧바로 실거래하지 말고,")
        print("  - 다른 기간(--days 730 등)과 다른 시기에서도 통과하는지,")
        print("  - 소액 모의/실거래로 체결·슬리피지가 가정과 맞는지 먼저 확인하세요.")
    else:
        print("검증 구간 성적이 기준(평균 R > 0.1, PF > 1.3, 거래 100건 이상)을 통과하지 못했습니다.")
        print("학습 구간에서 아무리 좋아 보여도 처음 보는 구간에서 유지되지 않으면, 그 설정은 과거에 맞춘 우연일 가능성이 큽니다.")
        print("이 경우 해당 설정으로 실거래를 하지 마세요.")


if __name__ == "__main__":
    main()