"""
세 가지 설정을 같은 데이터로 한 번에 백테스트해서 표로 비교합니다.

  A. 원본 v4      : 반전 없음 (INVERT=False, 부분익절/트레일링 등 관리 ON)
  B. 반전 same    : 방향만 반대, SL/TP 거리는 원신호 그대로 (관리 ON)
  C. 반전 mirror  : 방향 반대 + SL/TP 맞바꿈 (관리 OFF, 시간청산만)

사용법 (봇 파일과 같은 폴더에서)
  python comparebacktest.py --days 180
  python comparebacktest.py --days 365 --module bitcoinv5
  python comparebacktest.py --days 180 --lev 1 --frac 0.3      # 사이즈를 줄여서 비교

참고
  - 모든 설정에 같은 사이징(증거금 90% x frac x 레버리지)을 적용하므로 공정한 비교입니다.
  - avg_R / profit_factor 는 사이즈와 거의 무관한 '전략 자체의 우열' 지표입니다.
    return_pct / max_dd_pct 는 사이즈(lev, frac)에 크게 좌우됩니다.
  - AI 판단은 백테스트에 포함되지 않습니다.
"""
import argparse
import importlib
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=180, help="백테스트 기간(일), 180 이상 권장")
ap.add_argument("--module", default="bitcoin_inverse_v5", help="봇 파이썬 파일 이름(.py 제외)")
ap.add_argument("--lev", type=int, default=3, help="고정 레버리지")
ap.add_argument("--frac", type=float, default=1.0, help="증거금 비중 (0.2~1.0, 90%% 한도 대비)")
ap.add_argument("--equity", type=float, default=1000.0, help="시작 자본(USDT)")
args = ap.parse_args()

bot = importlib.import_module(args.module)
bot.BT_LEVERAGE = args.lev
bot.BT_FRAC = args.frac

# ---------- 데이터는 한 번만 내려받아 3가지 설정이 공유 ----------
now_ms = bot.exchange.milliseconds()
day_ms = 86400000
print(f"데이터 수집 중... ({args.days}일)")
d15 = bot.add_indicators(bot.fetch_history('15m', now_ms - (args.days + 3) * day_ms)).reset_index(drop=True)
h1 = bot.add_indicators(bot.fetch_history('1h', now_ms - (args.days + 15) * day_ms)).reset_index(drop=True)
h4 = bot.add_indicators(bot.fetch_history('4h', now_ms - (args.days + 60) * day_ms)).reset_index(drop=True)

CONFIGS = [
    ("A. 원본 v4 (반전X)", dict(INVERT=False, INVERT_MODE="same", MANAGE=True)),
    ("B. 반전 same", dict(INVERT=True, INVERT_MODE="same", MANAGE=True)),
    ("C. 반전 mirror", dict(INVERT=True, INVERT_MODE="mirror", MANAGE=False)),
]

rows = []
detail = {}
for name, cfg in CONFIGS:
    for k, v in cfg.items():
        setattr(bot, k, v)
    print(f"\n>>> 실행 중: {name}")
    res = bot.simulate(d15, h1, h4, equity0=args.equity, verbose=False)
    trades = res.pop("_trades")
    detail[name] = trades
    rows.append({"설정": name, **res})

table = pd.DataFrame(rows).set_index("설정")
cols = ["trades", "win_rate", "avg_R", "avg_win_R", "avg_loss_R", "profit_factor",
        "return_pct", "max_dd_pct", "final_equity"]
table = table[[c for c in cols if c in table.columns]]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(f"\n===== 비교 결과 ({args.days}일, 레버리지 {args.lev}x, 증거금 비중 {args.frac:.0%} of 90%) =====")
print(table.round(2).to_string())

print("\n-- 전략별 평균 R --")
for name, trades in detail.items():
    by = {}
    for t in trades:
        by.setdefault(t['strategy'], []).append(t['R'])
    line = " | ".join(f"{k}: {sum(v) / len(v):+.2f}R ({len(v)}건)" for k, v in by.items()) or "거래 없음"
    print(f"{name:<18} {line}")

print("\n-- 판단 기준 --")
print("  avg_R > 0 그리고 profit_factor > 1.3 그리고 거래 100건 이상 -> 검토할 가치가 있음")
print("  모든 설정이 avg_R < 0 이면 -> 신호 엔진 자체에 우위가 없는 것 (방향을 뒤집어도 해결 안 됨)")
print("  기간을 바꿔서(90/180/365일) 결과가 일관적인지도 확인하세요. 한 기간만 좋은 건 우연일 수 있습니다.")

table.round(4).to_csv("compare_backtest_result.csv", encoding="utf-8-sig")
print("\n결과 저장: compare_backtest_result.csv")