"""
v4 -> v5 패치 스크립트
  AI 가 '거래당 위험 비율'과 '레버리지(5~20배)'를 정하고, 코드가 안전장치로 한도를 강제합니다.

사용법 (원본 파일과 같은 폴더에서)
  python make_v5.py bitcoinv3.py bitcoinv5.py
  python bitcoinv5.py --dry          # 먼저 신호만 확인
  python bitcoinv5.py                # 실거래
"""
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "bitcoinv3.py"
dst = sys.argv[2] if len(sys.argv) > 2 else "bitcoinv5.py"
code = open(src, encoding="utf-8").read()


def replace_once(text, old, new):
    assert text.count(old) == 1, f"패치 위치를 찾지 못했습니다: {old[:60]!r}"
    return text.replace(old, new)


def replace_slice(text, start_marker, end_marker, new_block, start_from=0):
    s = text.index(start_marker, start_from)
    e = text.index(end_marker, s)
    return text[:s] + new_block + text[e:]


# ---------------------------------------------------------------- 1) 설정
code = replace_once(code, '''RISK_PER_TRADE = 0.005       # 거래당 최대 손실 = 자본의 0.5%
MAX_LEVERAGE = 5
MAX_MARGIN_USAGE = 0.30
''', '''RISK_PER_TRADE = 0.005       # AI 실패/백테스트 기본값 (거래당 손실 = 자본의 0.5%)
MIN_RISK = 0.005             # AI 가 고를 수 있는 위험 비율 하한 0.5%
MAX_RISK = 0.02              # AI 가 고를 수 있는 위험 비율 상한 2.0% (하드캡)
MIN_LEVERAGE = 1             # AI 레버리지 하한
MAX_LEVERAGE = 5            # AI 레버리지 상한 (하드캡)
MAX_MARGIN_USAGE = 0.90      # 증거금으로 쓸 수 있는 자본 비율 (입금액의 최대 90%)
LIQ_SAFETY = 3.0             # 청산가까지 거리가 SL 거리의 최소 3배 이상이어야 진입
''')

# 시험용 200 USDT 상한 해제 (.env 에 MAX_NOTIONAL_USD 를 지정하면 그 값 사용)
code = replace_once(code, 'os.getenv("MAX_NOTIONAL_USD", "200")', 'os.getenv("MAX_NOTIONAL_USD", "0")')

# ---------------------------------------------------------------- 2) 포지션 사이징
CALC_POSITION = r'''def calc_position(equity, entry, sl_dist, risk_pct=RISK_PER_TRADE, leverage=MIN_LEVERAGE,
                  notional_cap=None):
    """위험 기반 사이징 + AI 레버리지.
    - 손절 시 손실 = 자본 x risk_pct (MIN_RISK~MAX_RISK 로 강제 클램프)
    - 레버리지는 MIN~MAX 로 클램프하고, 청산가가 손절가보다 충분히 멀도록 추가 제한
    - 명목금액 <= 자본 x MAX_MARGIN_USAGE x 레버리지 (입금액 한도)"""
    risk_pct = min(MAX_RISK, max(MIN_RISK, float(risk_pct)))
    sl_pct = sl_dist / entry
    max_lev_liq = int(0.8 / (LIQ_SAFETY * sl_pct))      # 청산 안전거리를 지키는 최대 레버리지
    lev = int(min(MAX_LEVERAGE, max(MIN_LEVERAGE, int(leverage)), max_lev_liq))
    if lev < MIN_LEVERAGE:
        return None                                       # 손절폭이 너무 넓어 안전하지 않음
    risk_amt = equity * risk_pct
    notional = (risk_amt / sl_dist) * entry
    cap = equity * MAX_MARGIN_USAGE * lev
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
    return {"qty": qty, "notional": notional, "leverage": lev, "risk_amt": qty * sl_dist}


'''
code = replace_slice(code, "def calc_position(", "# ===================== 거래소 헬퍼", CALC_POSITION)

# ---------------------------------------------------------------- 3) AI 검토
LLM_REVIEW = r'''def llm_review(sig, d, h1, h4):
    """신호 승인/거부 + 위험 비율 + 레버리지. 실패 시 None. (값은 코드가 한 번 더 클램프)"""
    global _client
    try:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(timeout=20)
        c = d.iloc[-1]
        payload = {
            "signal": {k: sig[k] for k in ("strategy", "side", "score", "rr", "reasons")},
            "stop_distance_pct": round(sig['sl_dist'] / sig['entry'] * 100, 3),
            "price": round(float(c['close']), 1),
            "15m": {k: round(float(c[k]), 2) for k in ("rsi", "adx", "atr", "vol_z", "macd_h")},
            "1h": {k: round(float(h1[k]), 2) for k in ("rsi", "adx")} | {"trend": bias(h1, 'ema20', 'ema50')},
            "4h": {k: round(float(h4[k]), 2) for k in ("rsi", "adx")} | {"trend": bias(h4, 'ema50', 'ema200')},
            "last_6_candles_15m": d.tail(6)[['open', 'high', 'low', 'close']].round(1).values.tolist(),
            "news": fetch_news(),
            "strategy_recent_perf": strategy_stats(),
            "limits": {"risk_pct": [MIN_RISK, MAX_RISK], "leverage": [MIN_LEVERAGE, MAX_LEVERAGE]},
        }
        sys_prompt = (
            "You are the position-sizing and risk manager of an automated BTC futures bot. A rule-based engine "
            "found a setup with high reward-to-risk (low win rate is by design). Decide: (1) APPROVE or VETO, "
            "(2) risk_pct = fraction of account equity lost if the stop is hit, within the given limits, "
            "(3) leverage as an integer within the given limits. Notes: position size is derived from risk_pct "
            "and the stop distance, so leverage does NOT change the loss at the stop; it only sets how much "
            "margin is used and how far liquidation is. Use LOW values (risk near the minimum, leverage near "
            "the minimum) by default. Use higher values ONLY when 1h and 4h trends agree with the side, the "
            "score is high, volume confirms, there is no opposing news, and the strategy's recent performance "
            "is positive. Use lower values after poor recent performance, high volatility, or conflicting "
            "signals. VETO only with concrete evidence (opposing major news, clear higher-timeframe conflict, "
            "clearly poor recent performance). "
            'Reply ONLY JSON: {"verdict":"APPROVE"|"VETO","risk_pct":number,"leverage":integer,'
            '"reason":"<=200 chars"}')
        r = _client.chat.completions.create(
            model=LLM_MODEL, max_tokens=200, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        out = json.loads(r.choices[0].message.content)
        out["verdict"] = "VETO" if str(out.get("verdict", "")).upper() == "VETO" else "APPROVE"
        out["risk_pct"] = min(MAX_RISK, max(MIN_RISK, float(out.get("risk_pct", MIN_RISK))))
        out["leverage"] = int(min(MAX_LEVERAGE, max(MIN_LEVERAGE, int(out.get("leverage", MIN_LEVERAGE)))))
        return out
    except Exception as e:
        print("LLM 검토 실패:", e)
        return None


'''
code = replace_slice(code, "def llm_review(", "# ===================== 포지션 관리", LLM_REVIEW)

# ---------------------------------------------------------------- 4) 신호 사이클
CYCLE = r'''    risk_pct, lev_pick, note = RISK_PER_TRADE, MIN_LEVERAGE, ""
    if USE_LLM:
        rv = llm_review(sig, d15, h1, h4)
        if rv is None:
            if sig['score'] < 75:
                save_signal(sig, False, "LLM 실패 + 점수 미달")
                return
            note = "LLM 실패, 기본값(최소 위험/최소 레버리지)"
        elif rv['verdict'] == 'VETO':
            print("AI 거부:", rv.get('reason'))
            save_signal(sig, False, "VETO: " + str(rv.get('reason')))
            return
        else:
            risk_pct, lev_pick, note = rv['risk_pct'], rv['leverage'], str(rv.get('reason', ''))
            print(f"AI 승인 (위험 {risk_pct * 100:.2f}%, 레버리지 {lev_pick}x): {note}")

    pos = calc_position(equity, sig['entry'], sig['sl_dist'], risk_pct, lev_pick, MAX_NOTIONAL_USD or None)
'''
s = code.index('    size_mult, note = 1.0, ""')
e = code.index("    if not pos:", s)
code = code[:s] + CYCLE + code[e:]

code = code.replace("Bitcoin Bot v4 시작", "Bitcoin Bot v5 시작 (AI 사이징/레버리지)")

open(dst, "w", encoding="utf-8").write(code)
print(f"완료: {dst}")