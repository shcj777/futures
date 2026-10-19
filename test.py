import os, requests, ccxt
from dotenv import load_dotenv
load_dotenv(override=True)

print("내 공인 IP:", requests.get("https://api.ipify.org").text)
print("ccxt 버전:", ccxt.__version__)

ex = ccxt.binance({
    "apiKey": os.getenv("BINANCE_API_KEY").strip(),
    "secret": os.getenv("BINANCE_SECRET_KEY").strip(),
    "options": {"defaultType": "future", "portfolioMargin": True, "fetchCurrencies": False},
})
print(ex.papiGetBalance())