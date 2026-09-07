"""
ETF 日线历史行情高速前复权下载器 (基于东方财富官方前复权接口 HTTP 直连，彻底绝杀除权假跳空)
"""

import sys
import time
import json
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

# 加入父目录
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RAW_DATA_DIR, DATA_DIR, START_DATE
from universe import get_all_symbols


def fetch_qfq_etf(symbol: str, name: str) -> pd.DataFrame:
    """获取单个 ETF 的全量前复权 (QFQ) 日线历史，杜绝拆分与分红假跳空"""
    # 市场标识: 5/6 开头为 1 (上交所), 1/0/3 开头为 0 (深交所)
    secid = f"1.{symbol}" if (symbol.startswith("5") or symbol.startswith("6")) else f"0.{symbol}"
    url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58&klt=101&fqt=1&end=20500101&lmt=3500"
    
    print(f"[*] 正在下载 {symbol} ({name}) 前复权日线行情...")
    for retry in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))
                if "data" in raw_json and raw_json["data"] and "klines" in raw_json["data"]:
                    klines = [x.split(",") for x in raw_json["data"]["klines"]]
                    break
        except Exception as e:
            print(f"[-] 下载失败 (重试 {retry+1}/3): {e}")
            time.sleep(1.0)
    else:
        print(f"[!] 最终下载失败: {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=["date", "open", "close", "high", "low", "volume", "amount", "amplitude"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 过滤起始时间
    df = df[df["date"] >= START_DATE].copy()
    
    # 强制数值类型转换
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    df = df[(df["volume"] > 0) & (df["close"] > 0)].reset_index(drop=True)
    
    # 基础收益与特征工程
    df["symbol"] = symbol
    df["name"] = name
    df["ret_1d"] = df["close"].pct_change().fillna(0.0)
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    
    # 均线系统
    for w in [5, 10, 20, 60, 120, 200]:
        df[f"ma_{w}"] = df["close"].rolling(w, min_periods=w//2).mean()
        df[f"bias_{w}"] = (df["close"] - df[f"ma_{w}"]) / (df[f"ma_{w}"] + 1e-8)
        
    # 动量系统 (过去 20, 60, 120, 252 日前复权收益率)
    for w in [20, 60, 120, 252]:
        df[f"mom_{w}d"] = df["close"].pct_change(w).fillna(0.0)
        
    # 波动率系统 (20日真实波动率与真实波幅)
    df["vol_20d"] = df["ret_1d"].rolling(20, min_periods=5).std() * np.sqrt(242)
    df["vol_60d"] = df["ret_1d"].rolling(60, min_periods=10).std() * np.sqrt(242)
    
    # 夏普动量 (动量除以年化波动率)
    df["sharp_mom_20d"] = df["mom_20d"] / (df["vol_20d"] + 1e-4)
    df["sharp_mom_60d"] = df["mom_60d"] / (df["vol_60d"] + 1e-4)

    return df


def download_all_qfq_etfs():
    all_symbols = get_all_symbols()
    print(f"=== 开始全量下载前复权 (QFQ) ETF 历史数据，共计 {len(all_symbols)} 只标的 ===")
    
    dfs = []
    for sym, info in all_symbols.items():
        df_sym = fetch_qfq_etf(sym, info["name"])
        if not df_sym.empty:
            out_p = RAW_DATA_DIR / f"{sym}.parquet"
            df_sym.to_parquet(out_p, index=False)
            print(f"[+] 保存成功: {sym} -> {out_p} ({len(df_sym)} 行, {df_sym['date'].min().date()} ~ {df_sym['date'].max().date()})")
            dfs.append(df_sym)
        time.sleep(0.15)
        
    if not dfs:
        raise RuntimeError("未能成功下载任何 ETF 数据！")
        
    # 合并全市场面板
    panel_all = pd.concat(dfs, ignore_index=True)
    panel_p = DATA_DIR / "panel_daily.parquet"
    panel_all.to_parquet(panel_p, index=False)
    print(f"\n[***] 前复权全量面板合并完成: {panel_p} (总行数: {len(panel_all)})")
    return panel_all


if __name__ == "__main__":
    download_all_qfq_etfs()
