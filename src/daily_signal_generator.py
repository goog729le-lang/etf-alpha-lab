"""
每日 14:45 盘中交易行为生成引擎 (Daily Action Generator)
针对当前冠军策略 (Domestic_Sector_Mom40_MA60_MONTHLY_DYNAMIC_CANARY)，
在交易日 14:45 自动抓取实时盘口行情，结合历史 K 线计算即时信号，
生成确定的实盘调仓行为建议 (HOLD / BUY / SELL / STAY_CASH)。
"""

import sys
import json
import urllib.request
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 加入项目路径
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from config import RAW_DATA_DIR, RULES_DIR, RESULTS_DIR
from universe import get_all_symbols


SIGNALS_DIR = RESULTS_DIR / "daily_signals"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_realtime_quotes(symbols: list) -> dict:
    """从稳定高速的行情接口批量抓取实时盘口价格 (14:45 点位)"""
    def to_tx_code(s):
        prefix = "sh" if s.startswith("5") or s.startswith("6") else "sz"
        return f"{prefix}{s}"

    q_str = ",".join([to_tx_code(s) for s in symbols])
    url = f"http://qt.gtimg.cn/q={q_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    quotes = {}
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            text = resp.read().decode("gbk", errors="ignore")
            for line in text.strip().split(";"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("~")
                if len(parts) > 30:
                    code = parts[2]
                    name = parts[1]
                    price = float(parts[3])
                    prev_close = float(parts[4])
                    open_p = float(parts[5])
                    volume = float(parts[6]) # 手
                    dt_str = parts[30] # YYYYMMDDHHMMSS
                    quotes[code] = {
                        "name": name,
                        "price": price if price > 0 else prev_close,
                        "prev_close": prev_close,
                        "open": open_p,
                        "volume": volume,
                        "timestamp": dt_str,
                    }
    except Exception as e:
        print(f"[!] 实时行情拉取异常: {e}，将尝试回退至历史最新收盘价")
    return quotes


def is_trade_day(today_dt: pd.Timestamp) -> bool:
    """通过是否有有效盘中交易或非周末进行交易日初筛"""
    if today_dt.weekday() >= 5: # 周六、周日
        return False
    return True


def generate_today_action():
    # 1. 加载冠军策略元数据
    champ_json_p = RULES_DIR / "champion_strategy.json"
    if not champ_json_p.exists():
        raise FileNotFoundError(f"未找到冠军策略文件: {champ_json_p}")
    
    with open(champ_json_p, "r", encoding="utf-8") as fp:
        champ_meta = json.load(fp)
        
    p = champ_meta["params"]
    mom_w = p.get("mom", 40)
    ma_w = p.get("ma", 60)
    def_m = p.get("def", "DYNAMIC_CANARY")
    rebal_freq = p.get("rebal", "MONTHLY")
    
    print(f"[*] 当前冠军策略: {champ_meta['champion_name']}")
    print(f"[*] 策略参数: 动量回溯={mom_w}日, 均线滤波={ma_w}日, 调仓频率={rebal_freq}, 防守模式={def_m}")

    # 2. 读取历史持仓最新记录 (确认昨日/当前实盘所持标的)
    nav_csv_p = RESULTS_DIR / "reports" / "champion_daily_nav.csv"
    if nav_csv_p.exists():
        df_nav = pd.read_csv(nav_csv_p, index_col=0, parse_dates=True)
        prev_holding = df_nav["holding_symbol"].iloc[-1]
        last_hist_date = df_nav.index[-1]
    else:
        prev_holding = "CASH"
        last_hist_date = pd.Timestamp.now().normalize()

    # 3. 收集策略所有涉及标的
    domestic_pool = [
        "159915", "512480", "515050", "588000", "512000", "512800",
        "159928", "512690", "512010", "512660", "516160"
    ]
    all_needed_syms = list(set(domestic_pool + ["510300", "518880", "511010", "510880"]))
    all_sym_dict = get_all_symbols()

    # 4. 加载本地日线数据
    panel_data = {}
    for sym in all_needed_syms:
        p_path = RAW_DATA_DIR / f"{sym}.parquet"
        if p_path.exists():
            df_sym = pd.read_parquet(p_path)
            df_sym["date"] = pd.to_datetime(df_sym["date"])
            panel_data[sym] = df_sym.sort_values("date").reset_index(drop=True)

    # 5. 拉取实时 14:45 盘口价格
    quotes = fetch_realtime_quotes(all_needed_syms)
    now_ts = datetime.now()
    today_date_str = now_ts.strftime("%Y-%m-%d")
    today_dt = pd.to_datetime(today_date_str)

    # 6. 将今日盘中行情拼接进历史序列 (作为最新未收盘 Bar)
    close_series_dict = {}
    for sym in all_needed_syms:
        if sym in panel_data:
            s_df = panel_data[sym].copy()
            # 如果本地最新数据已经包含今日，则直接使用
            if s_df["date"].iloc[-1].strftime("%Y-%m-%d") == today_date_str:
                s_series = s_df.set_index("date")["close"]
            else:
                # 拼接今日 14:45 价格
                curr_price = quotes.get(sym, {}).get("price", s_df["close"].iloc[-1])
                today_row = pd.Series([curr_price], index=[today_dt])
                s_series = pd.concat([s_df.set_index("date")["close"], today_row])
            close_series_dict[sym] = s_series

    df_close = pd.DataFrame(close_series_dict).sort_index().ffill()
    
    # 7. 计算关键技术指标
    df_ma = df_close.rolling(ma_w, min_periods=ma_w // 2).mean()
    df_mom = df_close.pct_change(mom_w, fill_method=None)

    today_p = df_close.loc[today_dt]
    today_m = df_ma.loc[today_dt]
    today_mom = df_mom.loc[today_dt]

    # 大盘金丝雀 (510300) 状态
    c_p = today_p.get("510300", np.nan)
    c_m = today_m.get("510300", np.nan)
    is_market_bull = bool(pd.notna(c_p) and pd.notna(c_m) and c_p > c_m)
    market_status = "多头运行 (进攻模式)" if is_market_bull else "跌破均线 (防守模式)"

    # 8. 判断今日是否为月末调仓再平衡日
    # 若下一个自然日跨月，或今日为周五且下周一跨月，则触发月末决策
    is_month_end = False
    next_day = today_dt + pd.Timedelta(days=1)
    if next_day.month != today_dt.month:
        is_month_end = True
    elif today_dt.weekday() == 4: # 周五，看下周一
        mon_day = today_dt + pd.Timedelta(days=3)
        if mon_day.month != today_dt.month:
            is_month_end = True

    # 9. 状态机推演
    target_holding = prev_holding
    action_type = "HOLD"
    action_reason = "今日非月末调仓日，且持仓未触及止损线，继续保持原持仓。"

    # 9.1 日度止损检查 (最高优先级)
    stop_loss_triggered = False
    if prev_holding != "CASH":
        p_held = today_p.get(prev_holding, np.nan)
        m_held = today_m.get(prev_holding, np.nan)
        if pd.notna(p_held) and pd.notna(m_held) and p_held < m_held * 0.98:
            stop_loss_triggered = True
            target_holding = "CASH"
            action_type = "SELL"
            loss_pct = (p_held / m_held - 1.0) * 100.0
            action_reason = f"【触发严苛止损断路器】持仓标的 {prev_holding} ({all_sym_dict.get(prev_holding, {}).get('name', '')}) 现价 {p_held:.3f} 跌破 {ma_w}日均线 ({m_held:.3f}) 达到 {loss_pct:.2f}% (超过 2.0% 止损阈值)，必须立即清仓切回现金 CASH 避险！"

    # 9.2 月末再平衡决策 (若未触发盘中毒性止损)
    if not stop_loss_triggered and is_month_end:
        if is_market_bull:
            # 行业池中选拔
            valid_atk = [
                s for s in domestic_pool
                if pd.notna(today_p.get(s)) and pd.notna(today_m.get(s)) and today_p[s] > today_m[s]
                and pd.notna(today_mom.get(s)) and today_mom[s] > 0
            ]
            if valid_atk:
                valid_scores = today_mom[valid_atk].dropna()
                target_holding = valid_scores.idxmax() if len(valid_scores) > 0 else "CASH"
                action_reason = f"【月末多头再平衡】大盘处于多头环境，行业动量池中评选出龙头标的 {target_holding} ({all_sym_dict.get(target_holding, {}).get('name', '')})，40日动量为 {valid_scores[target_holding]*100:+.2f}%。"
            else:
                target_holding = "CASH"
                action_reason = "【月末防守】大盘虽在均线上方，但 11 大行业无一满足多头动量过滤条件，全部切回现金 CASH。"
        else:
            # 大盘走熊 -> 动态金丝雀防守
            if def_m == "DYNAMIC_CANARY":
                p_gold, m_gold, sc_gold = today_p.get("518880"), today_m.get("518880"), today_mom.get("518880")
                p_bond, m_bond, sc_bond = today_p.get("511010"), today_m.get("511010"), today_mom.get("511010")
                gold_ok = pd.notna(p_gold) and pd.notna(m_gold) and p_gold > m_gold and pd.notna(sc_gold) and sc_gold > 0
                bond_ok = pd.notna(p_bond) and pd.notna(m_bond) and p_bond > m_bond and pd.notna(sc_bond) and sc_bond > 0
                if gold_ok and bond_ok:
                    target_holding = "518880" if sc_gold >= sc_bond else "511010"
                    asset_name = all_sym_dict.get(target_holding, {}).get("name", "")
                    action_reason = f"【月末动态金丝雀】大盘跌破均线，避险资产中黄金与国债皆走强，动量更优选持有 {target_holding} ({asset_name})。"
                elif gold_ok:
                    target_holding = "518880"
                    action_reason = "【月末动态金丝雀】大盘跌破均线，国债走弱，黄金站上均线，持有黄金 518880。"
                elif bond_ok:
                    target_holding = "511010"
                    action_reason = "【月末动态金丝雀】大盘跌破均线，黄金走弱，国债站上均线，持有国债 511010。"
                else:
                    target_holding = "CASH"
                    action_reason = "【月末动态金丝雀】大盘走熊且黄金、国债均跌破均线，100% 切换为 CASH 现金空仓避险。"
            else:
                target_holding = def_m

        if target_holding != prev_holding:
            action_type = "BUY" if prev_holding == "CASH" else "SWITCH"
        else:
            action_type = "HOLD" if target_holding != "CASH" else "STAY_CASH"

    # 9.3 若未调仓未止损，明确行动分类
    if not stop_loss_triggered and not is_month_end:
        if target_holding == "CASH":
            action_type = "STAY_CASH"
            action_reason = "当前处于空仓现金观望期，大盘或持仓未到调仓时机，保持现金。"
        else:
            action_type = "HOLD"
            p_held = today_p.get(target_holding, 0.0)
            m_held = today_m.get(target_holding, 0.0)
            action_reason = f"继续持有 {target_holding} ({all_sym_dict.get(target_holding, {}).get('name', '')})。现价 {p_held:.3f}，{ma_w}日均线 {m_held:.3f}，运行正常无破位。"

    # 10. 汇总输出结构体
    action_result = {
        "generated_at": now_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": today_date_str,
        "action_type": action_type, # HOLD / BUY / SELL / SWITCH / STAY_CASH
        "target_holding": target_holding,
        "target_name": all_sym_dict.get(target_holding, {}).get("name", "现金 (CASH)"),
        "prev_holding": prev_holding,
        "prev_name": all_sym_dict.get(prev_holding, {}).get("name", "现金 (CASH)"),
        "is_month_end": is_month_end,
        "stop_loss_triggered": stop_loss_triggered,
        "market_benchmark": {
            "symbol": "510300 (沪深300ETF)",
            "current_price": round(float(c_p), 3) if pd.notna(c_p) else None,
            "ma_filter": round(float(c_m), 3) if pd.notna(c_m) else None,
            "status": market_status,
        },
        "action_reason": action_reason,
        "champion_strategy": champ_meta["champion_name"],
    }

    # 导出 JSON
    json_path = SIGNALS_DIR / "latest_action.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(action_result, fp, indent=4, ensure_ascii=False)
    print(f"[+] 今日交易指令已生成 JSON: {json_path}")

    # 导出格式化 Markdown 研报 (供 GitHub Step Summary 展现)
    md_path = SIGNALS_DIR / "LATEST_ACTION.md"
    write_action_markdown(action_result, md_path)
    print(f"[+] 今日交易指令已生成 Markdown: {md_path}")
    return action_result


def write_action_markdown(res: dict, md_path: Path):
    action_icon = {
        "HOLD": "🛡️ **【继续持有 (HOLD)】**",
        "BUY": "🚀 **【开仓买入 (BUY)】**",
        "SELL": "⚠️ **【止损清仓 (SELL)】**",
        "SWITCH": "🔄 **【调仓换标的 (SWITCH)】**",
        "STAY_CASH": "☕ **【保持现金 (STAY CASH)】**",
    }.get(res["action_type"], res["action_type"])

    content = f"""# 🔔 ETF Alpha Lab 每日实盘推演与决策指令

> **生成时间**: `{res['generated_at']}` (北京时间盘中 14:45 真实快照)  
> **所处交易日**: `{res['trade_date']}`  
> **服务冠军模型**: `{res['champion_strategy']}`  

---

## 一、 今日核心操作指令

### {action_icon}

| 项目 | 决策状态 | 详情说明 |
| :--- | :---: | :--- |
| **当前持有标的** | `{res['prev_holding']}` | {res['prev_name']} |
| **今日目标持仓** | **`{res['target_holding']}`** | **{res['target_name']}** |
| **操作建议动作** | **`{res['action_type']}`** | 建议于 14:50 ~ 15:00 收盘前执行到位 |
| **月末再平衡触发** | `{'是 (Month-End)' if res['is_month_end'] else '否 (常规交易日)'}` | 动量调仓窗口判断 |
| **日度止损断路器** | `{'🚨 已触发 (跌破均线2%)' if res['stop_loss_triggered'] else '正常 (安全阈值内)'}` | 风险控制硬闸门 |

---

## 二、 市场大盘多空金丝雀状态

* **大盘基准**: `{res['market_benchmark']['symbol']}`
* **即时价格 (14:45)**: `{res['market_benchmark']['current_price']}`
* **60日生命线**: `{res['market_benchmark']['ma_filter']}`
* **多空研判**: **{res['market_benchmark']['status']}**

---

## 三、 决策逻辑与归因

> {res['action_reason']}

---
*注：本指令由系统在交易日 14:45 依据前复权历史与盘中实时行情纯自动计算输出，杜绝任何人工随意干预与主观情绪偏差。*
"""
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write(content)


if __name__ == "__main__":
    generate_today_action()
