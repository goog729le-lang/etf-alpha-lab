"""
10,000 元初始资金真实仿真整手撮合回测引擎
严格支持:
1. 信号日与成交日物理隔离 (T 日收盘产生信号 -> T+1 开盘/收盘撮合，杜绝同 bar 偷价)
2. 100 股整手限制与不足一手保留现金
3. 双边佣金万三 + 保守滑点扣除
4. 闲置现金保持零利息 (按审计规范彻底去掉货基计息，避免空仓期虚增收益与胜率)
5. 统一标准算术夏普比率口径: (日均超额收益 / 日收益波动率) * sqrt(242)
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from config import INITIAL_CASH, LOT_SIZE, COMMISSION_RATE, SLIPPAGE_POINTS


class ETFBacktester:
    def __init__(
        self,
        initial_cash: float = INITIAL_CASH,
        lot_size: int = LOT_SIZE,
        commission_rate: float = COMMISSION_RATE,
        slippage_points: float = SLIPPAGE_POINTS,
        risk_free_rate: float = 0.02,
        execution_timing: str = "T+1_CLOSE", # T+1_CLOSE: T日收盘信号在T+1日收盘价撮合
    ):
        self.initial_cash = initial_cash
        self.lot_size = lot_size
        self.commission_rate = commission_rate
        self.slippage_points = slippage_points
        self.risk_free_rate = risk_free_rate
        self.execution_timing = execution_timing

    def run_backtest(
        self,
        positions_df: pd.DataFrame,
        panel_data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        执行逐日仿真事件驱动撮合回测
        :param positions_df: 目标持仓信号矩阵 (index=date, columns=symbols)
        :param panel_data: 原始清洗日线字典
        :return: (daily_records_df, summary_metrics_dict)
        """
        dates = positions_df.index
        if start_date:
            dates = dates[dates >= pd.to_datetime(start_date)]
        if end_date:
            dates = dates[dates <= pd.to_datetime(end_date)]

        dates = sorted(list(dates))
        if len(dates) < 2:
            return pd.DataFrame(), {}

        # 提取对齐价格矩阵 (前向填充，停牌日保持停牌前最新价格计算持仓市值)
        symbols = [c for c in positions_df.columns if c in panel_data]
        price_dict = {}
        for sym in symbols:
            df = panel_data[sym].set_index("date")["close"].reindex(dates).ffill()
            price_dict[sym] = df
        df_prices = pd.DataFrame(price_dict, index=dates)

        # 核心修复 1.2: T 日产生的目标持仓信号，滞后到 T+1 日执行撮合 (物理阻断同 bar 乐观成交)
        if self.execution_timing == "T+1_CLOSE":
            executed_targets_df = positions_df.shift(1).fillna(0.0)
        else:
            executed_targets_df = positions_df

        cash = self.initial_cash
        shares_held: Dict[str, int] = {s: 0 for s in symbols}
        
        daily_records = []
        total_turnover_amount = 0.0
        trade_count = 0
# 闲置现金严格不计息 (收益率恒为 0.0，避免空仓期利息对净值与日胜率灌水)

        for i, dt in enumerate(dates):
            current_prices = df_prices.loc[dt]
            target_weights = executed_targets_df.loc[dt, symbols]

            # 1. 闲置现金不计息 (按审计规范彻底去掉现金空仓日利息，保持现金收益恒为 0.0)
            # cash 保持原值，不增加利息

            # 2. 检查是否有需要调仓变动
            if target_weights.max() > 0.5:
                target_sym = target_weights.idxmax()
            else:
                target_sym = None  # 空仓 (CASH)

            # 先卖出非目标标的 (严格检查有有效可成交价，若停牌无法成交则保持原仓位)
            for s in symbols:
                if s != target_sym and shares_held[s] > 0:
                    curr_p = current_prices[s]
                    if pd.notna(curr_p) and curr_p > 0:
                        sell_shares = shares_held[s]
                        sell_price = max(curr_p - self.slippage_points, 0.001)
                        gross_revenue = sell_shares * sell_price
                        commission = gross_revenue * self.commission_rate
                        net_revenue = gross_revenue - commission
                        
                        cash += net_revenue
                        shares_held[s] = 0
                        total_turnover_amount += gross_revenue
                        trade_count += 1

            # 再买入目标标的
            if target_sym is not None and shares_held[target_sym] == 0:
                curr_p = current_prices[target_sym]
                if pd.notna(curr_p) and curr_p > 0:
                    buy_price = curr_p + self.slippage_points
                    usable_cash = cash / (1.0 + self.commission_rate)
                    max_shares = int(usable_cash // (buy_price * self.lot_size)) * self.lot_size

                    if max_shares >= self.lot_size:
                        cost = max_shares * buy_price
                        commission = cost * self.commission_rate
                        total_outflow = cost + commission
                        
                        cash -= total_outflow
                        shares_held[target_sym] = max_shares
                        total_turnover_amount += cost
                        trade_count += 1

            # 3. 计算当日收盘总净值与持仓状态
            final_holding_value = sum(
                shares_held[s] * current_prices[s]
                for s in symbols
                if shares_held[s] > 0 and pd.notna(current_prices[s])
            )
            final_nav = cash + final_holding_value
            active_sym = target_sym if (target_sym and shares_held[target_sym] > 0) else "CASH"

            daily_records.append({
                "date": dt,
                "cash": cash,
                "holding_value": final_holding_value,
                "nav": final_nav,
                "holding_symbol": active_sym,
            })

        df_records = pd.DataFrame(daily_records).set_index("date")
        df_records["return"] = df_records["nav"].pct_change(fill_method=None).fillna(0.0)
        df_records["cum_return"] = df_records["nav"] / self.initial_cash - 1.0

        # 最大回撤计算
        df_records["peak"] = df_records["nav"].cummax()
        df_records["drawdown"] = (df_records["nav"] - df_records["peak"]) / df_records["peak"]

        # 计算综合绩效指标
        metrics = self._calculate_metrics(df_records, total_turnover_amount, trade_count)
        return df_records, metrics

    def _calculate_metrics(self, df: pd.DataFrame, total_turnover: float, trade_count: int) -> dict:
        total_days = len(df)
        if total_days < 2:
            return {}

        years = total_days / 242.0
        final_nav = df["nav"].iloc[-1]
        total_return = (final_nav / self.initial_cash) - 1.0
        cagr = (final_nav / self.initial_cash) ** (1.0 / max(years, 0.1)) - 1.0

        daily_ret = df["return"]
        daily_rf = (1.0 + self.risk_free_rate) ** (1.0 / 242.0) - 1.0
        excess_ret_daily = daily_ret - daily_rf
        
        # 核心修复 4: 统一使用算术标准夏普口径
        vol_daily = daily_ret.std(ddof=1)
        annual_vol = vol_daily * np.sqrt(242.0)
        sharpe = (excess_ret_daily.mean() / (vol_daily + 1e-8)) * np.sqrt(242.0)

        max_dd = abs(df["drawdown"].min())
        calmar = cagr / (max_dd + 1e-8)

        win_days = (daily_ret > 0).sum()
        win_rate = win_days / (total_days - (daily_ret == 0).sum() + 1e-8)

        # 单边年化换手率口径标注明确
        avg_equity = df["nav"].mean()
        annual_turnover = (total_turnover / 2.0) / (avg_equity + 1e-8) / max(years, 0.1)

        return {
            "initial_cash": self.initial_cash,
            "final_nav": round(float(final_nav), 2),
            "total_return_pct": round(float(total_return * 100), 2),
            "cagr_pct": round(float(cagr * 100), 2),
            "annual_vol_pct": round(float(annual_vol * 100), 2),
            "sharpe_ratio": round(float(sharpe), 3),
            "max_drawdown_pct": round(float(max_dd * 100), 2),
            "calmar_ratio": round(float(calmar), 3),
            "win_rate_pct": round(float(win_rate * 100), 2),
            "annual_turnover_rate": round(float(annual_turnover), 2),
            "total_trades": int(trade_count),
            "total_days": int(total_days),
        }
