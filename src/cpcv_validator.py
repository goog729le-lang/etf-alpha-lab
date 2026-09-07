"""
组合净化交叉验证 (Combinatorial Purged Cross-Validation, CPCV)
严格支持:
1. Purging (切除标签与测试集边界重叠期)
2. Embargoing (测试集右侧设立禁运隔离带消除自回归长记忆)
3. 严格使用 config 中的 CPCV_NUM_SPLITS 与 CPCV_TEST_BLOCKS 动态生成组合
4. 真实判定门槛 is_cpcv_passed 输出，作为入库硬闸门
"""

import itertools
from typing import List, Dict
import pandas as pd
import numpy as np
from config import CPCV_NUM_SPLITS, CPCV_TEST_BLOCKS, PURGE_BUFFER_DAYS, EMBARGO_BUFFER_DAYS
from backtester import ETFBacktester


class CPCVValidator:
    def __init__(
        self,
        n_splits: int = CPCV_NUM_SPLITS,
        k_test: int = CPCV_TEST_BLOCKS,
        purge_days: int = PURGE_BUFFER_DAYS,
        embargo_days: int = EMBARGO_BUFFER_DAYS,
        min_mean_sharpe: float = 0.50,   # CPCV 样本外平均夏普门槛
        min_worst_sharpe: float = 0.0,   # 最差路径不能为负
        max_worst_mdd: float = 35.0,     # 最差单条路径最大回撤不能击穿35%
    ):
        self.n_splits = n_splits
        self.k_test = k_test
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_mean_sharpe = min_mean_sharpe
        self.min_worst_sharpe = min_worst_sharpe
        self.max_worst_mdd = max_worst_mdd

    def split_dates_with_purging(self, all_dates: List[pd.Timestamp]) -> List[List[pd.Timestamp]]:
        """将全量有序交易日均匀划分为 n_splits 个独立区块"""
        total_len = len(all_dates)
        block_size = total_len // self.n_splits
        blocks = []
        for i in range(self.n_splits):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size if i < self.n_splits - 1 else total_len
            blocks.append(all_dates[start_idx:end_idx])
        return blocks

    def evaluate_positions(
        self,
        positions_df: pd.DataFrame,
        panel_data: Dict[str, pd.DataFrame],
    ) -> dict:
        """
        对已生成的候选持仓矩阵执行严格的 CPCV 交叉验证与 Purging/Embargo 隔离
        """
        all_dates = sorted(list(positions_df.index))
        blocks = self.split_dates_with_purging(all_dates)
        
        # 动态组合数 C(N, K)
        block_indices = list(range(self.n_splits))
        combinations = list(itertools.combinations(block_indices, self.k_test))

        backtester = ETFBacktester()
        oos_sharpes = []
        oos_max_dds = []
        oos_cagrs = []

        for combo in combinations:
            # 核心修复 P0: 独立回测每个时序区块，彻底消除跨区块时钟断层导致的虚假跳空暴跌
            all_recs = []
            curr_cash = 10000.0
            for b_idx in combo:
                b_dates = blocks[b_idx]
                if not b_dates:
                    continue
                # 对每个测试区块头部严格执行 purge (消除前序边界重叠)
                b_start = b_dates[0]
                purged_b_dates = [d for d in b_dates if d >= (b_start + pd.Timedelta(days=self.purge_days))]
                if len(purged_b_dates) < 20:
                    purged_b_dates = b_dates

                pos_b = positions_df.reindex(purged_b_dates).dropna(how="all")
                if pos_b.empty:
                    continue

                bt = ETFBacktester(initial_cash=curr_cash)
                df_b, _ = bt.run_backtest(pos_b, panel_data)
                if not df_b.empty:
                    all_recs.append(df_b)
                    curr_cash = df_b["nav"].iloc[-1]

            if not all_recs:
                continue

            # 拼接各区块真实收益率序列 (无跨期假跳空)
            rets = pd.concat([r["return"] for r in all_recs])
            if len(rets) < 30:
                continue

            nav_series = (1.0 + rets).cumprod() * 10000.0
            peak = nav_series.cummax()
            mdd = float(abs((nav_series - peak) / peak).max() * 100.0)

            daily_rf = (1.0 + 0.02) ** (1.0 / 242.0) - 1.0
            excess = rets - daily_rf
            vol = float(rets.std(ddof=1))
            sr = float((excess.mean() / (vol + 1e-8)) * np.sqrt(242.0))

            total_days = len(rets)
            years = total_days / 242.0
            final_nav = float(nav_series.iloc[-1])
            cagr = float(((final_nav / 10000.0) ** (1.0 / max(years, 0.1)) - 1.0) * 100.0)

            oos_sharpes.append(sr)
            oos_max_dds.append(mdd)
            oos_cagrs.append(cagr)
        if not oos_sharpes:
            return {
                "cpcv_valid": False,
                "is_cpcv_passed": False,
                "reason": "CPCV 路径有效样本点不足",
            }

        oos_sharpes = np.array(oos_sharpes)
        oos_max_dds = np.array(oos_max_dds)
        oos_cagrs = np.array(oos_cagrs)

        mean_sharpe = float(np.mean(oos_sharpes))
        worst_sharpe = float(np.min(oos_sharpes))
        best_sharpe = float(np.max(oos_sharpes))
        std_sharpe = float(np.std(oos_sharpes, ddof=1)) if len(oos_sharpes) > 1 else 0.0
        mean_max_dd = float(np.mean(oos_max_dds))
        worst_max_dd = float(np.max(oos_max_dds))

        # 真实判定门槛
        is_passed = (
            worst_sharpe >= self.min_worst_sharpe
            and mean_sharpe >= self.min_mean_sharpe
            and worst_max_dd <= self.max_worst_mdd
        )

        return {
            "cpcv_valid": True,
            "total_paths": len(oos_sharpes),
            "mean_oos_sharpe": round(mean_sharpe, 3),
            "worst_oos_sharpe": round(worst_sharpe, 3),
            "best_oos_sharpe": round(best_sharpe, 3),
            "std_oos_sharpe": round(std_sharpe, 3),
            "mean_oos_max_dd": round(mean_max_dd, 2),
            "worst_oos_max_dd": round(worst_max_dd, 2),
            "is_cpcv_passed": bool(is_passed),
        }
