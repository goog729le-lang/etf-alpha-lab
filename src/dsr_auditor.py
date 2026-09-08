"""
通缩夏普比率审计器 (Deflated Sharpe Ratio, DSR)
基于 Marcos Lopez de Prado 极值理论，对搜索试验次数执行极值惩罚
"""

import numpy as np
import scipy.stats as stats
import pandas as pd


class DSRAuditor:
    @staticmethod
    def calculate_dsr(
        returns: pd.Series,
        trials_count: int,
        var_trials_sharpe: float = 0.20,
        risk_free_rate: float = 0.02,
    ) -> dict:
        """
        计算非正态分布下的通缩夏普比率 (统一采用算术超额收益口径)
        """
        T = len(returns)
        if T < 30:
            return {"dsr_probability": 0.0, "is_significant_95": False, "is_significant_90": False, "reason": "样本量不足"}

        daily_rf = (1.0 + risk_free_rate) ** (1.0 / 242.0) - 1.0
        excess_returns = returns - daily_rf

        mean_ex = excess_returns.mean()
        std_ret = returns.std(ddof=1) + 1e-8
        sr_daily = mean_ex / std_ret
        sr_annual = sr_daily * np.sqrt(242.0)

        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns, fisher=False))  # 传统峰度，正态分布为3

        # 极值理论预期最大虚假夏普 E[max(SR)]
        euler_mascheroni = 0.5772156649
        k = max(int(trials_count), 2)
        
        z_k = (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / k) + euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (k * np.e))
        expected_max_sr_annual = np.sqrt(max(var_trials_sharpe, 1e-6)) * z_k
        expected_max_sr_daily = expected_max_sr_annual / np.sqrt(242.0)

        # 概率夏普比率 (PSR) 统计误差
        denom_sq = 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily ** 2)
        if denom_sq <= 0:
            denom_sq = 1.0
        se_sr = np.sqrt(denom_sq / (T - 1.0))

        z_stat = (sr_daily - expected_max_sr_daily) / (se_sr + 1e-8)
        dsr_prob = float(stats.norm.cdf(z_stat))

        return {
            "realized_sharpe_annual": round(float(sr_annual), 3),
            "expected_max_spurious_sharpe": round(float(expected_max_sr_annual), 3),
            "trials_count": k,
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "dsr_probability": round(dsr_prob, 4),
            "is_significant_95": bool(dsr_prob >= 0.95),
            "is_significant_90": bool(dsr_prob >= 0.90),
        }
