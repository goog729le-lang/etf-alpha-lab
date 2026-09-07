from pathlib import Path

# 项目路径定义
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
RULES_DIR = RESULTS_DIR / "discovered_rules"
REPORTS_DIR = RESULTS_DIR / "reports"

# 确保关键目录存在
for d in [DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, RULES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 交易与资金硬约束
INITIAL_CASH = 10000.0        # 初始试验资金：10,000 元
LOT_SIZE = 100                # A 股 ETF 最小买入单位：100 股（1手）
COMMISSION_RATE = 0.0003      # 券商实盘交易佣金：双边万分之三，无印花税
SLIPPAGE_POINTS = 0.001       # 回测保守滑点：每手 0.001 元（约 1 个基点）

# 时间周期约束
START_DATE = "2018-01-01"     # 历史回测起点
TRAIN_END_DATE = "2024-12-31" # 样本内演化切分点（用于初始基线探索）
TEST_START_DATE = "2025-01-01"# 样本外验证起点

# CPCV 交叉验证设置
CPCV_NUM_SPLITS = 6           # 全时段划分为 6 个时序区块
CPCV_TEST_BLOCKS = 2          # 每次选取 2 个区块作为盲测集，组合数 C(6, 2) = 15 条平行路径
PURGE_BUFFER_DAYS = 7         # 标签重叠物理净化隔离带
EMBARGO_BUFFER_DAYS = 10      # 自相关长记忆物理禁运隔离带
