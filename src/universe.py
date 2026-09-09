"""
ETF 资产池定义与分类管理
"""

# 金丝雀防守与宏观参照资产 (Canary Assets)
CANARY_UNIVERSE = {
    "511010": {"name": "国债ETF", "market": "sh", "type": "BOND_DEFENSE"},
    "510300": {"name": "沪深300ETF", "market": "sh", "type": "EQUITY_BENCHMARK"},
    "518880": {"name": "黄金ETF", "market": "sh", "type": "COMMODITY_DEFENSE"},
}

# 核心行业/风格轮动资产池 (Core Universe) - 高流动性主流品种
CORE_UNIVERSE = {
    # 科技成长
    "512480": {"name": "半导体ETF", "market": "sh", "sector": "Technology"},
    "588000": {"name": "科创50ETF", "market": "sh", "sector": "Technology"},
    "159915": {"name": "创业板ETF", "market": "sz", "sector": "Growth"},
    "515050": {"name": "通信ETF", "market": "sh", "sector": "Technology"},
    
    # 大金融与价值防御
    "512000": {"name": "券商ETF", "market": "sh", "sector": "Financial"},
    "512800": {"name": "银行ETF", "market": "sh", "sector": "Financial"},
    "510880": {"name": "红利ETF", "market": "sh", "sector": "Value_Dividend"},
    
    # 大消费与医药
    "159928": {"name": "消费ETF", "market": "sz", "sector": "Consumer"},
    "512690": {"name": "酒ETF", "market": "sh", "sector": "Consumer"},
    "512010": {"name": "医药ETF", "market": "sh", "sector": "Healthcare"},
    
    # 制造周期与新能源
    "516160": {"name": "新能源ETF", "market": "sh", "sector": "NewEnergy"},
    "512660": {"name": "军工ETF", "market": "sh", "sector": "Defense"},
    
    # 资源与上游周期 (高股息与大宗)
    "515220": {"name": "煤炭ETF", "market": "sh", "sector": "Energy_Coal"},
    "512400": {"name": "有色金属ETF", "market": "sh", "sector": "Materials_Metals"},
    
    # 数字传媒与新兴应用
    "512980": {"name": "传媒ETF", "market": "sh", "sector": "TMT_Media"},
    
    # 中小盘宽基风格
    "510500": {"name": "500ETF", "market": "sh", "sector": "MidCap_Core"},
    "512100": {"name": "1000ETF", "market": "sh", "sector": "SmallCap_Growth"},
    
    # 全球多元配置 (T+0 跨境高弹性)
    "513100": {"name": "纳斯达克ETF", "market": "sh", "sector": "Global"},
    "513050": {"name": "中概互联ETF", "market": "sh", "sector": "Global_Tech"},
}

# 域外压力测试池 (OOD Stress Universe) - 检验策略避坑与泛化能力
STRESS_UNIVERSE = {
    "159985": {"name": "豆粕ETF", "market": "sz", "sector": "Agriculture_Commodity"},
    "159865": {"name": "养殖ETF", "market": "sz", "sector": "Agriculture_Cycle"},
    "515210": {"name": "钢铁ETF", "market": "sh", "sector": "Heavy_Industry"},
}

def get_all_symbols():
    """获取全量 ETF 代码字典"""
    all_dict = {}
    all_dict.update(CANARY_UNIVERSE)
    all_dict.update(CORE_UNIVERSE)
    all_dict.update(STRESS_UNIVERSE)
    return all_dict
