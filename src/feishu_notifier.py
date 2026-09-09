"""
飞书富文本互动卡片通知模块 (Feishu Card Notifier)
支持自建应用 (Bot) 通过 open_id 或手机号自动发送每日 14:45 实盘交易决策卡片。
"""

import os
import json
import urllib.request
from typing import Optional


def get_tenant_access_token(app_id: str, app_secret: str) -> Optional[str]:
    """通过 app_id 与 app_secret 获取 tenant_access_token"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("code") == 0:
                return data.get("tenant_access_token")
            print(f"[-] 获取飞书 token 失败: {data.get('msg')}")
    except Exception as e:
        print(f"[-] 请求飞书 token 异常: {e}")
    return None


def get_open_id_by_mobile(token: str, mobile: str) -> Optional[str]:
    """通过手机号查询用户 open_id"""
    url = "https://open.feishu.cn/open-apis/contact/v3/users/batch_get_id?user_id_type=open_id"
    payload = json.dumps({"mobiles": [mobile]}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("code") == 0:
                user_list = data.get("data", {}).get("user_list", [])
                if user_list and user_list[0].get("user_id"):
                    return user_list[0].get("user_id")
    except Exception as e:
        print(f"[-] 手机号转换 open_id 异常: {e}")
    return None


def build_feishu_card(action_res: dict) -> dict:
    """构建美观直观的飞书互动卡片 (Interactive Card)"""
    action_type = action_res.get("action_type", "HOLD")
    
    # 配色与状态映射
    template_color = {
        "HOLD": "green",
        "BUY": "blue",
        "SELL": "red",
        "SWITCH": "orange",
        "STAY_CASH": "grey",
    }.get(action_type, "blue")

    action_label = {
        "HOLD": "🛡️【继续持有 (HOLD)】",
        "BUY": "🚀【开仓买入 (BUY)】",
        "SELL": "⚠️【止损清仓 (SELL)】",
        "SWITCH": "🔄【调仓换标 (SWITCH)】",
        "STAY_CASH": "☕【保持现金 (STAY CASH)】",
    }.get(action_type, action_type)

    bench = action_res.get("market_benchmark", {})
    bench_price = bench.get("current_price", "--")
    bench_ma = bench.get("ma_filter", "--")
    bench_br = bench.get("breadth_pct", "--")
    bench_br_th = bench.get("breadth_threshold_pct", "--")
    bench_status = bench.get("status", "--")

    card = {
        "config": {
            "wide_screen_mode": True
        },
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"🔔 ETF Alpha Lab 今日实盘指令: {action_label}"
            },
            "template": template_color
        },
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**所处交易日**\n`{action_res.get('trade_date')}`"
                        }
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**快照时间**\n`{action_res.get('generated_at')}`"
                        }
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**当前持有**\n`{action_res.get('prev_holding')}` ({action_res.get('prev_name')})"
                        }
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**目标持仓**\n🎯 **`{action_res.get('target_holding')}`** ({action_res.get('target_name')})"
                        }
                    }
                ]
            },
            {
                "tag": "hr"
            },
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"📊 **大盘基准与市场广度 (510300)**\n* **现价 (14:45)**: `{bench_price}` | **60日均线**: `{bench_ma}`\n* **行业市场广度**: `{bench_br}%` (多头门槛: `{bench_br_th}%`)\n* **多空研判**: **{bench_status}**"
                }
            },
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"💡 **决策依据与操作建议**\n> {action_res.get('action_reason')}\n\n*建议于今日 **14:50 ~ 15:00** 收盘前执行完毕。*"
                }
            },
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"模型: {action_res.get('champion_strategy')} | 严格纯境内A股无前视回测实盘跟踪"
                    }
                ]
            }
        ]
    }
    return card


def send_feishu_notification(
    action_res: dict,
    app_id: Optional[str] = None,
    app_secret: Optional[str] = None,
    open_id: Optional[str] = None,
    mobile: Optional[str] = None,
) -> bool:
    """发送飞书互动卡片通知"""
    app_id = app_id or os.environ.get("FEISHU_APP_ID")
    app_secret = app_secret or os.environ.get("FEISHU_APP_SECRET")
    open_id = open_id or os.environ.get("FEISHU_OPEN_ID")
    mobile = mobile or os.environ.get("FEISHU_USER_MOBILE")

    if not app_id or not app_secret:
        print("[*] 未配置 FEISHU_APP_ID 或 FEISHU_APP_SECRET，跳过飞书推送。")
        return False

    token = get_tenant_access_token(app_id, app_secret)
    if not token:
        return False

    # 若未直接提供 open_id 但有手机号，则动态解析
    if not open_id and mobile:
        open_id = get_open_id_by_mobile(token, mobile)

    if not open_id:
        print("[-] 未能获取有效的接收人 open_id，无法发送飞书消息。")
        return False

    card_content = build_feishu_card(action_res)
    send_url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id"
    payload = json.dumps({
        "receive_id": open_id,
        "msg_type": "interactive",
        "content": json.dumps(card_content)
    }).encode("utf-8")

    req = urllib.request.Request(
        send_url, data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            if res_data.get("code") == 0:
                print(f"[+] 飞书实盘决策卡片已成功推送到用户 (open_id: {open_id[:8]}***, 手机号: {mobile or '已绑定'})")
                return True
            else:
                print(f"[-] 飞书消息发送失败: {res_data}")
    except Exception as e:
        print(f"[-] 发送飞书消息异常: {e}")
    return False


if __name__ == "__main__":
    # 独立测试入口
    test_action = {
        "generated_at": "2026-09-08 14:45:00",
        "trade_date": "2026-09-08",
        "action_type": "HOLD",
        "target_holding": "518880",
        "target_name": "黄金ETF",
        "prev_holding": "518880",
        "prev_name": "黄金ETF",
        "market_benchmark": {
            "symbol": "510300 (沪深300ETF)",
            "current_price": 4.643,
            "ma_filter": 4.766,
            "status": "跌破均线 (防守模式)",
        },
        "action_reason": "继续持有 518880 (黄金ETF)。现价 9.106，60日均线 8.762，运行正常无破位。",
        "champion_strategy": "Domestic_Sector_Mom40_MA60_MONTHLY_DYNAMIC_CANARY",
    }
    # 从环境变量读取配置 (由 GitHub Secrets 或本地环境提供，严禁硬编码敏感凭证)
    send_feishu_notification(test_action)
