import requests, os, time
from dotenv import load_dotenv
load_dotenv()

print("Calling impromptu team meeting...\n")

agents = [
    ("Marcus Chen - Chief Orchestrator", "DISCORD_WEBHOOK_MARCUS", "Team, thank you for joining this impromptu meeting. I need status updates from everyone."),
    ("Victoria Sterling - Chief Risk Officer", "DISCORD_WEBHOOK_VICTORIA", "Risk status: GREEN. All positions within limits. Portfolio heat at 23%. We're well-positioned."),
    ("James Thornton - Strategy Team Lead", "DISCORD_WEBHOOK_JAMES", "Currently testing mean reversion on AAPL/MSFT pair. Backtests look promising - 58% win rate."),
    ("Elena Rodriguez - Data Team Lead", "DISCORD_WEBHOOK_ELENA", "Data feeds: OPERATIONAL. Polygon API 99.9% uptime. Detected 2 high-confidence setups in tech sector."),
    ("Derek Washington - Execution Team Lead", "DISCORD_WEBHOOK_DEREK", "Execution quality: EXCELLENT. Average slippage 0.02%. All systems ready for next trading session."),
    ("Dr. Sophia Nakamura - Research Team Lead", "DISCORD_WEBHOOK_SOPHIA", "TDA neural network showing 87% accuracy on validation set. Recommend moving to paper trading soon.")
]

for name, env_var, message in agents:
    url = os.getenv(env_var)
    if url:
        response = requests.post(url, json={"username": name, "content": message})
        print(f"✅ {name.split()[0]} reported")
        time.sleep(1)
    else:
        print(f"❌ {name} - webhook not found")

print("\n🎯 Meeting complete! Check #morning-standup in Discord to see all reports.")
