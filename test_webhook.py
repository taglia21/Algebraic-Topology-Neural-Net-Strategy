import requests, os
from dotenv import load_dotenv
load_dotenv()
url = os.getenv('DISCORD_WEBHOOK_MARCUS')
if url:
    r = requests.post(url, json={"username": "Marcus Chen", "content": "Team of Rivals system is ONLINE!"})
    print("SUCCESS!" if r.status_code == 204 else f"Error: {r.status_code}")
else:
    print("No webhook URL found")
