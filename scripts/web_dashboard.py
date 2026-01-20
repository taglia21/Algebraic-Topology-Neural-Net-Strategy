#!/usr/bin/env python3
"""
Simple Web Dashboard Server
===========================
Serves a real-time performance dashboard on port 8080
"""

import http.server
import socketserver
import json
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PORT = 8080
DAILY_LOG = Path('/opt/Algebraic-Topology-Neural-Net-Strategy/logs/daily_metrics.json')
PERF_LOG = Path('/opt/Algebraic-Topology-Neural-Net-Strategy/logs/performance_log.json')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>TDA Hedge Fund Dashboard</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ color: #00d4ff; margin: 0; font-size: 2.5em; }}
        .header p {{ color: #888; margin: 10px 0; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #16213e; border-radius: 10px; padding: 20px; text-align: center; }}
        .card h3 {{ color: #888; margin: 0 0 10px 0; font-size: 0.9em; text-transform: uppercase; }}
        .card .value {{ font-size: 2em; font-weight: bold; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        .neutral {{ color: #00d4ff; }}
        .positions {{ background: #16213e; border-radius: 10px; padding: 20px; }}
        .positions h2 {{ color: #00d4ff; margin: 0 0 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ text-align: left; color: #888; padding: 10px; border-bottom: 1px solid #333; }}
        td {{ padding: 10px; border-bottom: 1px solid #222; }}
        tr:hover {{ background: #1a2744; }}
        .refresh {{ text-align: center; color: #666; font-size: 0.8em; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 TDA Hedge Fund Dashboard</h1>
        <p>Last updated: {timestamp}</p>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>Portfolio Value</h3>
            <div class="value neutral">${equity:,.2f}</div>
        </div>
        <div class="card">
            <h3>Cash</h3>
            <div class="value">${cash:,.2f}</div>
        </div>
        <div class="card">
            <h3>Invested</h3>
            <div class="value">${invested:,.2f}</div>
        </div>
        <div class="card">
            <h3>Unrealized P&L</h3>
            <div class="value {pl_class}">${unrealized_pl:+,.2f}</div>
        </div>
        <div class="card">
            <h3>Total Return</h3>
            <div class="value {return_class}">{total_return:+.2f}%</div>
        </div>
        <div class="card">
            <h3>Sharpe Ratio</h3>
            <div class="value">{sharpe:.2f}</div>
        </div>
        <div class="card">
            <h3>Max Drawdown</h3>
            <div class="value negative">-{max_dd:.2f}%</div>
        </div>
        <div class="card">
            <h3>Positions</h3>
            <div class="value neutral">{position_count}</div>
        </div>
    </div>
    
    <div class="positions">
        <h2>Top Holdings</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Market Value</th>
                <th>P&L</th>
                <th>% Change</th>
            </tr>
            {position_rows}
        </table>
    </div>
    
    <p class="refresh">Auto-refreshes every 60 seconds</p>
</body>
</html>
"""

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            try:
                if DAILY_LOG.exists():
                    with open(DAILY_LOG) as f:
                        data = json.load(f)
                else:
                    data = {'snapshot': {}, 'metrics': {}}
                
                snap = data.get('snapshot', {})
                metrics = data.get('metrics', {})
                
                equity = snap.get('equity', 100000)
                cash = snap.get('cash', 100000)
                invested = equity - cash
                unrealized_pl = snap.get('total_unrealized_pl', 0)
                total_return = metrics.get('total_return_pct', 0)
                
                # Build position rows
                position_rows = ""
                for pos in snap.get('positions', [])[:20]:
                    pl_class = 'positive' if pos['unrealized_pl'] >= 0 else 'negative'
                    position_rows += f"""
                    <tr>
                        <td><strong>{pos['symbol']}</strong></td>
                        <td>${pos['market_value']:,.2f}</td>
                        <td class="{pl_class}">${pos['unrealized_pl']:+,.2f}</td>
                        <td class="{pl_class}">{pos['unrealized_plpc']:+.1f}%</td>
                    </tr>
                    """
                
                html = HTML_TEMPLATE.format(
                    timestamp=snap.get('timestamp', 'N/A')[:19].replace('T', ' '),
                    equity=equity,
                    cash=cash,
                    invested=invested,
                    unrealized_pl=unrealized_pl,
                    pl_class='positive' if unrealized_pl >= 0 else 'negative',
                    total_return=total_return,
                    return_class='positive' if total_return >= 0 else 'negative',
                    sharpe=metrics.get('sharpe_ratio', 0),
                    max_dd=metrics.get('max_drawdown_pct', 0),
                    position_count=snap.get('position_count', 0),
                    position_rows=position_rows,
                )
                
                self.wfile.write(html.encode())
            except Exception as e:
                self.wfile.write(f"Error: {e}".encode())
        elif self.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            if DAILY_LOG.exists():
                with open(DAILY_LOG) as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(b'{}')
        else:
            super().do_GET()

def main():
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Dashboard running at http://134.209.40.95:{PORT}")
        httpd.serve_forever()

if __name__ == '__main__':
    main()
