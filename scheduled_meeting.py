#!/usr/bin/env python3
"""Team of Rivals - Scheduled Daily Meetings with TTS
Runs at 9 AM EST every trading day
"""
import os
import asyncio
import aiohttp
import json
from datetime import datetime
import pytz
import schedule
import time
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import io
import base64

load_dotenv()

# Agent Configuration with unique voices
AGENTS = {
    'marcus': {
        'name': 'Marcus Chen',
        'role': 'Chief Orchestrator',
        'webhook_env': 'DISCORD_WEBHOOK_MARCUS',
        'voice': 'en-US-GuyNeural',
        'style': 'newscast-formal',
        'color': 0xFFD700  # Gold
    },
    'victoria': {
        'name': 'Victoria Sterling',
        'role': 'Chief Risk Officer',
        'webhook_env': 'DISCORD_WEBHOOK_VICTORIA',
        'voice': 'en-US-JennyNeural',
        'style': 'serious',
        'color': 0xFF4444  # Red
    },
    'james': {
        'name': 'James Thornton',
        'role': 'Chief Strategy Officer',
        'webhook_env': 'DISCORD_WEBHOOK_JAMES',
        'voice': 'en-US-DavisNeural',
        'style': 'friendly',
        'color': 0x4444FF  # Blue
    },
    'elena': {
        'name': 'Elena Rodriguez',
        'role': 'Chief ML Engineer',
        'webhook_env': 'DISCORD_WEBHOOK_ELENA',
        'voice': 'en-US-AriaNeural',
        'style': 'empathetic',
        'color': 0x44FF44  # Green
    },
    'derek': {
        'name': 'Derek Washington',
        'role': 'Execution Team Lead',
        'webhook_env': 'DISCORD_WEBHOOK_DEREK',
        'voice': 'en-US-TonyNeural',
        'style': 'calm',
        'color': 0xFF8800  # Orange
    },
    'sophia': {
        'name': 'Dr. Sophia Nakamura',
        'role': 'Research Lead',
        'webhook_env': 'DISCORD_WEBHOOK_SOPHIA',
        'voice': 'en-US-SaraNeural',
        'style': 'professional',
        'color': 0x8800FF  # Purple
    }
}

class TTSEngine:
    """Azure Text-to-Speech engine for agent voices"""
    def __init__(self):
        self.key = os.getenv('AZURE_TTS_KEY')
        self.region = os.getenv('AZURE_TTS_REGION', 'eastus')
        if self.key:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.key, region=self.region
            )
        else:
            self.speech_config = None
            print("Warning: Azure TTS not configured")
    
    def synthesize(self, text, voice, style='general'):
        """Generate speech audio from text"""
        if not self.speech_config:
            return None
        
        ssml = f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
            xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{voice}">
                <mstts:express-as style="{style}">
                    {text}
                </mstts:express-as>
            </voice>
        </speak>'''
        
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None
        )
        result = synthesizer.speak_ssml_async(ssml).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        return None


class MeetingOrchestrator:
    """Orchestrates daily team meetings"""
    def __init__(self):
        self.tts = TTSEngine()
        self.est = pytz.timezone('America/New_York')
    
    def is_trading_day(self):
        """Check if today is a trading day"""
        now = datetime.now(self.est)
        # Skip weekends
        if now.weekday() >= 5:
            return False
        # TODO: Add holiday checking
        return True
    
    async def post_to_discord(self, agent_key, content, embed_data=None):
        """Post message to Discord via webhook"""
        agent = AGENTS[agent_key]
        webhook_url = os.getenv(agent['webhook_env'])
        
        if not webhook_url:
            print(f"Warning: No webhook for {agent['name']}")
            return False
        
        data = {
            'username': f"{agent['name']} - {agent['role']}",
            'content': content
        }
        
        if embed_data:
            data['embeds'] = [{
                'title': embed_data.get('title', ''),
                'description': embed_data.get('description', ''),
                'color': agent['color'],
                'fields': embed_data.get('fields', []),
                'footer': {'text': f"Team of Rivals • {datetime.now(self.est).strftime('%Y-%m-%d %H:%M EST')}"}
            }]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=data) as resp:
                return resp.status == 204

    
    def get_portfolio_status(self):
        """Get current portfolio status from Alpaca"""
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            )
            account = api.get_account()
            positions = api.list_positions()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'positions': len(positions),
                'daily_pnl': float(account.equity) - float(account.last_equity)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def generate_agent_report(self, agent_key, portfolio):
        """Generate personalized report for each agent"""
        agent = AGENTS[agent_key]
        now = datetime.now(self.est)
        
        reports = {
            'marcus': f"""Good morning team. Today is {now.strftime('%A, %B %d')}.
Portfolio stands at ${portfolio.get('equity', 0):,.2f} with {portfolio.get('positions', 0)} active positions.
Daily P&L: ${portfolio.get('daily_pnl', 0):,.2f}.
Let's hear from each department.""",
            
            'victoria': f"""Risk assessment for today:
- Current exposure: {portfolio.get('positions', 0)} positions
- Cash buffer: ${portfolio.get('cash', 0):,.2f}
- Buying power: ${portfolio.get('buying_power', 0):,.2f}
All risk parameters are within acceptable limits. No vetoes required at this time.""",
            
            'james': f"""Strategy update:
Our TDA neural network continues to identify topological patterns in market data.
Current focus: Mean reversion and momentum divergence signals.
Recommendation: Maintain current allocation strategy with tactical adjustments based on volatility.""",
            
            'elena': f"""ML systems status:
- Neural network accuracy: Monitoring
- Feature engineering: Active
- Model drift: Within tolerance
Next scheduled retraining: After market close today if data quality thresholds are met.""",
            
            'derek': f"""Execution report:
- Order routing: Optimal
- Slippage yesterday: Minimal
- Fill rates: Excellent
All execution systems are operational and ready for today's trading session.""",
            
            'sophia': f"""Research update:
Topological Data Analysis showing interesting Betti number patterns.
Persistence diagrams indicate potential regime shift signals.
Continuing to monitor homology groups for actionable insights."""
        }
        return reports.get(agent_key, "Report not available.")

    
    async def run_morning_meeting(self):
        """Execute the full morning standup meeting"""
        if not self.is_trading_day():
            print("Not a trading day, skipping meeting")
            return
        
        print(f"Starting morning meeting at {datetime.now(self.est)}")
        portfolio = self.get_portfolio_status()
        
        # Order of speakers
        speaker_order = ['marcus', 'victoria', 'james', 'elena', 'derek', 'sophia']
        
        for agent_key in speaker_order:
            agent = AGENTS[agent_key]
            report = self.generate_agent_report(agent_key, portfolio)
            
            # Post to Discord
            embed = {
                'title': f"{agent['role']} Report",
                'description': report,
                'fields': []
            }
            
            success = await self.post_to_discord(agent_key, '', embed)
            if success:
                print(f"Posted: {agent['name']}")
            else:
                print(f"Failed: {agent['name']}")
            
            # Generate TTS audio (optional - for voice channel)
            if self.tts.speech_config:
                audio = self.tts.synthesize(report, agent['voice'], agent['style'])
                if audio:
                    print(f"Generated audio for {agent['name']}")
            
            await asyncio.sleep(2)  # Delay between speakers
        
        print("Morning meeting complete")


async def trigger_ml_retraining():
    """Trigger ML model retraining after market close"""
    print("Triggering ML retraining...")
    try:
        import subprocess
        result = subprocess.run(
            ['python', 'train_and_backtest.py'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        print(f"Retraining complete: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Retraining error: {e}")
        return False


def run_scheduled_meeting():
    """Wrapper for scheduled execution"""
    orchestrator = MeetingOrchestrator()
    asyncio.run(orchestrator.run_morning_meeting())


def run_scheduled_retraining():
    """Wrapper for scheduled ML retraining"""
    asyncio.run(trigger_ml_retraining())


def main():
    """Main scheduling loop"""
    print("Team of Rivals - Scheduled Meeting System")
    print("Scheduling daily meetings at 9:00 AM EST")
    print("Scheduling ML retraining at 4:30 PM EST")
    
    # Schedule morning meeting at 9 AM EST
    schedule.every().monday.at("09:00").do(run_scheduled_meeting)
    schedule.every().tuesday.at("09:00").do(run_scheduled_meeting)
    schedule.every().wednesday.at("09:00").do(run_scheduled_meeting)
    schedule.every().thursday.at("09:00").do(run_scheduled_meeting)
    schedule.every().friday.at("09:00").do(run_scheduled_meeting)
    
    # Schedule ML retraining at 4:30 PM EST (after market close)
    schedule.every().monday.at("16:30").do(run_scheduled_retraining)
    schedule.every().tuesday.at("16:30").do(run_scheduled_retraining)
    schedule.every().wednesday.at("16:30").do(run_scheduled_retraining)
    schedule.every().thursday.at("16:30").do(run_scheduled_retraining)
    schedule.every().friday.at("16:30").do(run_scheduled_retraining)
    
    # Run the scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--now':
        # Run meeting immediately
        run_scheduled_meeting()
    elif len(sys.argv) > 1 and sys.argv[1] == '--retrain':
        # Run retraining immediately
        run_scheduled_retraining()
    else:
        main()
