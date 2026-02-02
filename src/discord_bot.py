import discord
from discord.ext import commands, tasks
import asyncio
import logging
import os
from datetime import datetime
import pytz
import aiohttp
import base64

logger = logging.getLogger(__name__)

class DiscordBot:
    def __init__(self, team):
        self.team = team
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        self.guild_id = 1467608148855750832
        self.standup_channel_id = 1467608263297335448
        self.setup_commands()
        
    def setup_commands(self):
        @self.bot.event
        async def on_ready():
            logger.info(f'Discord bot logged in as {self.bot.user}')
            self.standup_task.start()
            
        @self.bot.command()
        async def status(ctx):
            """Check system status"""
            await ctx.send('✅ Trading system online. All 6 agents ready.')
            
        @self.bot.command()
        async def meeting(ctx):
            """Trigger manual standup"""
            await self.run_standup()
            
    @tasks.loop(hours=24)
    async def standup_task(self):
        """Daily standup at 9am EST"""
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)
        if now.hour == 9 and now.minute < 5:
            await self.run_standup()
            
    async def text_to_speech(self, text, voice_id):
        """Convert text to speech using Azure TTS"""
        azure_key = os.getenv('AZURE_TTS_KEY')
        azure_region = os.getenv('AZURE_TTS_REGION', 'eastus')
        
        if not azure_key:
            logger.warning('Azure TTS not configured')
            return None
            
        url = f'https://{azure_region}.tts.speech.microsoft.com/cognitiveservices/v1'
        headers = {
            'Ocp-Apim-Subscription-Key': azure_key,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3'
        }
        
        ssml = f'''<speak version='1.0' xml:lang='en-US'>
            <voice name='{voice_id}'>{text}</voice>
        </speak>'''
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=ssml) as resp:
                    if resp.status == 200:
                        return await resp.read()
        except Exception as e:
            logger.error(f'TTS error: {e}')
        return None
        
    async def run_standup(self):
        """Run morning standup meeting"""
        logger.info('Starting morning standup')
        channel = self.bot.get_channel(self.standup_channel_id)
        if not channel:
            logger.error('Standup channel not found')
            return
            
        await channel.send('📊 **Morning Standup - Team of Rivals**')
        await channel.send(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M EST")}')
        await channel.send('---')
        
        # Get reports from all agents
        reports = await self.team.generate_standup_reports()
        
        for report in reports:
            # Send text update
            message = f"**{report['agent']}** - {report['role']}\n{report['update']}"
            await channel.send(message)
            
            # Generate and send TTS audio
            audio = await self.text_to_speech(report['update'], report['voice_id'])
            if audio:
                # Save audio temporarily and send
                filename = f'/tmp/{report["agent"].replace(" ", "_")}.mp3'
                with open(filename, 'wb') as f:
                    f.write(audio)
                await channel.send(file=discord.File(filename))
                os.remove(filename)
                
            await asyncio.sleep(2)  # Pause between agents
            
        await channel.send('---')
        await channel.send('✅ Standup complete. Ready for trading day.')
        
    async def log_trade(self, order):
        """Log executed trade to Discord"""
        channel = self.bot.get_channel(self.standup_channel_id)
        if channel:
            await channel.send(f'💰 **Trade Executed**: {order.symbol} {order.side} {order.qty} @ {order.filled_avg_price}')
            
    async def log_veto(self, proposal):
        """Log vetoed trade to Discord"""
        channel = self.bot.get_channel(self.standup_channel_id)
        if channel:
            await channel.send(f'⛔ **Trade Vetoed**: {proposal["symbol"]} {proposal["side"]} {proposal["qty"]}')
            
    async def start(self, token):
        """Start Discord bot"""
        await self.bot.start(token)
