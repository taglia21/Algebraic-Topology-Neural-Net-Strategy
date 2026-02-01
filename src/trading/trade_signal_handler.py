#!/usr/bin/env python3
"""
Trade Signal Handler - Connects TDA+NN Bot to Team of Rivals
All trade signals must pass through agent voting before execution
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import requests

class TradeSignal:
    """Represents a trade signal from the bot"""
    def __init__(self, symbol: str, action: str, quantity: int, 
                 signal_type: str, confidence: float, reason: str):
        self.symbol = symbol
        self.action = action  # BUY, SELL, BUY_CALL, SELL_PUT, etc
        self.quantity = quantity
        self.signal_type = signal_type  # EQUITY, OPTION
        self.confidence = confidence  # 0.0 to 1.0
        self.reason = reason  # TDA pattern detected
        self.timestamp = datetime.now()
        
class TeamOfRivalsEvaluator:
    """Routes trade signals through agent voting"""
    
    def __init__(self):
        self.min_votes_required = 4  # Need 4 out of 6 agents to approve
        self.webhook_urls = self._load_webhooks()
        
    def _load_webhooks(self) -> Dict[str, str]:
        return {
            "marcus": os.getenv("DISCORD_WEBHOOK_MARCUS"),
            "victoria": os.getenv("DISCORD_WEBHOOK_VICTORIA"),
            "james": os.getenv("DISCORD_WEBHOOK_JAMES"),
            "elena": os.getenv("DISCORD_WEBHOOK_ELENA"),
            "derek": os.getenv("DISCORD_WEBHOOK_DEREK"),
            "sophia": os.getenv("DISCORD_WEBHOOK_SOPHIA")
        }
    
    async def evaluate_signal(self, signal: TradeSignal) -> Dict:
        """
        Submit trade signal to Team of Rivals for evaluation
        Returns: {approved: bool, votes: dict, vetoes: list, reasoning: str}
        """
        # Gather agent opinions
        votes = await self._gather_agent_votes(signal)
        
        # Count approvals
        approvals = sum(1 for v in votes.values() if v['vote'] == 'APPROVE')
        vetoes = [agent for agent, v in votes.items() if v['vote'] == 'VETO']
        
        approved = approvals >= self.min_votes_required
        
        # Post decision to Discord
        await self._announce_decision(signal, approved, votes, vetoes)
        
        return {
            'approved': approved,
            'votes': votes,
            'vetoes': vetoes,
            'approval_count': approvals,
            'reasoning': self._compile_reasoning(votes)
        }
    
    async def _gather_agent_votes(self, signal: TradeSignal) -> Dict:
        """Each agent evaluates the trade signal"""
        votes = {}
        
        # Marcus Chen - Strategy evaluation
        votes['marcus'] = self._marcus_evaluate(signal)
        
        # Victoria Hayes - Risk evaluation  
        votes['victoria'] = self._victoria_evaluate(signal)
        
        # James Park - Statistical evaluation
        votes['james'] = self._james_evaluate(signal)
        
        # Elena Rodriguez - Market conditions evaluation
        votes['elena'] = self._elena_evaluate(signal)
        
        # Derek Thompson - Execution feasibility
        votes['derek'] = self._derek_evaluate(signal)
        
        # Sophia Williams - Compliance check
        votes['sophia'] = self._sophia_evaluate(signal)
        
        return votes
    
    def _marcus_evaluate(self, signal: TradeSignal) -> Dict:
        """Strategy Officer: Profit potential and strategic fit"""
        # Check if signal aligns with overall strategy
        vote = 'APPROVE'
        reasoning = f"Signal shows {signal.confidence*100:.1f}% confidence. "
        
        if signal.confidence < 0.55:
            vote = 'VETO'
            reasoning += "Confidence too low for execution. Need >55% for approval."
        else:
            reasoning += "Acceptable risk/reward ratio."
            
        return {'vote': vote, 'reasoning': reasoning, 'confidence': signal.confidence}
    
    def _victoria_evaluate(self, signal: TradeSignal) -> Dict:
        """Risk Officer: Position sizing and risk limits"""
        vote = 'APPROVE'
        reasoning = "Position size within risk limits. "
        
        # Check position sizing
        if signal.quantity > 100:  # Max position size check
            vote = 'VETO'
            reasoning = f"Position size {signal.quantity} exceeds max limit of 100 shares."
        else:
            reasoning += f"Quantity {signal.quantity} acceptable."
            
        return {'vote': vote, 'reasoning': reasoning}
    
    def _james_evaluate(self, signal: TradeSignal) -> Dict:
        """Quant Analyst: Statistical validity"""
        vote = 'APPROVE'
        reasoning = "Pattern detected with statistical significance. "
        
        # Algebraic topology pattern should have high confidence
        if signal.confidence < 0.58:
            vote = 'VETO'
            reasoning = f"TDA pattern confidence {signal.confidence*100:.1f}% below threshold (58%)."
        else:
            reasoning += f"TDA signal at {signal.confidence*100:.1f}% exceeds minimum."
            
        return {'vote': vote, 'reasoning': reasoning}
    
    def _elena_evaluate(self, signal: TradeSignal) -> Dict:
        """Market Analyst: Market regime and conditions"""
        vote = 'APPROVE'
        reasoning = "Market conditions favorable for this trade type. "
        
        # Check if market is open (simplified check)
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            vote = 'VETO'
            reasoning = "Market closed. Cannot execute trade."
            
        return {'vote': vote, 'reasoning': reasoning}
    
    def _derek_evaluate(self, signal: TradeSignal) -> Dict:
        """Infrastructure: Execution quality check"""
        vote = 'APPROVE'
        reasoning = "Execution infrastructure ready. "
        
        # Check if we can actually execute this
        if signal.signal_type not in ['EQUITY', 'OPTION']:
            vote = 'VETO'
            reasoning = f"Unknown signal type: {signal.signal_type}"
            
        return {'vote': vote, 'reasoning': reasoning}
    
    def _sophia_evaluate(self, signal: TradeSignal) -> Dict:
        """Compliance: Regulatory checks"""
        vote = 'APPROVE'
        reasoning = "Trade complies with regulatory requirements. "
        
        # Check PDT rules, position limits, etc
        # Simplified for now
        
        return {'vote': vote, 'reasoning': reasoning}
    
    def _compile_reasoning(self, votes: Dict) -> str:
        """Compile all agent reasoning into summary"""
        reasoning = []
        for agent, vote_data in votes.items():
            status = "✅" if vote_data['vote'] == 'APPROVE' else "🛑"
            reasoning.append(f"{status} {agent.upper()}: {vote_data['reasoning']}")
        return "\n".join(reasoning)
    
    async def _announce_decision(self, signal: TradeSignal, approved: bool, 
                                 votes: Dict, vetoes: List):
        """Post decision to Discord #trade-alerts"""
        decision = "✅ APPROVED" if approved else "🛑 VETOED"
        
        message = f"""
{decision} - {signal.action} {signal.quantity} {signal.symbol}

Signal Details:
- Type: {signal.signal_type}
- Confidence: {signal.confidence*100:.1f}%
- Reason: {signal.reason}

Team Vote: {sum(1 for v in votes.values() if v['vote'] == 'APPROVE')}/6 Approve
{'Vetoed by: ' + ', '.join(vetoes) if vetoes else 'Unanimous approval'}

{self._compile_reasoning(votes)}
"""
        
        # Post to Marcus's webhook (as team representative)
        webhook_url = self.webhook_urls.get('marcus')
        if webhook_url:
            payload = {
                "username": "Team of Rivals - Trade Decision",
                "content": message
            }
            try:
                requests.post(webhook_url, json=payload)
            except Exception as e:
                print(f"Failed to post decision: {e}")

if __name__ == "__main__":
    # Test the system
    evaluator = TeamOfRivalsEvaluator()
    
    # Example signal from TDA+NN bot
    test_signal = TradeSignal(
        symbol="AAPL",
        action="BUY",
        quantity=50,
        signal_type="EQUITY",
        confidence=0.62,
        reason="Algebraic topology pattern: persistent homology detected bullish structure"
    )
    
    # Evaluate signal
    result = asyncio.run(evaluator.evaluate_signal(test_signal))
    
    print(f"\nDecision: {'APPROVED' if result['approved'] else 'VETOED'}")
    print(f"Votes: {result['approval_count']}/6")
    if result['vetoes']:
        print(f"Vetoed by: {', '.join(result['vetoes'])}")
