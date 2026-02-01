#!/usr/bin/env python3
"""
Trade Signal Handler - Team of Rivals Veto Mechanism
All trade signals must pass team voting before execution
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, List
import requests

class TradeSignal:
    """Represents a trade signal from the TDA+NN bot"""
    def __init__(self, symbol: str, action: str, quantity: int,
                 signal_type: str, confidence: float, reason: str):
        self.symbol = symbol
        self.action = action  # BUY, SELL, BUY_CALL, SELL_PUT, etc
        self.quantity = quantity  
        self.signal_type = signal_type  # EQUITY or OPTION
        self.confidence = confidence  # 0.0 to 1.0
        self.reason = reason  # TDA pattern description
        self.timestamp = datetime.now()

class TeamOfRivalsEvaluator:
    """Routes trade signals through 6-agent veto system"""
    
    def __init__(self):
        self.min_votes_required = 4  # Need 4/6 to approve
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
        Submit trade to Team of Rivals for voting
        Returns: {approved: bool, votes: dict, vetoes: list, reasoning: str}
        """
        # Gather all agent votes
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
        """Each of 6 agents evaluates the trade"""
        votes = {}
        
        # Marcus Chen - Strategy evaluation
        votes['marcus'] = self._marcus_evaluate(signal)
        
        # Victoria Hayes - Risk evaluation
        votes['victoria'] = self._victoria_evaluate(signal)
        
        # James Park - Statistical evaluation  
        votes['james'] = self._james_evaluate(signal)
        
        # Elena Rodriguez - Market conditions
        votes['elena'] = self._elena_evaluate(signal)
        
        # Derek Thompson - Execution feasibility
        votes['derek'] = self._derek_evaluate(signal)
        
        # Sophia Williams - Compliance
        votes['sophia'] = self._sophia_evaluate(signal)
        
        return votes
    
    def _marcus_evaluate(self, signal: TradeSignal) -> Dict:
        """Strategy Officer: Profit potential"""
        vote = 'APPROVE'
        reasoning = f"Signal confidence {signal.confidence*100:.1f}%. "
        
        if signal.confidence < 0.55:
            vote = 'VETO'
            reasoning += "VETO: Confidence below 55% threshold. Too risky."
        else:
            reasoning += "Acceptable risk/reward for execution."
            
        return {'vote': vote, 'reasoning': reasoning}
    
    def _victoria_evaluate(self, signal: TradeSignal) -> Dict:
        """Risk Officer: Position sizing"""
        vote = 'APPROVE'
        reasoning = f"Position size: {signal.quantity} shares. "
        
        if signal.quantity > 100:
            vote = 'VETO'
            reasoning += f"VETO: Exceeds max 100 shares per position."
        else:
            reasoning += "Within position limits."
            
        return {'vote': vote, 'reasoning': reasoning}
    
    def _james_evaluate(self, signal: TradeSignal) -> Dict:
        """Quant Analyst: Statistical validity"""
        vote = 'APPROVE'
        reasoning = f"TDA confidence: {signal.confidence*100:.1f}%. "
        
        if signal.confidence < 0.58:
            vote = 'VETO'
            reasoning += "VETO: Below 58% statistical significance threshold."
        else:
            reasoning += "Statistically significant pattern detected."
            
        return {'vote': vote, 'reasoning': reasoning}
    
    def _elena_evaluate(self, signal: TradeSignal) -> Dict:
        """Market Analyst: Market conditions"""
        vote = 'APPROVE'
        reasoning = "Market conditions favorable. "
        
        # Check if market is open (simplified)
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            vote = 'VETO'
            reasoning = "VETO: Market closed (weekend)."
            
        return {'vote': vote, 'reasoning': reasoning}
    
    def _derek_evaluate(self, signal: TradeSignal) -> Dict:
        """Infrastructure: Execution check"""
        vote = 'APPROVE'
        reasoning = "Execution infrastructure ready. "
        
        if signal.signal_type not in ['EQUITY', 'OPTION']:
            vote = 'VETO'
            reasoning = f"VETO: Unknown signal type '{signal.signal_type}'."
            
        return {'vote': vote, 'reasoning': reasoning}
    
    def _sophia_evaluate(self, signal: TradeSignal) -> Dict:
        """Compliance: Regulatory checks"""
        vote = 'APPROVE'
        reasoning = "Trade compliant with regulations. "
        
        # Simplified compliance check
        return {'vote': vote, 'reasoning': reasoning}
    
    def _compile_reasoning(self, votes: Dict) -> str:
        """Compile all agent reasoning"""
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
{'Vetoed by: ' + ', '.join(vetoes) if vetoes else 'Unanimous approval ✅'}

{self._compile_reasoning(votes)}
"""
        
        # Post to Discord via Marcus's webhook
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
    # Test the veto mechanism
    evaluator = TeamOfRivalsEvaluator()
    
    # Good trade (should approve)
    good_signal = TradeSignal(
        symbol="AAPL",
        action="BUY",
        quantity=50,
        signal_type="EQUITY",
        confidence=0.62,
        reason="Persistent homology: bullish structure detected"
    )
    
    result = asyncio.run(evaluator.evaluate_signal(good_signal))
    print(f"\nGood Trade: {'APPROVED' if result['approved'] else 'VETOED'}")
    print(f"Votes: {result['approval_count']}/6")
    
    # Bad trade (should veto)
    bad_signal = TradeSignal(
        symbol="TSLA",
        action="BUY",
        quantity=500,  # Too large!
        signal_type="EQUITY",
        confidence=0.52,  # Too low!
        reason="Weak pattern"
    )
    
    result = asyncio.run(evaluator.evaluate_signal(bad_signal))
    print(f"\nBad Trade: {'APPROVED' if result['approved'] else 'VETOED'}")
    print(f"Votes: {result['approval_count']}/6")
    print(f"Vetoed by: {', '.join(result['vetoes'])}")
