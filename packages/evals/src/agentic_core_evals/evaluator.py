import logging
from typing import Any, Callable, Awaitable
from agentic_core.llm_providers import ILLMClient
from agentic_core.engines.engine import AgentRunner
from agentic_core.utils import heuristic_json_parse

logger = logging.getLogger(__name__)

class TrajectoryEvaluator:
    """Evaluates agent execution trajectories against datasets using LLM-as-a-Judge."""
    def __init__(self, judge_llm: ILLMClient):
        self.judge_llm = judge_llm

    async def evaluate(self, runner_factory: Callable[[], Awaitable[AgentRunner]], dataset: list[dict[str, str]]) -> list[dict[str, Any]]:
        results = []
        for i, data in enumerate(dataset):
            logger.info(f"Evaluating sample {i+1}/{len(dataset)}...")
            runner = await runner_factory()
            response = await runner.run_turn(data["input"])
            
            prompt = (
                "You are an impartial Judge evaluating an AI agent.\n"
                f"User Input: {data['input']}\n"
                f"Expected Outcome: {data['expected']}\n"
                f"Agent Actual Response: {response.text}\n"
                "Provide a JSON object with 'score' (0 to 10) and 'reasoning' (string explaining the score)."
            )
            
            judge_resp_text = ""
            async for chunk in self.judge_llm.ask([{"role": "user", "content": prompt}]):
                if chunk.text: judge_resp_text += chunk.text
                
            try:
                eval_result = heuristic_json_parse(judge_resp_text)
            except Exception:
                eval_result = {"score": 0, "reasoning": "Judge failed to output valid JSON."}
                
            results.append({
                "input": data["input"],
                "expected": data["expected"],
                "actual": response.text,
                "score": eval_result.get("score", 0),
                "reasoning": eval_result.get("reasoning", "")
            })
            
        return results
