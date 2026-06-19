"""
This module has been removed to eliminate architectural bloat.

Please use `agentic_core.engines.dag_engine.DAGAgentRunner` which now natively 
supports state-aware conditional routing, Context Reducers, and cyclical control-flow abstractions,
entirely replacing the need for an unstructured swarm engine.
"""

class StatefulSwarmEngine:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "StatefulSwarmEngine has been completely removed to enforce a single source of truth for orchestration. "
            "Migrate to DAGAgentRunner."
        )
