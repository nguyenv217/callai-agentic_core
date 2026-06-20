from agentic_core.engines.dag_engine import GraphAgentRunner

class StudioExporter:
    """Foundational tools for Observability Dashboards."""
    @staticmethod
    def export_graph_to_mermaid(graph: GraphAgentRunner) -> str:
        """Helper function to abstract Mermaid generation for web UI rendering."""
        return graph.to_mermaid()
