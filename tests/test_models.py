import pytest
from agentic_core.models import AgentResponse, DAGNodeResponse, DAGResponse, StreamEvent, StreamEventType

def test_agent_response_to_dict():
    resp = AgentResponse(text="Hello", reasoning="Thinking", usage={"tokens": 10})
    d = resp.to_dict()
    assert d["text"] == "Hello"
    assert d["reasoning"] == "Thinking"
    assert d["usage"] == {"tokens": 10}
    assert d["error"] is None

def test_dag_node_response_to_dict():
    agent_resp = AgentResponse(text="Success")
    resp = DAGNodeResponse(state="SUCCESS", result=agent_resp, failed_by="NodeA")
    d = resp.to_dict()
    assert d["state"] == "SUCCESS"
    assert d["failed_by"] == "NodeA"
    assert d["result"] == agent_resp

def test_dag_response_to_dict():
    agent_resp = AgentResponse(text="Success")
    node_resp = DAGNodeResponse(state="SUCCESS", result=agent_resp)
    dag_resp = DAGResponse(nodes={"Node1": node_resp})
    d = dag_resp.to_dict()
    assert "Node1" in d["nodes"]
    assert d["nodes"]["Node1"]["state"] == "SUCCESS"
    
def test_stream_event():
    event = StreamEvent(type=StreamEventType.TEXT, content="Chunk")
    assert event.type == StreamEventType.TEXT
    assert event.content == "Chunk"
