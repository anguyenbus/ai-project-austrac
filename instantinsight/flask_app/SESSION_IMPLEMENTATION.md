# Flask App Session Management Implementation Guide

## Using Strands FileSessionManager for Demo Purposes

**Purpose:** Demo/Development Only
**Scope:** Per-visualization conversation persistence
**Cleanup:** Automatic on Flask app shutdown

---

## Overview

This guide provides a **simple FileSessionManager-based approach** for the Flask demo app to persist conversation history across visualization modifications. Since the Flask app is for **demo purposes only**, we use Strands' built-in [`FileSessionManager`](https://github.com/agentsea/strands) directly with automatic cleanup on app shutdown.

**Key Principles:**

- ✅ Simple, demo-appropriate implementation
- ✅ Uses Strands FileSessionManager directly (no wrapper)
- ✅ Per-visualization session tracking
- ✅ Auto-cleanup on app close
- ❌ Not production-ready (no encryption, limited security)
- ❌ File-based storage only (not scalable)

---

## Architecture

```
┌──────────────┐
│   Browser    │  User modifies visualization
└──────┬───────┘
       │ viz_id + modification_request
       ▼
┌─────────────────────────────────────────┐
│         Flask ModificationHandler       │
│  ┌─────────────────────────────────┐   │
│  │  Per-Viz Session Management     │   │
│  │  viz_id → FileSessionManager    │   │
│  └─────────────────────────────────┘   │
└──────┬──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  ./flask_sessions/               │
│    └─ viz_<uuid>/                │
│         └─ session.json          │  ← Strands manages this
└──────────────────────────────────┘
```

**Session Lifecycle:**

1. User creates visualization → Generate `viz_id` → Create FileSessionManager
2. User modifies visualization → Load session → Update agents → Save automatically
3. Flask app closes → Cleanup all sessions in `./flask_sessions/`

---

## Implementation

### Step 1: Update ModificationHandler

```python
# flask_app/modification_handler.py
from pathlib import Path
from typing import Any
import atexit
import shutil

from loguru import logger
from strands.session.file_session_manager import FileSessionManager

from src.agents.strand_agents.output.visualizer import VisualizationAgent
from src.agents.strand_agents.query.modification_decider import ModificationDecisionAgent
from src.rag.pipeline import Pipeline

class ModificationHandler:
    """Handles visualization modifications with session persistence."""
  
    def __init__(self, pipeline: Pipeline | None = None):
        """Initialize modification handler with session support."""
        self.pipeline = pipeline
    
        # Session configuration
        self.session_dir = Path("./flask_sessions")
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
        # Session managers per visualization
        self._session_managers: dict[str, FileSessionManager] = {}
    
        # Agent instances per session (reused across modifications)
        self._decision_agents: dict[str, ModificationDecisionAgent] = {}
        self._visualization_agents: dict[str, VisualizationAgent] = {}
    
        # Register cleanup on exit
        atexit.register(self._cleanup_sessions)
    
        logger.info(f"✅ ModificationHandler initialized with sessions in {self.session_dir}")
  
    def _get_session_manager(self, viz_id: str) -> FileSessionManager:
        """Get or create FileSessionManager for a visualization."""
        if viz_id not in self._session_managers:
            session_mgr = FileSessionManager(
                session_id=viz_id,
                storage_dir=str(self.session_dir),
            )
            self._session_managers[viz_id] = session_mgr
            logger.info(f"Created session for viz: {viz_id}")
    
        return self._session_managers[viz_id]
  
    def _get_decision_agent(self, viz_id: str) -> ModificationDecisionAgent:
        """Get or create ModificationDecisionAgent with session."""
        if viz_id not in self._decision_agents:
            session_mgr = self._get_session_manager(viz_id)
        
            # Create agent core with session manager
            from src.agents.strand_agents.query.modification_decider import ModificationDecisionCore
            core = ModificationDecisionCore(session_id=viz_id)
        
            # Attach session manager to agent
            core.agent.session_manager = session_mgr
        
            # Wrap in compatibility layer
            agent = ModificationDecisionAgent()
            agent.core = core
        
            self._decision_agents[viz_id] = agent
            logger.info(f"Created session-aware ModificationDecisionAgent for {viz_id}")
    
        return self._decision_agents[viz_id]
  
    def _get_visualization_agent(self, viz_id: str) -> VisualizationAgent:
        """Get or create VisualizationAgent with session."""
        if viz_id not in self._visualization_agents:
            session_mgr = self._get_session_manager(viz_id)
        
            # Create agent core with session manager
            from src.agents.strand_agents.output.visualizer import VisualizationCore
            core = VisualizationCore(session_id=viz_id)
        
            # Attach session manager to agent
            core.agent.session_manager = session_mgr
        
            # Wrap in compatibility layer
            agent = VisualizationAgent()
            agent.core = core
        
            self._visualization_agents[viz_id] = agent
            logger.info(f"Created session-aware VisualizationAgent for {viz_id}")
    
        return self._visualization_agents[viz_id]
  
    def process_modification_request(
        self,
        user_message: str,
        current_viz_data: dict,
        viz_id: str,
        conversation_history: list,
    ) -> dict:
        """
        Process modification request with session persistence.
    
        Args:
            user_message: User's modification request
            current_viz_data: Current visualization data
            viz_id: Visualization identifier (used as session_id)
            conversation_history: In-memory conversation history (for UI display)
    
        Returns:
            Dict with success, message, updated_figure, updated_context
        """
        try:
            # Get session-aware agents
            decision_agent = self._get_decision_agent(viz_id)
            visualization_agent = self._get_visualization_agent(viz_id)
        
            # NOTE: Agents now have conversation history in session
            # Strands FileSessionManager automatically persists:
            # - agent.messages (conversation turns)
            # - agent.state (custom state like viz metadata)
        
            context = current_viz_data.get("context", {})
            current_sql = context.get("sql_query", "")
            current_plotly_schema = context.get("plotly_schema", {})
        
            # Build compact schema for prompt
            compact_schema = self._build_compact_schema(
                current_plotly_schema,
                metadata=current_viz_data.get("metadata"),
                context=context,
            )
        
            logger.info(f"Processing modification: {user_message}")
        
            # Use decision agent (session-aware)
            decision_result = decision_agent.decide_processing_approach(
                historical_context=self._get_historical_context(viz_id, context),
                user_message=user_message,
                current_sql=current_sql,
                current_plotly_schema=compact_schema,
                user_id=viz_id,  # Use viz_id as user_id for session scoping
            )
        
            if not decision_result.get("success"):
                return {
                    "success": False,
                    "message": f"❌ Decision failed: {decision_result.get('error')}",
                }
        
            processing_method = decision_result.get("processing_method")
            logger.info(f"Decision: {processing_method}")
        
            # Route to appropriate handler (existing logic)
            if processing_method == "simple_visual_modification":
                result = self._handle_simple_visual(decision_result, current_viz_data, user_message)
            elif processing_method == "visualization_agent_only":
                result = self._handle_schema_only(
                    decision_result, current_viz_data, user_message, visualization_agent
                )
            elif processing_method == "simple_sql_modification":
                result = self._handle_simple_sql(decision_result, current_viz_data, user_message)
            else:
                result = self._handle_full_pipeline(
                    decision_result, current_viz_data, user_message,
                    self._get_historical_context(viz_id, context),
                    current_sql, compact_schema
                )
        
            # Session automatically saved by Strands after agent interactions
            return result
        
        except Exception as e:
            logger.error(f"Modification processing failed: {e}")
            return {"success": False, "message": f"❌ Error: {str(e)}"}
  
    def _get_historical_context(self, viz_id: str, context: dict) -> str:
        """Get historical context from session."""
        session_mgr = self._session_managers.get(viz_id)
    
        if session_mgr and hasattr(session_mgr, "agent"):
            # Extract conversation history from session
            messages = session_mgr.agent.messages
            if messages:
                # Format last 3 turns for context
                recent = messages[-6:] if len(messages) > 6 else messages
                history_parts = []
                for msg in recent:
                    role = getattr(msg, "role", "unknown")
                    content = getattr(msg, "content", "")
                    if content:
                        history_parts.append(f"{role}: {content[:100]}")
            
                return "\n".join(history_parts)
    
        # Fallback to original question
        original = context.get("original_question", "").strip()
        if original:
            return f"Original question: {original}"
        return ""
  
    def _handle_schema_only(
        self,
        decision_result: dict,
        viz_data: dict,
        user_message: str,
        visualization_agent: VisualizationAgent,  # Session-aware agent
    ) -> dict:
        """Handle schema-only modifications using session-aware VisualizationAgent."""
        try:
            current_data = viz_data.get("data")
            if current_data is None or current_data.empty:
                return {"success": False, "message": "❌ No data available"}
        
            # Use session-aware visualization agent
            viz_query = f"Create visualization: {decision_result.get('modify_schema', user_message)}"
            viz_result = visualization_agent.process(current_data, viz_query)
        
            if viz_result.get("success"):
                # Extract figure and update context
                from .visualization_utils import extract_plotly_figure
                updated_figure, viz_metadata = extract_plotly_figure(viz_result)
            
                if updated_figure:
                    context = viz_data.get("context", {})
                    new_context = {
                        "sql_query": context.get("sql_query", ""),
                        "original_question": context.get("original_question", ""),
                        "user_modification": user_message,
                        "plotly_schema": viz_result,
                        "data_shape": context.get("data_shape", ""),
                    }
                
                    if viz_metadata:
                        new_context.update({
                            "chart_type": viz_metadata.get("chart_type"),
                            "confidence": viz_metadata.get("confidence"),
                        })
                
                    return {
                        "success": True,
                        "message": f"✅ Updated visualization (session-aware)",
                        "updated_figure": updated_figure,
                        "updated_context": new_context,
                    }
        
            return {"success": False, "message": f"❌ Visualization failed"}
        
        except Exception as e:
            logger.error(f"Schema-only modification failed: {e}")
            return {"success": False, "message": f"❌ Error: {str(e)}"}
  
    def _cleanup_sessions(self) -> None:
        """Cleanup all session directories on Flask app shutdown."""
        try:
            if self.session_dir.exists():
                import shutil
                shutil.rmtree(self.session_dir)
                logger.info(f"✅ Cleaned up session directory: {self.session_dir}")
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
  
    # ... existing _handle_simple_visual, _handle_simple_sql, _handle_full_pipeline methods unchanged
```

---

## Session Data Structure

Strands FileSessionManager automatically creates:

```
flask_sessions/
├─ viz_<uuid-1>/
│   └─ session.json       # Contains agent.messages + agent.state
├─ viz_<uuid-2>/
│   └─ session.json
└─ viz_<uuid-3>/
    └─ session.json
```

**session.json structure (managed by Strands):**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Change this to a pie chart",
      "timestamp": "2025-10-23T04:00:00Z"
    },
    {
      "role": "assistant",
      "content": "I'll convert it to a pie chart...",
      "timestamp": "2025-10-23T04:00:05Z"
    }
  ],
  "state": {
    "viz_id": "viz_abc123",
    "chart_type": "pie",
    "modification_count": 3
  }
}
```

---

## Usage Example

```python
# flask_app/app.py
from flask import Flask, request, jsonify
from src.rag.pipeline import Pipeline
from .modification_handler import ModificationHandler

app = Flask(__name__)

# Initialize with session support
pipeline = Pipeline()
handler = ModificationHandler(pipeline=pipeline)

@app.route("/api/modify", methods=["POST"])
def modify_visualization():
    """Modify visualization with session persistence."""
    data = request.json
  
    result = handler.process_modification_request(
        user_message=data["message"],
        current_viz_data=data["viz_data"],
        viz_id=data["viz_id"],  # Used as session_id
        conversation_history=data.get("history", []),
    )
  
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
    # NOTE: handler._cleanup_sessions() called automatically via atexit
```

---

## Testing

```python
# tests/flask_app/test_session_handler.py
import pytest
from pathlib import Path
from flask_app.modification_handler import ModificationHandler

@pytest.fixture
def handler():
    """Create handler with test session directory."""
    handler = ModificationHandler()
    yield handler
    # Cleanup handled by atexit, but explicit for tests
    handler._cleanup_sessions()

def test_session_creation(handler):
    """Test session manager creation per viz_id."""
    viz_id = "test-viz-123"
  
    # First access creates session
    session_mgr = handler._get_session_manager(viz_id)
    assert session_mgr is not None
    assert viz_id in handler._session_managers
  
    # Second access returns same session
    session_mgr2 = handler._get_session_manager(viz_id)
    assert session_mgr is session_mgr2

def test_session_persistence(handler):
    """Test conversation history persistence."""
    viz_id = "test-viz-456"
  
    # Get agents (creates session)
    decision_agent = handler._get_decision_agent(viz_id)
  
    # Simulate agent interaction
    decision_agent.core.agent.messages.append({
        "role": "user",
        "content": "Make it blue",
    })
  
    # Session should persist automatically
    session_mgr = handler._get_session_manager(viz_id)
    assert len(session_mgr.agent.messages) == 1
    assert session_mgr.agent.messages[0]["content"] == "Make it blue"

def test_session_cleanup(handler):
    """Test session cleanup on shutdown."""
    viz_id = "test-viz-789"
    handler._get_session_manager(viz_id)
  
    # Verify session directory exists
    session_path = handler.session_dir / viz_id
    assert handler.session_dir.exists()
  
    # Cleanup
    handler._cleanup_sessions()
  
    # Verify cleaned up
    assert not handler.session_dir.exists()
```

---

## Quick Start Checklist

- [ ] Copy code above into [`modification_handler.py`](modification_handler.py)
- [ ] Add Strands dependency: `uv add strands`
- [ ] Create test file: `tests/flask_app/test_session_handler.py`
- [ ] Run tests: `uv run pytest tests/flask_app/test_session_handler.py`
- [ ] Start Flask app: `uv run python flask_app/run_flask.py`
- [ ] Verify sessions created in `./flask_sessions/`
- [ ] Test modification continuity across requests
- [ ] Stop Flask app → Verify cleanup
