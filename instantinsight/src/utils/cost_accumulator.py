"""Cost accumulator for tracking total costs across pipeline execution."""

from dataclasses import dataclass


@dataclass
class AgentCost:
    """Cost tracking for a single agent execution."""

    agent_name: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    model_id: str

    @property
    def total_cost(self) -> float:
        """Calculate total cost for this agent execution."""
        return self.input_cost + self.output_cost

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens for this agent execution."""
        return self.input_tokens + self.output_tokens


class CostAccumulator:
    """Accumulate costs from multiple agents during pipeline execution."""

    def __init__(self):
        """Initialize cost accumulator with empty state."""
        self.agent_costs: list[AgentCost] = []
        self.trace_id: str = None

    def add_agent_cost(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        input_cost: float,
        output_cost: float,
        model_id: str,
    ):
        """
        Add cost information for an agent.

        Args:
            agent_name: Name of the agent
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            input_cost: Cost for input tokens
            output_cost: Cost for output tokens
            model_id: Model identifier used

        """
        cost = AgentCost(
            agent_name=agent_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            model_id=model_id,
        )
        self.agent_costs.append(cost)

    def get_total_cost(self) -> float:
        """
        Get total cost across all agents.

        Returns:
            Total accumulated cost

        """
        return sum(cost.total_cost for cost in self.agent_costs)

    def get_total_tokens(self) -> int:
        """
        Get total tokens across all agents.

        Returns:
            Total token count

        """
        return sum(cost.total_tokens for cost in self.agent_costs)

    def get_cost_breakdown(self) -> dict[str, float]:
        """
        Get cost breakdown by agent.

        Returns:
            Dictionary mapping agent names to costs

        """
        return {cost.agent_name: cost.total_cost for cost in self.agent_costs}

    def set_trace_id(self, trace_id: str):
        """
        Set the current trace ID.

        Args:
            trace_id: Trace identifier to set

        """
        self.trace_id = trace_id

    def reset(self):
        """Reset the accumulator for a new pipeline execution."""
        self.agent_costs.clear()
        self.trace_id = None


# Global cost accumulator instance
cost_accumulator = CostAccumulator()
