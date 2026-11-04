"""Process-level services shared by graph nodes (kept out of checkpointed state)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.agents.planner.plan_validator import PlanValidator
from backend.agents.planner.requirement_analyzer import RequirementAnalyzer
from backend.agents.planner.test_compiler import TestCompiler
from backend.agents.governance.secret_manager import SecretManager, get_secret_manager
from backend.harness.model_gateway import ModelGateway, get_model_gateway


@dataclass
class GraphServices:
    gateway: ModelGateway = field(default_factory=get_model_gateway)
    secrets: SecretManager = field(default_factory=get_secret_manager)
    sessionmaker: Any = None  # async_sessionmaker or None (offline)
    compiler: TestCompiler = field(default_factory=TestCompiler)
    validator: PlanValidator = field(default_factory=PlanValidator)

    def analyzer(self) -> RequirementAnalyzer:
        return RequirementAnalyzer(self.gateway if self.gateway.available else None)


_services: GraphServices | None = None


def get_graph_services() -> GraphServices:
    global _services
    if _services is None:
        _services = GraphServices()
    return _services


def set_graph_services(services: GraphServices) -> None:
    global _services
    _services = services
