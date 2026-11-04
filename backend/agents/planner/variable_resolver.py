"""Runtime value resolution: literals, test data, generated data, captured values, secrets."""

from __future__ import annotations

import datetime as dt
import re
import secrets as pysecrets
import uuid
from dataclasses import dataclass

from backend.agents.governance.redaction import register_sensitive
from backend.agents.governance.secret_manager import SecretManager, SecretScope
from backend.core.exceptions import ValidationFailed
from backend.dsl.models import ValueSpec

TEMPLATE_RE = re.compile(r"\{\{\s*([A-Za-z_][\w.]*)\s*\}\}")


@dataclass
class ResolvedValue:
    text: str
    sensitive: bool
    source: str

    def display(self) -> str:
        return "••••••" if self.sensitive else self.text


class VariableResolver:
    def __init__(self, *, variables: dict[str, str], secret_manager: SecretManager, scope: SecretScope,
                 run_token: str, captured: dict[str, str] | None = None) -> None:
        self.variables = dict(variables)
        self.secret_manager = secret_manager
        self.scope = scope
        self.run_token = run_token
        self.captured = captured if captured is not None else {}
        self._generated: dict[str, str] = {}

    def generated(self, name: str) -> str:
        if name in self._generated:
            return self._generated[name]
        short = self.run_token[-6:]
        today = dt.date.today()
        value = {
            "generated_email": f"aitest.{short}@example.com",
            "generated_username": f"aitest_{short}",
            "generated_name": f"Test User {short.upper()}",
            "generated_phone": f"+1555{int(short, 16) % 10_000_000:07d}" if re.fullmatch(r"[0-9a-f]+", short)
            else "+15550100",
            "current_date": today.isoformat(),
            "today": today.isoformat(),
            "timestamp": dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S"),
            "uuid": str(uuid.uuid4()),
            "run_id": self.run_token,
        }.get(name)
        if value is None:
            if name.startswith("random_"):
                value = pysecrets.token_hex(4)
            elif name.startswith("generated_"):
                value = f"{name.removeprefix('generated_')}_{short}"
            else:
                raise ValidationFailed(f"Unknown generator {name!r}")
        self._generated[name] = value
        return value

    async def resolve(self, spec: ValueSpec | None) -> ResolvedValue:
        if spec is None or spec.source == "none":
            return ResolvedValue("", False, "none")
        if spec.source == "secret" and spec.reference:
            value = await self.secret_manager.resolve(self.scope, spec.reference)
            return ResolvedValue(value, True, "secret")
        if spec.source == "variable" and spec.reference:
            if spec.reference in self.captured:
                return ResolvedValue(self.captured[spec.reference], spec.sensitive, "captured")
            if spec.reference not in self.variables:
                raise ValidationFailed(f"Test data does not define {{{{{spec.reference}}}}}")
            value = self.variables[spec.reference]
            if spec.sensitive:
                register_sensitive(value)
            return ResolvedValue(value, spec.sensitive, "variable")
        if spec.source == "generated":
            return ResolvedValue(self.generated(spec.generator or spec.reference or "random_value"), False,
                                 "generated")
        if spec.values:
            return ResolvedValue(", ".join([await self.interpolate(v) for v in spec.values]), spec.sensitive, "literal")
        value = await self.interpolate(spec.value or "")
        if spec.sensitive:
            register_sensitive(value)
        return ResolvedValue(value, spec.sensitive, "literal")

    async def interpolate(self, template: str) -> str:
        out = template
        for name in TEMPLATE_RE.findall(template):
            if name in self.captured:
                replacement = self.captured[name]
            elif name in self.variables:
                replacement = self.variables[name]
            elif name.startswith(("generated_", "random_")) or name in ("current_date", "today", "timestamp", "uuid",
                                                                        "run_id"):
                replacement = self.generated(name)
            else:
                replacement = await self.secret_manager.resolve(self.scope, name)
            out = re.sub(r"\{\{\s*" + re.escape(name) + r"\s*\}\}", lambda _m, r=replacement: r, out)
        return out
