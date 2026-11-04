"""Scenario planner / test compiler (spec sections 7.3, 26.1 scenario_planner_agent).

Takes the analyzer's normalized spec and produces the executable plan: implicit
initial navigation, sequential IDs, risk hints, form groups for the form runner,
navigation expectations and expected postconditions.
"""

from __future__ import annotations

from backend.agents.planner.lexicon import risk_for_text
from backend.dsl.models import ActionType, Step, TargetSpec, TestSpec

FORM_ACTIONS = {ActionType.FILL, ActionType.TYPE, ActionType.SELECT, ActionType.CHECK, ActionType.UNCHECK,
                ActionType.CLEAR, ActionType.FILL_FORM, ActionType.UPLOAD}


class TestCompiler:
    name = "scenario_planner_agent"

    def compile(self, spec: TestSpec) -> TestSpec:
        steps = [s.model_copy(deep=True) for s in spec.steps]
        if spec.base_url and not _starts_with_navigation(steps):
            steps.insert(0, Step(id="step_000", index=0, original_text=f"(implicit) Open {spec.base_url}",
                                 action=ActionType.NAVIGATE, target=TargetSpec(intent="url", url=spec.base_url),
                                 risk_hint="navigation", expects_navigation=True, notes=["implicit_navigation"]))
        counter = [0]

        def renumber(items: list[Step], prefix: str = "") -> None:
            for step in items:
                counter[0] += 1
                step.index = counter[0]
                step.id = f"step_{counter[0]:03d}"
                renumber(step.then_steps)
                renumber(step.else_steps)

        renumber(steps)
        flat = [s for step in steps for s in step.iter_flat()]
        self._link_page_openers(flat)
        group = 0
        previous_form_action = False
        for step in flat:
            self._annotate_risk(step)
            if step.action in FORM_ACTIONS:
                if not previous_form_action:
                    group += 1
                step.notes = [n for n in step.notes if not n.startswith("form_group:")] + [f"form_group:{group}"]
                previous_form_action = True
            elif step.action == ActionType.CLICK and previous_form_action and step.risk_hint in ("submit", None):
                step.notes.append(f"submits_form_group:{group}")
                step.expects_navigation = step.expects_navigation or bool(
                    step.target and step.target.purpose in ("submit_login", "submit_register"))
                previous_form_action = False
            else:
                previous_form_action = False
            if step.expected_postcondition is None:
                step.expected_postcondition = _postcondition(step)
        compiled = spec.model_copy(update={"steps": steps})
        compiled.analyzer = {**spec.analyzer, "compiled": True, "step_count": len(flat)}
        return compiled

    @staticmethod
    def _link_page_openers(flat: list[Step]) -> None:
        """'Open the login page' + 'Click on the Login button': the click *is* how the page opens.

        The OPEN_PAGE step then only checks where we are instead of clicking the same control twice.
        """

        from backend.agents.grounding.text_match import similarity

        for current, nxt in zip(flat, flat[1:]):
            if current.action != ActionType.OPEN_PAGE or nxt.action != ActionType.CLICK or not current.target \
                    or not nxt.target:
                continue
            page = current.target.intent.replace(" page", "")
            same_purpose = bool(current.target.purpose and current.target.purpose == nxt.target.purpose)
            same_name = similarity(page, nxt.target.exact_name or nxt.target.intent) >= 0.8
            if same_purpose or same_name:
                current.notes.append(f"opened_by_next_step:{nxt.id}")
                nxt.notes.append(f"opens_page:{page}")
                nxt.expects_navigation = True

    @staticmethod
    def _annotate_risk(step: Step) -> None:
        if step.action in (ActionType.ASSERT, ActionType.WAIT, ActionType.CAPTURE_SCREENSHOT, ActionType.CAPTURE_VALUE,
                           ActionType.HOVER, ActionType.SCROLL):
            step.risk_hint = "read"
            return
        if step.action in (ActionType.NAVIGATE, ActionType.OPEN_PAGE, ActionType.GO_BACK, ActionType.RELOAD,
                           ActionType.SWITCH_TAB, ActionType.SWITCH_FRAME):
            step.risk_hint = "navigation"
            return
        if step.action in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.PRESS_KEY, ActionType.ACCEPT_DIALOG):
            text = " ".join(filter(None, [step.target.exact_name if step.target else None,
                                          step.target.intent if step.target else None, step.original_text]))
            risk = risk_for_text(text)
            if risk in ("destructive", "financial", "messaging"):
                step.risk_hint = risk  # type: ignore[assignment]
            elif step.risk_hint is None:
                step.risk_hint = risk or "input"  # type: ignore[assignment]
            return
        if step.risk_hint is None:
            step.risk_hint = "input"


def _starts_with_navigation(steps: list[Step]) -> bool:
    for step in steps:
        if step.action in (ActionType.NAVIGATE,):
            return True
        if step.action in (ActionType.WAIT, ActionType.CAPTURE_SCREENSHOT):
            continue
        return False
    return False


def _postcondition(step: Step) -> str | None:
    return {
        ActionType.NAVIGATE: "page loaded at the requested URL",
        ActionType.OPEN_PAGE: "requested page is displayed",
        ActionType.FILL: "field contains the entered value",
        ActionType.TYPE: "field contains the typed text",
        ActionType.CLEAR: "field is empty",
        ActionType.SELECT: "requested option is selected",
        ActionType.CHECK: "control is checked",
        ActionType.UNCHECK: "control is unchecked",
        ActionType.UPLOAD: "file is attached",
        ActionType.SWITCH_TAB: "requested tab is active",
        ActionType.ACCEPT_DIALOG: "dialog is closed",
        ActionType.DISMISS_DIALOG: "dialog is closed",
        ActionType.CLICK: "click produced an observable effect",
        ActionType.DRAG: "item moved to the destination",
        ActionType.DOWNLOAD: "download started",
    }.get(step.action)
