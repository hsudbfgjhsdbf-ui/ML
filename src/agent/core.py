"""Agent AI multi-agent fraud detection system (Approach 3).

Implements a modular multi-agent workflow that mirrors a team of human claims
investigators: a document/eligibility agent, a policy-verification agent, an
anomaly-detection agent, a historical-pattern agent and a reasoning/decision
agent, coordinated by a coordinator.

Each agent emits structured findings (with confidence and evidence). The
reasoning agent synthesises the findings into an explainable verdict
(Approved / Flagged / Rejected) with a natural-language explanation.

Note: This sandbox runs without external LLM credentials, so the agents use
deterministic, transparent heuristics trained on the reference distributions
of the dataset. The interfaces mirror what an LLM-backed implementation
(LangChain/LangGraph + Gemini) would expose, and the same state graph can be
backed by an LLM when an API key is provided.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.utils import setup_logging

logger = setup_logging()


@dataclass
class AgentFinding:
    """Structured output of a single agent."""

    agent: str
    status: str                 # pass | warn | fail
    confidence: float           # 0..1
    details: str
    evidence: list = field(default_factory=list)   # specific evidence strings
    severity: str = "low"       # low | medium | high | critical


@dataclass
class ClaimState:
    """Shared state object passed between agents (LangGraph-style graph state)."""

    claim: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    risk_score: float = 0.0
    verdict: str = ""
    explanation: str = ""
    agent_log: list = field(default_factory=list)

    def add_finding(self, f: AgentFinding):
        self.findings.append(dataclasses.asdict(f))
        self.agent_log.append(f.agent)


# --------------------------------------------------------------------------
# Reference statistics learned from the data (the "local database").
# --------------------------------------------------------------------------
class ReferenceDatabase:
    """Stores typical-cost and fraud-baseline statistics per claim context.

    Provides the comparison baselines used by the anomaly and policy agents.
    """

    def __init__(self, df: pd.DataFrame):
        self.specialty_cost = df.groupby("ProviderSpecialty")["ClaimAmount"].agg(
            ["mean", "std", "median", lambda s: np.percentile(s, 90)])
        self.specialty_cost.columns = ["mean", "std", "median", "p90"]
        self.type_cost = df.groupby("ClaimType")["ClaimAmount"].agg(["mean", "std", "median"])
        self.age_avg = df["PatientAge"].mean()
        self.age_std = df["PatientAge"].std()
        self.income_avg = df["PatientIncome"].mean()
        self.base_fraud = float((df["ClaimLegitimacy"] == "Fraud").mean())

    def typical_cost(self, specialty, claim_type):
        """Return (mean, std) typical cost for a specialty+type context."""
        m = self.specialty_cost.loc[specialty, "mean"] if specialty in self.specialty_cost.index else self.type_cost["mean"].mean()
        s = self.specialty_cost.loc[specialty, "std"] if specialty in self.specialty_cost.index else self.type_cost["std"].mean()
        return float(m), float(s)


# --------------------------------------------------------------------------
# The five agents
# --------------------------------------------------------------------------
class EligibilityAgent:
    """Agent 1 - document/eligibility verification (completeness & plausibility)."""

    def __init__(self, db: ReferenceDatabase):
        self.db = db

    def run(self, claim: dict) -> AgentFinding:
        missing = [k for k in ["ClaimAmount", "PatientAge", "ProviderSpecialty",
                               "ClaimType", "ClaimDate", "PatientGender"] if k not in claim]
        if missing:
            return AgentFinding("EligibilityAgent", "fail", 0.3,
                                f"Missing critical fields: {missing}",
                                evidence=[f"Missing: {m}" for m in missing],
                                severity="high")
        if claim["PatientAge"] < 0 or claim["ClaimAmount"] <= 0:
            return AgentFinding("EligibilityAgent", "fail", 0.4,
                                "Implausible age or non-positive claim amount.", severity="critical")
        return AgentFinding("EligibilityAgent", "pass", 0.9,
                            "All required fields present and plausible.")


class PolicyAgent:
    """Agent 2 - policy verification against coverage baselines."""

    def __init__(self, db: ReferenceDatabase):
        self.db = db

    def run(self, claim: dict) -> AgentFinding:
        # simple coverage heuristics: senior citizens (>=60) in India often
        # carry co-pay and higher risk; extreme claims may breach sum-insured.
        warnings = []
        severity = "low"
        conf = 0.7
        if claim["PatientAge"] >= 60:
            warnings.append("Senior citizen profile - higher co-pay / waiting-period risk.")
            severity = "medium"
            conf = 0.75
        mean, std = self.db.typical_cost(claim.get("ProviderSpecialty", ""),
                                         claim.get("ClaimType", ""))
        if claim["ClaimAmount"] > 3 * mean:
            warnings.append(
                f"Claim amount Rs {claim['ClaimAmount']:,.0f} >3x typical "
                f"({mean:,.0f}) for {claim.get('ProviderSpecialty')}.")
            severity = "high"
            conf = 0.8
        status = "pass" if not warnings else "warn"
        return AgentFinding("PolicyAgent", status, conf,
                            "; ".join(warnings) if warnings else "Claim within coverage baselines.",
                            evidence=warnings, severity=severity)


class AnomalyAgent:
    """Agent 3 - anomaly detection on cost, age and temporal patterns."""

    def __init__(self, db: ReferenceDatabase):
        self.db = db

    def run(self, claim: dict) -> AgentFinding:
        anomalies = []
        severity = "low"
        conf = 0.6
        mean, std = self.db.typical_cost(claim.get("ProviderSpecialty", ""),
                                         claim.get("ClaimType", ""))
        z = (claim["ClaimAmount"] - mean) / (std + 1e-6)
        if z > 2.0:
            anomalies.append(f"Claim amount {z:.1f} std-devs above specialty norm (possible inflation).")
            severity = "high"; conf = 0.8
        # elderly with high claim relative to income
        if claim.get("PatientAge", 0) > 75 and claim.get("PatientIncome", 1e9) < 5e4:
            anomalies.append("Very senior age with low declared income - socio-economic mismatch.")
            severity = "medium"; conf = 0.7
        status = "pass" if not anomalies else "warn"
        return AgentFinding("AnomalyAgent", status, conf,
                            "; ".join(anomalies) if anomalies else "No material anomalies detected.",
                            evidence=anomalies, severity=severity)


class HistoricalAgent:
    """Agent 4 - historical/contextual pattern analysis (provider-level baseline)."""

    def __init__(self, db: ReferenceDatabase):
        self.db = db

    def run(self, claim: dict) -> AgentFinding:
        mean, std = self.db.typical_cost(claim.get("ProviderSpecialty", ""),
                                         claim.get("ClaimType", ""))
        evidence = []
        severity = "low"; conf = 0.6
        spec = claim.get("ProviderSpecialty", "")
        if spec in self.db.specialty_cost.index and \
                claim.get("ClaimAmount", 0) > self.db.specialty_cost.loc[spec, "p90"]:
            evidence.append("Claim above the 90th percentile for its specialty.")
            severity = "medium"; conf = 0.7
        status = "pass" if not evidence else "warn"
        return AgentFinding("HistoricalAgent", status, conf,
                            "; ".join(evidence) if evidence else "No elevated historical risk signals.",
                            evidence=evidence, severity=severity)


class ReasoningAgent:
    """Agent 5 - synthesises findings into a verdict with explanation."""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {"approve": 0.25, "reject": 0.7}

    def run(self, state: ClaimState) -> ClaimState:
        # aggregate a risk score from finding severities and confidences
        weights = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        total = 0.0; wsum = 0.0
        signals = []
        for f in state.findings:
            w = weights.get(f["severity"], 1)
            sign = 0.0 if f["status"] == "pass" else (1 if f["status"] == "warn" else 1.0)
            if f["status"] != "pass":
                total += w * f["confidence"]
                signals.append(f)
            wsum += w
        state.risk_score = min(1.0, total / (wsum * 4 + 1e-6) * 4)

        if state.risk_score >= self.thresholds["reject"]:
            verdict = "Rejected"
            reason = "high risk"
        elif state.risk_score >= self.thresholds["approve"]:
            verdict = "Flagged"
            reason = "moderate risk requiring manual review"
        else:
            verdict = "Approved"
            reason = "low risk"
        state.verdict = verdict
        state.explanation = self.build_explanation(state, verdict, reason)
        return state

    def build_explanation(self, state, verdict, reason):
        lines = [f"Decision: {verdict} ({reason})."]
        if state.findings:
            lines.append("Agent findings:")
            for f in state.findings:
                lines.append(f"- {f['agent']}: [{f['status'].upper()}] {f['details']}")
        lines.append("This decision is backed by the evidence cited above and is "
                     "reproducible; flagged items require human review.")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Coordinator - orchestrates the agent workflow
# --------------------------------------------------------------------------
class Coordinator:
    """Manages the multi-agent workflow and compiles the final report."""

    def __init__(self, db: ReferenceDatabase, verbose: bool = True):
        self.verbose = verbose
        self.agents = {
            "eligibility": EligibilityAgent(db),
            "policy": PolicyAgent(db),
            "anomaly": AnomalyAgent(db),
            "historical": HistoricalAgent(db),
            "reasoning": ReasoningAgent(),
        }

    def process(self, claim: dict) -> ClaimState:
        """Run the full agent workflow for a single claim."""
        state = ClaimState(claim=claim)
        # sequential orchestration (order matters for the reasoning step)
        for key in ["eligibility", "policy", "anomaly", "historical"]:
            finding = self.agents[key].run(claim)
            state.add_finding(finding)
            if self.verbose:
                logger.info("[%s] %s | %s", finding.agent, finding.status, finding.details)
        state = self.agents["reasoning"].run(state)
        return state

    def process_batch(self, claims: list[dict], labels: list[int] | None = None) -> dict:
        """Process many claims and optionally score against ground truth."""
        results = []
        for claim in claims:
            st = self.process(claim)
            results.append({"claim": claim, "state": st})
        report = {"results": results}
        if labels is not None:
            preds = [1 if r["state"].verdict in ("Rejected", "Flagged") else 0 for r in results]
            from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
            report["metrics"] = {
                "accuracy": float(accuracy_score(labels, preds)),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
                "f2": float(fbeta_score(labels, preds, beta=2, zero_division=0)),
            }
        return report
