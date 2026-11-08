"""Evidence-grounded physical reasoning with an optional Cosmos 3 API client.

Reasoners may suggest hypotheses or probes; they NEVER generate executable
robot commands through this module. All references must point to observations
available *before* action selection. Claims without grounded refs are rejected.

The optional Cosmos client uses an OpenAI-compatible chat endpoint as described
in the official Cosmos3 Reasoner recipes. Network/model inference is NOT part
of the default tests; a fake transport verifies request/response behaviour.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import json
import math
import os
import urllib.request


@dataclass(frozen=True)
class PhysicalFact:
    key: str
    value: float
    unit: str
    source: str

    def __post_init__(self):
        if not self.key or not self.unit or not self.source:
            raise ValueError("Each physical fact requires key/unit/source")
        if not isinstance(self.value,(int,float)) or not math.isfinite(self.value):
            raise ValueError("Physical fact must have a finite numeric value")


@dataclass(frozen=True)
class ReasoningClaim:
    hypothesis: str
    verdict: str
    evidence_refs: tuple[str,...]
    suggested_probe: str
    rationale: str

    @classmethod
    def parse(cls,obj:dict, available:dict[str,PhysicalFact]):
        if not isinstance(obj,dict) or set(obj)!={"hypothesis","verdict","evidence_refs","suggested_probe","rationale"}:
            raise ValueError("Unexpected reasoning schema")
        refs=obj["evidence_refs"]
        if not isinstance(refs,list) or not refs or len(refs)!=len(set(refs)):
            raise ValueError("Reasoning requires unique nonempty fact references")
        if any(not isinstance(ref,str) or ref not in available for ref in refs):
            raise ValueError("Reasoner cited unavailable or future evidence")
        if obj["verdict"] not in ("support","reject","uncertain"):
            raise ValueError("Invalid verdict")
        for key in ("hypothesis","suggested_probe","rationale"):
            if not isinstance(obj[key],str) or not obj[key].strip() or len(obj[key])>2048:
                raise ValueError(f"Invalid {key}")
        return cls(obj["hypothesis"],obj["verdict"],tuple(refs),
                   obj["suggested_probe"],obj["rationale"])


def validate_facts(facts:list[PhysicalFact]) -> dict[str,PhysicalFact]:
    if not facts: raise ValueError("At least one measured physical fact required")
    by_key={}
    for fact in facts:
        if fact.key in by_key: raise ValueError("Duplicate physical fact")
        by_key[fact.key]=fact
    return by_key


class ReasoningGate:
    """Model-assisted hypothesis notes, gated by measured evidence references."""
    def __init__(self, reasoner:Callable[[dict],dict]):
        if not callable(reasoner): raise TypeError("Reasoner must be callable")
        self.reasoner=reasoner

    def assess(self, *, task:str, facts:list[PhysicalFact],
               proposals: list[str]) -> dict:
        available=validate_facts(facts)
        if not isinstance(task,str) or not task.strip():
            raise ValueError("Nonempty task required")
        if not proposals or any(not isinstance(x,str) or not x.strip() for x in proposals):
            raise ValueError("Nonempty named teacher proposals required")
        if len(set(proposals))!=len(proposals): raise ValueError("Duplicate teacher name")
        context=dict(task=task,
           facts=[dict(key=f.key,value=f.value,unit=f.unit,source=f.source)
                  for f in available.values()],
           teachers=proposals,
           instruction="Reason about observed physics. Cite only provided fact keys. Return hypothesis, verdict, evidence_refs, suggested_probe, rationale. Do not generate actuator commands.")
        raw=self.reasoner(context)
        claim=ReasoningClaim.parse(raw,available)
        return dict(
            claim=dict(hypothesis=claim.hypothesis,verdict=claim.verdict,
                       evidence_refs=list(claim.evidence_refs),
                       suggested_probe=claim.suggested_probe,rationale=claim.rationale),
            grounded_facts={key:dict(value=available[key].value,
                 unit=available[key].unit,source=available[key].source)
                 for key in claim.evidence_refs},
            control_authorised=False,
            note="Model reasoning is a hypothesis, not a physical measurement or control permission.",
        )


class Cosmos3ReasonerClient:
    """Optional Cosmos3 Reasoner over an OpenAI-compatible chat completions API.

    Pass a real serving endpoint and a model name from the deployment. Provider
    compatibility, checkpoint capability and visual understanding must be
    validated in the user's own environment. No remote call happens by default.
    """
    def __init__(self,base_url:str,model:str,token_env="COSMOS_API_KEY",timeout=60):
        if not isinstance(base_url,str) or not base_url.startswith(("http://","https://")):
            raise ValueError("Expected explicit HTTP(S) Cosmos Reasoner endpoint")
        if not model or not isinstance(model,str): raise ValueError("Model identifier required")
        if not 0<timeout<=600: raise ValueError("Timeout outside [0,600]")
        self.base_url=base_url.rstrip("/")
        self.model=model
        self.token_env=token_env
        self.timeout=timeout

    def __call__(self,context:dict) -> dict:
        url=self.base_url if self.base_url.endswith("/chat/completions") else self.base_url+"/v1/chat/completions"
        payload=dict(
            model=self.model,temperature=0,
            messages=[
                dict(role="system",content=(
                    "You are a physics hypothesis assistant. Do not output actuator "
                    "commands. Reply ONLY with one JSON object containing exactly "
                    "hypothesis, verdict, evidence_refs, suggested_probe, rationale. "
                    "Cite only observed fact keys. If uncertain, say uncertain."
                )),
                dict(role="user",content=json.dumps(context,sort_keys=True)),
            ],
        )
        data=json.dumps(payload,allow_nan=False).encode("utf-8")
        headers={"Content-Type":"application/json"}
        key=os.environ.get(self.token_env)
        if key: headers["Authorization"]="Bearer "+key
        request=urllib.request.Request(url,data=data,headers=headers,method="POST")
        with urllib.request.urlopen(request,timeout=self.timeout) as response:
            reply=json.loads(response.read().decode("utf-8"))
        content=reply["choices"][0]["message"]["content"]
        if isinstance(content,list):
            content="".join(c.get("text","") for c in content if isinstance(c,dict))
        if not isinstance(content,str):
            raise ValueError("Reasoner returned non-text response")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Reasoner did not return strict JSON") from exc
