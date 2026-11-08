"""Reasoning gate: cite measured facts, never grant control authority."""
import json
import unittest
from unittest.mock import patch
from vla_council.causalforge.physical_reasoning import (
    PhysicalFact, ReasoningGate, Cosmos3ReasonerClient
)

FACTS=[
    PhysicalFact("observed_force",12.5,"N","simulator_contact_sensor"),
    PhysicalFact("slip_distance",.025,"m","measured_state"),
]
GOOD={
    "hypothesis":"low contact friction",
    "verdict":"uncertain",
    "evidence_refs":["observed_force","slip_distance"],
    "suggested_probe":"perform a lower speed grasp test",
    "rationale":"measured slip occurred; cannot infer friction alone",
}

class ReasoningTests(unittest.TestCase):
    def test_grounded_claim(self):
        out=ReasoningGate(lambda context:GOOD).assess(
            task="diagnose slip",facts=FACTS,proposals=["pi","groot"])
        self.assertFalse(out["control_authorised"])
        self.assertEqual(out["claim"]["verdict"],"uncertain")
        self.assertIn("slip_distance",out["grounded_facts"])

    def test_unobserved_future_reference_rejected(self):
        bad={**GOOD,"evidence_refs":["future_contact"]}
        with self.assertRaisesRegex(ValueError,"unavailable"):
            ReasoningGate(lambda x:bad).assess(task="slip",facts=FACTS,proposals=["pi"])

    def test_empty_and_duplicate_fact_rejected(self):
        gate=ReasoningGate(lambda x:GOOD)
        with self.assertRaises(ValueError):
            gate.assess(task="slip",facts=[],proposals=["pi"])
        with self.assertRaises(ValueError):
            gate.assess(task="slip",facts=FACTS+[FACTS[0]],proposals=["pi"])

    def test_cosmos_api_shape_with_fake_transport_only(self):
        client=Cosmos3ReasonerClient("http://localhost:8000","cosmos3-reasoner")
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def read(self): return json.dumps({
                "choices":[{"message":{"content":json.dumps(GOOD)}}]
            }).encode()
        def fake_open(request,timeout):
            payload=json.loads(request.data)
            self.assertEqual(payload["model"],"cosmos3-reasoner")
            self.assertEqual(payload["messages"][0]["role"],"system")
            self.assertTrue(request.full_url.endswith("/v1/chat/completions"))
            return FakeResponse()
        with patch("urllib.request.urlopen",side_effect=fake_open):
            out=ReasoningGate(client).assess(task="slip",facts=FACTS,proposals=["pi"])
        self.assertEqual(out["claim"]["hypothesis"],GOOD["hypothesis"])

    def test_refuse_unstructured_reasoner_response(self):
        client=Cosmos3ReasonerClient("http://localhost:8000","test")
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def read(self): return b'{"choices":[{"message":{"content":"not JSON"}}]}'
        with patch("urllib.request.urlopen",return_value=FakeResponse()):
            with self.assertRaisesRegex(ValueError,"strict JSON"):
                client({"task":"slip"})

if __name__=="__main__":
    unittest.main()
