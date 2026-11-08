"""Semantic observations that preserve missing modalities across embodiments.

This is a *declared* shared input vocabulary, not a magic image/action encoder.
Missing optional features produce an explicit availability bit; a missing
measurement is never silently interpreted as a true zero. Unrecognised native
signals are retained as provenance, not guessed into physical quantities.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Feature:
    name: str
    unit: str
    frame: str
    required: bool = False

    def __post_init__(self):
        if not self.name or not self.unit or not self.frame:
            raise ValueError("Name, unit and reference frame required")
        if not isinstance(self.required,bool):
            raise TypeError("required must be bool")


@dataclass(frozen=True)
class ScalarObservation:
    value: float
    unit: str
    frame: str
    time_s: float
    source: str

    def __post_init__(self):
        if (not math.isfinite(self.value) or not math.isfinite(self.time_s)
            or self.time_s<0 or not self.unit or not self.frame or not self.source):
            raise ValueError("Invalid measurement, timestamp or provenance")


@dataclass(frozen=True)
class ModalityRef:
    """Checksum-addressed native image/video/text/tactile asset, not converted."""
    kind: str
    uri: str
    sha256: str
    schema: str

    def __post_init__(self):
        if self.kind not in ("image","video","text","depth","point_cloud","force","other"):
            raise ValueError("Unknown native modality")
        if not self.uri or not self.schema or len(self.sha256)!=64 or any(
            x not in "0123456789abcdef" for x in self.sha256):
            raise ValueError("Native modality must have URI, schema and SHA256")


class SemanticMapper:
    """Map measured named scalars to [values..., availability mask...] features."""
    def __init__(self,fields:tuple[Feature,...],*,max_skew_s=.05):
        if not fields or len({f.name for f in fields})!=len(fields):
            raise ValueError("Require nonempty unique semantic fields")
        if max_skew_s<0 or not math.isfinite(max_skew_s):
            raise ValueError("Invalid timestamp skew bound")
        self.fields=tuple(fields)
        self.max_skew_s=max_skew_s

    def encode(self,measurements:dict[str,ScalarObservation],
               *,modalities:tuple[ModalityRef,...]=()):
        if not isinstance(measurements,dict):
            raise TypeError("Expected named scalar measurements")
        known={f.name for f in self.fields}
        vals,mask,times=[],[],[]
        for feature in self.fields:
            item=measurements.get(feature.name)
            if item is None:
                if feature.required:
                    raise ValueError(f"Missing required semantic feature: {feature.name}")
                vals.append(0.)
                mask.append(0.)
                continue
            if not isinstance(item,ScalarObservation):
                raise TypeError("Expected typed ScalarObservation")
            if (item.unit,item.frame)!=(feature.unit,feature.frame):
                raise ValueError(f"Unverified units or frame for {feature.name}")
            vals.append(item.value)
            mask.append(1.)
            times.append(item.time_s)
        if times and max(times)-min(times)>self.max_skew_s+1e-12:
            raise ValueError("Asynchronous observations exceed timestamp skew bound")
        if any(not isinstance(m,ModalityRef) for m in modalities):
            raise TypeError("Native modality manifest must be typed")
        # Unknown data are retained for a model-specific encoder or later mapping.
        raw_unmapped={name:measurements[name] for name in measurements if name not in known}
        return {
            "task_features":tuple(vals+mask),
            "feature_order":tuple(f.name for f in self.fields),
            "availability":tuple(bool(m) for m in mask),
            "unmapped_observations":raw_unmapped,
            "native_modalities":modalities,
            "note":"Missing scalar values are distinguished by the availability mask. Native modalities remain separately referenced.",
        }
