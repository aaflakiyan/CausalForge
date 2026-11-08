"""Heterogeneous teacher evidence and shared multi-embodiment student baseline.

Native teacher action streams are retained, not merged. Only physical action
chunks with identical ActionSpec within each embodiment are distilled. One
student artifact has a shared task encoder and separate embodiment heads.
This is an offline CPU ridge baseline, NOT a pretrained VLA model.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from .vla_interface import ActionChunk, ActionSpec

VERSION = 1

def spec_to_dict(spec):
    return dict(robot=spec.robot, representation=spec.representation,
                ordered_axes=list(spec.ordered_axes), units=list(spec.units),
                frame=spec.frame, dt=spec.dt)

def spec_from_dict(x):
    if set(x) != {"robot","representation","ordered_axes","units","frame","dt"}:
        raise ValueError("Invalid action spec keys")
    return ActionSpec(robot=x["robot"],representation=x["representation"],
                      ordered_axes=tuple(x["ordered_axes"]),units=tuple(x["units"]),
                      frame=x["frame"],dt=float(x["dt"]))

@dataclass(frozen=True)
class TeacherEvidence:
    episode_id:str
    split:str
    teacher:str
    architecture:str
    checkpoint:str
    task_features:tuple[float,...]
    action:ActionChunk
    measured_loss:float
    safe:bool
    observation_digest:str
    native_schema:str
    native_action:tuple[tuple[float,...],...]
    evidence_ref:str

    def __post_init__(self):
        for field in ("episode_id","teacher","architecture","checkpoint","observation_digest","native_schema","evidence_ref"):
            value=getattr(self,field)
            if not isinstance(value,str) or not value.strip():
                raise ValueError(f"{field} must be nonempty")
        if self.split not in ("train","test"):
            raise ValueError("Split must be train/test")
        if not isinstance(self.safe,bool) or not isinstance(self.action,ActionChunk):
            raise TypeError("Require measured safety bool and ActionChunk")
        features=np.asarray(self.task_features,dtype=float)
        raw=np.asarray(self.native_action,dtype=float)
        if (features.ndim!=1 or not features.size or not np.isfinite(features).all()
                or raw.ndim!=2 or not raw.size or not np.isfinite(raw).all()):
            raise ValueError("Invalid task features or raw native action")
        if not isinstance(self.measured_loss,(int,float)) or not math.isfinite(self.measured_loss) or self.measured_loss<0:
            raise ValueError("Measured loss must be nonnegative finite")
        if len(self.observation_digest)!=64 or any(c not in "0123456789abcdef" for c in self.observation_digest):
            raise ValueError("Require SHA256 observation digest")

    def to_dict(self):
        return dict(schema_version=VERSION,episode_id=self.episode_id,split=self.split,
          teacher=self.teacher,architecture=self.architecture,checkpoint=self.checkpoint,
          task_features=list(self.task_features),
          action=dict(spec=spec_to_dict(self.action.spec),
                      actions=self.action.actions.tolist(),model_name=self.action.model_name),
          measured_loss=float(self.measured_loss),safe=self.safe,
          observation_digest=self.observation_digest,native_schema=self.native_schema,
          native_action=[list(x) for x in self.native_action],evidence_ref=self.evidence_ref)

    @classmethod
    def from_dict(cls, x):
        if x.get("schema_version")!=VERSION:
            raise ValueError("Unsupported evidence schema")
        a=x["action"]
        return cls(episode_id=x["episode_id"],split=x["split"],teacher=x["teacher"],
          architecture=x["architecture"],checkpoint=x["checkpoint"],
          task_features=tuple(x["task_features"]),
          action=ActionChunk(np.asarray(a["actions"],dtype=float),
                             spec_from_dict(a["spec"]),a["model_name"]),
          measured_loss=float(x["measured_loss"]),safe=x["safe"],
          observation_digest=x["observation_digest"],
          native_schema=x["native_schema"],
          native_action=tuple(tuple(y) for y in x["native_action"]),
          evidence_ref=x["evidence_ref"])

def save_evidence(path, records):
    if not records:
        raise ValueError("Cannot save empty evidence")
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text("".join(json.dumps(r.to_dict(),sort_keys=True,allow_nan=False)+"\n"
                         for r in records),encoding="utf-8")

def load_evidence(path):
    rows=[]
    for n,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if line.strip():
            try: rows.append(TeacherEvidence.from_dict(json.loads(line)))
            except (KeyError,ValueError,TypeError) as exc:
                raise ValueError(f"Evidence line {n}: {exc}") from exc
    if not rows: raise ValueError("Empty evidence")
    return rows

def validate_groups(records):
    if not records: raise ValueError("No evidence")
    dim=len(records[0].task_features)
    groups={}
    splits={}
    specs={}
    horizons={}
    identities=set()
    for r in records:
        if len(r.task_features)!=dim:
            raise ValueError("Semantic task features incompatible")
        if r.episode_id in splits and splits[r.episode_id]!=r.split:
            raise ValueError("Train/test leakage")
        splits[r.episode_id]=r.split
        robot=r.action.spec.robot
        if robot in specs and specs[robot]!=r.action.spec:
            raise ValueError("Physical action space differs within embodiment")
        specs[robot]=r.action.spec
        horizon=r.action.actions.shape[0]
        if robot in horizons and horizons[robot]!=horizon:
            raise ValueError("Action horizon differs: no silent interpolation")
        horizons[robot]=horizon
        identity=(r.episode_id,robot,r.teacher)
        if identity in identities: raise ValueError("Duplicate teacher")
        identities.add(identity)
        groups.setdefault((r.episode_id,robot),[]).append(r)
    for group in groups.values():
        a=group[0]
        if any(r.split!=a.split or r.task_features!=a.task_features
               or r.observation_digest!=a.observation_digest for r in group):
            raise ValueError("Inconsistent observation or split among teachers")
    return groups

def verified_teacher(group):
    safe=[r for r in group if r.safe]
    if not safe: raise ValueError("No measured-safe teacher")
    return min(safe,key=lambda r:(r.measured_loss,r.teacher))

class UnifiedStudent:
    """One serialised multihead model with a shared PCA task-feature encoder."""
    def __init__(self,latent_dim=8,ridge=.01):
        if latent_dim<1 or not math.isfinite(ridge) or ridge<=0: raise ValueError("Invalid model hyperparameters")
        self.latent_dim=latent_dim
        self.ridge=float(ridge)
        self.mean=self.scale=self.projection=None
        self.heads={}

    def _encode(self,x):
        arr=np.asarray(x,dtype=float)
        if arr.ndim!=2 or arr.shape[1]!=len(self.mean) or not np.isfinite(arr).all():
            raise ValueError("Invalid semantic feature vector")
        return ((arr-self.mean)/self.scale)@self.projection

    def fit(self,records):
        grouped=validate_groups(records)
        train=[verified_teacher(g) for g in grouped.values() if g[0].split=="train"]
        test=[g for g in grouped.values() if g[0].split=="test"]
        if not train or not test: raise ValueError("Train and test episodes required")
        features=np.asarray([r.task_features for r in train])
        self.mean=features.mean(axis=0)
        self.scale=np.maximum(features.std(axis=0),1e-6)
        _,_,v=np.linalg.svd((features-self.mean)/self.scale,full_matrices=False)
        self.projection=v[:min(self.latent_dim,*v.shape)].T.copy()
        encoded=self._encode(features)
        for robot in sorted({r.action.spec.robot for r in train}):
            indexes=[i for i,r in enumerate(train) if r.action.spec.robot==robot]
            if len(indexes)<2: raise ValueError(f"Too few examples for {robot}")
            target=np.stack([train[i].action.actions.ravel() for i in indexes])
            design=np.column_stack([np.ones(len(indexes)),encoded[indexes]])
            penalty=self.ridge*np.eye(design.shape[1]); penalty[0,0]=0
            weights=np.linalg.solve(design.T@design+penalty+1e-10*np.eye(design.shape[1]),design.T@target)
            self.heads[robot]=dict(spec=train[indexes[0]].action.spec,
                  horizon=train[indexes[0]].action.actions.shape[0],weights=weights,
                  training_examples=len(indexes))
        return self

    def predict(self,features,*,robot):
        if self.mean is None: raise RuntimeError("Untrained student")
        if robot not in self.heads: raise ValueError("Unknown robot embodiment")
        a=np.asarray(features,dtype=float)
        if a.ndim!=1: raise ValueError("Expected 1D task features")
        z=self._encode(a.reshape(1,-1))
        h=self.heads[robot]
        y=np.column_stack([np.ones(1),z])@h["weights"]
        return ActionChunk(y.reshape(h["horizon"],-1),h["spec"],"unified_student")

    def evaluate(self,records):
        groups=validate_groups(records)
        scores={}
        for (episode,robot),g in groups.items():
            if g[0].split!="test" or robot not in self.heads: continue
            try: ref=verified_teacher(g)
            except ValueError: continue
            pred=self.predict(ref.task_features,robot=robot)
            scores.setdefault(robot,[]).append(float(np.mean((pred.actions-ref.action.actions)**2)))
        if not scores: raise ValueError("No held-out evaluation samples")
        return dict(metric="offline_action_mse_to_verified_teacher",
          per_embodiment={k:dict(examples=len(v),mean_mse=float(np.mean(v))) for k,v in scores.items()},
          warning="Test outcome used ONLY to define offline reference. Not a live policy or proof of skill retention.")

    def to_dict(self):
        if self.mean is None: raise RuntimeError("Untrained")
        return dict(schema_version=VERSION,model="multihead_ridge",latent_dim=self.latent_dim,
          ridge=self.ridge,mean=self.mean.tolist(),scale=self.scale.tolist(),
          projection=self.projection.tolist(),
          heads={k:dict(spec=spec_to_dict(h["spec"]),horizon=h["horizon"],
                  weights=h["weights"].tolist(),training_examples=h["training_examples"])
                 for k,h in self.heads.items()})

    @classmethod
    def from_dict(cls,obj):
        if obj.get("schema_version")!=VERSION or obj.get("model")!="multihead_ridge":
            raise ValueError("Unsupported student format")
        result=cls(latent_dim=int(obj["latent_dim"]),ridge=float(obj["ridge"]))
        result.mean=np.asarray(obj["mean"],dtype=float)
        result.scale=np.asarray(obj["scale"],dtype=float)
        result.projection=np.asarray(obj["projection"],dtype=float)
        if (result.projection.ndim!=2 or result.projection.shape[0]!=len(result.mean)
           or len(result.scale)!=len(result.mean) or np.any(result.scale<=0)
           or not np.isfinite(result.projection).all()):
            raise ValueError("Corrupt shared encoder")
        for robot,h in obj["heads"].items():
            spec=spec_from_dict(h["spec"])
            weights=np.asarray(h["weights"],dtype=float)
            horizon=int(h["horizon"])
            if (robot!=spec.robot or horizon<1 or
               weights.shape!=(result.projection.shape[1]+1,horizon*len(spec.ordered_axes))
               or not np.isfinite(weights).all()):
                raise ValueError("Corrupt embodiment head")
            result.heads[robot]=dict(spec=spec,weights=weights,horizon=horizon,
                                     training_examples=h["training_examples"])
        return result

def evidence_digest(rows):
    text="\n".join(json.dumps(r.to_dict(),sort_keys=True,allow_nan=False) for r in rows)+"\n"
    return hashlib.sha256(text.encode()).hexdigest()
