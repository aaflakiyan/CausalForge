"""Semantic observation mappers must not erase missing modalities or units."""
import hashlib
import unittest
from vla_council.causalforge.semantic_mapper import (
    Feature,ScalarObservation,ModalityRef,SemanticMapper
)

SCHEMA=(Feature("goal_x","m","world",True),
        Feature("normal_force","N","tool",False),
        Feature("object_mass","kg","world",False))

class SemanticMapperTests(unittest.TestCase):
    def test_missing_optional_features_are_masked_not_assumed_zero(self):
        mapper=SemanticMapper(SCHEMA)
        image=ModalityRef("image","s3://private-dataset/episode/image.png",
                          hashlib.sha256(b"image bytes").hexdigest(),"rgb8/640x480")
        mapped=mapper.encode({"goal_x":ScalarObservation(.24,"m","world",1.,"sim")},
                             modalities=(image,))
        self.assertEqual(mapped["task_features"],(.24,0.,0.,1.,0.,0.))
        self.assertEqual(mapped["availability"],(True,False,False))
        self.assertEqual(mapped["native_modalities"],(image,))
    def test_real_zero_is_distinct_from_missing(self):
        m=SemanticMapper(SCHEMA)
        present=m.encode({"goal_x":ScalarObservation(0.,"m","world",1.,"sim"),
                          "normal_force":ScalarObservation(0.,"N","tool",1.,"ft")})
        self.assertEqual(present["task_features"][:3],(0.,0.,0.))
        self.assertEqual(present["availability"],(True,True,False))
    def test_wrong_units_or_frame_refused(self):
        m=SemanticMapper(SCHEMA)
        for u,f in (("cm","world"),("m","base")):
            with self.subTest(u=u,f=f),self.assertRaises(ValueError):
                m.encode({"goal_x":ScalarObservation(10.,u,f,1.,"sim")})
    def test_stale_mixed_sensor_readings_refused(self):
        m=SemanticMapper(SCHEMA,max_skew_s=.02)
        with self.assertRaisesRegex(ValueError,"timestamp"):
            m.encode({"goal_x":ScalarObservation(.2,"m","world",1.,"sim"),
                      "normal_force":ScalarObservation(5.,"N","tool",1.3,"ft")})
    def test_unknown_data_retained_without_guessing(self):
        m=SemanticMapper(SCHEMA)
        other=ScalarObservation(3.,"C","camera",1.,"thermal")
        out=m.encode({"goal_x":ScalarObservation(.2,"m","world",1.,"sim"),
                      "unknown_temperature":other})
        self.assertEqual(out["unmapped_observations"]["unknown_temperature"],other)
    def test_required_feature_and_invalid_modality_rejected(self):
        m=SemanticMapper(SCHEMA)
        with self.assertRaisesRegex(ValueError,"Missing required"):
            m.encode({})
        with self.assertRaises(ValueError):
            ModalityRef("video","x","invalid","x")

if __name__=="__main__":
    unittest.main()
