
from app import Use_model

def test_predict_face():
    tmp_path = "drake-continue-de-marquer-l-histoire_643ff073894c9.jpeg"
    pred = Use_model.predict_face(tmp_path)
    assert isinstance(pred, str)
    assert pred == "Drake"
