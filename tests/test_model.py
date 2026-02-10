from pathlib import Path

from app import Use_model

def test_predict_face():
    repo_root = Path(__file__).resolve().parents[1]
    img_path = repo_root / "drake-continue-de-marquer-l-histoire_643ff073894c9.jpeg"
    pred = Use_model.predict_face(str(img_path))
    assert isinstance(pred, str)
    assert pred == "Drake"
