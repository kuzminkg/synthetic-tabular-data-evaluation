from synthrep.models import make_estimator


def test_logreg_default_c():
    model = make_estimator("logreg")
    assert model.C == 1.0


def test_logreg_custom_c():
    model = make_estimator("logreg", logreg_c=0.5)
    assert model.C == 0.5
