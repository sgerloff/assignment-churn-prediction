import pytest

import churn_prediction_model

def test_semantic_version_format():
    version = churn_prediction_model.__version__
    version_numbers = version.split(".")
    pytest.assume(len(version_numbers) == 3)
    pytest.assume(all([version_number.isdigit() for version_number in version_numbers]))
