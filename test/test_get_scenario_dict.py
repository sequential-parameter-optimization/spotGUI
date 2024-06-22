import pytest
from unittest.mock import patch
import copy

from spotGUI.tuner.spotRun import get_scenario_dict

@pytest.fixture
def scenario_entries_mock():
    return {"entry1": "value1", "entry2": "value2"}

@pytest.fixture
def prep_models_mock():
    return ["prep_model1", "prep_model2"]

# Test when scenario is "river"
@patch("spotGUI.tuner.spotRun.get_river_prep_models")
@patch("spotGUI.tuner.spotRun.get_river_regression_datasets")
@patch("spotGUI.tuner.spotRun.get_river_rules_core_model_names")
@patch("spotGUI.tuner.spotRun.get_regression_metric_sklearn_levels")
@patch("spotGUI.tuner.spotRun.get_river_regression_core_model_names")
@patch("spotGUI.tuner.spotRun.get_river_binary_classification_datasets")
@patch("spotGUI.tuner.spotRun.get_classification_metric_sklearn_levels")
@patch("spotGUI.tuner.spotRun.get_river_classification_core_model_names")
@patch("spotGUI.tuner.spotRun.get_scenario_entries")
def test_get_scenario_dict_river(get_scenario_entries_mock, get_river_classification_core_model_names_mock,
                                  get_classification_metric_sklearn_levels_mock, get_river_binary_classification_datasets_mock,
                                  get_river_regression_core_model_names_mock, get_regression_metric_sklearn_levels_mock,
                                  get_river_rules_core_model_names_mock, get_river_regression_datasets_mock,
                                  get_river_prep_models_mock, scenario_entries_mock, prep_models_mock):
    # Setup mock return values
    get_scenario_entries_mock.return_value = scenario_entries_mock
    get_river_classification_core_model_names_mock.return_value = ["model1", "model2"]
    get_classification_metric_sklearn_levels_mock.return_value = ["level1", "level2"]
    get_river_binary_classification_datasets_mock.return_value = ["dataset1", "dataset2"]
    get_river_regression_core_model_names_mock.return_value = ["model3", "model4"]
    get_regression_metric_sklearn_levels_mock.return_value = ["level3", "level4"]
    get_river_rules_core_model_names_mock.return_value = ["model5", "model6"]
    get_river_regression_datasets_mock.return_value = ["dataset3", "dataset4"]
    get_river_prep_models_mock.return_value = prep_models_mock

    expected = {
        "classification_task": {
            "core_model_names": ["model1", "model2"],
            "metric_sklearn_levels": ["level1", "level2"],
            "datasets": ["dataset1", "dataset2"],
            "prep_models": copy.deepcopy(prep_models_mock),
            **scenario_entries_mock
        },
        "regression_task": {
            "core_model_names": ["model3", "model4"],
            "metric_sklearn_levels": ["level3", "level4"],
            "datasets": ["dataset3", "dataset4"],
            "prep_models": copy.deepcopy(prep_models_mock),
            **scenario_entries_mock
        },
        "rules_task": {
            "core_model_names": ["model5", "model6"],
            "metric_sklearn_levels": ["level3", "level4"],  # Assuming this is not a mistake in the original function
            "datasets": ["dataset3", "dataset4"],
            "prep_models": copy.deepcopy(prep_models_mock),
            **scenario_entries_mock
        }
    }

    result = get_scenario_dict("river")
    assert result == expected


# Test when scenario is not "river"
def test_get_scenario_dict_not_river():
    assert get_scenario_dict("not_river") is None