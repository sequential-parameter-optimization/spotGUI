import pytest
from spotGUI.tuner.spotRun import run_spot_python_experiment
from spotPython.utils.init import fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init
from spotPython.light.regression.netlightregression import NetLightRegression
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.spot import spot

def test_run_spot_python_experiment_save_only():
    # Define the input parameters for the function
    save_only = True
    fun_control = fun_control_init(PREFIX="test")
    design_control = design_control_init()
    surrogate_control = surrogate_control_init()
    optimizer_control = optimizer_control_init()

    add_core_model_to_fun_control(fun_control=fun_control,
                                  core_model=NetLightRegression,
                                  hyper_dict=LightHyperDict)

    # Call the function
    result = run_spot_python_experiment(
        save_only,
        fun_control,
        design_control,
        surrogate_control,
        optimizer_control,
    )
    # Assert the expected output
    assert len(result) == 7
    assert result[0] is not None 
    assert isinstance(result[1], spot.Spot)  # spot_tuner should be an instance of spot.Spot
