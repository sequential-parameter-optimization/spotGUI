import numpy as np
import matplotlib.pyplot as plt
import pylab
import torch

from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.utils.init import fun_control_init, design_control_init, surrogate_control_init
from spotPython.hyperparameters.values import set_control_key_value

from spotPython.spot import spot
from spotPython.utils.tensorboard import start_tensorboard

from spotPython.utils.device import getDevice
from spotPython.utils.eda import gen_design_table
from spotPython.fun.hyperlight import HyperLight
from spotPython.light.regression.netlightregression import NetLightRegression
from spotPython.light.regression.netlightregression2 import NetLightRegression2

from spotPython.light.regression.transformerlightregression import TransformerLightRegression
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.utils.file import save_experiment, load_experiment
from spotPython.data.lightdatamodule import LightDataModule
from spotPython.data.pkldataset import PKLDataset
from spotPython.data.diabetes import Diabetes


def run_spot_python_experiment(
    _L_in,
    _L_out,
    coremodel,
    MAX_TIME=1,
    INIT_SIZE=5,
    PREFIX="0000-spot",
    FUN_EVALS=10,
    FUN_REPEATS=1,
    n_total=None,
    perc_train=0.6,
    data_set="Diabetes",
    filename="PhishingData.csv",
    directory="./userData",
    n_samples=1_250,
    n_features=9,
    log_level=50,
    DATA_PKL_NAME="DATA.pickle",
    NOISE=False,
    OCBA_DELTA=0,
    REPEATS=2,
    WORKERS=0,
    DEVICE=getDevice(),
    DEVICES=1,
    TEST_SIZE=0.3,
    K_FOLDS=5,
) -> spot.Spot:
    """Runs a spot experiment."""

    fun_control = fun_control_init(
        _L_in=_L_in,
        _L_out=_L_out,
        PREFIX=PREFIX,
        TENSORBOARD_CLEAN=True,
        device=DEVICE,
        enable_progress_bar=False,
        fun_evals=FUN_EVALS,
        fun_repeats=FUN_REPEATS,
        log_level=50,
        max_time=MAX_TIME,
        num_workers=WORKERS,
        ocba_delta=OCBA_DELTA,
        show_progress=True,
        test_size=TEST_SIZE,
        tolerance_x=np.sqrt(np.spacing(1)),
        verbosity=1,
        noise=NOISE,
    )

    # Data set
    if data_set == "Diabetes":
        dataset = Diabetes(feature_type=torch.float32, target_type=torch.float32)
        set_control_key_value(control_dict=fun_control, key="data_set", value=dataset, replace=True)
        print(len(dataset))

    # TODO: Add more data sets, e.g. user specific data sets in ./userData:
    # dataset = PKLDataset(
    #     directory="./userData/",
    #     filename="data_sensitive.pkl",
    #     target_column=target,
    #     feature_type=torch.float32,
    #     target_type=torch.float32,
    #     rmNA=True,
    # )
    # set_control_key_value(control_dict=fun_control, key="data_set", value=dataset, replace=True)
    # print(len(dataset))

    # Core model
    if coremodel == "NetLightRegression2":
        add_core_model_to_fun_control(
            fun_control=fun_control, core_model=NetLightRegression2, hyper_dict=LightHyperDict
        )
        print(gen_design_table(fun_control))
    elif coremodel == "NetLightRegression":
        add_core_model_to_fun_control(fun_control=fun_control, core_model=NetLightRegression, hyper_dict=LightHyperDict)
        print(gen_design_table(fun_control))
    elif coremodel == "TransformerLightRegression":
        add_core_model_to_fun_control(
            fun_control=fun_control, core_model=TransformerLightRegression, hyper_dict=LightHyperDict
        )
        print(gen_design_table(fun_control))

    fun = HyperLight(log_level=50).fun

    design_control = design_control_init(
        init_size=INIT_SIZE,
        repeats=REPEATS,
    )

    # TODO: Pass more surrogate options
    surrogate_control = surrogate_control_init(
        noise=True,
        n_theta=2,
        min_Lambda=1e-6,
        max_Lambda=10,
        log_level=50,
    )

    spot_tuner = spot.Spot(
        fun=fun, fun_control=fun_control, design_control=design_control, surrogate_control=surrogate_control
    )
    spot_tuner.run()

    SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control)

    # tensorboard --logdir="runs/"

    # stop_tensorboard(p_open)
    return spot_tuner, fun_control


def parallel_plot(spot_tuner):
    fig = spot_tuner.parallel_plot()
    fig.show()


def contour_plot(spot_tuner):
    spot_tuner.plot_important_hyperparameter_contour(show=False)
    pylab.show()


def importance_plot(spot_tuner):
    plt.figure()
    spot_tuner.plot_importance(show=False)
    plt.show()


def progress_plot(spot_tuner):
    spot_tuner.plot_progress(show=False)
    plt.show()
