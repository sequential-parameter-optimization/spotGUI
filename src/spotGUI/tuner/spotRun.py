import matplotlib.pyplot as plt
import pprint
import pylab
from spotPython.spot import spot
from spotPython.utils.tensorboard import start_tensorboard
from spotPython.utils.device import getDevice
from spotPython.utils.eda import gen_design_table
from spotPython.fun.hyperlight import HyperLight
from spotPython.utils.file import save_experiment, load_experiment
from spotPython.utils.init import fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init


def run_spot_python_experiment(
    save_only,
    fun_control,
    design_control,
    surrogate_control,
    optimizer_control,
) -> spot.Spot:
    """Runs a spot experiment."""

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
    # if coremodel == "NetLightRegression2":
    #     add_core_model_to_fun_control(
    #         fun_control=fun_control, core_model=NetLightRegression2, hyper_dict=LightHyperDict
    #     )
    #     print(gen_design_table(fun_control))
    # elif coremodel == "NetLightRegression":
    #     add_core_model_to_fun_control(fun_control=fun_control, core_model=NetLightRegression,
    #        hyper_dict=LightHyperDict)
    #     print(gen_design_table(fun_control))
    # elif coremodel == "TransformerLightRegression":
    #     add_core_model_to_fun_control(
    #         fun_control=fun_control, core_model=TransformerLightRegression, hyper_dict=LightHyperDict
    #     )
    print("\nfun_control in spotRun():", fun_control)

    pprint.pprint(fun_control)

    print(gen_design_table(fun_control))

    fun = HyperLight(log_level=10).fun

    spot_tuner = spot.Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
        surrogate_control=surrogate_control,
        optimizer_control=optimizer_control,
    )
    # save_experiment(spot_tuner, fun_control, design_control, surrogate_control, optimizer_control)

    if save_only:
        return spot_tuner, fun_control, design_control, surrogate_control, optimizer_control
    else:
        spot_tuner.run()

        SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control)

        # tensorboard --logdir="runs/"

        # stop_tensorboard(p_open)
        return spot_tuner, fun_control, design_control, surrogate_control, optimizer_control


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
