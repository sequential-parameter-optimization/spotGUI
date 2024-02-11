import matplotlib.pyplot as plt
import pylab
from spotPython.spot import spot
from spotPython.utils.tensorboard import start_tensorboard, stop_tensorboard
from spotPython.utils.eda import gen_design_table
from spotPython.fun.hyperlight import HyperLight
from spotPython.utils.file import save_experiment

# TODO: Implement a function to load the experiment
# from spotPython.utils.file import load_experiment


def run_spot_python_experiment(
    save_only,
    fun_control,
    design_control,
    surrogate_control,
    optimizer_control,
    fun=HyperLight(log_level=50).fun,
) -> spot.Spot:
    """Runs a spot experiment."""

    print("\nfun_control in spotRun():", fun_control)

    print(gen_design_table(fun_control))

    spot_tuner = spot.Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
        surrogate_control=surrogate_control,
        optimizer_control=optimizer_control,
    )

    # TODO: Fix error when saving the experiment w/o spot run execution
    # SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control, design_control, surrogate_control, optimizer_control)
    SPOT_PKL_NAME = None
    if save_only:
        return SPOT_PKL_NAME, spot_tuner, fun_control, design_control, surrogate_control, optimizer_control
    else:
        p_open = start_tensorboard()

        spot_tuner.run()

        SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control)

        # tensorboard --logdir="runs/"

        stop_tensorboard(p_open)
        return SPOT_PKL_NAME, spot_tuner, fun_control, design_control, surrogate_control, optimizer_control


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
