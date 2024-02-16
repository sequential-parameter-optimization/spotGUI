import matplotlib.pyplot as plt
import pylab
import pprint
from spotPython.spot import spot
from spotPython.utils.tensorboard import start_tensorboard, stop_tensorboard
from spotPython.utils.eda import gen_design_table
from spotPython.fun.hyperlight import HyperLight
from spotPython.utils.file import save_experiment
from spotPython.utils.file import load_experiment

from spotRiver.evaluation.eval_bml import eval_oml_horizon
from spotRiver.evaluation.eval_bml import plot_bml_oml_horizon_metrics
from spotPython.plot.validation import plot_roc_from_dataframes
from spotPython.plot.validation import plot_confusion_matrix
from spotPython.hyperparameters.values import get_one_core_model_from_X
from spotPython.hyperparameters.values import get_default_hyperparameters_as_array
from spotGUI.eda.pairplot import generate_pairplot


def run_spot_python_experiment(
    save_only,
    fun_control,
    design_control,
    surrogate_control,
    optimizer_control,
    fun=HyperLight(log_level=50).fun,
    show_data_only=False,
) -> spot.Spot:
    """Runs a spot experiment.

    Args:
        save_only (bool): If True, the experiment will be saved and the spot run will not be executed.
        show_data_only (bool): If True, the data will be shown and the spot run will not be executed.
        fun_control (dict): A dictionary with the function control parameters.
        design_control (dict): A dictionary with the design control parameters.
        surrogate_control (dict): A dictionary with the surrogate control parameters.
        optimizer_control (dict): A dictionary with the optimizer control parameters.
        fun (function): The function to be optimized.

    Returns:
        spot.Spot: The spot experiment.

    """
    SPOT_PKL_NAME = None

    print("\nfun_control in spotRun():")
    pprint.pprint(fun_control)
    print(gen_design_table(fun_control))

    spot_tuner = spot.Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
        surrogate_control=surrogate_control,
        optimizer_control=optimizer_control,
    )

    if show_data_only:
        train = fun_control["train"]
        train_size = len(train)
        test = fun_control["test"]
        test_size = len(test)
        target_column = fun_control["target_column"]
        print(f"\nTrain data summary:\n {train.describe(include='all')}")
        print(f"\nTest data summary:\n {test.describe(include='all')}")
        # if the data set has more than 1000 entries,
        # select 1000 random samples to display
        train_sample = test_sample = False
        if train_size > 1000:
            train = train.sample(n=1000)
            train_sample = True
        if test_size > 1000:
            test = test.sample(n=1000)
            test_sample = True

        # generate a histogram of the target column
        plt.figure()
        # Create the first subplot for the training data
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, index 1
        train[target_column].hist()
        plt.title("Train Data")
        # Create the second subplot for the test data
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, index 2
        test[target_column].hist()
        plt.title("Test Data")

        generate_pairplot(
            data=train, target_column=target_column, title="Train Data", sample=train_sample, size=train_size
        )
        generate_pairplot(data=test, target_column=target_column, title="Test Data", sample=test_sample, size=test_size)
        plt.show()
        return SPOT_PKL_NAME, spot_tuner, fun_control, design_control, surrogate_control, optimizer_control

    if save_only:
        if "spot_writer" in fun_control and fun_control["spot_writer"] is not None:
            fun_control["spot_writer"].close()
        SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control, design_control, surrogate_control, optimizer_control)
        return SPOT_PKL_NAME, spot_tuner, fun_control, design_control, surrogate_control, optimizer_control
    else:
        p_open = start_tensorboard()
        # TODO: Implement X_Start handling
        # X_start = get_default_hyperparameters_as_array(fun_control)
        spot_tuner.run()
        SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control, design_control, surrogate_control, optimizer_control)
        # tensorboard --logdir="runs/"
        stop_tensorboard(p_open)
        return SPOT_PKL_NAME, spot_tuner, fun_control, design_control, surrogate_control, optimizer_control


def load_and_run_spot_python_experiment(spot_pkl_name) -> spot.Spot:
    """Loads and runs a spot experiment.

    Args:
        spot_pkl_name (str): The name of the spot experiment file.

    Returns:
        spot.Spot: The spot experiment.

    """
    (spot_tuner, fun_control, design_control, surrogate_control, optimizer_control) = load_experiment(spot_pkl_name)
    print("\nLoaded fun_control in spotRun():")
    pprint.pprint(fun_control)
    print(gen_design_table(fun_control))
    p_open = start_tensorboard()
    spot_tuner.run()
    SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control, design_control, surrogate_control, optimizer_control)
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


def compare_tuned_default(spot_tuner, fun_control) -> None:
    X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1, -1))
    print(f"X = {X}")
    model_spot = get_one_core_model_from_X(X, fun_control)
    df_eval_spot, df_true_spot = eval_oml_horizon(
        model=model_spot,
        train=fun_control["train"],
        test=fun_control["test"],
        target_column=fun_control["target_column"],
        horizon=fun_control["horizon"],
        oml_grace_period=fun_control["oml_grace_period"],
        metric=fun_control["metric_sklearn"],
    )
    X_start = get_default_hyperparameters_as_array(fun_control)
    model_default = get_one_core_model_from_X(X_start, fun_control)
    df_eval_default, df_true_default = eval_oml_horizon(
        model=model_default,
        train=fun_control["train"],
        test=fun_control["test"],
        target_column=fun_control["target_column"],
        horizon=fun_control["horizon"],
        oml_grace_period=fun_control["oml_grace_period"],
        metric=fun_control["metric_sklearn"],
    )

    df_labels = ["default", "spot"]
    # Create a figure with 1 row and 2 columns of subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # First Plot
    plot_confusion_matrix(
        df=df_true_default,
        title="Default",
        y_true_name=fun_control["target_column"],
        y_pred_name="Prediction",
        show=False,
        ax=axs[0],
    )
    # plt.figure(1)
    # Second Plot
    plot_confusion_matrix(
        df=df_true_spot,
        title="Spot",
        y_true_name=fun_control["target_column"],
        y_pred_name="Prediction",
        show=False,
        ax=axs[1],
    )
    # plt.figure(2)
    # Third Plot
    plot_roc_from_dataframes(
        [df_true_default, df_true_spot],
        model_names=["default", "spot"],
        target_column=fun_control["target_column"],
        show=False,
    )
    plt.figure(1)
    # Fourth Plot
    plot_bml_oml_horizon_metrics(
        df_eval=[df_eval_default, df_eval_spot],
        log_y=False,
        df_labels=df_labels,
        metric=fun_control["metric_sklearn"],
        filename=None,
        show=False,
    )
    plt.figure(2)
    plt.show()
