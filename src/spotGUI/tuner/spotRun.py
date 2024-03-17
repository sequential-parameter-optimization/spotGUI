import matplotlib.pyplot as plt
import sklearn.metrics
import pylab
import pprint
import numpy as np
import os
from math import inf
from tkinter import filedialog as fd
from spotPython.spot import spot
from spotPython.utils.tensorboard import start_tensorboard, stop_tensorboard
from spotPython.utils.eda import gen_design_table
from spotPython.fun.hyperlight import HyperLight
from spotPython.utils.file import save_experiment
from spotPython.utils.file import load_experiment
from spotPython.utils.metrics import get_metric_sign

import river
from spotRiver.evaluation.eval_bml import eval_oml_horizon
from spotRiver.evaluation.eval_bml import plot_bml_oml_horizon_metrics
from spotPython.plot.validation import plot_roc_from_dataframes
from spotPython.plot.validation import plot_confusion_matrix
from spotPython.hyperparameters.values import get_one_core_model_from_X
from spotPython.hyperparameters.values import get_default_hyperparameters_as_array
from spotGUI.eda.pairplot import generate_pairplot


def get_report_file_name(fun_control):
    """Returns the name of the report file.

    Args:
        fun_control (dict):

    Returns:
        str: The name of the report file.
    """
    PREFIX = fun_control["PREFIX"]
    REP_NAME = "spot_" + PREFIX + "_report.txt"
    return REP_NAME


def get_result(spot_tuner, fun_control):
    """Gets the result of the spot experiment.

    Args:
        spot_tuner (spot.Spot): The spot experiment.
        fun_control (dict): A dictionary with the function control parameters.

    Returns:
        str: result of the spot experiment from gen_design_table().

    """
    if spot_tuner is not None and fun_control is not None:
        return gen_design_table(fun_control=fun_control, spot=spot_tuner)


def run_spot_python_experiment(
    save_only,
    fun_control,
    design_control,
    surrogate_control,
    optimizer_control,
    fun=HyperLight(log_level=50).fun,
    show_data_only=False,
    tensorboard_start=True,
    tensorboard_stop=True,
    tuner_report=True,
) -> spot.Spot:
    """Runs a spot experiment.

    Args:
        save_only (bool):
            If True, the experiment will be saved and the spot run will not be executed.
        show_data_only (bool):
            If True, the data will be shown and the spot run will not be executed.
        fun_control (dict):
            A dictionary with the function control parameters.
        design_control (dict):
            A dictionary with the design control parameters.
        surrogate_control (dict):
            A dictionary with the surrogate control parameters.
        optimizer_control (dict):
            A dictionary with the optimizer control parameters.
        fun (function):
            The function to be optimized.
        tensorboard_start (bool):
            If True, the tensorboard process will be started before the spot run.
            Default is True.
        tensorboard_stop (bool):
            If True, the tensorboard process will be stopped after the spot run.
            Default is True.

    Returns:
        spot.Spot: The spot experiment.

    """
    SPOT_PKL_NAME = None
    p_open = None

    print("\nfun_control in spotRun():")
    pprint.pprint(fun_control)

    if tuner_report:
        REP_NAME = get_report_file_name(fun_control)
        # write the formatted fun_control to a file
        with open(REP_NAME, "w") as file:
            file.write(gen_design_table(fun_control))
            file.write("\n\n")
            file.write(pprint.pformat(fun_control))
            file.write("\n\n")
            file.write(pprint.pformat(surrogate_control))
            file.write("\n\n")
            file.write(pprint.pformat(design_control))
            file.write("\n\n")
            file.write(pprint.pformat(optimizer_control))
            file.write("\n\n")
        print("Control dicts written to SPOT report txt file.")
        # close the file
        file.close()

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
        return SPOT_PKL_NAME, spot_tuner, fun_control, design_control, surrogate_control, optimizer_control, p_open

    if save_only:
        if "spot_writer" in fun_control and fun_control["spot_writer"] is not None:
            fun_control["spot_writer"].close()
        SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control, design_control, surrogate_control, optimizer_control)
        return SPOT_PKL_NAME, spot_tuner, fun_control, design_control, surrogate_control, optimizer_control, p_open
    else:
        if tensorboard_start:
            p_open = start_tensorboard()
        # TODO: Implement X_Start handling
        # X_start = get_default_hyperparameters_as_array(fun_control)
        spot_tuner.run()
        SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control, design_control, surrogate_control, optimizer_control)
        # tensorboard --logdir="runs/"
        if tuner_report:
            REP_NAME = get_report_file_name(fun_control)
            # write the formatted fun_control to a file
            with open(REP_NAME, "a") as file:
                file.write(gen_design_table(fun_control=fun_control, spot=spot_tuner))
        # close the file
        file.close()
        print(gen_design_table(fun_control=fun_control, spot=spot_tuner))
        if tensorboard_stop:
            stop_tensorboard(p_open)
            p_open = None
        return SPOT_PKL_NAME, spot_tuner, fun_control, design_control, surrogate_control, optimizer_control, p_open


def load_and_run_spot_python_experiment(spot_pkl_name) -> spot.Spot:
    """Loads and runs a spot experiment.

    Args:
        spot_pkl_name (str): The name of the spot experiment file.

    Returns:
        spot.Spot: The spot experiment.

    """
    p_open = None
    (spot_tuner, fun_control, design_control, surrogate_control, optimizer_control) = load_experiment(spot_pkl_name)
    print("\nLoaded fun_control in spotRun():")
    pprint.pprint(fun_control)
    print(gen_design_table(fun_control))
    p_open = start_tensorboard()
    spot_tuner.run()
    SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control, design_control, surrogate_control, optimizer_control)
    # tensorboard --logdir="runs/"
    stop_tensorboard(p_open)
    return SPOT_PKL_NAME, spot_tuner, fun_control, design_control, surrogate_control, optimizer_control, p_open


def parallel_plot(spot_tuner, fun_control):
    fig = spot_tuner.parallel_plot()
    plt.title(fun_control["PREFIX"])
    fig.show()


def contour_plot(spot_tuner, fun_control):
    spot_tuner.plot_important_hyperparameter_contour(show=False, max_imp=3, threshold=0, title=fun_control["PREFIX"])
    pylab.show()
    # plt.show()


def importance_plot(spot_tuner, fun_control):
    plt.figure()
    spot_tuner.plot_importance(show=False)
    plt.title(fun_control["PREFIX"])
    plt.show()


def progress_plot(spot_tuner, fun_control):
    spot_tuner.plot_progress(show=False)
    plt.title(fun_control["PREFIX"])
    plt.show()


def plot_confusion_matrices(spot_tuner, fun_control, show=False) -> None:
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
    # Second Plot
    plot_confusion_matrix(
        df=df_true_spot,
        title="Spot",
        y_true_name=fun_control["target_column"],
        y_pred_name="Prediction",
        show=False,
        ax=axs[1],
    )
    # add a title to the figure
    fig.suptitle(fun_control["PREFIX"])
    if show:
        plt.show()


def plot_rocs(spot_tuner, fun_control, show=False) -> None:
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
    plot_roc_from_dataframes(
        [df_true_default, df_true_spot],
        model_names=["default", "spot"],
        target_column=fun_control["target_column"],
        show=show,
        title=fun_control["PREFIX"],
    )


def compare_tuned_default(spot_tuner, fun_control, show=False) -> None:
    print(vars(spot_tuner))
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
    plot_bml_oml_horizon_metrics(
        df_eval=[df_eval_default, df_eval_spot],
        log_y=False,
        df_labels=df_labels,
        metric=fun_control["metric_sklearn"],
        filename=None,
        show=show,
        title=fun_control["PREFIX"],
    )


def all_compare_tuned_default(spot_tuner, fun_control) -> None:
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


def destroy_entries(entries):
    """
    Destroys all non-None entries in the provided list of entries.

    Args:
        entries: A list of entries to be destroyed.

    Returns:
        None
    """
    if entries is not None:
        for entry in entries:
            if entry is not None:
                entry.destroy()


def load_file_dialog():
    """
    Opens a file dialog to select a file.

    Args:
        None

    Returns:
        str: The name of the selected file.

    """
    current_dir = os.getcwd()
    filetypes = (("Pickle files", "*.pickle"), ("All files", "*.*"))
    filename = fd.askopenfilename(title="Select a Pickle File", initialdir=current_dir, filetypes=filetypes)
    return filename


def get_core_model_from_name(core_model_name):
    """
    Returns the core model name and instance from a core model name.

    Args:
        core_model_name (str): The name of the core model.

    Returns:
        Tuple: The core model name and instance.
    """
    core_model_module = core_model_name.split(".")[0]
    coremodel = core_model_name.split(".")[1]
    core_model_instance = getattr(getattr(river, core_model_module), coremodel)
    return coremodel, core_model_instance


def get_n_total(n_total):
    """
    Returns the number of total iterations.

    Args:
        n_total (str): The number of total iterations.

    Returns:
        int: The number of total iterations.
    """
    if n_total == "None" or n_total == "All":
        n_total = None
    else:
        n_total = int(n_total)
    return n_total


def get_fun_evals(fun_evals):
    """
    Returns the number of function evaluations.

    Args:
        fun_evals (str): The number of function evaluations.

    Returns:
        int: The number of function evaluations.
    """
    if fun_evals == "None" or fun_evals == "inf":
        fun_evals_val = inf
    else:
        fun_evals_val = int(fun_evals)
    return fun_evals_val


def get_lambda_min_max(lambda_min_max):
    """
    Returns the minimum and maximum lambda values.

    Args:
        lambda_min_max (str): The minimum and maximum lambda values.

    Returns:
        Tuple: The minimum and maximum lambda values.
    """
    lbd = lambda_min_max.split(",")
    # if len(lbd) != 2, set the lambda values to the default values
    if len(lbd) != 2:
        lbd = ["1e-6", "1e2"]
    lbd_min = float(lbd[0])
    lbd_max = float(lbd[1])
    if lbd_min < 0:
        lbd_min = 1e-6
    if lbd_max < 0:
        lbd_max = 1e2
    return lbd_min, lbd_max


def get_oml_grace_period(oml_grace_period):
    """
    Returns the grace period for the online machine learning evaluation.

    Args:
        oml_grace_period (str): The grace period for the online machine learning evaluation.

    Returns:
        int: The grace period for the online machine learning evaluation.
    """
    if oml_grace_period == "None":
        oml_grace_period = None
    else:
        oml_grace_period = int(oml_grace_period)
    return oml_grace_period


def get_prep_model(prepmodel_name):
    """
    Get the preprocessing model from the name.

    Args:
        prepmodel_name (str): The name of the preprocessing model.

    Returns:
        river.preprocessing: The preprocessing model.

    """
    if prepmodel_name == "None":
        prepmodel = None
    else:
        prepmodel = getattr(river.preprocessing, prepmodel_name)
    return prepmodel


def get_metric_sklearn(metric_name):
    """
    Returns the metric from the metric name.

    Args:
        metric_name (str): The name of the metric.

    Returns:
        sklearn.metrics: The metric from the metric name.
    """
    metric_sklearn = getattr(sklearn.metrics, metric_name)
    return metric_sklearn


def get_weights(metric_name, metric_weights, default_weights=["1000,1,1"]):
    """
    Returns the weights for the metric.

    Args:
        metric_name (str): The name of the metric.
        metric_weights (str): The weights for the metric.
        default_weights (list): The default weights for the metric.

    Returns:
        np.array: The weights for the metric.
    """
    weight_sgn = get_metric_sign(metric_name)
    mw = metric_weights.split(",")
    if len(mw) != 3:
        mw = default_weights
    weights = np.array([weight_sgn * float(mw[0]), float(mw[1]), float(mw[2])])
    return weights


def get_kriging_noise(lbd_min, lbd_max):
    """Get the kriging noise based on the lambda values.
    If the lambda values are both 0, the kriging noise is False.
    Otherwise, the kriging noise is True.

    Args:
        lbd_min (float): The minimum lambda value.
        lbd_max (float): The maximum lambda value.

    Returns:
        bool: The kriging noise.

    """
    if lbd_min == 0.0 and lbd_max == 0.0:
        return False
    else:
        return True
