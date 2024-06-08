import sys
import pprint
import webbrowser
import river.preprocessing
from spotRiver.data.river_hyper_dict import RiverHyperDict
import river
from river import forest, tree, linear_model
from river import preprocessing
import tkinter as tk
import os
import copy
import numpy as np
from tkinter import ttk, StringVar
from idlelib.tooltip import Hovertip
from spotPython.utils.init import fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init
from spotPython.hyperparameters.values import add_core_model_to_fun_control, set_control_hyperparameter_value
from spotGUI.tuner.spotRun import (
    run_spot_python_experiment,
    contour_plot,
    parallel_plot,
    importance_plot,
    progress_plot,
    actual_vs_prediction,
    compare_tuned_default,
    all_compare_tuned_default,
    plot_confusion_matrices,
    plot_rocs,
    destroy_entries,
    load_file_dialog,
    get_report_file_name,
    get_result,
    get_n_total,
    get_fun_evals,
    get_lambda_min_max,
    get_oml_grace_period,
    get_weights,
    get_kriging_noise,
)
from spotPython.utils.eda import gen_design_table
from spotPython.utils.convert import map_to_True_False
from spotPython.utils.file import load_dict_from_file
from spotRiver.fun.hyperriver import HyperRiver
from spotRiver.data.selector import get_river_dataset_from_name
from spotRiver.utils.data_conversion import split_df
from spotPython.utils.file import load_experiment as load_experiment_spot
from spotPython.hyperparameters.values import (get_prep_model,
                                               get_metric_sklearn, get_core_model_from_name)

classification_core_model_names = [
    "linear_model.LogisticRegression",
    "forest.AMFClassifier",
    "forest.ARFClassifier",
    "tree.ExtremelyFastDecisionTreeClassifier",
    "tree.HoeffdingTreeClassifier",
    "tree.HoeffdingAdaptiveTreeClassifier",
    "tree.SGTClassifier",
]
classification_metric_levels = [
    "accuracy_score",
    "cohen_kappa_score",
    "f1_score",
    "hamming_loss",
    "hinge_loss",
    "jaccard_score",
    "matthews_corrcoef",
    "precision_score",
    "recall_score",
    "roc_auc_score",
    "zero_one_loss",
]
river_binary_classification_datasets = ["Phishing", "Bananas", "CreditCard", "Elec2", "Higgs", "HTTP"]

regression_core_model_names = [
    "linear_model.LinearRegression",
    "tree.HoeffdingTreeRegressor",
    "forest.AMFRegressor",
    "forest.ARFRegressor",
    "tree.HoeffdingAdaptiveTreeRegressor",
    "tree.SGTRegressor",
]

regression_metric_levels = [
    "mean_absolute_error",
    "explained_variance_score",
    "max_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_squared_log_error",
    "root_mean_squared_log_error",
    "median_absolute_error",
    "r2_score",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
    "mean_absolute_percentage_error",
    "d2_absolute_error_score",
    "d2_pinball_score",
    "d2_tweedie_score",
]

river_regression_datasets = ["ChickWeights", "Bikes", "Taxis", "TrumpApproval"]

task_entries = dict(
    core_model_names=[],
    metric_levels=[],
    datasets=[],
    core_model_combo=None,
    data_set_combo=None,
    n_total_entry=None,
    target_type_entry=None,
    test_size_entry=None,
    prep_model_combo=None,
    shuffle=None,
    max_sp_entry=None,
    max_time_entry=None,
    fun_evals_entry=None,
    init_size_entry=None,
    noise_entry=None,
    lambda_min_max_entry=None,
    seed_entry=None,
    metric_combo=None,
    metric_weights_entry=None,
    horizon_entry=None,
    oml_grace_period_entry=None,
    prefix_entry=None,
    tb_clean=None,
    tb_start=None,
    tb_stop=None,
)

prep_models = ["AdaptiveStandardScaler", "MaxAbsScaler", "MinMaxScaler", "StandardScaler", "None"]

task_dict = {"classification_tab": copy.deepcopy(task_entries), "regression_tab": copy.deepcopy(task_entries)}

task_dict["classification_tab"]["core_model_names"] = classification_core_model_names
task_dict["classification_tab"]["metric_levels"] = classification_metric_levels
task_dict["classification_tab"]["datasets"] = river_binary_classification_datasets
task_dict["regression_tab"]["core_model_names"] = regression_core_model_names
task_dict["regression_tab"]["metric_levels"] = regression_metric_levels
task_dict["regression_tab"]["datasets"] = river_regression_datasets
task_dict["classification_tab"]["prep_models"] = copy.deepcopy(prep_models)
task_dict["regression_tab"]["prep_models"] = copy.deepcopy(prep_models)


spot_tuner = None
rhd = RiverHyperDict()
# Max number of keys (hyperparameters):
n_keys = 25

label_dict = {
    "label": [None] * n_keys,
    "default_entry": [None] * n_keys,
    "lower_bound_entry": [None] * n_keys,
    "upper_bound_entry": [None] * n_keys,
    "factor_level_entry": [None] * n_keys,
    "transform_entry": [None] * n_keys,
}
hyper_dict = {"classification_tab": dict(copy.deepcopy(label_dict)), "regression_tab": dict(copy.deepcopy(label_dict))}


def call_compare_tuned_default():
    if spot_tuner is not None and fun_control is not None:
        compare_tuned_default(spot_tuner, fun_control, show=True)


def call_actual_vs_prediction():
    if spot_tuner is not None and fun_control is not None:
        actual_vs_prediction(spot_tuner, fun_control, show=True)


def call_all_compare_tuned_default():
    if spot_tuner is not None and fun_control is not None:
        all_compare_tuned_default(spot_tuner, fun_control, show=True)


def call_plot_confusion_matrices():
    if spot_tuner is not None and fun_control is not None:
        plot_confusion_matrices(spot_tuner, fun_control, show=True)


def call_plot_rocs():
    if spot_tuner is not None and fun_control is not None:
        plot_rocs(spot_tuner, fun_control, show=True)


def call_parallel_plot():
    if spot_tuner is not None:
        parallel_plot(spot_tuner=spot_tuner, fun_control=fun_control)


def call_contour_plot():
    if spot_tuner is not None:
        contour_plot(spot_tuner=spot_tuner, fun_control=fun_control)


def call_importance_plot():
    if spot_tuner is not None:
        importance_plot(spot_tuner=spot_tuner, fun_control=fun_control)


def call_progress_plot():
    if spot_tuner is not None:
        progress_plot(spot_tuner=spot_tuner, fun_control=fun_control)


def show_result():
    if spot_tuner is not None and fun_control is not None:
        res = get_result(spot_tuner=spot_tuner, fun_control=fun_control)
        print(f"Result: {res}")
        # add a text window to the result tab that shows the content of the REP_NAME file
        result_text = tk.Text(result_tab)
        result_text.grid(row=1, column=1, rowspan=1, columnspan=10, sticky="W")
        # open the result file and write its content to the text window
        # clean the text window
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, res)
        # resize the text window to fit the content
        result_text.config(width=200, height=25)


# def show_report():
#     if fun_control is not None:
#         REP_NAME = get_report_file_name(fun_control=fun_control)
#         # add a text window to the report tab that shows the content of the REP_NAME file
#         report_text = tk.Text(report_tab)
#         report_text.grid(row=1, column=1, rowspan=1, columnspan=10, sticky="W")
#         # open the report file and write its content to the text window
#         with open(REP_NAME, "r") as file:
#             # clean the text window
#             report_text.delete("1.0", tk.END)
#             report_text.insert(tk.END, file.read())
#             # resize the text window to fit the content
#             report_text.config(width=250, height=50)
#         # close the file
#         file.close()


def run_experiment(tab_task, save_only=False, show_data_only=False):
    global spot_tuner, fun_control, hyper_dict, task_dict

    print(f"tab_task in run_experiment(): {tab_task.name}")
    noise = map_to_True_False(task_dict[tab_task.name]["noise_entry"].get())
    n_total = get_n_total(task_dict[tab_task.name]["n_total_entry"].get())
    fun_evals_val = get_fun_evals(task_dict[tab_task.name]["fun_evals_entry"].get())
    max_surrogate_points = int(task_dict[tab_task.name]["max_sp_entry"].get())
    seed = int(task_dict[tab_task.name]["seed_entry"].get())
    test_size = float(task_dict[tab_task.name]["test_size_entry"].get())
    target_type = task_dict[tab_task.name]["target_type_entry"].get()
    core_model_name = task_dict[tab_task.name]["core_model_combo"].get()
    print(f"core_model_name: {core_model_name}")
    lbd_min, lbd_max = get_lambda_min_max(task_dict[tab_task.name]["lambda_min_max_entry"].get())
    prep_model_name = task_dict[tab_task.name]["prep_model_combo"].get()
    print(f"prep_model_name: {prep_model_name}")
    if prep_model_name.endswith(".py"):
        print(f"prep_model_name = {prep_model_name}")
        sys.path.insert(0, "./userPrepModel")
        # remove the file extension from the prep_model_name
        prep_model_name = prep_model_name[:-3]
        print(f"prep_model_name = {prep_model_name}")
        __import__(prep_model_name)
        prepmodel = sys.modules[prep_model_name].set_prep_model()
    else:
        prepmodel = get_prep_model(prep_model_name)
    metric_sklearn = get_metric_sklearn(task_dict[tab_task.name]["metric_combo"].get())
    weights = get_weights(
        task_dict[tab_task.name]["metric_combo"].get(), task_dict[tab_task.name]["metric_weights_entry"].get()
    )
    oml_grace_period = get_oml_grace_period(task_dict[tab_task.name]["oml_grace_period_entry"].get())
    data_set_name = task_dict[tab_task.name]["data_set_combo"].get()
    print(f"data_set_name: {data_set_name}")
    print(f"task_dict[tab_task.name]['datasets']: {task_dict[tab_task.name]['datasets']}")
    dataset, n_samples = get_river_dataset_from_name(
        data_set_name=data_set_name, n_total=n_total, river_datasets=task_dict[tab_task.name]["datasets"]
    )
    shuffle = bool(task_dict[tab_task.name]["shuffle"].get())
    train, test, n_samples = split_df(
        dataset=dataset, test_size=test_size, target_type=target_type, seed=seed, shuffle=shuffle, stratify=None
    )

    TENSORBOARD_CLEAN = bool(task_dict[tab_task.name]["tb_clean"].get())
    tensorboard_start = bool(task_dict[tab_task.name]["tb_start"].get())
    tensorboard_stop = bool(task_dict[tab_task.name]["tb_stop"].get())

    # Initialize the fun_control dictionary with the static parameters,
    # i.e., the parameters that are not hyperparameters (depending on the core model)
    fun_control = fun_control_init(
        PREFIX=task_dict[tab_task.name]["prefix_entry"].get(),
        TENSORBOARD_CLEAN=TENSORBOARD_CLEAN,
        core_model_name=core_model_name,
        data_set_name=data_set_name,
        fun_evals=fun_evals_val,
        fun_repeats=1,
        horizon=int(task_dict[tab_task.name]["horizon_entry"].get()),
        max_surrogate_points=max_surrogate_points,
        max_time=float(task_dict[tab_task.name]["max_time_entry"].get()),
        metric_sklearn=metric_sklearn,
        noise=noise,
        n_samples=n_samples,
        ocba_delta=0,
        oml_grace_period=oml_grace_period,
        prep_model=prepmodel,
        seed=seed,
        target_column="y",
        target_type=target_type,
        test=test,
        test_size=test_size,
        train=train,
        tolerance_x=np.sqrt(np.spacing(1)),
        verbosity=1,
        weights=weights,
        log_level=50,
    )

    # TODO:
    # Check the handling of l1/l2 in LogisticRegression. A note (from the River documentation):
    # > For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
    # Therefore, we set l1 bounds to 0.0:
    # modify_hyper_parameter_bounds(fun_control, "l1", bounds=[0.0, 0.0])
    # set_control_hyperparameter_value(fun_control, "l1", [0.0, 0.0])
    # modify_hyper_parameter_levels(fun_control, "optimizer", ["SGD"])

    # TODO:
    #  Enable user specific core models. An example is given below:
    # from spotPython.hyperparameters.values import add_core_model_to_fun_control
    # import sys
    # sys.path.insert(0, './userModel')
    # import river.tree
    # import river_hyper_dict
    # add_core_model_to_fun_control(fun_control=fun_control,
    #                             core_model=river.tree.HoeffdingTreeRegressor,
    #                             hyper_dict=river_hyper_dict.RiverHyperDict)

    coremodel, core_model_instance = get_core_model_from_name(core_model_name)
    add_core_model_to_fun_control(
        core_model=core_model_instance,
        fun_control=fun_control,
        hyper_dict=RiverHyperDict,
        filename=None,
    )
    dict = rhd.hyper_dict[coremodel]
    for i, (key, value) in enumerate(dict.items()):
        if dict[key]["type"] == "int":
            set_control_hyperparameter_value(
                fun_control,
                key,
                [
                    int(hyper_dict[tab_task.name]["lower_bound_entry"][i].get()),
                    int(hyper_dict[tab_task.name]["upper_bound_entry"][i].get()),
                ],
            )
        if (dict[key]["type"] == "factor") and (dict[key]["core_model_parameter_type"] == "bool"):
            set_control_hyperparameter_value(
                fun_control,
                key,
                [
                    int(hyper_dict[tab_task.name]["lower_bound_entry"][i].get()),
                    int(hyper_dict[tab_task.name]["upper_bound_entry"][i].get()),
                ],
            )
        if dict[key]["type"] == "float":
            set_control_hyperparameter_value(
                fun_control,
                key,
                [
                    float(hyper_dict[tab_task.name]["lower_bound_entry"][i].get()),
                    float(hyper_dict[tab_task.name]["upper_bound_entry"][i].get()),
                ],
            )
        if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
            fle = hyper_dict[tab_task.name]["factor_level_entry"][i].get()
            # convert the string to a list of strings
            fle = fle.split()
            set_control_hyperparameter_value(fun_control, key, fle)
            fun_control["core_model_hyper_dict"][key].update({"upper": len(fle) - 1})
    design_control = design_control_init(
        init_size=int(task_dict[tab_task.name]["init_size_entry"].get()),
        repeats=1,
    )
    surrogate_control = surrogate_control_init(
        # If lambda is set to 0, no noise will be used in the surrogate
        # Otherwise use noise in the surrogate:
        noise=get_kriging_noise(lbd_min, lbd_max),
        n_theta=2,
        min_Lambda=lbd_min,
        max_Lambda=lbd_max,
        log_level=50,
    )
    print("surrogate_control in run_experiment():")
    pprint.pprint(surrogate_control)
    optimizer_control = optimizer_control_init()
    print(gen_design_table(fun_control))
    (
        SPOT_PKL_NAME,
        spot_tuner,
        fun_control,
        design_control,
        surrogate_control,
        optimizer_control,
        p_open,
    ) = run_spot_python_experiment(
        save_only=save_only,
        show_data_only=show_data_only,
        fun_control=fun_control,
        design_control=design_control,
        surrogate_control=surrogate_control,
        optimizer_control=optimizer_control,
        fun=HyperRiver(log_level=fun_control["log_level"]).fun_oml_horizon,
        tensorboard_start=tensorboard_start,
        tensorboard_stop=tensorboard_stop,
    )
    if SPOT_PKL_NAME is not None and save_only:
        print(f"\nExperiment successfully saved. Configuration saved as: {SPOT_PKL_NAME}")
    elif SPOT_PKL_NAME is not None and not save_only:
        print(f"\nExperiment successfully terminated. Result saved as: {SPOT_PKL_NAME}")
    elif show_data_only:
        print("\nData shown. No result saved.")
    else:
        print("\nExperiment failed. No result saved.")


def load_experiment(tab_task):
    global spot_tuner, fun_control, task_dict
    filename = load_file_dialog()
    if filename:
        spot_tuner, fun_control, design_control, surrogate_control, optimizer_control = load_experiment_spot(filename)
        print("\nfun_control in load_experiment():")
        pprint.pprint(fun_control)

        task_dict[tab_task.name]["data_set_combo"].delete(0, tk.END)
        data_set_name = fun_control["data_set_name"]

        if data_set_name == "CSVDataset" or data_set_name == "PKLDataset":
            filename = vars(fun_control["data_set"])["filename"]
            print("filename: ", filename)
            task_dict[tab_task.name]["data_set_combo"].set(filename)
        else:
            task_dict[tab_task.name]["data_set_combo"].set(data_set_name)

        # static parameters, that are not hyperparameters (depending on the core model)

        task_dict[tab_task.name]["n_total_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["n_total_entry"].insert(0, str(fun_control["n_total"]))

        task_dict[tab_task.name]["test_size_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["test_size_entry"].insert(0, str(fun_control["test_size"]))

        task_dict[tab_task.name]["target_type_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["target_type_entry"].insert(0, str(fun_control["target_type"]))

        task_dict[tab_task.name]["prep_model_combo"].delete(0, tk.END)
        # prep_model_name = fun_control["prep_model"].__class__.__name__
        prep_model_name = fun_control["prep_model"].__name__
        task_dict[tab_task.name]["prep_model_combo"].set(prep_model_name)

        task_dict[tab_task.name]["max_time_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["max_time_entry"].insert(0, str(fun_control["max_time"]))

        task_dict[tab_task.name]["fun_evals_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["fun_evals_entry"].insert(0, str(fun_control["fun_evals"]))

        task_dict[tab_task.name]["init_size_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["init_size_entry"].insert(0, str(design_control["init_size"]))

        task_dict[tab_task.name]["prefix_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["prefix_entry"].insert(0, str(fun_control["PREFIX"]))

        task_dict[tab_task.name]["noise_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["noise_entry"].insert(0, str(fun_control["noise"]))

        task_dict[tab_task.name]["max_sp_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["max_sp_entry"].insert(0, str(fun_control["max_surrogate_points"]))

        task_dict[tab_task.name]["seed_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["seed_entry"].insert(0, str(fun_control["seed"]))

        task_dict[tab_task.name]["metric_combo"].delete(0, tk.END)
        metric_name = fun_control["metric_sklearn"].__name__
        task_dict[tab_task.name]["metric_combo"].set(metric_name)

        task_dict[tab_task.name]["metric_weights_entry"].delete(0, tk.END)
        wghts = fun_control["weights"]
        # take the absolute value of all weights
        wghts = [abs(w) for w in wghts]
        task_dict[tab_task.name]["metric_weights_entry"].insert(0, f"{wghts[0]}, {wghts[1]}, {wghts[2]}")

        task_dict[tab_task.name]["lambda_min_max_entry"].delete(0, tk.END)
        min_lbd = surrogate_control["min_Lambda"]
        max_lbd = surrogate_control["max_Lambda"]
        task_dict[tab_task.name]["lambda_min_max_entry"].insert(0, f"{min_lbd}, {max_lbd}")

        task_dict[tab_task.name]["horizon_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["horizon_entry"].insert(0, str(fun_control["horizon"]))

        task_dict[tab_task.name]["oml_grace_period_entry"].delete(0, tk.END)
        task_dict[tab_task.name]["oml_grace_period_entry"].insert(0, str(fun_control["oml_grace_period"]))

        destroy_entries(hyper_dict[tab_task.name]["label"])
        destroy_entries(hyper_dict[tab_task.name]["default_entry"])
        destroy_entries(hyper_dict[tab_task.name]["lower_bound_entry"])
        destroy_entries(hyper_dict[tab_task.name]["upper_bound_entry"])
        destroy_entries(hyper_dict[tab_task.name]["transform_entry"])

        if hyper_dict[tab_task.name]["factor_level_entry"] is not None:
            for i in range(len(hyper_dict[tab_task.name]["factor_level_entry"])):
                if hyper_dict[tab_task.name]["factor_level_entry"][i] is not None and not isinstance(
                    hyper_dict[tab_task.name]["factor_level_entry"][i], StringVar
                ):
                    hyper_dict[tab_task.name]["factor_level_entry"][i].destroy()

        update_entries_from_dict(fun_control["core_model_hyper_dict"], tab_task=tab_task, hyper_dict=hyper_dict)

        task_dict[tab_task.name]["core_model_combo"].delete(0, tk.END)
        task_dict[tab_task.name]["core_model_combo"].set(fun_control["core_model_name"])


def update_entries_from_dict(dict, tab_task, hyper_dict):
    # global hyper_dict
    # global label, default_entry, lower_bound_entry, upper_bound_entry, transform_entry, factor_level_entry
    # n_keys = len(dict)
    # # Create a list of labels and entries with the same length as the number of keys in the dictionary
    # label = [None] * n_keys
    # default_entry = [None] * n_keys
    # lower_bound_entry = [None] * n_keys
    # upper_bound_entry = [None] * n_keys
    # factor_level_entry = [None] * n_keys
    # transform_entry = [None] * n_keys
    print(f"tab_task in update_entries_from_dict(): {tab_task.name}\n")
    print(f"dict in update_entries_from_dict(): {dict}\n")
    print("hyper_dict in update_entries_from_dict():")

    for i, (key, value) in enumerate(dict.items()):
        if (
            dict[key]["type"] == "int"
            or dict[key]["type"] == "float"
            or dict[key]["core_model_parameter_type"] == "bool"
        ):
            hyper_dict[tab_task.name]["label"][i] = tk.Label(tab_task, text=key)
            hyper_dict[tab_task.name]["label"][i].grid(row=i + 3, column=2, sticky="W")
            hyper_dict[tab_task.name]["label"][i].update()

            hyper_dict[tab_task.name]["default_entry"][i] = tk.Label(tab_task, text=dict[key]["default"])
            hyper_dict[tab_task.name]["default_entry"][i].grid(row=i + 3, column=3, sticky="W")
            hyper_dict[tab_task.name]["default_entry"][i].update()

            hyper_dict[tab_task.name]["lower_bound_entry"][i] = tk.Entry(tab_task)
            hyper_dict[tab_task.name]["lower_bound_entry"][i].insert(0, dict[key]["lower"])
            hyper_dict[tab_task.name]["lower_bound_entry"][i].grid(row=i + 3, column=4, sticky="W")
            hyper_dict[tab_task.name]["lower_bound_entry"][i].update()

            hyper_dict[tab_task.name]["upper_bound_entry"][i] = tk.Entry(tab_task)
            hyper_dict[tab_task.name]["upper_bound_entry"][i].insert(0, dict[key]["upper"])
            hyper_dict[tab_task.name]["upper_bound_entry"][i].grid(row=i + 3, column=5, sticky="W")
            hyper_dict[tab_task.name]["upper_bound_entry"][i].update()

            hyper_dict[tab_task.name]["transform_entry"][i] = tk.Label(tab_task, text=dict[key]["transform"], padx=15)
            hyper_dict[tab_task.name]["transform_entry"][i].grid(row=i + 3, column=6, sticky="W")
            hyper_dict[tab_task.name]["transform_entry"][i].update()

        if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
            hyper_dict[tab_task.name]["label"][i] = tk.Label(tab_task, text=key)
            hyper_dict[tab_task.name]["label"][i].grid(row=i + 3, column=2, sticky="W")
            hyper_dict[tab_task.name]["label"][i].update()

            hyper_dict[tab_task.name]["default_entry"][i] = tk.Label(tab_task, text=dict[key]["default"])
            hyper_dict[tab_task.name]["default_entry"][i].grid(row=i + 3, column=3, sticky="W")
            hyper_dict[tab_task.name]["default_entry"][i].update()

            hyper_dict[tab_task.name]["factor_level_entry"][i] = tk.Entry(tab_task)

            hyper_dict[tab_task.name]["factor_level_entry"][i].insert(0, dict[key]["levels"])
            hyper_dict[tab_task.name]["factor_level_entry"][i].grid(
                row=i + 3, column=4, columnspan=2, sticky=tk.W + tk.E
            )
            hyper_dict[tab_task.name]["factor_level_entry"][i].update()


def create_first_column(tab_task):
    global hyper_dict, task_dict

    print(f"tab_task in create_first_column(): {tab_task.name}")

    # colummns 0+1: Data
    core_model_label = tk.Label(tab_task, text="Core model:")
    core_model_label.grid(row=1, column=0, sticky="W")

    core_model_label = tk.Label(tab_task, text="Select core model:")
    core_model_label.grid(row=2, column=0, sticky="W")
    for filename in os.listdir("userModel"):
        if filename.endswith(".json"):
            task_dict[tab_task.name]["core_model_names"].append(os.path.splitext(filename)[0])
    task_dict[tab_task.name]["core_model_combo"] = ttk.Combobox(
        tab_task, values=task_dict[tab_task.name]["core_model_names"]
    )
    # Default selection, the first core model in the list:
    task_dict[tab_task.name]["core_model_combo"].set(task_dict[tab_task.name]["core_model_names"][0])
    task_dict[tab_task.name]["core_model_combo"].bind(
        "<<ComboboxSelected>>", lambda e: update_hyperparams(event=None, tab_task=tab_task)
    )
    task_dict[tab_task.name]["core_model_combo"].grid(row=2, column=1)
    update_hyperparams(event=None, tab_task=tab_task)

    data_label = tk.Label(tab_task, text="Data options:")
    data_label.grid(row=3, column=0, sticky="W")

    data_set_label = tk.Label(tab_task, text="Select data_set:")
    data_set_label.grid(row=4, column=0, sticky="W")
    message = (
        "The data set.\n"
        "User specified data sets must have the target value in the last column.\n"
        "They are assumed to be in the directory 'userData'.\n"
        "The data set must be a CSV file."
    )
    data_set_tip = Hovertip(
        data_set_label,
        message,
    )
    data_set_values = task_dict[tab_task.name]["datasets"]
    # get all *.csv files in the data directory "userData" and append them to the list of data_set_values
    data_set_values.extend([f for f in os.listdir("userData") if f.endswith(".csv") or f.endswith(".pkl")])
    task_dict[tab_task.name]["data_set_combo"] = ttk.Combobox(tab_task, values=data_set_values)
    # Default selection, the first data set in the list:
    task_dict[tab_task.name]["data_set_combo"].set(data_set_values[0])
    task_dict[tab_task.name]["data_set_combo"].grid(row=4, column=1)

    n_total_label = tk.Label(tab_task, text="n_total (int|All):")
    n_total_label.grid(row=5, column=0, sticky="W")
    task_dict[tab_task.name]["n_total_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["n_total_entry"].insert(0, "All")
    task_dict[tab_task.name]["n_total_entry"].grid(row=5, column=1, sticky="W")

    test_size_label = tk.Label(tab_task, text="test_size (perc.):")
    test_size_label.grid(row=6, column=0, sticky="W")
    task_dict[tab_task.name]["test_size_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["test_size_entry"].insert(0, "0.30")
    task_dict[tab_task.name]["test_size_entry"].grid(row=6, column=1, sticky="W")

    target_type_label = tk.Label(tab_task, text="target_type (int|float):")
    target_type_label.grid(row=7, column=0, sticky="W")
    task_dict[tab_task.name]["target_type_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["target_type_entry"].insert(0, "int")
    task_dict[tab_task.name]["target_type_entry"].grid(row=7, column=1, sticky="W")

    prep_model_label = tk.Label(tab_task, text="Select preprocessing model")
    prep_model_label.grid(row=8, column=0, sticky="W")
    prep_model_values = task_dict[tab_task.name]["prep_models"]
    prep_model_values.extend([f for f in os.listdir("userPrepModel") if f.endswith(".py") and not f.startswith("__")])
    task_dict[tab_task.name]["prep_model_combo"] = ttk.Combobox(tab_task, values=prep_model_values)
    task_dict[tab_task.name]["prep_model_combo"].set("StandardScaler")
    task_dict[tab_task.name]["prep_model_combo"].grid(row=8, column=1)

    task_dict[tab_task.name]["shuffle"] = tk.BooleanVar()
    task_dict[tab_task.name]["shuffle"].set(True)
    shuffle_checkbutton = tk.Checkbutton(tab_task, text="Shuffle data", variable=task_dict[tab_task.name]["shuffle"])
    shuffle_checkbutton.grid(row=9, column=1, sticky="W")

    # columns 0+1: Experiment
    experiment_label = tk.Label(tab_task, text="Experiment options:")
    experiment_label.grid(row=10, column=0, sticky="W")

    max_time_label = tk.Label(tab_task, text="MAX_TIME (min):")
    max_time_label.grid(row=11, column=0, sticky="W")
    task_dict[tab_task.name]["max_time_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["max_time_entry"].insert(0, "1")
    task_dict[tab_task.name]["max_time_entry"].grid(row=11, column=1)

    fun_evals_label = tk.Label(tab_task, text="FUN_EVALS (int|inf):")
    fun_evals_label.grid(row=12, column=0, sticky="W")
    task_dict[tab_task.name]["fun_evals_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["fun_evals_entry"].insert(0, "30")
    task_dict[tab_task.name]["fun_evals_entry"].grid(row=12, column=1)

    init_size_label = tk.Label(tab_task, text="INIT_SIZE (int):")
    init_size_label.grid(row=13, column=0, sticky="W")
    task_dict[tab_task.name]["init_size_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["init_size_entry"].insert(0, "5")
    task_dict[tab_task.name]["init_size_entry"].grid(row=13, column=1)

    noise_label = tk.Label(tab_task, text="NOISE (bool):")
    noise_label.grid(row=14, column=0, sticky="W")
    task_dict[tab_task.name]["noise_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["noise_entry"].insert(0, "True")
    task_dict[tab_task.name]["noise_entry"].grid(row=14, column=1)

    lambda_min_max_label = tk.Label(tab_task, text="Lambda (nugget): min, max:")
    lambda_min_max_label.grid(row=15, column=0, sticky="W")
    message = (
        "The min max values for Kriging.\n"
        "If set to 0, 0, no noise will be used in the surrogate.\n"
        "Default is 1e-3, 1e2."
    )
    lambda_min_max_tip = Hovertip(
        lambda_min_max_label,
        message,
    )
    task_dict[tab_task.name]["lambda_min_max_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["lambda_min_max_entry"].insert(0, "1e-3, 1e2")
    task_dict[tab_task.name]["lambda_min_max_entry"].grid(row=15, column=1)

    max_sp_label = tk.Label(tab_task, text="max surrogate points (int):")
    max_sp_label.grid(row=16, column=0, sticky="W")
    task_dict[tab_task.name]["max_sp_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["max_sp_entry"].insert(0, "30")
    task_dict[tab_task.name]["max_sp_entry"].grid(row=16, column=1)

    seed_label = tk.Label(tab_task, text="seed (int):")
    seed_label.grid(row=17, column=0, sticky="W")
    task_dict[tab_task.name]["seed_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["seed_entry"].insert(0, "123")
    task_dict[tab_task.name]["seed_entry"].grid(row=17, column=1)

    # columns 0+1: Evaluation
    experiment_label = tk.Label(tab_task, text="Evaluation options:")
    experiment_label.grid(row=18, column=0, sticky="W")

    metric_label = tk.Label(tab_task, text="metric (sklearn):")
    metric_label.grid(row=19, column=0, sticky="W")
    task_dict[tab_task.name]["metric_combo"] = ttk.Combobox(tab_task, values=task_dict[tab_task.name]["metric_levels"])
    # Default selection, the first metric in the list:
    task_dict[tab_task.name]["metric_combo"].set(task_dict[tab_task.name]["metric_levels"][0])
    task_dict[tab_task.name]["metric_combo"].grid(row=19, column=1)

    metric_weights_label = tk.Label(tab_task, text="weights: y,time,mem (>0.0):")
    metric_weights_label.grid(row=20, column=0, sticky="W")
    message = (
        "The weights for metric, time, and memory.\n"
        "All values are positive real numbers and should be separated by a comma.\n"
        "If the metric is to be minimized, the weights will be automatically adopted.\n"
        "If '1,0,0' is selected, only the metric is considered.\n"
        "If '1000,1,1' is selected, the metric is considered 1000 times more important than time and memory."
    )
    metric_weights_tip = Hovertip(
        metric_weights_label,
        message,
    )
    task_dict[tab_task.name]["metric_weights_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["metric_weights_entry"].insert(0, "1000, 1, 1")
    task_dict[tab_task.name]["metric_weights_entry"].grid(row=20, column=1)

    horizon_label = tk.Label(tab_task, text="horizon (int):")
    horizon_label.grid(row=21, column=0, sticky="W")
    task_dict[tab_task.name]["horizon_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["horizon_entry"].insert(0, "10")
    task_dict[tab_task.name]["horizon_entry"].grid(row=21, column=1)

    oml_grace_period_label = tk.Label(tab_task, text="oml_grace_period (int|None):")
    oml_grace_period_label.grid(row=22, column=0, sticky="W")
    task_dict[tab_task.name]["oml_grace_period_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["oml_grace_period_entry"].insert(0, "None")
    task_dict[tab_task.name]["oml_grace_period_entry"].grid(row=22, column=1)
    message = "The grace period for online learning (OML).\n" "If None, the grace period is set to the horizon."
    oml_grace_period_tip = Hovertip(
        oml_grace_period_label,
        message,
    )

    # Experiment name:
    experiment_label = tk.Label(tab_task, text="Experiment Name:")
    experiment_label.grid(row=23, column=0, sticky="W")

    prefix_label = tk.Label(tab_task, text="Name prefix (str):")
    prefix_label.grid(row=24, column=0, sticky="W")
    task_dict[tab_task.name]["prefix_entry"] = tk.Entry(tab_task)
    task_dict[tab_task.name]["prefix_entry"].insert(0, "00")
    task_dict[tab_task.name]["prefix_entry"].grid(row=24, column=1)


def create_second_column(tab_task):
    # colummns 2-6: Model
    model_label = tk.Label(tab_task, text="Model options:")
    model_label.grid(row=1, column=2, sticky="W")

    hparam_label = tk.Label(tab_task, text="Hyperparameters:")
    hparam_label.grid(row=2, column=2, sticky="W")

    model_label = tk.Label(tab_task, text="Default values:")
    model_label.grid(row=2, column=3, sticky="W")

    model_label = tk.Label(tab_task, text="Lower bounds:")
    model_label.grid(row=2, column=4, sticky="W")

    model_label = tk.Label(tab_task, text="Upper bounds:")
    model_label.grid(row=2, column=5, sticky="W")

    model_label = tk.Label(tab_task, text="Transformation:")
    model_label.grid(row=2, column=6, sticky="W")


def create_third_column(tab_task):
    global task_dict
    # column 8: Save and run button
    tb_label = tk.Label(tab_task, text="Tensorboard options:")
    tb_label.grid(row=1, column=8, sticky="W")

    task_dict[tab_task.name]["tb_clean"] = tk.BooleanVar()
    task_dict[tab_task.name]["tb_clean"].set(True)
    tf_clean_checkbutton = tk.Checkbutton(
        tab_task, text="TENSORBOARD_CLEAN", variable=task_dict[tab_task.name]["tb_clean"]
    )
    tf_clean_checkbutton.grid(row=2, column=8, sticky="W")
    message = (
        "If checked, tensorboard's run dir will be moved to runs_OLD\n"
        "and a new, empty runs dir will be used.\n"
        "Does only work with Unix systems."
    )
    tf_clean_tip = Hovertip(
        tf_clean_checkbutton,
        message,
    )

    task_dict[tab_task.name]["tb_start"] = tk.BooleanVar()
    task_dict[tab_task.name]["tb_start"].set(True)
    tf_start_checkbutton = tk.Checkbutton(
        tab_task, text="Start TENSORBOARD", variable=task_dict[tab_task.name]["tb_start"]
    )
    tf_start_checkbutton.grid(row=3, column=8, sticky="W")

    task_dict[tab_task.name]["tb_stop"] = tk.BooleanVar()
    task_dict[tab_task.name]["tb_stop"].set(True)
    tf_stop_checkbutton = tk.Checkbutton(
        tab_task, text="Stop TENSORBOARD", variable=task_dict[tab_task.name]["tb_stop"]
    )
    tf_stop_checkbutton.grid(row=4, column=8, sticky="W")

    tb_text = tk.Label(tab_task, text="Open browser logging:")
    tb_text.grid(row=5, column=8, sticky="W")
    tb_link = tk.Label(tab_task, text="http://localhost:6006", fg="blue", cursor="hand2")
    tb_link.bind("<Button-1>", lambda e: webbrowser.open_new("http://localhost:6006"))
    tb_link.grid(row=5, column=9, sticky="W")

    spot_doc = tk.Label(tab_task, text="Open SPOT documentation:")
    spot_doc.grid(row=6, column=8, sticky="W")
    spot_link = tk.Label(tab_task, text="spotPython documentation", fg="blue", cursor="hand2")
    spot_link.bind(
        "<Button-1>", lambda e: webbrowser.open_new("https://sequential-parameter-optimization.github.io/spotPython/")
    )
    spot_link.grid(row=6, column=9, sticky="W")

    spot_river_doc = tk.Label(tab_task, text="Open spotRiver documentation:")
    spot_river_doc.grid(row=7, column=8, sticky="W")
    spot_river_link = tk.Label(tab_task, text="spotRiver documentation", fg="blue", cursor="hand2")
    spot_river_link.bind(
        "<Button-1>", lambda e: webbrowser.open_new("https://sequential-parameter-optimization.github.io/spotRiver/")
    )
    spot_river_link.grid(row=7, column=9, sticky="W")

    river_doc = tk.Label(tab_task, text="Open River documentation:")
    river_doc.grid(row=8, column=8, sticky="W")
    river_doc_link = tk.Label(tab_task, text="River documentation", fg="blue", cursor="hand2")
    river_doc_link.bind("<Button-1>", lambda e: webbrowser.open_new("https://riverml.xyz/latest/api/overview/"))
    river_doc_link.grid(row=8, column=9, sticky="W")

    data_button = ttk.Button(
        tab_task, text="Show Data", command=lambda: run_experiment(tab_task=tab_task, show_data_only=True)
    )
    data_button.grid(row=10, column=8, columnspan=2, sticky="E")
    load_button = ttk.Button(tab_task, text="Load Experiment", command=lambda: load_experiment(tab_task=tab_task))
    load_button.grid(row=11, column=8, columnspan=2, sticky="E")
    save_button = ttk.Button(
        tab_task, text="Save Experiment", command=lambda: run_experiment(tab_task=tab_task, save_only=True)
    )
    save_button.grid(row=12, column=8, columnspan=2, sticky="E")
    run_button = ttk.Button(tab_task, text="Run Experiment", command=lambda: run_experiment(tab_task=tab_task))
    run_button.grid(row=13, column=8, columnspan=2, sticky="E")


def update_hyperparams(event, tab_task):
    global hyper_dict, task_dict

    print(f"\n\ntab_task in update_hyperparams(): {tab_task.name}")

    destroy_entries(hyper_dict[tab_task.name]["label"])
    destroy_entries(hyper_dict[tab_task.name]["default_entry"])
    destroy_entries(hyper_dict[tab_task.name]["lower_bound_entry"])
    destroy_entries(hyper_dict[tab_task.name]["upper_bound_entry"])
    destroy_entries(hyper_dict[tab_task.name]["transform_entry"])
    if hyper_dict[tab_task.name]["factor_level_entry"] is not None:
        for i in range(len(hyper_dict[tab_task.name]["factor_level_entry"])):
            if hyper_dict[tab_task.name]["factor_level_entry"][i] is not None and not isinstance(
                hyper_dict[tab_task.name]["factor_level_entry"][i], StringVar
            ):
                hyper_dict[tab_task.name]["factor_level_entry"][i].destroy()

    core_model_name = task_dict[tab_task.name]["core_model_combo"].get()
    coremodel = core_model_name.split(".")[1]
    # if model is a key in rhd.hyper_dict set dict = rhd.hyper_dict[model]
    if coremodel in rhd.hyper_dict:
        dict = rhd.hyper_dict[coremodel]
    else:
        dict = load_dict_from_file(coremodel, dirname="userModel")
    update_entries_from_dict(dict, tab_task=tab_task, hyper_dict=hyper_dict)


# Create the main application window
app = tk.Tk()
app.title("Spot River Hyperparameter Tuning GUI")

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(app)

# Create the "Classification" tab
classification_tab = ttk.Frame(notebook)
classification_tab.name = "classification_tab"
notebook.add(classification_tab, text="Binary classification")
create_first_column(tab_task=classification_tab)
create_second_column(tab_task=classification_tab)
create_third_column(tab_task=classification_tab)

# Create the "Regression" tab
regression_tab = ttk.Frame(notebook)
regression_tab.name = "regression_tab"
notebook.add(regression_tab, text="Regression")
create_first_column(tab_task=regression_tab)
create_second_column(regression_tab)
create_third_column(regression_tab)

# Create the "Analysis" tab
analysis_tab = ttk.Frame(notebook)
notebook.add(analysis_tab, text="Analysis")

# Create the "Result" tab
result_tab = ttk.Frame(notebook)
notebook.add(result_tab, text="Result")

notebook.pack()

# Add the Logo image in all tabs
logo_image = tk.PhotoImage(file="images/spotlogo.png")
logo_label = tk.Label(classification_tab, image=logo_image)
logo_label.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="W")

analysis_logo_label = tk.Label(analysis_tab, image=logo_image)
analysis_logo_label.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="W")

regression_logo_label = tk.Label(regression_tab, image=logo_image)
regression_logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

result_logo_label = tk.Label(result_tab, image=logo_image)
result_logo_label.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="W")

# Analysis options
analysis_label = tk.Label(analysis_tab, text="Analysis options:")
analysis_label.grid(row=1, column=0, sticky="W")

progress_plot_button = ttk.Button(analysis_tab, text="Progress plot", command=call_progress_plot)
progress_plot_button.grid(row=1, column=1, columnspan=2, sticky="W")

compare_tuned_default_button = ttk.Button(
    analysis_tab, text="Compare tuned vs. default", command=call_compare_tuned_default
)
compare_tuned_default_button.grid(row=2, column=1, columnspan=2, sticky="W")

actual_vs_prediction_button = ttk.Button(
    analysis_tab, text="Actual versus predicted", command=call_actual_vs_prediction
)
actual_vs_prediction_button.grid(row=3, column=1, columnspan=2, sticky="W")

plot_confusion_matrices_button = ttk.Button(
    analysis_tab, text="Confusion matrices", command=call_plot_confusion_matrices
)
plot_confusion_matrices_button.grid(row=4, column=1, columnspan=2, sticky="W")

plot_rocs_button = ttk.Button(analysis_tab, text="ROC", command=call_plot_rocs)
plot_rocs_button.grid(row=5, column=1, columnspan=2, sticky="W")

importance_plot_button = ttk.Button(analysis_tab, text="Importance plot", command=call_importance_plot)
importance_plot_button.grid(row=6, column=1, columnspan=2, sticky="W")

contour_plot_button = ttk.Button(analysis_tab, text="Contour plot", command=call_contour_plot)
contour_plot_button.grid(row=7, column=1, columnspan=2, sticky="W")

parallel_plot_button = ttk.Button(analysis_tab, text="Parallel plot (Browser)", command=call_parallel_plot)
parallel_plot_button.grid(row=8, column=1, columnspan=2, sticky="W")

# Result options:
show_result_button = ttk.Button(result_tab, text="Show result", command=show_result)
show_result_button.grid(row=1, column=0, columnspan=2, sticky="W")
# Run the mainloop

app.mainloop()
