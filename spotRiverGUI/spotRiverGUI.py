import pprint
import webbrowser
import sklearn.metrics
from spotRiver.data.river_hyper_dict import RiverHyperDict
import river
from river import forest, tree, linear_model
from river import preprocessing
import tkinter as tk
from spotRiver.data.selector import data_selector
import os
import numpy as np
from tkinter import ttk, StringVar
from idlelib.tooltip import Hovertip
import math
from spotPython.utils.init import fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.hyperparameters.values import (
    set_control_hyperparameter_value,
    get_var_type,
    get_var_name,
    get_bound_values,
)
from spotGUI.tuner.spotRun import (
    run_spot_python_experiment,
    contour_plot,
    parallel_plot,
    importance_plot,
    progress_plot,
    compare_tuned_default,
    all_compare_tuned_default,
    plot_confusion_matrices,
    plot_rocs,
    destroy_entries,
    load_file_dialog,
    get_report_file_name,
    get_result,
)
from spotPython.utils.eda import gen_design_table
from spotPython.utils.file import load_dict_from_file
from spotRiver.fun.hyperriver import HyperRiver
from spotRiver.utils.data_conversion import convert_to_df
from spotRiver.data.csvdataset import CSVDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from spotRiver.utils.data_conversion import rename_df_to_xy
from spotPython.utils.metrics import get_metric_sign
from spotPython.utils.file import load_experiment as load_experiment_spot

core_model_names = [
    "forest.AMFClassifier",
    "tree.ExtremelyFastDecisionTreeClassifier",
    "tree.HoeffdingTreeClassifier",
    "tree.HoeffdingAdaptiveTreeClassifier",
    "linear_model.LogisticRegression",
]
metric_levels = [
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
river_binary_classification_datasets = ["Bananas", "CreditCard", "Elec2", "Higgs", "HTTP", "Phishing"]
spot_tuner = None
rhd = RiverHyperDict()
#
n_keys = 25
label = [None] * n_keys
default_entry = [None] * n_keys
lower_bound_entry = [None] * n_keys
upper_bound_entry = [None] * n_keys
factor_level_entry = [None] * n_keys
transform_entry = [None] * n_keys


def get_river_dataset_from_name(
    data_set_name,
    n_total=None,
    river_datasets=None,
):
    """Converts a data set name to a pandas DataFrame.

    Args:
        data_set_name (str):
            The name of the data set.
            If the data set name is not in river_datasets, the data set is assumed to be a CSV file.
        n_total (int):
            The number of samples to be used from the data set.
            If n_total is None, the full data set is used.
            Defaults to None.
        river_datasets (list):
            A list of the available river data sets.
            If the data set name is not in river_datasets,
            the data set is assumed to be a CSV file.

    Returns:
        pd.DataFrame:
            The data set as a pandas DataFrame.
        n_samples (int):
            The number of samples in the data set.
    """
    print(f"data_set_name: {data_set_name}")
    print("river_datasets: ", river_datasets)
    # data_set ends with ".csv" or data_set ends with ".pkl":
    if data_set_name.endswith(".csv"):
        print(f"data_set_name: {data_set_name}")
        dataset = CSVDataset(filename=data_set_name, directory="./userData/").data
        n_samples = dataset.shape[0]
    elif data_set_name in river_datasets:
        dataset, n_samples = data_selector(
            data_set=data_set_name,
        )
        # convert the river datasets to a pandas DataFrame, the target column
        # of the resulting DataFrame is target_column
        dataset = convert_to_df(dataset, target_column="y", n_total=n_total)
    return dataset, n_samples


def run_experiment(save_only=False, show_data_only=False):
    global spot_tuner, fun_control, label, default_entry, lower_bound_entry, upper_bound_entry, factor_level_entry

    n_total = n_total_entry.get()

    noise = noise_entry.get()
    if noise.lower() == "true":
        noise = True
    else:
        noise = False

    if n_total == "None" or n_total == "All":
        n_total = None
    else:
        n_total = int(n_total)

    fun_evals = fun_evals_entry.get()
    if fun_evals == "None" or fun_evals == "inf":
        fun_evals_val = math.inf
    else:
        fun_evals_val = int(fun_evals)

    seed = int(seed_entry.get())
    test_size = float(test_size_entry.get())

    core_model_name = core_model_combo.get()

    # lambda (Kriging nugget)
    lambda_min_max = lambda_min_max_entry.get()
    # split the string into a list of strings
    lbd = lambda_min_max.split(",")
    # if len(lbd) != 2, set the lambda values to the default values [-3, 2]
    if len(lbd) != 2:
        lbd = ["1e-6", "1e2"]

    # metrics
    metric_name = metric_combo.get()
    metric_sklearn = getattr(sklearn.metrics, metric_name)
    weight_sgn = get_metric_sign(metric_name)
    metric_weights = metric_weights_entry.get()

    # split the string into a list of strings
    mw = metric_weights.split(",")
    # if len(mw) != 3, set the weights to the default values [1000, 1, 1]
    if len(mw) != 3:
        mw = ["1000", "1", "1"]
    weights = np.array([weight_sgn * float(mw[0]), float(mw[1]), float(mw[2])])
    # River specific parameters
    oml_grace_period = oml_grace_period_entry.get()
    if oml_grace_period == "None":
        oml_grace_period = None
    else:
        oml_grace_period = int(oml_grace_period)

    data_set_name = data_set_combo.get()
    dataset, n_samples = get_river_dataset_from_name(
        data_set_name=data_set_name, n_total=n_total,
        river_datasets=river_binary_classification_datasets
    )

    # TODO: implement this as a function in spotRun.py
    # Rename the columns of a DataFrame to x1, x2, ..., xn, y.
    # From now on we assume that the target column is called "y" and
    # is of type int (binary classification)
    df = rename_df_to_xy(df=dataset, target_column="y")
    df["y"] = df["y"].astype(int)
    target_column = "y"
    # split the data set into a training and a test set,
    # where the test set is a percentage of the data set given as test_size:
    X = df.drop(columns=[target_column])
    Y = df[target_column]
    # Split the data into training and test sets
    # test_size is the percentage of the data that should be held over for testing
    # random_state is a seed for the random number generator to make your train and test splits reproducible
    train_features, test_features, train_target, test_target = train_test_split(
        X, Y, test_size=test_size, random_state=seed
    )
    # combine the training features and the training target into a training DataFrame
    train = pd.concat([train_features, train_target], axis=1)
    test = pd.concat([test_features, test_target], axis=1)
    n_samples = train.shape[0] + test.shape[0]

    # Get the selected prep and core model and add it to the fun_control dictionary
    prepmodel = prep_model_combo.get()
    if prepmodel == "StandardScaler":
        prep_model = preprocessing.StandardScaler()
    elif prepmodel == "MinMaxScaler":
        prep_model = preprocessing.MinMaxScaler()
    else:
        prep_model = None

    TENSORBOARD_CLEAN = bool(tb_clean.get())
    tensorboard_start = bool(tb_start.get())
    tensorboard_stop = bool(tb_stop.get())

    # Initialize the fun_control dictionary with the static parameters,
    # i.e., the parameters that are not hyperparameters (depending on the core model)
    fun_control = fun_control_init(
        PREFIX=prefix_entry.get(),
        TENSORBOARD_CLEAN=TENSORBOARD_CLEAN,
        core_model_name=core_model_name,
        data_set_name=data_set_name,
        fun_evals=fun_evals_val,
        fun_repeats=1,
        horizon=int(horizon_entry.get()),
        max_time=float(max_time_entry.get()),
        metric_sklearn=metric_sklearn,
        noise=noise,
        n_samples=n_samples,
        ocba_delta=0,
        oml_grace_period=oml_grace_period,
        prep_model=prep_model,
        seed=seed,
        target_column="y",
        test=test,
        test_size=test_size,
        train=train,
        tolerance_x=np.sqrt(np.spacing(1)),
        verbosity=1,
        weights=weights,
        log_level=50,
    )

    core_model_module = core_model_name.split(".")[0]
    coremodel = core_model_name.split(".")[1]
    core_model_instance = getattr(getattr(river, core_model_module), coremodel)

    add_core_model_to_fun_control(
        core_model=core_model_instance,
        fun_control=fun_control,
        hyper_dict=RiverHyperDict,
        filename=None,
    )
    dict = rhd.hyper_dict[coremodel]

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

    for i, (key, value) in enumerate(dict.items()):
        if dict[key]["type"] == "int":
            set_control_hyperparameter_value(
                fun_control, key, [int(lower_bound_entry[i].get()), int(upper_bound_entry[i].get())]
            )
        if (dict[key]["type"] == "factor") and (dict[key]["core_model_parameter_type"] == "bool"):
            # fun_control["core_model_hyper_dict"][key].update({"lower": int(lower_bound_entry[i].get())})
            # fun_control["core_model_hyper_dict"][key].update({"upper": int(upper_bound_entry[i].get())})
            set_control_hyperparameter_value(fun_control, key, [int(lower_bound_entry[i].get()), int(upper_bound_entry[i].get())])
        if dict[key]["type"] == "float":
            set_control_hyperparameter_value(
                fun_control, key, [float(lower_bound_entry[i].get()), float(upper_bound_entry[i].get())]
            )
        if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
            fle = factor_level_entry[i].get()
            # convert the string to a list of strings
            fle = fle.split()
            set_control_hyperparameter_value(fun_control, key, fle)
            fun_control["core_model_hyper_dict"][key].update({"upper": len(fle) - 1})

    design_control = design_control_init(
        init_size=int(init_size_entry.get()),
        repeats=1,
    )

    kriging_noise = True
    lbd_min = float(lbd[0])
    lbd_max = float(lbd[1])
    if lbd_min < 0:
        lbd_min = 1e-6
    if lbd_max < 0:
        lbd_max = 1e2
    if lbd_min == 0.0 and lbd_max == 0.0:
        kriging_noise = False
    surrogate_control = surrogate_control_init(
        noise=kriging_noise,
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


def load_experiment():
    global label, default_entry, lower_bound_entry, upper_bound_entry, transform_entry, factor_level_entry, spot_tuner, fun_control
    filename = load_file_dialog()
    if filename:
        spot_tuner, fun_control, design_control, surrogate_control, optimizer_control = load_experiment_spot(filename)
        print("\nfun_control in load_experiment():")
        pprint.pprint(fun_control)

        data_set_combo.delete(0, tk.END)
        data_set_name = fun_control["data_set_name"]

        if data_set_name == "CSVDataset" or data_set_name == "PKLDataset":
            filename = vars(fun_control["data_set"])["filename"]
            print("filename: ", filename)
            data_set_combo.set(filename)
        else:
            data_set_combo.set(data_set_name)

        # static parameters, that are not hyperparameters (depending on the core model)

        n_total_entry.delete(0, tk.END)
        n_total_entry.insert(0, str(fun_control["n_total"]))

        test_size_entry.delete(0, tk.END)
        test_size_entry.insert(0, str(fun_control["test_size"]))

        prep_model_combo.delete(0, tk.END)
        prep_model_name = fun_control["prep_model"].__class__.__name__
        prep_model_combo.set(prep_model_name)

        max_time_entry.delete(0, tk.END)
        max_time_entry.insert(0, str(fun_control["max_time"]))

        fun_evals_entry.delete(0, tk.END)
        fun_evals_entry.insert(0, str(fun_control["fun_evals"]))

        init_size_entry.delete(0, tk.END)
        init_size_entry.insert(0, str(design_control["init_size"]))

        prefix_entry.delete(0, tk.END)
        prefix_entry.insert(0, str(fun_control["PREFIX"]))

        noise_entry.delete(0, tk.END)
        noise_entry.insert(0, str(fun_control["noise"]))

        seed_entry.delete(0, tk.END)
        seed_entry.insert(0, str(fun_control["seed"]))

        metric_combo.delete(0, tk.END)
        metric_name = fun_control["metric_sklearn"].__name__
        metric_combo.set(metric_name)

        metric_weights_entry.delete(0, tk.END)
        wghts = fun_control["weights"]
        # take the absolute value of all weights
        wghts = [abs(w) for w in wghts]
        metric_weights_entry.insert(0, f"{wghts[0]}, {wghts[1]}, {wghts[2]}")
        # metric_weights_entry.insert(0, str(fun_control["weights"]))

        lambda_min_max_entry.delete(0, tk.END)
        min_lbd = surrogate_control["min_Lambda"]
        max_lbd = surrogate_control["max_Lambda"]
        lambda_min_max_entry.insert(0, f"{min_lbd}, {max_lbd}")

        horizon_entry.delete(0, tk.END)
        horizon_entry.insert(0, str(fun_control["horizon"]))

        oml_grace_period_entry.delete(0, tk.END)
        oml_grace_period_entry.insert(0, str(fun_control["oml_grace_period"]))

        destroy_entries(label)
        destroy_entries(default_entry)
        destroy_entries(lower_bound_entry)
        destroy_entries(upper_bound_entry)
        destroy_entries(transform_entry)

        if factor_level_entry is not None:
            for i in range(len(factor_level_entry)):
                if factor_level_entry[i] is not None and not isinstance(factor_level_entry[i], StringVar):
                    factor_level_entry[i].destroy()

        update_entries_from_dict(fun_control["core_model_hyper_dict"])
        # Modeloptions
        core_model_combo.delete(0, tk.END)
        core_model_combo.set(fun_control["core_model_name"])


def call_compare_tuned_default():
    if spot_tuner is not None and fun_control is not None:
        compare_tuned_default(spot_tuner, fun_control, show=True)


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


def update_entries_from_dict(dict):
    global label, default_entry, lower_bound_entry, upper_bound_entry, transform_entry, factor_level_entry
    n_keys = len(dict)
    # Create a list of labels and entries with the same length as the number of keys in the dictionary
    label = [None] * n_keys
    default_entry = [None] * n_keys
    lower_bound_entry = [None] * n_keys
    upper_bound_entry = [None] * n_keys
    factor_level_entry = [None] * n_keys
    transform_entry = [None] * n_keys
    for i, (key, value) in enumerate(dict.items()):
        if (
            dict[key]["type"] == "int"
            or dict[key]["type"] == "float"
            or dict[key]["core_model_parameter_type"] == "bool"
        ):
            # Create a label with the key as text
            label[i] = tk.Label(run_tab, text=key)
            label[i].grid(row=i + 3, column=2, sticky="W")
            label[i].update()
            # # Create an entry with the default value as the default text
            # default_entry[i] = tk.Entry(run_tab)
            # default_entry[i].insert(0, dict[key]["default"])
            # default_entry[i].grid(row=i + 2, column=3, sticky="W")
            # default_entry[i].update()
            # Create an entry with the default value as the default text
            default_entry[i] = tk.Label(run_tab, text=dict[key]["default"])
            default_entry[i].grid(row=i + 3, column=3, sticky="W")
            default_entry[i].update()
            # add the lower bound values in column 4
            lower_bound_entry[i] = tk.Entry(run_tab)
            lower_bound_entry[i].insert(0, dict[key]["lower"])
            lower_bound_entry[i].grid(row=i + 3, column=4, sticky="W")
            # add the upper bound values in column 5
            upper_bound_entry[i] = tk.Entry(run_tab)
            upper_bound_entry[i].insert(0, dict[key]["upper"])
            upper_bound_entry[i].grid(row=i + 3, column=5, sticky="W")
            # add the transformation values in column 6
            transform_entry[i] = tk.Label(run_tab, text=dict[key]["transform"], padx=15)
            transform_entry[i].grid(row=i + 3, column=6, sticky="W")

        if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
            # Create a label with the key as text
            label[i] = tk.Label(run_tab, text=key)
            label[i].grid(row=i + 3, column=2, sticky="W")
            label[i].update()
            # Create an entry with the default value as the default text
            default_entry[i] = tk.Label(run_tab, text=dict[key]["default"])
            default_entry[i].grid(row=i + 3, column=3, sticky="W")
            # add the lower bound values in column 2
            factor_level_entry[i] = tk.Entry(run_tab)
            # TODO: replace " " with ", " for the levels
            # lvls = dict[key]["levels"]
            # factor_level_entry[i].insert(0, ", ".join(lvls))
            # generates several commas, if used several times
            factor_level_entry[i].insert(0, dict[key]["levels"])
            factor_level_entry[i].grid(row=i + 3, column=4, columnspan=2, sticky=tk.W + tk.E)


def update_hyperparams(event):
    global label, default_entry, lower_bound_entry, upper_bound_entry, factor_level_entry, transform_entry

    destroy_entries(label)
    destroy_entries(default_entry)
    destroy_entries(lower_bound_entry)
    destroy_entries(upper_bound_entry)
    destroy_entries(transform_entry)

    if factor_level_entry is not None:
        for i in range(len(factor_level_entry)):
            if factor_level_entry[i] is not None and not isinstance(factor_level_entry[i], StringVar):
                factor_level_entry[i].destroy()

    # coremodel = core_model_combo.get()
    core_model = core_model_combo.get()
    coremodel = core_model.split(".")[1]
    # if model is a key in rhd.hyper_dict set dict = rhd.hyper_dict[model]
    if coremodel in rhd.hyper_dict:
        dict = rhd.hyper_dict[coremodel]
    else:
        dict = load_dict_from_file(coremodel, dirname="userModel")
    update_entries_from_dict(dict)


# Create the main application window
app = tk.Tk()
app.title("Spot River Hyperparameter Tuning GUI")

# generate a list of StringVar() objects of size n_keys
for i in range(n_keys):
    factor_level_entry.append(StringVar())

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(app)
# notebook.pack(fill='both', expand=True)

# Create and pack entry fields for the "Run" tab
run_tab = ttk.Frame(notebook)
notebook.add(run_tab, text="Binary classification")

# colummns 0+1: Data

core_model_label = tk.Label(run_tab, text="Core model:")
core_model_label.grid(row=1, column=0, sticky="W")

core_model_label = tk.Label(run_tab, text="Select core model:")
core_model_label.grid(row=2, column=0, sticky="W")
for filename in os.listdir("userModel"):
    if filename.endswith(".json"):
        core_model_names.append(os.path.splitext(filename)[0])
core_model_combo = ttk.Combobox(run_tab, values=core_model_names)
# core_model_combo.set("Select Model")  # Default selection
core_model_combo.set("tree.HoeffdingTreeClassifier")  # Default selection
core_model_combo.bind("<<ComboboxSelected>>", update_hyperparams)
core_model_combo.grid(row=2, column=1)
update_hyperparams(None)

data_label = tk.Label(run_tab, text="Data options:")
data_label.grid(row=3, column=0, sticky="W")

data_set_label = tk.Label(run_tab, text="Select data_set:")
data_set_label.grid(row=4, column=0, sticky="W")
data_set_tip = Hovertip(data_set_label, "The data set.\n User specified data sets must have the target value in the last column.\n They are assumed to be in the directory 'userData'.\n The data set can be a CSV file.")
data_set_values = river_binary_classification_datasets
# get all *.csv files in the data directory "userData" and append them to the list of data_set_values
data_set_values.extend([f for f in os.listdir("userData") if f.endswith(".csv") or f.endswith(".pkl")])
data_set_combo = ttk.Combobox(run_tab, values=data_set_values)
data_set_combo.set("Phishing")  # Default selection
data_set_combo.grid(row=4, column=1)

n_total_label = tk.Label(run_tab, text="n_total (int|All):")
n_total_label.grid(row=5, column=0, sticky="W")
n_total_entry = tk.Entry(run_tab)
n_total_entry.insert(0, "All")
n_total_entry.grid(row=5, column=1, sticky="W")

test_size_label = tk.Label(run_tab, text="test_size (perc.):")
test_size_label.grid(row=6, column=0, sticky="W")
test_size_entry = tk.Entry(run_tab)
test_size_entry.insert(0, "0.30")
test_size_entry.grid(row=6, column=1, sticky="W")

prep_model_label = tk.Label(run_tab, text="Select preprocessing model")
prep_model_label.grid(row=7, column=0, sticky="W")
prep_model_values = ["MinMaxScaler", "StandardScaler", "None"]
prep_model_combo = ttk.Combobox(run_tab, values=prep_model_values)
prep_model_combo.set("StandardScaler")
prep_model_combo.grid(row=7, column=1)

# columns 0+1: Experiment
experiment_label = tk.Label(run_tab, text="Experiment options:")
experiment_label.grid(row=8, column=0, sticky="W")

max_time_label = tk.Label(run_tab, text="MAX_TIME (min):")
max_time_label.grid(row=9, column=0, sticky="W")
max_time_entry = tk.Entry(run_tab)
max_time_entry.insert(0, "1")
max_time_entry.grid(row=9, column=1)

fun_evals_label = tk.Label(run_tab, text="FUN_EVALS (int|inf):")
fun_evals_label.grid(row=10, column=0, sticky="W")
fun_evals_entry = tk.Entry(run_tab)
fun_evals_entry.insert(0, "30")
fun_evals_entry.grid(row=10, column=1)

init_size_label = tk.Label(run_tab, text="INIT_SIZE (int):")
init_size_label.grid(row=11, column=0, sticky="W")
init_size_entry = tk.Entry(run_tab)
init_size_entry.insert(0, "5")
init_size_entry.grid(row=11, column=1)

noise_label = tk.Label(run_tab, text="NOISE (bool):")
noise_label.grid(row=12, column=0, sticky="W")
noise_entry = tk.Entry(run_tab)
noise_entry.insert(0, "True")
noise_entry.grid(row=12, column=1)

lambda_min_max_label = tk.Label(run_tab, text="Lambda (nugget): min, max:")
lambda_min_max_label.grid(row=13, column=0, sticky="W")
lambda_min_max_tip = Hovertip(lambda_min_max_label, "The min max values for Kriging.\nIf set to 0, 0, no noise will be used in the surrogate.\Default is -3, 2.")
lambda_min_max_entry = tk.Entry(run_tab)
lambda_min_max_entry.insert(0, "1e-3, 1e2")
lambda_min_max_entry.grid(row=13, column=1)

seed_label = tk.Label(run_tab, text="seed (int):")
seed_label.grid(row=14, column=0, sticky="W")
seed_entry = tk.Entry(run_tab)
seed_entry.insert(0, "123")
seed_entry.grid(row=14, column=1)

# columns 0+1: Evaluation
experiment_label = tk.Label(run_tab, text="Evaluation options:")
experiment_label.grid(row=15, column=0, sticky="W")


metric_label = tk.Label(run_tab, text="metric (sklearn):")
metric_label.grid(row=16, column=0, sticky="W")
metric_combo = ttk.Combobox(run_tab, values=metric_levels)
metric_combo.set("accuracy_score")  # Default selection
metric_combo.grid(row=16, column=1)

metric_weights_label = tk.Label(run_tab, text="weights: y,time,mem (>0.0):")
metric_weights_label.grid(row=17, column=0, sticky="W")
metric_weights_tip = Hovertip(metric_weights_label, "The weights for metric, time, and memory.\nAll values are positive real numbers and should be separated by a comma.\nIf the metric is to be minimized, the weights will be automatically adopted.\nIf '1,0,0' is selected, only the metric is considered.\nIf '1000,1,1' is selected, the metric is considered 1000 times more important than time and memory.")
metric_weights_entry = tk.Entry(run_tab)
metric_weights_entry.insert(0, "1000, 1, 1")
metric_weights_entry.grid(row=17, column=1)


horizon_label = tk.Label(run_tab, text="horizon (int):")
horizon_label.grid(row=18, column=0, sticky="W")
horizon_entry = tk.Entry(run_tab)
horizon_entry.insert(0, "10")
horizon_entry.grid(row=18, column=1)

oml_grace_period_label = tk.Label(run_tab, text="oml_grace_period (int|None):")
oml_grace_period_label.grid(row=19, column=0, sticky="W")
oml_grace_period_entry = tk.Entry(run_tab)
oml_grace_period_entry.insert(0, "None")
oml_grace_period_entry.grid(row=19, column=1)
oml_grace_period_tip = Hovertip(oml_grace_period_label, "The grace period for online learning (OML).\n If None, the grace period is set to the horizon.")

# Experiment name:
experiment_label = tk.Label(run_tab, text="Experiment Name:")
experiment_label.grid(row=20, column=0, sticky="W")

prefix_label = tk.Label(run_tab, text="Name prefix (str):")
prefix_label.grid(row=21, column=0, sticky="W")
prefix_entry = tk.Entry(run_tab)
prefix_entry.insert(0, "00")
prefix_entry.grid(row=21, column=1)


# colummns 2-6: Model
model_label = tk.Label(run_tab, text="Model options:")
model_label.grid(row=1, column=2, sticky="W")

hparam_label = tk.Label(run_tab, text="Hyperparameters:")
hparam_label.grid(row=2, column=2, sticky="W")

model_label = tk.Label(run_tab, text="Default values:")
model_label.grid(row=2, column=3, sticky="W")

model_label = tk.Label(run_tab, text="Lower bounds:")
model_label.grid(row=2, column=4, sticky="W")

model_label = tk.Label(run_tab, text="Upper bounds:")
model_label.grid(row=2, column=5, sticky="W")

model_label = tk.Label(run_tab, text="Transformation:")
model_label.grid(row=2, column=6, sticky="W")

# core_model_label = tk.Label(run_tab, text="Core model")
# core_model_label.grid(row=1, column=2, sticky="W")
# for filename in os.listdir("userModel"):
#     if filename.endswith(".json"):
#         core_model_names.append(os.path.splitext(filename)[0])
# core_model_combo = ttk.Combobox(run_tab, values=core_model_names)
# # core_model_combo.set("Select Model")  # Default selection
# core_model_combo.set("tree.HoeffdingTreeClassifier")  # Default selection
# core_model_combo.bind("<<ComboboxSelected>>", update_hyperparams)
# core_model_combo.grid(row=1, column=3)
# update_hyperparams(None)


# column 8: Save and run button
tb_label = tk.Label(run_tab, text="Tensorboard options:")
tb_label.grid(row=1, column=8, sticky="W")

tb_clean = tk.BooleanVar()
tb_clean.set(True)
tf_clean_checkbutton = tk.Checkbutton(run_tab, text="TENSORBOARD_CLEAN", variable=tb_clean)
tf_clean_checkbutton.grid(row=2, column=8, sticky="W")
tf_clean_tip = Hovertip(tf_clean_checkbutton, "If checked, tensorboard's run dir will be moved to runs_OLD\nand a new, empty runs dir will be used.\nDoes only work with Unix systems.")

tb_start = tk.BooleanVar()
tb_start.set(True)
tf_start_checkbutton = tk.Checkbutton(run_tab, text="Start TENSORBOARD", variable=tb_start)
tf_start_checkbutton.grid(row=3, column=8, sticky="W")

tb_stop = tk.BooleanVar()
tb_stop.set(True)
tf_stop_checkbutton = tk.Checkbutton(run_tab, text="Stop TENSORBOARD", variable=tb_stop)
tf_stop_checkbutton.grid(row=4, column=8, sticky="W")

tb_text = tk.Label(run_tab, text="Open browser logging:")
tb_text.grid(row=5, column=8, sticky="W")
tb_link = tk.Label(run_tab, text="http://localhost:6006", fg="blue", cursor="hand2")
tb_link.bind("<Button-1>", lambda e: webbrowser.open_new("http://localhost:6006"))
tb_link.grid(row=5, column=9, sticky="W")

spot_doc = tk.Label(run_tab, text="Open SPOT documentation:")
spot_doc.grid(row=6, column=8, sticky="W")
spot_link = tk.Label(run_tab, text="spotPython documentation", fg="blue", cursor="hand2")
spot_link.bind("<Button-1>", lambda e: webbrowser.open_new("https://sequential-parameter-optimization.github.io/spotPython/"))
spot_link.grid(row=6, column=9, sticky="W")

spot_river_doc = tk.Label(run_tab, text="Open spotRiver documentation:")
spot_river_doc.grid(row=7, column=8, sticky="W")
spot_river_link = tk.Label(run_tab, text="spotRiver documentation", fg="blue", cursor="hand2")
spot_river_link.bind("<Button-1>", lambda e: webbrowser.open_new("https://sequential-parameter-optimization.github.io/spotRiver/"))
spot_river_link.grid(row=7, column=9, sticky="W")

river_doc = tk.Label(run_tab, text="Open River documentation:")
river_doc.grid(row=8, column=8, sticky="W")
river_doc_link = tk.Label(run_tab, text="River documentation", fg="blue", cursor="hand2")
river_doc_link.bind("<Button-1>", lambda e: webbrowser.open_new("https://riverml.xyz/latest/api/overview/"))
river_doc_link.grid(row=8, column=9, sticky="W")

data_button = ttk.Button(run_tab, text="Show Data", command=lambda: run_experiment(show_data_only=True))
data_button.grid(row=10, column=8, columnspan=2, sticky="E")
load_button = ttk.Button(run_tab, text="Load Experiment", command=load_experiment)
load_button.grid(row=11, column=8, columnspan=2, sticky="E")
save_button = ttk.Button(run_tab, text="Save Experiment", command=lambda: run_experiment(save_only=True))
save_button.grid(row=12, column=8, columnspan=2, sticky="E")
run_button = ttk.Button(run_tab, text="Run Experiment", command=run_experiment)
run_button.grid(row=13, column=8, columnspan=2, sticky="E")


# TODO: Create and pack the "Regression" tab with a button to run the analysis
# regression_tab = ttk.Frame(notebook)
# notebook.add(regression_tab, text="Regression")

# # colummns 0+1: Data

# regression_data_label = tk.Label(regression_tab, text="Data options:")
# regression_data_label.grid(row=0, column=0, sticky="W")

# # colummns 2+3: Model
# regression_model_label = tk.Label(regression_tab, text="Model options:")
# regression_model_label.grid(row=0, column=2, sticky="W")

# # columns 4+5: Experiment
# regression_experiment_label = tk.Label(regression_tab, text="Experiment options:")
# regression_experiment_label.grid(row=0, column=4, sticky="W")


# Create and pack the "Analysis" tab
analysis_tab = ttk.Frame(notebook)
notebook.add(analysis_tab, text="Analysis")

result_tab = ttk.Frame(notebook)
notebook.add(result_tab, text="Result")

notebook.pack()

# Add the Logo image in both tabs
logo_image = tk.PhotoImage(file="images/spotlogo.png")
logo_label = tk.Label(run_tab, image=logo_image)
logo_label.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="W")

analysis_label = tk.Label(analysis_tab, text="Analysis options:")
analysis_label.grid(row=1, column=0, sticky="W")

progress_plot_button = ttk.Button(analysis_tab, text="Progress plot", command=call_progress_plot)
progress_plot_button.grid(row=1, column=1, columnspan=2, sticky="W")

compare_tuned_default_button = ttk.Button(
    analysis_tab, text="Compare tuned vs. default", command=call_compare_tuned_default
)
compare_tuned_default_button.grid(row=2, column=1, columnspan=2, sticky="W")

plot_confusion_matrices_button = ttk.Button(
    analysis_tab, text="Confusion matrices", command=call_plot_confusion_matrices
)
plot_confusion_matrices_button.grid(row=3, column=1, columnspan=2, sticky="W")

plot_rocs_button = ttk.Button(
    analysis_tab, text="ROC", command=call_plot_rocs
)
plot_rocs_button.grid(row=4, column=1, columnspan=2, sticky="W")

importance_plot_button = ttk.Button(analysis_tab, text="Importance plot", command=call_importance_plot)
importance_plot_button.grid(row=5, column=1, columnspan=2, sticky="W")

contour_plot_button = ttk.Button(analysis_tab, text="Contour plot", command=call_contour_plot)
contour_plot_button.grid(row=6, column=1, columnspan=2, sticky="W")

parallel_plot_button = ttk.Button(analysis_tab, text="Parallel plot (Browser)", command=call_parallel_plot)
parallel_plot_button.grid(row=7, column=1, columnspan=2, sticky="W")

analysis_logo_label = tk.Label(analysis_tab, image=logo_image)
analysis_logo_label.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="W")

# regression_logo_label = tk.Label(regression_tab, image=logo_image)
# regression_logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

result_logo_label = tk.Label(result_tab, image=logo_image)
result_logo_label.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="W")

show_result_button = ttk.Button(result_tab, text="Show result", command=show_result)
show_result_button.grid(row=1, column=0, columnspan=2, sticky="W")
# Run the mainloop

app.mainloop()
