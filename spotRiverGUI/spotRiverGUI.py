from sklearn.metrics import accuracy_score
from spotRiver.data.river_hyper_dict import RiverHyperDict
from river.forest import AMFClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.linear_model import LogisticRegression
from spotRiver.utils.data_conversion import convert_to_df
from river import preprocessing
import tkinter as tk
from spotRiver.data.selector import data_selector
import os
import numpy as np
from tkinter import ttk, StringVar
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
)
from spotPython.utils.eda import gen_design_table
from spotPython.utils.file import load_dict_from_file, load_core_model_from_file
from spotRiver.fun.hyperriver import HyperRiver

spot_tuner = None
rhd = RiverHyperDict()
#
n_keys = 25
label = [None] * n_keys
default_entry = [None] * n_keys
lower_bound_entry = [None] * n_keys
upper_bound_entry = [None] * n_keys
factor_level_entry = [None] * n_keys


def run_experiment(save_only=False):
    global spot_tuner, fun_control, label, default_entry, lower_bound_entry, upper_bound_entry, factor_level_entry

    n_total = n_total_entry.get()
    if n_total == "None" or n_total == "All":
        n_total = None
    else:
        n_total = int(n_total)

    fun_evals = fun_evals_entry.get()
    if fun_evals == "None" or fun_evals == "inf":
        fun_evals_val = math.inf
    else:
        fun_evals_val = int(fun_evals)

    # River specific parameters
    oml_grace_period = oml_grace_period_entry.get()
    if oml_grace_period == "None" or oml_grace_period == "n_train":
        oml_grace_period = None
    else:
        oml_grace_period = int(oml_grace_period)

    # check if data_set provided by spotRiver as a data set from the river package
    data_set = data_set_combo.get()
    filename = None
    directory = None
    target = target_column_entry.get()
    n_features = None
    n_samples = None
    converters = None
    parse_dates = None

    # check if data_set is provided as .csv
    if data_set.endswith(".csv"):
        directory = "./userData/"
        filename = data_set
        target_column = target_column_entry.get()
        # TODO: This is correct for the Phishing data set,
        # but needs to be adapted for other data sets:
        n_samples = 1_250
        n_features = 9
        parse_dates = None
        converters = {
            "empty_server_form_handler": float,
            "popup_window": float,
            "https": float,
            "request_from_other_domain": float,
            "anchor_from_other_domain": float,
            "is_popular": float,
            "long_url": float,
            "age_of_domain": int,
            "ip_in_url": int,
            "is_phishing": lambda x: x == "1",
            }

    # TODO: Add user specified types for feature and target columns:
    # feature_type=getattr(torch, feature_type_entry.get())
    # target_type=getattr(torch, target_type_entry.get())

    # TODO: This is a template for user specified data sets:
    # if data_set == "PhishingData.csv":
    #     filename = "PhishingData.csv"
    #     directory = "./userData"
    #     n_samples = 1_250
    #     n_features = 9
    #     parse_dates = None
    #     converters = {
    #         "empty_server_form_handler": float,
    #         "popup_window": float,
    #         "https": float,
    #         "request_from_other_domain": float,
    #         "anchor_from_other_domain": float,
    #         "is_popular": float,
    #         "long_url": float,
    #         "age_of_domain": int,
    #         "ip_in_url": int,
    #         "is_phishing": lambda x: x == "1",
    #         }

    dataset, n_samples = data_selector(
        data_set=data_set,
        filename=filename,
        directory=directory,
        target=target,
        n_features=n_features,
        n_samples=n_samples,
        converters=converters,
        parse_dates=parse_dates,
    )
    # target_column is the name of the target column in the resulting data frame df:
    target_column = "y"
    df = convert_to_df(dataset, target_column=target_column, n_total=n_total)
    df.columns = [f"x{i}" for i in range(1, dataset.n_features + 1)] + ["y"]
    df["y"] = df["y"].astype(int)
    # update n_samples to the actual number of samples in the data set,
    # because n_total might be smaller than n_samples which results in a smaller data set:
    test_size = float(test_size_entry.get())
    n_samples = len(df)
    n_train = int((1.0 - test_size) * n_samples)
    train = df[:n_train]
    print(f"train = {train.describe(include='all')}")
    test = df[n_train:]
    print(f"test = {test.describe(include='all')}")

    # Initialize the fun_control dictionary with the static parameters,
    # i.e., the parameters that are not hyperparameters (depending on the core model)
    fun_control = fun_control_init(
        PREFIX=prefix_entry.get(),
        TENSORBOARD_CLEAN=True,
        fun_evals=fun_evals_val,
        fun_repeats=1,
        # TODO: add horizon to fun_control and not as an update below
        # horizon=int(horizon_entry.get()),
        max_time=float(max_time_entry.get()),
        noise=bool(noise_entry.get()),
        ocba_delta=0,
        data_set=dataset,
        test_size=test_size,
        test=test,
        train=train,
        tolerance_x=np.sqrt(np.spacing(1)),
        verbosity=1,
        log_level=50,
    )

    # Get the selected prep and core model and add it to the fun_control dictionary
    prepmodel = prep_model_combo.get()
    if prepmodel == "StandardScaler":
        prep_model = preprocessing.StandardScaler()
    elif prepmodel == "MinMaxScaler":
        prep_model = preprocessing.MinMaxScaler()
    else:
        prep_model = None

    coremodel = core_model_combo.get()
    if coremodel == "LogisticRegression":
        add_core_model_to_fun_control(
            core_model=LogisticRegression,
            fun_control=fun_control,
            hyper_dict=RiverHyperDict,
            filename=None,
        )
        dict = rhd.hyper_dict[coremodel]
        # modify_hyper_parameter_bounds(fun_control, "l2", bounds=[0.0, 0.01])
        # set_control_hyperparameter_value(fun_control, "l2", [0.0, 0.01])
        # Note (from the River documentation):
        # For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
        # Therefore, we set l1 bounds to 0.0:
        # modify_hyper_parameter_bounds(fun_control, "l1", bounds=[0.0, 0.0])
        # set_control_hyperparameter_value(fun_control, "l1", [0.0, 0.0])
        # modify_hyper_parameter_levels(fun_control, "optimizer", ["SGD"])
    elif coremodel == "AMFClassifier":
        add_core_model_to_fun_control(
            core_model=AMFClassifier, fun_control=fun_control, hyper_dict=RiverHyperDict, filename=None
        )
        set_control_hyperparameter_value(fun_control, "n_estimators", [2, 10])
        set_control_hyperparameter_value(fun_control, "step", [0.5, 2])
        dict = rhd.hyper_dict[coremodel]
    elif coremodel == "HoeffdingAdaptiveTreeClassifier":
        add_core_model_to_fun_control(
            core_model=HoeffdingAdaptiveTreeClassifier,
            fun_control=fun_control,
            hyper_dict=RiverHyperDict,
            filename=None,
        )
        dict = rhd.hyper_dict[coremodel]
    else:
        core_model = load_core_model_from_file(coremodel, dirname="userModel")
        dict = load_dict_from_file(coremodel, dirname="userModel")
        fun_control.update({"core_model": core_model})
        fun_control.update({"core_model_hyper_dict": dict})
        var_type = get_var_type(fun_control)
        var_name = get_var_name(fun_control)
        lower = get_bound_values(fun_control, "lower", as_list=False)
        upper = get_bound_values(fun_control, "upper", as_list=False)
        fun_control.update({"var_type": var_type, "var_name": var_name, "lower": lower, "upper": upper})

    for i, (key, value) in enumerate(dict.items()):
        if dict[key]["type"] == "int" or "core_model_parameter_type" == "bool":
            set_control_hyperparameter_value(
                fun_control, key, [int(lower_bound_entry[i].get()), int(upper_bound_entry[i].get())]
            )
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

    weights = np.array([-1, 1 / 1000, 1 / 1000]) * 10_000.0
    weight_coeff = 1.0

    fun_control.update(
        {
            "train": train,
            "oml_grace_period": oml_grace_period,
            "test": test,
            "n_samples": n_samples,
            "target_column": target_column,
            "prep_model": prep_model,
            "oml_grace_period": oml_grace_period,
            "weights": weights,
            "weight_coeff": weight_coeff,
            "metric_sklearn": accuracy_score,
            "horizon": int(horizon_entry.get()),
        }
    )

    print(gen_design_table(fun_control))

    design_control = design_control_init(
        init_size=int(init_size_entry.get()),
        repeats=1,
    )

    surrogate_control = surrogate_control_init(
        noise=True,
        n_theta=2,
        min_Lambda=1e-6,
        max_Lambda=10,
        log_level=50,
    )

    optimizer_control = optimizer_control_init()

    (
        SPOT_PKL_NAME,
        spot_tuner,
        fun_control,
        design_control,
        surrogate_control,
        optimizer_control,
    ) = run_spot_python_experiment(
        save_only=save_only,
        fun_control=fun_control,
        design_control=design_control,
        surrogate_control=surrogate_control,
        optimizer_control=optimizer_control,
        fun=HyperRiver(log_level=fun_control["log_level"]).fun_oml_horizon,
    )
    if SPOT_PKL_NAME is not None and save_only:
        print(f"\nExperiment successfully saved. Configuration saved as: {SPOT_PKL_NAME}")
    elif SPOT_PKL_NAME is not None and not save_only:
        print(f"\nExperiment successfully terminated. Result saved as: {SPOT_PKL_NAME}")
    else:
        print("\nExperiment failed. No result saved.")


def call_compare_tuned_default():
    if spot_tuner is not None and fun_control is not None:
        compare_tuned_default(spot_tuner, fun_control)


def call_parallel_plot():
    if spot_tuner is not None:
        parallel_plot(spot_tuner)


def call_contour_plot():
    if spot_tuner is not None:
        contour_plot(spot_tuner)


def call_importance_plot():
    if spot_tuner is not None:
        importance_plot(spot_tuner)


def call_progress_plot():
    if spot_tuner is not None:
        progress_plot(spot_tuner)


def update_hyperparams():
    global label, default_entry, lower_bound_entry, upper_bound_entry, factor_level_entry
    coremodel = core_model_combo.get()
    # if model is a key in rhd.hyper_dict set dict = rhd.hyper_dict[model]
    if coremodel in rhd.hyper_dict:
        dict = rhd.hyper_dict[coremodel]
    else:
        dict = load_dict_from_file(coremodel, dirname="userModel")
    n_keys = len(dict)
    # Create a list of labels and entries with the same length as the number of keys in the dictionary
    label = [None] * n_keys
    default_entry = [None] * n_keys
    lower_bound_entry = [None] * n_keys
    upper_bound_entry = [None] * n_keys
    factor_level_entry = [None] * n_keys
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
            # Create an entry with the default value as the default text
            default_entry[i] = tk.Entry(run_tab)
            default_entry[i].insert(0, dict[key]["default"])
            default_entry[i].grid(row=i + 3, column=3, sticky="W")
            default_entry[i].update()
            # add the lower bound values in column 2
            lower_bound_entry[i] = tk.Entry(run_tab)
            lower_bound_entry[i].insert(0, dict[key]["lower"])
            lower_bound_entry[i].grid(row=i + 3, column=4, sticky="W")
            # add the upper bound values in column 3
            upper_bound_entry[i] = tk.Entry(run_tab)
            upper_bound_entry[i].insert(0, dict[key]["upper"])
            upper_bound_entry[i].grid(row=i + 3, column=5, sticky="W")
            print(f"GUI: Insert hyperparam val: {key}, {lower_bound_entry[i].get()}, {upper_bound_entry[i].get()}")
        if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
            # Create a label with the key as text
            label[i] = tk.Label(run_tab, text=key)
            label[i].grid(row=i + 3, column=2, sticky="W")
            label[i].update()
            # Create an entry with the default value as the default text
            default_entry[i] = tk.Entry(run_tab)
            default_entry[i].insert(0, dict[key]["default"])
            default_entry[i].grid(row=i + 3, column=3, sticky="W")
            # add the lower bound values in column 2
            factor_level_entry[i] = tk.Entry(run_tab)
            # TODO: replace " " with ", " for the levels
            print(f"GUI: dict[key][levels]: {dict[key]['levels']}")
            factor_level_entry[i].insert(0, dict[key]["levels"])
            factor_level_entry[i].grid(row=i + 3, column=4, sticky="W")
            print(f"GUI: Key: {key}. Inserting control hyperparameter value: {factor_level_entry[i].get()}")


# Create the main application window
app = tk.Tk()
app.title("Spot River Hyperparameter Tuning GUI")

# generate a list of StringVar() objects of size n_keys
for i in range(n_keys):
    factor_level_entry.append(StringVar())
    print(f"factor_level_entry[{i}]: {factor_level_entry[i]}")

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(app)
# notebook.pack(fill='both', expand=True)

# Create and pack entry fields for the "Run" tab
run_tab = ttk.Frame(notebook)
notebook.add(run_tab, text="Binary classification")

# colummns 0+1: Data

data_label = tk.Label(run_tab, text="Data options:")
data_label.grid(row=0, column=0, sticky="W")

data_set_label = tk.Label(run_tab, text="Select data_set:")
data_set_label.grid(row=1, column=0, sticky="W")
data_set_values = [
    "Bananas",
    "CreditCard",
    "Elec2",
    "Higgs",
    "HTTP",
    "MaliciousURL",
    "Phishing",
    "SMSSpam",
    "SMTP",
    "TREC07"
]
# get all *.csv files in the data directory "userData" and append them to the list of data_set_values
data_set_values.extend([f for f in os.listdir("userData") if f.endswith(".csv") or f.endswith(".pkl")])
data_set_combo = ttk.Combobox(run_tab, values=data_set_values)
data_set_combo.set("Phishing")  # Default selection
data_set_combo.grid(row=1, column=1)

target_column_label = tk.Label(run_tab, text="target_column:")
target_column_label.grid(row=2, column=0, sticky="W")
target_column_entry = tk.Entry(run_tab)
target_column_entry.insert(0, "is_phishing")
target_column_entry.grid(row=2, column=1, sticky="W")


n_total_label = tk.Label(run_tab, text="n_total:")
n_total_label.grid(row=3, column=0, sticky="W")
n_total_entry = tk.Entry(run_tab)
n_total_entry.insert(0, "All")
n_total_entry.grid(row=3, column=1, sticky="W")

test_size_label = tk.Label(run_tab, text="test_size:")
test_size_label.grid(row=3, column=0, sticky="W")
test_size_entry = tk.Entry(run_tab)
test_size_entry.insert(0, "0.10")
test_size_entry.grid(row=3, column=1, sticky="W")

# columns 0+1: Experiment
experiment_label = tk.Label(run_tab, text="Experiment options:")
experiment_label.grid(row=4, column=0, sticky="W")

max_time_label = tk.Label(run_tab, text="MAX_TIME:")
max_time_label.grid(row=5, column=0, sticky="W")
max_time_entry = tk.Entry(run_tab)
max_time_entry.insert(0, "1")
max_time_entry.grid(row=5, column=1)

fun_evals_label = tk.Label(run_tab, text="FUN_EVALS:")
fun_evals_label.grid(row=6, column=0, sticky="W")
fun_evals_entry = tk.Entry(run_tab)
fun_evals_entry.insert(0, "inf")
fun_evals_entry.grid(row=6, column=1)

init_size_label = tk.Label(run_tab, text="INIT_SIZE:")
init_size_label.grid(row=7, column=0, sticky="W")
init_size_entry = tk.Entry(run_tab)
init_size_entry.insert(0, "3")
init_size_entry.grid(row=7, column=1)

prefix_label = tk.Label(run_tab, text="PREFIX:")
prefix_label.grid(row=8, column=0, sticky="W")
prefix_entry = tk.Entry(run_tab)
prefix_entry.insert(0, "00")
prefix_entry.grid(row=8, column=1)

noise_label = tk.Label(run_tab, text="NOISE:")
noise_label.grid(row=9, column=0, sticky="W")
noise_entry = tk.Entry(run_tab)
noise_entry.insert(0, "TRUE")
noise_entry.grid(row=9, column=1)

horizon_label = tk.Label(run_tab, text="horizon:")
horizon_label.grid(row=10, column=0, sticky="W")
horizon_entry = tk.Entry(run_tab)
horizon_entry.insert(0, "1")
horizon_entry.grid(row=10, column=1)

oml_grace_period_label = tk.Label(run_tab, text="oml_grace_period:")
oml_grace_period_label.grid(row=11, column=0, sticky="W")
oml_grace_period_entry = tk.Entry(run_tab)
oml_grace_period_entry.insert(0, "n_train")
oml_grace_period_entry.grid(row=11, column=1)


# colummns 2-5: Model
model_label = tk.Label(run_tab, text="Model options:")
model_label.grid(row=0, column=2, sticky="W")

model_label = tk.Label(run_tab, text="Default values:")
model_label.grid(row=0, column=3, sticky="W")

model_label = tk.Label(run_tab, text="Lower bounds:")
model_label.grid(row=0, column=4, sticky="W")

model_label = tk.Label(run_tab, text="Upper bounds:")
model_label.grid(row=0, column=5, sticky="W")

prep_model_label = tk.Label(run_tab, text="Select preprocessing model")
prep_model_label.grid(row=1, column=2, sticky="W")
prep_model_values = ["MinMaxScaler", "StandardScaler", "None"]
prep_model_combo = ttk.Combobox(run_tab, values=prep_model_values)
prep_model_combo.set("StandardScaler")  # Default selection
prep_model_combo.grid(row=1, column=3)

core_model_label = tk.Label(run_tab, text="Select core model")
core_model_label.grid(row=2, column=2, sticky="W")
core_model_values = ["AMFClassifier", "HoeffdingAdaptiveTreeClassifier", "LogisticRegression"]
for filename in os.listdir("userModel"):
    if filename.endswith(".json"):
        core_model_values.append(os.path.splitext(filename)[0])
core_model_combo = ttk.Combobox(run_tab, values=core_model_values, postcommand=update_hyperparams)
core_model_combo.set("LogisticRegression")  # Default selection
core_model_combo.bind("<<ComboboxSelected>>", update_hyperparams())
core_model_combo.grid(row=2, column=3)


update_hyperparams()

# column 6: Save and run button
save_button = ttk.Button(run_tab, text="Save Experiment", command=lambda: run_experiment(save_only=True))
save_button.grid(row=7, column=6, columnspan=2, sticky="E")
run_button = ttk.Button(run_tab, text="Run Experiment", command=run_experiment)
run_button.grid(row=8, column=6, columnspan=2, sticky="E")

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


# Create and pack the "Analysis" tab with a button to run the analysis
analysis_tab = ttk.Frame(notebook)
notebook.add(analysis_tab, text="Analysis")

notebook.pack()


# Add the Logo image in both tabs
logo_image = tk.PhotoImage(file="images/spotlogo.png")
logo_label = tk.Label(run_tab, image=logo_image)
logo_label.grid(row=0, column=8, rowspan=1, columnspan=1)

analysis_label = tk.Label(analysis_tab, text="Analysis options:")
analysis_label.grid(row=0, column=1, sticky="W")

progress_plot_button = ttk.Button(analysis_tab, text="Progress plot", command=call_progress_plot)
progress_plot_button.grid(row=1, column=1, columnspan=2, sticky="W")

compare_tuned_default_button = ttk.Button(
    analysis_tab, text="Compare tuned vs. default", command=call_compare_tuned_default
)
compare_tuned_default_button.grid(row=2, column=1, columnspan=2, sticky="W")

importance_plot_button = ttk.Button(analysis_tab, text="Importance plot", command=call_importance_plot)
importance_plot_button.grid(row=3, column=1, columnspan=2, sticky="W")

contour_plot_button = ttk.Button(analysis_tab, text="Contour plot", command=call_contour_plot)
contour_plot_button.grid(row=4, column=1, columnspan=2, sticky="W")

parallel_plot_button = ttk.Button(analysis_tab, text="Parallel plot (Browser)", command=call_parallel_plot)
parallel_plot_button.grid(row=5, column=1, columnspan=2, sticky="W")


analysis_logo_label = tk.Label(analysis_tab, image=logo_image)
analysis_logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

# regression_logo_label = tk.Label(regression_tab, image=logo_image)
# regression_logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

# Run the mainloop

app.mainloop()
