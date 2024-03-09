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
from tkinter import filedialog as fd
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
    destroy_entries
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
from spotPython.utils.file import load_dict_from_file, load_core_model_from_file
from spotPython.utils.file import load_experiment as load_experiment_spot

core_model_values = [
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


def run_experiment(save_only=False, show_data_only=False):
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

    target_column = target_column_entry.get()
    test_size = float(test_size_entry.get())

    # metrics
    metric_name = metric_combo.get()
    metric_sklearn = getattr(sklearn.metrics, metric_name)
    weight_sgn = get_metric_sign(metric_name)
    metric_weights = metric_weights_entry.get()
    print(f"metric_weights = {metric_weights}")
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

    data_set = data_set_combo.get()
    # if the user has not specified a data set, take the default data set:
    # this can be one of the following:
    river_datasets = ["Bananas", "CreditCard", "Elec2", "Higgs", "HTTP", "Phishing"]
    if data_set in river_datasets:
        dataset, n_samples = data_selector(
            data_set=data_set,
        )
        # convert the river datasets to a pandas DataFrame, the target column
        # of the resulting DataFrame is target_column
        df = convert_to_df(dataset, target_column=target_column, n_total=n_total)
    # data_set ends with ".csv" or data_set ends with ".pkl":
    elif data_set.endswith(".csv"):
        df = CSVDataset(filename=data_set, directory="./userData/").data
        n_samples = df.shape[0]
    # Rename the columns of a DataFrame to x1, x2, ..., xn, y.
    # From now on we assume that the target column is called "y" and
    # is of type int (binary classification)
    df = rename_df_to_xy(df=df, target_column="y")
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
        X, Y, test_size=test_size, random_state=42
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

    # Initialize the fun_control dictionary with the static parameters,
    # i.e., the parameters that are not hyperparameters (depending on the core model)
    fun_control = fun_control_init(
        PREFIX=prefix_entry.get(),
        TENSORBOARD_CLEAN=TENSORBOARD_CLEAN,
        fun_evals=fun_evals_val,
        fun_repeats=1,
        horizon=int(horizon_entry.get()),
        max_time=float(max_time_entry.get()),
        metric_sklearn=metric_sklearn,
        noise=bool(noise_entry.get()),
        n_samples=n_samples,
        ocba_delta=0,
        oml_grace_period=oml_grace_period,
        prep_model=prep_model,
        target_column=target_column,
        test=test,
        test_size=test_size,
        train=train,
        tolerance_x=np.sqrt(np.spacing(1)),
        verbosity=1,
        weights=weights,
        log_level=50,
    )

    core_model = core_model_combo.get()
    core_model_module = core_model.split(".")[0]
    coremodel = core_model.split(".")[1]
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
    #  Enable user specific core models:
    # if core_model is not in core_model_values:
    #     core_model = load_core_model_from_file(coremodel, dirname="userModel")
    #     dict = load_dict_from_file(coremodel, dirname="userModel")
    #     fun_control.update({"core_model": core_model})
    #     fun_control.update({"core_model_hyper_dict": dict})
    #     var_type = get_var_type(fun_control)
    #     var_name = get_var_name(fun_control)
    #     lower = get_bound_values(fun_control, "lower", as_list=False)
    #     upper = get_bound_values(fun_control, "upper", as_list=False)
    #     fun_control.update({"var_type": var_type, "var_name": var_name, "lower": lower, "upper": upper})

    for i, (key, value) in enumerate(dict.items()):
        if dict[key]["type"] == "int":
            set_control_hyperparameter_value(
                fun_control, key, [int(lower_bound_entry[i].get()), int(upper_bound_entry[i].get())]
            )
        if (dict[key]["type"] == "factor") and (dict[key]["core_model_parameter_type"] == "bool"):
            fun_control["core_model_hyper_dict"][key].update({"lower": int(lower_bound_entry[i].get())})
            fun_control["core_model_hyper_dict"][key].update({"upper": int(upper_bound_entry[i].get())})
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

    surrogate_control = surrogate_control_init(
        noise=True,
        n_theta=2,
        min_Lambda=1e-6,
        max_Lambda=10,
        log_level=50,
    )

    optimizer_control = optimizer_control_init()

    print(gen_design_table(fun_control))

    (
        SPOT_PKL_NAME,
        spot_tuner,
        fun_control,
        design_control,
        surrogate_control,
        optimizer_control,
    ) = run_spot_python_experiment(
        save_only=save_only,
        show_data_only=show_data_only,
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
    elif show_data_only:
        print("\nData shown. No result saved.")
    else:
        print("\nExperiment failed. No result saved.")


def load_experiment():
    global label, default_entry, lower_bound_entry, upper_bound_entry, transform_entry, factor_level_entry, menu, choices, select, selectValue
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filetypes = (("Pickle files", "*.pickle"), ("All files", "*.*"))
    filename = fd.askopenfilename(title="Select a Pickle File", initialdir=current_dir, filetypes=filetypes)
    if filename:
        spot_tuner, fun_control, design_control, surrogate_control, optimizer_control = load_experiment_spot(filename)

        # TODO spottuner = -> laden aus der Pickle datei. Damit dann analysis nachträglich gestartet werden kann
        data_set_combo.delete(0, tk.END)
        # target_column_entry.delete(0, tk.END)
        data_set_name = fun_control["data_set"].__class__.__name__
        print(f"\ndata_set_name: {data_set_name}\n")

        if data_set_name == "CSVDataset" or data_set_name == "PKLDataset":
            # target_column_entry.insert(0, str(vars(fun_control["data_set"])["target_column"]))
            filename = vars(fun_control["data_set"])["filename"]
            print("filename: ", filename)
            # TODO nicht neuen EIntrag hginzufügen sondern einen asuwählen. Ist sicherlich anders. Soinst müssten einträge doppelt sein.
            data_set_combo.set(filename)
        else:
            # target_column_entry.insert(0, "target")
            data_set_combo.set(data_set_name)

        # static parameters, that are not hyperparameters (depending on the core model)

        target_column_entry.delete(0, tk.END)
        target_column_entry.insert(0, str(fun_control["target_column"]))

        n_total_entry.delete(0, tk.END)
        n_total_entry.insert(0, str(fun_control["n_total"]))

        test_size_entry.delete(0, tk.END)
        test_size_entry.insert(0, str(fun_control["test_size"]))

        # TODO spottuner = -> laden aus der Pickle datei. Damit dann analysis nachträglich gestartet werden kann
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

        metric_combo.delete(0, tk.END)
        metric_name = fun_control["prep_model"].__class__.__name__
        metric_combo.set(metric_name)

        metric_weights_entry.delete(0, tk.END)
        metric_weights_entry.insert(0, str(fun_control["weights"]))

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
        # hier direkt über name zugreifen, da kein Objekt, sondern eine Klasse übergeben wird
        print(f"Core model: {fun_control['core_model']}")
        print(f"Core model.__class__: {fun_control['core_model'].__class__}")
        print(f"Core model.__name__: {fun_control['core_model'].__name__}")
        core_model_combo.set(fun_control["core_model"].__name__)

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
            label[i].grid(row=i + 2, column=2, sticky="W")
            label[i].update()
            # Create an entry with the default value as the default text
            default_entry[i] = tk.Entry(run_tab)
            default_entry[i].insert(0, dict[key]["default"])
            default_entry[i].grid(row=i + 2, column=3, sticky="W")
            default_entry[i].update()
            # add the lower bound values in column 4
            lower_bound_entry[i] = tk.Entry(run_tab)
            lower_bound_entry[i].insert(0, dict[key]["lower"])
            lower_bound_entry[i].grid(row=i + 2, column=4, sticky="W")
            # add the upper bound values in column 5
            upper_bound_entry[i] = tk.Entry(run_tab)
            upper_bound_entry[i].insert(0, dict[key]["upper"])
            upper_bound_entry[i].grid(row=i + 2, column=5, sticky="W")
            # add the transformation values in column 6
            transform_entry[i] = tk.Entry(run_tab)
            transform_entry[i].insert(0, dict[key]["transform"])
            transform_entry[i].grid(row=i + 2, column=6, sticky="W")

        if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
            # Create a label with the key as text
            label[i] = tk.Label(run_tab, text=key)
            label[i].grid(row=i + 2, column=2, sticky="W")
            label[i].update()
            # Create an entry with the default value as the default text
            default_entry[i] = tk.Entry(run_tab)
            default_entry[i].insert(0, dict[key]["default"])
            default_entry[i].grid(row=i + 2, column=3, sticky="W")
            # add the lower bound values in column 2
            factor_level_entry[i] = tk.Entry(run_tab)
            # TODO: replace " " with ", " for the levels
            factor_level_entry[i].insert(0, dict[key]["levels"])
            factor_level_entry[i].grid(row=i + 2, column=4, columnspan=2, sticky=tk.W + tk.E)


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

data_label = tk.Label(run_tab, text="Data options:")
data_label.grid(row=0, column=0, sticky="W")

data_set_label = tk.Label(run_tab, text="Select data_set:")
data_set_label.grid(row=1, column=0, sticky="W")
data_set_values = ["Bananas", "CreditCard", "Elec2", "Higgs", "HTTP", "Phishing"]
# get all *.csv files in the data directory "userData" and append them to the list of data_set_values
data_set_values.extend([f for f in os.listdir("userData") if f.endswith(".csv") or f.endswith(".pkl")])
data_set_combo = ttk.Combobox(run_tab, values=data_set_values)
data_set_combo.set("Phishing")  # Default selection
data_set_combo.grid(row=1, column=1)

target_column_label = tk.Label(run_tab, text="target_column (str):")
target_column_label.grid(row=2, column=0, sticky="W")
target_column_entry = tk.Entry(run_tab)
target_column_entry.insert(0, "y")
target_column_entry.grid(row=2, column=1, sticky="W")

n_total_label = tk.Label(run_tab, text="n_total (int|All):")
n_total_label.grid(row=3, column=0, sticky="W")
n_total_entry = tk.Entry(run_tab)
n_total_entry.insert(0, "All")
n_total_entry.grid(row=3, column=1, sticky="W")

test_size_label = tk.Label(run_tab, text="test_size (perc.):")
test_size_label.grid(row=4, column=0, sticky="W")
test_size_entry = tk.Entry(run_tab)
test_size_entry.insert(0, "0.30")
test_size_entry.grid(row=4, column=1, sticky="W")

prep_model_label = tk.Label(run_tab, text="Select preprocessing model")
prep_model_label.grid(row=5, column=0, sticky="W")
prep_model_values = ["MinMaxScaler", "StandardScaler", "None"]
prep_model_combo = ttk.Combobox(run_tab, values=prep_model_values)
prep_model_combo.set("StandardScaler")  # Default selection
prep_model_combo.grid(row=5, column=1)

# columns 0+1: Experiment
experiment_label = tk.Label(run_tab, text="Experiment options:")
experiment_label.grid(row=6, column=0, sticky="W")

max_time_label = tk.Label(run_tab, text="MAX_TIME (min):")
max_time_label.grid(row=7, column=0, sticky="W")
max_time_entry = tk.Entry(run_tab)
max_time_entry.insert(0, "1")
max_time_entry.grid(row=7, column=1)

fun_evals_label = tk.Label(run_tab, text="FUN_EVALS (int|inf):")
fun_evals_label.grid(row=8, column=0, sticky="W")
fun_evals_entry = tk.Entry(run_tab)
fun_evals_entry.insert(0, "30")
fun_evals_entry.grid(row=8, column=1)

init_size_label = tk.Label(run_tab, text="INIT_SIZE (int):")
init_size_label.grid(row=9, column=0, sticky="W")
init_size_entry = tk.Entry(run_tab)
init_size_entry.insert(0, "5")
init_size_entry.grid(row=9, column=1)

prefix_label = tk.Label(run_tab, text="PREFIX (str):")
prefix_label.grid(row=10, column=0, sticky="W")
prefix_entry = tk.Entry(run_tab)
prefix_entry.insert(0, "00")
prefix_entry.grid(row=10, column=1)

noise_label = tk.Label(run_tab, text="NOISE (bool):")
noise_label.grid(row=11, column=0, sticky="W")
noise_entry = tk.Entry(run_tab)
noise_entry.insert(0, "TRUE")
noise_entry.grid(row=11, column=1)

# columns 0+1: Evaluation
experiment_label = tk.Label(run_tab, text="Evaluation options:")
experiment_label.grid(row=12, column=0, sticky="W")


metric_label = tk.Label(run_tab, text="metric (sklearn):")
metric_label.grid(row=13, column=0, sticky="W")
metric_combo = ttk.Combobox(run_tab, values=metric_levels)
metric_combo.set("accuracy_score")  # Default selection
metric_combo.grid(row=13, column=1)

metric_weights_label = tk.Label(run_tab, text="weights: y,time,mem (>0.0):")
metric_weights_label.grid(row=14, column=0, sticky="W")
metric_weights_entry = tk.Entry(run_tab)
metric_weights_entry.insert(0, "1000, 1, 1")
metric_weights_entry.grid(row=14, column=1)

horizon_label = tk.Label(run_tab, text="horizon (int):")
horizon_label.grid(row=15, column=0, sticky="W")
horizon_entry = tk.Entry(run_tab)
horizon_entry.insert(0, "10")
horizon_entry.grid(row=15, column=1)

oml_grace_period_label = tk.Label(run_tab, text="oml_grace_period (int|None):")
oml_grace_period_label.grid(row=16, column=0, sticky="W")
oml_grace_period_entry = tk.Entry(run_tab)
oml_grace_period_entry.insert(0, "None")
oml_grace_period_entry.grid(row=16, column=1)


# colummns 2-6: Model
model_label = tk.Label(run_tab, text="Model options:")
model_label.grid(row=0, column=2, sticky="W")

model_label = tk.Label(run_tab, text="Default values:")
model_label.grid(row=0, column=3, sticky="W")

model_label = tk.Label(run_tab, text="Lower bounds:")
model_label.grid(row=0, column=4, sticky="W")

model_label = tk.Label(run_tab, text="Upper bounds:")
model_label.grid(row=0, column=5, sticky="W")

model_label = tk.Label(run_tab, text="Transformation:")
model_label.grid(row=0, column=6, sticky="W")

core_model_label = tk.Label(run_tab, text="Core model")
core_model_label.grid(row=1, column=2, sticky="W")
for filename in os.listdir("userModel"):
    if filename.endswith(".json"):
        core_model_values.append(os.path.splitext(filename)[0])
core_model_combo = ttk.Combobox(run_tab, values=core_model_values)
core_model_combo.set("Select Model")  # Default selection
core_model_combo.bind("<<ComboboxSelected>>", update_hyperparams)
core_model_combo.grid(row=1, column=3)


# column 8: Save and run button
tb_clean = tk.BooleanVar()
tb_clean.set(True)
tf_clean_checkbutton = tk.Checkbutton(run_tab, text="TENSORBOARD_CLEAN", variable=tb_clean)
tf_clean_checkbutton.grid(row=2, column=8, sticky="E")
data_button = ttk.Button(run_tab, text="Show Data", command=lambda: run_experiment(show_data_only=True))
data_button.grid(row=7, column=8, columnspan=2, sticky="E")
save_button = ttk.Button(run_tab, text="Save Experiment", command=lambda: run_experiment(save_only=True))
save_button.grid(row=8, column=8, columnspan=2, sticky="E")
run_button = ttk.Button(run_tab, text="Run Experiment", command=run_experiment)
run_button.grid(row=9, column=8, columnspan=2, sticky="E")
load_button = ttk.Button(run_tab, text="Load Experiment", command=load_experiment)
load_button.grid(row=10, column=8, columnspan=2, sticky="E")

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
