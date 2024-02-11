import numpy as np
import tkinter as tk
from tkinter import ttk, StringVar
import math
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.utils.init import fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.hyperparameters.values import set_control_hyperparameter_value
from spotGUI.tuner.spotRun import (
    run_spot_python_experiment,
    contour_plot,
    parallel_plot,
    importance_plot,
    progress_plot,
)
from spotPython.light.regression.netlightregression import NetLightRegression
from spotPython.light.regression.netlightregression2 import NetLightRegression2
from spotPython.light.regression.transformerlightregression import TransformerLightRegression
from spotPython.utils.eda import gen_design_table
from spotPython.data.diabetes import Diabetes
import torch

spot_tuner = None
# Create a LightHyperDict object
lhd = LightHyperDict()
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
    perc_train = float(perc_train_entry.get())

    fun_evals = fun_evals_entry.get()
    if fun_evals == "None" or fun_evals == "inf":
        fun_evals_val = math.inf
    else:
        fun_evals_val = int(fun_evals)
    
    data_set = data_set_combo.get()
    if data_set == "Diabetes":
        dataset = Diabetes(feature_type=torch.float32, target_type=torch.float32)
        print(f"Diabetes data set: {len(dataset)}")
    else:
        print(f"Error: Data set {data_set} not available.")        
        return

    # Initialize the fun_control dictionary with the static parameters,
    # i.e., the parameters that are not hyperparameters (depending on the core model)
    fun_control = fun_control_init(
        _L_in=int(lin_entry.get()),
        _L_out=int(lout_entry.get()),
        PREFIX = prefix_entry.get(),
        TENSORBOARD_CLEAN=True,
        fun_evals=fun_evals_val,
        fun_repeats=1,
        max_time = float(max_time_entry.get()),
        noise = bool(noise_entry.get()),
        ocba_delta=0,
        test_size=0.4,
        tolerance_x=np.sqrt(np.spacing(1)),
        verbosity=1,
        data_set = dataset,
        log_level =10,
    )
    
    # Get the selected core model and add it to the fun_control dictionary
    coremodel = core_model_combo.get()
    if coremodel == "NetLightRegression2":
        add_core_model_to_fun_control(
            fun_control=fun_control, core_model=NetLightRegression2, hyper_dict=LightHyperDict
        )
        print(gen_design_table(fun_control))
    elif coremodel == "NetLightRegression":
        add_core_model_to_fun_control(fun_control=fun_control, core_model=NetLightRegression,
           hyper_dict=LightHyperDict)
        print(gen_design_table(fun_control))
    elif coremodel == "TransformerLightRegression":
        add_core_model_to_fun_control(
            fun_control=fun_control, core_model=TransformerLightRegression, hyper_dict=LightHyperDict
        )

    # update the keys and values in the fun_control dictionary with the keys and values in the dict dictionary
    dict = lhd.hyper_dict[coremodel]
    n_keys = len(dict)
    print(f"n_keys in the dictionary: {n_keys}")    
    for i, (key, value) in enumerate(dict.items()):
        if dict[key]["type"] == "int":
            print(f"fun_control: Setting control hyperparameter value: {key}, {lower_bound_entry[i].get()}, {upper_bound_entry[i].get()}")
            set_control_hyperparameter_value(fun_control, key, [int(lower_bound_entry[i].get()), int(upper_bound_entry[i].get())])
        if dict[key]["type"] == "float":
            print(f"fun_control: Setting control hyperparameter value: {key}, {lower_bound_entry[i].get()}, {upper_bound_entry[i].get()}")
            set_control_hyperparameter_value(fun_control, key, [float(lower_bound_entry[i].get()), float(upper_bound_entry[i].get())])
        if dict[key]["type"] == "factor":
            print(f"fun_control: getting control hyperparameter value: {key}, {factor_level_entry[i].get()}")
            fle = factor_level_entry[i].get()
            # convert the string to a list of strings
            fle = fle.split()
            print(f"fun_control: Key {key}: setting control hyperparameter value: {fle}")
            set_control_hyperparameter_value(fun_control, key,fle)
            # fun_control["core_model"][key].update({"upper": len(fle)})
            # print the values from 'core_model_hyper_dict' in the fun_control dictionary
            print("\n****\nfun_control['core_model_hyper_dict'][key] in run_experiment():", fun_control['core_model_hyper_dict'][key])
            fun_control['core_model_hyper_dict'][key].update({"upper": len(fle) - 1})
            print("\n****\nfun_control['core_model_hyper_dict'][key] in run_experiment():", fun_control['core_model_hyper_dict'][key])

    print("\nfun_control in run_experiment():", fun_control)
    # print the values from 'core_model_hyper_dict' in the fun_control dictionary
    # print("\nfun_control['core_model_hyper_dict'] in run_experiment():", fun_control['core_model_hyper_dict'])

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


    SPOT_PKL_NAME, spot_tuner, fun_control, design_control, surrogate_control, optimizer_control = run_spot_python_experiment(
        save_only=save_only,
        fun_control=fun_control,
        design_control=design_control,
        surrogate_control=surrogate_control,
        optimizer_control=optimizer_control,
    )


def call_parallel_plot():
    if spot_tuner is not None:
        parallel_plot(spot_tuner)


def call_contour_plot():
    if spot_tuner is not None:
        contour_plot(spot_tuner)


def call_progress_plot():
    if spot_tuner is not None:
        progress_plot(spot_tuner)


def call_importance_plot():
    if spot_tuner is not None:
        importance_plot(spot_tuner)


def update_hyperparams():
    global label, default_entry, lower_bound_entry, upper_bound_entry, factor_level_entry
    model = core_model_combo.get()
    dict = lhd.hyper_dict[model]
    n_keys = len(dict)
    print(f"n_keys in the dictionary: {n_keys}")
    # Create a list of labels and entries with the same length as the number of keys in the dictionary
    label = [None] * n_keys
    default_entry = [None] * n_keys
    lower_bound_entry = [None] * n_keys
    upper_bound_entry = [None] * n_keys
    factor_level_entry = [None] * n_keys
    for i, (key, value) in enumerate(dict.items()):
        if dict[key]["type"] == "int" or dict[key]["type"] == "float":
            # Create a label with the key as text
            label[i] = tk.Label(run_tab, text=key)
            label[i].grid(row=i + 2, column=2, sticky="W")
            label[i].update()
            # Create an entry with the default value as the default text
            default_entry[i] = tk.Entry(run_tab)
            default_entry[i].insert(0, dict[key]["default"])
            default_entry[i].grid(row=i + 2, column=3, sticky="W")
            default_entry[i].update()
            # add the lower bound values in column 2
            lower_bound_entry[i] = tk.Entry(run_tab)
            lower_bound_entry[i].insert(0, dict[key]["lower"])
            lower_bound_entry[i].grid(row=i + 2, column=4, sticky="W")
            # add the upper bound values in column 3
            upper_bound_entry[i] = tk.Entry(run_tab)
            upper_bound_entry[i].insert(0, dict[key]["upper"])
            upper_bound_entry[i].grid(row=i + 2, column=5, sticky="W")
            print(f"GUI: Inserting control hyperparameter value: {key}, {lower_bound_entry[i].get()}, {upper_bound_entry[i].get()}")
        if dict[key]["type"] == "factor":
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
            print(f"GUI: dict[key][levels]: {dict[key]['levels']}")
            factor_level_entry[i].insert(0, dict[key]["levels"])
            factor_level_entry[i].grid(row=i + 2, column=4, sticky="W")
            print(f"GUI: Key: {key}. Inserting control hyperparameter value: {factor_level_entry[i].get()}")


# Create the main application window
app = tk.Tk()
app.title("Spot Python Hyperparameter Tuning GUI")

# generate a list of StringVar() objects of size n_keys
for i in range(n_keys):
    factor_level_entry.append(StringVar())
    print(f"factor_level_entry[{i}]: {factor_level_entry[i]}")


# Create a notebook (tabbed interface)
notebook = ttk.Notebook(app)
# notebook.pack(fill='both', expand=True)

# Create and pack entry fields for the "Run" tab
run_tab = ttk.Frame(notebook)
notebook.add(run_tab, text="Pytorch Lightning")

# colummns 0+1: Data

data_label = tk.Label(run_tab, text="Data options:")
data_label.grid(row=0, column=0, sticky="W")

data_set_label = tk.Label(run_tab, text="Select data_set:")
data_set_label.grid(row=1, column=0, sticky="W")
data_set_values = [
    "Diabetes",
    "USER",
]
data_set_combo = ttk.Combobox(run_tab, values=data_set_values)
data_set_combo.set("Diabetes")  # Default selection
data_set_combo.grid(row=1, column=1)


n_total_label = tk.Label(run_tab, text="n_total:")
n_total_label.grid(row=2, column=0, sticky="W")
n_total_entry = tk.Entry(run_tab)
n_total_entry.insert(0, "All")
n_total_entry.grid(row=2, column=1, sticky="W")

perc_train_label = tk.Label(run_tab, text="perc_train:")
perc_train_label.grid(row=3, column=0, sticky="W")
perc_train_entry = tk.Entry(run_tab)
perc_train_entry.insert(0, "0.90")
perc_train_entry.grid(row=3, column=1, sticky="W")

lin_label = tk.Label(run_tab, text="_L_in:")
lin_label.grid(row=4, column=0, sticky="W")
lin_entry = tk.Entry(run_tab)
lin_entry.insert(0, "10")
lin_entry.grid(row=4, column=1, sticky="W")

lout_label = tk.Label(run_tab, text="_L_out:")
lout_label.grid(row=5, column=0, sticky="W")
lout_entry = tk.Entry(run_tab)
lout_entry.insert(0, "1")
lout_entry.grid(row=5, column=1, sticky="W")


# colummns 2+3: Model
model_label = tk.Label(run_tab, text="Model options:")
model_label.grid(row=0, column=2, sticky="W")
model_label = tk.Label(run_tab, text="Default values:")
model_label.grid(row=0, column=3, sticky="W")
model_label = tk.Label(run_tab, text="Lower bounds:")
model_label.grid(row=0, column=4, sticky="W")
model_label = tk.Label(run_tab, text="Upper bounds:")
model_label.grid(row=0, column=5, sticky="W")
core_model_label = tk.Label(run_tab, text="Select core model")
core_model_label.grid(row=1, column=2, sticky="W")
core_model_values = ["NetLightRegression", "NetLightRegression2", "TransformerLightRegression"]
core_model_combo = ttk.Combobox(run_tab, values=core_model_values, postcommand=update_hyperparams)
core_model_combo.set("NetLightRegression")  # Default selection
core_model_combo.bind("<<ComboboxSelected>>", update_hyperparams())
core_model_combo.grid(row=1, column=3)

update_hyperparams()
print(f"\ndict after update: {dict}\n")


# columns 4+5: Experiment
experiment_label = tk.Label(run_tab, text="Experiment options:")
experiment_label.grid(row=0, column=6, sticky="W")

max_time_label = tk.Label(run_tab, text="MAX_TIME:")
max_time_label.grid(row=1, column=6, sticky="W")
max_time_entry = tk.Entry(run_tab)
max_time_entry.insert(0, "1")
max_time_entry.grid(row=1, column=7)

fun_evals_label = tk.Label(run_tab, text="FUN_EVALS:")
fun_evals_label.grid(row=2, column=6, sticky="W")
fun_evals_entry = tk.Entry(run_tab)
fun_evals_entry.insert(0, "inf")
fun_evals_entry.grid(row=2, column=7)

init_size_label = tk.Label(run_tab, text="INIT_SIZE:")
init_size_label.grid(row=3, column=6, sticky="W")
init_size_entry = tk.Entry(run_tab)
init_size_entry.insert(0, "6")
init_size_entry.grid(row=3, column=7)

prefix_label = tk.Label(run_tab, text="PREFIX:")
prefix_label.grid(row=4, column=6, sticky="W")
prefix_entry = tk.Entry(run_tab)
prefix_entry.insert(0, "SPOT_0000")
prefix_entry.grid(row=4, column=7)

noise_label = tk.Label(run_tab, text="NOISE:")
noise_label.grid(row=5, column=6, sticky="W")
noise_entry = tk.Entry(run_tab)
noise_entry.insert(0, "TRUE")
noise_entry.grid(row=5, column=7)


# column 8: Save and run button
save_button = ttk.Button(run_tab, text="Save Experiment", command=lambda: run_experiment(save_only=True))
save_button.grid(row=7, column=8, columnspan=2, sticky="E")
run_button = ttk.Button(run_tab, text="Run Experiment", command=run_experiment)
run_button.grid(row=8, column=8, columnspan=2, sticky="E")


# Create and pack the "Regression" tab with a button to run the analysis
river_tab = ttk.Frame(notebook)
notebook.add(river_tab, text="River")

# colummns 0+1: Data

river_data_label = tk.Label(river_tab, text="Data options:")
river_data_label.grid(row=0, column=0, sticky="W")

# colummns 2+3: Model
river_model_label = tk.Label(river_tab, text="Model options:")
river_model_label.grid(row=0, column=2, sticky="W")

# columns 4+5: Experiment
river_experiment_label = tk.Label(river_tab, text="Experiment options:")
river_experiment_label.grid(row=0, column=4, sticky="W")


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

importance_plot_button = ttk.Button(analysis_tab, text="Importance plot", command=call_importance_plot)
importance_plot_button.grid(row=3, column=1, columnspan=2, sticky="W")

contour_plot_button = ttk.Button(analysis_tab, text="Contour plot", command=call_contour_plot)
contour_plot_button.grid(row=4, column=1, columnspan=2, sticky="W")

parallel_plot_button = ttk.Button(analysis_tab, text="Parallel plot (Browser)", command=call_parallel_plot)
parallel_plot_button.grid(row=5, column=1, columnspan=2, sticky="W")


analysis_logo_label = tk.Label(analysis_tab, image=logo_image)
analysis_logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

river_logo_label = tk.Label(river_tab, image=logo_image)
river_logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

# Run the mainloop

app.mainloop()