import os
import numpy as np
import tkinter as tk
from tkinter import ttk, StringVar
import math
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
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
)
from spotPython.light.regression.netlightregression import NetLightRegression
from spotPython.light.regression.netlightregression2 import NetLightRegression2
from spotPython.light.regression.transformerlightregression import TransformerLightRegression
from spotPython.utils.eda import gen_design_table
from spotPython.data.diabetes import Diabetes
import torch
from spotPython.data.csvdataset import CSVDataset
from spotPython.data.pkldataset import PKLDataset
from spotPython.utils.file import load_dict_from_file, load_core_model_from_file

spot_tuner = None
lhd = LightHyperDict()
#
n_keys = 25
label = [None] * n_keys
default_entry = [None] * n_keys
lower_bound_entry = [None] * n_keys
upper_bound_entry = [None] * n_keys
factor_level_entry = [None] * n_keys
transform_entry = [None] * n_keys

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

    # check if data_set provided by spotPython as a DataSet object
    data_set = data_set_combo.get()
    if data_set == "Diabetes":
        dataset = Diabetes(feature_type=torch.float32, target_type=torch.float32)
        print(f"Diabetes data set: {len(dataset)}")
    # check if data_set is available as .csv
    elif data_set.endswith(".csv"):
        dataset = CSVDataset(
            directory="./userData/",
            filename=data_set,
            target_column=target_column_entry.get(),
            feature_type=getattr(torch, feature_type_entry.get()),
            target_type=getattr(torch, target_type_entry.get()),
        )
        print(len(dataset))
    #  check if data_set is available as .pkl
    elif data_set.endswith(".pkl"):
        dataset = PKLDataset(
            directory="./userData/",
            filename=data_set,
            target_column=target_column_entry.get(),
            feature_type=getattr(torch, feature_type_entry.get()),
            target_type=getattr(torch, target_type_entry.get()),
        )
        print(len(dataset))
    else:
        print(f"Error: Data set {data_set} not available.")
        return

    # Initialize the fun_control dictionary with the static parameters,
    # i.e., the parameters that are not hyperparameters (depending on the core model)
    fun_control = fun_control_init(
        _L_in=int(lin_entry.get()),
        _L_out=int(lout_entry.get()),
        PREFIX=prefix_entry.get(),
        TENSORBOARD_CLEAN=True,
        fun_evals=fun_evals_val,
        fun_repeats=1,
        max_time=float(max_time_entry.get()),
        noise=bool(noise_entry.get()),
        ocba_delta=0,
        data_set=dataset,
        test_size=float(test_size_entry.get()),
        tolerance_x=np.sqrt(np.spacing(1)),
        verbosity=1,
        log_level=50,
    )

    # Get the selected core model and add it to the fun_control dictionary
    coremodel = core_model_combo.get()
    if coremodel == "NetLightRegression2":
        add_core_model_to_fun_control(
            fun_control=fun_control, core_model=NetLightRegression2, hyper_dict=LightHyperDict
        )
        dict = lhd.hyper_dict[coremodel]
    elif coremodel == "NetLightRegression":
        add_core_model_to_fun_control(fun_control=fun_control, core_model=NetLightRegression, hyper_dict=LightHyperDict)
        dict = lhd.hyper_dict[coremodel]
    elif coremodel == "TransformerLightRegression":
        add_core_model_to_fun_control(
            fun_control=fun_control, core_model=TransformerLightRegression, hyper_dict=LightHyperDict
        )
        dict = lhd.hyper_dict[coremodel]
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
        if dict[key]["type"] == "int":
            set_control_hyperparameter_value(
                fun_control, key, [int(lower_bound_entry[i].get()), int(upper_bound_entry[i].get())]
            )
        if dict[key]["type"] == "float":
            set_control_hyperparameter_value(
                fun_control, key, [float(lower_bound_entry[i].get()), float(upper_bound_entry[i].get())]
            )
        if dict[key]["type"] == "factor":

            # load from combo box and add to empty list
            fle = []
            for name, var in choices[i].items():
                if(var.get() == 1):
                    fle.append(name)

            set_control_hyperparameter_value(fun_control, key, fle)
            fun_control["core_model_hyper_dict"][key].update({"upper": len(fle) - 1})

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
    )
    if SPOT_PKL_NAME is not None and save_only:
        print(f"\nExperiment successfully saved. Configuration saved as: {SPOT_PKL_NAME}")
    elif SPOT_PKL_NAME is not None and not save_only:
        print(f"\nExperiment successfully terminated. Result saved as: {SPOT_PKL_NAME}")
    else:
        print("\nExperiment failed. No result saved.")


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

def selection_changed(i):
    if factor_level_entry[i]["text"] == "":
        factor_level_entry[i]["text"] = "Select something"

def show_selection(choices, i):
    factor_level_entry[i]["text"] = ""
    first_element = True
    all_selected = True  # Flag to check if all choices are selected
    for j, (name, var) in enumerate(choices.items()):
        if var.get() == 1:
            if first_element:
                factor_level_entry[i]["text"] = name
                first_element = False
            else:
                factor_level_entry[i]["text"] += ", " + name
        else:
            all_selected = False  # Set flag to False if any choice is not selected
    if all_selected:
        selectValue[i].set(1)  # Check the checkbox if all choices are selected
    else:
        selectValue[i].set(0)  # Uncheck the checkbox if not all choices are selected
    selection_changed(i)

def selectAll(choices, i):
    if(selectValue[i].get() == 1):
        for name, var in choices.items():
            var.set(1)
    elif(selectValue[i].get() == 0):
        for name, var in choices.items():
            var.set(0)
    show_selection(choices, i)

def update_hyperparams(event):
    global label, default_entry, lower_bound_entry, upper_bound_entry, transform_entry, factor_level_entry, menu, choices, select, selectValue

    if label is not None:
        for i in range(len(label)):
            if label[i] is not None:
                label[i].destroy()

    if default_entry is not None:
        for i in range(len(default_entry)):
            if default_entry[i] is not None:
                default_entry[i].destroy()

    if lower_bound_entry is not None:
        for i in range(len(lower_bound_entry)):
            if lower_bound_entry[i] is not None:
                lower_bound_entry[i].destroy()

    if upper_bound_entry is not None:
        for i in range(len(upper_bound_entry)):
            if upper_bound_entry[i] is not None:
                upper_bound_entry[i].destroy()

    if transform_entry is not None:
        for i in range(len(transform_entry)):
            if transform_entry[i] is not None:
                transform_entry[i].destroy()

    if factor_level_entry is not None:
        for i in range(len(factor_level_entry)):
            if factor_level_entry[i] is not None and not isinstance(factor_level_entry[i], StringVar):
                factor_level_entry[i].destroy()
                
    coremodel = core_model_combo.get()
    # if model is a key in lhd.hyper_dict set dict = lhd.hyper_dict[model]
    if coremodel in lhd.hyper_dict:
        dict = lhd.hyper_dict[coremodel]
    else:
        dict = load_dict_from_file(coremodel, dirname="userModel")
    n_keys = len(dict)
    # Create a list of labels and entries with the same length as the number of keys in the dictionary
    label = [None] * n_keys
    default_entry = [None] * n_keys
    lower_bound_entry = [None] * n_keys
    upper_bound_entry = [None] * n_keys
    transform_entry = [None] * n_keys
    factor_level_entry = [None] * n_keys
    choices = [None] * n_keys
    menu = [None] * n_keys
    selectValue = [tk.IntVar(value=0) for _ in range(n_keys)]
    select = [None] * n_keys
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
            print(f"GUI: Insert hyperparam val: {key}, {lower_bound_entry[i].get()}, {upper_bound_entry[i].get()}")
            # add the transformation values in column 6
            transform_entry[i] = tk.Entry(run_tab)
            transform_entry[i].insert(0, dict[key]["transform"])
            transform_entry[i].grid(row=i + 2, column=6, sticky="W")
        if dict[key]["type"] == "factor":
            # Create a label with the key as text
            label[i] = tk.Label(run_tab, text=key)
            label[i].grid(row=i + 2, column=2, sticky="W")
            label[i].update()
            # Create an entry with the default value as the default text
            default_entry[i] = tk.Entry(run_tab)
            default_entry[i].insert(0, dict[key]["default"])
            default_entry[i].grid(row=i + 2, column=3, sticky="W")

            # Factor_Levels
            factor_level_entry[i]= tk.Menubutton(run_tab, text="Select something", indicatoron=True, borderwidth=1, relief="raised")
            menu[i] = tk.Menu(factor_level_entry[i], tearoff=False )
            factor_level_entry[i].configure(menu=menu[i])
            factor_level_entry[i].grid(row=i + 2, column=4, columnspan=2, sticky=tk.W + tk.E)

            choices[i] = {}
            for choice in dict[key]["levels"]:
                choices[i][choice] = tk.IntVar(value=0)
                menu[i].add_checkbutton(label=choice, variable=choices[i][choice], 
                                    onvalue=1, offvalue=0, command=lambda i=i, choices=choices[i]: show_selection(choices, i))

            select[i] = tk.Checkbutton(run_tab,text="Select all", variable=selectValue[i], onvalue=1, offvalue=0, command=lambda i=i, choices=choices[i]: selectAll(choices, i))
            select[i].grid(row=i + 2, column=6, sticky=tk.W)



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
data_set_values = ["Diabetes"]
# get all *.csv files in the data directory "userData" and append them to the list of data_set_values
data_set_values.extend([f for f in os.listdir("userData") if f.endswith(".csv") or f.endswith(".pkl")])
data_set_combo = ttk.Combobox(run_tab, values=data_set_values)
data_set_combo.set("Diabetes")  # Default selection
data_set_combo.grid(row=1, column=1)

feature_type_label = tk.Label(run_tab, text="torch feature_type:")
feature_type_label.grid(row=2, column=0, sticky="W")
feature_type_entry = tk.Entry(run_tab)
feature_type_entry.insert(0, "float32")
feature_type_entry.grid(row=2, column=1, sticky="W")

target_type_label = tk.Label(run_tab, text="torch target_type:")
target_type_label.grid(row=3, column=0, sticky="W")
target_type_entry = tk.Entry(run_tab)
target_type_entry.insert(0, "float32")
target_type_entry.grid(row=3, column=1, sticky="W")

target_column_label = tk.Label(run_tab, text="target_column:")
target_column_label.grid(row=4, column=0, sticky="W")
target_column_entry = tk.Entry(run_tab)
target_column_entry.insert(0, "target")
target_column_entry.grid(row=4, column=1, sticky="W")

n_total_label = tk.Label(run_tab, text="n_total:")
n_total_label.grid(row=5, column=0, sticky="W")
n_total_entry = tk.Entry(run_tab)
n_total_entry.insert(0, "All")
n_total_entry.grid(row=5, column=1, sticky="W")

test_size_label = tk.Label(run_tab, text="test_size:")
test_size_label.grid(row=6, column=0, sticky="W")
test_size_entry = tk.Entry(run_tab)
test_size_entry.insert(0, "0.10")
test_size_entry.grid(row=6, column=1, sticky="W")

lin_label = tk.Label(run_tab, text="_L_in:")
lin_label.grid(row=7, column=0, sticky="W")
lin_entry = tk.Entry(run_tab)
lin_entry.insert(0, "10")
lin_entry.grid(row=7, column=1, sticky="W")

lout_label = tk.Label(run_tab, text="_L_out:")
lout_label.grid(row=8, column=0, sticky="W")
lout_entry = tk.Entry(run_tab)
lout_entry.insert(0, "1")
lout_entry.grid(row=8, column=1, sticky="W")


# columns 0+1: Experiment
experiment_label = tk.Label(run_tab, text="Experiment options:")
experiment_label.grid(row=9, column=0, sticky="W")

max_time_label = tk.Label(run_tab, text="MAX_TIME:")
max_time_label.grid(row=10, column=0, sticky="W")
max_time_entry = tk.Entry(run_tab)
max_time_entry.insert(0, "1")
max_time_entry.grid(row=10, column=1)

fun_evals_label = tk.Label(run_tab, text="FUN_EVALS:")
fun_evals_label.grid(row=11, column=0, sticky="W")
fun_evals_entry = tk.Entry(run_tab)
fun_evals_entry.insert(0, "inf")
fun_evals_entry.grid(row=11, column=1)

init_size_label = tk.Label(run_tab, text="INIT_SIZE:")
init_size_label.grid(row=12, column=0, sticky="W")
init_size_entry = tk.Entry(run_tab)
init_size_entry.insert(0, "6")
init_size_entry.grid(row=12, column=1)

prefix_label = tk.Label(run_tab, text="PREFIX:")
prefix_label.grid(row=13, column=0, sticky="W")
prefix_entry = tk.Entry(run_tab)
prefix_entry.insert(0, "0000-0000")
prefix_entry.grid(row=13, column=1)

noise_label = tk.Label(run_tab, text="NOISE:")
noise_label.grid(row=14, column=0, sticky="W")
noise_entry = tk.Entry(run_tab)
noise_entry.insert(0, "TRUE")
noise_entry.grid(row=14, column=1)


# colummns 2 - 5: Model
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
core_model_values = ["NetLightRegression", "NetLightRegression2", "TransformerLightRegression"]
for filename in os.listdir("userModel"):
    if filename.endswith(".json"):
        core_model_values.append(os.path.splitext(filename)[0])
core_model_combo = ttk.Combobox(run_tab, values=core_model_values)
core_model_combo.set("Select model")  # Default selection
core_model_combo.bind("<<ComboboxSelected>>", update_hyperparams)
core_model_combo.grid(row=1, column=3)


# column 8: Save and run button
save_button = ttk.Button(run_tab, text="Save Experiment", command=lambda: run_experiment(save_only=True))
save_button.grid(row=7, column=8, columnspan=2, sticky="E")
run_button = ttk.Button(run_tab, text="Run Experiment", command=run_experiment)
run_button.grid(row=8, column=8, columnspan=2, sticky="E")

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

# TODO: Add logo to river tab
# river_logo_label = tk.Label(river_tab, image=logo_image)
# river_logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

# Run the mainloop



app.mainloop()
