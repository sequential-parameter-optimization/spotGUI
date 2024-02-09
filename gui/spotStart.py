import tkinter as tk
from tkinter import ttk
from spotPython.hyperdict.light_hyper_dict import LightHyperDict

from spotGUI.tuner.spotRun import (
    run_spot_python_experiment,
    contour_plot,
    parallel_plot,
    importance_plot,
    progress_plot,
)

result = None
fun_control = None


def run_experiment(save_only=False):
    global result, fun_control
    MAX_TIME = float(max_time_entry.get())
    INIT_SIZE = int(init_size_entry.get())
    PREFIX = prefix_entry.get()
    NOISE = bool(noise_entry.get())
    n_total = n_total_entry.get()
    if n_total == "None" or n_total == "All":
        n_total = None
    else:
        n_total = int(n_total)
    perc_train = float(perc_train_entry.get())
    lin = int(lin_entry.get())
    lout = int(lout_entry.get())
    data_set = data_set_combo.get()
    core_model = core_model_combo.get()

    result, fun_control, design_control, surrogate_control, optimizer_control = run_spot_python_experiment(
        _L_in=lin,
        _L_out=lout,
        MAX_TIME=MAX_TIME,
        INIT_SIZE=INIT_SIZE,
        PREFIX=PREFIX,
        NOISE=NOISE,
        data_set=data_set,
        coremodel=core_model,
        log_level=50,
        save_only=save_only
    )


def call_parallel_plot():
    if result is not None:
        parallel_plot(result)


def call_contour_plot():
    if result is not None:
        contour_plot(result)


def call_progress_plot():
    if result is not None:
        progress_plot(result)


def call_importance_plot():
    if result is not None:
        importance_plot(result)


def update_hyperparams():
    model = core_model_combo.get()
    dict = lhd.hyper_dict[model]
    for i, (key, value) in enumerate(dict.items()):
        if dict[key]["type"] == "int" or dict[key]["type"] == "float":
            # Create a label with the key as text
            label = tk.Label(run_tab, text=key)
            label.grid(row=i + 2, column=2, sticky="W")
            label.update()
            print(f"key: {key}, value: {value}")
            # Create an entry with the default value as the default text
            default_entry = tk.Entry(run_tab)
            default_entry.insert(0, dict[key]["default"])
            default_entry.grid(row=i + 2, column=3, sticky="W")
            # add the lower bound values in column 2
            lower_bound_entry = tk.Entry(run_tab)
            lower_bound_entry.insert(0, dict[key]["lower"])
            lower_bound_entry.grid(row=i + 2, column=4, sticky="W")
            # add the upper bound values in column 3
            upper_bound_entry = tk.Entry(run_tab)
            upper_bound_entry.insert(0, dict[key]["upper"])
            upper_bound_entry.grid(row=i + 2, column=5, sticky="W")
        if dict[key]["type"] == "factor":
            # Create a label with the key as text
            label = tk.Label(run_tab, text=key)
            label.grid(row=i + 2, column=2, sticky="W")
            label.update()
            # Create an entry with the default value as the default text
            default_entry = tk.Entry(run_tab)
            default_entry.insert(0, dict[key]["default"])
            default_entry.grid(row=i + 2, column=3, sticky="W")
            # add the lower bound values in column 2
            factor_level_entry = tk.Entry(run_tab)
            # add a comma to each level
            # dict[key]["levels"] = ", ".join(dict[key]["levels"])
            factor_level_entry.insert(0, dict[key]["levels"])
            factor_level_entry.grid(row=i + 2, column=4, sticky="W")


# Create the main application window
app = tk.Tk()
app.title("Spot Python Hyperparameter Tuning GUI")

# Create a LightHyperDict object
lhd = LightHyperDict()

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


# columns 4+5: Experiment
experiment_label = tk.Label(run_tab, text="Experiment options:")
experiment_label.grid(row=0, column=6, sticky="W")

max_time_label = tk.Label(run_tab, text="MAX_TIME:")
max_time_label.grid(row=1, column=6, sticky="W")
max_time_entry = tk.Entry(run_tab)
max_time_entry.insert(0, "1")
max_time_entry.grid(row=1, column=7)

init_size_label = tk.Label(run_tab, text="INIT_SIZE:")
init_size_label.grid(row=2, column=6, sticky="W")
init_size_entry = tk.Entry(run_tab)
init_size_entry.insert(0, "6")
init_size_entry.grid(row=2, column=7)

prefix_label = tk.Label(run_tab, text="PREFIX:")
prefix_label.grid(row=3, column=6, sticky="W")
prefix_entry = tk.Entry(run_tab)
prefix_entry.insert(0, "SPOT_0000")
prefix_entry.grid(row=3, column=7)

noise_label = tk.Label(run_tab, text="NOISE:")
noise_label.grid(row=4, column=6, sticky="W")
noise_entry = tk.Entry(run_tab)
noise_entry.insert(0, "TRUE")
noise_entry.grid(row=4, column=7)


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
