import customtkinter
import os
import webbrowser
import copy
from spotGUI.tuner.spotRun import (
    progress_plot,
    contour_plot,
    importance_plot,
    load_file_dialog,
    get_scenario_dict,
)
from PIL import Image
import time
from spotPython.utils.eda import gen_design_table
import tkinter as tk
import sys
from spotGUI.ctk.SelectOptions import SelectOptionMenuFrame
from spotGUI.ctk.HyperparameterFrame import NumHyperparameterFrame, CatHyperparameterFrame
from spotPython.utils.file import load_experiment as load_experiment_spot
from spotPython.hyperparameters.values import (
    get_river_prep_model,
    get_river_core_model_from_name,
    get_core_model_from_name,
    get_prep_model,
    get_sklearn_scaler,
)
from spotRiver.hyperdict.river_hyper_dict import RiverHyperDict
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.hyperdict.sklearn_hyper_dict import SklearnHyperDict


class CTkApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        # name of the progress file
        self.progress_file = "progress.txt"
        # if the progress file exists, delete it
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        current_path = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_path, "images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "spotlogo.png")), size=(85, 37))
        self.geometry(f"{1600}x{1200}")
        self.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)
        self.entry_width = 80

    def change_appearance_mode_event(self, new_appearance_mode: str):
        print(f"Appearance Mode changed to: {new_appearance_mode}")
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def plot_progress_button_event(self):
        if self.spot_tuner is not None:
            progress_plot(spot_tuner=self.spot_tuner, fun_control=self.fun_control)

    def plot_contour_button_event(self):
        if self.spot_tuner is not None:
            contour_plot(spot_tuner=self.spot_tuner, fun_control=self.fun_control)

    def plot_importance_button_event(self):
        if self.spot_tuner is not None:
            importance_plot(spot_tuner=self.spot_tuner, fun_control=self.fun_control)

    def print_tuned_design(self):
        text = gen_design_table(self.fun_control)
        self.textbox.delete("1.0", tk.END)
        self.textbox.insert(tk.END, text)

    def update_text(self):
        # This method runs in a separate thread to update the text area
        while True:  # Infinite loop to continuously update the textbox
            try:
                with open("progress.txt", "r") as file:
                    lines = file.readlines()
                    last_line = lines[-1] if lines else ""
                    # Get the last line or an empty string if file is empty
                    # text = file.read()  # Read the entire file
                    self.textbox.delete("1.0", tk.END)
                    self.textbox.insert(tk.END, last_line)
            except FileNotFoundError:
                # text = "File not found."
                # self.textbox.insert(tk.END, text)
                pass
            time.sleep(1)  # Wait for 1 second before the next update

    def label_button_frame_event(self, item):
        print(f"label button frame clicked: {item}")

    def select_data_frame_event(self, new_data: str):
        print(f"Data modified: {new_data}")
        print(f"Data Selection modified: {self.select_data_frame.get_selected_optionmenu_item()}")

    def create_num_hp_frame(self, dict=None):
        # create new num_hp_frame
        self.num_hp_frame = NumHyperparameterFrame(
            master=self.hp_main_frame, width=640, command=self.label_button_frame_event, entry_width=self.entry_width
        )

        self.num_hp_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.num_hp_frame.add_header()
        if self.scenario == "river":
            coremodel, core_model_instance = get_river_core_model_from_name(self.core_model_name)
        else:
            coremodel, core_model_instance = get_core_model_from_name(self.core_model_name)
        if dict is None:
            dict = self.hyperdict().hyper_dict[coremodel]
        for i, (key, value) in enumerate(dict.items()):
            if (
                dict[key]["type"] == "int"
                or dict[key]["type"] == "float"
                or dict[key]["core_model_parameter_type"] == "bool"
            ):
                self.num_hp_frame.add_num_item(
                    hp=key,
                    default=value["default"],
                    lower=value["lower"],
                    upper=value["upper"],
                    transform=value["transform"],
                )

    def create_cat_hp_frame(self, dict=None):
        self.cat_hp_frame = CatHyperparameterFrame(master=self.hp_main_frame, command=self.label_button_frame_event)
        self.cat_hp_frame.grid(row=2, column=0, padx=0, pady=0, sticky="nsew")
        if self.scenario == "river":
            coremodel, core_model_instance = get_river_core_model_from_name(self.core_model_name)
        else:
            coremodel, core_model_instance = get_core_model_from_name(self.core_model_name)
        if dict is None:
            dict = self.hyperdict().hyper_dict[coremodel]
        empty = True
        for i, (key, value) in enumerate(dict.items()):
            if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
                if empty:
                    self.cat_hp_frame.add_header()
                    empty = False
                self.cat_hp_frame.add_cat_item(
                    hp=key, default=value["default"], levels=value["levels"], transform=value["transform"]
                )

    def create_core_model_frame(self, row, column):
        # create new core model frame
        self.select_core_model_frame = SelectOptionMenuFrame(
            master=self.sidebar_frame,
            command=self.select_core_model_frame_event,
            item_list=self.scenario_dict[self.task_name]["core_model_names"],
            item_default=None,
            title="Select Core Model",
        )
        self.select_core_model_frame.grid(row=row, column=column, padx=15, pady=15, sticky="nsew")
        self.select_core_model_frame.configure(width=500)
        self.core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()

    def create_prep_model_frame(self, row, column):
        if self.scenario == "lightning":
            self.select_prep_model_frame.destroy()
        else:
            self.prep_model_values = self.scenario_dict[self.task_name]["prep_models"]
            if self.prep_model_values is not None:
                self.prep_model_values.extend(
                    [f for f in os.listdir("userPrepModel") if f.endswith(".py") and not f.startswith("__")]
                )
                self.select_prep_model_frame = SelectOptionMenuFrame(
                    master=self.sidebar_frame,
                    command=self.select_prep_model_frame_event,
                    item_list=self.prep_model_values,
                    item_default=None,
                    title="Select Prep Model",
                )
                self.select_prep_model_frame.grid(row=row, column=column, padx=15, pady=15, sticky="nsew")
                self.select_prep_model_frame.configure(width=500)

    def create_scaler_frame(self, row, column):
        if self.scenario == "lightning" or self.scenario == "river":
            self.select_scaler_frame.destroy()
        else:
            self.scaler_values = self.scenario_dict[self.task_name]["scalers"]
            directory = "userScaler"
            if os.path.exists(directory):
                self.scaler_values.extend(
                    [f for f in os.listdir(directory) if f.endswith(".py") and not f.startswith("__")]
                )
            if self.scaler_values is not None:
                self.select_scaler_frame = SelectOptionMenuFrame(
                    master=self.sidebar_frame,
                    command=self.select_scaler_frame_event,
                    item_list=self.scaler_values,
                    item_default=None,
                    title="Select Scaler",
                )
                self.select_scaler_frame.grid(row=row, column=column, padx=15, pady=15, sticky="nsew")
                self.select_scaler_frame.configure(width=500)
            else:
                self.select_scaler_frame = None

    def create_select_data_frame(self, row, column):
        data_set_values = copy.deepcopy(self.scenario_dict[self.task_name]["datasets"])
        data_set_values.extend([f for f in os.listdir("userData") if f.endswith(".csv") or f.endswith(".pkl")])
        self.select_data_frame = SelectOptionMenuFrame(
            master=self.sidebar_frame,
            command=self.select_data_frame_event,
            item_list=data_set_values,
            item_default=None,
            title="Select Data",
        )
        self.select_data_frame.grid(row=row, column=column, padx=15, pady=15, sticky="nswe")
        self.select_data_frame.configure(width=500)
        self.data_set_name = self.select_data_frame.get_selected_optionmenu_item()

    def create_metric_sklearn_levels_frame(self, row, column):
        self.select_metric_sklearn_levels_frame = SelectOptionMenuFrame(
            master=self.sidebar_frame,
            command=self.select_metric_sklearn_levels_frame_event,
            item_list=self.scenario_dict[self.task_name]["metric_sklearn_levels"],
            item_default=None,
            title="Select sklearn metric",
        )
        self.select_metric_sklearn_levels_frame.grid(row=row, column=0, padx=15, pady=15, sticky="nsew")
        self.select_metric_sklearn_levels_frame.configure(width=500)

    def select_core_model_frame_event(self, new_core_model: str):
        self.core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()
        self.num_hp_frame.destroy()
        self.create_num_hp_frame()
        self.cat_hp_frame.destroy()
        self.create_cat_hp_frame()

    def select_prep_model_frame_event(self, new_prep_model: str):
        print(f"Prep Model modified: {self.select_prep_model_frame.get_selected_optionmenu_item()}")

    def select_scaler_frame_event(self, new_scaler: str):
        print(f"Scaler modified: {self.select_scaler_frame.get_selected_optionmenu_item()}")

    def check_user_prep_model(self, prep_model_name) -> object:
        """Check if the prep model is a user defined prep model.
        If it is a user defined prep model, import the prep model from the userPrepModel directory.
        Otherwise, get the prep model from the river.preprocessing module.

        Args:
            prep_model_name (str): The name of the prep model.

        Returns:
            prepmodel: The prep model object.

        """
        if prep_model_name.endswith(".py"):
            print(f"prep_model_name = {prep_model_name}")
            sys.path.insert(0, "./userPrepModel")
            # remove the file extension from the prep_model_name
            prep_model_name = prep_model_name[:-3]
            print(f"prep_model_name = {prep_model_name}")
            __import__(prep_model_name)
            prepmodel = sys.modules[prep_model_name].set_prep_model()
        elif self.scenario == "river":
            # get the river prep model from river.preprocessing
            prepmodel = get_river_prep_model(prep_model_name)
        else:
            # get the prep model from the sklearn.preprocessing module
            prepmodel = get_prep_model(prep_model_name)
        return prepmodel

    def check_user_scaler(self, scaler_name) -> object:
        """Check if the scaler is a user defined scaler.
        Aplies to sklearn and lightning scenarios.
        If it is a user defined scaler, import the scaler from the userScaler directory.
        Otherwise, get the scaler from the sklearn.preprocessing module.

        Args:
            scaler_name (str): The name of the scaler.

        Returns:
            scaler: The scaler object.

        """
        if scaler_name.endswith(".py"):
            print(f"scaler_name = {scaler_name}")
            sys.path.insert(0, "./userScaler")
            # remove the file extension from the scaler_name
            scaler_name = scaler_name[:-3]
            print(f"scaler_name = {scaler_name}")
            __import__(scaler_name)
            scaler = sys.modules[scaler_name].set_scaler()
        else:
            # get the prep model from the sklearn.preprocessing module
            scaler = get_sklearn_scaler(scaler_name)
        return scaler

    def select_metric_sklearn_levels_frame_event(self, new_metric_sklearn_levels: str):
        print(f"Metric sklearn modified: {self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item()}")
        self.metric_sklearn_name = self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item()

    def change_scenario_event(self, new_scenario: str):
        print(f"Scenario changed to: {new_scenario}")
        self.scenario_dict = get_scenario_dict(scenario=new_scenario)
        if hasattr(self, "select_scaler_frame"):
            self.select_scaler_frame.destroy()
        if new_scenario == "river":
            self.scenario = "river"
            self.hyperdict = RiverHyperDict
        elif new_scenario == "sklearn":
            self.scenario = "sklearn"
            self.hyperdict = SklearnHyperDict
            self.create_scaler_frame(row=4, column=0)
        elif new_scenario == "lightning":
            self.scenario = "lightning"
            self.hyperdict = LightHyperDict
        else:
            print("Error: Scenario not found")
        self.select_prep_model_frame.destroy()
        self.create_prep_model_frame(row=3, column=0)
        self.select_core_model_frame.destroy()
        self.create_core_model_frame(row=5, column=0)
        self.num_hp_frame.destroy()
        self.create_num_hp_frame()
        self.cat_hp_frame.destroy()
        self.create_cat_hp_frame()
        self.select_data_frame.destroy()
        self.create_select_data_frame(row=6, column=0)
        self.create_experiment_eval_frame()

    def change_task_event(self, new_task: str):
        print(f"Task changed to: {new_task}")
        if new_task == "Binary Classification":
            self.task_name = "classification_task"
        elif new_task == "Regression":
            self.task_name = "regression_task"
        elif new_task == "Rules":
            self.task_name = "rules_task"
        else:
            print("Error: Task not found")
        self.select_prep_model_frame.destroy()
        self.create_prep_model_frame(row=3, column=0)
        if hasattr(self, "select_scaler_frame"):
            self.select_scaler_frame.destroy()
            self.create_scaler_frame(row=4, column=0)
        self.select_core_model_frame.destroy()
        self.create_core_model_frame(row=5, column=0)
        self.num_hp_frame.destroy()
        self.create_num_hp_frame()
        self.cat_hp_frame.destroy()
        self.create_cat_hp_frame()
        self.select_data_frame.destroy()
        self.create_select_data_frame(row=6, column=0)
        self.select_metric_sklearn_levels_frame.destroy()
        self.create_metric_sklearn_levels_frame(row=7, column=0)

    def run_button_event(self):
        self.run_experiment()
        # self.print_tuned_design()

    def save_button_event(self):
        self.save_experiment()

    def print_data_botton_event(self):
        self.print_data()

    def make_sidebar_frame(self):
        self.sidebar_frame = customtkinter.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(3, weight=1)
        #
        # Inside the sidebar frame
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text=self.logo_text,
            image=self.logo_image,
            compound="left",
            font=customtkinter.CTkFont(size=20, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=10, pady=(7.5, 2.5), sticky="ew")
        #
        # ................. Scenario Frame ....................................... #
        # create scenario frame inside sidebar frame
        self.scenario_frame = SelectOptionMenuFrame(
            master=self.sidebar_frame,
            command=self.change_scenario_event,
            item_list=["lightning", "sklearn", "river"],
            item_default="river",
            title="Select Scenario",
        )
        self.scenario_frame.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")
        self.scenario_frame.configure(width=500)
        # ................. Task Frame ....................................... #
        # create task frame inside sidebar frame
        self.task_frame = SelectOptionMenuFrame(
            master=self.sidebar_frame,
            command=self.change_task_event,
            item_list=["Binary Classification", "Regression", "Rules"],
            item_default="Regression",
            title="Select Task",
        )
        self.task_frame.grid(row=2, column=0, padx=15, pady=15, sticky="nsew")
        self.task_frame.configure(width=500)
        #
        # ................. Prep Model Frame ....................................... #
        # create select prep model frame inside sidebar frame
        # if key "prep_models" exists in the scenario_dict, get the prep models from the scenario_dict
        if "prep_models" in self.scenario_dict[self.task_name]:
            self.create_prep_model_frame(row=3, column=0)
        #
        # ................. Scaler Frame ....................................... #
        # create select scaler frame inside sidebar frame
        # if key "scaler" exists in the scenario_dict, get the scaler from the scenario_dict
        if "scalers" in self.scenario_dict[self.task_name]:
            self.create_scaler_frame(row=4, column=0)
        #
        # ................. Core Model Frame ....................................... #
        print(f"scenario_dict = {self.scenario_dict}")
        self.core_model_name = self.scenario_dict[self.task_name]["core_model_names"][0]
        # Uncomment to get user defined core models (not useful for spotRiver):
        # for filename in os.listdir("userModel"):
        #     if filename.endswith(".json"):
        #         self.core_model_name.append(os.path.splitext(filename)[0])
        # create core model frame inside sidebar frame
        self.create_core_model_frame(row=5, column=0)  #
        #  ................. Data Frame ....................................... #
        # select data frame in data main frame
        self.create_select_data_frame(row=6, column=0)
        #
        # # create plot data button
        # self.plot_data_button = customtkinter.CTkButton(
        #     master=self.sidebar_frame, text="Plot Data", command=self.plot_data_button_event
        # )
        # self.plot_data_button.grid(row=6, column=0, sticky="nsew", padx=10, pady=10)
        # #
        # ................. Metric Frame ....................................... #
        # create select metric_sklearn levels frame inside sidebar frame
        self.select_metric_sklearn_levels_frame = SelectOptionMenuFrame(
            master=self.sidebar_frame,
            command=self.select_metric_sklearn_levels_frame_event,
            item_list=self.scenario_dict[self.task_name]["metric_sklearn_levels"],
            item_default=None,
            title="Select sklearn metric",
        )
        self.select_metric_sklearn_levels_frame.grid(row=7, column=0, padx=15, pady=15, sticky="nsew")
        self.select_metric_sklearn_levels_frame.configure(width=500)
        #
        # ................. Appearance Frame ....................................... #
        # create appearance mode frame
        customtkinter.set_appearance_mode("System")
        self.appearance_frame = SelectOptionMenuFrame(
            master=self.sidebar_frame,
            width=500,
            command=self.change_appearance_mode_event,
            item_list=["System", "Light", "Dark"],
            item_default="System",
            title="Appearance Mode",
        )
        self.appearance_frame.grid(row=8, column=0, padx=15, pady=15, sticky="ew")
        #
        self.scaling_label = customtkinter.CTkLabel(self.appearance_frame, text="UI Scaling", anchor="w")
        self.scaling_label.grid(row=2, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(
            self.appearance_frame, values=["100%", "80%", "90%", "110%", "120%"], command=self.change_scaling_event
        )
        self.scaling_optionemenu.grid(row=3, column=0, padx=15, pady=15, sticky="ew")

    def make_experiment_frame(self):
        self.experiment_main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.experiment_main_frame.grid(row=0, column=1, sticky="nsew")
        #
        # experiment frame title in experiment main frame
        self.experiment_main_frame_title = customtkinter.CTkLabel(
            self.experiment_main_frame,
            text="Experiment Options",
            font=customtkinter.CTkFont(size=20, weight="bold"),
            corner_radius=6,
        )
        self.experiment_main_frame_title.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        #
        # ................. Experiment_Data Frame .......................................#
        # create experiment data_frame with widgets in experiment_main frame
        self.experiment_data_frame = customtkinter.CTkFrame(self.experiment_main_frame, corner_radius=6)
        self.experiment_data_frame.grid(row=1, column=0, sticky="ew")
        #
        # experiment_data frame title
        self.experiment_data_frame_title = customtkinter.CTkLabel(
            self.experiment_data_frame, text="Data Options", font=customtkinter.CTkFont(weight="bold"), corner_radius=6
        )
        self.experiment_data_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        #
        # n_total entry in experiment_data frame
        self.n_total_label = customtkinter.CTkLabel(self.experiment_data_frame, text="n_total", corner_radius=6)
        self.n_total_label.grid(row=1, column=0, padx=0, pady=(10, 0), sticky="w")
        self.n_total_var = customtkinter.StringVar(value="None")
        self.n_total_entry = customtkinter.CTkEntry(
            self.experiment_data_frame, textvariable=self.n_total_var, width=self.entry_width
        )
        self.n_total_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        #
        # test_size entry in experiment_data frame
        self.test_size_label = customtkinter.CTkLabel(self.experiment_data_frame, text="test_size", corner_radius=6)
        self.test_size_label.grid(row=2, column=0, padx=0, pady=(10, 0), sticky="w")
        self.test_size_var = customtkinter.StringVar(value="0.3")
        self.test_size_entry = customtkinter.CTkEntry(
            self.experiment_data_frame, textvariable=self.test_size_var, width=self.entry_width
        )
        self.test_size_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        #
        # shuffle data in experiment_data frame
        self.shuffle_var = customtkinter.StringVar(value="False")
        self.shuffle_checkbox = customtkinter.CTkCheckBox(
            self.experiment_data_frame,
            text="ShuffleData",
            command=None,
            variable=self.shuffle_var,
            onvalue="True",
            offvalue="False",
        )
        self.shuffle_checkbox.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # ................. Experiment_Model Frame .......................................#
        # create experiment_model frame with widgets in experiment_main frame
        self.experiment_model_frame = customtkinter.CTkFrame(self.experiment_main_frame, corner_radius=6)
        self.experiment_model_frame.grid(row=3, column=0, sticky="ew")
        #
        # experiment_model frame title
        self.experiment_model_frame_title = customtkinter.CTkLabel(
            self.experiment_model_frame,
            text="Model Options",
            font=customtkinter.CTkFont(weight="bold"),
            corner_radius=6,
        )
        self.experiment_model_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        #
        # max_time entry in experiment_model frame
        self.max_time_label = customtkinter.CTkLabel(self.experiment_model_frame, text="max_time", corner_radius=6)
        self.max_time_label.grid(row=1, column=0, padx=0, pady=(10, 0), sticky="w")
        self.max_time_var = customtkinter.StringVar(value="1")
        self.max_time_entry = customtkinter.CTkEntry(
            self.experiment_model_frame, textvariable=self.max_time_var, width=self.entry_width
        )
        self.max_time_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        #
        self.fun_evals_label = customtkinter.CTkLabel(self.experiment_model_frame, text="fun_evals", corner_radius=6)
        self.fun_evals_label.grid(row=2, column=0, padx=0, pady=(10, 0), sticky="w")
        self.fun_evals_var = customtkinter.StringVar(value="30")
        self.fun_evals_entry = customtkinter.CTkEntry(
            self.experiment_model_frame, textvariable=self.fun_evals_var, width=self.entry_width
        )
        self.fun_evals_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        #
        # init_size entry in experiment_model frame
        self.init_size_label = customtkinter.CTkLabel(self.experiment_model_frame, text="init_size", corner_radius=6)
        self.init_size_label.grid(row=3, column=0, padx=0, pady=(10, 0), sticky="w")
        self.init_size_var = customtkinter.StringVar(value="5")
        self.init_size_entry = customtkinter.CTkEntry(
            self.experiment_model_frame, textvariable=self.init_size_var, width=self.entry_width
        )
        self.init_size_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        #
        self.lambda_min_max_label = customtkinter.CTkLabel(
            self.experiment_model_frame, text="lambda_min_max", corner_radius=6
        )
        self.lambda_min_max_label.grid(row=4, column=0, padx=0, pady=(10, 0), sticky="w")
        self.lambda_min_max_var = customtkinter.StringVar(value="1e-3, 1e2")
        self.lambda_min_max_entry = customtkinter.CTkEntry(
            self.experiment_model_frame, textvariable=self.lambda_min_max_var, width=self.entry_width
        )
        self.lambda_min_max_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        #
        self.max_sp_label = customtkinter.CTkLabel(self.experiment_model_frame, text="max_sp", corner_radius=6)
        self.max_sp_label.grid(row=5, column=0, padx=0, pady=(10, 0), sticky="w")
        self.max_sp_var = customtkinter.StringVar(value="30")
        self.max_sp_entry = customtkinter.CTkEntry(
            self.experiment_model_frame, textvariable=self.max_sp_var, width=self.entry_width
        )
        self.max_sp_entry.grid(row=5, column=1, padx=10, pady=10, sticky="w")
        #
        self.seed_label = customtkinter.CTkLabel(self.experiment_model_frame, text="seed", corner_radius=6)
        self.seed_label.grid(row=6, column=0, padx=0, pady=(10, 0), sticky="w")
        self.seed_var = customtkinter.StringVar(value="123")
        self.seed_entry = customtkinter.CTkEntry(
            self.experiment_model_frame, textvariable=self.seed_var, width=self.entry_width
        )
        self.seed_entry.grid(row=6, column=1, padx=10, pady=10, sticky="w")
        #
        # noise data in experiment_model frame
        self.noise_var = customtkinter.StringVar(value="True")
        self.noise_checkbox = customtkinter.CTkCheckBox(
            self.experiment_model_frame,
            text="noise",
            command=None,
            variable=self.noise_var,
            onvalue="True",
            offvalue="False",
        )
        self.noise_checkbox.grid(row=7, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # ................. River: Experiment_Eval Options Frame .................................#
        self.create_experiment_eval_frame()

    def make_hyperparameter_frame(self):
        self.hp_main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.hp_main_frame.grid(row=0, column=2, sticky="nsew")
        #
        # create hyperparameter title frame in hyperparameter main frame
        self.hp_main_frame_title = customtkinter.CTkLabel(
            self.hp_main_frame,
            text="Hyperparameter",
            font=customtkinter.CTkFont(size=20, weight="bold"),
            corner_radius=6,
        )
        self.hp_main_frame_title.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        #
        self.create_num_hp_frame()
        #
        self.create_cat_hp_frame()

    def make_execution_frame(self):
        self.execution_main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.execution_main_frame.grid(row=0, column=3, sticky="nsew")
        #
        # execution frame title in execution main frame
        self.execution_main_frame_title = customtkinter.CTkLabel(
            self.execution_main_frame,
            text="Run Options",
            font=customtkinter.CTkFont(size=20, weight="bold"),
            corner_radius=6,
        )
        self.execution_main_frame_title.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        #
        # ................. execution_tb Frame .......................................#
        # create execution_tb frame with widgets in execution_main frame
        self.execution_tb_frame = customtkinter.CTkFrame(self.execution_main_frame, corner_radius=6)
        self.execution_tb_frame.grid(row=1, column=0, sticky="ew")
        #
        # execution_tb frame title
        self.execution_tb_frame_title = customtkinter.CTkLabel(
            self.execution_tb_frame,
            text="Tensorboard Options",
            font=customtkinter.CTkFont(weight="bold"),
            corner_radius=6,
        )
        self.execution_tb_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # tb_clean in execution_tb frame
        self.tb_clean_var = customtkinter.StringVar(value="True")
        self.tb_clean_checkbox = customtkinter.CTkCheckBox(
            self.execution_tb_frame,
            text="TENSORBOARD_CLEAN",
            command=None,
            variable=self.tb_clean_var,
            onvalue="True",
            offvalue="False",
        )
        self.tb_clean_checkbox.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")
        # tb_start in execution_tb frame
        self.tb_start_var = customtkinter.StringVar(value="True")
        self.tb_start_checkbox = customtkinter.CTkCheckBox(
            self.execution_tb_frame,
            text="Start Tensorboard",
            command=None,
            variable=self.tb_start_var,
            onvalue="True",
            offvalue="False",
        )
        self.tb_start_checkbox.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="w")
        # tb_stop in execution_tb frame
        self.tb_stop_var = customtkinter.StringVar(value="True")
        self.tb_stop_checkbox = customtkinter.CTkCheckBox(
            self.execution_tb_frame,
            text="Stop Tensorboard",
            command=None,
            variable=self.tb_stop_var,
            onvalue="True",
            offvalue="False",
        )
        self.tb_stop_checkbox.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        self.browser_link_label = customtkinter.CTkLabel(
            self.execution_tb_frame,
            text="Open http://localhost:6006",
            text_color=("blue", "orange"),
            cursor="hand2",
            corner_radius=6,
        )
        self.browser_link_label.bind("<Button-1>", lambda e: webbrowser.open_new("http://localhost:6006"))
        self.browser_link_label.grid(row=4, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # ................. execution_docs Frame .......................................#
        # create execution_docs frame with widgets in execution_main frame
        self.execution_docs_frame = customtkinter.CTkFrame(self.execution_main_frame, corner_radius=6)
        self.execution_docs_frame.grid(row=2, column=0, sticky="ew")
        #
        # execution_model frame title
        self.execution_docs_frame_title = customtkinter.CTkLabel(
            self.execution_docs_frame, text="Documentation", font=customtkinter.CTkFont(weight="bold"), corner_radius=6
        )
        self.execution_docs_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        # spot_doc entry in execution_model frame
        self.spot_link_label = customtkinter.CTkLabel(
            self.execution_docs_frame,
            text="spotPython documentation",
            text_color=("blue", "orange"),
            cursor="hand2",
            corner_radius=6,
        )
        self.spot_link_label.bind(
            "<Button-1>",
            lambda e: webbrowser.open_new("https://sequential-parameter-optimization.github.io/spotPython/"),
        )
        self.spot_link_label.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # spotriver_doc entry in execution_model frame
        self.spotriver_link_label = customtkinter.CTkLabel(
            self.execution_docs_frame,
            text="spotRiver documentation",
            text_color=("blue", "orange"),
            cursor="hand2",
            corner_radius=6,
        )
        self.spotriver_link_label.bind(
            "<Button-1>",
            lambda e: webbrowser.open_new("https://sequential-parameter-optimization.github.io/spotRiver/"),
        )
        self.spotriver_link_label.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # river_link entry in execution_model frame
        self.river_link_label = customtkinter.CTkLabel(
            self.execution_docs_frame,
            text="River documentation",
            text_color=("blue", "orange"),
            cursor="hand2",
            corner_radius=6,
        )
        self.river_link_label.bind(
            "<Button-1>", lambda e: webbrowser.open_new("https://riverml.xyz/latest/api/overview/")
        )
        self.river_link_label.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="w")

        #
        # ................. Execution_Experiment_Name Frame .......................................#
        # create experiment data_frame with widgets in experiment_main frame
        self.experiment_name_frame = customtkinter.CTkFrame(self.execution_main_frame, corner_radius=6)
        self.experiment_name_frame.grid(row=3, column=0, sticky="ew")
        #
        # experiment_data frame title
        self.experiment_name_frame_title = customtkinter.CTkLabel(
            self.experiment_name_frame,
            text="New experiment name",
            font=customtkinter.CTkFont(weight="bold"),
            corner_radius=6,
        )
        self.experiment_name_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        #
        # experiment_name entry in experiment_name frame
        self.experiment_name_var = customtkinter.StringVar(value="000")
        self.experiment_name_entry = customtkinter.CTkEntry(
            self.experiment_name_frame, textvariable=self.experiment_name_var, width=self.entry_width
        )
        self.experiment_name_entry.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        # ................. Run_Experiment_Name Frame .......................................#
        # create experiment_run_frame with widgets in experiment_main frame
        self.experiment_run_frame = customtkinter.CTkFrame(self.execution_main_frame, corner_radius=6)
        self.experiment_run_frame.grid(row=4, column=0, sticky="ew")
        #
        # experiment_data frame title
        self.experiment_run_frame_title = customtkinter.CTkLabel(
            self.experiment_run_frame, text="Execute", font=customtkinter.CTkFont(weight="bold"), corner_radius=6
        )
        self.experiment_run_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # create print_data button
        self.print_data_button = customtkinter.CTkButton(
            master=self.experiment_run_frame, text="Print Data", command=self.print_data_botton_event
        )
        self.print_data_button.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        # create save button
        self.save_button = customtkinter.CTkButton(
            master=self.experiment_run_frame, text="Save", command=self.save_button_event
        )
        self.save_button.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        # create run button
        self.run_button = customtkinter.CTkButton(
            master=self.experiment_run_frame, text="Run", command=self.run_button_event
        )
        self.run_button.grid(row=4, column=0, sticky="ew", padx=10, pady=10)

    def make_analysis_frame(self):
        self.analysis_main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.analysis_main_frame.grid(row=0, column=4, sticky="nsew")
        #
        # analysis frame title in analysis main frame
        self.analysis_main_frame_title = customtkinter.CTkLabel(
            self.analysis_main_frame,
            text="Analysis",
            font=customtkinter.CTkFont(size=20, weight="bold"),
            corner_radius=6,
        )
        self.analysis_main_frame_title.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        #
        # ................. Run_Analysis Frame .......................................#
        # create analysis_run_frame with widgets in analysis_main frame
        self.analysis_run_frame = customtkinter.CTkFrame(self.analysis_main_frame, corner_radius=6)
        self.analysis_run_frame.grid(row=1, column=0, sticky="ew")
        #
        # analysis_data frame title
        self.analysis_run_frame_title = customtkinter.CTkLabel(
            self.analysis_run_frame,
            text="Loaded experiment",
            font=customtkinter.CTkFont(weight="bold"),
            corner_radius=6,
        )
        self.analysis_run_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        # Create Loaded Experiment Entry
        self.loaded_label = customtkinter.CTkLabel(self.analysis_run_frame, text="None", corner_radius=6)
        self.loaded_label.grid(row=1, column=0, padx=0, pady=(10, 0), sticky="w")
        # self.loaded_label.configure(text=self.experiment_name)
        # create load button
        self.load_button = customtkinter.CTkButton(
            master=self.analysis_run_frame, text="Load", command=self.load_button_event
        )
        self.load_button.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        # create plot progress button
        self.plot_progress_button = customtkinter.CTkButton(
            master=self.analysis_run_frame, text="Plot Progress", command=self.plot_progress_button_event
        )
        self.plot_progress_button.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        #
        # ................. Hyperparameter_Analysis Frame .......................................#
        # create analysis_hyperparameter_frame with widgets in analysis_main frame
        self.analysis_hyperparameter_frame = customtkinter.CTkFrame(self.analysis_main_frame, corner_radius=6)
        self.analysis_hyperparameter_frame.grid(row=4, column=0, sticky="ew")
        #
        # analysis_data frame title
        self.analysis_hyperparameter_frame_title = customtkinter.CTkLabel(
            self.analysis_hyperparameter_frame,
            text="Hyperparameter",
            font=customtkinter.CTkFont(weight="bold"),
            corner_radius=6,
        )
        self.analysis_hyperparameter_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # create contour plot  button
        self.contour_button = customtkinter.CTkButton(
            master=self.analysis_hyperparameter_frame, text="Contour plots", command=self.plot_contour_button_event
        )
        self.contour_button.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        #
        # create importance button
        self.importance_button = customtkinter.CTkButton(
            master=self.analysis_hyperparameter_frame, text="Importance", command=self.plot_importance_button_event
        )
        self.importance_button.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

    def load_button_event(self):
        filename = load_file_dialog()
        if filename:
            (
                self.spot_tuner,
                self.fun_control,
                self.design_control,
                self.surrogate_control,
                self.optimizer_control,
            ) = load_experiment_spot(filename)
            #
            self.scenario_frame.set_selected_optionmenu_item(self.fun_control["scenario"])
            self.task_frame.set_selected_optionmenu_item(self.fun_control["task"])
            print(f'Task set to loaded tast:{self.fun_control["task"]}')
            self.change_task_event(self.fun_control["task"])
            #
            self.select_core_model_frame.set_selected_optionmenu_item(self.fun_control["core_model_name"])
            print(f'Core model set to loaded core model:{self.fun_control["core_model_name"]}')
            self.core_model_name = self.fun_control["core_model_name"]
            #
            if "prep_models" in self.scenario_dict[self.task_name]:
                self.select_prep_model_frame.set_selected_optionmenu_item(self.fun_control["prep_model_name"])
                print(f'Prep model set to loaded prep model:{self.fun_control["prep_model_name"]}')
                self.prep_model_name = self.fun_control["prep_model_name"]
            #
            if "scalers" in self.scenario_dict[self.task_name]:
                self.select_scaler_frame.set_selected_optionmenu_item(self.fun_control["scaler_name"])
                print(f'Scaler set to loaded scaler:{self.fun_control["scaler_name"]}')
                self.scaler_name = self.fun_control["scaler_name"]
            #
            self.select_data_frame.set_selected_optionmenu_item(self.fun_control["data_set_name"])
            print(f'Data set set to loaded data set:{self.fun_control["data_set_name"]}')
            self.data_set_name = self.fun_control["data_set_name"]
            #
            self.select_metric_sklearn_levels_frame.set_selected_optionmenu_item(
                self.fun_control["metric_sklearn_name"]
            )
            print(f'Metric set to loaded metric:{self.fun_control["metric_sklearn_name"]}')
            self.metric_sklearn_name = self.fun_control["metric_sklearn_name"]
            #
            self.n_total = self.fun_control["n_total"]
            if self.n_total is None:
                self.n_total = "None"
            self.n_total_entry.delete(0, "end")
            self.n_total_entry.insert(0, self.n_total)
            #
            self.test_size = self.fun_control["test_size"]
            self.test_size_entry.delete(0, "end")
            self.test_size_entry.insert(0, self.test_size)
            #
            self.shuffle = self.fun_control["shuffle"]
            self.shuffle_checkbox.deselect()
            if self.shuffle:
                self.shuffle_checkbox.select()
            #
            self.max_time = self.fun_control["max_time"]
            self.max_time_entry.delete(0, "end")
            self.max_time_entry.insert(0, self.max_time)
            #
            self.fun_evals = self.fun_control["fun_evals"]
            if not isinstance(self.fun_evals, int):
                self.fun_evals = "inf"
            self.fun_evals_entry.delete(0, "end")
            self.fun_evals_entry.insert(0, self.fun_evals)
            #
            self.init_size = self.design_control["init_size"]
            self.init_size_entry.delete(0, "end")
            self.init_size_entry.insert(0, self.init_size)
            #
            self.lambda_min_max = [self.surrogate_control["min_Lambda"], self.surrogate_control["max_Lambda"]]
            self.lambda_min_max_entry.delete(0, "end")
            self.lambda_min_max_entry.insert(0, f"{self.lambda_min_max[0]}, {self.lambda_min_max[1]}")
            #
            self.max_sp = self.fun_control["max_surrogate_points"]
            self.max_sp_entry.delete(0, "end")
            self.max_sp_entry.insert(0, self.max_sp)
            #
            self.seed = self.fun_control["seed"]
            self.seed_entry.delete(0, "end")
            self.seed_entry.insert(0, self.seed)
            #
            self.noise = self.fun_control["noise"]
            self.noise_checkbox.deselect()
            if self.noise:
                self.noise_checkbox.select()
            #
            # ----------------- River specific ----------------- #
            if self.fun_control["scenario"] == "river":
                self.weights = self.fun_control["weights_entry"]
                self.weights_entry.delete(0, "end")
                self.weights_entry.insert(0, self.weights)
                #
                self.horizon = self.fun_control["horizon"]
                self.horizon_entry.delete(0, "end")
                self.horizon_entry.insert(0, self.horizon)
                #
                self.oml_grace_period = self.fun_control["oml_grace_period"]
                if self.oml_grace_period is None:
                    self.oml_grace_period = "None"
                self.oml_grace_period_entry.delete(0, "end")
                self.oml_grace_period_entry.insert(0, self.oml_grace_period)
            #
            # ----------------- Hyperparameter ----------------- #
            self.num_hp_frame.destroy()
            self.create_num_hp_frame(dict=self.fun_control["core_model_hyper_dict"])
            self.cat_hp_frame.destroy()
            self.create_cat_hp_frame(dict=self.fun_control["core_model_hyper_dict"])
            #
            # ----------------- Run Options ----------------- #
            self.experiment_name = self.fun_control["PREFIX"]
            self.loaded_label.configure(text=self.experiment_name)
            self.experiment_name_entry.delete(0, "end")
            self.experiment_name_entry.insert(0, self.experiment_name)

    def create_experiment_eval_frame(self):
        if self.scenario == "river":
            # ................. Experiment_Eval Frame .......................................#
            # create experiment_eval frame with widgets in experiment_main frame
            self.experiment_eval_frame = customtkinter.CTkFrame(self.experiment_main_frame, corner_radius=6)
            self.experiment_eval_frame.grid(row=4, column=0, sticky="ew")
            #
            # experiment_eval frame title
            self.experiment_eval_frame_title = customtkinter.CTkLabel(
                self.experiment_eval_frame,
                text="Eval Options",
                font=customtkinter.CTkFont(weight="bold"),
                corner_radius=6,
            )
            self.experiment_eval_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
            #
            # weights entry in experiment_model frame
            self.weights_label = customtkinter.CTkLabel(self.experiment_eval_frame, text="weights", corner_radius=6)
            self.weights_label.grid(row=1, column=0, padx=0, pady=(10, 0), sticky="w")
            self.weights_var = customtkinter.StringVar(value="1000, 1, 1")
            self.weights_entry = customtkinter.CTkEntry(
                self.experiment_eval_frame, textvariable=self.weights_var, width=self.entry_width
            )
            self.weights_entry.grid(row=1, column=1, padx=0, pady=10, sticky="w")
            # horizon entry in experiment_model frame
            self.horizon_label = customtkinter.CTkLabel(self.experiment_eval_frame, text="horizon", corner_radius=6)
            self.horizon_label.grid(row=2, column=0, padx=0, pady=(10, 0), sticky="w")
            self.horizon_var = customtkinter.StringVar(value="10")
            self.horizon_entry = customtkinter.CTkEntry(
                self.experiment_eval_frame, textvariable=self.horizon_var, width=self.entry_width
            )
            self.horizon_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
            # oml_grace_periond entry in experiment_model frame
            self.oml_grace_period_label = customtkinter.CTkLabel(
                self.experiment_eval_frame, text="oml_grace_period", corner_radius=6
            )
            self.oml_grace_period_label.grid(row=3, column=0, padx=0, pady=(10, 0), sticky="w")
            self.oml_grace_period_var = customtkinter.StringVar(value="None")
            self.oml_grace_period_entry = customtkinter.CTkEntry(
                self.experiment_eval_frame, textvariable=self.oml_grace_period_var, width=self.entry_width
            )
            self.oml_grace_period_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        elif self.scenario == "lightning" or "sklearn":
            self.experiment_eval_frame.destroy()
