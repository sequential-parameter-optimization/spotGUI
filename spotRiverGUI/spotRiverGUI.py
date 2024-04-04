from concurrent.futures import ThreadPoolExecutor
# from threading import Thread
import tkinter as tk
import time
import json
import threading
import customtkinter
import pprint
import webbrowser
import os
import numpy as np
import copy
import sys
from PIL import Image
from spotPython.utils.init import fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init

from spotRiver.data.river_hyper_dict import RiverHyperDict
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
    get_core_model_from_name,
    get_n_total,
    get_fun_evals,
    get_lambda_min_max,
    get_oml_grace_period,
    get_prep_model,
    get_metric_sklearn,
    get_weights,
    get_kriging_noise,
    get_task_dict,
    show_data,
    show_y_hist,
)
from spotRiver.data.selector import get_river_dataset_from_name
from spotPython.utils.convert import map_to_True_False, check_type
from spotRiver.utils.data_conversion import split_df
from spotPython.utils.numpy2json import NumpyEncoder
from spotPython.hyperparameters.values import (
    add_core_model_to_fun_control,
    set_control_hyperparameter_value,
    update_fun_control_with_hyper_num_cat_dicts,
)
from spotPython.utils.eda import gen_design_table
from spotRiver.fun.hyperriver import HyperRiver
from spotPython.utils.file import load_experiment as load_experiment_spot
from spotPython.utils.file import get_experiment_filename

customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class SelectScrollableComboBoxFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, item_list, item_default, command=None, **kwargs):
        super().__init__(master, **kwargs)

        self.combobox_var = customtkinter.StringVar(value=item_default)
        combobox = customtkinter.CTkComboBox(
            self, values=item_list, command=self.combobox_callback, variable=self.combobox_var
        )
        combobox.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.combobox_var.set(item_default)

    def combobox_callback(self, choice):
        print("combobox dropdown selected:", choice)

    def get_selected_item(self):
        return self.combobox_var.get()


class SelectOptionMenuFrame(customtkinter.CTkFrame):
    def __init__(self, master, title, item_list, item_default, command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.title = title
        self.title_label = customtkinter.CTkLabel(self, text=self.title, corner_radius=6)
        self.title_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        print(f"item_list: {item_list}")

        if item_default is None:
            item_default = item_list[0]
        self.optionmenu_var = customtkinter.StringVar(value=item_default)
        optionmenu = customtkinter.CTkOptionMenu(self, values=item_list, command=command, variable=self.optionmenu_var)
        optionmenu.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.optionmenu_var.set(item_default)

    def get_selected_optionmenu_item(self):
        return self.optionmenu_var.get()

    def set_selected_optionmenu_item(self, item):
        self.optionmenu_var.set(item)


class NumHyperparameterFrame(customtkinter.CTkFrame):
    def __init__(self, master, command=None, entry_width=120, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.entry_width = entry_width

        self.command = command
        self.hp_list = []
        self.default_list = []
        self.lower_list = []
        self.upper_list = []
        self.transform_list = []
        self.level_list = []

    def add_header(self):
        header_hp = customtkinter.CTkLabel(self, text="Hyperparameter", corner_radius=6)
        header_hp.grid(row=0, column=0, padx=0, pady=(10, 0), sticky="w")
        header_hp = customtkinter.CTkLabel(self, text="Default", corner_radius=6)
        header_hp.grid(row=0, column=1, padx=0, pady=(10, 0), sticky="w")
        header_hp = customtkinter.CTkLabel(self, text="Lower", corner_radius=6)
        header_hp.grid(row=0, column=2, padx=0, pady=(10, 0), sticky="w")
        header_hp = customtkinter.CTkLabel(self, text="Upper", corner_radius=6)
        header_hp.grid(row=0, column=3, padx=0, pady=(10, 0), sticky="w")
        header_hp = customtkinter.CTkLabel(self, text="Transform", corner_radius=6)
        header_hp.grid(row=0, column=4, padx=0, pady=(10, 0), sticky="w")

    def add_num_item(self, hp, default, lower, upper, transform):
        self.hp_col = customtkinter.CTkLabel(self, text=hp, compound="left", padx=5, anchor="w")
        self.default_col = customtkinter.CTkLabel(self, text=default, compound="left", padx=5, anchor="w")
        self.lower_col = customtkinter.CTkEntry(self, width=self.entry_width)
        self.lower_col.insert(0, str(lower))
        self.upper_col = customtkinter.CTkEntry(self, width=self.entry_width)
        self.upper_col.insert(0, str(upper))
        self.transform_col = customtkinter.CTkLabel(self, text=transform, compound="left", padx=5, anchor="w")

        self.hp_col.grid(row=1 + len(self.hp_list), column=0, pady=(0, 10), sticky="w")
        self.default_col.grid(row=1 + len(self.default_list), column=1, pady=(0, 10), sticky="w")
        self.lower_col.grid(row=1 + len(self.lower_list), column=2, pady=(0, 10), sticky="w")
        self.upper_col.grid(row=1 + len(self.upper_list), column=3, pady=(0, 10), sticky="w")
        self.transform_col.grid(row=1 + len(self.transform_list), column=4, pady=(0, 10), padx=5, sticky="w")
        self.hp_list.append(self.hp_col)
        self.default_list.append(self.default_col)
        self.lower_list.append(self.lower_col)
        self.upper_list.append(self.upper_col)
        self.transform_list.append(self.transform_col)

    def get_num_item(self) -> dict:
        """
        Get the values from self.hp_list, self.default_list, self.lower_list, self.upper_list,
        and self.transform_list and put lower and upper in a dictionary with the corresponding
        hyperparameter (hp) as key.

        Note:
            Method is designed for numerical parameters.

        Args:
            None

        Returns:
            num_hp_dict (dict): dictionary with hyperparameter as key and values
            as dictionary with lower and upper values.
        """
        num_hp_dict = {}
        for label, default, lower, upper, transform in zip(
            self.hp_list, self.default_list, self.lower_list, self.upper_list, self.transform_list
        ):
            num_hp_dict[label.cget("text")] = dict(
                lower=lower.get(),
                upper=upper.get(),
            )
        return num_hp_dict

    def remove_num_item(self, item):
        for label, default, lower, upper, transform in zip(
            self.hp_list, self.default_list, self.lower_list, self.upper_list, self.transform_list
        ):
            if item == label.cget("text"):
                label.destroy()
                default.destroy()
                lower.destroy()
                upper.destroy()
                transform.destroy()
                self.hp_list.remove(label)
                self.default_list.remove(default)
                self.lower_list.remove(lower)
                self.upper_list.remove(upper)
                self.transform_list.remove(transform)
                return


class CatHyperparameterFrame(customtkinter.CTkFrame):
    def __init__(self, master, command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.command = command
        self.hp_list = []
        self.default_list = []
        self.lower_list = []
        self.upper_list = []
        self.transform_list = []
        self.levels_list = []

    def add_header(self):
        header_hp = customtkinter.CTkLabel(self, text="Hyperparameter", corner_radius=6)
        header_hp.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        header_hp = customtkinter.CTkLabel(self, text="Default", corner_radius=6)
        header_hp.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="ew")
        header_hp = customtkinter.CTkLabel(self, text="Levels", corner_radius=6)
        header_hp.grid(row=0, column=2, padx=10, pady=(10, 0), sticky="ew")

    def add_cat_item(self, hp, default, levels, transform):
        self.hp_col = customtkinter.CTkLabel(self, text=hp, compound="left", padx=5, anchor="w")
        self.default_col = customtkinter.CTkLabel(self, text=default, compound="left", padx=5, anchor="w")
        self.levels_col = customtkinter.CTkTextbox(self, width=400, height=1)
        string_items = " ".join(levels)
        self.levels_col.insert("0.0", string_items)

        self.hp_col.grid(row=1 + len(self.hp_list), column=0, pady=(0, 10), sticky="w")
        self.default_col.grid(row=1 + len(self.default_list), column=1, pady=(0, 10), sticky="w")
        self.levels_col.grid(row=1 + len(self.levels_list), column=2, pady=(0, 10), sticky="w")
        self.hp_list.append(self.hp_col)
        self.default_list.append(self.default_col)
        self.levels_list.append(self.levels_col)

    def get_cat_item(self):
        """
        Get the values self.hp_list, self.default_list, self.levels_list,
        and put lower and upper in a dictionary with the corresponding
        hyperparameter (hp) as key.

        Note:
            Method is designed for categorical parameters.

        Args:
            None

        Returns:
            num_hp_dict (dict): dictionary with hyperparameter as key and values
            as dictionary with lower and upper values.
        """
        cat_hp_dict = {}
        for label, default, levels in zip(self.hp_list, self.default_list, self.levels_list):
            cat_hp_dict[label.cget("text")] = dict(
                levels=levels.get("0.0", "end-1c"),
            )
        return cat_hp_dict

    def remove_cat_item(self, item):
        for (
            label,
            default,
            levels,
        ) in zip(self.hp_list, self.default_list, self.level_list):
            if item == label.cget("text"):
                label.destroy()
                default.destroy()
                levels.destroy()
                self.hp_list.remove(label)
                self.default_list.remove(default)
                self.lower_list.remove(levels)
                return


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("spotRiver GUI")
        self.geometry(f"{1600}x{900}")
        self.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)
        self.entry_width = 80
        # dictionary name for the database
        # similar for all spotRiver experiments
        self.db_dict_name = "spotRiver_db.json"
        # name of the progress file
        self.progress_file = "progress.txt"
        # if the progress file exists, delete it
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)

        current_path = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_path, "images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "spotlogo.png")), size=(85, 37))

        self.rhd = RiverHyperDict()
        self.task_name = "regression_tab"
        self.task_dict = get_task_dict()
        pprint.pprint(self.task_dict)
        self.core_model_name = self.task_dict[self.task_name]["core_model_names"][0]
        # Uncomment to get user defined core models (not useful for spotRiver):
        # for filename in os.listdir("userModel"):
        #     if filename.endswith(".json"):
        #         self.core_model_name.append(os.path.splitext(filename)[0])

        # ---------------- Sidebar Frame --------------------------------------- #
        # ---------------------------------------------------------------------- #
        # create sidebar frame with widgets in row 0 and column 0
        self.sidebar_frame = customtkinter.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        #
        # Inside the sidebar frame
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="    SPOTRiver",
            image=self.logo_image,
            compound="left",
            font=customtkinter.CTkFont(size=20, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=10, pady=(7.5, 2.5), sticky="ew")
        #
        # ................. Task Frame ....................................... #
        # create task frame inside sidebar frame
        self.task_frame = SelectOptionMenuFrame(
            master=self.sidebar_frame,
            command=self.change_task_event,
            item_list=["Binary Classification", "Regression"],
            item_default="Regression",
            title="Select Task",
        )
        self.task_frame.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")
        self.task_frame.configure(width=500)
        #
        # ................. Core Model Frame ....................................... #
        # create core model frame inside sidebar frame
        self.create_core_model_frame(row=2, column=0)
        #
        # ................. Prep Model Frame ....................................... #
        # create select prep model frame inside sidebar frame
        self.prep_model_values = self.task_dict[self.task_name]["prep_models"]
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
        self.select_prep_model_frame.grid(row=3, column=0, padx=15, pady=15, sticky="nsew")
        self.select_prep_model_frame.configure(width=500)
        #
        #  ................. Data Frame ....................................... #
        # select data frame in data main frame
        self.create_select_data_frame(row=4, column=0)
        #
        # create plot data button
        self.plot_data_button = customtkinter.CTkButton(
            master=self.sidebar_frame, text="Plot Data", command=self.plot_data_button_event
        )
        self.plot_data_button.grid(row=6, column=0, sticky="nsew", padx=10, pady=10)
        #
        # ................. Metric Frame ....................................... #
        # create select metric_sklearn levels frame inside sidebar frame
        self.select_metric_sklearn_levels_frame = SelectOptionMenuFrame(
            master=self.sidebar_frame,
            command=self.select_metric_sklearn_levels_frame_event,
            item_list=self.task_dict[self.task_name]["metric_sklearn_levels"],
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
        #
        # ----------------- Experiment_Main Frame -------------------------------------- #
        # ------------------------------------------------------------------------------ #
        # create experiment main frame with widgets in row 0 and column 1
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
        # ................. Experiment_Eval Frame .......................................#
        # create experiment_eval frame with widgets in experiment_main frame
        self.experiment_eval_frame = customtkinter.CTkFrame(self.experiment_main_frame, corner_radius=6)
        self.experiment_eval_frame.grid(row=4, column=0, sticky="ew")
        #
        # experiment_eval frame title
        self.experiment_eval_frame_title = customtkinter.CTkLabel(
            self.experiment_eval_frame, text="Eval Options", font=customtkinter.CTkFont(weight="bold"), corner_radius=6
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
        #
        # ------------------ Hyperparameter Main Frame ------------------------------------- #
        # create hyperparameter main frame with widgets in row 0 and column 2
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
        #
        # ----------------- Execution_Main Frame -------------------------------------- #
        # create execution_main frame with widgets in row 0 and column 4
        self.execution_main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.execution_main_frame.grid(row=0, column=4, sticky="nsew")
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
        # create save button
        self.save_button = customtkinter.CTkButton(
            master=self.experiment_run_frame, text="Save", command=self.save_button_event
        )
        self.save_button.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        # create run button
        self.run_button = customtkinter.CTkButton(
            master=self.experiment_run_frame, text="Run", command=self.run_button_event
        )
        self.run_button.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

        # ----------------- Analysis_Main Frame -------------------------------------- #
        # create analysis_main frame with widgets in row 0 and column 3
        self.analysis_main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.analysis_main_frame.grid(row=0, column=3, sticky="nsew")
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
            self.analysis_run_frame, text="Loaded experiment", font=customtkinter.CTkFont(weight="bold"), corner_radius=6
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
        # ................. Comparison_Analysis Frame .......................................#
        # create analysis_comparison_frame with widgets in analysis_main frame
        self.analysis_comparison_frame = customtkinter.CTkFrame(self.analysis_main_frame, corner_radius=6)
        self.analysis_comparison_frame.grid(row=4, column=0, sticky="ew")
        #
        # analysis_data frame title
        self.analysis_comparison_frame_title = customtkinter.CTkLabel(
            self.analysis_comparison_frame,
            text="Comparisons",
            font=customtkinter.CTkFont(weight="bold"),
            corner_radius=6,
        )
        self.analysis_comparison_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        # create tuned default button
        self.compare_tuned_default_button = customtkinter.CTkButton(
            master=self.analysis_comparison_frame,
            text="Tuned vs. default",
            command=self.plot_tuned_default_button_event,
        )
        self.compare_tuned_default_button.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        #
        # create actual prediction button
        self.compare_actual_prediction_button = customtkinter.CTkButton(
            master=self.analysis_comparison_frame,
            text="Actual vs. prediction",
            command=self.plot_actual_prediction_button_event,
        )
        self.compare_actual_prediction_button.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        #
        # ................. Hyperparameter_Analysis Frame .......................................#
        # create analysis_hyperparameter_frame with widgets in analysis_main frame
        self.analysis_hyperparameter_frame = customtkinter.CTkFrame(self.analysis_main_frame, corner_radius=6)
        self.analysis_hyperparameter_frame.grid(row=5, column=0, sticky="ew")
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
        #
        # ................. Classification_Analysis Frame .......................................#
        # create analysis_classification_frame with widgets in analysis_main frame
        self.analysis_classification_frame = customtkinter.CTkFrame(self.analysis_main_frame, corner_radius=6)
        self.analysis_classification_frame.grid(row=6, column=0, sticky="ew")
        #
        # analysis_data frame title
        self.analysis_classification_frame_title = customtkinter.CTkLabel(
            self.analysis_classification_frame,
            text="Classification",
            font=customtkinter.CTkFont(weight="bold"),
            corner_radius=6,
        )
        self.analysis_classification_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # create confusion plot  button
        self.confusion_button = customtkinter.CTkButton(
            master=self.analysis_classification_frame, text="Confusion matrix", command=self.plot_confusion_button_event
        )
        self.confusion_button.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        #
        # create roc button
        self.roc_button = customtkinter.CTkButton(
            master=self.analysis_classification_frame, text="ROC", command=self.plot_roc_button_event
        )
        self.roc_button.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        #
        # ------------------- Textbox Frame ------------------------------------ #
        # create textbox in row 1 and column 0
        self.textbox = customtkinter.CTkTextbox(self)
        self.textbox.grid(row=1, column=0, columnspan=5, padx=(10, 10), pady=10, sticky="nsew")
        self.textbox.configure(height=20, width=10)
        self.textbox.insert(tk.END, "Welcome to SPOTRiver\n")
        #
        # Start the thread that will update the text area
        # update_thread = threading.Thread(target=self.update_text, daemon=True)
        # update_thread.start()
        # e = ThreadPoolExecutor(max_workers=1)
        # e.submit(self.update_text)
        # e.shutdown(wait=False)

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

    def plot_progress_button_event(self):
        if self.spot_tuner is not None:
            progress_plot(spot_tuner=self.spot_tuner, fun_control=self.fun_control)

    def plot_tuned_default_button_event(self):
        if self.spot_tuner is not None:
            compare_tuned_default(spot_tuner=self.spot_tuner, fun_control=self.fun_control, show=True)

    def plot_actual_prediction_button_event(self):
        if self.spot_tuner is not None:
            actual_vs_prediction(spot_tuner=self.spot_tuner, fun_control=self.fun_control)

    def plot_contour_button_event(self):
        if self.spot_tuner is not None:
            contour_plot(spot_tuner=self.spot_tuner, fun_control=self.fun_control)

    def plot_importance_button_event(self):
        if self.spot_tuner is not None:
            importance_plot(spot_tuner=self.spot_tuner, fun_control=self.fun_control)

    def plot_confusion_button_event(self):
        if self.spot_tuner is not None:
            plot_confusion_matrices(spot_tuner=self.spot_tuner, fun_control=self.fun_control, show=True)

    def plot_roc_button_event(self):
        if self.spot_tuner is not None:
            plot_rocs(spot_tuner=self.spot_tuner, fun_control=self.fun_control, show=True)

    def label_button_frame_event(self, item):
        print(f"label button frame clicked: {item}")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        print(f"Appearance Mode changed to: {new_appearance_mode}")
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

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
        print(f"self.core_model_name: {self.core_model_name}")
        coremodel, core_model_instance = get_core_model_from_name(self.core_model_name)
        if dict is None:
            dict = self.rhd.hyper_dict[coremodel]
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
        print(f"self.core_model_name: {self.core_model_name}")
        coremodel, core_model_instance = get_core_model_from_name(self.core_model_name)
        if dict is None:
            dict = self.rhd.hyper_dict[coremodel]
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
            item_list=self.task_dict[self.task_name]["core_model_names"],
            item_default=None,
            title="Select Core Model",
        )
        self.select_core_model_frame.grid(row=row, column=column, padx=15, pady=15, sticky="nsew")
        self.select_core_model_frame.configure(width=500)
        self.core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()

    def create_select_data_frame(self, row, column):
        data_set_values = copy.deepcopy(self.task_dict[self.task_name]["datasets"])
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

    def select_core_model_frame_event(self, new_core_model: str):
        self.core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()
        self.num_hp_frame.destroy()
        self.create_num_hp_frame()
        self.cat_hp_frame.destroy()
        self.create_cat_hp_frame()

    def select_prep_model_frame_event(self, new_prep_model: str):
        print(f"Prep Model modified: {self.select_prep_model_frame.get_selected_optionmenu_item()}")

    def select_metric_sklearn_levels_frame_event(self, new_metric_sklearn_levels: str):
        print(f"Metric sklearn modified: {self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item()}")
        self.metric_sklearn_name = self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item()

    def change_task_event(self, new_task: str):
        print(f"Task changed to: {new_task}")
        if new_task == "Binary Classification":
            self.task_name = "classification_tab"
        elif new_task == "Regression":
            self.task_name = "regression_tab"
        else:
            print("Error: Task not found")
        self.select_core_model_frame.destroy()
        self.create_core_model_frame(row=2, column=0)
        self.num_hp_frame.destroy()
        self.create_num_hp_frame()
        self.cat_hp_frame.destroy()
        self.create_cat_hp_frame()
        self.select_data_frame.destroy()
        self.create_select_data_frame(row=4, column=0)

    def run_button_event(self):
        self.save_only = False
        self.run_experiment()
        # self.print_tuned_design()

    def save_button_event(self):
        self.save_only = True
        self.run_experiment()

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
            self.task_frame.set_selected_optionmenu_item(self.fun_control["task"])
            self.change_task_event(self.fun_control["task"])
            #
            self.select_core_model_frame.set_selected_optionmenu_item(self.fun_control["core_model_name"])
            self.core_model_name = self.fun_control["core_model_name"]
            #
            self.select_prep_model_frame.set_selected_optionmenu_item(self.fun_control["prep_model_name"])
            self.prep_model_name = self.fun_control["prep_model_name"]
            #
            self.select_data_frame.set_selected_optionmenu_item(self.fun_control["data_set_name"])
            self.data_set_name = self.fun_control["data_set_name"]
            #
            self.select_metric_sklearn_levels_frame.set_selected_optionmenu_item(
                self.fun_control["metric_sklearn_name"]
            )
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
            self.num_hp_frame.destroy()
            self.create_num_hp_frame(dict=self.fun_control["core_model_hyper_dict"])
            self.cat_hp_frame.destroy()
            self.create_cat_hp_frame(dict=self.fun_control["core_model_hyper_dict"])
            #
            self.experiment_name = self.fun_control["PREFIX"]
            self.loaded_label.configure(text=self.experiment_name)
            self.experiment_name_entry.delete(0, "end")
            self.experiment_name_entry.insert(0, self.experiment_name)
            #
            # self.print_tuned_design()

    def plot_data_button_event(self):
        self.prepare_data()
        show_y_hist(train=self.train, test=self.test, target_column="y")
        # TODO: show_data
        # show_data(train=self.train,
        #           test=self.test,
        #           target_column=self.target_column,
        #           n_samples=1000)
        print("\nData shown. No result saved.")

    def prepare_data(self):
        self.data_set_name = self.select_data_frame.get_selected_optionmenu_item()
        dataset, self.n_samples = get_river_dataset_from_name(
            data_set_name=self.data_set_name,
            n_total=get_n_total(self.n_total_var.get()),
            river_datasets=self.task_dict[self.task_name]["datasets"],
        )
        val = copy.deepcopy(dataset.iloc[0, -1])
        self.target_type = check_type(val)
        if self.target_type == "bool" or self.target_type == "str":
            # convert the target column to 0 and 1
            dataset["y"] = dataset["y"].astype(int)
        self.test_size = float(self.test_size_var.get())
        self.seed = int(self.seed_var.get())
        self.shuffle = map_to_True_False(self.shuffle_var.get())
        self.train, self.test, self.n_samples = split_df(
            dataset=dataset,
            test_size=self.test_size,
            target_type=self.target_type,
            seed=self.seed,
            shuffle=self.shuffle,
            stratify=None,
        )

    def run_experiment(self):
        task_name = self.task_frame.get_selected_optionmenu_item()
        #
        core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()
        #
        prep_model_name = self.select_prep_model_frame.get_selected_optionmenu_item()
        prepmodel = self.check_user_prep_model(prep_model_name=prep_model_name)
        #
        self.prepare_data()
        # data_set_name = self.select_data_frame.get_selected_optionmenu_item()
        # dataset, n_samples = get_river_dataset_from_name(data_set_name=data_set_name,
        #                                             n_total=get_n_total(self.n_total_var.get()),
        #                                             river_datasets=self.task_dict[self.task_name]["datasets"])
        #
        metric_sklearn_name = self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item()
        metric_sklearn = get_metric_sklearn(self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item())
        #
        n_total = get_n_total(self.n_total_var.get())
        #
        # test_size = float(self.test_size_var.get())
        #
        # shuffle = map_to_True_False(self.shuffle_var.get())
        #
        max_time = float(self.max_time_var.get())
        #
        fun_evals = get_fun_evals(self.fun_evals_var.get())
        #
        init_size = int(self.init_size_var.get())
        #
        lbd_min, lbd_max = get_lambda_min_max(self.lambda_min_max_var.get())
        #
        max_surrogate_points = int(self.max_sp_var.get())
        #
        # seed = int(self.seed_var.get())
        #
        noise = map_to_True_False(self.noise_var.get())
        #
        weights_entry = self.weights_var.get()
        weights = get_weights(
            self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item(), self.weights_var.get()
        )
        #
        horizon = int(self.horizon_var.get())
        #
        oml_grace_period = get_oml_grace_period(self.oml_grace_period_var.get())
        #
        TENSORBOARD_CLEAN = map_to_True_False(self.tb_clean_var.get())
        tensorboard_start = map_to_True_False(self.tb_start_var.get())
        tensorboard_stop = map_to_True_False(self.tb_stop_var.get())
        PREFIX = self.experiment_name_entry.get()
        #
        # val = copy.deepcopy(dataset.iloc[0, -1])
        # target_type = check_type(val)
        # if target_type == "bool" or target_type == "str":
        #     # convert the target column to 0 and 1
        #     dataset["y"] = dataset["y"].astype(int)
        # train, test, n_samples = split_df(dataset=dataset,
        #                                   test_size=test_size,
        #                                   target_type=target_type,
        #                                   seed=seed,
        #                                   shuffle=shuffle,
        #                                   stratify=None)
        fun_control = fun_control_init(
            PREFIX=PREFIX,
            TENSORBOARD_CLEAN=TENSORBOARD_CLEAN,
            core_model_name=core_model_name,
            data_set_name=self.data_set_name,
            db_dict_name=self.db_dict_name,
            fun_evals=fun_evals,
            fun_repeats=1,
            horizon=horizon,
            max_surrogate_points=max_surrogate_points,
            max_time=max_time,
            metric_sklearn=metric_sklearn,
            metric_sklearn_name=metric_sklearn_name,
            noise=noise,
            n_samples=self.n_samples,
            n_total=n_total,
            ocba_delta=0,
            oml_grace_period=oml_grace_period,
            prep_model=prepmodel,
            prep_model_name=prep_model_name,
            progress_file=self.progress_file,
            seed=self.seed,
            shuffle=self.shuffle,
            task=task_name,
            target_column="y",
            target_type=self.target_type,
            test=self.test,
            test_size=self.test_size,
            train=self.train,
            tolerance_x=np.sqrt(np.spacing(1)),
            verbosity=1,
            weights=weights,
            weights_entry=weights_entry,
            log_level=50,
        )
        coremodel, core_model_instance = get_core_model_from_name(core_model_name)
        add_core_model_to_fun_control(
            core_model=core_model_instance,
            fun_control=fun_control,
            hyper_dict=RiverHyperDict,
            filename=None,
        )
        dict = self.rhd.hyper_dict[coremodel]
        num_dict = self.num_hp_frame.get_num_item()
        cat_dict = self.cat_hp_frame.get_cat_item()
        update_fun_control_with_hyper_num_cat_dicts(fun_control, num_dict, cat_dict, dict)
        pprint.pprint(fun_control["core_model_hyper_dict"])
        design_control = design_control_init(
            init_size=init_size,
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
        # call spot:
        run_spot_python_experiment(
            save_only=self.save_only,
            fun_control=fun_control,
            design_control=design_control,
            surrogate_control=surrogate_control,
            optimizer_control=optimizer_control,
            fun=HyperRiver(log_level=fun_control["log_level"]).fun_oml_horizon,
            tensorboard_start=tensorboard_start,
            tensorboard_stop=tensorboard_stop,
        )
        if self.save_only:
            print("\nExperiment saved.")
        elif not self.save_only:
            print("\nExperiment started.")
        else:
            print("\nExperiment failed. No result saved.")

    def check_user_prep_model(self, prep_model_name):
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
        return prepmodel


# TODO:
# Check the handling of l1/l2 in LogisticRegression. A note (from the River documentation):
# > For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
# Therefore, we set l1 bounds to 0.0:
# modify_hyper_parameter_bounds(fun_control, "l1", bounds=[0.0, 0.0])
# set_control_hyperparameter_value(fun_control, "l1", [0.0, 0.0])
# modify_hyper_parameter_levels(fun_control, "optimizer", ["SGD"])

if __name__ == "__main__":
    customtkinter.set_appearance_mode("light")
    app = App()
    app.mainloop()
