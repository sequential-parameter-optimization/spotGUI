import tkinter as tk
import tkinter.messagebox
import customtkinter
import pprint
import webbrowser

import numpy as np
import copy
import sys
from PIL import Image
from spotPython.utils.init import (fun_control_init,
                                   design_control_init,
                                   surrogate_control_init,
                                   optimizer_control_init)

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
)
from spotRiver.data.selector import get_river_dataset_from_name
from spotPython.utils.convert import map_to_True_False
from spotRiver.utils.data_conversion import split_df
from spotPython.hyperparameters.values import add_core_model_to_fun_control, set_control_hyperparameter_value
from spotPython.utils.eda import gen_design_table
from spotRiver.fun.hyperriver import HyperRiver

# customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
# customtkinter.set_task("Binary Classification")  # Tasks: "Binary Classification", "Regression"
# customtkinter.set_core_model("System")
# customtkinter.set_prep_model("System")

# customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"


class SelectScrollableComboBoxFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, item_list, item_default, command=None, **kwargs):
        super().__init__(master, **kwargs)

        self.combobox_var = customtkinter.StringVar(value=item_default)
        combobox = customtkinter.CTkComboBox(self,
                                             values=item_list,
                                             command=self.combobox_callback,
                                             variable=self.combobox_var)
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

        self.title = customtkinter.CTkLabel(self, text=self.title, corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        print(f"item_list: {item_list}")

        if item_default is None:
            item_default = item_list[0]
        self.optionmenu_var = customtkinter.StringVar(value=item_default)
        optionmenu = customtkinter.CTkOptionMenu(self,
                                                 values=item_list,
                                                 command=command,
                                                 variable=self.optionmenu_var)
        optionmenu.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.optionmenu_var.set(item_default)

    def get_selected_optionmenu_item(self):
        return self.optionmenu_var.get()


class NumHyperparameterFrame(customtkinter.CTkFrame):
    def __init__(self, master, command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.command = command
        self.hp_list = []
        self.default_list = []
        self.lower_list = []
        self.upper_list = []
        self.transform_list = []
        self.level_list = []

    def add_header(self):
        header_hp = customtkinter.CTkLabel(self, text="Hyperparameter", corner_radius=6)
        header_hp.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        header_hp = customtkinter.CTkLabel(self, text="Default", corner_radius=6)
        header_hp.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="ew")
        header_hp = customtkinter.CTkLabel(self, text="Lower",  corner_radius=6)
        header_hp.grid(row=0, column=2, padx=10, pady=(10, 0), sticky="ew")
        header_hp = customtkinter.CTkLabel(self, text="Upper", corner_radius=6)
        header_hp.grid(row=0, column=3, padx=10, pady=(10, 0), sticky="ew")
        header_hp = customtkinter.CTkLabel(self, text="Transform",  corner_radius=6)
        header_hp.grid(row=0, column=4, padx=10, pady=(10, 0), sticky="ew")

    def add_num_item(self, hp, default, lower, upper, transform):
        self.hp_col = customtkinter.CTkLabel(self, text=hp, compound="left", padx=5, anchor="w")
        self.default_col = customtkinter.CTkLabel(self, text=default, compound="left", padx=5, anchor="w")
        self.lower_col = customtkinter.CTkEntry(self)
        self.lower_col.insert(0, str(lower))
        self.upper_col = customtkinter.CTkEntry(self)
        self.upper_col.insert(0, str(upper))
        self.transform_col = customtkinter.CTkLabel(self,
                                                text=transform,
                                                compound="left",
                                                padx=5,
                                                anchor="w")

        self.hp_col.grid(row=1+len(self.hp_list), column=0, pady=(0, 10), sticky="w")
        self.default_col.grid(row=1+len(self.default_list), column=1, pady=(0, 10), sticky="w")
        self.lower_col.grid(row=1+len(self.lower_list), column=2, pady=(0, 10), sticky="w")
        self.upper_col.grid(row=1+len(self.upper_list), column=3, pady=(0, 10), sticky="w")
        self.transform_col.grid(row=1+len(self.transform_list), column=4, pady=(0, 10), padx=5)
        self.hp_list.append(self.hp_col)
        self.default_list.append(self.default_col)
        self.lower_list.append(self.lower_col)
        self.upper_list.append(self.upper_col)
        self.transform_list.append(self.transform_col)

    def get_num_item(self):
        # get the values from self.lower_col.get() and self.upper_col.get() and put them in a 
        # dictionary with the corresponding hyperparameter hh as key
        num_hp_dict = {}
        for label, default, lower, upper, transform in zip(self.hp_list,
                                                                self.default_list,
                                                                self.lower_list,
                                                                self.upper_list,
                                                                self.transform_list):
            num_hp_dict[label.cget("text")] = dict(
                                                lower=lower.get(),
                                                upper=upper.get(),
                                              )
        return num_hp_dict

    def remove_num_item(self, item):
        for label, default, lower, upper, transform in zip(self.hp_list,
                                                                self.default_list,
                                                                self.lower_list,
                                                                self.upper_list,
                                                                self.transform_list):
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
        header_hp = customtkinter.CTkLabel(self, text="Hyperparameter",  corner_radius=6)
        header_hp.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        header_hp = customtkinter.CTkLabel(self, text="Default", corner_radius=6)
        header_hp.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="ew")
        header_hp = customtkinter.CTkLabel(self, text="Levels",  corner_radius=6)
        header_hp.grid(row=0, column=2, padx=10, pady=(10, 0), sticky="ew")

    def add_cat_item(self, hp, default, levels, transform):
        self.hp_col = customtkinter.CTkLabel(self, text=hp, compound="left", padx=5, anchor="w")
        self.default_col = customtkinter.CTkLabel(self, text=default, compound="left", padx=5, anchor="w")
        self.levels_col = customtkinter.CTkTextbox(self, width=400, height=1)
        string_items = ' '.join(levels)
        self.levels_col.insert("0.0", string_items)

        self.hp_col.grid(row=1+len(self.hp_list), column=0, pady=(0, 10), sticky="w")
        self.default_col.grid(row=1+len(self.default_list), column=1, pady=(0, 10), sticky="w")
        self.levels_col.grid(row=1+len(self.levels_list), column=2, pady=(0, 10), sticky="w")
        self.hp_list.append(self.hp_col)
        self.default_list.append(self.default_col)
        self.levels_list.append(self.levels_col)

    def get_cat_item(self):
        # get the values from self.levels_col.get() and put them in a
        # dictionary with the corresponding hyperparameter hh as key
        num_hp_dict = {}
        for label, default, levels in zip(self.hp_list,
                                        self.default_list,
                                        self.levels_list):
            num_hp_dict[label.cget("text")] = dict(
                                                levels=levels.get("0.0", "end-1c"),
                                              )
        return num_hp_dict

    def remove_cat_item(self, item):
        for label, default, levels, in zip(self.hp_list,
                                           self.default_list,
                                           self.level_list):
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
        self.geometry(f"{1720}x{1020}")
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.rhd = RiverHyperDict()
        self.task_name = "regression_tab"
        self.task_dict = get_task_dict()
        pprint.pprint(self.task_dict)
        self.core_model_name = self.task_dict[self.task_name]["core_model_names"][0]

        # ---------------- Sidebar Frame --------------------------------------- #
        # create sidebar frame with widgets in row 0 and column 0
        self.sidebar_frame = customtkinter.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        # Inside the sidebar frame
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame,
                                                 text="SpotRiver",
                                                 font=customtkinter.CTkFont(size=20,
                                                                            weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=10, pady=2.5)
        #
        # create task frame inside sidebar frame
        self.task_frame = SelectOptionMenuFrame(master=self.sidebar_frame,
                                                command=self.change_task_event,
                                                item_list=["Binary Classification",
                                                           "Regression"],
                                                item_default="Regression",
                                                title="Select Task")
        self.task_frame.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")
        self.task_frame.configure(width=500)
        #
        # create core model frame inside sidebar frame
        self.create_core_model_frame(row=2, column=0)
        # create select prep model frame inside sidebar frame
        self.select_prep_model_frame = SelectOptionMenuFrame(master=self.sidebar_frame,
                                                           command=self.select_prep_model_frame_event,
                                                           item_list=self.task_dict[self.task_name]["prep_models"],
                                                           item_default=None,
                                                           title="Select Prep Model")
        self.select_prep_model_frame.grid(row=3, column=0, padx=15, pady=15, sticky="nsew")
        self.select_prep_model_frame.configure(width=500)
        # select data frame in data main frame
        self.create_select_data_frame(row=4, column=0)
        #
        # create select metric levels frame inside sidebar frame
        self.select_metric_levels_frame = SelectOptionMenuFrame(master=self.sidebar_frame,
                                                           command=self.select_metric_levels_frame_event,
                                                           item_list=self.task_dict[self.task_name]["metric_levels"],
                                                           item_default=None,
                                                           title="Select Metric")
        self.select_metric_levels_frame.grid(row=5, column=0, padx=15, pady=15, sticky="nsew")
        self.select_metric_levels_frame.configure(width=500)
        # create appearance mode frame
        self.appearance_frame = SelectOptionMenuFrame(master=self.sidebar_frame,
                                                width=500,
                                                command=self.change_appearance_mode_event,
                                                item_list=["Light", "Dark", "System"],
                                                item_default="System",
                                                title="Appearance Mode")
        self.appearance_frame.grid(row=6, column=0, padx=15, pady=15, sticky="nsew")
        # self.appearance_frame.configure(width=500)

        # ----------------- Experiment_Main Frame -------------------------------------- #
        # create experiment main frame with widgets in row 0 and column 1
        self.experiment_main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.experiment_main_frame.grid(row=0, column=1, sticky="nsew")
        #
        # experiment frame title in experiment main frame
        self.experiment_main_frame_title = customtkinter.CTkLabel(self.experiment_main_frame,
                                                            text="Experiment Options",
                                                            font=customtkinter.CTkFont(size=20, weight="bold"),
                                                            corner_radius=6)
        self.experiment_main_frame_title.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        #
        # ................. Experiment_Data Frame .......................................#
        # create experiment data_frame with widgets in experiment_main frame
        self.experiment_data_frame = customtkinter.CTkFrame(self.experiment_main_frame, corner_radius=6)
        self.experiment_data_frame.grid(row=1, column=0, sticky="ew")
        #
        # experiment_data frame title
        self.experiment_data_frame_title = customtkinter.CTkLabel(self.experiment_data_frame,
                                                            text="Data Options",
                                                            font=customtkinter.CTkFont(weight="bold"),
                                                            corner_radius=6)
        self.experiment_data_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        #
        # n_total entry in experiment_data frame
        self.n_total_label = customtkinter.CTkLabel(self.experiment_data_frame,
                                                    text="n_total", corner_radius=6)
        self.n_total_label.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.n_total_var = customtkinter.StringVar(value="All")
        self.n_total_entry_frame = customtkinter.CTkEntry(self.experiment_data_frame,
                                                          textvariable=self.n_total_var)
        self.n_total_entry_frame.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        #
        # test_size entry in experiment_data frame
        self.test_size_label = customtkinter.CTkLabel(self.experiment_data_frame,
                                                    text="test_size", corner_radius=6)
        self.test_size_label.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.test_size_var = customtkinter.StringVar(value="0.3")
        self.test_size_entry_frame = customtkinter.CTkEntry(self.experiment_data_frame,
                                                          textvariable=self.test_size_var)
        self.test_size_entry_frame.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        #
        # shuffle data in experiment_data frame
        self.shuffle_var = customtkinter.StringVar(value="False")
        self.shuffle_checkbox = customtkinter.CTkCheckBox(self.experiment_data_frame,
                                             text="ShuffleData",
                                             command=None,
                                             variable=self.shuffle_var,
                                             onvalue="True",
                                             offvalue="False")
        self.shuffle_checkbox.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="w")
        # ................. Experiment_Model Frame .......................................#
        # create experiment_model frame with widgets in experiment_main frame
        self.experiment_model_frame = customtkinter.CTkFrame(self.experiment_main_frame, corner_radius=6)
        self.experiment_model_frame.grid(row=3, column=0, sticky="ew")
        #
        # experiment_model frame title
        self.experiment_model_frame_title = customtkinter.CTkLabel(self.experiment_model_frame,
                                                            text="Model Options",
                                                            font=customtkinter.CTkFont(weight="bold"),
                                                            corner_radius=6)
        self.experiment_model_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        #
        # max_time entry in experiment_model frame
        self.max_time_label = customtkinter.CTkLabel(self.experiment_model_frame,
                                                    text="max_time", corner_radius=6)
        self.max_time_label.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.max_time_var = customtkinter.StringVar(value="1")
        self.max_time_entry_frame = customtkinter.CTkEntry(self.experiment_model_frame,
                                                          textvariable=self.max_time_var)
        self.max_time_entry_frame.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        #
        self.fun_evals_label = customtkinter.CTkLabel(self.experiment_model_frame,
                                                    text="fun_evals", corner_radius=6)
        self.fun_evals_label.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.fun_evals_var = customtkinter.StringVar(value="30")
        self.fun_evals_entry_frame = customtkinter.CTkEntry(self.experiment_model_frame,
                                                          textvariable=self.fun_evals_var)
        self.fun_evals_entry_frame.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        #
        # init_size entry in experiment_model frame
        self.init_size_label = customtkinter.CTkLabel(self.experiment_model_frame,
                                                    text="init_size", corner_radius=6)
        self.init_size_label.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.init_size_var = customtkinter.StringVar(value="5")
        self.init_size_entry_frame = customtkinter.CTkEntry(self.experiment_model_frame,
                                                          textvariable=self.init_size_var)
        self.init_size_entry_frame.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        #
        self.lambda_min_max_label = customtkinter.CTkLabel(self.experiment_model_frame,
                                                    text="lambda_min_max", corner_radius=6)
        self.lambda_min_max_label.grid(row=4, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.lambda_min_max_var = customtkinter.StringVar(value="1e-3, 1e2")
        self.lambda_min_max_entry_frame = customtkinter.CTkEntry(self.experiment_model_frame,
                                                          textvariable=self.lambda_min_max_var)
        self.lambda_min_max_entry_frame.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        #
        self.max_sp_label = customtkinter.CTkLabel(self.experiment_model_frame,
                                                    text="max_sp", corner_radius=6)
        self.max_sp_label.grid(row=5, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.max_sp_var = customtkinter.StringVar(value="30")
        self.max_sp_entry_frame = customtkinter.CTkEntry(self.experiment_model_frame,
                                                          textvariable=self.max_sp_var)
        self.max_sp_entry_frame.grid(row=5, column=1, padx=10, pady=10, sticky="w")
        #
        self.seed_label = customtkinter.CTkLabel(self.experiment_model_frame,
                                                    text="seed", corner_radius=6)
        self.seed_label.grid(row=6, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.seed_var = customtkinter.StringVar(value="123")
        self.seed_entry_frame = customtkinter.CTkEntry(self.experiment_model_frame,
                                                          textvariable=self.seed_var)
        self.seed_entry_frame.grid(row=6, column=1, padx=10, pady=10, sticky="w")
        #
        # noise data in experiment_model frame
        self.noise_var = customtkinter.StringVar(value="True")
        self.noise_checkbox = customtkinter.CTkCheckBox(self.experiment_model_frame,
                                             text="noise",
                                             command=None,
                                             variable=self.noise_var,
                                             onvalue="True",
                                             offvalue="False")
        self.noise_checkbox.grid(row=7, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # ................. Experiment_Eval Frame .......................................#
        # create experiment_eval frame with widgets in experiment_main frame
        self.experiment_eval_frame = customtkinter.CTkFrame(self.experiment_main_frame, corner_radius=6)
        self.experiment_eval_frame.grid(row=4, column=0, sticky="ew")
        #
        # experiment_eval frame title
        self.experiment_eval_frame_title = customtkinter.CTkLabel(self.experiment_eval_frame,
                                                            text="Eval Options",
                                                            font=customtkinter.CTkFont(weight="bold"),
                                                            corner_radius=6)
        self.experiment_eval_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        #
        # weights entry in experiment_model frame
        self.weights_label = customtkinter.CTkLabel(self.experiment_eval_frame,
                                                    text="weights", corner_radius=6)
        self.weights_label.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.weights_var = customtkinter.StringVar(value="1000, 1, 1")
        self.weights_entry_frame = customtkinter.CTkEntry(self.experiment_eval_frame,
                                                          textvariable=self.weights_var)
        self.weights_entry_frame.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        # horizon entry in experiment_model frame
        self.horizon_label = customtkinter.CTkLabel(self.experiment_eval_frame,
                                                    text="horizon", corner_radius=6)
        self.horizon_label.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.horizon_var = customtkinter.StringVar(value="10")
        self.horizon_entry_frame = customtkinter.CTkEntry(self.experiment_eval_frame,
                                                          textvariable=self.horizon_var)
        self.horizon_entry_frame.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        # oml_grace_periond entry in experiment_model frame
        self.oml_grace_period_label = customtkinter.CTkLabel(self.experiment_eval_frame,
                                                    text="oml_grace_period", corner_radius=6)
        self.oml_grace_period_label.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.oml_grace_period_var = customtkinter.StringVar(value="None")
        self.oml_grace_period_entry_frame = customtkinter.CTkEntry(self.experiment_eval_frame,
                                                          textvariable=self.oml_grace_period_var)
        self.oml_grace_period_entry_frame.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # ------------------ Hyperparameter Main Frame ------------------------------------- #
        # create hyperparameter main frame with widgets in row 0 and column 2
        self.hp_main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.hp_main_frame.grid(row=0, column=2, sticky="nsew")
        # self.hp_main_frame.grid_rowconfigure((0, 1, 2), weight=0)
        #
        # create hyperparameter title frame in hyperparameter main frame
        self.hp_main_frame_title = customtkinter.CTkLabel(self.hp_main_frame,
                                                          text="Hyperparameter",
                                                          font=customtkinter.CTkFont(size=20, weight="bold"),
                                                          corner_radius=6)
        self.hp_main_frame_title.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        #
        self.create_num_hp_frame()
        #
        self.create_cat_hp_frame()

        # ----------------- Execution_Main Frame -------------------------------------- #
        # create execution_main frame with widgets in row 0 and column 3
        self.execution_main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.execution_main_frame.grid(row=0, column=3, sticky="nsew")
        #
        # execution frame title in execution main frame
        self.execution_main_frame_title = customtkinter.CTkLabel(self.execution_main_frame,
                                                            text="Run Options",
                                                            font=customtkinter.CTkFont(size=20, weight="bold"),
                                                            corner_radius=6)
        self.execution_main_frame_title.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        #
        # ................. execution_tb Frame .......................................#
        # create execution_tb frame with widgets in execution_main frame
        self.execution_tb_frame = customtkinter.CTkFrame(self.execution_main_frame, corner_radius=6)
        self.execution_tb_frame.grid(row=1, column=0, sticky="ew")
        #
        # execution_tb frame title
        self.execution_tb_frame_title = customtkinter.CTkLabel(self.execution_tb_frame,
                                                            text="Tensorboard Options",
                                                            font=customtkinter.CTkFont(weight="bold"),
                                                            corner_radius=6)
        self.execution_tb_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # tb_clean in execution_tb frame
        self.tb_clean_var = customtkinter.StringVar(value="True")
        self.tb_clean_checkbox = customtkinter.CTkCheckBox(self.execution_tb_frame,
                                             text="TENSORBOARD_CLEAN",
                                             command=None,
                                             variable=self.tb_clean_var,
                                             onvalue="True",
                                             offvalue="False")
        self.tb_clean_checkbox.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")
        # tb_start in execution_tb frame
        self.tb_start_var = customtkinter.StringVar(value="True")
        self.tb_start_checkbox = customtkinter.CTkCheckBox(self.execution_tb_frame,
                                             text="Start Tensorboard",
                                             command=None,
                                             variable=self.tb_start_var,
                                             onvalue="True",
                                             offvalue="False")
        self.tb_start_checkbox.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="w")
        # tb_stop in execution_tb frame
        self.tb_stop_var = customtkinter.StringVar(value="True")
        self.tb_stop_checkbox = customtkinter.CTkCheckBox(self.execution_tb_frame,
                                             text="Stop Tensorboard",
                                             command=None,
                                             variable=self.tb_stop_var,
                                             onvalue="True",
                                             offvalue="False")
        self.tb_stop_checkbox.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        self.browser_link_label = customtkinter.CTkLabel(self.execution_tb_frame,
                                                    text="Open http://localhost:6006",
                                                    text_color=("blue", "orange"),
                                                    cursor="hand2",
                                                    corner_radius=6)
        self.browser_link_label.bind("<Button-1>", lambda e: webbrowser.open_new("http://localhost:6006"))
        self.browser_link_label.grid(row=4, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # ................. execution_docs Frame .......................................#
        # create execution_docs frame with widgets in execution_main frame
        self.execution_docs_frame = customtkinter.CTkFrame(self.execution_main_frame, corner_radius=6)
        self.execution_docs_frame.grid(row=2, column=0, sticky="ew")
        #
        # execution_model frame title
        self.execution_docs_frame_title = customtkinter.CTkLabel(self.execution_docs_frame,
                                                            text="Documentation",
                                                            font=customtkinter.CTkFont(weight="bold"),
                                                            corner_radius=6)
        self.execution_docs_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        # spot_doc entry in execution_model frame
        self.spot_link_label = customtkinter.CTkLabel(self.execution_docs_frame,
                                                    text="spotPython documentation",
                                                    text_color=("blue", "orange"),
                                                    cursor="hand2",
                                                    corner_radius=6)
        self.spot_link_label.bind("<Button-1>", lambda e: webbrowser.open_new("https://sequential-parameter-optimization.github.io/spotPython/"))
        self.spot_link_label.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # spotriver_doc entry in execution_model frame
        self.spotriver_link_label = customtkinter.CTkLabel(self.execution_docs_frame,
                                                    text="spotRiver documentation",
                                                    text_color=("blue", "orange"),
                                                    cursor="hand2",
                                                    corner_radius=6)
        self.spotriver_link_label.bind("<Button-1>", lambda e: webbrowser.open_new("https://sequential-parameter-optimization.github.io/spotRiver/"))
        self.spotriver_link_label.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # river_link entry in execution_model frame
        self.river_link_label = customtkinter.CTkLabel(self.execution_docs_frame,
                                                    text="River documentation",
                                                    text_color=("blue", "orange"),
                                                    cursor="hand2",
                                                    corner_radius=6)
        self.river_link_label.bind("<Button-1>", lambda e: webbrowser.open_new("https://riverml.xyz/latest/api/overview/"))
        self.river_link_label.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="w")

        #
        # ................. Execution_Experiment_Name Frame .......................................#
        # create experiment data_frame with widgets in experiment_main frame
        self.experiment_name_frame = customtkinter.CTkFrame(self.execution_main_frame, corner_radius=6)
        self.experiment_name_frame.grid(row=3, column=0, sticky="ew")
        #
        # experiment_data frame title
        self.experiment_name_frame_title = customtkinter.CTkLabel(self.experiment_name_frame,
                                                            text="Experiment Name",
                                                            font=customtkinter.CTkFont(weight="bold"),
                                                            corner_radius=6)
        self.experiment_name_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        #
        # experiment_name entry in experiment_name frame
        self.experiment_name_var = customtkinter.StringVar(value="000")
        self.experiment_name_entry = customtkinter.CTkEntry(self.experiment_name_frame,
                                                          textvariable=self.experiment_name_var)
        self.experiment_name_entry.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        # ................. Run_Experiment_Name Frame .......................................#
        # create experiment_run_frame with widgets in experiment_main frame
        self.experiment_run_frame = customtkinter.CTkFrame(self.execution_main_frame, corner_radius=6)
        self.experiment_run_frame.grid(row=4, column=0, sticky="ew")
        #
        # experiment_data frame title
        self.experiment_run_frame_title = customtkinter.CTkLabel(self.experiment_run_frame,
                                                            text="Execute",
                                                            font=customtkinter.CTkFont(weight="bold"),
                                                            corner_radius=6)
        self.experiment_run_frame_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        #
        # create plot data button
        self.plot_data_button = customtkinter.CTkButton(master=self.experiment_run_frame,
                                                        text="Plot Data",
                                                        command=self.plot_data_button_event)
        self.plot_data_button.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        # create save button
        self.save_button = customtkinter.CTkButton(master=self.experiment_run_frame,
                                                  text="Save",
                                                  command=self.save_button_event)
        self.save_button.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        # create load button
        self.load_button = customtkinter.CTkButton(master=self.experiment_run_frame,
                                                  text="Load",
                                                  command=self.load_button_event)
        self.load_button.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        # create run button
        self.run_button = customtkinter.CTkButton(master=self.experiment_run_frame,
                                                  text="Run",
                                                  command=self.run_button_event)
        self.run_button.grid(row=4, column=0, sticky="ew", padx=10, pady=10)

        # ------------------- Textbox Frame ------------------------------------ #
        # create textbox in row 1 and column 0
        self.textbox = customtkinter.CTkTextbox(self)
        self.textbox.grid(row=1, column=0, columnspan=4, padx=(20, 0), pady=(20, 0), sticky="nsew")

    def label_button_frame_event(self, item):
        print(f"label button frame clicked: {item}")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        print(f"Appearance Mode changed to: {new_appearance_mode}")
        customtkinter.set_appearance_mode(new_appearance_mode)

    def select_data_frame_event(self, new_data: str):
        print(f"Data modified: {new_data}")
        print(f"Data Selection modified: {self.select_data_frame.get_selected_optionmenu_item()}")

    def create_num_hp_frame(self):
        # create new num_hp_frame
        self.num_hp_frame = NumHyperparameterFrame(master=self.hp_main_frame,
                                                   width=640,
                                                   command=self.label_button_frame_event)

        self.num_hp_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.num_hp_frame.add_header()
        print(f"self.core_model_name: {self.core_model_name}")
        coremodel, core_model_instance = get_core_model_from_name(self.core_model_name)
        dict = self.rhd.hyper_dict[coremodel]
        pprint.pprint(dict)
        for i, (key, value) in enumerate(dict.items()):
            if (dict[key]["type"] == "int" or dict[key]["type"] == "float"
                or dict[key]["core_model_parameter_type"] == "bool"):
                self.num_hp_frame.add_num_item(hp=key,
                                               default=value["default"],
                                               lower=value["lower"],
                                               upper=value["upper"],
                                               transform=value["transform"])

    def create_cat_hp_frame(self):
        self.cat_hp_frame = CatHyperparameterFrame(master=self.hp_main_frame,
                                                   command=self.label_button_frame_event)
        self.cat_hp_frame.grid(row=2, column=0, padx=0, pady=0, sticky="nsew")
        print(f"self.core_model_name: {self.core_model_name}")
        coremodel, core_model_instance = get_core_model_from_name(self.core_model_name)
        dict = self.rhd.hyper_dict[coremodel]
        pprint.pprint(dict)
        empty = True
        for i, (key, value) in enumerate(dict.items()):
            if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
                if empty:
                    self.cat_hp_frame.add_header()
                    empty = False
                self.cat_hp_frame.add_cat_item(hp=key,
                                               default=value["default"],
                                               levels=value["levels"],
                                               transform=value["transform"])

    def create_core_model_frame(self, row, column):
        # create new core model frame
        self.select_core_model_frame = SelectOptionMenuFrame(master=self.sidebar_frame,
                                                             command=self.select_core_model_frame_event,
                                                             item_list=self.task_dict[self.task_name]["core_model_names"],
                                                             item_default=None,
                                                             title="Select Core Model")
        self.select_core_model_frame.grid(row=row, column=column, padx=15, pady=15, sticky="nsew")
        self.select_core_model_frame.configure(width=500)
        self.core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()

    def create_select_data_frame(self, row, column):
        self.select_data_frame = SelectOptionMenuFrame(master=self.sidebar_frame,
                                                           command=self.select_data_frame_event,
                                                           item_list=self.task_dict[self.task_name]["datasets"],
                                                           item_default=None,
                                                           title="Select Data")
        self.select_data_frame.grid(row=row, column=column, padx=15, pady=15, sticky="nswe")
        self.select_data_frame.configure(width=500)
        self.data_name = self.select_data_frame.get_selected_optionmenu_item()

    def select_core_model_frame_event(self, new_core_model: str):
        self.core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()
        self.num_hp_frame.destroy()
        self.create_num_hp_frame()
        self.cat_hp_frame.destroy()
        self.create_cat_hp_frame()

    def select_prep_model_frame_event(self, new_prep_model: str):
        print(f"Prep Model modified: {self.select_prep_model_frame.get_selected_optionmenu_item()}")

    def select_metric_levels_frame_event(self, new_metric_levels: str):
        print(f"Metric modified: {self.select_metric_levels_frame.get_selected_optionmenu_item()}")

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
        self.show_data_only = False
        self.run_experiment()

    def save_button_event(self):
        print("Save button clicked")
        self.save_only = True
        self.show_data_only = False
        self.run_experiment()

    def load_button_event(self):
        print("Load button clicked")

    def plot_data_button_event(self):
        print("Plot Data button clicked")
        self.save_only = False
        self.show_data_only = True
        self.run_experiment()

    def check_type(self, value):
        if isinstance(value, (int, np.integer)):
            return "int"
        elif isinstance(value, (float, np.floating)):
            return "float"
        elif isinstance(value, (str, np.str_)):
            return "str"
        elif isinstance(value, (bool, np.bool_)):
            return "bool"
        else:
            return None

    def run_experiment(self):
        noise = map_to_True_False(self.noise_var.get())
        n_total = get_n_total(self.n_total_var.get())
        fun_evals_val = get_fun_evals(self.fun_evals_var.get())
        max_surrogate_points = int(self.max_sp_var.get())
        seed = int(self.seed_var.get())
        test_size = float(self.test_size_var.get())
        core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()
        lbd_min, lbd_max = get_lambda_min_max(self.lambda_min_max_var.get())
        self.prep_model_name = self.select_prep_model_frame.get_selected_optionmenu_item()
        prepmodel = self.check_user_prep_model()
        metric_sklearn = get_metric_sklearn(self.select_metric_levels_frame.get_selected_optionmenu_item())
        weights = get_weights(self.select_metric_levels_frame.get_selected_optionmenu_item(), self.weights_var.get())
        oml_grace_period = get_oml_grace_period(self.oml_grace_period_var.get())
        data_set_name = self.select_data_frame.get_selected_optionmenu_item()
        dataset, n_samples = get_river_dataset_from_name(data_set_name=data_set_name,
                                                         n_total=n_total,
                                                         river_datasets=self.task_dict[self.task_name]["datasets"])
        print(f"dataset: {dataset.head()}")
        val = copy.deepcopy(dataset.iloc[0, -1])
        target_type = self.check_type(val)
        if target_type == "bool" or target_type == "str":
            # convert the target column to 0 and 1
            dataset["y"] = dataset["y"].astype(int)
        shuffle = map_to_True_False(self.shuffle_var.get())
        train, test, n_samples = split_df(dataset=dataset,
                                          test_size=test_size,
                                          target_type=target_type,
                                          seed=seed,
                                          shuffle=shuffle,
                                          stratify=None)
        TENSORBOARD_CLEAN = map_to_True_False(self.tb_clean_var.get())
        tensorboard_start = map_to_True_False(self.tb_start_var.get())
        tensorboard_stop = map_to_True_False(self.tb_stop_var.get())

        # Initialize the fun_control dictionary with the static parameters,
        # i.e., the parameters that are not hyperparameters (depending on the core model)
        fun_control = fun_control_init(
            PREFIX=self.experiment_name_entry.get(),
            TENSORBOARD_CLEAN=TENSORBOARD_CLEAN,
            core_model_name=core_model_name,
            data_set_name=data_set_name,
            fun_evals=fun_evals_val,
            fun_repeats=1,
            horizon=int(self.horizon_var.get()),
            max_surrogate_points=max_surrogate_points,
            max_time=float(self.max_time_var.get()),
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
        coremodel, core_model_instance = get_core_model_from_name(core_model_name)
        add_core_model_to_fun_control(
            core_model=core_model_instance,
            fun_control=fun_control,
            hyper_dict=RiverHyperDict,
            filename=None,
        )
        dict = self.rhd.hyper_dict[coremodel]
        pprint.pprint(dict)
        num_dict = self.num_hp_frame.get_num_item()
        cat_dict = self.cat_hp_frame.get_cat_item()
        for i, (key, value) in enumerate(dict.items()):
            if dict[key]["type"] == "int":
                set_control_hyperparameter_value(
                    fun_control,
                    key,
                    [
                        int(num_dict[key]["lower"]),
                        int(num_dict[key]["upper"]),
                    ],
                )
            if (dict[key]["type"] == "factor") and (dict[key]["core_model_parameter_type"] == "bool"):
                set_control_hyperparameter_value(
                    fun_control,
                    key,
                    [
                        int(num_dict[key]["lower"]),
                        int(num_dict[key]["upper"]),
                    ],
                )
            if dict[key]["type"] == "float":
                set_control_hyperparameter_value(
                    fun_control,
                    key,
                    [
                        float(num_dict[key]["lower"]),
                        float(num_dict[key]["upper"]),
                    ],
                )
            if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
                fle = cat_dict[key]["levels"]
                # convert the string to a list of strings
                fle = fle.split()
                set_control_hyperparameter_value(fun_control, key, fle)
                fun_control["core_model_hyper_dict"][key].update({"upper": len(fle) - 1})
            pprint.pprint(fun_control["core_model_hyper_dict"])
        design_control = design_control_init(
            init_size=int(self.init_size_var.get()), repeats=1,
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
            save_only=self.save_only,
            show_data_only=self.show_data_only,
            fun_control=fun_control,
            design_control=design_control,
            surrogate_control=surrogate_control,
            optimizer_control=optimizer_control,
            fun=HyperRiver(log_level=fun_control["log_level"]).fun_oml_horizon,
            tensorboard_start=tensorboard_start,
            tensorboard_stop=tensorboard_stop,
        )
        if SPOT_PKL_NAME is not None and self.save_only:
            print(f"\nExperiment successfully saved. Configuration saved as: {SPOT_PKL_NAME}")
        elif SPOT_PKL_NAME is not None and not self.save_only:
            print(f"\nExperiment successfully terminated. Result saved as: {SPOT_PKL_NAME}")
        elif self.show_data_only:
            print("\nData shown. No result saved.")
        else:
            print("\nExperiment failed. No result saved.")

    def check_user_prep_model(self):
        if self.prep_model_name.endswith(".py"):
            print(f"prep_model_name = {self.prep_model_name}")
            sys.path.insert(0, "./userPrepModel")
            # remove the file extension from the prep_model_name
            prep_model_name = self.prep_model_name[:-3]
            print(f"prep_model_name = {prep_model_name}")
            __import__(prep_model_name)
            prepmodel = sys.modules[prep_model_name].set_prep_model()
        else:
            prepmodel = get_prep_model(self.prep_model_name)
        return prepmodel

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




if __name__ == "__main__":
    customtkinter.set_appearance_mode("light")
    app = App()
    app.mainloop()