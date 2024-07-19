from spotPython.data.csvdataset import CSVDataset as spotPythonCSVDataset
from spotPython.data.pkldataset import PKLDataset as spotPythonPKLDataset
import tkinter as tk
import customtkinter
import pprint
import os
import numpy as np
import copy
from spotPython.utils.init import fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init
from spotGUI.ctk.CTk import CTkApp, SelectOptionMenuFrame

from spotRiver.hyperdict.river_hyper_dict import RiverHyperDict
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotGUI.tuner.spotRun import (
    save_spot_python_experiment,
    run_spot_python_experiment,
    actual_vs_prediction_river,
    compare_river_tuned_default,
    plot_confusion_matrices_river,
    plot_rocs_river,
    get_n_total,
    get_fun_evals,
    get_lambda_min_max,
    get_oml_grace_period,
    get_weights,
    get_kriging_noise,
    get_scenario_dict,
    show_y_hist,
)
from spotRiver.data.selector import get_river_dataset_from_name
from spotPython.utils.convert import map_to_True_False, set_dataset_target_type, check_type
from spotRiver.utils.data_conversion import split_df
from spotPython.hyperparameters.values import (
    add_core_model_to_fun_control,
    get_core_model_from_name,
    get_river_core_model_from_name,
    get_metric_sklearn,
    update_fun_control_with_hyper_num_cat_dicts,
)
from spotRiver.fun.hyperriver import HyperRiver
from spotPython.fun.hyperlight import HyperLight
from spotPython.fun.hypersklearn import HyperSklearn
from spotPython.utils.metrics import get_metric_sign


class spotPythonApp(CTkApp):
    def __init__(self):
        super().__init__()
        self.title("spotRiver GUI")
        self.logo_text = "    SPOTPython"

        self.scenario = "river"
        self.hyperdict = RiverHyperDict
        self.task_name = "regression_task"
        self.scenario_dict = get_scenario_dict(scenario=self.scenario)
        pprint.pprint(self.scenario_dict)
        # ---------------------------------------------------------------------- #
        # ---------------- 0 Sidebar Frame --------------------------------------- #
        # ---------------------------------------------------------------------- #
        # create sidebar frame with widgets in row 0 and column 0
        self.make_sidebar_frame()
        # ---------------------------------------------------------------------- #
        # ----------------- 1 Experiment_Main Frame ------------------------------ #
        # ---------------------------------------------------------------------- #
        # create experiment main frame with widgets in row 0 and column 1
        #
        self.make_experiment_frame()
        # ---------------------------------------------------------------------- #
        # ------------------ 2 Hyperparameter Main Frame ----------------------- #
        # ---------------------------------------------------------------------- #
        # create hyperparameter main frame with widgets in row 0 and column 2
        self.make_hyperparameter_frame()
        #
        # ---------------------------------------------------------------------- #
        # ----------------- 3 Execution_Main Frame ----------------------------- #
        # ---------------------------------------------------------------------- #
        # create execution_main frame with widgets in row 0 and column 4
        self.make_execution_frame()

        # ---------------------------------------------------------------------- #
        # ----------------- 4 Analysis_Main Frame ------------------------------ #
        # ---------------------------------------------------------------------- #
        # create analysis_main frame with widgets in row 0 and column 3
        self.make_analysis_frame()
        #
        # ................. Comparison_Analysis Frame .......................................#
        # create analysis_comparison_frame with widgets in analysis_main frame
        self.analysis_comparison_frame = customtkinter.CTkFrame(self.analysis_main_frame, corner_radius=6)
        self.analysis_comparison_frame.grid(row=5, column=0, sticky="ew")
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
        self.compare_river_tuned_default_button = customtkinter.CTkButton(
            master=self.analysis_comparison_frame,
            text="Tuned vs. default",
            command=self.plot_tuned_default_button_event,
        )
        self.compare_river_tuned_default_button.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        #
        # create actual prediction button
        self.compare_actual_prediction_button = customtkinter.CTkButton(
            master=self.analysis_comparison_frame,
            text="Actual vs. prediction",
            command=self.plot_actual_prediction_button_event,
        )
        self.compare_actual_prediction_button.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
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

        # ---------------------------------------------------------------------- #
        # ------------------- Textbox Frame ------------------------------------ #
        # ---------------------------------------------------------------------- #
        # create textbox in row 1 and column 0
        self.textbox = customtkinter.CTkTextbox(self)
        self.textbox.grid(row=1, column=0, columnspan=5, padx=(10, 10), pady=10, sticky="nsew")
        self.textbox.configure(height=20, width=10)
        self.textbox.insert(tk.END, "Welcome to SPOTPython\n")
        #
        # ---------------------------------------------------------------------- #

    # -------------- River specific plots ----------------- #
    def plot_tuned_default_button_event(self):
        if self.spot_tuner is not None:
            compare_river_tuned_default(spot_tuner=self.spot_tuner, fun_control=self.fun_control, show=True)

    def plot_actual_prediction_button_event(self):
        if self.spot_tuner is not None:
            actual_vs_prediction_river(spot_tuner=self.spot_tuner, fun_control=self.fun_control)

    def plot_confusion_button_event(self):
        if self.spot_tuner is not None:
            plot_confusion_matrices_river(spot_tuner=self.spot_tuner, fun_control=self.fun_control, show=True)

    def plot_roc_button_event(self):
        if self.spot_tuner is not None:
            plot_rocs_river(spot_tuner=self.spot_tuner, fun_control=self.fun_control, show=True)

    def plot_data_button_event(self):
        train, test, n_samples, target_type = self.prepare_data()
        show_y_hist(train=train, test=test, target_column="y")
        # TODO: show_data
        # show_data(train=self.train,
        #           test=self.test,
        #           target_column=self.target_column,
        #           n_samples=1000)
        print("\nData shown. No result saved.")

    def get_data(self):
        seed = int(self.seed_var.get())
        test_size = self.test_size_var.get()
        # if test_size contains a point, it is a float, otherwise an integer:
        if "." in test_size:
            test_size = float(test_size)
        else:
            test_size = int(test_size)
        shuffle = map_to_True_False(self.shuffle_var.get())
        if self.scenario == "river" or self.scenario == "sklearn":
            data_set_name = self.select_data_frame.get_selected_optionmenu_item()
            dataset, n_samples = get_river_dataset_from_name(
                data_set_name=data_set_name,
                n_total=get_n_total(self.n_total_var.get()),
                river_datasets=self.scenario_dict[self.task_name]["datasets"],
            )
        val = copy.deepcopy(dataset.iloc[0, -1])
        target_type = check_type(val)
        dataset = set_dataset_target_type(dataset=dataset, target="y")
        return dataset, n_samples, target_type, seed, test_size, shuffle

    def print_data(self):
        if self.scenario == "lightning":
            data_set_name = self.select_data_frame.get_selected_optionmenu_item()
            print(f"\nData set name: {data_set_name}")
            if data_set_name.endswith(".csv"):
                data_set = spotPythonCSVDataset(filename=data_set_name, directory="./userData/")
            elif data_set_name.endswith(".pkl"):
                data_set = spotPythonPKLDataset(filename=data_set_name, directory="./userData/")
            else:
                raise ValueError("Invalid data set format. Check userData directory.")
            n_samples = len(data_set)
            print(f"Number of samples: {n_samples}")
            n_cols = data_set.__ncols__()
            print(f"Data set number of columns: {n_cols}")
            # Set batch size for DataLoader
            batch_size = 5
            # Create DataLoader
            from torch.utils.data import DataLoader
            dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
            # Iterate over the data in the DataLoader
            for batch in dataloader:
                inputs, targets = batch
                print(f"Batch Size for Display: {inputs.size(0)}")
                print(f"Inputs Shape: {inputs.shape}")
                print(f"Targets Shape: {targets.shape}")
                print("---------------")
                print(f"Inputs: {inputs}")
                print(f"Targets: {targets}")
                break
        else:
            dataset, n_samples, target_type, seed, test_size, shuffle = self.get_data()
            print("\nDataset in prepare_data():")
            print(f"n_samples: {n_samples}")
            print(f"target_type: {target_type}")
            print(f"seed: {seed}")
            print(f"test_size: {test_size}")
            print(f"shuffle: {shuffle}")
            print(f"{dataset.describe(include='all')}")
            print(f"Header of the dataset:\n {dataset.head()}")

    def prepare_data(self):
        dataset, n_samples, target_type, seed, test_size, shuffle = self.get_data()
        train, test, n_samples = split_df(
            dataset=dataset,
            test_size=test_size,
            target_type=target_type,
            seed=seed,
            shuffle=shuffle,
            stratify=None,
        )
        return train, test, n_samples, target_type

    def prepare_experiment(self):
        log_level = 50
        verbosity = 1
        tolerance_x = np.sqrt(np.spacing(1))
        ocba_delta = 0
        repeats = 1
        fun_repeats = 1
        target_column = "y"
        n_theta = 2
        eval = None

        task_name = self.task_frame.get_selected_optionmenu_item()
        core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()
        # if self has the attribute select_prep_model_frame, get the selected optionmenu item
        if hasattr(self, "select_prep_model_frame"):
            prep_model_name = self.select_prep_model_frame.get_selected_optionmenu_item()
            prepmodel = self.check_user_prep_model(prep_model_name=prep_model_name)
        else:
            prep_model_name = None
            prepmodel = None

        # if self has the attribute select_scaler_frame, get the selected optionmenu item
        if hasattr(self, "select_scaler_frame"):
            scaler_name = self.select_scaler_frame.get_selected_optionmenu_item()
            scaler = self.check_user_prep_model(prep_model_name=scaler_name)
        else:
            scaler_name = None
            scaler = None
        
        data_set_name = self.select_data_frame.get_selected_optionmenu_item()

        seed = int(self.seed_var.get())
        test_size = self.test_size_var.get()
        # if test_size contains a point, it is a float, otherwise an integer:
        if "." in test_size:
            test_size = float(test_size)
        else:
            test_size = int(test_size)
        shuffle = map_to_True_False(self.shuffle_var.get())
        if hasattr(self, "select_metric_sklearn_levels_frame"):
            metric_sklearn_name = self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item()
            metric_sklearn = get_metric_sklearn(metric_sklearn_name)
        else:
            metric_sklearn_name = None
            metric_sklearn = None

        n_cols = None  # number of features in the data_set
        n_total = get_n_total(self.n_total_var.get())
        max_time = float(self.max_time_var.get())
        fun_evals = get_fun_evals(self.fun_evals_var.get())
        init_size = int(self.init_size_var.get())
        noise = map_to_True_False(self.noise_var.get())

        lbd_min, lbd_max = get_lambda_min_max(self.lambda_min_max_var.get())
        kriging_noise = get_kriging_noise(lbd_min, lbd_max)
        max_surrogate_points = int(self.max_sp_var.get())

        TENSORBOARD_CLEAN = map_to_True_False(self.tb_clean_var.get())
        self.tensorboard_start = map_to_True_False(self.tb_start_var.get())
        self.tensorboard_stop = map_to_True_False(self.tb_stop_var.get())
        PREFIX = self.experiment_name_entry.get()

        # ----------------- River specific ----------------- #
        # dictionary name for the database
        # similar for all spotRiver experiments
        if self.scenario == "river":
            data_set = None
            db_dict_name = None  # experimental, do not use
            train, test, n_samples, target_type = self.prepare_data()
            weights_entry = self.weights_var.get()
            weights = get_weights(
                self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item(), self.weights_var.get()
            )
            horizon = int(self.horizon_var.get())
            oml_grace_period = get_oml_grace_period(self.oml_grace_period_var.get())
            self.fun = HyperRiver(log_level=log_level).fun_oml_horizon
        elif self.scenario == "lightning":
            db_dict_name = None  # experimental, do not use
            train = None
            test = None
            n_samples = None
            target_type = None
            weights = 1.0
            weights_entry = None
            horizon = None
            oml_grace_period = None
            self.fun = HyperLight(log_level=log_level).fun
            if data_set_name.endswith(".csv"):
                data_set = spotPythonCSVDataset(filename=data_set_name, directory="./userData/")
            elif data_set_name.endswith(".pkl"):
                data_set = spotPythonPKLDataset(filename=data_set_name, directory="./userData/")
            else:
                raise ValueError("Invalid data set format. Check userData directory.")
            n_samples = len(data_set)
            n_cols = data_set.__ncols__()
            print(f"Data set number of columns: {data_set.__ncols__()}")
            # Set batch size for DataLoader
            batch_size = 5
            # Create DataLoader
            from torch.utils.data import DataLoader
            dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
            # Iterate over the data in the DataLoader
            for batch in dataloader:
                inputs, targets = batch
                print(f"Batch Size: {inputs.size(0)}")
                print(f"Inputs Shape: {inputs.shape}")
                print(f"Targets Shape: {targets.shape}")
                print("---------------")
                print(f"Inputs: {inputs}")
                print(f"Targets: {targets}")
                break
        elif self.scenario == "sklearn":
            eval = "evaluate_hold_out" # "eval_test" #
            data_set = None
            db_dict_name = None  # experimental, do not use
            train, test, n_samples, target_type = self.prepare_data()
            print(f"train: {train}")
            print(f"test: {test}")
            print(f"n_samples: {n_samples}")
            print(f"target_type: {target_type}")
            weights = get_metric_sign(metric_sklearn_name)
            weights_entry = None
            horizon = None
            oml_grace_period = None
            self.fun = HyperSklearn(log_level=log_level).fun_sklearn
        # ----------------- fun_control ----------------- #
        self.fun_control = fun_control_init(
            _L_in=n_cols, # number of input features
            _L_out=1,
            _torchmetric=None,
            PREFIX=PREFIX,
            TENSORBOARD_CLEAN=TENSORBOARD_CLEAN,
            core_model_name=core_model_name,
            data_set_name=data_set_name,
            data_set=data_set,
            db_dict_name=db_dict_name,
            eval=eval,
            fun_evals=fun_evals,
            fun_repeats=fun_repeats,
            horizon=horizon,
            max_surrogate_points=max_surrogate_points,
            max_time=max_time,
            metric_sklearn=metric_sklearn,
            metric_sklearn_name=metric_sklearn_name,
            noise=noise,
            n_samples=n_samples,
            n_total=n_total,
            ocba_delta=ocba_delta,
            oml_grace_period=oml_grace_period,
            prep_model=prepmodel,
            prep_model_name=prep_model_name,
            progress_file=self.progress_file,
            scaler=scaler,
            scaler_name=scaler_name,
            scenario=self.scenario,
            seed=seed,
            shuffle=shuffle,
            task=task_name,
            target_column=target_column,
            target_type=target_type,
            test=test,
            test_size=test_size,
            train=train,
            tolerance_x=tolerance_x,
            verbosity=verbosity,
            weights=weights,
            weights_entry=weights_entry,
            log_level=log_level,
        )
        if self.scenario == "river":
            coremodel, core_model_instance = get_river_core_model_from_name(core_model_name)
        else:
            coremodel, core_model_instance = get_core_model_from_name(core_model_name)
        add_core_model_to_fun_control(
            core_model=core_model_instance,
            fun_control=self.fun_control,
            hyper_dict=self.hyperdict,
            filename=None,
        )
        dict = self.hyperdict().hyper_dict[coremodel]
        num_dict = self.num_hp_frame.get_num_item()
        cat_dict = self.cat_hp_frame.get_cat_item()
        update_fun_control_with_hyper_num_cat_dicts(self.fun_control, num_dict, cat_dict, dict)

        # ----------------- design_control ----------------- #
        self.design_control = design_control_init(
            init_size=init_size,
            repeats=repeats,
        )

        # ----------------- surrogate_control ----------------- #
        self.surrogate_control = surrogate_control_init(
            # If lambda is set to 0, no noise will be used in the surrogate
            # Otherwise use noise in the surrogate:
            noise=kriging_noise,
            n_theta=n_theta,
            min_Lambda=lbd_min,
            max_Lambda=lbd_max,
            log_level=log_level,
        )

        # ----------------- optimizer_control ----------------- #
        self.optimizer_control = optimizer_control_init()

    def save_experiment(self):
        self.prepare_experiment()
        save_spot_python_experiment(
            fun_control=self.fun_control,
            design_control=self.design_control,
            surrogate_control=self.surrogate_control,
            optimizer_control=self.optimizer_control,
            fun=self.fun
        )
        print("\nExperiment saved.")

    def run_experiment(self):
        self.prepare_experiment()
        run_spot_python_experiment(
            fun_control=self.fun_control,
            design_control=self.design_control,
            surrogate_control=self.surrogate_control,
            optimizer_control=self.optimizer_control,
            fun=self.fun,
            tensorboard_start=self.tensorboard_start,
            tensorboard_stop=self.tensorboard_stop,
        )
        print("\nExperiment finished.")


# TODO:
# Check the handling of l1/l2 in LogisticRegression. A note (from the River documentation):
# > For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
# Therefore, we set l1 bounds to 0.0:
# modify_hyper_parameter_bounds(fun_control, "l1", bounds=[0.0, 0.0])
# set_control_hyperparameter_value(fun_control, "l1", [0.0, 0.0])
# modify_hyper_parameter_levels(fun_control, "optimizer", ["SGD"])

if __name__ == "__main__":
    customtkinter.set_appearance_mode("light")
    customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
    app = spotPythonApp()
    app.mainloop()
