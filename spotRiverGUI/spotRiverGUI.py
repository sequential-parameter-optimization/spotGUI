import tkinter as tk
import customtkinter
import pprint
import os
import numpy as np
import copy
from spotPython.utils.init import fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init
from spotGUI.ctk.CTk import CTkApp, SelectOptionMenuFrame

from spotRiver.hyperdict.river_hyper_dict import RiverHyperDict
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
    get_metric_sklearn,
    update_fun_control_with_hyper_num_cat_dicts,
)
from spotRiver.fun.hyperriver import HyperRiver


class RiverApp(CTkApp):
    def __init__(self):
        super().__init__()

        self.scenario = "river"
        self.hyperdict = RiverHyperDict
        self.title("spotRiver GUI")
        self.logo_text = "    SPOTRiver"

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
        self.textbox.insert(tk.END, "Welcome to SPOTRiver\n")
        #
        # Start the thread that will update the text area
        # update_thread = threading.Thread(target=self.update_text, daemon=True)
        # update_thread.start()
        # e = ThreadPoolExecutor(max_workers=1)
        # e.submit(self.update_text)
        # e.shutdown(wait=False)

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

    def prepare_data(self):
        seed = int(self.seed_var.get())
        test_size = self.test_size_var.get()
        # if test_size contains a point, it is a float, otherwise an integer:
        if "." in test_size:
            test_size = float(test_size)
        else:
            test_size = int(test_size)
        shuffle = map_to_True_False(self.shuffle_var.get())
        data_set_name = self.select_data_frame.get_selected_optionmenu_item()
        dataset, n_samples = get_river_dataset_from_name(
            data_set_name=data_set_name,
            n_total=get_n_total(self.n_total_var.get()),
            river_datasets=self.scenario_dict[self.task_name]["datasets"],
        )
        val = copy.deepcopy(dataset.iloc[0, -1])
        target_type = check_type(val)
        dataset = set_dataset_target_type(dataset=dataset, target="y")
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

        task_name = self.task_frame.get_selected_optionmenu_item()
        core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()
        prep_model_name = self.select_prep_model_frame.get_selected_optionmenu_item()
        prepmodel = self.check_user_prep_model(prep_model_name=prep_model_name)

        data_set_name = self.select_data_frame.get_selected_optionmenu_item()

        seed = int(self.seed_var.get())
        test_size = self.test_size_var.get()
        # if test_size contains a point, it is a float, otherwise an integer:
        if "." in test_size:
            test_size = float(test_size)
        else:
            test_size = int(test_size)
        shuffle = map_to_True_False(self.shuffle_var.get())
        metric_sklearn_name = self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item()
        metric_sklearn = get_metric_sklearn(metric_sklearn_name)

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
        db_dict_name = "spotRiver_db.json"
        train, test, n_samples, target_type = self.prepare_data()
        weights_entry = self.weights_var.get()
        weights = get_weights(
            self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item(), self.weights_var.get()
        )
        horizon = int(self.horizon_var.get())
        oml_grace_period = get_oml_grace_period(self.oml_grace_period_var.get())
        self.fun = HyperRiver(log_level=log_level).fun_oml_horizon

        # ----------------- fun_control ----------------- #
        self.fun_control = fun_control_init(
            PREFIX=PREFIX,
            TENSORBOARD_CLEAN=TENSORBOARD_CLEAN,
            core_model_name=core_model_name,
            data_set_name=data_set_name,
            db_dict_name=db_dict_name,
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
    app = RiverApp()
    app.mainloop()
