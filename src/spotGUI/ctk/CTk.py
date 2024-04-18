import customtkinter
import os
import copy
from spotGUI.tuner.spotRun import progress_plot, contour_plot, importance_plot, get_core_model_from_name, get_prep_model
from PIL import Image
import time
from spotPython.utils.eda import gen_design_table
import tkinter as tk
import sys


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
