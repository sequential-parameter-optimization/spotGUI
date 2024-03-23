import tkinter
import tkinter.messagebox
import customtkinter
import pprint

import os
from PIL import Image

from spotGUI.tuner.spotRun import get_task_dict, get_core_model_from_name
from spotRiver.data.river_hyper_dict import RiverHyperDict

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


class CheckboxFrame(customtkinter.CTkFrame):
    def __init__(self, master, text, value, command=None, **kwargs):
        super().__init__(master)
        self.checkbox_var = customtkinter.StringVar(value=value)
        checkbox = customtkinter.CTkCheckBox(self,
                                             text=text,
                                             command=command,
                                             variable=self.checkbox_var,
                                             onvalue="True",
                                             offvalue="False")
        checkbox.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

    def get_checkbox_var(self):
        return self.checkbox_var.get()


class NumHyperparameterFrame(customtkinter.CTkScrollableFrame):
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


class CatHyperparameterFrame(customtkinter.CTkScrollableFrame):
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
        self.levels_col.insert("0.0", str(levels))

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
        # self.geometry(f"{1400}x{780}")
        self.resizable(True, True)
        # configure grid layout (4x4)
        # self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.rhd = RiverHyperDict()

        self.task_name = "regression_tab"
        self.task_dict = get_task_dict()
        pprint.pprint(self.task_dict)
        self.core_model_name = self.task_dict[self.task_name]["core_model_names"][0]

        # set default values
        # these values can be changed by the GUI and will be passed to spot
        # self.appearance_mode_value = "Dark"
        # self.data_set = "data 0"
        # self.shuffle = None

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkScrollableFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame,
                                                 text="SpotRiver GUI",
                                                 font=customtkinter.CTkFont(size=20,
                                                                            weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # create data main frame with widgets
        self.data_main_frame = customtkinter.CTkScrollableFrame(self, width=240, corner_radius=0)
        self.data_main_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.data_main_frame.grid_rowconfigure(4, weight=1)

        # create hyperparameter main frame with widgets
        self.hp_main_frame = customtkinter.CTkFrame(self, width=720, corner_radius=0)
        self.hp_main_frame.grid(row=0, column=2, rowspan=5, sticky="nsew")
        self.hp_main_frame.grid_rowconfigure(4, weight=1)

        # Execution main frame
        self.exec_main_frame = customtkinter.CTkFrame(self, width=256, corner_radius=0)
        self.exec_main_frame.grid(row=0, column=7, rowspan=2, sticky="nsew")
        self.exec_main_frame.grid_rowconfigure(4, weight=1)
        # create run button
        self.run_button = customtkinter.CTkButton(master=self.exec_main_frame,
                                             text="Run",
                                             command=self.run_button_event)
        self.run_button.grid(row=0, column=1, pady=(0, 10), padx=5)

        self.task_frame = SelectOptionMenuFrame(master=self.sidebar_frame,
                                                width=500,
                                                command=self.change_task_event,
                                                item_list=["Binary Classification",
                                                           "Regression"],
                                                item_default="Regression",
                                                title="Select Task")
        self.task_frame.grid(row=3, column=0, padx=15, pady=15, sticky="ns")
        self.task_frame.configure(width=500)

        # create appearance mode frame
        self.appearance_frame = SelectOptionMenuFrame(master=self.sidebar_frame,
                                                width=500,
                                                command=self.change_appearance_mode_event,
                                                item_list=["Light", "Dark", "System"],
                                                item_default="System",
                                                title="Appearance Mode")
        self.appearance_frame.grid(row=7, column=0, padx=15, pady=15, sticky="ns")
        self.appearance_frame.configure(width=500)

        # create select data set frame
        self.hp__main_frame_title = customtkinter.CTkLabel(self.data_main_frame,
                                            text="Data",
                                            font=customtkinter.CTkFont(size=20,
                                                            weight="bold"),
                                            corner_radius=6)
        self.hp__main_frame_title.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="ew")
        self.select_data_frame = SelectOptionMenuFrame(master=self.data_main_frame,
                                                           width=500,
                                                           command=self.select_data_frame_event,
                                                           item_list=self.task_dict[self.task_name]["datasets"],
                                                           item_default=None,
                                                           title="Select Data")
        self.select_data_frame.grid(row=1, column=1, padx=15, pady=15, sticky="ns")
        self.select_data_frame.configure(width=500)

        # shuffle data in data main frame
        self.shuffle_checkbox_frame = CheckboxFrame(self.data_main_frame,
                                                    text="shuffle",
                                                    value="True",)
        self.shuffle_checkbox_frame.grid(row=2, column=1, padx=(0, 10), pady=(10, 0), sticky="nsew")
        self.shuffle = self.shuffle_checkbox_frame.get_checkbox_var()

        # create core model frame
        self.create_core_model_frame()

        # create select prep model frame
        self.select_prep_model_frame = SelectOptionMenuFrame(master=self.sidebar_frame,
                                                           width=500,
                                                           command=self.select_prep_model_frame_event,
                                                           item_list=self.task_dict[self.task_name]["prep_models"],
                                                           item_default=None,
                                                           title="Select Prep Model")
        self.select_prep_model_frame.grid(row=5, column=0, padx=15, pady=15, sticky="ns")
        self.select_prep_model_frame.configure(width=200)

        # create scrollable label and button frame
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.hp__main_frame_title = customtkinter.CTkLabel(self.hp_main_frame,
                                                           text="Hyperparameter",
                                                           font=customtkinter.CTkFont(size=20,
                                                                            weight="bold"),
                                                           corner_radius=6)
        self.hp__main_frame_title.grid(row=0, column=3, padx=10, pady=(10, 0), sticky="ew")
        self.create_num_hp_frame()
        self.create_cat_hp_frame()

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
                                                                 height=480,
                                                                 command=self.label_button_frame_event,
                                                                 label_text="Numerical Hyperparameters",
                                                                 corner_radius=0)
        self.num_hp_frame.grid(row=1, column=3, padx=0, pady=0, sticky="nsew")
        self.num_hp_frame.add_header()
        print(f"self.core_model_name: {self.core_model_name}")
        coremodel, core_model_instance = get_core_model_from_name(self.core_model_name)
        dict = self.rhd.hyper_dict[coremodel]
        pprint.pprint(dict)
        for i, (key, value) in enumerate(dict.items()):
            if (dict[key]["type"] == "int"
                or dict[key]["type"] == "float"
                or dict[key]["core_model_parameter_type"] == "bool"):
                self.num_hp_frame.add_num_item(hp=key,
                                               default=value["default"],
                                               lower=value["lower"],
                                               upper=value["upper"],
                                               transform=value["transform"])

    def create_cat_hp_frame(self):
        self.cat_hp_frame = CatHyperparameterFrame(master=self.hp_main_frame,
                                                                 width=640,
                                                                 command=self.label_button_frame_event,
                                                                 label_text="Categorical Hyperparameters",
                                                                 corner_radius=0)
        self.cat_hp_frame.grid(row=2, column=3, padx=0, pady=0, sticky="nsew")
        self.cat_hp_frame.add_header()
        print(f"self.core_model_name: {self.core_model_name}")
        coremodel, core_model_instance = get_core_model_from_name(self.core_model_name)
        dict = self.rhd.hyper_dict[coremodel]
        pprint.pprint(dict)
        for i, (key, value) in enumerate(dict.items()):
            if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
                self.cat_hp_frame.add_cat_item(hp=key,
                                               default=value["default"],
                                               levels=value["levels"],
                                               transform=value["transform"])

    def create_core_model_frame(self):
        # create new core model frame
        self.select_core_model_frame = SelectOptionMenuFrame(master=self.sidebar_frame,
                                                             width=500,
                                                             command=self.select_core_model_frame_event,
                                                             item_list=self.task_dict[self.task_name]["core_model_names"],
                                                             item_default=None,
                                                             title="Select Core Model")
        self.select_core_model_frame.grid(row=4, column=0, padx=15, pady=15, sticky="ns")
        self.select_core_model_frame.configure(width=500)
        self.core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()

    def select_core_model_frame_event(self, new_core_model: str):
        print(f"Core Model modified: {new_core_model}")
        self.core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()
        print(f"New self.core_model_name: {self.core_model_name}")
        # TODO: Modify the hyperparameters based on the core model
        # destroy old num_hp_frame
        self.num_hp_frame.destroy()
        # create new num_hp_frame
        self.create_num_hp_frame()
        self.cat_hp_frame.destroy()
        self.create_cat_hp_frame()

    def select_prep_model_frame_event(self, new_prep_model: str):
        print(f"Prep Model modified: {new_prep_model}")
        print(f"Prep Model modified: {self.select_prep_model_frame.get_selected_optionmenu_item()}")

    def change_task_event(self, new_task: str):
        print(f"Task changed to: {new_task}")
        if new_task == "Binary Classification":
            self.task_name = "classification_tab"
        elif new_task == "Regression":
            self.task_name = "regression_tab"
        else:
            print("Error: Task not found")
        # destroy old core model frame
        self.select_core_model_frame.destroy()
        # create new core model frame
        self.create_core_model_frame()
        print(f"New self.core_model_name: {self.core_model_name}")
        # destroy old num_hp_frame
        self.num_hp_frame.destroy()
        # create new num_hp_frame
        self.create_num_hp_frame()
        self.cat_hp_frame.destroy()
        self.create_cat_hp_frame()

        # destroy old data frame
        self.select_data_frame.destroy()
        # create new data frame
        self.select_data_frame = SelectOptionMenuFrame(master=self.data_main_frame,
                                                           width=500,
                                                           command=self.select_data_frame_event,
                                                           item_list=self.task_dict[self.task_name]["datasets"],
                                                           item_default=None,
                                                           title="Select Data")
        self.select_data_frame.grid(row=1, column=1, padx=15, pady=15, sticky="ns")
        self.select_data_frame.configure(width=500)

    def run_button_event(self):
        print("Run button clicked")
        print("Data:", self.select_data_frame.get_selected_optionmenu_item())
        print("Shuffle:", self.shuffle_checkbox_frame.get_checkbox_var())
        print("Core Model:", self.select_core_model_frame.get_selected_optionmenu_item())
        print("Prep Model:", self.select_prep_model_frame.get_selected_optionmenu_item())
        print("Numerical Hyperparameters:", self.num_hp_frame.get_num_item())
        print("Categorical Hyperparameters:", self.cat_hp_frame.get_cat_item())


if __name__ == "__main__":
    customtkinter.set_appearance_mode("light")
    app = App()
    app.mainloop()