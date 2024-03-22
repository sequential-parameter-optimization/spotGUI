import customtkinter
import os
from PIL import Image


class NumHyperparameterFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.command = command
        self.radiobutton_variable = customtkinter.StringVar()
        self.label_list = []
        self.default_list = []
        self.lower_list = []
        self.upper_list = []
        self.transformation_list = []
        self.level_list = []

    def add_header(self):
        header_hp = customtkinter.CTkLabel(self, text="Hyperparameter", fg_color="gray30", corner_radius=6)
        header_hp.grid(row=len(self.label_list), column=0, padx=10, pady=(10, 0), sticky="ew")

    def add_num_item(self, item, image=None):
        label = customtkinter.CTkLabel(self, text=item, compound="left", padx=5, anchor="w")
        default = customtkinter.CTkLabel(self, text="Default", compound="left", padx=5, anchor="w")
        lower = customtkinter.CTkLabel(self, text="Lower", compound="left", padx=5, anchor="w")
        upper = customtkinter.CTkLabel(self, text="Upper", compound="left", padx=5, anchor="w")
        transformation = customtkinter.CTkLabel(self, text="Transformation", compound="left", padx=5, anchor="w")

        label.grid(row=len(self.label_list), column=0, pady=(0, 10), sticky="w")
        default.grid(row=len(self.default_list), column=1, pady=(0, 10), sticky="w")
        lower.grid(row=len(self.lower_list), column=2, pady=(0, 10), sticky="w")
        upper.grid(row=len(self.upper_list), column=3, pady=(0, 10), sticky="w")
        transformation.grid(row=len(self.transformation_list), column=4, pady=(0, 10), padx=5)
        self.label_list.append(label)
        self.default_list.append(default)
        self.lower_list.append(lower)
        self.upper_list.append(upper)
        self.transformation_list.append(transformation)

    def remove_num_item(self, item):
        for label, default, lower, upper, transformation in zip(self.label_list,
                                                                self.default_list,
                                                                self.lower_list,
                                                                self.upper_list,
                                                                self.transformation_list):
            if item == label.cget("text"):
                label.destroy()
                default.destroy()
                lower.destroy()
                upper.destroy()
                transformation.destroy()
                self.label_list.remove(label)
                self.default_list.remove(default)
                self.lower_list.remove(lower)
                self.upper_list.remove(upper)
                self.transformation_list.remove(transformation)
                return


class CatHyperparameterFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.command = command
        self.radiobutton_variable = customtkinter.StringVar()
        self.label_list = []
        self.default_list = []
        self.lower_list = []
        self.upper_list = []
        self.transformation_list = []
        self.level_list = []

    def add_header(self):
        header_hp = customtkinter.CTkLabel(self, text="Hyperparameter", fg_color="gray30", corner_radius=6)
        header_hp.grid(row=len(self.label_list), column=0, padx=10, pady=(10, 0), sticky="ew")

    def add_cat_item(self, item, image=None):
        label = customtkinter.CTkLabel(self, text=item, compound="left", padx=5, anchor="w")
        default = customtkinter.CTkLabel(self, text="Default", compound="left", padx=5, anchor="w")
        level = customtkinter.CTkLabel(self, text="Levels", compound="left", padx=5, anchor="w")
        transformation = customtkinter.CTkLabel(self, text="Transformation", compound="left", padx=5, anchor="w")

        label.grid(row=len(self.label_list), column=0, pady=(0, 10), sticky="w")
        default.grid(row=len(self.default_list), column=1, pady=(0, 10), sticky="w")
        level.grid(row=len(self.level_list), column=2, pady=(0, 10), padx=5)
        transformation.grid(row=len(self.transformation_list), column=4, pady=(0, 10), padx=5)
        self.label_list.append(label)
        self.default_list.append(default)
        self.level_list.append(level)
        self.transformation_list.append(transformation)

    def remove_cat_item(self, item):
        for label, default, level, transformation in zip(self.label_list,
                                                        self.default_list,
                                                        self.level_list,
                                                        self.transformation_list):
            if item == label.cget("text"):
                label.destroy()
                default.destroy()
                level.destroy()
                transformation.destroy()
                self.label_list.remove(label)
                self.default_list.remove(default)
                self.lower_list.remove(level)
                self.transformation_list.remove(transformation)
                return


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("spotRiver GUI")
        self.grid_rowconfigure(0, weight=1)
        self.columnconfigure(2, weight=1)

        # create scrollable label and button frame
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.num_hp_frame = NumHyperparameterFrame(master=self,
                                                                 width=500,
                                                                 command=self.label_button_frame_event,
                                                                 label_text="NumHyperparameterFrame",
                                                                 corner_radius=0)
        self.cat_hp_frame = CatHyperparameterFrame(master=self,
                                                                 width=500,
                                                                 command=self.label_button_frame_event,
                                                                 label_text="CatHyperparameterFrame",
                                                                 corner_radius=0)
        self.num_hp_frame.grid(row=0, column=2, padx=0, pady=0, sticky="nsew")
        self.cat_hp_frame.grid(row=1, column=2, padx=0, pady=0, sticky="nsew")
        n_num_items = 3
        n_cat_items = 2
        for i in range(n_num_items):
            self.num_hp_frame.add_num_item(f"Item {i}")
        for i in range(n_cat_items):
            self.cat_hp_frame.add_cat_item(f"Item {i}")

    def label_button_frame_event(self, item):
        print(f"label button frame clicked: {item}")


if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")
    app = App()
    app.mainloop()