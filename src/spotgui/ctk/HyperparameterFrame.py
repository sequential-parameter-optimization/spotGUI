import customtkinter


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

    def get_cat_item(self) -> dict:
        """
        Get the values self.hp_list, self.default_list, self.levels_list,
        and put lower and upper in a dictionary with the corresponding
        hyperparameter (hp) as key.

        Note:
            Method is designed for categorical parameters.

        Returns:
            num_hp_dict (dict):
                dictionary with hyperparameter as key and values
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
