import customtkinter


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
        # if self.optionmenu_var exists, return the value of the optionmenu_var
        if self.optionmenu_var:
            return self.optionmenu_var.get()
        else:
            return None

    def set_selected_optionmenu_item(self, item):
        self.optionmenu_var.set(item)
