import customtkinter
import os


class CTkApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        # name of the progress file
        self.progress_file = "progress.txt"
        # if the progress file exists, delete it
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        print(f"Appearance Mode changed to: {new_appearance_mode}")
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
