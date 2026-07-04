import customtkinter as ctk

# --- Global Appearance Settings ---
ctk.set_appearance_mode("Dark")  # Forces the "black mode" UI
ctk.set_default_color_theme("blue")  # Options: "blue", "dark-blue", "green"

class ModernDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Control Dashboard")
        self.geometry("900x600")
        
        # Configure grid layout (1 row, 2 columns)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ==========================================
        # 1. SIDEBAR (Navigation)
        # ==========================================
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1) # Pushes bottom widgets down

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Control Panel", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.nav_button_1 = ctk.CTkButton(self.sidebar_frame, text="Live Monitor")
        self.nav_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.nav_button_2 = ctk.CTkButton(self.sidebar_frame, text="Experiment Setup", fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
        self.nav_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.nav_button_3 = ctk.CTkButton(self.sidebar_frame, text="Data Export", fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
        self.nav_button_3.grid(row=3, column=0, padx=20, pady=10)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))

        # ==========================================
        # 2. MAIN CONTENT AREA
        # ==========================================
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure((0, 1), weight=1)

        # Header
        self.header_label = ctk.CTkLabel(self.main_frame, text="System Overview", font=ctk.CTkFont(size=24, weight="bold"))
        self.header_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(0, 20), sticky="w")

        # --- Top Cards (e.g., Live Stats) ---
        self.card_1 = ctk.CTkFrame(self.main_frame)
        self.card_1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.card_1, text="Motor Position", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        ctk.CTkLabel(self.card_1, text="14.50 mm", font=ctk.CTkFont(size=28)).pack(pady=(0, 15))

        self.card_2 = ctk.CTkFrame(self.main_frame)
        self.card_2.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.card_2, text="Total Counts (ROI)", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        ctk.CTkLabel(self.card_2, text="84,201", font=ctk.CTkFont(size=28, text_color="#2FA572")).pack(pady=(0, 15))

        # --- Control Section ---
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=20, sticky="nsew")
        self.control_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.control_frame, text="Scan Progress", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        self.progressbar = ctk.CTkProgressBar(self.control_frame)
        self.progressbar.grid(row=0, column=1, padx=20, pady=(20, 10), sticky="ew")
        self.progressbar.set(0.65) # Simulating 65% complete

        self.start_btn = ctk.CTkButton(self.control_frame, text="Start Scan", fg_color="#2FA572", hover_color="#208259")
        self.start_btn.grid(row=1, column=0, padx=20, pady=(10, 20))
        
        self.stop_btn = ctk.CTkButton(self.control_frame, text="Emergency Stop", fg_color="#C24641", hover_color="#91322E")
        self.stop_btn.grid(row=1, column=1, padx=20, pady=(10, 20), sticky="w")

        # --- Log / Output Box ---
        self.log_box = ctk.CTkTextbox(self.main_frame, height=150)
        self.log_box.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.log_box.insert("0.0", "System initialized...\nHardware connected.\nReady for measurement sequence...\n")
        self.log_box.configure(state="disabled") # Make it read-only

    # --- Functions ---
    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

if __name__ == "__main__":
    app = ModernDashboard()
    app.mainloop()