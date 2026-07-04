import os
import sys
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import customtkinter as ctk
from pylablib.devices import Thorlabs

# ==========================================
# 1. GLOBAL HARDWARE SETUP & CONSTANTS
# ==========================================
PIXETDIR = r"C:\Program Files\PIXet Pro"
SAVE_DIR = r"C:\Users\Lab\Desktop\My_Scans"
SERIAL_NUM = "83000001" # Replace with your TDC001 serial number
EXPOSURE_TIME = 1.0     # Default exposure time in seconds

# Ensure the save directory exists to prevent popups
os.makedirs(SAVE_DIR, exist_ok=True)

# Add PIXet API to path
sys.path.append(PIXETDIR)
try:
    import pypixet
except ImportError:
    print("WARNING: Could not import pypixet. Ensure you are running 64-bit Python.")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_and_calibrate_t3p(filepath):
    """
    Parses the .t3p file and applies hardware calibrations.
    """
    # ---------------------------------------------------------
    # INSERT YOUR CUSTOM .T3P PARSING LOGIC HERE
    # Reminder: HW Trigger packets do not always come in 16-byte blocks.
    # FToA counts ToA counter overflows, and packets end with a specific 
    # tab-separated sequence.
    # ---------------------------------------------------------
    
    # Placeholder: replace this with your actual parsed 2D NumPy array
    raw_matrix = np.random.randint(0, 50, size=(256, 256)) 
    
    # CRITICAL CALIBRATION FOR CdTe CHIPS:
    # Rotate matrix because the first pixel is bottom-left relative to readout wires.
    calibrated_matrix = np.rot90(raw_matrix, k=1) 
    
    return calibrated_matrix

def visually_select_roi(frame_matrix):
    """Opens an interactive window to drag and select the ROI."""
    fig, ax = plt.subplots()
    ax.imshow(frame_matrix, cmap='viridis', origin='lower')
    plt.title("Click and drag to select ROI. Close window when done.")
    
    roi_coords = [0, frame_matrix.shape[0], 0, frame_matrix.shape[1]]
    
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        roi_coords[0] = max(0, min(y1, y2))
        roi_coords[1] = min(frame_matrix.shape[0], max(y1, y2))
        roi_coords[2] = max(0, min(x1, x2))
        roi_coords[3] = min(frame_matrix.shape[1], max(x1, x2))

    rs = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
    plt.show(block=True) # Pauses execution until window is closed
    return roi_coords

# ==========================================
# 3. DASHBOARD UI & THREADING LOGIC
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ModernDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MTS50-Z8 & AdvaPIX Control Dashboard")
        self.geometry("950x650")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # State Variables
        self.scan_thread = None
        self.stop_requested = False
        self.camera = None
        self.motor = None
        self.pixet_core = None

        self.setup_ui()
        self.initialize_hardware()

    def setup_ui(self):
        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(self.sidebar_frame, text="Control Panel", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))

        # --- Main Content Area ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(self.main_frame, text="System Overview", font=ctk.CTkFont(size=24, weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(0, 20), sticky="w")

        # Live Stats Cards
        self.card_1 = ctk.CTkFrame(self.main_frame)
        self.card_1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.card_1, text="Current Position (mm)", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        self.lbl_position = ctk.CTkLabel(self.card_1, text="--", font=ctk.CTkFont(size=28))
        self.lbl_position.pack(pady=(0, 15))

        self.card_2 = ctk.CTkFrame(self.main_frame)
        self.card_2.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.card_2, text="Latest ROI Counts", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        self.lbl_counts = ctk.CTkLabel(self.card_2, text="--", font=ctk.CTkFont(size=28, text_color="#2FA572"))
        self.lbl_counts.pack(pady=(0, 15))

        # Experiment Setup Controls
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=20, sticky="nsew")
        
        ctk.CTkLabel(self.control_frame, text="Motor Scan Parameters", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=6, padx=20, pady=(10, 10), sticky="w")

        ctk.CTkLabel(self.control_frame, text="Start (mm):").grid(row=1, column=0, padx=(20, 5), pady=10, sticky="e")
        self.entry_start = ctk.CTkEntry(self.control_frame, width=80)
        self.entry_start.insert(0, "0.0")
        self.entry_start.grid(row=1, column=1, padx=5, pady=10, sticky="w")

        ctk.CTkLabel(self.control_frame, text="End (mm):").grid(row=1, column=2, padx=10, pady=10, sticky="e")
        self.entry_end = ctk.CTkEntry(self.control_frame, width=80)
        self.entry_end.insert(0, "10.0")
        self.entry_end.grid(row=1, column=3, padx=5, pady=10, sticky="w")

        ctk.CTkLabel(self.control_frame, text="Step (mm):").grid(row=1, column=4, padx=10, pady=10, sticky="e")
        self.entry_step = ctk.CTkEntry(self.control_frame, width=80)
        self.entry_step.insert(0, "0.5")
        self.entry_step.grid(row=1, column=5, padx=5, pady=10, sticky="w")

        self.progressbar = ctk.CTkProgressBar(self.control_frame)
        self.progressbar.grid(row=2, column=0, columnspan=6, padx=20, pady=(10, 5), sticky="ew")
        self.progressbar.set(0.0)

        self.start_btn = ctk.CTkButton(self.control_frame, text="Start Scan", fg_color="#2FA572", hover_color="#208259", command=self.start_scan_sequence)
        self.start_btn.grid(row=3, column=0, columnspan=3, padx=20, pady=20, sticky="ew")
        
        self.stop_btn = ctk.CTkButton(self.control_frame, text="Emergency Stop", fg_color="#C24641", hover_color="#91322E", command=self.emergency_stop)
        self.stop_btn.grid(row=3, column=3, columnspan=3, padx=20, pady=20, sticky="ew")

        # Output Log Box
        self.log_box = ctk.CTkTextbox(self.main_frame, height=150)
        self.log_box.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    def log(self, message):
        """Thread-safe UI logging"""
        self.log_box.insert(ctk.END, message + "\n")
        self.log_box.see(ctk.END)

    def initialize_hardware(self):
        self.log("Initializing Hardware...")
        try:
            # Init PIXet
            pypixet.start()
            self.pixet_core = pypixet.pixet
            devices = self.pixet_core.devicesByType(self.pixet_core.PX_DEVTYPE_TPX3)
            if not devices:
                self.log("ERROR: No AdvaPIX camera found.")
            else:
                self.camera = devices[0]
                self.log(f"Camera Connected: {self.camera.name()}")

            # Init Thorlabs
            self.motor = Thorlabs.KinesisMotor(SERIAL_NUM)
            self.log(f"Motor Connected (SN: {SERIAL_NUM}). Homing...")
            # self.motor.home() # Uncomment this when running on live hardware
            self.log("Hardware ready.")
        except Exception as e:
            self.log(f"Hardware initialization skipped/failed: {e}")

    def emergency_stop(self):
        self.log(">>> STOP REQUESTED. Finishing current step... <<<")
        self.stop_requested = True

    def start_scan_sequence(self):
        """Phase 1: Grab parameters and take the scout shot (Runs in Main UI Thread)"""
        if self.camera is None or self.motor is None:
            self.log("Hardware not connected. Cannot run scan.")
            return

        try:
            start_pos = float(self.entry_start.get())
            end_pos = float(self.entry_end.get())
            step_size = float(self.entry_step.get())
        except ValueError:
            self.log("Error: Invalid numeric inputs.")
            return

        self.stop_requested = False
        self.start_btn.configure(state="disabled") # Prevent double-clicks

        # Take Scout Shot
        self.log("Taking scout frame for ROI selection...")
        test_file = os.path.join(SAVE_DIR, "roi_test_frame.t3p")
        self.camera.doSimpleAcquisition(1, EXPOSURE_TIME, self.pixet_core.PX_FTYPE_AUTODETECT, test_file)
        
        test_matrix = load_and_calibrate_t3p(test_file)
        
        # Pop up visual selector
        self.roi_coords = visually_select_roi(test_matrix)
        self.log(f"ROI Locked: Y[{self.roi_coords[0]}:{self.roi_coords[1]}], X[{self.roi_coords[2]}:{self.roi_coords[3]}]")

        # Start Phase 2: The Motor Loop (in a background thread)
        self.scan_thread = threading.Thread(target=self.execute_motor_scan, args=(start_pos, end_pos, step_size))
        self.scan_thread.start()

    def execute_motor_scan(self, start_pos, end_pos, step_size):
        """Phase 2: Automated Loop (Runs in Background Thread)"""
        num_steps = int(abs(end_pos - start_pos) / step_size) + 1
        measured_positions = []
        photon_counts = []

        roi_y1, roi_y2, roi_x1, roi_x2 = self.roi_coords

        self.log("Starting automated scan loop...")

        for step in range(num_steps):
            if self.stop_requested:
                self.log("Scan aborted by user.")
                break

            target_pos = start_pos + (step * step_size)
            
            # 1. Move Hardware
            self.motor.move_to(target_pos)
            self.motor.wait_move()
            
            # 2. Acquire Image (Bypasses popup using os.path.join)
            filename = os.path.join(SAVE_DIR, f"scan_pos_{target_pos:.2f}.t3p")
            self.camera.doSimpleAcquisition(1, EXPOSURE_TIME, self.pixet_core.PX_FTYPE_AUTODETECT, filename)
            
            # 3. Process Data
            frame_matrix = load_and_calibrate_t3p(filename)
            roi_data = frame_matrix[roi_y1:roi_y2, roi_x1:roi_x2]
            total_counts = np.sum(roi_data)
            
            # 4. Save Data to array
            measured_positions.append(target_pos)
            photon_counts.append(total_counts)

            # 5. Update UI Safely
            progress = (step + 1) / num_steps
            self.progressbar.set(progress)
            self.lbl_position.configure(text=f"{target_pos:.2f}")
            self.lbl_counts.configure(text=f"{total_counts}")
            self.log(f"Step {step+1}/{num_steps} | Pos: {target_pos:.2f}mm | Counts: {total_counts}")

        self.log("Scan Complete.")
        
        # Reset UI
        self.start_btn.configure(state="normal")
        self.progressbar.set(1.0)

        # Plot Final Results (Must be launched in main thread if using specific matplotlib backends)
        self.after(500, lambda: self.plot_results(measured_positions, photon_counts))

    def plot_results(self, positions, counts):
        if not positions:
            return
        plt.figure(figsize=(8, 5))
        plt.plot(positions, counts, marker='o', linestyle='-', color='b')
        plt.title('Total Photon Counts in ROI vs. Motor Position')
        plt.xlabel('Motor Position (mm)')
        plt.ylabel('Photon Counts')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def on_closing(self):
        """Cleanup hardware before closing app"""
        try:
            if self.motor: self.motor.close()
            pypixet.exit()
        except:
            pass
        self.destroy()

if __name__ == "__main__":
    app = ModernDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()