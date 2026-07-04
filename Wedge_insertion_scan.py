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
SAVE_DIR = r"C:\Eldar\Eldar_IFM\esrf_ifm_data\Automatic_wedge_insertion"
SERIAL_NUM = "83860502"

# Ensure the scan directory exists immediately to suppress native Windows dialogs
os.makedirs(SAVE_DIR, exist_ok=True)

# Add PIXet API folder to system environment path
sys.path.append(PIXETDIR)
try:
    import pypixet
except ImportError:
    print("CRITICAL WARNING: Could not import pypixet.")
    print("Ensure your Python architecture (32/64-bit) matches your PIXet installation.")


# ==========================================
# 2. DATA PROCESSING & INTERACTIVE ROI
# ==========================================
def load_and_calibrate_frame(filepath):
    """
    Loads a PIXet ASCII text frame matrix and applies CdTe orientation correction.
    """
    # Simply load the 256x256 grid of text numbers directly into a 2D NumPy array
    raw_matrix = np.loadtxt(filepath)

    # CRITICAL CADMIUM TELLURIDE (CdTe) CALIBRATION:
    # Rotate the matrix 90 degrees counter-clockwise because the first pixel indexing
    # begins at the physical bottom-left corner relative to the readout wires.
    calibrated_matrix = np.rot90(raw_matrix, k=1)

    return calibrated_matrix


def visually_select_roi(frame_matrix):
    """Opens an interactive matplotlib selector to slice out the region of interest."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frame_matrix, cmap='viridis', origin='lower')
    plt.title("Click & Drag to define ROI rectangle.\nClose this window to lock coordinates.")

    # Default fall-back bounds equal to the whole chip matrix
    roi_coords = [0, frame_matrix.shape[0], 0, frame_matrix.shape[1]]

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Enforce strict matrix slicing constraints [ymin:ymax, xmin:xmax]
        roi_coords[0] = max(0, min(y1, y2))
        roi_coords[1] = min(frame_matrix.shape[0], max(y1, y2))
        roi_coords[2] = max(0, min(x1, x2))
        roi_coords[3] = min(frame_matrix.shape[1], max(x1, x2))

    rs = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
    plt.show(block=True)  # Block execution until the user closes the window
    return roi_coords


# ==========================================
# 3. MASTER CUSTOMTKINTER DASHBOARD
# ==========================================
ctk.set_appearance_mode("Dark")  # Force explicit Black Mode theme
ctk.set_default_color_theme("blue")


class MasterControlDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ESRF Automated Hardware Scan - MTS50-Z8 & AdvaPIX")
        self.geometry("980x680")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Threading and Engine States
        self.scan_thread = None
        self.stop_requested = False
        self.camera = None
        self.motor = None
        self.pixet_core = None
        self.roi_coords = None

        self.setup_ui_layout()
        self.connect_hardware_drivers()

    def setup_ui_layout(self):
        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.sidebar_frame, text="ESRF System Panel",
                     font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=20, pady=25)

        # --- Main Control Station Area ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(self.main_frame, text="Real-Time Data Acquisition (Frames Mode)",
                     font=ctk.CTkFont(size=24, weight="bold")).grid(row=0, column=0, columnspan=2, padx=10,
                                                                    pady=(0, 20), sticky="w")

        # Telemetry Stat Displays
        self.card_pos = ctk.CTkFrame(self.main_frame)
        self.card_pos.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.card_pos, text="Motor Extension (mm)", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        self.lbl_position = ctk.CTkLabel(self.card_pos, text="--", font=ctk.CTkFont(size=32))
        self.lbl_position.pack(pady=(0, 15))

        self.card_cnt = ctk.CTkFrame(self.main_frame)
        self.card_cnt.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.card_cnt, text="ROI Integral Counts", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        self.lbl_counts = ctk.CTkLabel(self.card_cnt, text="--", font=ctk.CTkFont(size=32, text_color="#2FA572"))
        self.lbl_counts.pack(pady=(0, 15))

        # Config Parameter Entry Grids
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=20, sticky="nsew")

        ctk.CTkLabel(self.control_frame, text="Experiment Profile Slicing",
                     font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=8, padx=20, pady=(15, 10),
                                                           sticky="w")

        # User Input Boxes
        ctk.CTkLabel(self.control_frame, text="Start (mm):").grid(row=1, column=0, padx=(15, 2), pady=10, sticky="e")
        self.entry_start = ctk.CTkEntry(self.control_frame, width=70)
        self.entry_start.insert(0, "0.0")
        self.entry_start.grid(row=1, column=1, padx=2, pady=10, sticky="w")

        ctk.CTkLabel(self.control_frame, text="End (mm):").grid(row=1, column=2, padx=10, pady=10, sticky="e")
        self.entry_end = ctk.CTkEntry(self.control_frame, width=70)
        self.entry_end.insert(0, "5.0")
        self.entry_end.grid(row=1, column=3, padx=2, pady=10, sticky="w")

        ctk.CTkLabel(self.control_frame, text="Step (mm):").grid(row=1, column=4, padx=10, pady=10, sticky="e")
        self.entry_step = ctk.CTkEntry(self.control_frame, width=70)
        self.entry_step.insert(0, "0.2")
        self.entry_step.grid(row=1, column=5, padx=2, pady=10, sticky="w")

        ctk.CTkLabel(self.control_frame, text="Exposure (s):").grid(row=1, column=6, padx=10, pady=10, sticky="e")
        self.entry_exposure = ctk.CTkEntry(self.control_frame, width=70)
        self.entry_exposure.insert(0, "1.0")
        self.entry_exposure.grid(row=1, column=7, padx=(2, 15), pady=10, sticky="w")

        # Dynamic Bars & Execution Controls
        self.progressbar = ctk.CTkProgressBar(self.control_frame)
        self.progressbar.grid(row=2, column=0, columnspan=8, padx=20, pady=(15, 5), sticky="ew")
        self.progressbar.set(0.0)

        self.start_btn = ctk.CTkButton(self.control_frame, text="Launch Experiment Pipeline",
                                       fg_color="#2FA572", hover_color="#208259", command=self.arm_scout_sequence)
        self.start_btn.grid(row=3, column=0, columnspan=4, padx=20, pady=20, sticky="ew")

        self.stop_btn = ctk.CTkButton(self.control_frame, text="Abrupt Interrupt (E-Stop)",
                                      fg_color="#C24641", hover_color="#91322E", command=self.trigger_emergency_stop)
        self.stop_btn.grid(row=3, column=4, columnspan=4, padx=20, pady=20, sticky="ew")

        # Terminal Simulation Output Box
        self.log_box = ctk.CTkTextbox(self.main_frame, height=170)
        self.log_box.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    def log(self, message):
        """Thread-safe standard console stream output update."""
        self.log_box.insert(ctk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_box.see(ctk.END)

    def connect_hardware_drivers(self):
        self.log("Initializing Hardware Drivers...")
        try:
            # Initialize ADVACAM Core
            pypixet.start()
            self.pixet_core = pypixet.pixet
            devices = self.pixet_core.devicesByType(self.pixet_core.PX_DEVTYPE_TPX3)
            if not devices:
                self.log("CORE ERROR: AdvaPIX detector stack undetected via USB.")
            else:
                self.camera = devices[0]
                self.log(f"Detector Synchronized: {self.camera.name()}")

            # Initialize Thorlabs Cube Controller
            self.motor = Thorlabs.KinesisMotor(SERIAL_NUM)
            self.log(f"Controller Bound (SN: {SERIAL_NUM}). Commencing axis homing...")
            # self.motor.home()  # Uncomment this during live integration
            self.log("Hardware deployment sequence standing by.")
        except Exception as e:
            self.log(f"Driver connection faulted or missing hardware: {e}")

    def trigger_emergency_stop(self):
        self.log(">>> SYSTEM CRITICAL INTERRUPT CAUGHT. Aborting after active cycle finishes... <<<")
        self.stop_requested = True

    def arm_scout_sequence(self):
        if self.camera is None or self.motor is None:
            self.log("FAIL: Hardware endpoints uninitialized. Cannot command execution.")
            return

        try:
            start_pos = float(self.entry_start.get())
            end_pos = float(self.entry_end.get())
            step_size = float(self.entry_step.get())
            exp_time = float(self.entry_exposure.get())
        except ValueError:
            self.log("INPUT ERROR: Non-numeric parsing exception detected in configuration fields.")
            return

        self.stop_requested = False
        self.start_btn.configure(state="disabled")

        self.log("Executing single scout reference frame to evaluate beam position...")
        test_file = os.path.join(SAVE_DIR, "scout_reference_frame.txt")

        # Pulls dynamic parameters directly from UI input and forces ASCII format output
        self.camera.doSimpleAcquisition(1, exp_time, self.pixet_core.PX_FTYPE = "TXT", test_file)

        test_matrix = load_and_calibrate_frame(test_file)

        # Render visual selection map (Blocks current thread till window closes)
        self.roi_coords = visually_select_roi(test_matrix)
        self.log(
            f"ROI Boundary Matrices Set -> Y[{self.roi_coords[0]}:{self.roi_coords[1]}], X[{self.roi_coords[2]}:{self.roi_coords[3]}]")

        # Hand off control to background thread to shield Main Loop UI response
        self.scan_thread = threading.Thread(target=self.execute_background_loop,
                                            args=(start_pos, end_pos, step_size, exp_time))
        self.scan_thread.start()

    def execute_background_loop(self, start_pos, end_pos, step_size, exp_time):
        """Phase 2: Automated step scan loop executing strictly on an independent thread."""
        num_steps = int(abs(end_pos - start_pos) / step_size) + 1
        positions_profile = []
        counts_profile = []

        y1, y2, x1, x2 = self.roi_coords
        self.log(f"Beginning structured scan pipeline profile targeting {num_steps} iterations...")

        for step in range(num_steps):
            if self.stop_requested:
                self.log("Pipeline breakdown sequence executed by administrative override.")
                break

            target_pos = start_pos + (step * step_size)

            # Step Axis Engine
            self.motor.move_to(target_pos)
            self.motor.wait_move()

            # Direct Storage File Sinking (TXT matrix output format)
            filename = os.path.join(SAVE_DIR, f"scan_pos_{target_pos:.2f}.txt")
            self.camera.doSimpleAcquisition(1, exp_time, self.pixet_core.PX_FTYPE = "TXT", filename)

            # Load ASCII data trivially via Numpy text parsing
            frame_matrix = load_and_calibrate_frame(filename)
            roi_slice = frame_matrix[y1:y2, x1:x2]
            summed_integral = np.sum(roi_slice)

            # Commit tracking variables
            positions_profile.append(target_pos)
            counts_profile.append(summed_integral)

            # Direct Thread-Safe UI Component Redraw
            current_progress = (step + 1) / num_steps
            self.progressbar.set(current_progress)
            self.lbl_position.configure(text=f"{target_pos:.2f}")
            self.lbl_counts.configure(text=f"{summed_integral:,}")
            self.log(
                f"Sweep {step + 1}/{num_steps} Complete | Position: {target_pos:.2f}mm | Counts: {summed_integral}")

        self.log("Pipeline routine complete. Parking motor. Closing links.")

        # Normalize UI State Elements
        self.start_btn.configure(state="normal")
        self.progressbar.set(1.0)

        # Safely delay and signal generation plotting mapping to avoid frame rendering violations
        self.after(400, lambda: self.render_profile_plot(positions_profile, counts_profile))

    def render_profile_plot(self, positions, counts):
        if not positions:
            return
        plt.figure(figsize=(8, 4.5))
        plt.plot(positions, counts, marker='o', linestyle='-', linewidth=2, color='#1f77b4')
        plt.title('Integrated Photon Flux Profile vs. Motor Position Target')
        plt.xlabel('Axis Spatial Displacement Position (mm)')
        plt.ylabel('Summed Photon Intensity (Counts)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def handle_shutdown_cleanup(self):
        """Graceful hardware detachment handler."""
        try:
            if self.motor:
                self.motor.close()
            pypixet.exit()
        except:
            pass
        self.destroy()


if __name__ == "__main__":
    app = MasterControlDashboard()
    app.protocol("WM_DELETE_WINDOW", app.handle_shutdown_cleanup)
    app.mainloop()