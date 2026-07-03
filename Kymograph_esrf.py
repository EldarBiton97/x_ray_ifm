"""
Modern Kymograph Dashboard using CustomTkinter
Features: Automated Data Ingestion, Metadata Tracking, Remarks, and Dark Mode UI.
Updated: Pandas I/O speedup, Corrected filename ingestion, Bottom-left Origin fix, UI Freeze Bypass.
"""

import os
import glob
import shutil
import json
from datetime import datetime
import customtkinter as ctk
from tkinter import filedialog
import tkinter.messagebox as messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from natsort import natsorted

# Set Matplotlib to dark mode to match CustomTkinter
plt.style.use('dark_background')

# Set CustomTkinter Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==========================================
# 1. Settings
# ==========================================
dt = 0.001           # seconds per frame
bin_frames = 1       # use 1 for no binning; try 5 or 10 if signal is weak

# ==========================================
# 2. Main Application Class
# ==========================================
class KymographApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Interferometry Kymograph Dashboard")
        self.geometry("1500x950")

        self.folder_path = ""
        self.files = []
        self.display_image = None

        # Start with the Data Ingestion Screen
        self.build_ingestion_screen()

    # ==========================================
    # SCREEN 0: DATA INGESTION & METADATA
    # ==========================================
    def build_ingestion_screen(self):
        self.ingest_frame = ctk.CTkFrame(self)
        self.ingest_frame.pack(fill="both", expand=True, padx=50, pady=50)

        ctk.CTkLabel(self.ingest_frame, text="Data Ingestion & Metadata Setup", font=("Arial", 28, "bold")).pack(pady=(40, 20))

        # --- File Paths ---
        path_frame = ctk.CTkFrame(self.ingest_frame, fg_color="transparent")
        path_frame.pack(fill="x", padx=100, pady=10)

        ctk.CTkLabel(path_frame, text="PIXet Inbox (Source):", width=150, anchor="e").grid(row=0, column=0, padx=10, pady=10)
        self.ui_inbox = ctk.CTkEntry(path_frame, width=400)
        self.ui_inbox.insert(0, os.path.join(os.getcwd(), "PIXet_Inbox"))
        self.ui_inbox.grid(row=0, column=1, padx=10)
        ctk.CTkButton(path_frame, text="Browse", width=80, command=lambda: self.browse_folder(self.ui_inbox)).grid(row=0, column=2)

        ctk.CTkLabel(path_frame, text="Archive Destination:", width=150, anchor="e").grid(row=1, column=0, padx=10, pady=10)
        self.ui_archive = ctk.CTkEntry(path_frame, width=400)
        self.ui_archive.insert(0, os.path.join(os.getcwd(), "Archived_Measurements"))
        self.ui_archive.grid(row=1, column=1, padx=10)
        ctk.CTkButton(path_frame, text="Browse", width=80, command=lambda: self.browse_folder(self.ui_archive)).grid(row=1, column=2)

        # --- Parameters ---
        param_frame = ctk.CTkFrame(self.ingest_frame)
        param_frame.pack(fill="x", padx=100, pady=30)

        ctk.CTkLabel(param_frame, text="Measurement Parameters", font=("Arial", 18, "bold")).pack(pady=10)

        grid_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        grid_frame.pack(pady=10)

        ctk.CTkLabel(grid_frame, text="Phase Object:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.ui_phase = ctk.CTkEntry(grid_frame, width=300, placeholder_text="e.g., Si(333), Air, Glass")
        self.ui_phase.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        ctk.CTkLabel(grid_frame, text="AC State:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.ui_ac = ctk.CTkSwitch(grid_frame, text="ON", progress_color="cyan")
        self.ui_ac.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        ctk.CTkLabel(grid_frame, text="Stabilizer State:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.ui_stab = ctk.CTkSwitch(grid_frame, text="ON", progress_color="magenta")
        self.ui_stab.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        ctk.CTkLabel(grid_frame, text="Remarks:").grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.ui_remarks = ctk.CTkEntry(grid_frame, width=400, placeholder_text="Any issues, alignment changes, or general notes...")
        self.ui_remarks.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # --- Action Buttons ---
        btn_frame = ctk.CTkFrame(self.ingest_frame, fg_color="transparent")
        btn_frame.pack(pady=20)

        ctk.CTkButton(btn_frame, text="Ingest, Rename & Analyze", font=("Arial", 16, "bold"),
                      fg_color="green", hover_color="darkgreen", height=50, width=250,
                      command=self.execute_ingestion).pack(side="left", padx=20)

        ctk.CTkButton(btn_frame, text="Load Existing Archive (Skip Ingest)", font=("Arial", 14),
                      fg_color="gray30", hover_color="gray20", height=50,
                      command=self.load_existing_archive).pack(side="left", padx=20)

    def browse_folder(self, entry_widget):
        folder = filedialog.askdirectory()
        if folder:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, folder)

    def execute_ingestion(self):
        inbox = self.ui_inbox.get()
        archive_root = self.ui_archive.get()
        phase = self.ui_phase.get().strip() or "None"
        remarks = self.ui_remarks.get().strip()
        ac = "ON" if self.ui_ac.get() else "OFF"
        stab = "ON" if self.ui_stab.get() else "OFF"

        search_pattern = os.path.join(inbox, '*_Event.txt')
        target_files = glob.glob(search_pattern)
        if not target_files:
            messagebox.showerror("Error", f"No PIXet Event files found in:\n{inbox}")
            return

        timestamp_folder = datetime.now().strftime("%b%d_%H%M")
        safe_phase = "".join([c for c in phase if c.isalnum() or c in "()-_"])
        new_folder_name = f"{timestamp_folder}_{safe_phase}_AC-{ac}_Stab-{stab}"
        self.folder_path = os.path.join(archive_root, new_folder_name)

        os.makedirs(self.folder_path, exist_ok=True)

        all_txt_files = glob.glob(os.path.join(inbox, '*.txt*'))
        for f in all_txt_files:
            shutil.move(f, os.path.join(self.folder_path, os.path.basename(f)))

        metadata = {
            "timestamp_full": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase_object": phase,
            "ac_state": ac,
            "stabilizer_state": stab,
            "remarks": remarks,
            "original_source": inbox,
            "frames_moved": len(target_files)
        }
        with open(os.path.join(self.folder_path, "metadata.json"), "w") as mfile:
            json.dump(metadata, mfile, indent=4)

        self.load_data()

    def load_existing_archive(self):
        self.folder_path = filedialog.askdirectory(title='Select an existing archive folder')
        if not self.folder_path: return
        self.load_data()

    def load_data(self):
        search_pattern = os.path.join(self.folder_path, '*_Event.txt')
        self.files = natsorted(glob.glob(search_pattern))
        self.num_files = len(self.files)

        if self.num_files == 0:
            messagebox.showerror("Error", "No Event frames found in the selected directory.")
            return

        self.num_bins = self.num_files // bin_frames
        self.dt_bin = dt * bin_frames
        self.time = np.arange(self.num_bins) * self.dt_bin

        # Force UI update so it doesn't look frozen
        self.title(f"Loading {self.num_files} frames... Please wait.")
        self.update()

        # Read first frame
        first_frame = pd.read_csv(self.files[0], sep=r'\s+', header=None).values
        self.Ny, self.Nx = first_frame.shape

        sum_image = np.zeros((self.Ny, self.Nx))

        # 50-Frame Bypass for instant UI loading
        preview_limit = min(50, self.num_files)
        print(f"Building master preview image from first {preview_limit} frames...")

        for f in self.files[:preview_limit]:
            sum_image += pd.read_csv(f, sep=r'\s+', header=None).values

        self.display_image = np.log10(sum_image + 1)

        # Restore title
        self.title("Interferometry Kymograph Dashboard")

        self.ingest_frame.destroy()
        self.build_main_dashboard()

    # ==========================================
    # DASHBOARD TABS
    # ==========================================
    def build_main_dashboard(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        self.tab_setup = self.tabview.add("1. ROI Setup")
        self.tab_raw = self.tabview.add("2. Raw Kymographs")
        self.tab_norm = self.tabview.add("3. Normalized Kymographs")

        self.build_setup_tab()

    def build_setup_tab(self):
        frame_plot = ctk.CTkFrame(self.tab_setup, fg_color="transparent")
        frame_plot.pack(side="left", fill="both", expand=True)

        frame_controls = ctk.CTkFrame(self.tab_setup, width=320)
        frame_controls.pack(side="right", fill="y", padx=10, pady=10)

        self.fig_setup, self.ax_setup = plt.subplots(figsize=(8, 8))

        # Bottom-left origin fix for the physical detector geometry
        self.ax_setup.imshow(self.display_image, cmap='gray', origin='lower')
        self.ax_setup.set_title("Select a Spot to edit, then drag to resize")
        self.fig_setup.tight_layout()

        self.rect1 = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='cyan', facecolor='cyan', alpha=0.2)
        self.rect2 = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='magenta', facecolor='magenta', alpha=0.2)
        self.ax_setup.add_patch(self.rect1)
        self.ax_setup.add_patch(self.rect2)

        self.canvas_setup = FigureCanvasTkAgg(self.fig_setup, master=frame_plot)
        self.canvas_setup.draw()
        self.canvas_setup.get_tk_widget().pack(side="top", fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_setup, frame_plot)
        toolbar.update()

        w, h = self.Nx // 10, self.Ny // 10
        self.active_roi = "Spot 1"
        self.updating_ui = False

        self.coords = {
            'x1a': ctk.StringVar(value=str(self.Nx // 4 - w // 2)),
            'x1b': ctk.StringVar(value=str(self.Nx // 4 + w // 2)),
            'y1a': ctk.StringVar(value=str(self.Ny // 2 - h // 2)),
            'y1b': ctk.StringVar(value=str(self.Ny // 2 + h // 2)),
            'x2a': ctk.StringVar(value=str(3 * self.Nx // 4 - w // 2)),
            'x2b': ctk.StringVar(value=str(3 * self.Nx // 4 + w // 2)),
            'y2a': ctk.StringVar(value=str(self.Ny // 2 - h // 2)),
            'y2b': ctk.StringVar(value=str(self.Ny // 2 + h // 2)),
        }

        def sync_static_rectangles():
            try:
                x1a, x1b = int(self.coords['x1a'].get()), int(self.coords['x1b'].get())
                y1a, y1b = int(self.coords['y1a'].get()), int(self.coords['y1b'].get())
                self.rect1.set_bounds(min(x1a, x1b), min(y1a, y1b), abs(x1b-x1a), abs(y1b-y1a))

                x2a, x2b = int(self.coords['x2a'].get()), int(self.coords['x2b'].get())
                y2a, y2b = int(self.coords['y2a'].get()), int(self.coords['y2b'].get())
                self.rect2.set_bounds(min(x2a, x2b), min(y2a, y2b), abs(x2b-x2a), abs(y2b-y2a))
                self.canvas_setup.draw_idle()
            except ValueError:
                pass

        def update_text_from_plot(eclick=None, erelease=None):
            if self.updating_ui: return
            self.updating_ui = True

            ext = self.rs.extents
            if self.active_roi == "Spot 1":
                self.coords['x1a'].set(str(int(round(ext[0]))))
                self.coords['x1b'].set(str(int(round(ext[1]))))
                self.coords['y1a'].set(str(int(round(ext[2]))))
                self.coords['y1b'].set(str(int(round(ext[3]))))
            else:
                self.coords['x2a'].set(str(int(round(ext[0]))))
                self.coords['x2b'].set(str(int(round(ext[1]))))
                self.coords['y2a'].set(str(int(round(ext[2]))))
                self.coords['y2b'].set(str(int(round(ext[3]))))

            sync_static_rectangles()
            self.updating_ui = False

        def update_plot_from_text(*args):
            if self.updating_ui: return
            self.updating_ui = True
            sync_static_rectangles()
            try:
                if self.active_roi == "Spot 1":
                    self.rs.extents = (int(self.coords['x1a'].get()), int(self.coords['x1b'].get()),
                                       int(self.coords['y1a'].get()), int(self.coords['y1b'].get()))
                else:
                    self.rs.extents = (int(self.coords['x2a'].get()), int(self.coords['x2b'].get()),
                                       int(self.coords['y2a'].get()), int(self.coords['y2b'].get()))
            except ValueError:
                pass
            self.updating_ui = False

        def change_active_roi(value):
            self.active_roi = value
            update_plot_from_text()

        self.rs = RectangleSelector(self.ax_setup, update_text_from_plot,
                                    useblit=True, interactive=True, drag_from_anywhere=True,
                                    props=dict(facecolor='none', edgecolor='white', linestyle='--', linewidth=2, alpha=0.8))

        for var in self.coords.values():
            var.trace_add('write', update_plot_from_text)

        ctk.CTkLabel(frame_controls, text="Active Editor Tool", font=("Arial", 16, "bold")).pack(pady=(15, 5))

        self.roi_switch = ctk.CTkSegmentedButton(frame_controls, values=["Spot 1", "Spot 2"], command=change_active_roi)
        self.roi_switch.set("Spot 1")
        self.roi_switch.pack(fill="x", padx=10, pady=(0, 15))

        def create_coord_inputs(parent, title, keys, text_color):
            section_frame = ctk.CTkFrame(parent, fg_color="transparent")
            section_frame.pack(fill="x", pady=5, padx=10)
            ctk.CTkLabel(section_frame, text=title, font=("Arial", 16, "bold"), text_color=text_color).pack(pady=(5, 5))

            labels = ["X Min:", "X Max:", "Y Min:", "Y Max:"]
            for key, label in zip(keys, labels):
                row = ctk.CTkFrame(section_frame, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=label, width=60, anchor='e').pack(side="left", padx=5)
                ctk.CTkEntry(row, textvariable=self.coords[key], width=100).pack(side="left")

        create_coord_inputs(frame_controls, "Spot 1 (Cyan)", ['x1a', 'x1b', 'y1a', 'y1b'], '#00FFFF')
        create_coord_inputs(frame_controls, "Spot 2 (Magenta)", ['x2a', 'x2b', 'y2a', 'y2b'], '#FF00FF')

        self.btn_process = ctk.CTkButton(frame_controls, text="Process Kymographs",
                                         fg_color="green", hover_color="darkgreen", height=40,
                                         font=("Arial", 14, "bold"), command=self.process_data)
        self.btn_process.pack(pady=20, fill="x", padx=20)

        self.lbl_status = ctk.CTkLabel(frame_controls, text=f"Data Path: .../{os.path.basename(self.folder_path)}", text_color="gray", font=("Arial", 12, "italic"))
        self.lbl_status.pack()

        self.updating_ui = False
        update_plot_from_text()

    def process_data(self):
        self.btn_process.configure(state="disabled")
        self.lbl_status.configure(text="Processing frames... Please wait.", text_color="#2980b9")
        self.update()

        x1a, x1b = int(self.coords['x1a'].get()), int(self.coords['x1b'].get())
        y1a, y1b = int(self.coords['y1a'].get()), int(self.coords['y1b'].get())
        x2a, x2b = int(self.coords['x2a'].get()), int(self.coords['x2b'].get())
        y2a, y2b = int(self.coords['y2a'].get()), int(self.coords['y2b'].get())

        self.width1 = x1b - x1a + 1
        self.width2 = x2b - x2a + 1

        self.kymo1 = np.zeros((self.num_bins, self.width1))
        self.kymo2 = np.zeros((self.num_bins, self.width2))
        self.counts_spot1 = np.zeros(self.num_bins)
        self.counts_spot2 = np.zeros(self.num_bins)

        for b in range(self.num_bins):
            bin_frame = np.zeros((self.Ny, self.Nx))
            for i in range(b * bin_frames, (b + 1) * bin_frames):
                bin_frame += pd.read_csv(self.files[i], sep=r'\s+', header=None).values

            spot1 = bin_frame[y1a:y1b+1, x1a:x1b+1]
            spot2 = bin_frame[y2a:y2b+1, x2a:x2b+1]

            self.kymo1[b, :] = np.sum(spot1, axis=0)
            self.kymo2[b, :] = np.sum(spot2, axis=0)
            self.counts_spot1[b] = np.sum(spot1)
            self.counts_spot2[b] = np.sum(spot2)

        eps = np.finfo(float).eps
        kymo1_norm = self.kymo1 / np.maximum(np.sum(self.kymo1, axis=1, keepdims=True), eps)
        kymo2_norm = self.kymo2 / np.maximum(np.sum(self.kymo2, axis=1, keepdims=True), eps)

        self.kymo1_hp = kymo1_norm - np.mean(kymo1_norm, axis=0, keepdims=True)
        self.kymo2_hp = kymo2_norm - np.mean(kymo2_norm, axis=0, keepdims=True)

        np.savetxt(os.path.join(self.folder_path, 'spot1_raw.csv'), self.kymo1, delimiter=',')
        np.savetxt(os.path.join(self.folder_path, 'spot2_raw.csv'), self.kymo2, delimiter=',')
        np.savetxt(os.path.join(self.folder_path, 'spot1_hp.csv'), self.kymo1_hp, delimiter=',')
        np.savetxt(os.path.join(self.folder_path, 'spot2_hp.csv'), self.kymo2_hp, delimiter=',')
        pd.DataFrame({'Time_s': self.time, 'Spot1': self.counts_spot1, 'Spot2': self.counts_spot2}).to_csv(
            os.path.join(self.folder_path, 'counts.csv'), index=False)

        self.lbl_status.configure(text="Done! Building Tabs...", text_color="green")
        self.update()

        self.build_result_tab(self.tab_raw, self.kymo1, self.kymo2, 'gray', is_hp=False)
        self.build_result_tab(self.tab_norm, self.kymo1_hp, self.kymo2_hp, 'seismic', is_hp=True)

        self.btn_process.configure(state="normal")
        self.lbl_status.configure(text="Data Processed & Saved.", text_color="gray")
        self.tabview.set("2. Raw Kymographs")

    def build_result_tab(self, parent_frame, data1, data2, colormap, is_hp):
        control_frame = ctk.CTkFrame(parent_frame, width=280)
        control_frame.pack(side="left", fill="y", padx=10, pady=10)

        plot_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 2.5, 2])

        ax1 = fig.add_subplot(gs[0:2, 0])
        ax2 = fig.add_subplot(gs[0:2, 1])
        ax_line = fig.add_subplot(gs[2, :])

        im1 = ax1.imshow(data1, aspect='auto', cmap=colormap, extent=[1, self.width1, self.time[-1], self.time[0]])
        ax1.set_title('Spot 1' + (' (High-Pass)' if is_hp else ' (Raw)'))
        fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(data2, aspect='auto', cmap=colormap, extent=[1, self.width2, self.time[-1], self.time[0]])
        ax2.set_title('Spot 2' + (' (High-Pass)' if is_hp else ' (Raw)'))
        fig.colorbar(im2, ax=ax2)

        line1, = ax_line.plot(self.time, self.counts_spot1, label='Spot 1', color='cyan')
        line2, = ax_line.plot(self.time, self.counts_spot2, label='Spot 2', color='magenta')

        ax_line.set_xlabel('Time [s]')
        ax_line.set_ylabel('Total Photon Counts')
        ax_line.grid(True, alpha=0.2)
        ax_line.legend()

        for ax in [ax1, ax2]:
            ax.set_ylabel('Time [s]')
            ax.set_xlabel('Horizontal Pixel')

        fig.tight_layout(pad=3.0)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()

        ctk.CTkLabel(control_frame, text="Display Settings", font=("Arial", 20, "bold")).pack(pady=(20, 10))

        def update_img1(val):
            lim = np.percentile(np.abs(data1) if is_hp else data1, float(val))
            im1.set_clim(vmin=-lim if is_hp else 0, vmax=lim)
            lbl_val1.configure(text=f"{val:.1f}%")
            canvas.draw_idle()

        def update_img2(val):
            lim = np.percentile(np.abs(data2) if is_hp else data2, float(val))
            im2.set_clim(vmin=-lim if is_hp else 0, vmax=lim)
            lbl_val2.configure(text=f"{val:.1f}%")
            canvas.draw_idle()

        frame_s1 = ctk.CTkFrame(control_frame, fg_color="transparent")
        frame_s1.pack(fill="x", pady=15, padx=15)
        ctk.CTkLabel(frame_s1, text="Spot 1 Saturation:").pack(anchor='w')
        s1 = ctk.CTkSlider(frame_s1, from_=50 if is_hp else 90, to=100, command=update_img1)
        s1.set(98.0 if is_hp else 99.5)
        s1.pack(side="left", fill="x", expand=True, pady=5)
        lbl_val1 = ctk.CTkLabel(frame_s1, text=f"{s1.get():.1f}%", width=50)
        lbl_val1.pack(side="right")

        frame_s2 = ctk.CTkFrame(control_frame, fg_color="transparent")
        frame_s2.pack(fill="x", pady=15, padx=15)
        ctk.CTkLabel(frame_s2, text="Spot 2 Saturation:").pack(anchor='w')
        s2 = ctk.CTkSlider(frame_s2, from_=50 if is_hp else 90, to=100, command=update_img2)
        s2.set(98.0 if is_hp else 99.5)
        s2.pack(side="left", fill="x", expand=True, pady=5)
        lbl_val2 = ctk.CTkLabel(frame_s2, text=f"{s2.get():.1f}%", width=50)
        lbl_val2.pack(side="right")

        ctk.CTkFrame(control_frame, height=2, fg_color="gray30").pack(fill="x", pady=20, padx=20)

        ctk.CTkLabel(control_frame, text="Histogram Overlay", font=("Arial", 16, "bold")).pack(pady=(0, 10))

        var1 = ctk.BooleanVar(value=True)
        var2 = ctk.BooleanVar(value=True)

        def toggle_lines():
            line1.set_visible(var1.get())
            line2.set_visible(var2.get())
            ax_line.relim()
            ax_line.autoscale_view()

            vis_lines = [l for l in [line1, line2] if l.get_visible()]
            if ax_line.get_legend(): ax_line.get_legend().remove()
            if vis_lines: ax_line.legend(vis_lines, [l.get_label() for l in vis_lines])
            canvas.draw_idle()

        sw1 = ctk.CTkSwitch(control_frame, text="Spot 1 Counts", variable=var1, command=toggle_lines, progress_color="cyan")
        sw1.pack(anchor='w', padx=20, pady=10)

        sw2 = ctk.CTkSwitch(control_frame, text="Spot 2 Counts", variable=var2, command=toggle_lines, progress_color="magenta")
        sw2.pack(anchor='w', padx=20, pady=10)

        update_img1(s1.get())
        update_img2(s2.get())

if __name__ == "__main__":
    app = KymographApp()
    app.mainloop()