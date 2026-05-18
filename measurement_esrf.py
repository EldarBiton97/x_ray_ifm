import customtkinter as ctk
import nidaqmx
from nidaqmx.constants import (Edge, CountDirection, AcquisitionType, FrequencyUnits, Level, TaskMode, TriggerType)
import numpy as np
import time
import h5py
import threading

# Set the theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class DAQControlApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("ESRF MI-1545: Detector Sync Control")
        self.geometry("900x850")
        
        # --- Data Mapping ---
        self.bin_options = {
            '1 µs (1 MHz)': 1e6, 
            '10 µs (100 kHz)': 1e5, 
            '100 µs (10 kHz)': 1e4,
            '1 ms (1 kHz)': 1e3, 
            '10 ms (100 Hz)': 1e2, 
            '100 ms (10 Hz)': 10.0
        }
        
        self.build_ui()
        self.update_hardware_calc() # Run initial calculation
        self.stop_flag = False

    def build_ui(self):
        # --- Top Frame: Setup ---
        setup_frame = ctk.CTkFrame(self)
        setup_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(setup_frame, text="Measurement Setup", font=("Arial", 20, "bold")).pack(pady=10)
        
        # File Name
        file_frame = ctk.CTkFrame(setup_frame, fg_color="transparent")
        file_frame.pack(pady=5, fill="x", padx=20)
        ctk.CTkLabel(file_frame, text="File Prefix:").pack(side="left", padx=10)
        self.ui_filename = ctk.CTkEntry(file_frame, width=300)
        self.ui_filename.insert(0, "esrf_px5_data")
        self.ui_filename.pack(side="left")

        # Time Inputs
        time_frame = ctk.CTkFrame(setup_frame, fg_color="transparent")
        time_frame.pack(pady=10, fill="x", padx=20)
        
        ctk.CTkLabel(time_frame, text="Days:").pack(side="left", padx=5)
        self.ui_days = ctk.CTkEntry(time_frame, width=60)
        self.ui_days.insert(0, "0")
        self.ui_days.pack(side="left", padx=5)
        self.ui_days.bind("<KeyRelease>", self.update_hardware_calc)
        
        ctk.CTkLabel(time_frame, text="Hours:").pack(side="left", padx=5)
        self.ui_hours = ctk.CTkEntry(time_frame, width=60)
        self.ui_hours.insert(0, "1")
        self.ui_hours.pack(side="left", padx=5)
        self.ui_hours.bind("<KeyRelease>", self.update_hardware_calc)
        
        ctk.CTkLabel(time_frame, text="Minutes:").pack(side="left", padx=5)
        self.ui_minutes = ctk.CTkEntry(time_frame, width=60)
        self.ui_minutes.insert(0, "0")
        self.ui_minutes.pack(side="left", padx=5)
        self.ui_minutes.bind("<KeyRelease>", self.update_hardware_calc)

        ctk.CTkLabel(time_frame, text="Split Every (hr):").pack(side="left", padx=(30, 5))
        self.ui_split_hours = ctk.CTkEntry(time_frame, width=60)
        self.ui_split_hours.insert(0, "1")
        self.ui_split_hours.pack(side="left", padx=5)

        # DAQ Parameters
        daq_frame = ctk.CTkFrame(setup_frame, fg_color="transparent")
        daq_frame.pack(pady=10, fill="x", padx=20)
        
        ctk.CTkLabel(daq_frame, text="Bin Size:").pack(side="left", padx=10)
        self.ui_bin_size = ctk.CTkOptionMenu(daq_frame, values=list(self.bin_options.keys()), command=self.update_hardware_calc)
        self.ui_bin_size.set('1 µs (1 MHz)')
        self.ui_bin_size.pack(side="left", padx=10)

        ctk.CTkLabel(daq_frame, text="Chunk (s):").pack(side="left", padx=(30, 10))
        self.ui_chunk_val = ctk.CTkLabel(daq_frame, text="10")
        self.ui_chunk_val.pack(side="left")
        self.ui_chunk = ctk.CTkSlider(daq_frame, from_=1, to=60, number_of_steps=59, command=self.slider_event)
        self.ui_chunk.set(10)
        self.ui_chunk.pack(side="left", padx=10)

        # Hardware Calc Display
        self.ui_hardware_calc = ctk.CTkLabel(setup_frame, text="", justify="left", font=("Consolas", 14))
        self.ui_hardware_calc.pack(pady=15)

        # --- Middle Frame: Controls & Status ---
        control_frame = ctk.CTkFrame(self, fg_color="transparent")
        control_frame.pack(pady=10, padx=20, fill="x")

        self.btn_start = ctk.CTkButton(control_frame, text="START MEASUREMENT", fg_color="green", hover_color="darkgreen", height=50, command=self.launch_daq_experiment)
        self.btn_start.pack(side="left", expand=True, padx=10)
        
        self.btn_stop = ctk.CTkButton(control_frame, text="STOP MEASUREMENT", fg_color="red", hover_color="darkred", height=50, state="disabled", command=self.stop_measurement)
        self.btn_stop.pack(side="right", expand=True, padx=10)

        # Progress
        self.ui_progress = ctk.CTkProgressBar(self, width=600)
        self.ui_progress.set(0)
        self.ui_progress.pack(pady=15)
        self.ui_progress_text = ctk.CTkLabel(self, text="0.0% (0 / 0 s)", font=("Arial", 14))
        self.ui_progress_text.pack()

        self.ui_status = ctk.CTkLabel(self, text="Live Status: Idle (Waiting to Start)", text_color="gray", font=("Arial", 16, "bold"))
        self.ui_status.pack(pady=10)

        # --- Bottom Frame: Console Log ---
        self.console_out = ctk.CTkTextbox(self, height=200, state="disabled", font=("Consolas", 12))
        self.console_out.pack(pady=10, padx=20, fill="both", expand=True)

    def slider_event(self, value):
        self.ui_chunk_val.configure(text=str(int(value)))
        self.update_hardware_calc()

    def get_int_safe(self, entry_widget, default=0):
        try:
            return int(entry_widget.get())
        except ValueError:
            return default

    def update_hardware_calc(self, *args):
        duration_s = (self.get_int_safe(self.ui_days) * 86400) + (self.get_int_safe(self.ui_hours) * 3600) + (self.get_int_safe(self.ui_minutes) * 60)
        chunk_s = int(self.ui_chunk.get())
        bin_rate = self.bin_options[self.ui_bin_size.get()]
        
        peak_ram_mb = ((chunk_s * bin_rate * 16.0) / 1e6) + 50.0 
        disk_gb_uncompressed = ((duration_s * bin_rate * 1.0) / 1e6) / 1024.0
        disk_gb_compressed = disk_gb_uncompressed * 0.15 
        
        calc_text = (
            f"⚡ PEAK RAM REQUIRED: ~{peak_ram_mb:.0f} MB\n"
            f"💾 EST. TOTAL DISK SPACE (Uncompressed): ~{disk_gb_uncompressed:.4f} GB\n"
            f"📦 EST. TOTAL DISK SPACE (HDF5 Compressed): ~{disk_gb_compressed:.4f} GB"
        )
        self.ui_hardware_calc.configure(text=calc_text)

    def log_msg(self, msg):
        self.console_out.configure(state="normal")
        self.console_out.insert("end", msg + "\n")
        self.console_out.see("end")
        self.console_out.configure(state="disabled")

    def stop_measurement(self):
        self.stop_flag = True
        self.btn_stop.configure(state="disabled")
        self.ui_status.configure(text="Live Status: Stopping... Waiting for current chunk to finish securely.", text_color="red")
        self.log_msg("\n[!] STOP SIGNAL RECEIVED. Securing data and stopping hardware...")

    def launch_daq_experiment(self):
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        thread = threading.Thread(target=self.daq_worker, daemon=True)
        thread.start()

    def daq_worker(self):
        self.stop_flag = False
        
        self.console_out.configure(state="normal")
        self.console_out.delete("0.0", "end")
        self.console_out.configure(state="disabled")
        
        self.ui_status.configure(text="Live Status: Initializing Setup...", text_color="orange")
        
        total_duration_s = (self.get_int_safe(self.ui_days) * 86400) + (self.get_int_safe(self.ui_hours) * 3600) + (self.get_int_safe(self.ui_minutes) * 60)
        split_s = self.get_int_safe(self.ui_split_hours, 1) * 3600
        chunk_s = float(int(self.ui_chunk.get()))
        binRateHz = float(self.bin_options[self.ui_bin_size.get()])
        
        if total_duration_s <= 0:
            self.log_msg("Error: Total duration must be greater than 0.")
            self.ui_status.configure(text="Live Status: Error - Invalid Duration", text_color="red")
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            return
            
        if split_s <= 0:
            split_s = total_duration_s

        self.ui_progress.set(0)
        self.ui_progress_text.configure(text=f"0.0% (0 / {total_duration_s} s)")

        bin_s = 1.0 / binRateHz
        chunk_bins = int(round(chunk_s * binRateHz))
        num_splits = int(np.ceil(total_duration_s / split_s))
        
        self.log_msg(f"--- EXPERIMENT SETTINGS ---")
        self.log_msg(f"File Prefix    : {self.ui_filename.get()}")
        self.log_msg(f"Total Duration : {total_duration_s} seconds")
        self.log_msg(f"File Splitting : Every {split_s} seconds")
        self.log_msg(f"Total Files    : {num_splits}")
        self.log_msg(f"Bin Resolution : {bin_s * 1e6:.1f} µs ({binRateHz} Hz)")
        self.log_msg(f"Chunk Size     : {chunk_s} seconds ({chunk_bins} bins per chunk)")
        self.log_msg(f"---------------------------\n")

        dev = "Dev1"
        startOutPFI, startInPFI = f"/{dev}/PFI1", f"/{dev}/PFI11"
        binInPFI, binOutPFI = f"/{dev}/PFI10", f"/{dev}/PFI14"
        px5PFI = f"/{dev}/PFI0"
        
        px5Ctr, binCtr, startCtr = f"{dev}/ctr0", f"{dev}/ctr2", f"{dev}/ctr3"
        startFreqHz, startDuty = 1.0, 0.0001
        buffer_size = int(chunk_bins * 5) 

        try:
            self.ui_status.configure(text="Live Status: Initializing NI-DAQmx Hardware Tasks...", text_color="orange")
            
            with nidaqmx.Task() as tBIN, nidaqmx.Task() as tPX5, nidaqmx.Task() as tSTART:
                tBIN.co_channels.add_co_pulse_chan_freq(binCtr, units=FrequencyUnits.HZ, idle_state=Level.LOW, freq=binRateHz, duty_cycle=0.5).co_pulse_term = binOutPFI
                tBIN.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=buffer_size)

                tPX5.ci_channels.add_ci_count_edges_chan(px5Ctr, edge=Edge.RISING, initial_count=0, count_direction=CountDirection.COUNT_UP).ci_count_edges_term = px5PFI
                tPX5.timing.cfg_samp_clk_timing(rate=binRateHz, source=binInPFI, active_edge=Edge.RISING, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=buffer_size)
                tPX5.triggers.arm_start_trigger.trig_type = TriggerType.DIGITAL_EDGE
                tPX5.triggers.arm_start_trigger.dig_edge_edge, tPX5.triggers.arm_start_trigger.dig_edge_src = Edge.RISING, startInPFI

                tSTART.co_channels.add_co_pulse_chan_freq(startCtr, units=FrequencyUnits.HZ, idle_state=Level.LOW, freq=startFreqHz, duty_cycle=startDuty).co_pulse_term = startOutPFI
                tSTART.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)

                for task in [tBIN, tPX5, tSTART]: task.control(TaskMode.TASK_COMMIT)
                tBIN.start(); tPX5.start()
                time.sleep(0.05)
                tSTART.start()

                self.log_msg("Hardware Recording Started. Sync Heartbeat Active...")
                
                last_px5_cum = 0
                total_px5 = 0
                global_time_read = 0.0 

                # --- MASTER SPLITTING LOOP ---
                for split_idx in range(num_splits):
                    if self.stop_flag: break 
                    
                    hdf5_filename = f"{self.ui_filename.get()}_{split_idx:03d}.h5"
                    remaining_s = total_duration_s - (split_idx * split_s)
                    current_split_s = min(split_s, remaining_s)
                    current_split_chunks = int(np.ceil(current_split_s / chunk_s))
                    
                    with h5py.File(hdf5_filename, "w") as f:
                        self.log_msg(f"[→] Opened new file: {hdf5_filename}")
                        dset_px5 = f.create_dataset("px5CountsPerBin", shape=(0,), maxshape=(None,), dtype='uint8', chunks=(chunk_bins,), compression="gzip")
                        f.attrs['bin_s'], f.attrs['split_duration_s'] = bin_s, current_split_s
                        
                        # --- CHUNK READING LOOP ---
                        for i in range(current_split_chunks):
                            if self.stop_flag: break 
                            
                            self.ui_status.configure(text=f"Live Status: Recording File {split_idx+1}/{num_splits} (Chunk {i+1}/{current_split_chunks}) | Total PX5 Counts: {total_px5}", text_color="#2980b9")
                            
                            time_read_so_far = i * chunk_s
                            if time_read_so_far + chunk_s > current_split_s:
                                bins_to_read = int(round((current_split_s - time_read_so_far) * binRateHz))
                            else:
                                bins_to_read = chunk_bins
                                
                            if bins_to_read <= 0: break
                            
                            px5_raw = np.array(tPX5.read(number_of_samples_per_channel=bins_to_read, timeout=chunk_s + 5.0), dtype=np.uint32)

                            px5_diff = np.diff(np.concatenate(([last_px5_cum], px5_raw))).astype(np.int64)
                            px5_diff[px5_diff < 0] += 4294967296
                            px5_counts = np.clip(px5_diff, 0, 255).astype(np.uint8)

                            current_size = dset_px5.shape[0]
                            dset_px5.resize((current_size + bins_to_read,))
                            dset_px5[current_size:] = px5_counts

                            last_px5_cum = px5_raw[-1]
                            total_px5 += np.sum(px5_counts)
                            
                            # Update Progress Bar
                            global_time_read += (bins_to_read / binRateHz)
                            progress_val = min(global_time_read / total_duration_s, 1.0)
                            self.ui_progress.set(progress_val)
                            self.ui_progress_text.configure(text=f"{(progress_val * 100):.1f}% ({global_time_read:.0f} / {total_duration_s} s)")

            if self.stop_flag:
                self.log_msg(f"\nMeasurement Aborted by User! All data safely saved.")
                self.ui_status.configure(text="Live Status: Measurement Aborted. Data Saved.", text_color="red")
            else:
                self.ui_progress.set(1.0)
                self.ui_progress_text.configure(text=f"100% ({total_duration_s} / {total_duration_s} s)")
                self.log_msg(f"\nMeasurement Complete! All files safely saved.")
                self.ui_status.configure(text="Live Status: Measurement Complete! All Data Saved.", text_color="green")
                
        except Exception as e:
            self.log_msg(f"\nHardware Error Encountered: {e}")
            self.ui_status.configure(text="Live Status: Hardware Error! Check console log.", text_color="red")
            
        finally:
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")

if __name__ == "__main__":
    app = DAQControlApp()
    app.mainloop()