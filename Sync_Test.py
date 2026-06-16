import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
import h5py
import os
import xml.etree.ElementTree as ET
import base64
from numba import njit
from scipy.signal import correlate, correlation_lags  # <--- Added for Sync Test
import re
import mmap

# ==============================================================================
# PART 1: THE T3P PARSER & STRICT GEOMETRY VALIDATION
# DAQ/PX5-LOCKED VERSION (SEQUENTIAL STATE MACHINE)
# ==============================================================================

def _unwrap_28bit_ticks(raw_ticks, period=268435456.0):
    raw_ticks = np.asarray(raw_ticks, dtype=np.float64)
    if raw_ticks.size == 0: return raw_ticks.copy()
    half_period = period / 2.0
    deltas = np.diff(raw_ticks)
    roll_forward = (deltas < -half_period).astype(np.int64)
    roll_backward = (deltas > half_period).astype(np.int64)
    cycles = np.zeros(raw_ticks.size, dtype=np.int64)
    cycles[1:] = np.cumsum(roll_forward - roll_backward)
    return raw_ticks + cycles * period

def _deduplicate_heartbeat_ticks(raw_trigger_ticks, min_separation_ticks=20_000_000.0, period=268435456.0):
    raw_trigger_ticks = np.asarray(raw_trigger_ticks, dtype=np.float64)
    if raw_trigger_ticks.size == 0: return np.array([], dtype=np.float64)
    unwrapped_all = _unwrap_28bit_ticks(raw_trigger_ticks, period=period)
    groups = []
    current_group = [unwrapped_all[0]]
    for val in unwrapped_all[1:]:
        if (val - current_group[-1]) < min_separation_ticks:
            current_group.append(val)
        else:
            groups.append(current_group)
            current_group = [val]
    groups.append(current_group)
    unique_heartbeats = np.array([np.median(g) for g in groups], dtype=np.float64)
    return unique_heartbeats

def _build_heartbeat_daq_times(heartbeat_ticks, expected_ticks_per_second=40_000_000.0, max_fractional_interval_error=0.25):
    heartbeat_ticks = np.asarray(heartbeat_ticks, dtype=np.float64)
    if heartbeat_ticks.size < 2:
        raise ValueError("Need at least two unique DAQ heartbeat records to synchronize.")
    intervals = np.diff(heartbeat_ticks)
    if np.any(intervals <= 0):
        raise ValueError("Heartbeat ticks are not strictly increasing.")
    steps = np.rint(intervals / expected_ticks_per_second).astype(np.int64)
    if np.any(steps < 1):
        raise ValueError("Invalid heartbeat interval.")
    residual_ticks = intervals - steps.astype(np.float64) * expected_ticks_per_second
    frac_error = np.abs(residual_ticks) / expected_ticks_per_second
    bad = np.nonzero(frac_error > max_fractional_interval_error)[0]
    if bad.size > 0:
        raise ValueError(f"Heartbeat interval not consistent. Bad index {bad[0]}.")
    heartbeat_daq_times = np.zeros(heartbeat_ticks.size, dtype=np.float64)
    heartbeat_daq_times[1:] = np.cumsum(steps).astype(np.float64)
    return heartbeat_daq_times, intervals, steps, residual_ticks

def load_advacam_t3p(
    filepath,
    *,
    strict_two_chip=True,
    max_unmapped_fraction=1e-6,
    expected_ticks_per_second=40_000_000.0,
    heartbeat_duplicate_separation_ticks=20_000_000.0,
    drop_unanchored_photons=True,
    return_diagnostics=False
):
    print(f"Loading Timepix3 data from {filepath}...")

    # ------------------------------------------------------------------
    # 1. FAST MEMORY-MAPPED REGEX PARSING
    # ------------------------------------------------------------------
    # We match ANY 6-column text injection (including shutter states like 11/14/15)
    # so we can cleanly strip them from the binary stream.
    hw_trigger_pattern = re.compile(rb'\d+\t\d+\t\d+\t\d+\t\d+\t\d+\r?\n')
    
    raw_trigger_ticks = []
    binary_chunks = []
    
    with open(filepath, "rb") as f:
        # Check if empty
        f.seek(0, 2)
        if f.tell() == 0:
            raise ValueError("File is empty.")
        f.seek(0)
        
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        last_idx = 0
        
        # Isolate text injections and save the pure binary data
        for match in hw_trigger_pattern.finditer(mm):
            start, end = match.span()
            
            # Slice out the pure binary chunk that occurred before this text
            chunk = mm[last_idx:start]
            
            # Ensure the chunk perfectly aligns to 16 bytes. If a buffer glitch 
            # interrupted a packet right before the text, this drops the broken bytes.
            valid_len = (len(chunk) // 16) * 16
            if valid_len > 0:
                binary_chunks.append(chunk[:valid_len])
                
            # Parse the text trigger
            text_str = match.group(0).decode('ascii', errors='ignore').strip()
            parts = text_str.split('\t')
            
            if len(parts) == 6:
                trigger_id = int(parts[5])
                
                # ONLY use overflow=10 (Heartbeat Sync) for time synchronization
                if trigger_id == 10:
                    try:
                        # Mask to 28 bits as per your original synchronization math
                        heartbeat_toa = int(parts[2]) & 0x0FFFFFFF
                        raw_trigger_ticks.append(float(heartbeat_toa))
                    except ValueError:
                        pass
                        
            last_idx = end
            
        # Append the final binary chunk after the last text injection
        chunk = mm[last_idx:]
        valid_len = (len(chunk) // 16) * 16
        if valid_len > 0:
            binary_chunks.append(chunk[:valid_len])
            
        mm.close()

    # Glue all the perfectly aligned 16-byte chunks together
    binary_data = b"".join(binary_chunks)

    # ------------------------------------------------------------------
    # 2. Parse binary photon records (High-Speed NumPy Engine)
    # ------------------------------------------------------------------
    dt = np.dtype([
        ("matrixIdx", "<u4"),
        ("toa", "<u8"),
        ("overflow", "u1"),
        ("ftoa", "u1"),
        ("tot", "<u2"),
    ])

    data = np.frombuffer(binary_data, dtype=dt)

    if len(data) == 0:
        raise ValueError("No binary photon records were found.")

    # ------------------------------------------------------------------
    # 3. Strict two-chip geometry validation
    # ------------------------------------------------------------------
    n_raw = len(data)
    chip_ids_all = data["matrixIdx"] // 65536

    print("\n--- RAW CHIP DIAGNOSTIC ---")
    for cid in range(4):
        c = np.count_nonzero(chip_ids_all == cid)
        print(f"Chip {cid}: {c:,} photon records")

    if strict_two_chip:
        valid_mask = data["matrixIdx"] < 131072
    else:
        valid_mask = data["matrixIdx"] < 262144

    n_keep = int(np.count_nonzero(valid_mask))
    n_drop = n_raw - n_keep
    drop_frac = n_drop / max(n_raw, 1)

    print(f"Dropped unmapped-chip records: {n_drop:,} / {n_raw:,} = {100.0 * drop_frac:.6f}%")
    print("---------------------------\n")

    if strict_two_chip and drop_frac > max_unmapped_fraction:
        raise ValueError(
            f"Detected {n_drop:,} impossible matrixIdx records ({100.0 * drop_frac:.6f}%). "
            "This suggests byte corruption or a geometry mismatch."
        )

    clean_data = data[valid_mask]
    if len(clean_data) == 0:
        raise ValueError("No valid photon records remain after filtering.")

    matrixIdx = clean_data["matrixIdx"].astype(np.uint32)
    ftoa = clean_data["ftoa"].astype(np.float64)
    tot = clean_data["tot"].astype(np.float64)
    raw_toa = (clean_data["toa"] & 0x0FFFFFFF).astype(np.float64)

    # ------------------------------------------------------------------
    # 4. Unwrap photon ToA per chip
    # ------------------------------------------------------------------
    unwrapped_toa = np.zeros_like(raw_toa, dtype=np.float64)
    chip_ids = matrixIdx // 65536

    print("\n--- PHOTON ToA UNWRAP DIAGNOSTIC ---")
    for cid in np.unique(chip_ids):
        mask = (chip_ids == cid)
        c_toa = raw_toa[mask]
        if c_toa.size == 0: continue
        deltas = np.diff(c_toa)
        non_monotonic = int(np.count_nonzero(deltas < 0))
        rollover_like = int(np.count_nonzero(deltas < -134217728.0))
        print(f"Chip {int(cid)}: {non_monotonic:,} negative ToA steps, {rollover_like:,} rollover-like steps.")
        unwrapped_toa[mask] = _unwrap_28bit_ticks(c_toa)
    print("------------------------------------\n")

    photon_tick_fine = unwrapped_toa - (ftoa / 16.0)

    # ------------------------------------------------------------------
    # 5. Build DAQ heartbeat anchors
    # ------------------------------------------------------------------
    if len(raw_trigger_ticks) < 2:
        raise ValueError("Fewer than two ASCII DAQ heartbeat packets found.")

    heartbeat_ticks = _deduplicate_heartbeat_ticks(
        raw_trigger_ticks,
        min_separation_ticks=heartbeat_duplicate_separation_ticks
    )

    if heartbeat_ticks.size < 2:
        raise ValueError("Fewer than two unique DAQ heartbeats remain.")

    heartbeat_daq_times, hb_intervals, hb_steps, hb_residual_ticks = _build_heartbeat_daq_times(
        heartbeat_ticks,
        expected_ticks_per_second=expected_ticks_per_second
    )

    print("\n--- DAQ HEARTBEAT DIAGNOSTIC ---")
    print(f"Raw ASCII heartbeat packets: {len(raw_trigger_ticks):,}")
    print(f"Unique DAQ heartbeats:       {len(heartbeat_ticks):,}")
    print(f"First heartbeat tick:        {heartbeat_ticks[0]:.3f}")
    print(f"Last heartbeat tick:         {heartbeat_ticks[-1]:.3f}")
    print(f"DAQ time span:               {heartbeat_daq_times[0]:.6f} s to {heartbeat_daq_times[-1]:.6f} s")
    print("--------------------------------\n")

    # ------------------------------------------------------------------
    # 6. Piecewise-linear DAQ-clock synchronization
    # ------------------------------------------------------------------
    hb_indices = np.searchsorted(heartbeat_ticks, photon_tick_fine, side="right") - 1
    anchored = (hb_indices >= 0) & (hb_indices < heartbeat_ticks.size - 1)
    n_unanchored = int(np.count_nonzero(~anchored))

    if n_unanchored > 0:
        frac_unanchored = n_unanchored / max(len(photon_tick_fine), 1)
        msg = f"{n_unanchored:,} photon records ({100.0 * frac_unanchored:.6f}%) are outside the heartbeat-covered time range."
        if drop_unanchored_photons:
            print("WARNING:", msg, "They will be dropped.")
        else:
            raise ValueError(msg)

    matrixIdx = matrixIdx[anchored]
    tot = tot[anchored]
    photon_tick_fine = photon_tick_fine[anchored]
    hb_indices = hb_indices[anchored]

    H0 = heartbeat_ticks[hb_indices]
    H1 = heartbeat_ticks[hb_indices + 1]
    T0 = heartbeat_daq_times[hb_indices]
    T1 = heartbeat_daq_times[hb_indices + 1]

    interval_ticks = H1 - H0
    interval_seconds = T1 - T0

    if np.any(interval_ticks <= 0):
        raise ValueError("Non-positive heartbeat interval encountered.")

    frac = (photon_tick_fine - H0) / interval_ticks
    bad_frac = (frac < -1e-9) | (frac > 1.0 + 1e-9)
    if np.any(bad_frac):
        raise ValueError("Internal synchronization error.")

    time_s = T0 + frac * interval_seconds

    # ------------------------------------------------------------------
    # 7. Sort output by DAQ/PX5 time
    # ------------------------------------------------------------------
    sort_idx = np.argsort(time_s)
    time_s = time_s[sort_idx]
    tot = tot[sort_idx]
    matrixIdx = matrixIdx[sort_idx]

    print("\n--- SYNCHRONIZED OUTPUT DIAGNOSTIC ---")
    print(f"Returned photon records: {len(time_s):,}")
    if len(time_s) > 0:
        print(f"First photon time:       {time_s[0]:.9f} s")
        print(f"Last photon time:        {time_s[-1]:.9f} s")
        print(f"Duration:                {time_s[-1] - time_s[0]:.9f} s")
    print("--------------------------------------\n")

    if return_diagnostics:
        diagnostics = {
            "unanchored_photon_records": n_unanchored,
        }
        return time_s, tot, matrixIdx, diagnostics

    return time_s, tot, matrixIdx

def load_calibration_matrices(xml_filepath):
    print(f"Extracting surrogate calibration matrices from {xml_filepath}...")
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    matrices = {}
    for param in ['caliba', 'calibb', 'calibc', 'calibt']:
        tags = root.findall(f'.//{param}')
        data_list = [np.frombuffer(base64.b64decode(tag.text.strip()), dtype=np.float64) for tag in tags]
        matrices[param] = np.concatenate(data_list)
    return matrices['caliba'], matrices['calibb'], matrices['calibc'], matrices['calibt']

# ==============================================================================
# PART 2: THE RE-ENGINEERED HARDWARE PIPELINES
# ==============================================================================
@njit
def map_matrix_idx(idx):
    chip_id = idx // 65536
    local_idx = idx % 65536
    cx = local_idx // 256
    cy = local_idx % 256
    valid = True
    if chip_id == 0:
        x = np.float64(cx); y = np.float64(255 - cy)
    elif chip_id == 1:
        x = np.float64(511 - cx); y = np.float64(cy)
    else:
        x = -1.0; y = -1.0; valid = False
    return x, y, chip_id, valid

@njit
def calc_energy_safe(tot, a, b, c, t):
    if a == 0.0 or tot <= 0.0: return np.nan
    B = b - (a * t) - tot
    C = (tot * t) - (b * t) - c
    D = B * B - 4.0 * a * C
    if D < 0.0: return np.nan
    
    sqrtD = np.sqrt(D)
    e1 = (-B + sqrtD) / (2.0 * a)
    e2 = (-B - sqrtD) / (2.0 * a)
    
    best = np.nan
    if np.isfinite(e1) and e1 > max(0.0, t): best = e1
    if np.isfinite(e2) and e2 > max(0.0, t):
        if np.isnan(best): best = e2
        else:
            pred1 = a * best + b - c / (best - t)
            pred2 = a * e2 + b - c / (e2 - t)
            if abs(pred2 - tot) < abs(pred1 - tot): best = e2
    return best

@njit
def cpu_fast_pipeline(time_array, tot_array_in, matrix_idx_array, a_mat, b_mat, c_mat, t_mat):
    n = len(tot_array_in)
    out_energy = np.full(n, np.nan, dtype=np.float64)
    out_x = np.zeros(n, dtype=np.float64)
    out_y = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        idx = matrix_idx_array[i]
        tot = tot_array_in[i]
        
        x, y, chip_id, valid_geom = map_matrix_idx(idx)
        if not valid_geom: continue
            
        out_energy[i] = calc_energy_safe(tot, a_mat[idx], b_mat[idx], c_mat[idx], t_mat[idx])
        out_x[i] = x; out_y[i] = y
            
    return out_x, out_y, out_energy, tot_array_in.copy(), time_array.copy(), np.ones(n, dtype=np.int32), np.zeros(n, dtype=np.int32), 0

@njit
def cpu_cluster_pipeline_fixed(time_array, tot_array_in, matrix_idx_array, a_mat, b_mat, c_mat, t_mat, time_window_s, max_span_s):
    n = len(tot_array_in)
    x_all = np.zeros(n, dtype=np.float64)
    y_all = np.zeros(n, dtype=np.float64)
    e_all = np.zeros(n, dtype=np.float64)
    chip_all = np.zeros(n, dtype=np.int32)
    valid_geom = np.zeros(n, dtype=np.bool_)
    valid_energy = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        idx = matrix_idx_array[i]
        tot = tot_array_in[i]
        x, y, chip_id, is_valid = map_matrix_idx(idx)
        x_all[i] = x; y_all[i] = y; chip_all[i] = chip_id
        
        if not is_valid or tot <= 0.0:
            valid_geom[i] = False; continue
            
        valid_geom[i] = True
        energy = calc_energy_safe(tot, a_mat[idx], b_mat[idx], c_mat[idx], t_mat[idx])
        if not np.isnan(energy):
            e_all[i] = energy
            valid_energy[i] = True

    out_energy = np.zeros(n, dtype=np.float64)
    out_x = np.zeros(n, dtype=np.float64)
    out_y = np.zeros(n, dtype=np.float64)
    out_tot = np.zeros(n, dtype=np.float64)
    out_time = np.zeros(n, dtype=np.float64)
    out_size = np.zeros(n, dtype=np.int32)
    out_bad_e = np.zeros(n, dtype=np.int32)

    assigned = np.zeros(n, dtype=np.bool_)
    queue = np.empty(n, dtype=np.int64) 
    cluster_count = 0
    span_rejected_count = 0

    for i in range(n):
        if not valid_geom[i] or assigned[i]: continue

        assigned[i] = True
        queue[0] = i
        q_head = 0; q_tail = 1

        cluster_e = 0.0; cluster_tot = 0.0
        cluster_min_t = time_array[i]; cluster_max_t = time_array[i]
        bad_energy_count = 0
        max_e = -1.0; max_x = x_all[i]; max_y = y_all[i]

        while q_head < q_tail:
            curr_idx = queue[q_head]
            q_head += 1

            t_curr = time_array[curr_idx]
            if valid_energy[curr_idx]:
                e_curr = e_all[curr_idx]
                cluster_e += e_curr
                if e_curr > max_e:
                    max_e = e_curr; max_x = x_all[curr_idx]; max_y = y_all[curr_idx]
            else:
                bad_energy_count += 1
                    
            cluster_tot += tot_array_in[curr_idx]

            # BACKWARD SCAN
            j = curr_idx - 1
            while j >= 0 and (t_curr - time_array[j]) <= time_window_s:
                if not assigned[j] and valid_geom[j]:
                    if chip_all[curr_idx] == chip_all[j]:
                        dx = abs(x_all[curr_idx] - x_all[j])
                        dy = abs(y_all[curr_idx] - y_all[j])
                        if dx <= 1.0 and dy <= 1.0 and not (dx == 0.0 and dy == 0.0): 
                            cand_min_t = min(cluster_min_t, time_array[j])
                            cand_max_t = max(cluster_max_t, time_array[j])
                            if (cand_max_t - cand_min_t) <= max_span_s:
                                assigned[j] = True
                                queue[q_tail] = j; q_tail += 1
                                cluster_min_t = cand_min_t; cluster_max_t = cand_max_t
                            else:
                                span_rejected_count += 1
                j -= 1

            # FORWARD SCAN
            j = curr_idx + 1
            while j < n and (time_array[j] - t_curr) <= time_window_s:
                if not assigned[j] and valid_geom[j]:
                    if chip_all[curr_idx] == chip_all[j]:
                        dx = abs(x_all[curr_idx] - x_all[j])
                        dy = abs(y_all[curr_idx] - y_all[j])
                        if dx <= 1.0 and dy <= 1.0 and not (dx == 0.0 and dy == 0.0):
                            cand_min_t = min(cluster_min_t, time_array[j])
                            cand_max_t = max(cluster_max_t, time_array[j])
                            if (cand_max_t - cand_min_t) <= max_span_s:
                                assigned[j] = True
                                queue[q_tail] = j; q_tail += 1
                                cluster_min_t = cand_min_t; cluster_max_t = cand_max_t
                            else:
                                span_rejected_count += 1
                j += 1 

        valid_energy_pixels = q_tail - bad_energy_count
        if valid_energy_pixels == 0: out_energy[cluster_count] = np.nan
        else: out_energy[cluster_count] = cluster_e
            
        out_tot[cluster_count] = cluster_tot
        out_time[cluster_count] = cluster_min_t
        out_x[cluster_count] = max_x; out_y[cluster_count] = max_y
        out_size[cluster_count] = q_tail
        out_bad_e[cluster_count] = bad_energy_count
        cluster_count += 1

    return out_x[:cluster_count], out_y[:cluster_count], out_energy[:cluster_count], out_tot[:cluster_count], out_time[:cluster_count], out_size[:cluster_count], out_bad_e[:cluster_count], span_rejected_count

# ==============================================================================
# PART 3: DASHBOARD & FILTERS
# ==============================================================================
@njit
def fast_roi_filter(x, y, t, tot, e, bad_e, x0, x1, y0, y1, t0, t1):
    n = len(x)
    count = 0
    for i in range(n):
        if x0 <= x[i] < x1 and y0 <= y[i] < y1 and t0 <= t[i] <= t1: count += 1
    t_out = np.empty(count, dtype=t.dtype)
    tot_out = np.empty(count, dtype=tot.dtype)
    e_out = np.empty(count, dtype=e.dtype)
    bad_e_out = np.empty(count, dtype=bad_e.dtype)
    idx = 0
    for i in range(n):
        if x0 <= x[i] < x1 and y0 <= y[i] < y1 and t0 <= t[i] <= t1:
            t_out[idx], tot_out[idx], e_out[idx], bad_e_out[idx] = t[i], tot[i], e[i], bad_e[i]
            idx += 1
    return t_out, tot_out, e_out, bad_e_out

def load_daq_h5(filepath):
    if not filepath or not os.path.exists(filepath): return np.array([]), np.array([])
    with h5py.File(filepath, 'r') as f:
        px5_counts, bin_s = f['px5CountsPerBin'][:], f.attrs['bin_s']
    px5Cum = np.concatenate(([0], np.cumsum(px5_counts, dtype=np.float64)))
    return np.arange(len(px5Cum)) * bin_s, px5Cum

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AnalysisApp(ctk.CTk):
    def __init__(self, raw_t, raw_tot, raw_matrix, a_m, b_m, c_m, t_m, daq_t_s, px5Cum):
        super().__init__()
        self.title("Master IFM Dashboard")
        self.geometry("1600x950")
        
        self.raw_t, self.raw_tot, self.raw_matrix = raw_t, raw_tot, raw_matrix
        self.calib_mats = (a_m, b_m, c_m, t_m)
        self.daq_t_s, self.px5Cum = daq_t_s, px5Cum
        self.has_h5 = len(self.daq_t_s) > 0
        
        self.x, self.y, self.energy, self.tot, self.t_s = None, None, None, None, None
        self.cluster_sizes, self.bad_e_counts = None, None
        self.cached_engine = None 
        
        self.max_x, self.max_y = 512, 256
        self.colors = ['red', 'blue', 'cyan', 'magenta', 'orange', 'purple', 'lime']
        
        self.setup_ui()

    def get_float(self, entry, default=0.0):
        try: return float(entry.get())
        except ValueError: return default
        
    def get_int(self, entry, default=0):
        try: return int(entry.get())
        except ValueError: return default

    def setup_ui(self):
        self.sidebar = ctk.CTkScrollableFrame(self, width=400)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        
        self.plot_area = ctk.CTkFrame(self)
        self.plot_area.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar, text="Port Boundaries", font=("Arial", 16, "bold")).pack(pady=(10, 5))
        self.roi_vars = []
        
        ports_grid_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        ports_grid_frame.pack(fill="x", pady=2)
        
        ports_grid_frame.grid_columnconfigure(0, weight=1)
        ports_grid_frame.grid_columnconfigure(1, weight=1)
        
        for i in range(6):
            row = i // 2
            col = i % 2
            
            frame = ctk.CTkFrame(ports_grid_frame)
            frame.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
            
            chk_var = ctk.BooleanVar(value=False)
            chk = ctk.CTkCheckBox(frame, text=f"Port {i+1}", variable=chk_var, text_color=self.colors[i], width=60)
            chk.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
            
            x_min = ctk.CTkEntry(frame, width=40); x_min.insert(0, str(10 + i*40))
            x_max = ctk.CTkEntry(frame, width=40); x_max.insert(0, str(30 + i*40))
            x_min.grid(row=0, column=1, padx=2); x_max.grid(row=0, column=2, padx=2)
            
            y_min = ctk.CTkEntry(frame, width=40); y_min.insert(0, "100")
            y_max = ctk.CTkEntry(frame, width=40); y_max.insert(0, "150")
            y_min.grid(row=1, column=1, padx=2); y_max.grid(row=1, column=2, padx=2)
            
            self.roi_vars.append({'chk': chk_var, 'is_whole': False, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})

        all_pixels_frame = ctk.CTkFrame(ports_grid_frame, fg_color="transparent")
        all_pixels_frame.grid(row=3, column=0, padx=2, pady=5, sticky="nsew")
        chk_var_all = ctk.BooleanVar(value=True)
        chk_all = ctk.CTkCheckBox(all_pixels_frame, text="All Pixels", variable=chk_var_all, text_color=self.colors[6], font=("Arial", 13, "bold"))
        chk_all.pack(side="left", padx=5, pady=5)
        self.roi_vars.append({'chk': chk_var_all, 'is_whole': True})

        px5_frame = ctk.CTkFrame(ports_grid_frame, fg_color="transparent")
        px5_frame.grid(row=3, column=1, padx=2, pady=5, sticky="nsew")
        self.show_px5_var = ctk.BooleanVar(value=True) 
        self.chk_px5 = ctk.CTkCheckBox(px5_frame, text="Show PX5", variable=self.show_px5_var, text_color='green', font=("Arial", 13, "bold"))
        self.chk_px5.pack(side="left", padx=5, pady=5)

        ctk.CTkLabel(self.sidebar, text="Global Settings", font=("Arial", 16, "bold")).pack(pady=(20, 5))
        e_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        e_frame.pack(fill="x")
        ctk.CTkLabel(e_frame, text="Energy (keV):").pack(side="left")
        self.e_min = ctk.CTkEntry(e_frame, width=60); self.e_min.insert(0, "5.0")
        self.e_max = ctk.CTkEntry(e_frame, width=60); self.e_max.insert(0, "150.0")
        self.e_max.pack(side="right", padx=5); self.e_min.pack(side="right", padx=5)

        max_t = np.ceil(np.max(self.raw_t)) if len(self.raw_t) > 0 else 1000.0
        init_t = min(300.0, max_t)
        t_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        t_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(t_frame, text="Time Zoom (s):").pack(side="left")
        self.t_min = ctk.CTkEntry(t_frame, width=60); self.t_min.insert(0, "0.0")
        self.t_max = ctk.CTkEntry(t_frame, width=60); self.t_max.insert(0, str(init_t))
        self.t_max.pack(side="right", padx=5); self.t_min.pack(side="right", padx=5)

        bin_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bin_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(bin_frame, text="Bin Width (ms):").pack(side="left")
        self.bin_ms = ctk.CTkEntry(bin_frame, width=60); self.bin_ms.insert(0, "100.0")
        self.bin_ms.pack(side="right", padx=5)

        ctk.CTkLabel(self.sidebar, text="Hardware Engine", font=("Arial", 16, "bold")).pack(pady=(20, 5))
        self.engine_drop = ctk.CTkOptionMenu(
            self.sidebar, 
            values=['1. No Charge Sharing', '2. Charge Sharing']
        )
        self.engine_drop.pack(fill="x", pady=5)

        cs_frame1 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        cs_frame1.pack(fill="x", pady=2)
        ctk.CTkLabel(cs_frame1, text="CS Window (ns):").pack(side="left")
        self.cs_window = ctk.CTkEntry(cs_frame1, width=80); self.cs_window.insert(0, "50.0")
        self.cs_window.pack(side="right")
        
        cs_frame2 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        cs_frame2.pack(fill="x", pady=2)
        ctk.CTkLabel(cs_frame2, text="Max CS Span (ns):").pack(side="left")
        self.cs_max_span = ctk.CTkEntry(cs_frame2, width=80); self.cs_max_span.insert(0, "100.0")
        self.cs_max_span.pack(side="right")

        ctk.CTkLabel(self.sidebar, text="Analysis Mode:", anchor="w").pack(fill="x", padx=20, pady=(5, 0))
        self.mode_drop = ctk.CTkOptionMenu(self.sidebar, values=['Bomb Out', 'Bomb In'])
        if not self.has_h5: self.mode_drop.set('Bomb Out')
        self.mode_drop.pack(fill="x", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(self.sidebar, text="Dark Port (O-Beam):", anchor="w").pack(fill="x", padx=20, pady=(0, 0))
        self.dark_port_drop = ctk.CTkOptionMenu(self.sidebar, values=[f'Port {i+1}' for i in range(6)])
        self.dark_port_drop.set('Port 3') 
        self.dark_port_drop.pack(fill="x", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(self.sidebar, text="Reference Port:", anchor="w").pack(fill="x", padx=20, pady=(0, 0))
        self.ref_port_drop = ctk.CTkOptionMenu(self.sidebar, values=[f'Port {i+1}' for i in range(6)])
        self.ref_port_drop.set('Port 6')
        self.ref_port_drop.pack(fill="x", padx=20, pady=(0, 10))
        
        offset_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        offset_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(offset_frame, text="PX5 Offset (s):").pack(side="left")
        self.px5_offset = ctk.CTkEntry(offset_frame, width=80)
        self.px5_offset.insert(0, "0.0")
        self.px5_offset.pack(side="right")
        
        err_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        err_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(err_frame, text="Dark Err (%):").pack(side="left")
        self.base_err = ctk.CTkEntry(err_frame, width=80); self.base_err.insert(0, "0.0")
        self.base_err.pack(side="right")

        self.btn_update = ctk.CTkButton(self.sidebar, text="UPDATE DASHBOARD", fg_color="green", hover_color="darkgreen", height=40, command=self.update_plots)
        self.btn_update.pack(fill="x", pady=(20, 5))
        
        # --- NEW BUTTON: Integrated Sync Test ---
        self.btn_sync = ctk.CTkButton(self.sidebar, text="RUN SYNC TEST (CO-57)", fg_color="#9B59B6", hover_color="#8E44AD", height=40, command=self.run_sync_test)
        self.btn_sync.pack(fill="x", pady=(0, 20))

        self.results_txt = ctk.CTkTextbox(self.sidebar, height=250, font=("Consolas", 14), state="disabled")
        self.results_txt.pack(fill="x", pady=10)

        self.fig = plt.figure(figsize=(12, 8))
        self.gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])
        self.ax1 = self.fig.add_subplot(self.gs[0, 0]); self.ax2 = self.fig.add_subplot(self.gs[0, 1]) 
        self.ax3 = self.fig.add_subplot(self.gs[1, 0]); self.ax4 = self.fig.add_subplot(self.gs[1, 1]) 
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==============================================================================
    # INTEGRATED CROSS-CORRELATION ENGINE
    # ==============================================================================
    # ==============================================================================
    # INTEGRATED CROSS-CORRELATION ENGINE (PATCHED)
    # ==============================================================================
    def run_sync_test(self):
        if not self.has_h5:
            self.results_txt.configure(state="normal")
            self.results_txt.insert("end", "\n[ERROR] No PX5 data loaded.\n")
            self.results_txt.configure(state="disabled")
            return

        if self.t_s is None: self.update_plots()
        if len(self.daq_t_s) < 2: return
        
        # 1. Obey the UI Time Zoom and UI Bin Size!
        t_start = self.get_float(self.t_min)
        t_end = self.get_float(self.t_max)
        bin_s = self.get_float(self.bin_ms) / 1000.0

        if bin_s <= 0 or t_end <= t_start: return

        # 2. Filter AdvaPIX Energy
        e_min, e_max = self.get_float(self.e_min), self.get_float(self.e_max)
        selected_engine = self.engine_drop.get()
        
        has_valid_e = ~np.isnan(self.energy)
        good_energy_cluster = (self.bad_e_counts == 0)
        in_energy_range = has_valid_e & (self.energy >= e_min) & (self.energy <= e_max)
        
        if "2." in selected_engine: counting_mask = in_energy_range & good_energy_cluster
        else: counting_mask = in_energy_range

        # Apply BOTH Energy AND Time constraints
        time_mask = (self.t_s >= t_start) & (self.t_s <= t_end)
        filtered_t_s = self.t_s[counting_mask & time_mask]

        if len(filtered_t_s) == 0:
            self.results_txt.configure(state="normal")
            self.results_txt.insert("end", f"\n[ERROR] No AdvaPIX hits in {e_min}-{e_max} keV between {t_start}-{t_end}s.\n")
            self.results_txt.configure(state="disabled")
            return

        self.btn_sync.configure(text="CALCULATING...", state="disabled")
        self.update()

        # 3. Bin AdvaPIX data using the UI bin size
        common_bin_edges = np.arange(t_start, t_end + bin_s, bin_s)
        adva_bins, _ = np.histogram(filtered_t_s, bins=common_bin_edges)
        
        # 4. Interpolate PX5 cumulative data into the exact same UI bins
        interp_px5 = np.interp(common_bin_edges, self.daq_t_s, self.px5Cum)
        px5_bins_trunc = np.diff(interp_px5)

        # 5. Normalize
        adva_norm = adva_bins - np.mean(adva_bins)
        px5_norm = px5_bins_trunc - np.mean(px5_bins_trunc)

        # 6. Cross-Correlate
        correlation = correlate(adva_norm, px5_norm, mode='full')
        lags = correlation_lags(len(adva_norm), len(px5_norm), mode='full')
        lag_times_s = lags * bin_s

        max_idx = np.argmax(correlation)
        best_lag_s = lag_times_s[max_idx]

        self.btn_sync.configure(text="RUN SYNC TEST (CO-57)", state="normal")
        
        res = f"\n--- SYNC CALIBRATION ---\nTime Window: {t_start} - {t_end} s\nBin Size: {bin_s*1e6:.1f} µs\nFound Math Peak: {best_lag_s*1000:.3f} ms\n"
        self.results_txt.configure(state="normal")
        self.results_txt.insert("end", res)
        self.results_txt.see("end")
        self.results_txt.configure(state="disabled")

        # 7. Clean TopLevel Plot (No ghost UI figures)
        top = ctk.CTkToplevel(self)
        top.title(f"Coincidence Check ({e_min}-{e_max} keV)")
        top.geometry("800x500")
        
        fig = plt.Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(lag_times_s * 1000, correlation, color='#9467bd', linewidth=1.5)
        ax.axvline(best_lag_s * 1000, color='red', linestyle='--', linewidth=2, label=f'Math Peak = {best_lag_s * 1000:.3f} ms')
        
        ax.set_title("Co-57 Coincidence Time Resolution", fontweight='bold')
        ax.set_xlabel(r"Time Difference ($\Delta t$) [ms]")
        ax.set_ylabel("Correlation Coefficient")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==============================================================================
    # EXISTING UPDATE PIPELINE
    # ==============================================================================
    def update_plots(self):
        self.btn_update.configure(state="disabled", text="CALCULATING...")
        self.update()
        
        selected_engine = self.engine_drop.get()
        window_ns = self.get_float(self.cs_window, 50.0)
        span_ns = self.get_float(self.cs_max_span, 100.0)
        if window_ns <= 0: window_ns = 50.0
        if span_ns < window_ns: span_ns = window_ns
        self.cs_window.delete(0, "end"); self.cs_window.insert(0, str(window_ns))
        self.cs_max_span.delete(0, "end"); self.cs_max_span.insert(0, str(span_ns))
        
        if "1." in selected_engine: state_key = selected_engine
        else: state_key = f"{selected_engine}_{window_ns}_{span_ns}"
        
        if self.cached_engine != state_key:
            self.btn_update.configure(text="PROCESSING RAW BINARY...")
            self.update()
            
            a_m, b_m, c_m, t_m = self.calib_mats
            
            if "1." in selected_engine:
                self.x, self.y, self.energy, self.tot, self.t_s, self.cluster_sizes, self.bad_e_counts, self.span_rejected = cpu_fast_pipeline(
                    self.raw_t, self.raw_tot, self.raw_matrix, a_m, b_m, c_m, t_m
                )
            elif "2." in selected_engine:
                self.x, self.y, self.energy, self.tot, self.t_s, self.cluster_sizes, self.bad_e_counts, self.span_rejected = cpu_cluster_pipeline_fixed(
                    self.raw_t, self.raw_tot, self.raw_matrix, a_m, b_m, c_m, t_m, window_ns/1e9, span_ns/1e9
                )
            
            self.cached_engine = state_key 

        e_min, e_max = self.get_float(self.e_min), self.get_float(self.e_max)
        
        has_valid_e = ~np.isnan(self.energy)
        good_energy_cluster = (self.bad_e_counts == 0)
        in_energy_range = has_valid_e & (self.energy >= e_min) & (self.energy <= e_max)
        
        if "2." in selected_engine: counting_mask = in_energy_range & good_energy_cluster
        else: counting_mask = in_energy_range

        x_f, y_f, t_f = self.x[counting_mask], self.y[counting_mask], self.t_s[counting_mask]
        tot_f = self.tot[counting_mask]
        
        n_before = len(self.energy)
        n_after = np.count_nonzero(counting_mask)
        
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear(); self.ax4.clear()
        
        img, _, _ = np.histogram2d(x_f, y_f, bins=[self.max_x, self.max_y], range=[[0, self.max_x], [0, self.max_y]])
        self.ax1.imshow(np.log10(img.T + 1), origin='lower', cmap='viridis', extent=[0, self.max_x, 0, self.max_y])
        self.ax1.set_title(f"X-Ray Hit Map ({e_min}-{e_max} keV)", fontweight='bold')
        self.ax1.set_xlabel("X (Pixels)")
        self.ax1.set_ylabel("Y (Pixels)")
        
        bin_s = self.get_float(self.bin_ms) / 1000.0
        t_range = [self.get_float(self.t_min), self.get_float(self.t_max)]
        common_bins = np.arange(t_range[0], t_range[1] + bin_s, bin_s)
        counts_total = []
        bad_e_roi_counts = []

        for i in range(7):
            roi = self.roi_vars[i]
            
            if 'is_whole' in roi and roi['is_whole']:
                rx = [0, 512]
                ry = [0, 256]
                label_name = "AdvaPIX"
                draw_box = False
            else:
                rx = [self.get_int(roi['x_min']), self.get_int(roi['x_max'])]
                ry = [self.get_int(roi['y_min']), self.get_int(roi['y_max'])]
                label_name = f'P{i+1}'
                draw_box = True

            if roi['chk'].get() and draw_box: 
                self.ax1.add_patch(Rectangle((rx[0], ry[0]), rx[1]-rx[0], ry[1]-ry[0], linewidth=2, edgecolor=self.colors[i], facecolor='none'))
            
            t_fin, tot_fin, e_fin, bad_e_fin = fast_roi_filter(
                self.x, self.y, self.t_s, self.tot, self.energy, self.bad_e_counts, 
                rx[0], rx[1], ry[0], ry[1], t_range[0], t_range[1]
            )
            
            port_valid_e_mask = (~np.isnan(e_fin)) & (e_fin >= e_min) & (e_fin <= e_max)
            port_strict_mask = port_valid_e_mask & (bad_e_fin == 0) if "2." in selected_engine else port_valid_e_mask
            
            counts_total.append(np.count_nonzero(port_strict_mask))
            bad_e_roi_counts.append(np.count_nonzero((~port_strict_mask) & (bad_e_fin > 0)))

            if roi['chk'].get() and np.count_nonzero(port_strict_mask) > 0: 
                c, edges = np.histogram(t_fin[port_strict_mask], bins=common_bins)
                self.ax2.plot(edges[:-1], c, color=self.colors[i], label=label_name, linewidth=1.5, alpha=0.8)
                self.ax3.hist(tot_fin[port_strict_mask], bins=np.arange(max(0, int(np.min(tot_fin[port_strict_mask])) - 5), int(np.max(tot_fin[port_strict_mask])) + 7) - 0.5, color=self.colors[i], alpha=0.5, histtype='stepfilled')
                
                e_fin_valid = e_fin[port_strict_mask]
                e_fin_valid = e_fin_valid[~np.isnan(e_fin_valid)]
                if len(e_fin_valid) > 0:
                    self.ax4.hist(e_fin_valid, bins=150, range=(e_min, e_max), color=self.colors[i], alpha=0.5, histtype='stepfilled')
                    
        c_px5 = 0
        mode = self.mode_drop.get()
        if self.has_h5:
            offset_s = self.get_float(self.px5_offset)
            interp_px5 = np.interp(common_bins + offset_s, self.daq_t_s, self.px5Cum)
            c_px5 = np.interp(t_range[1] + offset_s, self.daq_t_s, self.px5Cum) - np.interp(t_range[0] + offset_s, self.daq_t_s, self.px5Cum)
            
            if self.show_px5_var.get() and mode == 'Bomb In':
                self.ax2.plot(common_bins[:-1], np.diff(interp_px5), color='green', label='PX5', linewidth=1.5)
            
        self.ax2.set_title("Timeline", fontweight='bold')
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Counts")
        self.ax2.set_xlim(t_range)
        self.ax2.legend()

        tot_title = "Pixel ToT Dist" if "1." in selected_engine else "Cluster Summed ToT Dist"
        self.ax3.set_title(tot_title, fontweight='bold')
        self.ax3.set_xlabel("Time-over-Threshold (Clock Ticks)")
        self.ax3.set_ylabel("Counts")

        self.ax4.set_title("Energy Spectrum", fontweight='bold')
        self.ax4.set_xlabel("Energy (keV)")
        self.ax4.set_ylabel("Counts")

        self.fig.tight_layout()
        self.canvas.draw()

        dark_idx = int(self.dark_port_drop.get().split(" ")[1]) - 1
        ref_idx = int(self.ref_port_drop.get().split(" ")[1]) - 1
        
        D = float(counts_total[dark_idx])
        R = float(counts_total[ref_idx])
        A = float(c_px5)
        
        single_hits = np.count_nonzero(self.cluster_sizes == 1)
        multi_hits = np.count_nonzero(self.cluster_sizes > 1)
        
        res_text = f"--- DIAGNOSTICS ---\n"
        res_text += f"Singlets: {single_hits:,} | Multiplets: {multi_hits:,}\n"
        if "2." in selected_engine: 
            res_text += f"Span Rejected (Pile-up): {self.span_rejected:,}\n"
        res_text += f"Energy Accepted: {n_after:,}/{n_before:,} ({100*n_after/max(n_before,1):.2f}%)\n"
        
        res_text += f"\n--- RAW COUNTS ---\n"
        for i in range(6): 
            res_text += f"P{i+1}: {counts_total[i]:.0f} (+sys: {bad_e_roi_counts[i]})\n"
        res_text += f"All Pixels: {counts_total[6]:.0f} (+sys: {bad_e_roi_counts[6]})\n"
        res_text += f"PX5: {A:.0f}\n\n--- PHYSICS ---\n"
        
        if mode == 'Bomb Out':
            if R > 0: 
                res_text += f"Dark Error (P{dark_idx+1}/P{ref_idx+1}): {(D/R)*100:.4f}%\n"
            else: 
                res_text += f"Error: Reference P{ref_idx+1} has 0 counts\n"
        else:
            e_dark = self.get_float(self.base_err) / 100.0
            if R > 0:
                p_det_raw = (D / R) - e_dark
                p_det_clip = max(0.0, p_det_raw)
                p_abs = A / R if self.has_h5 else 0.0
                
                var_p_det = (D / (R**2)) + ((D**2) / (R**3))
                sig_p_det = np.sqrt(var_p_det) if var_p_det > 0 else 0.0
                
                var_p_abs = (A / (R**2)) + ((A**2) / (R**3))
                sig_p_abs = np.sqrt(var_p_abs) if var_p_abs > 0 else 0.0
                
                eta_raw = (p_det_raw / (p_det_raw + p_abs)) * 100.0 if (p_det_raw + p_abs) > 0 else 0.0
                eta_clip = (p_det_clip / (p_det_clip + p_abs)) * 100.0 if (p_det_clip + p_abs) > 0 else 0.0
                
                if (p_det_raw + p_abs) > 0:
                    d_eta_d_pdet = p_abs / ((p_det_raw + p_abs)**2)
                    d_eta_d_pabs = -p_det_raw / ((p_det_raw + p_abs)**2)
                    var_eta = (d_eta_d_pdet**2 * var_p_det) + (d_eta_d_pabs**2 * var_p_abs)
                    sig_eta = np.sqrt(var_eta) * 100.0
                else:
                    sig_eta = 0.0

                res_text += f"P(det): {p_det_clip:.5f} ± {sig_p_det:.5f} (raw: {p_det_raw:.5f})\n"
                res_text += f"P(abs): {p_abs:.5f} ± {sig_p_abs:.5f}\n"
                res_text += f"IFM Efficiency:    {eta_clip:.2f}% ± {sig_eta:.2f}%\n"
            else: 
                res_text += f"Error: Reference P{ref_idx+1} has 0 counts\n"

        self.results_txt.configure(state="normal"); self.results_txt.delete("0.0", "end"); self.results_txt.insert("0.0", res_text); self.results_txt.configure(state="disabled")
        self.btn_update.configure(state="normal", text="UPDATE DASHBOARD")

# ==============================================================================
# PART 4: EXECUTION & VALIDATION
# ==============================================================================
if __name__ == "__main__":
    t3p_file = r"G:\האחסון שלי\X-Ray-IFM\Test Files\Sync_test\sync_test_25.5_r2.t3p"
    h5_file = r"G:\האחסון שלי\X-Ray-IFM\Test Files\Sync_test\sync_test_25.5_px5_data_002.h5"
    xml_file = r"G:\האחסון שלי\ESRF IFM\AdvaPIX-D04-W0126-2 (1).xml" 

    t_sec, tot_hits, matrix_idx = load_advacam_t3p(t3p_file)

    print("\n--- RUNNING CALIBRATION VALIDATION ---")
    daq_t, px5_c = load_daq_h5(h5_file)
    a_m, b_m, c_m, t_m = load_calibration_matrices(xml_file)
    
    lengths = [len(a_m), len(b_m), len(c_m), len(t_m)]
    print(f"Calibration matrix lengths: a={lengths[0]:,}, b={lengths[1]:,}, c={lengths[2]:,}, t={lengths[3]:,}")
    if len(set(lengths)) != 1:
        raise ValueError(f"CRITICAL ERROR: Calibration arrays have mismatched lengths: {lengths}")
        
    if len(matrix_idx) > 0:
        max_valid_idx = matrix_idx.max()
        if max_valid_idx >= min(lengths):
            raise ValueError(f"CRITICAL ERROR: Detected valid matrixIdx {max_valid_idx}, but calibration matrix is only {min(lengths)} long!")
    print("Calibration Coverage: PASSED")
    print("----------------------------------\n")

    app = AnalysisApp(t_sec, tot_hits, matrix_idx, a_m, b_m, c_m, t_m, daq_t, px5_c)
    app.update_plots()
    app.mainloop()