import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os
import re
import mmap
import base64
import xml.etree.ElementTree as ET
from numba import njit


# ============================================================
# USER SETTINGS
# ============================================================

ENERGY_MIN_KEV = 10.0
ENERGY_MAX_KEV = 27.0

CS_WINDOW_NS = 50.0
CS_MAX_SPAN_NS = 100.0

# Fixed ROI for the analysis
ROI_X1 = 90
ROI_X2 = 128
ROI_Y1 = 200
ROI_Y2 = 233

# Time bin for kymograph
# 0.001 = 1 ms
KYMOGRAPH_BIN_S = 0.001

MAX_X = 512
MAX_Y = 256


# ============================================================
# T3P PARSER
# ============================================================

def _unwrap_28bit_ticks(raw_ticks, period=268435456.0, last_raw=None, current_cycles=0):
    raw_ticks = np.asarray(raw_ticks, dtype=np.float64)

    if raw_ticks.size == 0:
        return raw_ticks.copy(), last_raw, current_cycles

    if last_raw is not None:
        deltas = np.diff(np.insert(raw_ticks, 0, last_raw))
    else:
        deltas = np.diff(raw_ticks)

    half_period = period / 2.0

    roll_forward = (deltas < -half_period).astype(np.int64)
    roll_backward = (deltas > half_period).astype(np.int64)

    cycles = np.zeros(raw_ticks.size, dtype=np.int64)

    if last_raw is not None:
        cycles = np.cumsum(roll_forward - roll_backward) + current_cycles
    else:
        cycles[1:] = np.cumsum(roll_forward - roll_backward)
        cycles += current_cycles

    unwrapped = raw_ticks + cycles * period

    return unwrapped, raw_ticks[-1], cycles[-1]


def load_advacam_t3p(filepath, state=None, strict_two_chip=True, expected_ticks_per_second=40_000_000.0):
    print(f"Loading T3P: {os.path.basename(filepath)}")

    if state is None:
        state = {
            "chips": {},
            "hb": {"last_raw": None, "cycles": 0},
            "last_hb_tick": None,
            "last_daq_time": 0.0,
            "last_interval_ticks": expected_ticks_per_second,
        }

    hw_trigger_pattern = re.compile(rb"\d+\t\d+\t\d+\t\d+\t\d+\t\d+\r?\n")

    raw_trigger_ticks = []
    valid_chunks = []

    dt = np.dtype([
        ("matrixIdx", "<u4"),
        ("toa", "<u8"),
        ("overflow", "u1"),
        ("ftoa", "u1"),
        ("tot", "<u2"),
    ])

    max_idx_val = 131072 if strict_two_chip else 262144

    with open(filepath, "rb") as f:
        f.seek(0, 2)

        if f.tell() == 0:
            return np.array([]), np.array([]), np.array([]), state

        f.seek(0)
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        last_idx = 0

        for match in hw_trigger_pattern.finditer(mm):
            start, end = match.span()

            valid_len = ((start - last_idx) // 16) * 16

            if valid_len > 0:
                raw_chunk = np.ndarray(
                    shape=(valid_len // 16,),
                    dtype=dt,
                    buffer=mm,
                    offset=last_idx,
                )

                good = raw_chunk[raw_chunk["matrixIdx"] < max_idx_val].copy()
                valid_chunks.append(good)

            parts = match.group(0).decode("ascii", errors="ignore").strip().split("\t")

            if len(parts) == 6 and int(parts[5]) == 10:
                try:
                    raw_trigger_ticks.append(float(int(parts[2]) & 0x0FFFFFFF))
                except ValueError:
                    pass

            last_idx = end

        valid_len = ((len(mm) - last_idx) // 16) * 16

        if valid_len > 0:
            raw_chunk = np.ndarray(
                shape=(valid_len // 16,),
                dtype=dt,
                buffer=mm,
                offset=last_idx,
            )

            good = raw_chunk[raw_chunk["matrixIdx"] < max_idx_val].copy()
            valid_chunks.append(good)

        mm.close()

    if len(valid_chunks) == 0:
        return np.array([]), np.array([]), np.array([]), state

    clean_data = np.concatenate(valid_chunks)

    if len(clean_data) == 0:
        return np.array([]), np.array([]), np.array([]), state

    matrix_idx = clean_data["matrixIdx"].astype(np.uint32)
    ftoa = clean_data["ftoa"].astype(np.float64)
    tot = clean_data["tot"].astype(np.float64)

    raw_toa = (clean_data["toa"] & 0x0FFFFFFF).astype(np.float64)

    unwrapped_toa = np.zeros_like(raw_toa, dtype=np.float64)

    chip_ids = matrix_idx // 65536

    for cid in np.unique(chip_ids):
        mask = chip_ids == cid
        c_toa = raw_toa[mask]

        c_state = state["chips"].get(cid, {"last_raw": None, "cycles": 0})

        u_toa, last_raw, cycles = _unwrap_28bit_ticks(
            c_toa,
            last_raw=c_state["last_raw"],
            current_cycles=c_state["cycles"],
        )

        unwrapped_toa[mask] = u_toa
        state["chips"][cid] = {"last_raw": last_raw, "cycles": cycles}

    photon_tick_fine = unwrapped_toa - (ftoa / 16.0)

    # Heartbeats
    if len(raw_trigger_ticks) > 0:
        hb_unwrapped, hb_last_raw, hb_cycles = _unwrap_28bit_ticks(
            raw_trigger_ticks,
            last_raw=state["hb"]["last_raw"],
            current_cycles=state["hb"]["cycles"],
        )

        state["hb"] = {"last_raw": hb_last_raw, "cycles": hb_cycles}

        groups = []
        current_group = [hb_unwrapped[0]]

        for val in hb_unwrapped[1:]:
            if val - current_group[-1] < 20_000_000.0:
                current_group.append(val)
            else:
                groups.append(current_group)
                current_group = [val]

        groups.append(current_group)

        heartbeat_ticks = np.array([np.median(g) for g in groups], dtype=np.float64)
    else:
        heartbeat_ticks = np.array([], dtype=np.float64)

    current_anchors_ticks = []
    current_anchors_times = []

    if heartbeat_ticks.size > 0:
        if state["last_hb_tick"] is not None:
            prev_tick = state["last_hb_tick"]
            prev_time = state["last_daq_time"]
        else:
            prev_tick = heartbeat_ticks[0]
            prev_time = 0.0

            current_anchors_ticks.append(prev_tick)
            current_anchors_times.append(prev_time)

            heartbeat_ticks = heartbeat_ticks[1:]

        for hb in heartbeat_ticks:
            interval = hb - prev_tick
            step = max(1, int(np.rint(interval / expected_ticks_per_second)))
            new_time = prev_time + float(step)

            current_anchors_ticks.append(hb)
            current_anchors_times.append(new_time)

            state["last_interval_ticks"] = interval / step
            prev_tick = hb
            prev_time = new_time

        state["last_hb_tick"] = prev_tick
        state["last_daq_time"] = prev_time

    valid_anchor_ticks = (
        [state["last_hb_tick"]] if state["last_hb_tick"] is not None else []
    ) + current_anchors_ticks

    valid_anchor_times = (
        [state["last_daq_time"]] if state["last_hb_tick"] is not None else []
    ) + current_anchors_times

    valid_anchor_ticks = np.array(valid_anchor_ticks, dtype=np.float64)
    valid_anchor_times = np.array(valid_anchor_times, dtype=np.float64)

    if valid_anchor_ticks.size >= 2:
        hb_indices = np.searchsorted(valid_anchor_ticks, photon_tick_fine, side="right") - 1
        hb_indices = np.clip(hb_indices, 0, valid_anchor_ticks.size - 2)

        H0 = valid_anchor_ticks[hb_indices]
        H1 = valid_anchor_ticks[hb_indices + 1]
        T0 = valid_anchor_times[hb_indices]
        T1 = valid_anchor_times[hb_indices + 1]

        time_s = T0 + ((photon_tick_fine - H0) / (H1 - H0)) * (T1 - T0)

    elif valid_anchor_ticks.size == 1:
        H0 = valid_anchor_ticks[0]
        T0 = valid_anchor_times[0]

        time_s = T0 + (photon_tick_fine - H0) / state["last_interval_ticks"]

    else:
        time_s = photon_tick_fine / expected_ticks_per_second

    sort_idx = np.argsort(time_s)

    return time_s[sort_idx], tot[sort_idx], matrix_idx[sort_idx], state


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration_matrices(xml_filepath):
    print(f"Loading calibration XML: {xml_filepath}")

    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    matrices = {}

    for param in ["caliba", "calibb", "calibc", "calibt"]:
        tags = root.findall(f".//{param}")

        data_list = [
            np.frombuffer(base64.b64decode(tag.text.strip()), dtype=np.float64)
            for tag in tags
        ]

        matrices[param] = np.concatenate(data_list)

    return (
        matrices["caliba"],
        matrices["calibb"],
        matrices["calibc"],
        matrices["calibt"],
    )


# ============================================================
# GEOMETRY AND ENERGY
# ============================================================

@njit
def map_matrix_idx(idx):
    chip_id = idx // 65536
    local_idx = idx % 65536

    cx = local_idx // 256
    cy = local_idx % 256

    valid = True

    if chip_id == 0:
        x = np.float64(cx)
        y = np.float64(255 - cy)

    elif chip_id == 1:
        x = np.float64(511 - cx)
        y = np.float64(cy)

    else:
        x = -1.0
        y = -1.0
        valid = False

    return x, y, chip_id, valid


@njit
def calc_energy_safe(tot, a, b, c, t):
    if a == 0.0 or tot <= 0.0:
        return np.nan

    B = b - (a * t) - tot
    C = (tot * t) - (b * t) - c
    D = B * B - 4.0 * a * C

    if D < 0.0:
        return np.nan

    sqrtD = np.sqrt(D)

    e1 = (-B + sqrtD) / (2.0 * a)
    e2 = (-B - sqrtD) / (2.0 * a)

    best = np.nan

    if np.isfinite(e1) and e1 > max(0.0, t):
        best = e1

    if np.isfinite(e2) and e2 > max(0.0, t):
        if np.isnan(best):
            best = e2
        else:
            pred1 = a * best + b - c / (best - t)
            pred2 = a * e2 + b - c / (e2 - t)

            if abs(pred2 - tot) < abs(pred1 - tot):
                best = e2

    return best


# ============================================================
# CHARGE SHARING PIPELINE
# ============================================================

@njit
def cpu_cluster_pipeline_fixed(
    time_array,
    tot_array_in,
    matrix_idx_array,
    a_mat,
    b_mat,
    c_mat,
    t_mat,
    time_window_s,
    max_span_s,
):
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

        x_all[i] = x
        y_all[i] = y
        chip_all[i] = chip_id

        if not is_valid or tot <= 0.0:
            valid_geom[i] = False
            continue

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
        if not valid_geom[i] or assigned[i]:
            continue

        assigned[i] = True
        queue[0] = i

        q_head = 0
        q_tail = 1

        cluster_e = 0.0
        cluster_tot = 0.0

        cluster_min_t = time_array[i]
        cluster_max_t = time_array[i]

        bad_energy_count = 0

        max_e = -1.0
        max_x = x_all[i]
        max_y = y_all[i]

        while q_head < q_tail:
            curr_idx = queue[q_head]
            q_head += 1

            t_curr = time_array[curr_idx]

            if valid_energy[curr_idx]:
                e_curr = e_all[curr_idx]
                cluster_e += e_curr

                if e_curr > max_e:
                    max_e = e_curr
                    max_x = x_all[curr_idx]
                    max_y = y_all[curr_idx]
            else:
                bad_energy_count += 1

            cluster_tot += tot_array_in[curr_idx]

            j = curr_idx - 1

            while j >= 0 and (t_curr - time_array[j]) <= time_window_s:
                if (
                    not assigned[j]
                    and valid_geom[j]
                    and chip_all[curr_idx] == chip_all[j]
                ):
                    dx = abs(x_all[curr_idx] - x_all[j])
                    dy = abs(y_all[curr_idx] - y_all[j])

                    if dx <= 1.0 and dy <= 1.0 and not (dx == 0.0 and dy == 0.0):
                        cand_min_t = min(cluster_min_t, time_array[j])
                        cand_max_t = max(cluster_max_t, time_array[j])

                        if (cand_max_t - cand_min_t) <= max_span_s:
                            assigned[j] = True
                            queue[q_tail] = j
                            q_tail += 1
                            cluster_min_t = cand_min_t
                            cluster_max_t = cand_max_t
                        else:
                            span_rejected_count += 1

                j -= 1

            j = curr_idx + 1

            while j < n and (time_array[j] - t_curr) <= time_window_s:
                if (
                    not assigned[j]
                    and valid_geom[j]
                    and chip_all[curr_idx] == chip_all[j]
                ):
                    dx = abs(x_all[curr_idx] - x_all[j])
                    dy = abs(y_all[curr_idx] - y_all[j])

                    if dx <= 1.0 and dy <= 1.0 and not (dx == 0.0 and dy == 0.0):
                        cand_min_t = min(cluster_min_t, time_array[j])
                        cand_max_t = max(cluster_max_t, time_array[j])

                        if (cand_max_t - cand_min_t) <= max_span_s:
                            assigned[j] = True
                            queue[q_tail] = j
                            q_tail += 1
                            cluster_min_t = cand_min_t
                            cluster_max_t = cand_max_t
                        else:
                            span_rejected_count += 1

                j += 1

        valid_energy_pixels = q_tail - bad_energy_count

        if valid_energy_pixels == 0:
            out_energy[cluster_count] = np.nan
        else:
            out_energy[cluster_count] = cluster_e

        out_tot[cluster_count] = cluster_tot
        out_time[cluster_count] = cluster_min_t
        out_x[cluster_count] = max_x
        out_y[cluster_count] = max_y
        out_size[cluster_count] = q_tail
        out_bad_e[cluster_count] = bad_energy_count

        cluster_count += 1

    return (
        out_x[:cluster_count],
        out_y[:cluster_count],
        out_energy[:cluster_count],
        out_tot[:cluster_count],
        out_time[:cluster_count],
        out_size[:cluster_count],
        out_bad_e[:cluster_count],
        span_rejected_count,
    )


# ============================================================
# FILE SELECTION
# ============================================================

def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


def select_inputs():
    root = Tk()
    root.withdraw()

    folder = filedialog.askdirectory(title="Select folder containing T3P files")

    if not folder:
        raise RuntimeError("No T3P folder selected.")

    xml_file = filedialog.askopenfilename(
        title="Select calibration XML",
        filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
    )

    if not xml_file:
        raise RuntimeError("No calibration XML selected.")

    root.destroy()

    all_files = os.listdir(folder)

    t3p_files = sorted(
        [
            os.path.join(folder, f)
            for f in all_files
            if f.lower().endswith(".t3p")
        ],
        key=natural_sort_key,
    )

    if len(t3p_files) == 0:
        raise RuntimeError("No T3P files found.")

    print(f"Found {len(t3p_files)} T3P files.")

    return folder, t3p_files, xml_file


# ============================================================
# ANALYSIS AFTER FILTERING
# ============================================================

def build_filtered_image(x, y):
    img, _, _ = np.histogram2d(
        x,
        y,
        bins=[MAX_X, MAX_Y],
        range=[[0, MAX_X], [0, MAX_Y]],
    )

    # Return as image[y, x]
    return img.T


def build_roi_kymograph(x, y, t, bin_s):
    roi_mask = (
        (x >= ROI_X1)
        & (x <= ROI_X2)
        & (y >= ROI_Y1)
        & (y <= ROI_Y2)
    )

    x_roi = x[roi_mask]
    t_roi = t[roi_mask]

    if len(x_roi) == 0:
        raise RuntimeError("No filtered events inside the fixed ROI.")

    t0 = np.min(t)
    t1 = np.max(t)

    time_edges = np.arange(t0, t1 + bin_s, bin_s)

    x_edges = np.arange(ROI_X1, ROI_X2 + 2, 1)

    kymo, _, _ = np.histogram2d(
        t_roi,
        x_roi,
        bins=[time_edges, x_edges],
    )

    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

    counts_roi = np.sum(kymo, axis=1)

    return kymo, time_centers, counts_roi


def plot_and_save_results(
    output_dir,
    hit_map,
    kymo_raw,
    time_centers,
    counts_roi,
    x_f,
    y_f,
    e_f,
    t_f,
    tot_f,
):
    os.makedirs(output_dir, exist_ok=True)

    # Normalizations
    kymo_norm_roi = kymo_raw / np.maximum(np.sum(kymo_raw, axis=1, keepdims=True), np.finfo(float).eps)

    # Full filtered flux per time bin
    time_edges = np.arange(time_centers[0] - KYMOGRAPH_BIN_S / 2, time_centers[-1] + KYMOGRAPH_BIN_S, KYMOGRAPH_BIN_S)
    flux_full, _ = np.histogram(t_f, bins=time_edges)

    kymo_norm_global = kymo_raw / np.maximum(flux_full[:, None], np.finfo(float).eps)

    kymo_minus_mean = kymo_norm_roi - np.mean(kymo_norm_roi, axis=0, keepdims=True)

    avg_raw = np.mean(kymo_raw, axis=0)
    avg_norm_roi = np.mean(kymo_norm_roi, axis=0)
    avg_norm_global = np.mean(kymo_norm_global, axis=0)
    avg_minus_mean = np.mean(kymo_minus_mean, axis=0)

    # Save images/data
    np.save(os.path.join(output_dir, "filtered_hit_map_chargeSharing_10_27keV.npy"), hit_map)
    np.savetxt(os.path.join(output_dir, "filtered_hit_map_chargeSharing_10_27keV.csv"), hit_map, delimiter=",")

    np.save(os.path.join(output_dir, "kymo_raw.npy"), kymo_raw)
    np.save(os.path.join(output_dir, "kymo_fluxNorm_by_ROI.npy"), kymo_norm_roi)
    np.save(os.path.join(output_dir, "kymo_fluxNorm_by_global.npy"), kymo_norm_global)
    np.save(os.path.join(output_dir, "kymo_minus_mean.npy"), kymo_minus_mean)

    np.savetxt(os.path.join(output_dir, "kymo_raw.csv"), kymo_raw, delimiter=",")
    np.savetxt(os.path.join(output_dir, "kymo_fluxNorm_by_ROI.csv"), kymo_norm_roi, delimiter=",")
    np.savetxt(os.path.join(output_dir, "kymo_fluxNorm_by_global.csv"), kymo_norm_global, delimiter=",")
    np.savetxt(os.path.join(output_dir, "kymo_minus_mean.csv"), kymo_minus_mean, delimiter=",")

    events = np.column_stack((x_f, y_f, e_f, t_f, tot_f))
    np.savetxt(
        os.path.join(output_dir, "filtered_events_chargeSharing_10_27keV.csv"),
        events,
        delimiter=",",
        header="x,y,energy_keV,time_s,tot",
        comments="",
    )

    avg_profiles = np.column_stack((
        np.arange(ROI_X1, ROI_X2 + 1),
        avg_raw,
        avg_norm_roi,
        avg_norm_global,
        avg_minus_mean,
    ))

    np.savetxt(
        os.path.join(output_dir, "time_averaged_profiles.csv"),
        avg_profiles,
        delimiter=",",
        header="x_pixel,avg_raw,avg_fluxNorm_ROI,avg_fluxNorm_global,avg_minus_mean",
        comments="",
    )

    # Plot filtered hit map
    plt.figure(figsize=(9, 4.8))
    plt.imshow(np.log10(hit_map + 1), origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="log10(counts + 1)")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.title("Charge-sharing corrected hit map, 10–27 keV")

    rect = plt.Rectangle(
        (ROI_X1, ROI_Y1),
        ROI_X2 - ROI_X1 + 1,
        ROI_Y2 - ROI_Y1 + 1,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
    )
    plt.gca().add_patch(rect)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "filtered_hit_map_log.png"), dpi=300)

    # Plot raw kymograph
    plt.figure(figsize=(8, 5))
    plt.imshow(
        kymo_raw,
        origin="lower",
        aspect="auto",
        extent=[ROI_X1, ROI_X2, time_centers[0], time_centers[-1]],
        cmap="gray",
    )
    plt.colorbar(label="Counts")
    plt.xlabel("X pixel inside ROI")
    plt.ylabel("Time [s]")
    plt.title("Raw kymograph after CS + 10–27 keV")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kymo_raw.png"), dpi=300)

    # Plot flux-normalized kymograph by ROI
    plt.figure(figsize=(8, 5))
    plt.imshow(
        kymo_norm_roi,
        origin="lower",
        aspect="auto",
        extent=[ROI_X1, ROI_X2, time_centers[0], time_centers[-1]],
        cmap="turbo",
    )
    plt.colorbar(label="ROI-normalized intensity")
    plt.xlabel("X pixel inside ROI")
    plt.ylabel("Time [s]")
    plt.title("Flux-normalized kymograph by ROI counts")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kymo_fluxNorm_by_ROI.png"), dpi=300)

    # Plot flux-normalized minus mean
    plt.figure(figsize=(8, 5))
    plt.imshow(
        kymo_minus_mean,
        origin="lower",
        aspect="auto",
        extent=[ROI_X1, ROI_X2, time_centers[0], time_centers[-1]],
        cmap="turbo",
    )
    plt.colorbar(label="Residual")
    plt.xlabel("X pixel inside ROI")
    plt.ylabel("Time [s]")
    plt.title("Flux-normalized kymograph minus mean profile")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kymo_minus_mean.png"), dpi=300)

    # Average profiles
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(ROI_X1, ROI_X2 + 1), avg_raw, linewidth=1.5)
    plt.xlabel("X pixel")
    plt.ylabel("Mean counts")
    plt.title("Time-averaged raw profile")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_profile_raw.png"), dpi=300)

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(ROI_X1, ROI_X2 + 1), avg_norm_roi, linewidth=1.5)
    plt.xlabel("X pixel")
    plt.ylabel("Mean normalized intensity")
    plt.title("Time-averaged flux-normalized profile")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_profile_fluxNorm_ROI.png"), dpi=300)

    # Counts vs time
    plt.figure(figsize=(8, 4))
    plt.plot(time_centers, counts_roi, linewidth=1.2)
    plt.xlabel("Time [s]")
    plt.ylabel("ROI counts per bin")
    plt.title("Filtered ROI counts vs time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "counts_ROI_vs_time.png"), dpi=300)

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    folder, t3p_files, xml_file = select_inputs()

    output_dir = os.path.join(folder, "CS_10_27keV_analysis_output")

    a_m, b_m, c_m, t_m = load_calibration_matrices(xml_file)

    all_t = []
    all_tot = []
    all_matrix = []

    parser_state = None

    for file in t3p_files:
        t, tot, matrix, parser_state = load_advacam_t3p(file, state=parser_state)

        if len(t) == 0:
            continue

        all_t.append(t)
        all_tot.append(tot)
        all_matrix.append(matrix)

    if len(all_t) == 0:
        raise RuntimeError("No events loaded from T3P files.")

    raw_t = np.concatenate(all_t)
    raw_tot = np.concatenate(all_tot)
    raw_matrix = np.concatenate(all_matrix)

    sort_idx = np.argsort(raw_t)
    raw_t = raw_t[sort_idx]
    raw_tot = raw_tot[sort_idx]
    raw_matrix = raw_matrix[sort_idx]

    print("\nRunning charge sharing...")
    print(f"Raw hits: {len(raw_t):,}")

    x, y, energy, tot, t_s, cluster_sizes, bad_e_counts, span_rejected = cpu_cluster_pipeline_fixed(
        raw_t,
        raw_tot,
        raw_matrix,
        a_m,
        b_m,
        c_m,
        t_m,
        CS_WINDOW_NS / 1e9,
        CS_MAX_SPAN_NS / 1e9,
    )

    print(f"Clusters after charge sharing: {len(x):,}")
    print(f"Span rejected: {span_rejected:,}")

    # Energy filter and strict good-cluster filter
    mask = (
        (~np.isnan(energy))
        & (energy >= ENERGY_MIN_KEV)
        & (energy <= ENERGY_MAX_KEV)
        & (bad_e_counts == 0)
    )

    x_f = x[mask]
    y_f = y[mask]
    e_f = energy[mask]
    t_f = t_s[mask]
    tot_f = tot[mask]

    print("\nAfter final filter:")
    print(f"Energy range: {ENERGY_MIN_KEV}-{ENERGY_MAX_KEV} keV")
    print(f"Accepted events: {len(x_f):,}")

    if len(x_f) == 0:
        raise RuntimeError("No events survived the CS + 10–27 keV filter.")

    hit_map = build_filtered_image(x_f, y_f)

    # ============================================================
    # PLOT FILTERED IMAGE
    # ============================================================

    plt.figure(figsize=(10, 5))

    plt.imshow(
        np.log10(hit_map + 1),
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=[0, MAX_X, 0, MAX_Y]
    )

    plt.colorbar(label="log10(counts + 1)")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")

    plt.title(
        f"Filtered hit map after Charge Sharing\n"
        f"Energy filter: {ENERGY_MIN_KEV:.0f}-{ENERGY_MAX_KEV:.0f} keV, "
        f"accepted events = {len(x_f):,}"
    )

    # Draw the ROI on top of the filtered image
    rect = plt.Rectangle(
        (ROI_X1, ROI_Y1),
        ROI_X2 - ROI_X1 + 1,
        ROI_Y2 - ROI_Y1 + 1,
        edgecolor="red",
        facecolor="none",
        linewidth=2
    )

    plt.gca().add_patch(rect)

    plt.tight_layout()
    plt.show()

    print(f"Filtered hit-map total counts: {np.sum(hit_map):.0f}")
    print(f"Filtered hit-map max pixel: {np.max(hit_map):.0f}")

    kymo_raw, time_centers, counts_roi = build_roi_kymograph(
        x_f,
        y_f,
        t_f,
        KYMOGRAPH_BIN_S,
    )

    print("\nROI analysis:")
    print(f"x = {ROI_X1}:{ROI_X2}")
    print(f"y = {ROI_Y1}:{ROI_Y2}")
    print(f"Kymograph shape: {kymo_raw.shape}")
    print(f"Total counts in ROI: {np.sum(kymo_raw):.0f}")

    plot_and_save_results(
        output_dir,
        hit_map,
        kymo_raw,
        time_centers,
        counts_roi,
        x_f,
        y_f,
        e_f,
        t_f,
        tot_f,
    )

    print("\nDone.")
    print(f"Saved all outputs to:\n{output_dir}")


if __name__ == "__main__":
    main()