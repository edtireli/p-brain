"""Interactive start/menu helpers.

This module contains both GUI helpers (Tkinter) and headless helpers used by
batch runs (e.g. PAR/REC conversion). Tkinter is optional so the pipeline can
run on headless machines.
"""

# A GUI that enumerates over the data directory's subfolders and shows them to the user so that they may continue with analysis of a single subfolder.
# The idea is the user has MRI data within the subfolders, and analyses them sequentially or individually, and not in bulk.
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None
import sys
import os
import gzip
import glob
import json
import re

import utils.settings as settings
from utils.loading import (
    inject_flip_angle_into_sidecar_json,
    inject_parrec_metadata_into_sidecar_json,
    inject_repetition_time_excitation_into_sidecar_json,
    inject_turbo_factor_into_sidecar_json,
    read_flip_angle_deg_from_par,
    read_protocol_name_from_par,
    read_repetition_time_excitation_s_from_par,
    read_turbo_factor_from_par,
    sanitize_dcm2niix_basename,
)

from termcolor import colored

def print_banner():
    """Display the p-brain banner."""
    os.system('clear')
    line = "=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-="
    print(colored("         Welcome to p-brain - a neuroimaging & analysis tool", "white"))
    print(colored(line, "white"))
    print("")
    print(colored("        / /                                                  / /", "cyan"))
    print(colored("       / /    eeeee      eeeee  eeeee  eeeee e  eeeee       / /", "cyan"))
    print(colored("      / /     8   8      8   8  8   8  8   8 8  8   8      / /", "cyan"))
    print(colored("eeee / /      8eee8 eeee 8eee8e 8eee8e 8eee8 8e 8e  8     / /    eeee", "cyan"))
    print(colored("    / /       88         88   8 88   8 88  8 88 88  8    / /", "cyan"))
    print(colored("   / /        88         88eee8 88   8 88  8 88 88  8   / /", "cyan"))
    print("")
    print(colored("                  Developed by Edis Devin Tireli", "white"))
    print(colored("                     University of Copenhagen", "white"))
    print(colored(line, "white"))


availability_toggled = False


def select_log_number(data_root=None):
    """GUI for selecting a dataset.

    ``data_root`` specifies the directory containing subject folders. When not
    provided, the ``P_BRAIN_DATA_DIR`` environment variable is honoured before
    falling back to a local ``Data`` directory.
    """

    global selected_log_number

    if data_root is None:
        data_root = os.environ.get("P_BRAIN_DATA_DIR")
        if data_root is None:
            data_root = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'Data')
    data_root = os.path.abspath(data_root)

    if tk is None or ttk is None:
        raise RuntimeError("tkinter is not available in this Python; run with --id/--mode or install a Python build with Tk support")

    root = tk.Tk()
    root.title('Select Log Number')
    root.geometry("200x450")

    frame = tk.Frame(root)
    frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

    current_path = data_root
    log_numbers = []

    log_numbers_listbox = tk.Listbox(frame, height=20, width=30)
    log_numbers_listbox.pack(side=tk.TOP, padx=20, pady=5)

    def ensure_controls_flag(path):
        flag_path = os.path.join(path, 'controls.json')
        if not os.path.exists(flag_path):
            with open(flag_path, 'w') as f:
                json.dump({"controls": True}, f, indent=4)

    def refresh_list(path):
        nonlocal current_path, log_numbers
        current_path = path
        if os.path.basename(current_path).lower() == 'controls':
            ensure_controls_flag(current_path)
        log_numbers = [f.name for f in os.scandir(current_path) if f.is_dir()]
        log_numbers.sort()
        log_numbers_listbox.delete(0, tk.END)
        for item in log_numbers:
            log_numbers_listbox.insert(tk.END, item)
        log_numbers_listbox.yview(tk.END)
        if current_path != data_root:
            back_button.pack(side=tk.TOP, pady=5, anchor=tk.CENTER)
        else:
            back_button.pack_forget()

    def on_select(event):
        global selected_log_number
        if not log_numbers_listbox.curselection():
            return
        selected_log_number = log_numbers_listbox.get(log_numbers_listbox.curselection())
        selected_log_number = selected_log_number.rstrip('*')

    def on_double_click(event):
        if not log_numbers_listbox.curselection():
            return
        item = log_numbers_listbox.get(log_numbers_listbox.curselection())
        next_path = os.path.join(current_path, item)
        if os.path.isdir(next_path) and current_path == data_root and item.lower() == 'controls':
            refresh_list(next_path)
        else:
            on_select(None)
            on_accept()

    def on_accept():
        global selected_log_number
        if selected_log_number:
            root.destroy()

    def toggle_availability():
        global availability_toggled
        availability_toggled = not availability_toggled
        current_scroll = log_numbers_listbox.yview()
        for i, log in enumerate(log_numbers):
            path_to_check = os.path.join(current_path, log, 'Analysis', 'values.json')
            nifti_file_path = os.path.join(current_path, log, 'NIfTI', 'WIPDelRec-hperf120long.nii')
            nifti_file_exists = os.path.exists(nifti_file_path)
            if availability_toggled:
                display_text = log
                if nifti_file_exists:
                    display_text += '*'
                if os.path.exists(path_to_check):
                    log_numbers_listbox.delete(i)
                    log_numbers_listbox.insert(i, display_text)
                    log_numbers_listbox.itemconfig(i, {'fg': 'green'})
                else:
                    log_numbers_listbox.delete(i)
                    log_numbers_listbox.insert(i, display_text)
                    log_numbers_listbox.itemconfig(i, {'fg': 'red'})
            else:
                log_numbers_listbox.delete(i)
                log_numbers_listbox.insert(i, log)
                log_numbers_listbox.itemconfig(i, {'fg': 'white'})

        log_numbers_listbox.yview_moveto(current_scroll[0])

    def go_back():
        refresh_list(data_root)

    accept_button = ttk.Button(root, text="Accept", command=on_accept)
    availability_button = ttk.Button(root, text="Availability", command=toggle_availability)
    back_button = ttk.Button(root, text="Back", command=go_back)

    accept_button.pack(side=tk.TOP, pady=5, anchor=tk.CENTER)
    availability_button.pack(side=tk.TOP, pady=5, anchor=tk.CENTER)

    refresh_list(data_root)

    log_numbers_listbox.bind('<<ListboxSelect>>', on_select)
    log_numbers_listbox.bind('<Double-1>', on_double_click)

    selected_log_number = None
    root.mainloop()
    root.update()
    return selected_log_number


#A simple implementation of a CLI interface
import time
import os


def welcome_screen():
    print_banner()
    #print('----------------------------------------------------------------------')
    print(colored('=-=-= Choose between the following options =-=-==-=-==-=-==-=-==-=-=-=', 'white'))
    print('| '+colored('0', 'cyan')+' | View MRI images')
    print('| '+colored('1', 'cyan')+' | Compute M0 and T1 map from MRI data')
    print('| '+colored('2', 'cyan')+' | Generate Concentration Time Curves (CTC) based on ROI')
    print('| '+colored('3', 'cyan')+' | Generate Time-Shifted Concentration Curves (TSCC) from CTC')
    print('| '+colored('4', 'cyan')+' | Create Tissue (Grey/White matter) CTCs')
    print('| '+colored('5', 'cyan')+' | Compute BBB permeability and perfusion parameters')
    print('| '+colored('6', 'cyan')+' | Add analysis notes')
    print('| '+colored('7', 'cyan')+' | Addons')
    print('| '+colored('9', 'red')+' | ' +colored('Exit program', 'red'))
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))

def welcome_screen_pseudo():
    """Menu shown after running the automatic pipeline in pseudo mode."""
    print_banner()
    print(colored('=-=-= Choose between the following options =-=-==-=-==-=-==-=-=-=-=', 'white'))
    print('| '+colored('0', 'cyan')+' | View MRI images')
    print('| '+colored('1', 'cyan')+' | Compute M0 and T1 map from MRI data')
    print('| '+colored('2', 'cyan')+' | Change Input Function')
    print('| '+colored('3', 'cyan')+' | Generate Time-Shifted Concentration Curves (TSCC) from CTC')
    print('| '+colored('4', 'cyan')+' | Change Tissue (Grey/White matter) CTCs')
    print('| '+colored('5', 'cyan')+' | Compute BBB permeability and perfusion parameters')
    print('| '+colored('6', 'cyan')+' | Add analysis notes')
    print('| '+colored('7', 'cyan')+' | Addons')
    print('| '+colored('9', 'red')+' | ' +colored('Exit program', 'red'))
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    
def welcome_screen_choice():
    choice = input('['+colored('!', 'cyan')+'] Enter option ('+colored('1', 'cyan')+'-'+colored('7', 'cyan')+', or 9 to exit): ')
    
    if not choice.isdigit():
        print('[' +colored('!', 'red') +"] Only integer input!")
        time.sleep(2)
        print('[' +colored('!', 'cyan') +'] Try again!  ^-^')
        time.sleep(2)
        return welcome_screen_choice()  # Recursively call itself
    
    return int(choice) 


import subprocess
from collections import defaultdict
import os
import shutil
import hashlib
import time


def decompress_gz_in_directory(directory):
    """Decompress .gz files under a directory.

    WARNING: This can be extremely expensive on large cohorts.
    By default we skip decompressing `.nii.gz` because most tooling supports
    compressed NIfTI directly.
    """

    decompress_nii_gz = os.environ.get("PBRAIN_DECOMPRESS_NIFTI_GZ") == "1"
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.gz'):
                continue

            # Avoid expanding compressed NIfTI unless explicitly requested.
            if (not decompress_nii_gz) and file.endswith('.nii.gz'):
                continue

            gz_path = os.path.join(root, file)
            with gzip.open(gz_path, 'rb') as f_in:
                with open(gz_path[:-3], 'wb') as f_out:
                    f_out.write(f_in.read())


def parrec2nifti(directory, nifti_directory):
    """Convert Philips PAR/REC (or DICOM) to NIfTI using dcm2niix.

    Rules:
    - When PAR/REC pairs exist: convert per PAR only if matching NIfTI outputs
      are missing or older than the PAR ("outdated").
    - Always enrich JSON sidecar metadata from PAR when PAR is available.
    - When no PAR/REC exists: if NIfTI is missing, attempt recursive DICOM conversion.
    """

    directory = os.fspath(directory)
    nifti_directory = os.fspath(nifti_directory)
    if not directory or not os.path.isdir(directory):
        return
    if not nifti_directory:
        return

    os.makedirs(nifti_directory, exist_ok=True)

    retry_failed = (os.environ.get("P_BRAIN_DCM2NIIX_RETRY_FAILED", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    failed_dir = os.path.join(nifti_directory, ".pbrain_dcm2niix_failed")
    try:
        os.makedirs(failed_dir, exist_ok=True)
    except Exception:
        failed_dir = nifti_directory

    def _failure_marker_path(par_path: str) -> str:
        abs_path = os.path.abspath(os.fspath(par_path))
        h = hashlib.sha1(abs_path.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        stem = sanitize_dcm2niix_basename(stem) or "parrec"
        return os.path.join(failed_dir, f"{stem}_{h}.json")

    def _load_failure_marker(par_path: str) -> dict | None:
        p = _failure_marker_path(par_path)
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else None
        except Exception:
            return None

    def _write_failure_marker(par_path: str, *, stderr: str | None) -> None:
        p = _failure_marker_path(par_path)
        try:
            d = {
                "par_path": os.path.abspath(os.fspath(par_path)),
                "par_mtime": _mtime(par_path),
                "dcm2niix": dcm2niix_path,
                "stderr": (stderr or "")[-4000:],
                "ts": time.time(),
            }
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)
        except Exception:
            pass

    def _clear_failure_marker(par_path: str) -> None:
        p = _failure_marker_path(par_path)
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass

    def _ensure_common_tool_paths() -> None:
        """GUI apps on macOS often start with a minimal PATH.

        Homebrew installs to /opt/homebrew/bin on Apple Silicon.
        """

        try:
            current = os.environ.get("PATH") or ""
            parts = [p for p in current.split(":") if p]

            candidates = []
            for p in (
                "/opt/homebrew/bin",
                "/usr/local/bin",
                "/opt/local/bin",
            ):
                if os.path.isdir(p):
                    candidates.append(p)

            # Prepend missing candidates.
            new_parts = [p for p in candidates if p not in parts] + parts
            os.environ["PATH"] = ":".join(new_parts)
        except Exception:
            pass

    def _resolve_dcm2niix() -> str | None:
        # Explicit override
        override = (os.environ.get("P_BRAIN_DCM2NIIX") or "").strip()
        if override:
            if os.path.isfile(override) and os.access(override, os.X_OK):
                return override
            # Allow specifying just a name; fall through to which after PATH tweaks.

        _ensure_common_tool_paths()

        path = shutil.which(override) if override and os.sep not in override else None
        if path:
            return path

        path = shutil.which("dcm2niix")
        if path:
            return path

        # Last-resort common absolute locations
        for p in (
            "/opt/homebrew/bin/dcm2niix",
            "/usr/local/bin/dcm2niix",
            "/opt/local/bin/dcm2niix",
        ):
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        return None

    dcm2niix_path = _resolve_dcm2niix()

    def _nifti_path_from_json(json_path: str) -> str:
        base = json_path[:-5] if json_path.endswith('.json') else os.path.splitext(json_path)[0]
        gz = base + '.nii.gz'
        nii = base + '.nii'
        if os.path.exists(gz):
            return gz
        return nii

    def _find_parrec_pairs_one_level(root_dir: str):
        """Return list of full paths to .par files that have a matching .rec in the same directory.

        Scans root_dir and one level deep to match p-brain-web subject layouts.
        """

        candidates = [root_dir]
        try:
            for entry in os.scandir(root_dir):
                if entry.is_dir() and not entry.name.startswith('.'):
                    candidates.append(entry.path)
        except Exception:
            pass

        par_paths = []
        for base in candidates:
            try:
                files = [e.name for e in os.scandir(base) if e.is_file() and not e.name.startswith('.')]
            except Exception:
                continue

            pars = {}
            recs = set()
            for name in files:
                low = name.lower()
                stem, ext = os.path.splitext(low)
                if ext == '.par':
                    pars[stem] = name
                elif ext == '.rec':
                    recs.add(stem)

            for stem, par_name in pars.items():
                if stem in recs:
                    par_paths.append(os.path.join(base, par_name))

        return par_paths

    def _compact_name(name: str | None) -> str:
        if not name:
            return ""
        return re.sub(r"[^0-9A-Za-z]+", "", str(name)).lower()

    def _candidate_jsons_for_par(par_path: str) -> list[str]:
        protocol_name = read_protocol_name_from_par(par_path, default=None)
        protocol_sanitized = sanitize_dcm2niix_basename(protocol_name) if protocol_name else None
        protocol_compact = _compact_name(protocol_name)

        candidate_jsons: list[str] = []
        try:
            for fname in os.listdir(nifti_directory):
                if not fname.lower().endswith('.json'):
                    continue
                if fname.startswith('.'):
                    continue

                # Most robust: match using metadata inside the JSON itself.
                try:
                    json_path = os.path.join(nifti_directory, fname)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    if isinstance(meta, dict) and protocol_compact:
                        proto_c = _compact_name(meta.get('ProtocolName'))
                        series_c = _compact_name(meta.get('SeriesDescription'))
                        if proto_c == protocol_compact or series_c == protocol_compact:
                            candidate_jsons.append(json_path)
                            continue
                except Exception:
                    pass

                if protocol_sanitized:
                    base = os.path.splitext(fname)[0]
                    base_sanitized = sanitize_dcm2niix_basename(base)
                    if base_sanitized.lower() == protocol_sanitized.lower():
                        candidate_jsons.append(os.path.join(nifti_directory, fname))
                        continue

                # More robust match: compare compacted (alphanumeric-only) forms.
                if protocol_compact:
                    base = os.path.splitext(fname)[0]
                    base_compact = _compact_name(base)
                    if base_compact == protocol_compact:
                        candidate_jsons.append(os.path.join(nifti_directory, fname))

            if not candidate_jsons and protocol_sanitized:
                # Fallback: substring match.
                for fname in os.listdir(nifti_directory):
                    if not fname.lower().endswith('.json'):
                        continue
                    if fname.startswith('.'):
                        continue
                    base = os.path.splitext(fname)[0]
                    if protocol_sanitized.lower() in sanitize_dcm2niix_basename(base).lower():
                        candidate_jsons.append(os.path.join(nifti_directory, fname))
                        continue
                    if protocol_compact:
                        base_compact = _compact_name(base)
                        if protocol_compact in base_compact or base_compact in protocol_compact:
                            candidate_jsons.append(os.path.join(nifti_directory, fname))
        except Exception:
            return []

        return candidate_jsons

    def _mtime(path: str) -> float | None:
        try:
            return float(os.path.getmtime(path))
        except Exception:
            return None

    def _par_outputs_outdated(par_path: str, json_paths: list[str]) -> bool:
        """Return True when NIfTI/sidecar are missing or older than the PAR."""

        par_mtime = _mtime(par_path)
        if par_mtime is None:
            return False
        if not json_paths:
            return True

        newest_out = None
        for json_path in json_paths:
            nifti_path = _nifti_path_from_json(json_path)
            json_m = _mtime(json_path)
            nii_m = _mtime(nifti_path)
            if json_m is None or nii_m is None:
                return True
            out_m = max(json_m, nii_m)
            newest_out = out_m if newest_out is None else max(newest_out, out_m)

        # Coarse FS/zip extraction can smear mtimes; tolerate 1s.
        return newest_out is None or (par_mtime - newest_out) > 1.0

    def _enrich_from_par(par_path: str, json_paths: list[str]) -> None:
        flip_angle_deg = settings.FLIP_ANGLE_DEG
        if flip_angle_deg is None:
            flip_angle_deg = read_flip_angle_deg_from_par(par_path, default=None)
        tr_exc_s = read_repetition_time_excitation_s_from_par(par_path, default=None)
        turbo_factor = read_turbo_factor_from_par(par_path, default=None)

        for json_path in json_paths:
            nifti_path = _nifti_path_from_json(json_path)
            if flip_angle_deg is not None:
                inject_flip_angle_into_sidecar_json(nifti_path, flip_angle_deg)
            if tr_exc_s is not None:
                inject_repetition_time_excitation_into_sidecar_json(nifti_path, tr_exc_s)
            if turbo_factor is not None:
                inject_turbo_factor_into_sidecar_json(nifti_path, turbo_factor)
            inject_parrec_metadata_into_sidecar_json(nifti_path, par_path)

    def _run_dcm2niix(input_path: str) -> bool:
        nonlocal dcm2niix_path
        if dcm2niix_path is None:
            # Try again in case PATH was updated after process start.
            dcm2niix_path = _resolve_dcm2niix()

        if dcm2niix_path is None:
            # When invoked from p-brain-web we prefer to be non-fatal here; the
            # pipeline will later error clearly if DCE/T1/etc are missing.
            print(
                "[parrec2nifti] dcm2niix not found (PATH may be minimal when launched from the app); "
                "set P_BRAIN_DCM2NIIX=/full/path/to/dcm2niix or install into /opt/homebrew/bin or /usr/local/bin."
            )
            return False
        # -b y: emit JSON sidecars (metadata)
        # -r y: recurse into subfolders (common for DICOM exports)
        try:
            res = subprocess.run(
                [
                    dcm2niix_path,
                    "-b",
                    "y",
                    "-r",
                    "y",
                    "-f",
                    "%p",
                    "-o",
                    nifti_directory,
                    "-v",
                    "n",
                    input_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if res.returncode != 0:
                print(f"[parrec2nifti] dcm2niix failed for '{input_path}': {res.stderr}")
                # Persist failure so subsequent stages don't keep retrying the
                # same broken PAR/REC pair (common in p-brain-web stage runners).
                if os.path.splitext(str(input_path).lower())[1] == ".par":
                    _write_failure_marker(str(input_path), stderr=res.stderr)
                return False
            if os.path.splitext(str(input_path).lower())[1] == ".par":
                _clear_failure_marker(str(input_path))
            return True
        except Exception as e:
            print(f"[parrec2nifti] Exception running dcm2niix for '{input_path}': {e}")
            return False

    def _has_nifti(out_dir: str) -> bool:
        try:
            return any(
                f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")
                for f in os.listdir(out_dir)
            )
        except Exception:
            return False

    par_file_paths = _find_parrec_pairs_one_level(directory)

    if par_file_paths:
        # Convert/enrich each PAR independently.
        for par_path in par_file_paths:
            candidate_jsons = _candidate_jsons_for_par(par_path)
            needs_reconvert = _par_outputs_outdated(par_path, candidate_jsons)
            if needs_reconvert:
                marker = _load_failure_marker(par_path)
                par_mtime = _mtime(par_path)
                if (not retry_failed) and marker and marker.get("par_mtime") == par_mtime:
                    # Avoid spamming logs + wasting time if the PAR/REC is
                    # consistently unconvertible (e.g. truncated REC or unknown type).
                    pass
                else:
                    ok = _run_dcm2niix(par_path)
                    if ok:
                        print(f"Converted {os.path.basename(par_path)} successfully.")
                # Refresh after conversion (files may be overwritten).
                candidate_jsons = _candidate_jsons_for_par(par_path)

            # Always enrich sidecars from PAR when possible.
            if candidate_jsons:
                _enrich_from_par(par_path, candidate_jsons)
        return
    else:
        # No .PAR files. If no NIfTI exists yet, try DICOM conversion.
        if not _has_nifti(nifti_directory):
            # Heuristic: if there are DICOM-ish files in the subject root, run there;
            # otherwise run on the subject root anyway with recursive enabled.
            _run_dcm2niix(directory)

        # Avoid decompressing by default (can be hours on large cohorts).
        # Opt-in via: PBRAIN_DECOMPRESS_NIFTI_GZ=1
        if os.environ.get("PBRAIN_DECOMPRESS_NIFTI_GZ") == "1":
            decompress_gz_in_directory(nifti_directory)

