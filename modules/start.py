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


# ---------------------------------------------------------------------------
#  Philips PRIDE V5 XML → PAR header converter
# ---------------------------------------------------------------------------

def _convert_xml_to_par(xml_path: str) -> str | None:
    """Convert a Philips PRIDE V5 XML header into a V4.2 PAR text header.

    The companion ``.REC`` binary is identical regardless of whether the
    header is ``.PAR`` or ``.xml``, so generating a ``.PAR`` allows the
    existing ``dcm2niix`` / ``nibabel`` pipelines to work unchanged.

    Returns the path of the generated ``.PAR`` file, or ``None`` on failure.
    Based on xml2par v0.2 by Volker Biberger (GPL-3).
    """
    import xml.etree.ElementTree as _ET

    try:
        tree = _ET.parse(xml_path)
        root_el = tree.getroot()
    except Exception as exc:
        print(f"[xml2par] Failed to parse '{xml_path}': {exc}")
        return None

    # Validate structure: <PRIDE_V5> → <Series_Info>, <Image_Array>
    if root_el.tag != "PRIDE_V5" or len(root_el) < 2:
        print(f"[xml2par] '{xml_path}' is not a PRIDE_V5 XML – skipping.")
        return None

    series_info = root_el[0]
    image_array = root_el[1]

    # ── helpers ────────────────────────────────────────────────────────
    def _si(name: str) -> str:
        """Retrieve a Series_Info attribute value by Name."""
        for attr in series_info:
            if attr.get("Name") == name:
                return (attr.text or "").strip()
        return ""

    def _ii_key(name: str, idx: int) -> str:
        """Retrieve an Image_Info Key sub-element."""
        try:
            for k in image_array[idx][0]:  # first child is <Key>
                if k.get("Name") == name:
                    return (k.text or "").strip()
        except Exception:
            pass
        return ""

    def _ii(name: str, idx: int) -> str:
        """Retrieve an Image_Info attribute (non-Key)."""
        try:
            for attr in image_array[idx]:
                if attr.get("Name") == name:
                    return (attr.text or "").strip()
        except Exception:
            pass
        return ""

    def _fmt(val: str, digits: int) -> str:
        """Format a numeric string to *digits* decimal places."""
        try:
            return f"{float(val):.{digits}f}"
        except (ValueError, TypeError):
            return "0" + ("." + "0" * digits if digits else "")

    def _pad(val: str, width: int) -> str:
        """Right-align *val* within *width* characters."""
        s = str(val)
        if len(s) >= width:
            return " " + s[:width - 1]
        return " " * (width - len(s)) + s

    # ── output path ────────────────────────────────────────────────────
    par_path = os.path.splitext(xml_path)[0] + ".PAR"
    if os.path.isfile(par_path):
        # Already converted (or a real .PAR exists).
        return par_path

    try:
        with open(par_path, "w", encoding="utf-8") as f:
            w = f.write

            # ── header boilerplate ─────────────────────────────────────
            w("# === DATA DESCRIPTION FILE ======================================================\n")
            w("#\n# CAUTION - Investigational device.\n# Limited by Federal Law to investigational use.\n#\n")
            w(f"# Dataset name: {os.path.splitext(xml_path)[0]}\n")
            w("#\n# CLINICAL TRYOUT             Research image export tool     V4.2\n")
            w("#\n# === GENERAL INFORMATION ========================================================\n#\n")

            # ── general info ───────────────────────────────────────────
            w(f".    Patient name                       :   {_si('Patient Name')}\n")
            w(f".    Examination name                   :   {_si('Examination Name')}\n")
            w(f".    Protocol name                      :   {_si('Protocol Name')}\n")
            w(f".    Examination date/time              :   {_si('Examination Date')} / {_si('Examination Time')}\n")
            w( ".    Series Type                        :   Image   MRSERIES\n")
            w(f".    Acquisition nr                     :   {_si('Aquisition Number')}\n")
            w(f".    Reconstruction nr                  :   {_si('Reconstruction Number')}\n")
            w(f".    Scan Duration [sec]                :   {_fmt(_si('Scan Duration'), 0)}\n")
            w(f".    Max. number of cardiac phases      :   {_si('Max No Phases')}\n")
            w(f".    Max. number of echoes              :   {_si('Max No Echoes')}\n")
            w(f".    Max. number of slices/locations    :   {_si('Max No Slices')}\n")
            w(f".    Max. number of dynamics            :   {_si('Max No Dynamics')}\n")
            w(f".    Max. number of mixes               :   {_si('Max No Mixes')}\n")

            # Patient position
            pos = _si("Patient Position")
            position = ""
            if "HF" in pos:
                position += "Head First"
            elif "FF" in pos:
                position += "Feet First"
            if "S" in pos:
                position += " Supine"
            elif "P" in pos:
                position += " Prone"
            elif "D" in pos:
                position += " Decubitus"
            if not position:
                position = pos
            w(f".    Patient position                   :   {position}\n")

            # Preparation direction
            prep = _si("Preparation Direction")
            if "AP" in prep:
                prep = "Anterior-Posterior"
            elif "RL" in prep:
                prep = "Right-Left"
            elif "FH" in prep:
                prep = "Foot-Head"
            w(f".    Preparation direction              :   {prep}\n")

            w(f".    Technique                          :   {_si('Technique')}\n")
            w(f".    Scan resolution  (x, y)            :   {_si('Scan Resolution X')}  {_si('Scan Resolution Y')}\n")
            w(f".    Scan mode                          :   {_si('Scan Mode')}\n")

            # Repetition time (may be array – take first value)
            rt = _si("Repetition Times").split()[0] if _si("Repetition Times") else "0"
            w(f".    Repetition time [ms]               :   {_fmt(rt, 3)}\n")

            w(f".    FOV (ap,fh,rl) [mm]                :   {_fmt(_si('FOV AP'), 3)}  {_fmt(_si('FOV FH'), 3)}  {_fmt(_si('FOV RL'), 3)}\n")
            w(f".    Water Fat shift [pixels]           :   {_fmt(_si('Water Fat Shift'), 3)}\n")
            w(f".    Angulation midslice(ap,fh,rl)[degr]:   {_fmt(_si('Angulation AP'), 3)}  {_fmt(_si('Angulation FH'), 3)}  {_fmt(_si('Angulation RL'), 3)}\n")
            w(f".    Off Centre midslice(ap,fh,rl) [mm] :   {_fmt(_si('Off Center AP'), 3)}  {_fmt(_si('Off Center FH'), 3)}  {_fmt(_si('Off Center RL'), 3)}\n")

            def _yn(tag: str) -> str:
                v = _si(tag)
                return "0" if v == "N" else "1"

            w(f".    Flow compensation <0=no 1=yes> ?   :   {_yn('Flow Compensation')}\n")
            w(f".    Presaturation     <0=no 1=yes> ?   :   {_yn('Presaturation')}\n")

            pev = _si("Phase Encoding Velocity").split()
            while len(pev) < 3:
                pev.append("0")
            w(f".    Phase encoding velocity [cm/sec]   :   {_fmt(pev[0], 6)}  {_fmt(pev[1], 6)}  {_fmt(pev[2], 6)}\n")

            w(f".    MTC               <0=no 1=yes> ?   :   {_yn('MTC')}\n")
            w(f".    SPIR              <0=no 1=yes> ?   :   {_yn('SPIR')}\n")
            w(f".    EPI factor        <0,1=no EPI>     :   {_si('EPI factor') or '1'}\n")
            w(f".    Dynamic scan      <0=no 1=yes> ?   :   {_yn('Dynamic Scan')}\n")
            w(f".    Diffusion         <0=no 1=yes> ?   :   {_yn('Diffusion')}\n")
            w(f".    Diffusion echo time [ms]           :   {_fmt(_si('Diffusion Echo Time'), 4)}\n")
            w(f".    Max. number of diffusion values    :   {_si('Max No B Values') or '1'}\n")
            w(f".    Max. number of gradient orients    :   {_si('Max No Gradient Orients') or '1'}\n")
            w(f".    Number of label types   <0=no ASL> :   {_si('No Label Types') or '0'}\n")

            # ── pixel value explanation ────────────────────────────────
            w("#\n# === PIXEL VALUES =============================================================\n")
            w("#  PV = pixel value in REC file, FP = floating point value, DV = displayed value on console\n")
            w("#  RS = rescale slope,           RI = rescale intercept,    SS = scale slope\n")
            w("#  DV = PV * RS + RI             FP = DV / (RS * SS)\n")
            w("#\n# === IMAGE INFORMATION DEFINITION =============================================\n")
            w("#  The rest of this file contains ONE line per image, this line contains the following information:\n#\n")
            for desc in [
                "#  slice number                             (integer)",
                "#  echo number                              (integer)",
                "#  dynamic scan number                      (integer)",
                "#  cardiac phase number                     (integer)",
                "#  image_type_mr                            (integer)",
                "#  scanning sequence                        (integer)",
                "#  index in REC file (in images)            (integer)",
                "#  image pixel size (in bits)               (integer)",
                "#  scan percentage                          (integer)",
                "#  recon resolution (x y)                   (2*integer)",
                "#  rescale intercept                        (float)",
                "#  rescale slope                            (float)",
                "#  scale slope                              (float)",
                "#  window center                            (integer)",
                "#  window width                             (integer)",
                "#  image angulation (ap,fh,rl in degrees )  (3*float)",
                "#  image offcentre (ap,fh,rl in mm )        (3*float)",
                "#  slice thickness (in mm )                 (float)",
                "#  slice gap (in mm )                       (float)",
                "#  image_display_orientation                (integer)",
                "#  slice orientation ( TRA/SAG/COR )        (integer)",
                "#  fmri_status_indication                   (integer)",
                "#  image_type_ed_es  (end diast/end syst)   (integer)",
                "#  pixel spacing (x,y) (in mm)              (2*float)",
                "#  echo_time                                (float)",
                "#  dyn_scan_begin_time                      (float)",
                "#  trigger_time                             (float)",
                "#  diffusion_b_factor                       (float)",
                "#  number of averages                       (integer)",
                "#  image_flip_angle (in degrees)            (float)",
                "#  cardiac frequency   (bpm)                (integer)",
                "#  minimum RR-interval (in ms)              (integer)",
                "#  maximum RR-interval (in ms)              (integer)",
                "#  TURBO factor  <0=no turbo>               (integer)",
                "#  Inversion delay (in ms)                  (float)",
                "#  diffusion b value number    (imagekey!)  (integer)",
                "#  gradient orientation number (imagekey!)  (integer)",
                "#  contrast type                            (string)",
                "#  diffusion anisotropy type                (string)",
                "#  diffusion (ap, fh, rl)                   (3*float)",
                "#  label type (ASL)            (imagekey!)  (integer)",
            ]:
                w(desc + "\n")

            w("#\n# === IMAGE INFORMATION ==========================================================\n")
            w("#  sl ec  dyn ph ty    idx pix scan% rec size                (re)scale              window        angulation              offcentre        thick   gap   info      spacing     echo     dtime   ttime    diff  avg  flip    freq   RR-int  turbo delay b grad cont anis         diffusion       L.ty\n")
            w("\n")

            # ── per-image lines ────────────────────────────────────────
            IMAGE_TYPES = [
                "M", "R", "I", "P", "CR", "T0", "T1", "T2", "RHO", "SPECTRO",
                "DERIVED", "ADC", "RCBV", "RCBF", "MTT", "TTP", "FA", "EADC",
                "B0", "DELAY", "MAXRELENH", "RELENH", "MAXENH", "WASHIN",
                "WASHOUT", "BREVENH", "AREACURV", "ANATOMIC", "T_TEST",
                "STD_DEVIATION", "PERFUSION", "T2_STAR", "R2", "R2_STAR",
                "W", "IP", "OP", "F", "SPARE1", "SPARE2",
            ]
            SEQUENCES = ["IR", "SE", "FFE", "DERIVED", "PCA", "UNSPECIFIED", "SPECTRO", "SI"]
            CONTRAST_TYPES = [
                "DIFFUSION", "FLOW_ENCODED", "FLUID_ATTENUATED", "PERFUSION",
                "PROTON_DENSITY", "STIR", "TAGGING", "T1", "T2", "T2_STAR",
                "TOF", "UNKNOWN", "MIXED",
            ]

            # Pre-pass: compute uniform slice gap.  In Philips V5 XML the
            # first slice typically reports gap=0 while subsequent slices
            # carry the actual inter-slice gap.  nibabel requires a single
            # uniform value, so we take the mode of all non-zero gaps (or
            # 0 if all are zero).
            _all_gaps: list[float] = []
            for _gi in range(len(image_array)):
                try:
                    _gv = float(_ii("Slice Gap", _gi))
                except (ValueError, TypeError):
                    _gv = 0.0
                _all_gaps.append(_gv)
            _nonzero_gaps = [g for g in _all_gaps if g != 0.0]
            _uniform_gap = max(set(_nonzero_gaps), key=_nonzero_gaps.count) if _nonzero_gaps else 0.0

            for idx in range(len(image_array)):
                try:
                    _ii_key("Slice", idx)
                except Exception:
                    break

                line = ""
                line += _pad(_ii_key("Slice", idx), 3)
                line += _pad(_ii_key("Echo", idx), 4)
                line += _pad(_ii_key("Dynamic", idx), 5)
                line += _pad(_ii_key("Phase", idx), 3)

                # image type
                itype = _ii_key("Type", idx)
                try:
                    itype = str(IMAGE_TYPES.index(itype))
                except ValueError:
                    itype = "0"
                line += " " + itype

                # sequence
                seq = _ii_key("Sequence", idx)
                try:
                    seq = str(SEQUENCES.index(seq))
                except ValueError:
                    seq = "5"
                line += _pad(seq, 2)

                line += _pad(_ii_key("Index", idx), 6)
                line += _pad(_ii("Pixel Size", idx), 4)
                line += _pad(_fmt(_ii("Scan Percentage", idx), 0), 6)
                line += _pad(_ii("Resolution X", idx), 5)
                line += _pad(_ii("Resolution Y", idx), 5)
                line += _pad(_fmt(_ii("Rescale Intercept", idx), 5), 12)
                line += _pad(_fmt(_ii("Rescale Slope", idx), 5), 10)

                # scale slope — keep scientific notation
                ss = _ii("Scale Slope", idx).replace("E", "e")
                if not ss:
                    ss = "0.00000e+00"
                line += _pad(ss, 13)

                line += _pad(_fmt(_ii("Window Center", idx), 0), 6)
                line += _pad(_fmt(_ii("Window Width", idx), 0), 6)
                line += _pad(_fmt(_ii("Angulation AP", idx), 2), 7)
                line += _pad(_fmt(_ii("Angulation FH", idx), 2), 7)
                line += _pad(_fmt(_ii("Angulation RL", idx), 2), 7)
                line += _pad(_fmt(_ii("Offcenter AP", idx), 2), 8)
                line += _pad(_fmt(_ii("Offcenter FH", idx), 2), 8)
                line += _pad(_fmt(_ii("Offcenter RL", idx), 2), 8)
                line += _pad(_fmt(_ii("Slice Thickness", idx), 3), 7)
                line += _pad(_fmt(str(_uniform_gap), 3), 7)

                # display orientation
                disp = _ii("Display Orientation", idx)
                disp_map = ["NONE", "RIGHT90", "RIGHT180", "LEFT90",
                            "VM", "RIGHT90VM", "RIGHT180VM", "LEFT90VM"]
                try:
                    disp = str(disp_map.index(disp))
                except ValueError:
                    disp = "0"
                line += _pad(disp, 2)

                # slice orientation
                sl_or = (_ii("Slice Orientation", idx) or "").lower()
                if "tra" in sl_or:
                    sl_or = "1"
                elif "sag" in sl_or:
                    sl_or = "2"
                elif "cor" in sl_or:
                    sl_or = "3"
                else:
                    sl_or = "0"
                line += _pad(sl_or, 2)

                line += _pad(_fmt(_ii("fMRI Status Indication", idx), 0), 2)

                # ed/es type
                edes = _ii("Image Type Ed Es", idx)
                if "ED" in (edes or ""):
                    edes = "0"
                elif "ES" in (edes or ""):
                    edes = "1"
                else:
                    edes = "2"
                line += _pad(edes, 2)

                # pixel spacing (two values)
                ps = (_ii("Pixel Spacing", idx) or "0 0").split()
                while len(ps) < 2:
                    ps.append("0")
                line += _pad(_fmt(ps[0], 3), 7)
                line += _pad(_fmt(ps[1], 3), 7)

                line += _pad(_fmt(_ii("Echo Time", idx), 2), 7)
                line += _pad(_fmt(_ii("Dyn Scan Begin Time", idx), 2), 8)
                line += _pad(_fmt(_ii("Trigger Time", idx), 2), 9)
                line += _pad(_fmt(_ii("Diffusion B Factor", idx), 2), 8)
                line += _pad(_fmt(_ii("No Averages", idx), 0), 4)
                line += _pad(_fmt(_ii("Image Flip Angle", idx), 2), 8)
                line += _pad(_fmt(_ii("Cardiac Frequency", idx), 0), 6)
                line += _pad(_fmt(_ii("Min RR Interval", idx), 0), 5)
                line += _pad(_fmt(_ii("Max RR Interval", idx), 0), 5)
                line += _pad(_fmt(_ii("TURBO Factor", idx), 0), 6)
                line += _pad(_fmt(_ii("Inversion Delay", idx), 1), 6)
                line += _pad(_fmt(_ii_key("BValue", idx), 0), 3)
                line += _pad(_fmt(_ii_key("Grad Orient", idx), 0), 4)

                # contrast type
                contr = _ii("Contrast Type", idx)
                try:
                    contr = str(CONTRAST_TYPES.index(contr))
                except ValueError:
                    contr = "11"
                line += _pad(contr, 5)

                # anisotropy
                anis = _ii("Diffusion Anisotropy Type", idx)
                if anis in ("-", ""):
                    anis = "0"
                line += _pad(anis, 5)

                line += _pad(_fmt(_ii("Diffusion AP", idx), 3), 8)
                line += _pad(_fmt(_ii("Diffusion FH", idx), 3), 9)
                line += _pad(_fmt(_ii("Diffusion RL", idx), 3), 9)

                # label type
                lt = _ii_key("Label Type", idx)
                if lt in ("-", "", "L", "l"):
                    lt = "1"
                else:
                    lt = "0"
                line += _pad(lt, 3)

                w(line + "\n")

            w("\n# === END OF DATA DESCRIPTION FILE ===============================================\n")

        print(f"[xml2par] Generated '{os.path.basename(par_path)}' from '{os.path.basename(xml_path)}'.")
        return par_path

    except Exception as exc:
        print(f"[xml2par] Error writing '{par_path}': {exc}")
        # Clean up partial file.
        try:
            if os.path.isfile(par_path):
                os.remove(par_path)
        except Exception:
            pass
        return None


def _convert_xmlrec_pairs_in_directory(root_dir: str) -> list[str]:
    """Find XML/REC pairs under *root_dir* (one level deep) and generate .PAR headers.

    Returns list of newly generated .PAR paths.
    """
    candidates = [root_dir]
    try:
        for entry in os.scandir(root_dir):
            if entry.is_dir() and not entry.name.startswith("."):
                candidates.append(entry.path)
    except Exception:
        pass

    generated: list[str] = []
    for base in candidates:
        try:
            files = [e.name for e in os.scandir(base) if e.is_file() and not e.name.startswith(".")]
        except Exception:
            continue

        xmls: dict[str, str] = {}
        recs: set[str] = set()
        pars: set[str] = set()
        for name in files:
            low = name.lower()
            stem, ext = os.path.splitext(low)
            if ext == ".xml":
                xmls[stem] = name
            elif ext == ".rec":
                recs.add(stem)
            elif ext == ".par":
                pars.add(stem)

        for stem, xml_name in xmls.items():
            if stem in recs and stem not in pars:
                xml_path = os.path.join(base, xml_name)
                par_path = _convert_xml_to_par(xml_path)
                if par_path:
                    generated.append(par_path)

    return generated


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
        On Windows the PATH separator is ';' instead of ':'.
        """

        try:
            current = os.environ.get("PATH") or ""
            parts = [p for p in current.split(os.pathsep) if p]

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
            os.environ["PATH"] = os.pathsep.join(new_parts)
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

    def _nibabel_convert_parrec(par_path: str) -> bool:
        """Fallback converter: PAR/REC → NIfTI + JSON sidecar using nibabel.

        Used when dcm2niix is unavailable (common on Windows when not
        separately installed).  nibabel's ``parrec`` module can read
        Philips PAR/REC natively.

        Philips PAR/REC files often contain multiple image types per
        acquisition (magnitude=0, real=1, imaginary=2).  dcm2niix splits
        these into separate NIfTI files (``<name>.nii``, ``<name>_real.nii``,
        ``<name>_imaginary.nii``).  We replicate that behaviour here:
        only the **magnitude** volumes are saved to the main NIfTI so
        that downstream code (``build_voxel_matrix``, etc.) sees the
        same shape and content as it would with dcm2niix output.
        """
        try:
            import nibabel as _nib
        except ImportError:
            print("[parrec2nifti] nibabel not available – cannot convert PAR/REC.")
            return False

        try:
            # strict_sort=True makes nibabel group volumes by image type
            # first (mag → real → imag), then by dynamic within each type.
            # This lets us slice the first n_dynamics volumes to get
            # magnitude-only data, matching dcm2niix output.
            img = _nib.parrec.load(par_path, permit_truncated=True, strict_sort=True, scaling="fp")
        except Exception as exc:
            print(f"[parrec2nifti] nibabel failed to load '{par_path}': {exc}")
            return False

        # Determine output filename from protocol name (mirrors dcm2niix %p).
        # dcm2niix strips spaces from protocol names (e.g. "WIP TI_00120" → "WIPTI_00120")
        # so we replicate that behaviour here for consistent naming.
        protocol_name = read_protocol_name_from_par(par_path, default=None)
        if protocol_name:
            # Remove spaces first (dcm2niix behaviour), then sanitize remaining chars.
            stripped = str(protocol_name).replace(" ", "")
            out_stem = sanitize_dcm2niix_basename(stripped) or "parrec"
        else:
            out_stem = os.path.splitext(os.path.basename(par_path))[0]

        out_nii = os.path.join(nifti_directory, out_stem + ".nii")
        out_json = os.path.join(nifti_directory, out_stem + ".json")

        # --- Extract magnitude-only volumes when multiple types exist ------
        try:
            import numpy as _np
            hdr = img.header
            defs = hdr.image_defs
            unique_types = _np.unique(defs["image_type_mr"])
            data = _np.asanyarray(img.dataobj)

            if len(unique_types) > 1 and data.ndim == 4:
                # Multiple image types present (e.g. mag + real + imag).
                # With strict_sort=True the 4th dim is ordered:
                #   [mag_dyn1 … mag_dynN, real_dyn1 … real_dynN, imag_dyn1 … imag_dynN]
                n_dynamics = max(len(_np.unique(defs["dynamic scan number"])), 1)
                mag_data = data[:, :, :, :n_dynamics]
                save_img = _nib.Nifti1Image(mag_data, img.affine, img.header)
                print(
                    f"[parrec2nifti] Extracted magnitude volumes "
                    f"({n_dynamics}/{data.shape[3]}) from "
                    f"'{os.path.basename(par_path)}'"
                )
            else:
                # Single image type or 3-D volume — save as-is.
                save_img = img
        except Exception as exc:
            # If anything goes wrong with type splitting, fall back to
            # saving the full image (better than nothing).
            print(f"[parrec2nifti] Warning: could not split image types: {exc}")
            save_img = img

        # --- Reorient to LAS+ to match dcm2niix output orientation ---------
        # nibabel's parrec.load() keeps voxels in raw acquisition order.
        # dcm2niix (and nibabel's own parrec2nii CLI) reorient to LAS+.
        # Without this step images appear flipped/upside-down and XML ROI
        # coordinates (which assume dcm2niix orientation) land incorrectly.
        try:
            import numpy as _np
            from nibabel.orientations import io_orientation, apply_orientation, inv_ornt_aff

            _sv_data = _np.asanyarray(save_img.dataobj)
            _sv_aff = save_img.affine.copy()

            # Target LAS+: the np.diag([-1,1,1,1]) trick negates the R axis
            # so that io_orientation considers L (not R) as the "positive"
            # direction for axis-0 -- exactly what parrec2nii does.
            _ornt = io_orientation(_np.diag([-1, 1, 1, 1]).dot(_sv_aff))
            _las_identity = _np.array([[0, 1], [1, 1], [2, 1]])
            if not _np.array_equal(_ornt, _las_identity):
                _t_aff = inv_ornt_aff(_ornt, _sv_data.shape)
                _sv_aff = _np.dot(_sv_aff, _t_aff)
                _sv_data = apply_orientation(_sv_data, _ornt)
                save_img = _nib.Nifti1Image(_sv_data, _sv_aff)
        except Exception as exc:
            # Non-fatal: save with original orientation rather than crashing.
            print(f"[parrec2nifti] Warning: LAS+ reorientation failed: {exc}")

        try:
            _nib.save(save_img, out_nii)
        except Exception as exc:
            print(f"[parrec2nifti] nibabel failed to save NIfTI for '{par_path}': {exc}")
            return False

        # Build a minimal JSON sidecar so downstream metadata lookup works.
        sidecar: dict = {}
        if protocol_name:
            # Store the dcm2niix-style name (spaces stripped) for consistent matching.
            sidecar["ProtocolName"] = str(protocol_name).replace(" ", "")
            sidecar["SeriesDescription"] = str(protocol_name).replace(" ", "")
            sidecar["ProtocolNameOriginal"] = str(protocol_name)
        sidecar["ConversionSoftware"] = "nibabel"
        sidecar["ConversionSoftwareVersion"] = str(getattr(_nib, "__version__", "unknown"))

        # Extract rich metadata via _parrec_header_fields (uses nibabel internally).
        try:
            from utils.loading import _parrec_header_fields
            extra = _parrec_header_fields(par_path)
            for k, v in extra.items():
                if k not in sidecar:
                    sidecar[k] = v
        except Exception:
            pass

        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2)
                f.write("\n")
        except Exception:
            pass

        print(f"[parrec2nifti] Converted '{os.path.basename(par_path)}' via nibabel → {out_stem}.nii")
        return True

    # Auto-convert Philips XML/REC pairs into PAR so dcm2niix can handle them.
    _convert_xmlrec_pairs_in_directory(directory)

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
                    # dcm2niix previously failed for this PAR; try nibabel
                    # fallback instead of re-running dcm2niix.
                    if not marker.get("nibabel_attempted"):
                        ok = _nibabel_convert_parrec(par_path)
                        if ok:
                            _clear_failure_marker(par_path)
                        else:
                            # Mark that nibabel was also attempted so we don't
                            # retry every run.
                            try:
                                marker["nibabel_attempted"] = True
                                p = _failure_marker_path(par_path)
                                with open(p, "w", encoding="utf-8") as f:
                                    json.dump(marker, f, indent=2)
                            except Exception:
                                pass
                else:
                    ok = _run_dcm2niix(par_path)
                    if not ok:
                        # Fallback: use nibabel when dcm2niix is missing/broken
                        # (common on Windows without a separate dcm2niix install).
                        ok = _nibabel_convert_parrec(par_path)
                        if ok:
                            _clear_failure_marker(par_path)
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
            ok = _run_dcm2niix(directory)
            # If dcm2niix failed and there are still no NIfTI files, scan for
            # any PAR/REC pairs that may have been missed (e.g. XML→PAR
            # generated ones) and try nibabel fallback.
            if not ok and not _has_nifti(nifti_directory):
                leftover_pars = _find_parrec_pairs_one_level(directory)
                for lp in leftover_pars:
                    _nibabel_convert_parrec(lp)

        # Avoid decompressing by default (can be hours on large cohorts).
        # Opt-in via: PBRAIN_DECOMPRESS_NIFTI_GZ=1
        if os.environ.get("PBRAIN_DECOMPRESS_NIFTI_GZ") == "1":
            decompress_gz_in_directory(nifti_directory)

