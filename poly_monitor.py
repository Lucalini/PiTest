#!/usr/bin/env python3
"""
Poly USB headset audio tool.

Default mode opens a small Tkinter GUI with separate input/output device
dropdowns, a Record/Stop toggle, a Playback button, a level meter and a
status line. Use --monitor for the CLI live mic-to-headphone passthrough.

System deps (Pi):  ./install_pi.sh   (installs portaudio + python3-tk)
Python deps:       pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as e:
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR: Optional[Exception] = e
else:
    _TK_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Audio helpers (used by GUI and CLI monitor)
# ---------------------------------------------------------------------------

POLY_HINTS = ("poly", "plantronics")


def _query_devices() -> list[dict]:
    return list(sd.query_devices())


def _device_rate(idx: int, fallback: int = 48000) -> int:
    info = sd.query_devices(idx)
    r = int(float(info["default_samplerate"]) or 0)
    return r if r > 0 else fallback


def _describe(idx: int) -> str:
    d = sd.query_devices(idx)
    return (
        f"[{idx}] {d['name']!r}  "
        f"in={d['max_input_channels']} out={d['max_output_channels']}  "
        f"sr~{int(float(d['default_samplerate']) or 0)}Hz"
    )


def _pick_default(devices: list[dict], *, want_input: bool, want_output: bool) -> int:
    """Pick the best device. Prefer Poly/Plantronics; otherwise first match."""
    best_score = -1
    best_idx = -1
    for i, d in enumerate(devices):
        ins = int(d["max_input_channels"])
        outs = int(d["max_output_channels"])
        if want_input and ins < 1:
            continue
        if want_output and outs < 1:
            continue
        score = 0
        name = str(d["name"]).lower()
        if any(h in name for h in POLY_HINTS):
            score += 100
        score += ins + outs
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def list_devices() -> None:
    print("Audio devices:\n")
    for i, _ in enumerate(_query_devices()):
        print(_describe(i))
    print()


def _save_wav(path: Path, samples: np.ndarray, samplerate: int) -> None:
    if samples.size == 0:
        raise ValueError("No samples to save")
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    nch = samples.shape[1]
    if samples.dtype != np.int16:
        x = np.clip(samples.astype(np.float32), -1.0, 1.0)
        samples = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(samples.reshape(-1).tobytes())


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        nch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    data = data.reshape(-1, nch) if nch > 1 else data.reshape(-1, 1)
    return data, rate


# ---------------------------------------------------------------------------
# CLI live monitor: mic -> headphones
# ---------------------------------------------------------------------------


def _make_monitor_callback(in_ch: int, out_ch: int) -> Callable:
    def cb(indata: np.ndarray, outdata: np.ndarray, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if in_ch == 1 and out_ch >= 2:
            m = indata[:, 0]
            outdata[:, 0] = m
            outdata[:, 1] = m
        elif in_ch == out_ch:
            outdata[:] = indata
        else:
            n = min(in_ch, out_ch)
            outdata[:, :n] = indata[:, :n]
            if out_ch > n:
                outdata[:, n:] = 0
    return cb


def run_monitor(device: int, blocksize: int, latency: str) -> None:
    info = sd.query_devices(device)
    in_ch = min(2, int(info["max_input_channels"]))
    out_ch = min(2, int(info["max_output_channels"]))
    if in_ch < 1 or out_ch < 1:
        raise SystemExit(f"Device {_describe(device)} is not full-duplex usable.")
    rate = _device_rate(device)

    print(_describe(device))
    print(f"Streaming {in_ch}->{out_ch} ch @ {rate} Hz, blocksize={blocksize}, latency={latency}")
    print("Stop with Ctrl+C.\n")

    kwargs = dict(
        device=device,
        samplerate=rate,
        blocksize=blocksize,
        latency=latency,
        callback=_make_monitor_callback(in_ch, out_ch),
        channels=(in_ch, out_ch) if in_ch != out_ch else in_ch,
    )

    stop = False

    def _sig(_s, _f):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        with sd.Stream(**kwargs):
            while not stop:
                sd.sleep(200)
    except OSError as e:
        raise SystemExit(
            f"Audio error: {e}\n"
            "Tip: add user to 'audio' group: sudo usermod -aG audio $USER (re-login)"
        ) from e


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


@dataclass
class _DevOption:
    label: str
    index: int


class PolyGui:
    """Tkinter GUI: pick input + output device, record, play back."""

    def __init__(self, initial_device: Optional[int] = None) -> None:
        assert tk is not None and ttk is not None and messagebox is not None

        self.root = tk.Tk()
        self.root.title("Poly Audio - Record & Playback")
        self.root.minsize(520, 280)

        self._initial_device = initial_device
        self._recording = False
        self._record_stream: Optional[sd.InputStream] = None
        self._record_chunks: list[np.ndarray] = []
        self._record_lock = threading.Lock()
        self._wav_path: Optional[Path] = None
        self._level: float = 0.0
        self._playing = False

        self._input_options: list[_DevOption] = []
        self._output_options: list[_DevOption] = []

        self._build_ui()
        self._reload_devices()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._tick_meter)

    # ---- UI construction --------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}

        header = ttk.Label(
            self.root,
            text="Poly USB Headset Recorder",
            font=("TkDefaultFont", 14, "bold"),
        )
        header.pack(anchor="w", **pad)

        # Input row
        in_row = ttk.Frame(self.root)
        in_row.pack(fill="x", **pad)
        ttk.Label(in_row, text="Microphone:", width=12).pack(side="left")
        self._input_var = tk.StringVar()
        self._input_combo = ttk.Combobox(
            in_row, textvariable=self._input_var, state="readonly"
        )
        self._input_combo.pack(side="left", fill="x", expand=True)

        # Output row
        out_row = ttk.Frame(self.root)
        out_row.pack(fill="x", **pad)
        ttk.Label(out_row, text="Speakers:", width=12).pack(side="left")
        self._output_var = tk.StringVar()
        self._output_combo = ttk.Combobox(
            out_row, textvariable=self._output_var, state="readonly"
        )
        self._output_combo.pack(side="left", fill="x", expand=True)

        # Buttons
        btn_row = ttk.Frame(self.root)
        btn_row.pack(fill="x", **pad)
        self._btn_record = ttk.Button(
            btn_row, text="Record", width=14, command=self._toggle_record
        )
        self._btn_record.pack(side="left", padx=(0, 8))
        self._btn_play = ttk.Button(
            btn_row, text="Playback", width=14, command=self._playback, state="disabled"
        )
        self._btn_play.pack(side="left", padx=(0, 8))
        self._btn_refresh = ttk.Button(
            btn_row, text="Refresh devices", command=self._reload_devices
        )
        self._btn_refresh.pack(side="right")

        # Level meter
        meter_row = ttk.Frame(self.root)
        meter_row.pack(fill="x", **pad)
        ttk.Label(meter_row, text="Level:", width=12).pack(side="left")
        self._meter = ttk.Progressbar(
            meter_row, orient="horizontal", mode="determinate", maximum=100
        )
        self._meter.pack(side="left", fill="x", expand=True)

        # Status bar
        self._status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(
            self.root,
            textvariable=self._status_var,
            anchor="w",
            relief="sunken",
        )
        status.pack(fill="x", side="bottom")

    # ---- Device handling --------------------------------------------------

    def _reload_devices(self) -> None:
        devices = _query_devices()

        self._input_options = []
        self._output_options = []
        for i, d in enumerate(devices):
            label = f"{i}: {d['name']}"
            if int(d["max_input_channels"]) > 0:
                self._input_options.append(
                    _DevOption(f"{label}  (in={d['max_input_channels']})", i)
                )
            if int(d["max_output_channels"]) > 0:
                self._output_options.append(
                    _DevOption(f"{label}  (out={d['max_output_channels']})", i)
                )

        self._input_combo["values"] = [o.label for o in self._input_options]
        self._output_combo["values"] = [o.label for o in self._output_options]

        if not self._input_options and not self._output_options:
            self._set_status("No audio devices found. Plug in the headset and click Refresh.")
            return

        # Default selections
        if self._input_options:
            pick = self._initial_device if self._initial_device is not None else -1
            idx = next((j for j, o in enumerate(self._input_options) if o.index == pick), -1)
            if idx < 0:
                best = _pick_default(devices, want_input=True, want_output=False)
                idx = next((j for j, o in enumerate(self._input_options) if o.index == best), 0)
            self._input_combo.current(idx)

        if self._output_options:
            pick = self._initial_device if self._initial_device is not None else -1
            idx = next((j for j, o in enumerate(self._output_options) if o.index == pick), -1)
            if idx < 0:
                best = _pick_default(devices, want_input=False, want_output=True)
                idx = next((j for j, o in enumerate(self._output_options) if o.index == best), 0)
            self._output_combo.current(idx)

        self._set_status("Pick devices and press Record.")

    def _selected_input_idx(self) -> Optional[int]:
        i = self._input_combo.current()
        if i < 0 or i >= len(self._input_options):
            return None
        return self._input_options[i].index

    def _selected_output_idx(self) -> Optional[int]:
        i = self._output_combo.current()
        if i < 0 or i >= len(self._output_options):
            return None
        return self._output_options[i].index

    # ---- Status / meter ---------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    def _tick_meter(self) -> None:
        # Decay the visible level toward the latest sample.
        target = self._level if self._recording else 0.0
        current = float(self._meter["value"]) / 100.0
        next_val = max(target, current * 0.85)
        self._meter["value"] = max(0.0, min(1.0, next_val)) * 100.0
        self.root.after(80, self._tick_meter)

    # ---- Record -----------------------------------------------------------

    def _toggle_record(self) -> None:
        if self._recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self) -> None:
        dev = self._selected_input_idx()
        if dev is None:
            messagebox.showerror("Record", "Choose a microphone first.")
            return
        info = sd.query_devices(dev)
        in_ch = min(2, int(info["max_input_channels"]))
        if in_ch < 1:
            messagebox.showerror("Record", "Selected device has no microphone input.")
            return
        rate = _device_rate(dev)

        with self._record_lock:
            self._record_chunks = []

        def cb(indata: np.ndarray, frames, time, status) -> None:
            if status:
                self.root.after(0, lambda s=str(status): self._set_status(s))
            try:
                peak = float(np.max(np.abs(indata))) if indata.size else 0.0
            except Exception:
                peak = 0.0
            self._level = peak
            with self._record_lock:
                self._record_chunks.append(indata.copy())

        try:
            self._record_stream = sd.InputStream(
                device=dev,
                channels=in_ch,
                samplerate=rate,
                dtype="float32",
                callback=cb,
            )
            self._record_stream.start()
        except Exception as e:
            self._record_stream = None
            messagebox.showerror("Record", f"Could not open mic:\n{e}")
            return

        self._recording = True
        self._btn_record.configure(text="Stop")
        self._btn_play.configure(state="disabled")
        self._set_status(f"Recording... {rate} Hz, {in_ch} ch  (device {dev})")

    def _stop_record(self) -> None:
        self._recording = False
        self._btn_record.configure(text="Record")
        if self._record_stream is not None:
            try:
                self._record_stream.stop()
                self._record_stream.close()
            finally:
                self._record_stream = None

        with self._record_lock:
            chunks = self._record_chunks
            self._record_chunks = []

        if not chunks:
            self._set_status("No audio captured.")
            return

        audio = np.concatenate(chunks, axis=0)
        dev = self._selected_input_idx()
        rate = _device_rate(dev) if dev is not None else 48000

        fd, tmp = tempfile.mkstemp(prefix="poly_rec_", suffix=".wav")
        os.close(fd)
        path = Path(tmp)
        try:
            _save_wav(path, audio, rate)
        except Exception as e:
            messagebox.showerror("Save", str(e))
            self._set_status("Failed to save recording.")
            return

        self._wav_path = path
        self._btn_play.configure(state="normal")
        secs = audio.shape[0] / rate
        self._set_status(f"Saved {path.name}  ({secs:.1f}s @ {rate} Hz)")

    # ---- Playback ---------------------------------------------------------

    def _playback(self) -> None:
        if self._playing:
            return
        if self._wav_path is None or not self._wav_path.is_file():
            messagebox.showinfo("Playback", "Record something first.")
            return

        out_dev = self._selected_output_idx()
        if out_dev is None:
            messagebox.showerror("Playback", "Choose a speakers/output device first.")
            return

        info = sd.query_devices(out_dev)
        out_ch = min(int(info["max_output_channels"]), 2)
        if out_ch < 1:
            messagebox.showerror("Playback", "Selected device has no audio output.")
            return

        try:
            data, rate = _load_wav(self._wav_path)
        except Exception as e:
            messagebox.showerror("Playback", str(e))
            return

        if data.shape[1] == 1 and out_ch >= 2:
            data = np.column_stack([data[:, 0], data[:, 0]])
        elif data.shape[1] > out_ch:
            data = data[:, :out_ch]

        self._playing = True
        self._btn_play.configure(state="disabled")
        self._btn_record.configure(state="disabled")
        self._set_status("Playing...")

        def runner() -> None:
            err: Optional[Exception] = None
            try:
                sd.play(data, rate, device=out_dev)
                sd.wait()
            except Exception as ex:
                err = ex
            self.root.after(0, lambda e=err: self._play_finished(e))

        threading.Thread(target=runner, daemon=True).start()

    def _play_finished(self, err: Optional[Exception]) -> None:
        self._playing = False
        self._btn_play.configure(state="normal")
        self._btn_record.configure(state="normal")
        if err is not None:
            messagebox.showerror("Playback", str(err))
            self._set_status("Playback failed.")
        else:
            self._set_status("Playback finished.")

    # ---- Lifecycle --------------------------------------------------------

    def _on_close(self) -> None:
        try:
            if self._recording:
                self._stop_record()
            sd.stop()
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _ensure_gui_display() -> None:
    """If we were started over SSH but the Pi has a desktop on :0, point Tk there."""
    if os.environ.get("DISPLAY"):
        return
    if sys.platform != "linux":
        return
    sock_dir = Path("/tmp/.X11-unix")
    for disp, sock in ((":0", "X0"), (":1", "X1")):
        if (sock_dir / sock).exists():
            os.environ["DISPLAY"] = disp
            xauth = Path.home() / ".Xauthority"
            if xauth.is_file():
                os.environ.setdefault("XAUTHORITY", str(xauth))
            return


def run_gui(initial_device: Optional[int]) -> None:
    _ensure_gui_display()
    if tk is None:
        raise SystemExit(
            "Tkinter is not installed. On Raspberry Pi OS run: sudo apt install -y python3-tk\n"
            f"Import error: {_TK_IMPORT_ERROR}"
        )
    try:
        PolyGui(initial_device=initial_device).run()
    except tk.TclError as e:
        if "display" in str(e).lower():
            raise SystemExit(
                "No GUI display available.\n"
                "  - Run on the Pi's desktop, or\n"
                "  - From SSH with the Pi's desktop active: DISPLAY=:0 XAUTHORITY=~/.Xauthority python3 poly_monitor.py\n"
                "  - Or use the CLI passthrough: python3 poly_monitor.py --monitor\n"
                f"  ({e})"
            ) from e
        raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Poly USB headset record/playback GUI (default), or CLI live monitor."
    )
    p.add_argument("--monitor", action="store_true",
                   help="Run the CLI live mic->headphone passthrough instead of the GUI.")
    p.add_argument("--list-devices", action="store_true",
                   help="Print PortAudio device indices and exit.")
    p.add_argument("--device", type=int, default=None, metavar="INDEX",
                   help="Device index to preselect in the GUI, or to use for --monitor.")
    p.add_argument("--blocksize", type=int, default=512,
                   help="(monitor) Frames per callback. Default: 512.")
    p.add_argument("--latency", choices=("low", "high"), default="low",
                   help="(monitor) PortAudio latency hint.")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.monitor:
        dev = args.device
        if dev is None:
            dev = _pick_default(_query_devices(), want_input=True, want_output=True)
            if dev < 0:
                list_devices()
                raise SystemExit(
                    "No full-duplex device found. Pass --device <index> from the list."
                )
        run_monitor(dev, max(32, args.blocksize), args.latency)
        return

    run_gui(initial_device=args.device)


if __name__ == "__main__":
    main()
