#!/usr/bin/env python3
"""
Poly USB headset: GUI record & playback, or CLI live monitor (--monitor).

System deps: ./install_pi.sh (includes python3-tk on Pi for Tkinter)
Python: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError as e:
    tk = None  # type: ignore
    ttk = None  # type: ignore
    messagebox = None  # type: ignore
    _tk_import_error = e
else:
    _tk_import_error = None


def _pick_default_poly() -> Optional[int]:
    """Prefer a duplex device whose name suggests Poly / Plantronics."""
    devices = sd.query_devices()
    hints = ("poly", "plantronics")
    best: Optional[tuple[int, int]] = None
    for idx, d in enumerate(devices):
        name = d["name"].lower()
        if not any(h in name for h in hints):
            continue
        ins = int(d["max_input_channels"])
        outs = int(d["max_output_channels"])
        if ins < 1 or outs < 1:
            continue
        score = ins + outs
        if best is None or score > best[1]:
            best = (idx, score)
    return best[0] if best else None


def _describe_device(idx: int) -> str:
    d = sd.query_devices(idx)
    return (
        f"[{idx}] {d['name']!r}  "
        f"in={d['max_input_channels']} out={d['max_output_channels']}  "
        f"default_sr≈{d['default_samplerate']:.0f} Hz"
    )


def list_devices() -> None:
    print("Audio devices (use --device <index> for duplex USB headset):\n")
    for i in range(len(sd.query_devices())):
        print(_describe_device(i))
    print()


def make_callback(in_ch: int, out_ch: int):
    """Copy input to output; up-mix mono mic to stereo headphones if needed."""

    def callback(
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time,
        status,
    ) -> None:
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

    return callback


def run_monitor(device: int, blocksize: int, latency: str) -> None:
    info = sd.query_devices(device)
    in_ch = min(2, int(info["max_input_channels"]))
    out_ch = min(2, int(info["max_output_channels"]))
    if in_ch < 1 or out_ch < 1:
        raise SystemExit(f"Device {_describe_device(device)} is not full-duplex usable.")

    rate = int(float(info["default_samplerate"]))
    if rate <= 0:
        rate = 48000

    print(_describe_device(device))
    print(f"Streaming {in_ch}→{out_ch} ch @ {rate} Hz, blocksize={blocksize}, latency={latency}")
    print("Stop with Ctrl+C.\n")

    stream_kwargs = dict(
        device=device,
        samplerate=rate,
        blocksize=blocksize,
        latency=latency,
        callback=make_callback(in_ch, out_ch),
    )
    if in_ch != out_ch:
        stream_kwargs["channels"] = (in_ch, out_ch)
    else:
        stream_kwargs["channels"] = in_ch

    stop = False

    def handle_sig(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        with sd.Stream(**stream_kwargs):
            while not stop:
                sd.sleep(250)
    except OSError as e:
        raise SystemExit(
            f"Audio error: {e}\n"
            "Try: Permission denied → add user to group 'audio': sudo usermod -aG audio $USER"
        ) from e


def _device_rate(device_idx: int) -> int:
    info = sd.query_devices(device_idx)
    r = int(float(info["default_samplerate"]))
    return r if r > 0 else 48000


def _save_wav(path: Path, samples: np.ndarray, samplerate: int) -> None:
    if samples.size == 0:
        raise ValueError("No samples to save")
    if samples.dtype != np.int16:
        x = np.clip(samples.astype(np.float64), -1.0, 1.0)
        samples = (x * 32767.0).astype(np.int16)
    if samples.ndim == 1:
        nch = 1
        flat = samples
    else:
        nch = samples.shape[1]
        flat = samples.reshape(-1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(flat.tobytes())


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        nch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        data = data.reshape(-1, nch)
    else:
        data = data.reshape(-1, 1)
    return data, rate


class PolyGuiApp(tk.Tk):
    def __init__(self, initial_device: Optional[int] = None) -> None:
        super().__init__()
        self.title("Poly audio — record & playback")
        self.minsize(420, 220)
        self._recording = False
        self._record_stream: Optional[sd.InputStream] = None
        self._record_chunks: list[np.ndarray] = []
        self._record_lock = threading.Lock()
        self._wav_path: Optional[Path] = None

        pad = {"padx": 10, "pady": 6}

        ttk.Label(self, text="Audio device:").pack(anchor="w", **pad)
        self._device_var = tk.StringVar()
        self._combo = ttk.Combobox(self, textvariable=self._device_var, width=68, state="readonly")
        self._combo.pack(fill="x", **pad)
        self._reload_devices(initial_device)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", **pad)
        self._btn_record = ttk.Button(btn_row, text="Record", command=self._toggle_record)
        self._btn_record.pack(side="left", padx=(0, 8))
        self._btn_play = ttk.Button(btn_row, text="Playback", command=self._playback, state="disabled")
        self._btn_play.pack(side="left")
        self._btn_refresh = ttk.Button(btn_row, text="Refresh devices", command=lambda: self._reload_devices(None))
        self._btn_refresh.pack(side="right")

        self._status = ttk.Label(self, text="Idle.")
        self._status.pack(anchor="w", **pad)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _reload_devices(self, select_idx: Optional[int]) -> None:
        opts: list[str] = []
        self._indices: list[int] = []
        for i in range(len(sd.query_devices())):
            d = sd.query_devices(i)
            label = f"{i}: {d['name']} (in={d['max_input_channels']}, out={d['max_output_channels']})"
            opts.append(label)
            self._indices.append(i)
        self._combo["values"] = opts
        if not opts:
            self._set_status("No audio devices found.")
            return
        pick = 0
        if select_idx is not None and select_idx in self._indices:
            pick = self._indices.index(select_idx)
        elif _pick_default_poly() is not None:
            poly = _pick_default_poly()
            assert poly is not None
            if poly in self._indices:
                pick = self._indices.index(poly)
        self._combo.current(pick)
        self._set_status("Pick device with mic for Record and speakers for Playback.")

    def _selected_device_idx(self) -> int:
        i = self._combo.current()
        if i < 0:
            raise RuntimeError("No device selected")
        return self._indices[i]

    def _set_status(self, text: str) -> None:
        self._status.configure(text=text)

    def _toggle_record(self) -> None:
        if self._recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self) -> None:
        dev = self._selected_device_idx()
        info = sd.query_devices(dev)
        if int(info["max_input_channels"]) < 1:
            messagebox.showerror("Record", "Selected device has no microphone input.")
            return
        in_ch = min(2, int(info["max_input_channels"]))
        rate = _device_rate(dev)
        self._record_chunks = []
        self._recording = True
        self._btn_play.configure(state="disabled")
        self._btn_record.configure(text="Stop")

        def cb(indata: np.ndarray, frames: int, time, status) -> None:
            if status:
                self.after(0, lambda s=str(status): self._set_status(s))
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
        except OSError as e:
            self._recording = False
            self._btn_record.configure(text="Record")
            messagebox.showerror("Record", str(e))
            return

        self._set_status(f"Recording… {rate} Hz, {in_ch} ch (device {dev})")

    def _stop_record(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._btn_record.configure(text="Record")
        if self._record_stream is not None:
            self._record_stream.stop()
            self._record_stream.close()
            self._record_stream = None

        with self._record_lock:
            chunks = self._record_chunks
            self._record_chunks = []

        if not chunks:
            self._set_status("No audio captured.")
            return

        audio = np.concatenate(chunks, axis=0)
        dev = self._selected_device_idx()
        rate = _device_rate(dev)

        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="poly_rec_")
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
        self._set_status(f"Saved {path.name} — {audio.shape[0] / rate:.1f}s @ {rate} Hz")

    def _playback(self) -> None:
        if self._wav_path is None or not self._wav_path.is_file():
            messagebox.showinfo("Playback", "Record something first.")
            return
        dev = self._selected_device_idx()
        info = sd.query_devices(dev)
        if int(info["max_output_channels"]) < 1:
            messagebox.showerror("Playback", "Selected device has no audio output.")
            return

        try:
            data, rate = _load_wav(self._wav_path)
        except Exception as e:
            messagebox.showerror("Playback", str(e))
            return

        out_ch = min(int(info["max_output_channels"]), 2)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[1] == 1 and out_ch >= 2:
            data = np.column_stack([data[:, 0], data[:, 0]])
        elif data.shape[1] > out_ch:
            data = data[:, :out_ch]

        self._btn_play.configure(state="disabled")
        self._btn_record.configure(state="disabled")
        self._set_status("Playing…")

        def run() -> None:
            try:
                sd.play(data, rate, device=dev)
                sd.wait()
            except Exception as e:
                self.after(0, lambda err=e: messagebox.showerror("Playback", str(err)))
            finally:
                self.after(0, self._play_finished)

        threading.Thread(target=run, daemon=True).start()

    def _play_finished(self) -> None:
        self._btn_play.configure(state="normal")
        self._btn_record.configure(state="normal")
        self._set_status("Playback finished.")

    def _on_close(self) -> None:
        if self._recording:
            self._stop_record()
        sd.stop()
        self.destroy()


def run_gui(initial_device: Optional[int]) -> None:
    if tk is None:
        raise SystemExit(
            "Tkinter is not installed. On Raspberry Pi OS: sudo apt install python3-tk\n"
            f"Import error: {_tk_import_error}"
        )
    app = PolyGuiApp(initial_device=initial_device)
    app.mainloop()


def main() -> None:
    p = argparse.ArgumentParser(description="Poly USB headset — GUI record/playback or CLI live monitor.")
    p.add_argument(
        "--monitor",
        action="store_true",
        help="CLI: live mic→headphone monitor instead of opening the GUI.",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="Print sounddevice indices and exit.",
    )
    p.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Device index for --monitor or default selection in GUI.",
    )
    p.add_argument(
        "--blocksize",
        type=int,
        default=512,
        help="(monitor) Frames per callback. Default: 512.",
    )
    p.add_argument(
        "--latency",
        choices=("low", "high"),
        default="low",
        help="(monitor) PortAudio latency hint.",
    )
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.monitor:
        dev = args.device
        if dev is None:
            dev = _pick_default_poly()
            if dev is None:
                list_devices()
                raise SystemExit(
                    "No Poly/Plantronics duplex device found. "
                    "Pass --device <index> from the list above."
                )
        lat = "low" if args.latency == "low" else "high"
        run_monitor(dev, max(32, args.blocksize), lat)
        return

    run_gui(initial_device=args.device)


if __name__ == "__main__":
    main()
