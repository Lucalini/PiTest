#!/usr/bin/env python3
"""
Live monitor: Poly USB headset microphone → headset playback (sidetone-style).

Run on the Raspberry Pi with the Poly headset connected. Use --list-devices if
auto-detection fails, then pass --device <index>.

System deps (once): ./install_pi.sh
Python env: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import signal
import sys
from typing import Optional

import numpy as np
import sounddevice as sd


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


def main() -> None:
    p = argparse.ArgumentParser(description="Poly USB headset mic → headset playback monitor.")
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
        help="Duplex device index (see --list-devices). Default: auto Poly/Plantronics.",
    )
    p.add_argument(
        "--blocksize",
        type=int,
        default=512,
        help="Frames per callback (lower = less latency, more CPU). Default: 512.",
    )
    p.add_argument(
        "--latency",
        choices=("low", "high"),
        default="low",
        help="PortAudio latency hint. Default: low.",
    )
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

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


if __name__ == "__main__":
    main()
