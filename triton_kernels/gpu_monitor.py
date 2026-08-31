"""GPU clock, power and clock-event sampling through nvidia-smi.

A Python port of csrc/harness/gpu_monitor.h with the same fields, the same
background sampling and the same settle rule, so rows from the two harnesses
are recorded under the same conditions.
"""
import dataclasses
import subprocess
import threading
import time

FIELDS = ("clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,"
          "clocks_event_reasons.active,"
          "clocks_event_reasons_counters.sw_power_cap,"
          "clocks_event_reasons_counters.sw_thermal_slowdown,"
          "clocks_event_reasons_counters.hw_thermal_slowdown,"
          "clocks_event_reasons_counters.hw_power_brake_slowdown,"
          "enforced.power.limit")


@dataclasses.dataclass
class GpuSample:
    valid: bool = False
    t_s: float = 0.0            # time.monotonic()
    sm_mhz: int = -1
    mem_mhz: int = -1
    temp_c: int = -1
    power_w: float = -1.0
    reasons: str = ""
    power_limit_w: float = -1.0
    # Cumulative microseconds per slowdown reason; only differences matter.
    us_sw_power_cap: int = -1
    us_sw_thermal: int = -1
    us_hw_thermal: int = -1
    us_hw_power_brake: int = -1


def _int(s):
    try:
        return int(s)
    except ValueError:
        return -1


def _float(s):
    try:
        return float(s)
    except ValueError:
        return -1.0


def sample_gpu(selector=""):
    """One nvidia-smi sample. Unreadable fields map to -1 rather than failing."""
    cmd = ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"]
    if selector:
        cmd.append(f"--id={selector}")
    s = GpuSample(t_s=time.monotonic())
    try:
        line = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=10).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return s
    f = [p.strip() for p in line.split(",")]
    if len(f) != 10:
        return s
    s.sm_mhz, s.mem_mhz, s.temp_c = _int(f[0]), _int(f[1]), _int(f[2])
    s.power_w = _float(f[3])
    s.reasons = f[4] if f[4].startswith("0x") else ""
    s.us_sw_power_cap, s.us_sw_thermal = _int(f[5]), _int(f[6])
    s.us_hw_thermal, s.us_hw_power_brake = _int(f[7]), _int(f[8])
    s.power_limit_w = _float(f[9])
    s.valid = s.sm_mhz >= 0
    return s


class ClockSampler:
    """Samples in a background thread so the GPU stays loaded while sampling."""

    def __init__(self, selector, interval_s):
        self.selector = selector
        self.interval_s = interval_s
        self._samples = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        with self._lock:
            self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
            self._thread = None

    def snapshot(self):
        with self._lock:
            return list(self._samples)

    def _run(self):
        while not self._stop.is_set():
            s = sample_gpu(self.selector)
            if s.valid:
                with self._lock:
                    self._samples.append(s)
            self._stop.wait(self.interval_s)


def clock_settled(samples, window, tol):
    """Mean SM clock of the last `window` samples within `tol` of the
    `window` before them (means, not individual samples: a power-capped part
    dithers by several percent indefinitely)."""
    if window == 0 or len(samples) < 2 * window:
        return False
    earlier = sum(s.sm_mhz for s in samples[-2 * window:-window]) / window
    recent = sum(s.sm_mhz for s in samples[-window:]) / window
    return recent > 0 and abs(recent - earlier) <= tol * recent


def clock_window(samples, window):
    """Mean SM clock over the last `window` samples and its range in percent."""
    tail = samples[-window:] if window else []
    if not tail:
        return -1.0, -1.0
    mhz = [s.sm_mhz for s in tail]
    mean = sum(mhz) / len(mhz)
    return mean, (100.0 * (max(mhz) - min(mhz)) / mean if mean > 0 else -1.0)


def active_ms(before_us, after_us):
    if before_us < 0 or after_us < 0 or after_us < before_us:
        return -1.0
    return (after_us - before_us) / 1000.0
