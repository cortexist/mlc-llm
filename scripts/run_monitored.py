"""Run one Xavier workload with live telemetry and conservative stop limits.

Stream stdout to a local file over SSH as well as keeping the workload log on
the board, so the last samples survive another abrupt board power loss.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("--workload-log", required=True)
parser.add_argument("--max-temp", type=float, default=85)
parser.add_argument("--min-available-mb", type=int, default=2048)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    parser.error("a workload command is required")


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
log_lock = threading.Lock()


def emit(event):
    with log_lock:
        print(
            json.dumps({"time": datetime.datetime.now().astimezone().isoformat(), **event}),
            flush=True,
        )


def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


def snapshot():
    temps = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        name, value = read(zone / "type"), read(zone / "temp")
        if name and value:
            temps[name] = int(value) / 1000
    sensors = {}
    rails = {}
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        name = read(hwmon / "name")
        for pattern in ["oc*_irq_cnt", "pwm*", "rpm", "curr*_crit_alarm"]:
            for path in hwmon.glob(pattern):
                value = read(path)
                if value is not None:
                    sensors[f"{name}/{path.name}"] = value
        if name == "ina3221":
            for channel in range(1, 4):
                label = read(hwmon / f"in{channel}_label")
                voltage = read(hwmon / f"in{channel}_input")
                current = read(hwmon / f"curr{channel}_input")
                if label and voltage and current:
                    rails[label] = {
                        "mv": int(voltage),
                        "ma": int(current),
                        "mw": int(voltage) * int(current) / 1000,
                    }
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return dict(
        temperatures_c=temps,
        sensors=sensors,
        rails=rails,
        available_mb=int(mem["MemAvailable"].split()[0]) // 1024,
    )


stats = subprocess.Popen(
    ["tegrastats", "--interval", "1000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)


def forward_stats():
    for line in stats.stdout:
        emit({"tegrastats": line.strip()})


thread = threading.Thread(target=forward_stats, daemon=True)
thread.start()
workload = None
stopped = False
try:
    initial = snapshot()
    emit({"event": "start", "command": command, **initial})
    Path(args.workload_log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.workload_log, "w") as log:
        workload = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        while workload.poll() is None:
            sample = snapshot()
            emit(sample)
            soc_temps = [
                v
                for k, v in sample["temperatures_c"].items()
                if k in ["CPU-therm", "GPU-therm", "AUX-therm", "Tboard_tegra", "Tdiode_tegra"]
            ]
            if (soc_temps and max(soc_temps) >= args.max_temp) or sample[
                "available_mb"
            ] < args.min_available_mb:
                emit(
                    {
                        "event": "stop_limit",
                        "max_temp": args.max_temp,
                        "min_available_mb": args.min_available_mb,
                    }
                )
                stopped = True
                os.killpg(workload.pid, signal.SIGTERM)
                try:
                    workload.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(workload.pid, signal.SIGKILL)
                break
            time.sleep(1)
        status = workload.wait()
        emit({"event": "exit", "returncode": status, "stopped_by_monitor": stopped})
finally:
    if workload is not None and workload.poll() is None:
        os.killpg(workload.pid, signal.SIGTERM)
    stats.terminate()
    stats.wait(timeout=5)
raise SystemExit(125 if stopped else status)
