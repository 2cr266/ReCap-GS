"""Small dependency-free CSV logger for Gaussian-count trajectories."""

import csv
import os


FIELDS = (
    "global_iteration",
    "phase",
    "phase_iteration",
    "gaussian_count",
    "event",
)


def validate_count_interval(interval):
    if type(interval) is not int or interval <= 0:
        raise ValueError("gaussian_count_interval must be a positive integer")


def initialize_count_log(
    path,
    global_iteration,
    phase,
    phase_iteration,
    gaussian_count,
    event="initial",
):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerow(
            (
                int(global_iteration),
                str(phase),
                int(phase_iteration),
                int(gaussian_count),
                str(event),
            )
        )


def append_count_log(
    path,
    global_iteration,
    phase,
    phase_iteration,
    gaussian_count,
    event,
):
    if not path:
        return
    with open(path, "a", newline="") as handle:
        csv.writer(handle).writerow(
            (
                int(global_iteration),
                str(phase),
                int(phase_iteration),
                int(gaussian_count),
                str(event),
            )
        )
