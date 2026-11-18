from typing import Tuple


def _require_positive_integer(name, value):
    if type(value) is not int:
        raise TypeError("{} must be an integer".format(name))
    if value <= 0:
        raise ValueError("{} must be > 0".format(name))


def detail_hr_topology_events(
    hr_iteration: int,
    opacity_reset_interval: int,
) -> Tuple[str, ...]:
    _require_positive_integer("hr_iteration", hr_iteration)
    _require_positive_integer(
        "opacity_reset_interval",
        opacity_reset_interval,
    )

    events = []
    if hr_iteration <= 10_000:
        events.append("collect_stats")

    if (
        1_000 < hr_iteration <= 7_000
        and hr_iteration % 100 == 0
    ):
        events.append("detail_densify")
    elif (
        7_000 < hr_iteration <= 10_000
        and hr_iteration % 200 == 0
    ):
        events.append("detail_densify")

    if (
        hr_iteration <= 10_000
        and hr_iteration % opacity_reset_interval == 0
    ):
        events.append("opacity_reset")

    if (
        15_000 < hr_iteration < 30_000
        and hr_iteration % 3_000 == 0
    ):
        events.append("final_vcp")

    return tuple(events)
