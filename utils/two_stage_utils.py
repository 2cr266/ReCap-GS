import hashlib
import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StageClock:
    phase: str
    global_iteration: int
    hr_iteration: Optional[int]


@dataclass(frozen=True)
class TopologyConfig:
    densify_from_iteration: int = 500
    densify_until_iteration: int = 15_000
    lr_densify_until_iteration: int = 8_000
    densification_interval: int = 100
    opacity_reset_interval: int = 3_000
    final_vcp_from_iteration: int = 15_000
    final_vcp_until_iteration: int = 30_000
    final_vcp_interval: int = 3_000


@dataclass(frozen=True)
class StageState:
    global_iteration: int
    phase: str
    hr_switch_initialized: bool

@dataclass(frozen=True)
class StageTransition:
    state: StageState
    reset_hr_statistics: bool
    rebuild_camera_stack: bool


@dataclass(frozen=True)
class MultiScaleCameraPair:
    lr: object
    hr: object


SCALE_DEPENDENT_STAT_NAMES = (
    "xyz_gradient_accum",
    "xyz_gradient_accum_abs",
    "denom",
    "max_radii2D",
)

CHECKPOINT_SCHEMA_VERSION = 3
MAIN_OPTIMIZER_GROUPS = (
    "xyz",
    "f_dc",
    "opacity",
    "scaling",
    "rotation",
)
SH_OPTIMIZER_GROUPS = ("f_rest",)
OPTIMIZER_SIGNATURE_KEYS = (
    "position_lr_init",
    "position_lr_final",
    "position_lr_delay_mult",
    "position_lr_max_steps",
    "lowfeature_lr",
    "highfeature_lr",
    "opacity_lr",
    "scaling_lr",
    "rotation_lr",
)


def clock_for_iteration(global_iteration, switch_iteration):
    if global_iteration < 1:
        raise ValueError("global_iteration must be >= 1")
    if switch_iteration < 1:
        raise ValueError("switch_iteration must be >= 1")
    if global_iteration <= switch_iteration:
        return StageClock("lr", global_iteration, None)
    return StageClock("hr", global_iteration, global_iteration - switch_iteration)


def optimizer_schedule_iteration(clock):
    if clock.phase == "lr":
        return clock.global_iteration
    if clock.phase == "hr" and clock.hr_iteration is not None:
        return clock.hr_iteration
    raise ValueError("Invalid stage clock for optimizer scheduling")


def validate_stage_state(state, switch_iteration, total_iterations):
    if state.global_iteration < 0 or state.global_iteration > total_iterations:
        raise ValueError("checkpoint global_iteration is out of range")
    expected_phase = (
        "lr" if state.global_iteration <= switch_iteration else "hr"
    )
    if state.phase != expected_phase:
        raise ValueError("checkpoint phase does not match global_iteration")
    expected_initialized = state.global_iteration > switch_iteration
    if state.hr_switch_initialized != expected_initialized:
        raise ValueError(
            "checkpoint hr_switch_initialized does not match global_iteration"
        )
    return state


def advance_stage_state(
    state,
    next_global_iteration,
    switch_iteration,
    total_iterations,
):
    validate_stage_state(state, switch_iteration, total_iterations)
    if next_global_iteration != state.global_iteration + 1:
        raise ValueError("next_global_iteration must advance by exactly one")
    if next_global_iteration > total_iterations:
        raise ValueError("next_global_iteration exceeds total_iterations")

    clock = clock_for_iteration(next_global_iteration, switch_iteration)
    reset = clock.phase == "hr" and not state.hr_switch_initialized
    next_state = StageState(
        next_global_iteration,
        clock.phase,
        state.hr_switch_initialized or reset,
    )
    return StageTransition(next_state, reset, reset)


def reset_scale_dependent_statistics(gaussians):
    for name in SCALE_DEPENDENT_STAT_NAMES:
        if not hasattr(gaussians, name):
            raise AttributeError(
                "Gaussian model is missing scale statistic: {}".format(name)
            )
        value = getattr(gaussians, name)
        if not hasattr(value, "zero_"):
            raise TypeError(
                "Scale statistic {} does not support zero_()".format(name)
            )
        value.zero_()
    return SCALE_DEPENDENT_STAT_NAMES


def _camera_map(cameras, label):
    result = {}
    for camera in cameras:
        if camera.image_name in result:
            raise ValueError(
                "Duplicate {} camera name: {}".format(
                    label,
                    camera.image_name,
                )
            )
        result[camera.image_name] = camera
    return result


def cameras_by_name_order(
    cameras,
    ordered_names,
    label,
    require_all,
):
    camera_map = _camera_map(cameras, label)
    ordered_names = list(ordered_names)
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("{} contains duplicate Camera names".format(label))
    unknown = sorted(set(ordered_names) - set(camera_map))
    if unknown:
        raise ValueError(
            "{} contains unknown Camera names: {}".format(label, unknown)
        )
    if require_all and set(ordered_names) != set(camera_map):
        missing = sorted(set(camera_map) - set(ordered_names))
        raise ValueError(
            "{} is missing Camera names: {}".format(label, missing)
        )
    return [camera_map[name] for name in ordered_names]


def _comparable_camera_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_comparable_camera_value(item) for item in value)
    return value


def _camera_geometry(camera):
    required = ("R", "T", "FoVx", "FoVy")
    missing = [name for name in required if not hasattr(camera, name)]
    if missing:
        raise ValueError(
            "Camera geometry fields are missing: {}".format(missing)
        )
    return tuple(
        _comparable_camera_value(getattr(camera, name))
        for name in required
    )


def pair_multiscale_cameras(lr_cameras, hr_cameras, scale):
    if scale != 4:
        raise ValueError("Experiment 2 requires scale=4")
    lr_by_name = _camera_map(lr_cameras, "LR")
    hr_by_name = _camera_map(hr_cameras, "HR")
    if set(lr_by_name) != set(hr_by_name):
        missing_lr = sorted(set(hr_by_name) - set(lr_by_name))
        missing_hr = sorted(set(lr_by_name) - set(hr_by_name))
        raise ValueError(
            "LR/HR camera names differ: missing_lr={} missing_hr={}".format(
                missing_lr,
                missing_hr,
            )
        )

    pairs = []
    for hr_camera in hr_cameras:
        lr_camera = lr_by_name[hr_camera.image_name]
        if lr_camera.colmap_id != hr_camera.colmap_id:
            raise ValueError(
                "LR/HR COLMAP id mismatch for {}: lr={} hr={}".format(
                    hr_camera.image_name,
                    lr_camera.colmap_id,
                    hr_camera.colmap_id,
                )
            )
        if lr_camera.uid != hr_camera.uid:
            raise ValueError(
                "LR/HR camera uid mismatch for {}: lr={} hr={}".format(
                    hr_camera.image_name,
                    lr_camera.uid,
                    hr_camera.uid,
                )
            )
        if _camera_geometry(lr_camera) != _camera_geometry(hr_camera):
            raise ValueError(
                "LR/HR Camera geometry mismatch for {}".format(
                    hr_camera.image_name
                )
            )
        expected = (
            lr_camera.image_width * scale,
            lr_camera.image_height * scale,
        )
        actual = (hr_camera.image_width, hr_camera.image_height)
        if actual != expected:
            raise ValueError(
                "Pseudo-HR must be exactly {}x LR for {}: "
                "expected={} actual={}".format(
                    scale,
                    hr_camera.image_name,
                    expected,
                    actual,
                )
            )
        pairs.append(MultiScaleCameraPair(lr_camera, hr_camera))
    return pairs


def canonical_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_optimizer_signature(optimizer_type, values):
    if optimizer_type != "default":
        raise ValueError(
            "Experiment 2 requires optimizer_type=default "
            "for resumable checkpoints"
        )
    missing = [
        name for name in OPTIMIZER_SIGNATURE_KEYS if name not in values
    ]
    if missing:
        raise ValueError(
            "Missing optimizer signature values: {}".format(missing)
        )
    return {
        "optimizer_type": optimizer_type,
        "main_groups": list(MAIN_OPTIMIZER_GROUPS),
        "sh_groups": list(SH_OPTIMIZER_GROUPS),
        "hyperparameters": {
            name: values[name] for name in OPTIMIZER_SIGNATURE_KEYS
        },
    }


def build_checkpoint_metadata(
    stage_state,
    stage_config,
    optimizer_signature,
    manifest_hash,
):
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "global_iteration": stage_state.global_iteration,
        "stage_state": {
            "global_iteration": stage_state.global_iteration,
            "phase": stage_state.phase,
            "hr_switch_initialized": (
                stage_state.hr_switch_initialized
            ),
        },
        "stage_config": dict(stage_config),
        "optimizer_signature": optimizer_signature,
        "dataset_manifest_hash": manifest_hash,
    }


def validate_checkpoint_metadata(
    metadata,
    expected_stage_config,
    expected_signature,
    expected_manifest_hash,
):
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    required_metadata = {
        "schema_version",
        "global_iteration",
        "stage_state",
        "stage_config",
        "optimizer_signature",
        "dataset_manifest_hash",
    }
    missing_metadata = sorted(required_metadata - set(metadata))
    if missing_metadata:
        raise ValueError(
            "Checkpoint metadata fields are missing: {}".format(
                missing_metadata
            )
        )
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    if metadata.get("dataset_manifest_hash") != expected_manifest_hash:
        raise ValueError("Checkpoint dataset manifest hash mismatch")
    if metadata.get("optimizer_signature") != expected_signature:
        raise ValueError("Checkpoint optimizer signature mismatch")
    if metadata.get("stage_config") != expected_stage_config:
        raise ValueError("Checkpoint stage config mismatch")

    raw_state = metadata["stage_state"]
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint stage_state must be a dictionary")
    required_state = {
        "global_iteration",
        "phase",
        "hr_switch_initialized",
    }
    missing_state = sorted(required_state - set(raw_state))
    if missing_state:
        raise ValueError(
            "Checkpoint stage_state fields are missing: {}".format(
                missing_state
            )
        )
    if type(raw_state["global_iteration"]) is not int:
        raise ValueError("Checkpoint global_iteration must be an integer")
    if not isinstance(raw_state["phase"], str):
        raise ValueError("Checkpoint phase must be a string")
    if type(raw_state["hr_switch_initialized"]) is not bool:
        raise ValueError(
            "Checkpoint hr_switch_initialized must be a boolean"
        )
    state = StageState(
        raw_state["global_iteration"],
        raw_state["phase"],
        raw_state["hr_switch_initialized"],
    )
    if metadata.get("global_iteration") != state.global_iteration:
        raise ValueError("Checkpoint global iteration mismatch")
    return validate_stage_state(
        state,
        expected_stage_config["switch_iteration"],
        expected_stage_config["total_iterations"],
    )


def _validate_topology_config(config):
    if config.densify_from_iteration < 0:
        raise ValueError("densify_from_iteration must be >= 0")
    if config.densify_until_iteration <= config.densify_from_iteration:
        raise ValueError(
            "densify_until_iteration must exceed densify_from_iteration"
        )
    if config.lr_densify_until_iteration <= config.densify_from_iteration:
        raise ValueError(
            "lr_densify_until_iteration must exceed densify_from_iteration"
        )
    if config.densification_interval < 1:
        raise ValueError("densification_interval must be >= 1")
    if config.opacity_reset_interval < 1:
        raise ValueError("opacity_reset_interval must be >= 1")
    if config.final_vcp_interval < 1:
        raise ValueError("final_vcp_interval must be >= 1")


def topology_events(clock, config, white_background=False):
    _validate_topology_config(config)
    topology_iteration = (
        clock.global_iteration
        if clock.phase == "lr"
        else clock.hr_iteration
    )
    collect_stats = (
        clock.phase == "lr"
        and topology_iteration <= config.lr_densify_until_iteration
    ) or (
        clock.phase == "hr"
        and topology_iteration < config.densify_until_iteration
    )

    events = []
    if collect_stats:
        events.append("collect_stats")

    in_densify_window = (
        topology_iteration > config.densify_from_iteration
        and (
            (
                clock.phase == "lr"
                and topology_iteration <= config.lr_densify_until_iteration
            )
            or (
                clock.phase == "hr"
                and topology_iteration < config.densify_until_iteration
            )
        )
    )
    if (
        in_densify_window
        and topology_iteration % config.densification_interval == 0
    ):
        events.append("densify")

    if collect_stats and (
        topology_iteration % config.opacity_reset_interval == 0
        or (
            white_background
            and topology_iteration == config.densify_from_iteration
        )
    ):
        events.append("opacity_reset")

    if (
        clock.phase == "hr"
        and topology_iteration > config.final_vcp_from_iteration
        and topology_iteration < config.final_vcp_until_iteration
        and topology_iteration % config.final_vcp_interval == 0
    ):
        events.append("final_vcp")

    return tuple(events)
