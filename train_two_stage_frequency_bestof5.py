#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import copy
import gc
import json
import os
import random
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from random import randint

import torch
from fused_ssim import fused_ssim as fast_ssim
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui_ws, render_fastgs
from scene import Scene
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.detail_gaussian_model import DetailGaussianModel
from scripts.check_two_stage_inputs import (
    canonical_sha256,
    normalize_scene_relative_directory,
    validate_scene,
)
from utils.camera_utils import cameraList_from_camInfos
from utils.detail_densification_utils import (
    DetailDensifyConfig,
    build_detail_need_mask,
    sobel_magnitude,
)
from utils.detail_fast_utils import compute_gaussian_score_fastgs_detail
from utils.detail_two_stage_utils import detail_hr_topology_events
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.frequency_loss_utils_bestof5 import (
    FrequencyLossConfig,
    FrequencyTargetCache,
    compute_frequency_loss,
)
from utils.general_utils import safe_state
from utils.gaussian_count_logger import (
    append_count_log,
    initialize_count_log,
    validate_count_interval,
)
from utils.loss_utils import l1_loss
from utils.synthetic_two_stage_utils import (
    is_synthetic_scene,
    load_synthetic_camera_set,
    validate_synthetic_scene,
)
from utils.two_stage_utils import (
    OPTIMIZER_SIGNATURE_KEYS,
    StageState,
    TopologyConfig,
    advance_stage_state,
    build_checkpoint_metadata,
    build_optimizer_signature,
    cameras_by_name_order,
    clock_for_iteration,
    optimizer_schedule_iteration,
    pair_multiscale_cameras,
    reset_scale_dependent_statistics,
    topology_events,
    validate_checkpoint_metadata,
)
from utils.x8_scale_utils import (
    pair_multiscale_cameras_x8,
    validate_colmap_scene_x8,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


DETAIL_EXPERIMENT_ID = "two_stage_detail_guided_densification_v2"
DETAIL_SCHEDULE_VERSION = "detail_hr_schedule_v1"
DETAIL_STATE_SCHEMA_VERSION = 2
FREQUENCY_EXPERIMENT_ID = (
    "two_stage_detail_frequency_decoupled_4a_bestof5_memory_safe_v1"
)
FREQUENCY_LOSS_SCHEMA_VERSION = 1


def release_frequency_target_cache(frequency_target_cache, stage):
    """Release CPU target tensors after the final parameter update."""
    released_entries = frequency_target_cache.clear()
    collected_objects = gc.collect()
    torch.cuda.empty_cache()
    print(
        "[Best-of-5 memory release] stage={} released_frequency_targets={} "
        "gc_collected={}".format(
            stage,
            released_entries,
            collected_objects,
        )
    )
    return released_entries


def sample_fastgs_scoring_cameras(cameras):
    """Use all cameras when a sparse protocol provides fewer than ten."""
    cameras = list(cameras)
    if not cameras:
        raise ValueError("FastGS scoring requires at least one camera")
    if len(cameras) < 10:
        return cameras
    return sampling_cameras(cameras)


def _add_tensorboard_histogram(writer, tag, values, iteration):
    """Write a histogram without letting one known NumPy mismatch stop training."""
    try:
        writer.add_histogram(tag, values, iteration)
    except TypeError as error:
        message = str(error)
        known_numpy_ufunc_error = (
            "No loop matching the specified signature" in message
            and "ufunc greater" in message
        )
        if not known_numpy_ufunc_error:
            raise

        warning_attribute = "_fastgs_numpy_histogram_warning_shown"
        if not getattr(writer, warning_attribute, False):
            print(
                "[TensorBoard warning] Histogram logging is disabled because "
                "the installed PyTorch TensorBoard helper is incompatible "
                "with NumPy. Scalar logging and training continue unchanged."
            )
            setattr(writer, warning_attribute, True)
        return False
    return True


def load_independent_camera_set(dataset, image_dir, reference_cameras):
    if is_synthetic_scene(dataset.source_path):
        return load_synthetic_camera_set(
            dataset,
            image_dir,
            reference_cameras,
        )
    if not os.path.exists(os.path.join(dataset.source_path, "sparse")):
        raise ValueError("Two-stage FastGS requires a COLMAP scene")

    scene_info = sceneLoadTypeCallbacks["Colmap"](
        dataset.source_path,
        image_dir,
        dataset.eval,
    )
    camera_args = copy.copy(dataset)
    camera_args.images = image_dir
    infos_by_name = {}
    for info in scene_info.train_cameras:
        if info.image_name in infos_by_name:
            raise ValueError(
                "Duplicate independent Camera name: {}".format(
                    info.image_name
                )
            )
        infos_by_name[info.image_name] = info

    reference_names = [camera.image_name for camera in reference_cameras]
    if set(infos_by_name) != set(reference_names):
        raise ValueError(
            "Independent LR train split does not match HR train split"
        )
    ordered_infos = [infos_by_name[name] for name in reference_names]
    return cameraList_from_camInfos(ordered_infos, 1.0, camera_args)


def load_sparse_view_selection(source_path, manifest_path):
    if not manifest_path:
        return None, None
    resolved = manifest_path
    if not os.path.isabs(resolved):
        resolved = os.path.join(source_path, resolved)
    resolved = os.path.realpath(resolved)
    with open(resolved, "r", encoding="utf-8") as manifest_file:
        payload = json.load(manifest_file)
    selected = payload.get("selected_train_fastgs_names")
    if not isinstance(selected, list) or len(selected) < 3:
        raise ValueError(
            "Sparse-view manifest needs at least three selected train names"
        )
    names = [str(name) for name in selected]
    if len(set(names)) != len(names):
        raise ValueError("Sparse-view manifest contains duplicate train names")
    return set(names), payload


def attach_sparse_protocol_to_preflight(preflight, sparse_protocol):
    if sparse_protocol is None:
        return preflight
    hash_payload = dict(preflight["manifest"])
    hash_payload["sparse_view_protocol"] = sparse_protocol
    preflight["sparse_view_protocol"] = sparse_protocol
    preflight["manifest_hash"] = canonical_sha256(hash_payload)
    return preflight


def build_two_stage_cameras(
    scene,
    dataset,
    lr_image_dir,
    sr_scale,
    sparse_train_names=None,
):
    hr_cameras = scene.getTrainCameras()
    if is_synthetic_scene(dataset.source_path):
        hr_cameras = load_independent_camera_set(
            dataset,
            dataset.images,
            hr_cameras,
        )
        scene.train_cameras[1.0] = hr_cameras
    lr_cameras = load_independent_camera_set(
        dataset,
        lr_image_dir,
        hr_cameras,
    )
    if sparse_train_names is not None:
        available = {camera.image_name for camera in hr_cameras}
        missing = sorted(sparse_train_names - available)
        if missing:
            raise ValueError(
                "Sparse-view manifest names are absent from train split: {}".format(
                    missing
                )
            )
        hr_cameras = [
            camera
            for camera in hr_cameras
            if camera.image_name in sparse_train_names
        ]
        lr_cameras = [
            camera
            for camera in lr_cameras
            if camera.image_name in sparse_train_names
        ]
        scene.train_cameras[1.0] = hr_cameras
    pairs = (
        pair_multiscale_cameras_x8(lr_cameras, hr_cameras)
        if sr_scale == 8
        else pair_multiscale_cameras(lr_cameras, hr_cameras, sr_scale)
    )
    minimum_views = (
        3
        if is_synthetic_scene(dataset.source_path)
        or sparse_train_names is not None
        else 10
    )
    if len(pairs) < minimum_views:
        raise ValueError(
            "FastGS multiview scoring requires at least {} train cameras".format(
                minimum_views
            )
        )
    for pair in pairs:
        lr_image = pair.lr.original_image
        hr_image = pair.hr.original_image
        if (
            not torch.is_tensor(lr_image)
            or not torch.is_tensor(hr_image)
            or lr_image.ndim != 3
            or hr_image.ndim != 3
            or lr_image.shape[0] != 3
            or hr_image.shape[0] != 3
        ):
            raise ValueError(
                "LR/HR camera tensors must both have three RGB channels"
            )
        if lr_image.dtype != hr_image.dtype:
            raise ValueError("LR/HR camera tensors must use the same dtype")
        pair.hr.original_lr_image = pair.lr.original_image
    return [pair.lr for pair in pairs], [pair.hr for pair in pairs]


def build_stage_config(
    dataset,
    opt,
    pipe,
    switch_iteration,
    sr_scale,
    lr_image_dir,
    hr_image_dir,
    sparse_view_manifest=None,
):
    return {
        "switch_iteration": switch_iteration,
        "total_iterations": opt.iterations,
        "sr_scale": sr_scale,
        "lr_image_dir": lr_image_dir,
        "hr_image_dir": hr_image_dir,
        "sparse_view_manifest": sparse_view_manifest,
        "resolution": dataset.resolution,
        "white_background": dataset.white_background,
        "eval": dataset.eval,
        "sh_degree": dataset.sh_degree,
        "lambda_dssim": opt.lambda_dssim,
        "random_background": opt.random_background,
        "densify_from_iteration": opt.densify_from_iter,
        "densify_until_iteration": opt.densify_until_iter,
        "densification_interval": opt.densification_interval,
        "opacity_reset_interval": opt.opacity_reset_interval,
        "final_vcp_from_iteration": 15_000,
        "final_vcp_until_iteration": 30_000,
        "final_vcp_interval": 3_000,
        "loss_thresh": opt.loss_thresh,
        "grad_abs_thresh": opt.grad_abs_thresh,
        "grad_thresh": opt.grad_thresh,
        "dense": opt.dense,
        "mult": opt.mult,
        "separate_sh": pipe.separate_sh,
        "convert_SHs_python": pipe.convert_SHs_python,
        "compute_cov3D_python": pipe.compute_cov3D_python,
        "antialiasing": pipe.antialiasing,
    }


def optimizer_values(opt):
    return {
        name: getattr(opt, name) for name in OPTIMIZER_SIGNATURE_KEYS
    }


def _capture_optimizer_gradients(optimizer):
    gradients = {}
    for group in optimizer.param_groups:
        if len(group["params"]) != 1:
            raise ValueError(
                "Each FastGS optimizer group must contain one parameter"
            )
        name = group["name"]
        parameter = group["params"][0]
        gradients[name] = (
            None
            if parameter.grad is None
            else parameter.grad.detach().cpu().clone()
        )
    return gradients


def capture_pending_gradients(gaussians):
    if gaussians.optimizer is None or gaussians.shoptimizer is None:
        raise ValueError("FastGS optimizers are not initialized")
    return {
        "main": _capture_optimizer_gradients(gaussians.optimizer),
        "sh": _capture_optimizer_gradients(gaussians.shoptimizer),
    }


def _restore_optimizer_gradients(optimizer, saved_gradients, label):
    groups = {group["name"]: group for group in optimizer.param_groups}
    if set(groups) != set(saved_gradients):
        raise ValueError(
            "Checkpoint {} pending-gradient groups mismatch".format(label)
        )
    for name, group in groups.items():
        if len(group["params"]) != 1:
            raise ValueError(
                "Each FastGS optimizer group must contain one parameter"
            )
        parameter = group["params"][0]
        saved_gradient = saved_gradients[name]
        if saved_gradient is None:
            parameter.grad = None
            continue
        if tuple(saved_gradient.shape) != tuple(parameter.shape):
            raise ValueError(
                "Checkpoint pending gradient shape mismatch for {}".format(
                    name
                )
            )
        parameter.grad = saved_gradient.to(
            device=parameter.device,
            dtype=parameter.dtype,
        ).clone()


def restore_pending_gradients(gaussians, pending_gradients):
    if not isinstance(pending_gradients, dict):
        raise ValueError("Checkpoint pending_gradients must be a dictionary")
    if set(pending_gradients) != {"main", "sh"}:
        raise ValueError("Checkpoint pending-gradient optimizer set mismatch")
    _restore_optimizer_gradients(
        gaussians.optimizer,
        pending_gradients["main"],
        "main",
    )
    _restore_optimizer_gradients(
        gaussians.shoptimizer,
        pending_gradients["sh"],
        "sh",
    )


def capture_runtime_state(
    lr_cameras,
    hr_cameras,
    viewpoint_stack,
    background,
):
    return {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state(),
        "lr_camera_order": [camera.image_name for camera in lr_cameras],
        "hr_camera_order": [camera.image_name for camera in hr_cameras],
        "viewpoint_stack": [
            camera.image_name for camera in viewpoint_stack
        ],
        "background": background.detach().cpu().clone(),
    }


def restore_runtime_state(
    runtime_state,
    stage_state,
    lr_cameras,
    hr_cameras,
):
    required = {
        "python_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "lr_camera_order",
        "hr_camera_order",
        "viewpoint_stack",
        "background",
    }
    if not isinstance(runtime_state, dict):
        raise ValueError("Checkpoint runtime_state must be a dictionary")
    missing = sorted(required - set(runtime_state))
    if missing:
        raise ValueError(
            "Checkpoint runtime_state fields are missing: {}".format(
                missing
            )
        )
    lr_cameras = cameras_by_name_order(
        lr_cameras,
        runtime_state["lr_camera_order"],
        "checkpoint LR Camera order",
        True,
    )
    hr_cameras = cameras_by_name_order(
        hr_cameras,
        runtime_state["hr_camera_order"],
        "checkpoint HR Camera order",
        True,
    )
    active_cameras = (
        lr_cameras if stage_state.phase == "lr" else hr_cameras
    )
    viewpoint_stack = cameras_by_name_order(
        active_cameras,
        runtime_state["viewpoint_stack"],
        "checkpoint viewpoint stack",
        False,
    )
    saved_background = runtime_state["background"]
    if tuple(saved_background.shape) != (3,):
        raise ValueError("Checkpoint background must have shape (3,)")
    background = saved_background.to(
        device="cuda",
        dtype=torch.float32,
    ).clone()
    return lr_cameras, hr_cameras, viewpoint_stack, background


def activate_runtime_rng(runtime_state):
    random.setstate(runtime_state["python_random_state"])
    torch.set_rng_state(runtime_state["torch_cpu_rng_state"])
    torch.cuda.set_rng_state(runtime_state["torch_cuda_rng_state"])


def _detail_config_dictionary(detail_config):
    if not isinstance(detail_config, DetailDensifyConfig):
        raise TypeError("detail_config must be a DetailDensifyConfig")
    return asdict(detail_config)


def _frequency_config_dictionary(frequency_config):
    if not isinstance(frequency_config, FrequencyLossConfig):
        raise TypeError("frequency_config must be a FrequencyLossConfig")
    return asdict(frequency_config)


def validate_frequency_checkpoint_identity(
    payload,
    global_iteration,
    switch_iteration,
    frequency_config,
):
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    if type(global_iteration) is not int or type(switch_iteration) is not int:
        raise TypeError("Checkpoint iterations must be integers")
    frequency_experiment_id = payload.get("frequency_experiment_id")
    saved_frequency_config = payload.get("frequency_loss_config")
    frequency_schema_version = payload.get("frequency_loss_schema_version")
    if global_iteration <= switch_iteration:
        if frequency_experiment_id is not None and not isinstance(
            frequency_experiment_id,
            str,
        ):
            raise ValueError("Checkpoint frequency experiment id is invalid")
        if saved_frequency_config is not None and not isinstance(
            saved_frequency_config,
            dict,
        ):
            raise ValueError("Checkpoint frequency configuration is invalid")
        if frequency_schema_version is not None and type(
            frequency_schema_version
        ) is not int:
            raise ValueError("Checkpoint frequency loss schema is invalid")
        return
    if frequency_experiment_id != FREQUENCY_EXPERIMENT_ID:
        raise ValueError("HR checkpoint is not from this frequency experiment")
    if saved_frequency_config != _frequency_config_dictionary(
        frequency_config
    ):
        raise ValueError("Checkpoint frequency configuration mismatch")
    if frequency_schema_version != FREQUENCY_LOSS_SCHEMA_VERSION:
        raise ValueError("Checkpoint frequency loss schema mismatch")


def _validate_max_gaussians(max_gaussians):
    if type(max_gaussians) is not int:
        raise TypeError("max_gaussians must be an integer")
    if max_gaussians < 0:
        raise ValueError("max_gaussians must be >= 0")


def _validate_detail_state_payload(
    detail_state,
    expected_count,
    current_hr_iteration,
):
    if not isinstance(detail_state, dict):
        raise ValueError("Checkpoint detail_state must be a dictionary")
    required = {
        "detail_region_flag",
        "detail_extra_origin_flag",
        "detail_lineage_flag",
        "birth_hr_iteration",
    }
    if set(detail_state) != required:
        raise ValueError(
            "Checkpoint detail_state fields mismatch: {}".format(
                sorted(detail_state)
            )
        )
    detail_region_flag = detail_state["detail_region_flag"]
    detail_extra_origin_flag = detail_state["detail_extra_origin_flag"]
    detail_lineage_flag = detail_state["detail_lineage_flag"]
    birth_hr_iteration = detail_state["birth_hr_iteration"]
    detail_flags = (
        detail_region_flag,
        detail_extra_origin_flag,
        detail_lineage_flag,
    )
    if not all(torch.is_tensor(value) for value in detail_flags) or not (
        torch.is_tensor(birth_hr_iteration)
    ):
        raise ValueError("Checkpoint detail_state values must be tensors")
    if any(value.ndim != 1 for value in detail_flags) or (
        birth_hr_iteration.ndim != 1
    ):
        raise ValueError("Checkpoint detail tensors must be one-dimensional")
    if any(value.shape[0] != expected_count for value in detail_flags) or (
        birth_hr_iteration.shape[0] != expected_count
    ):
        raise ValueError(
            "Checkpoint detail metadata and Gaussian count mismatch"
        )
    for name, value in zip(
        (
            "detail_region_flag",
            "detail_extra_origin_flag",
            "detail_lineage_flag",
        ),
        detail_flags,
    ):
        if value.dtype != torch.bool:
            raise ValueError("{} must use torch.bool".format(name))
    if birth_hr_iteration.dtype != torch.int32:
        raise ValueError("birth_hr_iteration must use torch.int32")
    invalid_birth = (birth_hr_iteration < -1) | (
        birth_hr_iteration > current_hr_iteration
    )
    invalid_birth |= birth_hr_iteration == 0
    if bool(invalid_birth.any().item()):
        raise ValueError("Checkpoint birth_hr_iteration values are invalid")
    any_detail_flag = torch.logical_or(
        torch.logical_or(detail_region_flag, detail_extra_origin_flag),
        detail_lineage_flag,
    )
    if bool((any_detail_flag & (birth_hr_iteration < 1)).any().item()):
        raise ValueError("Detail Gaussians must have an HR birth iteration")
    if bool(
        (detail_extra_origin_flag & ~detail_region_flag).any().item()
    ):
        raise ValueError("detail_extra_origin_flag requires detail_region_flag")
    if bool(
        (detail_extra_origin_flag & ~detail_lineage_flag).any().item()
    ):
        raise ValueError("detail_extra_origin_flag requires detail_lineage_flag")
    return (
        detail_region_flag,
        detail_extra_origin_flag,
        detail_lineage_flag,
        birth_hr_iteration,
    )


def _validated_detail_state_tensors(
    detail_state,
    gaussians,
    current_hr_iteration,
):
    (
        detail_region_flag,
        detail_extra_origin_flag,
        detail_lineage_flag,
        birth_hr_iteration,
    ) = (
        _validate_detail_state_payload(
            detail_state,
            gaussians.get_xyz.shape[0],
            current_hr_iteration,
        )
    )
    device = gaussians.get_xyz.device
    return (
        detail_region_flag.to(device=device).clone(),
        detail_extra_origin_flag.to(device=device).clone(),
        detail_lineage_flag.to(device=device).clone(),
        birth_hr_iteration.to(device=device).clone(),
    )


def capture_detail_state(gaussians, state, switch_iteration):
    if state.global_iteration <= switch_iteration:
        return None
    current_hr_iteration = state.global_iteration - switch_iteration
    detail_state = {
        "detail_region_flag": gaussians.detail_region_flag,
        "detail_extra_origin_flag": gaussians.detail_extra_origin_flag,
        "detail_lineage_flag": gaussians.detail_lineage_flag,
        "birth_hr_iteration": gaussians.birth_hr_iteration,
    }
    region, extra_origin, lineage, births = _validated_detail_state_tensors(
        detail_state,
        gaussians,
        current_hr_iteration,
    )
    return {
        "detail_region_flag": region.detach().cpu(),
        "detail_extra_origin_flag": extra_origin.detach().cpu(),
        "detail_lineage_flag": lineage.detach().cpu(),
        "birth_hr_iteration": births.detach().cpu(),
    }


def build_checkpoint_payload(
    gaussians,
    opt,
    state,
    stage_config,
    manifest_hash,
    runtime_state,
    detail_config,
    frequency_config,
    max_gaussians,
    switch_iteration,
):
    signature = build_optimizer_signature(opt.optimizer_type, optimizer_values(opt))
    metadata = build_checkpoint_metadata(
        state,
        stage_config,
        signature,
        manifest_hash,
    )
    metadata["gaussian_capture"] = gaussians.capture(opt.optimizer_type)
    metadata["pending_gradients"] = capture_pending_gradients(gaussians)
    metadata["runtime_state"] = runtime_state
    payload = metadata
    payload["detail_experiment_id"] = DETAIL_EXPERIMENT_ID
    payload["detail_config"] = _detail_config_dictionary(detail_config)
    payload["detail_schedule_version"] = DETAIL_SCHEDULE_VERSION
    payload["detail_state_schema_version"] = DETAIL_STATE_SCHEMA_VERSION
    payload["max_gaussians"] = max_gaussians
    payload["frequency_experiment_id"] = FREQUENCY_EXPERIMENT_ID
    payload["frequency_loss_config"] = _frequency_config_dictionary(
        frequency_config
    )
    payload["frequency_loss_schema_version"] = (
        FREQUENCY_LOSS_SCHEMA_VERSION
    )
    payload["detail_state"] = capture_detail_state(
        gaussians,
        state,
        switch_iteration,
    )
    return payload


def save_checkpoint_atomic(path, payload):
    path = os.fspath(path)
    temporary = path + ".tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save_detail_metadata_sidecar(
    model_path,
    global_iteration,
    gaussians,
    switch_iteration,
):
    """Save visualization/ablation metadata beside an already-written PLY."""
    point_count = gaussians.get_xyz.shape[0]
    if global_iteration <= switch_iteration:
        detail_region_flag = None
        detail_extra_origin_flag = None
        detail_lineage_flag = None
        birth_hr_iteration = None
    else:
        current_hr_iteration = global_iteration - switch_iteration
        detail_state = {
            "detail_region_flag": gaussians.detail_region_flag,
            "detail_extra_origin_flag": gaussians.detail_extra_origin_flag,
            "detail_lineage_flag": gaussians.detail_lineage_flag,
            "birth_hr_iteration": gaussians.birth_hr_iteration,
        }
        (
            detail_region_flag,
            detail_extra_origin_flag,
            detail_lineage_flag,
            birth_hr_iteration,
        ) = _validated_detail_state_tensors(
            detail_state,
            gaussians,
            current_hr_iteration,
        )
        detail_region_flag = detail_region_flag.detach().cpu()
        detail_extra_origin_flag = detail_extra_origin_flag.detach().cpu()
        detail_lineage_flag = detail_lineage_flag.detach().cpu()
        birth_hr_iteration = birth_hr_iteration.detach().cpu()

    payload = {
        "schema_version": DETAIL_STATE_SCHEMA_VERSION,
        "experiment_id": DETAIL_EXPERIMENT_ID,
        "global_iteration": global_iteration,
        "gaussian_count": point_count,
        "detail_region_flag": detail_region_flag,
        "detail_extra_origin_flag": detail_extra_origin_flag,
        "detail_lineage_flag": detail_lineage_flag,
        "birth_hr_iteration": birth_hr_iteration,
    }
    iteration_directory = os.path.join(
        os.fspath(model_path),
        "point_cloud",
        "iteration_{}".format(global_iteration),
    )
    if not os.path.isdir(iteration_directory):
        raise RuntimeError(
            "PLY output directory is missing before detail sidecar save: {}".format(
                iteration_directory
            )
        )
    detail_metadata_sidecar = os.path.join(
        iteration_directory,
        "detail_metadata.pth",
    )
    temporary = detail_metadata_sidecar + ".tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, detail_metadata_sidecar)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return detail_metadata_sidecar


def restore_checkpoint(
    path,
    gaussians,
    opt,
    stage_config,
    manifest_hash,
    detail_config,
    frequency_config,
    max_gaussians,
    switch_iteration,
):
    payload = torch.load(path)
    if not isinstance(payload, dict):
        raise ValueError(
            "Two-stage checkpoints must use the versioned dictionary format"
        )
    signature = build_optimizer_signature(opt.optimizer_type, optimizer_values(opt))
    state = validate_checkpoint_metadata(
        payload,
        stage_config,
        signature,
        manifest_hash,
    )
    if "gaussian_capture" not in payload:
        raise ValueError("Checkpoint is missing gaussian_capture")
    if "pending_gradients" not in payload:
        raise ValueError("Checkpoint is missing pending_gradients")
    if "runtime_state" not in payload:
        raise ValueError("Checkpoint is missing runtime_state")
    detail_experiment_id = payload.get("detail_experiment_id")
    saved_detail_config = payload.get("detail_config")
    detail_state_schema_version = payload.get("detail_state_schema_version")
    detail_state = payload.get("detail_state")
    validate_frequency_checkpoint_identity(
        payload,
        state.global_iteration,
        switch_iteration,
        frequency_config,
    )
    if state.global_iteration <= switch_iteration:
        if detail_experiment_id is not None:
            if not isinstance(detail_experiment_id, str):
                raise ValueError("Checkpoint detail experiment id is invalid")
        if detail_state is not None:
            raise ValueError("LR checkpoint must not contain detail_state")
    else:
        if detail_experiment_id != DETAIL_EXPERIMENT_ID:
            raise ValueError("HR checkpoint is not from this detail experiment")
        if saved_detail_config != _detail_config_dictionary(detail_config):
            raise ValueError("Checkpoint detail configuration mismatch")
        if payload.get("detail_schedule_version") != DETAIL_SCHEDULE_VERSION:
            raise ValueError("Checkpoint detail schedule mismatch")
        if detail_state_schema_version != DETAIL_STATE_SCHEMA_VERSION:
            raise ValueError("Checkpoint detail state schema mismatch")
        if payload.get("max_gaussians") != max_gaussians:
            raise ValueError("Checkpoint max_gaussians mismatch")
        if detail_state is None:
            raise ValueError("HR checkpoint is missing detail_state")
        gaussian_capture = payload["gaussian_capture"]
        if not isinstance(gaussian_capture, (tuple, list)):
            raise ValueError("Checkpoint gaussian_capture must be a sequence")
        if len(gaussian_capture) < 2 or not torch.is_tensor(
            gaussian_capture[1]
        ):
            raise ValueError("Checkpoint gaussian_capture is missing xyz")
        captured_xyz = gaussian_capture[1]
        if captured_xyz.ndim != 2 or captured_xyz.shape[1] != 3:
            raise ValueError("Checkpoint Gaussian xyz must have shape [N,3]")
        _validate_detail_state_payload(
            detail_state,
            captured_xyz.shape[0],
            state.global_iteration - switch_iteration,
        )
    gaussians.restore(payload["gaussian_capture"], opt)
    restore_pending_gradients(gaussians, payload["pending_gradients"])
    if state.global_iteration > switch_iteration:
        region, extra_origin, lineage, births = _validated_detail_state_tensors(
            detail_state,
            gaussians,
            state.global_iteration - switch_iteration,
        )
        gaussians.detail_region_flag = region
        gaussians.detail_extra_origin_flag = extra_origin
        gaussians.detail_lineage_flag = lineage
        gaussians.birth_hr_iteration = births
    return state, payload["runtime_state"]


def _ensure_output_directory(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)

    os.makedirs(args.model_path, exist_ok=True)
    return args.model_path


def prepare_output_and_logger(args):
    _ensure_output_directory(args)
    print("Output folder: {}".format(args.model_path))
    with open(
        os.path.join(args.model_path, "cfg_args"),
        "w",
        encoding="utf-8",
    ) as cfg_log_file:
        cfg_log_file.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: not logging progress")
    return None


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(
            value,
            output_file,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        output_file.write("\n")


def _normalize_debug_map(value):
    value = value.detach().float()
    finite = torch.isfinite(value)
    if not bool(finite.any().item()):
        return torch.zeros_like(value)
    finite_values = value[finite]
    scale = torch.quantile(finite_values, 0.99)
    if float(scale.item()) <= 1e-8:
        scale = torch.max(finite_values)
    if float(scale.item()) <= 1e-8:
        return torch.zeros_like(value)
    normalized = torch.zeros_like(value)
    normalized[finite] = torch.clamp(
        finite_values / scale,
        min=0.0,
        max=1.0,
    )
    return normalized


def _save_debug_png(path, value, normalize=False):
    value = value.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if normalize:
        value = _normalize_debug_map(value)
    else:
        value = value.clamp(0.0, 1.0)
    value = value.cpu()
    if value.ndim == 2:
        array = (
            value.mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        Image.fromarray(array, mode="L").save(path)
        return
    if value.ndim == 3 and value.shape[0] >= 3:
        array = (
            value[:3]
            .permute(1, 2, 0)
            .contiguous()
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        Image.fromarray(array, mode="RGB").save(path)
        return
    raise ValueError("Unsupported debug image shape: {}".format(value.shape))


def _detail_gaussian_projection_heatmap(
    gaussians,
    camera,
    radii,
):
    height = int(camera.image_height)
    width = int(camera.image_width)
    device = gaussians.get_xyz.device
    heatmap = torch.zeros(
        (height, width),
        dtype=torch.float32,
        device=device,
    )
    point_count = gaussians.get_xyz.shape[0]
    if point_count == 0:
        return heatmap
    if radii.shape[0] != point_count:
        raise ValueError("Debug radii and Gaussian count mismatch")
    if gaussians.detail_lineage_flag.shape[0] != point_count:
        raise ValueError("Debug detail metadata and Gaussian count mismatch")

    xyz = gaussians.get_xyz
    projection = camera.full_proj_transform.to(
        device=device,
        dtype=xyz.dtype,
    )
    clip = xyz @ projection[:3, :] + projection[3, :]
    clip_w = clip[:, 3]
    valid = torch.isfinite(clip).all(dim=1)
    valid &= torch.abs(clip_w) > 1e-8
    valid &= radii.to(device=device) > 0
    valid &= gaussians.detail_lineage_flag
    safe_w = torch.where(valid, clip_w, torch.ones_like(clip_w))
    pixel_x = ((clip[:, 0] / safe_w + 1.0) * width - 1.0) * 0.5
    pixel_y = ((clip[:, 1] / safe_w + 1.0) * height - 1.0) * 0.5
    valid &= torch.isfinite(pixel_x) & torch.isfinite(pixel_y)
    valid &= (pixel_x >= 0.0) & (pixel_x < width)
    valid &= (pixel_y >= 0.0) & (pixel_y < height)
    if not bool(valid.any().item()):
        return heatmap
    x_index = pixel_x[valid].long()
    y_index = pixel_y[valid].long()
    linear_index = y_index * width + x_index
    heatmap.reshape(-1).scatter_add_(
        0,
        linear_index,
        torch.ones_like(linear_index, dtype=torch.float32),
    )
    return torch.log1p(heatmap)


@torch.no_grad()
def save_detail_debug_bundle(
    model_path,
    hr_iteration,
    camera,
    gaussians,
    pipe,
    background,
    opt,
    detail_config,
):
    debug_directory = os.path.join(
        model_path,
        "detail_debug",
        "hr_{:05d}".format(hr_iteration),
    )
    os.makedirs(debug_directory, exist_ok=True)
    debug_package = render_fastgs(
        camera,
        gaussians,
        pipe,
        background,
        opt.mult,
    )
    rendered = debug_package["render"].detach().clamp(0.0, 1.0)
    target = camera.original_image.cuda().detach().clamp(0.0, 1.0)
    if rendered.shape != target.shape:
        raise ValueError(
            "Debug render/target mismatch: {} vs {}".format(
                tuple(rendered.shape),
                tuple(target.shape),
            )
        )
    sobel = sobel_magnitude(target)
    error = torch.mean(torch.abs(rendered - target), dim=0)
    detail_mask, statistics = build_detail_need_mask(
        rendered,
        target,
        detail_config,
    )
    _save_debug_png(
        os.path.join(debug_directory, "pseudo_hr.png"),
        target,
    )
    _save_debug_png(
        os.path.join(debug_directory, "render.png"),
        rendered,
    )
    _save_debug_png(
        os.path.join(debug_directory, "sobel.png"),
        sobel,
        normalize=True,
    )
    _save_debug_png(
        os.path.join(debug_directory, "error.png"),
        error,
    )
    _save_debug_png(
        os.path.join(debug_directory, "detail_mask.png"),
        detail_mask.float(),
    )
    debug_radii = debug_package["radii"]
    del sobel, error, detail_mask, rendered, target, debug_package
    projection_heatmap = _detail_gaussian_projection_heatmap(
        gaussians,
        camera,
        debug_radii,
    )
    _save_debug_png(
        os.path.join(
            debug_directory,
            "detail_gaussian_projection.png",
        ),
        projection_heatmap,
        normalize=True,
    )
    return statistics


def _signed_residual_debug_image(value):
    value = value.detach().float()
    finite = torch.isfinite(value)
    if not bool(finite.any().item()):
        return torch.full_like(value, 0.5)
    magnitude = torch.abs(value[finite])
    scale = torch.quantile(magnitude, 0.99)
    if float(scale.item()) <= 1e-8:
        scale = torch.max(magnitude)
    if float(scale.item()) <= 1e-8:
        return torch.full_like(value, 0.5)
    result = torch.full_like(value, 0.5)
    result[finite] = torch.clamp(
        0.5 + 0.5 * value[finite] / scale,
        min=0.0,
        max=1.0,
    )
    return result


@torch.no_grad()
def save_frequency_debug_bundle(
    model_path,
    hr_iteration,
    camera,
    gaussians,
    pipe,
    background,
    opt,
    frequency_target_cache,
    frequency_config,
):
    if not hasattr(camera, "original_lr_image"):
        raise ValueError(
            "Frequency debug camera is missing its paired real-LR image"
        )
    debug_directory = os.path.join(
        model_path,
        "frequency_debug",
        "hr_{:05d}".format(hr_iteration),
    )
    os.makedirs(debug_directory, exist_ok=True)
    debug_package = render_fastgs(
        camera,
        gaussians,
        pipe,
        background,
        opt.mult,
    )
    render_hr = debug_package["render"]
    target_entry = frequency_target_cache.get(
        camera.image_name,
        camera.original_lr_image,
        camera.original_image,
    )
    target_lr = camera.original_lr_image.to(
        device=render_hr.device,
        dtype=render_hr.dtype,
        non_blocking=True,
    )
    target_sr = camera.original_image.to(
        device=render_hr.device,
        dtype=render_hr.dtype,
        non_blocking=True,
    )
    result = compute_frequency_loss(
        render_hr,
        target_lr,
        target_sr,
        target_entry,
        hr_iteration=hr_iteration,
        config=frequency_config,
        ssim_fn=fast_ssim,
    )
    residual_error = torch.mean(
        torch.abs(result.render_residual - result.target_residual),
        dim=0,
    )

    _save_debug_png(
        os.path.join(debug_directory, "01_real_lr.png"),
        target_lr,
    )
    _save_debug_png(
        os.path.join(debug_directory, "02_bicubic_lr.png"),
        target_sr - result.target_residual,
    )
    _save_debug_png(
        os.path.join(debug_directory, "03_pseudo_hr.png"),
        target_sr,
    )
    _save_debug_png(
        os.path.join(debug_directory, "04_target_residual.png"),
        _signed_residual_debug_image(result.target_residual),
    )
    _save_debug_png(
        os.path.join(debug_directory, "05_render_hr.png"),
        render_hr,
    )
    _save_debug_png(
        os.path.join(debug_directory, "06_render_residual.png"),
        _signed_residual_debug_image(result.render_residual),
    )
    _save_debug_png(
        os.path.join(debug_directory, "07_hf_mask.png"),
        result.hf_mask.float(),
    )
    _save_debug_png(
        os.path.join(debug_directory, "08_residual_error_heatmap.png"),
        residual_error,
        normalize=True,
    )
    _save_debug_png(
        os.path.join(debug_directory, "09_render_lr.png"),
        result.render_lr,
    )
    return {
        "hf_mask_ratio": result.mask_ratio,
        "target_residual_mean": result.target_residual_mean,
        "render_residual_mean": float(result.render_residual_mean.item()),
        "lr_psnr": float(result.lr_psnr.item()),
    }


def _validate_training_arguments(
    dataset,
    opt,
    pipe,
    switch_iteration,
    sr_scale,
):
    if switch_iteration < 1 or switch_iteration >= opt.iterations:
        raise ValueError(
            "switch_iteration must satisfy 1 <= switch_iteration < iterations"
        )
    if sr_scale not in (4, 8):
        raise ValueError("Two-stage training requires sr_scale=4 or sr_scale=8")
    if dataset.resolution != 1:
        raise ValueError(
            "Experiment 2 requires --resolution 1 for native LR/HR sizes"
        )
    if pipe.convert_SHs_python:
        raise ValueError(
            "FastGS render_fastgs does not support --convert_SHs_python"
        )
    build_optimizer_signature(opt.optimizer_type, optimizer_values(opt))
    validation_topology = TopologyConfig(
        densify_from_iteration=opt.densify_from_iter,
        densify_until_iteration=opt.densify_until_iter,
        lr_densify_until_iteration=switch_iteration,
        densification_interval=opt.densification_interval,
        opacity_reset_interval=opt.opacity_reset_interval,
    )
    topology_events(
        clock_for_iteration(1, switch_iteration),
        validation_topology,
        dataset.white_background,
    )


def training_two_stage_frequency(
    dataset,
    opt,
    pipe,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    websockets,
    switch_iteration,
    lr_image_dir,
    hr_image_dir,
    sr_scale,
    detail_config,
    frequency_config,
    max_gaussians,
    sparse_view_manifest,
    gaussian_count_log=None,
    gaussian_count_interval=100,
    gaussian_count_only=False,
):
    _validate_training_arguments(
        dataset,
        opt,
        pipe,
        switch_iteration,
        sr_scale,
    )
    _detail_config_dictionary(detail_config)
    _frequency_config_dictionary(frequency_config)
    _validate_max_gaussians(max_gaussians)
    validate_count_interval(gaussian_count_interval)
    lr_image_dir, _ = normalize_scene_relative_directory(
        dataset.source_path,
        lr_image_dir,
        "LR",
    )
    hr_image_dir, _ = normalize_scene_relative_directory(
        dataset.source_path,
        hr_image_dir,
        "HR",
    )
    sparse_train_names, sparse_protocol = load_sparse_view_selection(
        dataset.source_path,
        sparse_view_manifest,
    )
    stage_config = build_stage_config(
        dataset,
        opt,
        pipe,
        switch_iteration,
        sr_scale,
        lr_image_dir,
        hr_image_dir,
        sparse_view_manifest,
    )
    optimizer_signature = build_optimizer_signature(
        opt.optimizer_type,
        optimizer_values(opt),
    )
    synthetic_scene = is_synthetic_scene(dataset.source_path)
    point_cloud_path = (
        os.path.join(dataset.source_path, "points3d.ply")
        if synthetic_scene
        else os.path.join(
            dataset.source_path,
            "sparse",
            "0",
            "points3D.ply",
        )
    )
    point_cloud_was_present = os.path.isfile(point_cloud_path)
    if synthetic_scene:
        validator = validate_synthetic_scene
    elif sr_scale == 8:
        validator = validate_colmap_scene_x8
    else:
        validator = validate_scene
    preflight = validator(
        dataset.source_path,
        lr_image_dir,
        hr_image_dir,
        sr_scale,
        dataset.eval,
    )
    preflight = attach_sparse_protocol_to_preflight(
        preflight,
        sparse_protocol,
    )
    dataset.images = hr_image_dir
    _ensure_output_directory(dataset)
    gaussians = DetailGaussianModel(
        dataset.sh_degree,
        opt.optimizer_type,
    )
    gaussians.tmp_radii = None
    scene = Scene(dataset, gaussians)
    if not point_cloud_was_present and os.path.isfile(point_cloud_path):
        preflight = validator(
            dataset.source_path,
            lr_image_dir,
            hr_image_dir,
            sr_scale,
            dataset.eval,
        )
        preflight = attach_sparse_protocol_to_preflight(
            preflight,
            sparse_protocol,
        )
    manifest_hash = preflight["manifest_hash"]
    lr_cameras, hr_cameras = build_two_stage_cameras(
        scene,
        dataset,
        lr_image_dir,
        sr_scale,
        sparse_train_names,
    )

    restored_runtime_state = None
    if checkpoint:
        state, restored_runtime_state = restore_checkpoint(
            checkpoint,
            gaussians,
            opt,
            stage_config,
            manifest_hash,
            detail_config,
            frequency_config,
            max_gaussians,
            switch_iteration,
        )
        print(
            "Restored two-stage checkpoint at global iteration {}".format(
                state.global_iteration
            )
        )
    else:
        gaussians.training_setup(opt)
        state = StageState(0, "lr", False)

    tensorboard_writer = prepare_output_and_logger(dataset)
    initial_phase_iteration = (
        state.global_iteration
        if state.global_iteration <= switch_iteration
        else state.global_iteration - switch_iteration
    )
    initialize_count_log(
        gaussian_count_log,
        state.global_iteration,
        state.phase,
        initial_phase_iteration,
        gaussians.get_xyz.shape[0],
        "initial" if state.global_iteration == 0 else "resume",
    )
    _write_json(
        os.path.join(dataset.model_path, "two_stage_config.json"),
        {
            "stage_config": stage_config,
            "optimizer_signature": optimizer_signature,
            "dataset_manifest_hash": manifest_hash,
            "view_count": preflight["view_count"],
            "train_count": preflight["train_count"],
            "test_count": preflight["test_count"],
            "lr_sizes": preflight["lr_sizes"],
            "hr_sizes": preflight["hr_sizes"],
            "detail_experiment_id": DETAIL_EXPERIMENT_ID,
            "detail_schedule_version": DETAIL_SCHEDULE_VERSION,
            "detail_state_schema_version": DETAIL_STATE_SCHEMA_VERSION,
            "detail_config": _detail_config_dictionary(detail_config),
            "max_gaussians": max_gaussians,
            "frequency_experiment_id": FREQUENCY_EXPERIMENT_ID,
            "frequency_loss_schema_version": FREQUENCY_LOSS_SCHEMA_VERSION,
            "frequency_loss_config": _frequency_config_dictionary(
                frequency_config
            ),
        },
    )
    _write_json(
        os.path.join(dataset.model_path, "two_stage_manifest.json"),
        preflight,
    )

    topology_config = TopologyConfig(
        densify_from_iteration=opt.densify_from_iter,
        densify_until_iteration=opt.densify_until_iter,
        lr_densify_until_iteration=switch_iteration,
        densification_interval=opt.densification_interval,
        opacity_reset_interval=opt.opacity_reset_interval,
        final_vcp_from_iteration=15_000,
        final_vcp_until_iteration=30_000,
        final_vcp_interval=3_000,
    )

    background_color = (
        [1, 1, 1] if dataset.white_background else [0, 0, 0]
    )
    background = torch.tensor(
        background_color,
        dtype=torch.float32,
        device="cuda",
    )
    if restored_runtime_state is None:
        bg = (
            torch.rand((3), device="cuda")
            if opt.random_background
            else background
        )
        viewpoint_stack = []
    else:
        (
            lr_cameras,
            hr_cameras,
            viewpoint_stack,
            bg,
        ) = restore_runtime_state(
            restored_runtime_state,
            state,
            lr_cameras,
            hr_cameras,
        )
    viewpoint_indices = []
    if viewpoint_stack:
        viewpoint_indices = list(range(len(viewpoint_stack)))
    phase_time_ms = {"lr": 0.0, "hr": 0.0}
    detail_scoring_time_total_ms = 0.0
    ema_loss_for_log = 0.0
    remaining_iterations = opt.iterations - state.global_iteration
    progress_bar = tqdm(
        total=remaining_iterations,
        desc="Two-stage training progress",
    )
    iteration_start = torch.cuda.Event(enable_timing=True)
    iteration_end = torch.cuda.Event(enable_timing=True)
    if restored_runtime_state is not None:
        activate_runtime_rng(restored_runtime_state)
    fixed_debug_camera = min(hr_cameras, key=lambda camera: camera.image_name)
    frequency_target_cache = FrequencyTargetCache(frequency_config)

    for iteration in range(
        state.global_iteration + 1,
        opt.iterations + 1,
    ):
        transition = advance_stage_state(
            state,
            iteration,
            switch_iteration,
            opt.iterations,
        )
        state = transition.state
        clock = clock_for_iteration(iteration, switch_iteration)
        active_cameras = (
            lr_cameras if clock.phase == "lr" else hr_cameras
        )

        if transition.rebuild_camera_stack:
            viewpoint_stack = []
            viewpoint_indices = []
        if transition.reset_hr_statistics:
            reset_names = reset_scale_dependent_statistics(gaussians)
            gaussians.initialize_detail_tracking()
            torch.cuda.empty_cache()
            print(
                "[Stage switch] global={} hr_iter=1 reset={}".format(
                    iteration,
                    ",".join(reset_names),
                )
            )

        iteration_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1_000 == 0:
            gaussians.oneupSHdegree()

        if websockets:
            camera_index = network_gui_ws.curr_id
            if 0 <= camera_index < len(active_cameras):
                gui_camera = active_cameras[camera_index]
                gui_image = render_fastgs(
                    gui_camera,
                    gaussians,
                    pipe,
                    background,
                    opt.mult,
                    1.0,
                )["render"]
                network_gui_ws.latest_width = gui_camera.image_width
                network_gui_ws.latest_height = gui_camera.image_height
                network_gui_ws.latest_result = memoryview(
                    (
                        torch.clamp(gui_image, min=0, max=1.0) * 255
                    )
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )

        if not viewpoint_stack:
            viewpoint_stack = active_cameras.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        random_index = randint(0, len(viewpoint_indices) - 1)
        viewpoint_camera = viewpoint_stack.pop(random_index)
        viewpoint_indices.pop(random_index)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_package = render_fastgs(
            viewpoint_camera,
            gaussians,
            pipe,
            bg,
            opt.mult,
        )
        image = render_package["render"]
        viewspace_point_tensor = render_package["viewspace_points"]
        visibility_filter = render_package["visibility_filter"]
        radii = render_package["radii"]
        target_image = viewpoint_camera.original_image.cuda()
        if image.shape != target_image.shape:
            raise ValueError(
                "{} render/target shape mismatch for {}: {} vs {}".format(
                    clock.phase,
                    viewpoint_camera.image_name,
                    tuple(image.shape),
                    tuple(target_image.shape),
                )
            )

        frequency_result = None
        if clock.phase == "lr":
            loss_l1 = l1_loss(image, target_image)
            ssim_value = fast_ssim(
                image.unsqueeze(0),
                target_image.unsqueeze(0),
            )
            loss_dssim = 1.0 - ssim_value
            loss = (
                (1.0 - opt.lambda_dssim) * loss_l1
                + opt.lambda_dssim * loss_dssim
            )
        else:
            if not hasattr(viewpoint_camera, "original_lr_image"):
                raise ValueError(
                    "HR camera {} is missing its paired real-LR image".format(
                        viewpoint_camera.image_name
                    )
                )
            target_entry = frequency_target_cache.get(
                viewpoint_camera.image_name,
                viewpoint_camera.original_lr_image,
                viewpoint_camera.original_image,
            )
            target_lr = viewpoint_camera.original_lr_image.to(
                device=image.device,
                dtype=image.dtype,
                non_blocking=True,
            )
            frequency_result = compute_frequency_loss(
                image,
                target_lr,
                target_image,
                target_entry,
                hr_iteration=clock.hr_iteration,
                config=frequency_config,
                ssim_fn=fast_ssim,
            )
            loss_l1 = frequency_result.rgb_l1
            loss_dssim = frequency_result.rgb_dssim
            loss = frequency_result.total
        loss.backward()

        topology_diagnostics = {
            "normal_clone_candidates": 0,
            "detail_clone_candidates": 0,
            "normal_detail_clone_overlap_count": 0,
            "detail_only_clone_candidates": 0,
            "selected_detail_only_clone_count": 0,
            "normal_split_candidates": 0,
            "detail_split_candidates": 0,
            "normal_detail_split_overlap_count": 0,
            "detail_only_split_candidates": 0,
            "selected_detail_only_split_count": 0,
            "selected_clone_count": 0,
            "selected_split_count": 0,
            "detail_extra_children_count": 0,
            "detail_extra_net_gaussian_count": 0,
            "detail_region_alive_count": 0,
            "detail_extra_origin_alive_count": 0,
            "detail_lineage_alive_count": 0,
            "ordinary_pruned_count": 0,
            "final_vcp_removed_count": 0,
        }
        if clock.phase == "lr":
            events = topology_events(
                clock,
                topology_config,
                dataset.white_background,
            )
            with torch.no_grad():
                for event in events:
                    if event == "collect_stats":
                        gaussians.max_radii2D[visibility_filter] = torch.max(
                            gaussians.max_radii2D[visibility_filter],
                            radii[visibility_filter],
                        )
                        gaussians.add_densification_stats(
                            viewspace_point_tensor,
                            visibility_filter,
                        )
                    elif event == "densify":
                        size_threshold = (
                            20
                            if iteration > opt.opacity_reset_interval
                            else None
                        )
                        scoring_cameras = sample_fastgs_scoring_cameras(
                            active_cameras.copy()
                        )
                        importance_score, pruning_score = (
                            compute_gaussian_score_fastgs(
                                scoring_cameras,
                                gaussians,
                                pipe,
                                bg,
                                opt,
                                DENSIFY=True,
                            )
                        )
                        gaussians.densify_and_prune_fastgs(
                            max_screen_size=size_threshold,
                            min_opacity=0.005,
                            extent=scene.cameras_extent,
                            radii=radii,
                            args=opt,
                            importance_score=importance_score,
                            pruning_score=pruning_score,
                        )
                    elif event == "opacity_reset":
                        gaussians.reset_opacity()
                    else:
                        raise RuntimeError(
                            "Unknown LR topology event: {}".format(event)
                        )
        else:
            events = detail_hr_topology_events(
                clock.hr_iteration,
                opt.opacity_reset_interval,
            )
            with torch.no_grad():
                for event in events:
                    if event == "collect_stats":
                        gaussians.max_radii2D[visibility_filter] = torch.max(
                            gaussians.max_radii2D[visibility_filter],
                            radii[visibility_filter],
                        )
                        gaussians.add_densification_stats(
                            viewspace_point_tensor,
                            visibility_filter,
                        )
                    elif event == "detail_densify":
                        size_threshold = (
                            20
                            if clock.hr_iteration
                            > opt.opacity_reset_interval
                            else None
                        )
                        scoring_cameras = sample_fastgs_scoring_cameras(
                            active_cameras.copy()
                        )
                        detail_scoring_started = time.perf_counter()
                        score_result = compute_gaussian_score_fastgs_detail(
                            scoring_cameras,
                            gaussians,
                            pipe,
                            bg,
                            opt,
                            detail_config,
                        )
                        detail_scoring_time_ms = (
                            time.perf_counter() - detail_scoring_started
                        ) * 1_000.0
                        if os.environ.get("FRAMEWORK_VIS", "0") == "1":
                            from utils.framework_visualization import maybe_save_framework
                            maybe_save_framework(
                                scene.model_path, iteration, clock.hr_iteration,
                                scoring_cameras, gaussians, pipe, bg, opt,
                                detail_config, score_result,
                            )
                        detail_scoring_time_total_ms += detail_scoring_time_ms
                        topology_diagnostics.update(
                            gaussians.densify_and_prune_fastgs_detail(
                                max_screen_size=size_threshold,
                                min_opacity=0.005,
                                extent=scene.cameras_extent,
                                radii=radii,
                                args=opt,
                                importance_score=(
                                    score_result.importance_score
                                ),
                                pruning_score=score_result.pruning_score,
                                detail_score=score_result.detail_score,
                                detail_candidate_mask=(
                                    score_result.detail_candidate_mask
                                ),
                                detail_config=detail_config,
                                current_hr_iteration=clock.hr_iteration,
                                max_gaussians=max_gaussians,
                            )
                        )
                        visible_view_mean = float(
                            score_result.visible_view_count.to(
                                torch.float32
                            ).mean().item()
                        )
                        detail_support_mean = float(
                            score_result.detail_support_ratio.mean().item()
                        )
                        print(
                            "[Detail densify] global={} hr_iter={} "
                            "normal_clone_candidates={} "
                            "detail_clone_candidates={} "
                            "normal_detail_clone_overlap_count={} "
                            "detail_only_clone_candidates={} "
                            "normal_split_candidates={} "
                            "detail_split_candidates={} "
                            "normal_detail_split_overlap_count={} "
                            "detail_only_split_candidates={} "
                            "selected_clone_count={} selected_split_count={} "
                            "selected_detail_only_clone_count={} "
                            "selected_detail_only_split_count={} "
                            "ordinary_pruned={} candidates={} "
                            "mask_ratio={:.6f} mean_overlap={:.6f} "
                            "visible_views={:.3f} support_ratio={:.3f} "
                            "extra(children/net)={}/{} alive(region/extra/lineage)="
                            "{}/{}/{} total={} cap_reached={} scoring_ms={:.3f}".format(
                                iteration,
                                clock.hr_iteration,
                                topology_diagnostics[
                                    "normal_clone_candidates"
                                ],
                                topology_diagnostics[
                                    "detail_clone_candidates"
                                ],
                                topology_diagnostics[
                                    "normal_detail_clone_overlap_count"
                                ],
                                topology_diagnostics[
                                    "detail_only_clone_candidates"
                                ],
                                topology_diagnostics[
                                    "normal_split_candidates"
                                ],
                                topology_diagnostics[
                                    "detail_split_candidates"
                                ],
                                topology_diagnostics[
                                    "normal_detail_split_overlap_count"
                                ],
                                topology_diagnostics[
                                    "detail_only_split_candidates"
                                ],
                                topology_diagnostics[
                                    "selected_clone_count"
                                ],
                                topology_diagnostics[
                                    "selected_split_count"
                                ],
                                topology_diagnostics[
                                    "selected_detail_only_clone_count"
                                ],
                                topology_diagnostics[
                                    "selected_detail_only_split_count"
                                ],
                                topology_diagnostics[
                                    "ordinary_pruned_count"
                                ],
                                score_result.statistics[
                                    "detail_candidate_count"
                                ],
                                score_result.statistics[
                                    "detail_pixel_ratio"
                                ],
                                score_result.statistics["detail_mean_score"],
                                visible_view_mean,
                                detail_support_mean,
                                topology_diagnostics[
                                    "detail_extra_children_count"
                                ],
                                topology_diagnostics[
                                    "detail_extra_net_gaussian_count"
                                ],
                                topology_diagnostics[
                                    "detail_region_alive_count"
                                ],
                                topology_diagnostics[
                                    "detail_extra_origin_alive_count"
                                ],
                                topology_diagnostics[
                                    "detail_lineage_alive_count"
                                ],
                                topology_diagnostics[
                                    "total_gaussian_count"
                                ],
                                topology_diagnostics["cap_reached"],
                                detail_scoring_time_ms,
                            )
                        )
                        if tensorboard_writer:
                            tensorboard_writer.add_scalar(
                                "detail/mask_ratio",
                                score_result.statistics[
                                    "detail_pixel_ratio"
                                ],
                                iteration,
                            )
                            tensorboard_writer.add_scalar(
                                "detail/candidate_count",
                                score_result.statistics[
                                    "detail_candidate_count"
                                ],
                                iteration,
                            )
                            tensorboard_writer.add_scalar(
                                "detail/new_gaussian_count",
                                topology_diagnostics[
                                    "detail_extra_children_count"
                                ],
                                iteration,
                            )
                            tensorboard_writer.add_scalar(
                                "detail/total_gaussian_count",
                                topology_diagnostics[
                                    "total_gaussian_count"
                                ],
                                iteration,
                            )
                            tensorboard_writer.add_scalar(
                                "detail/flagged_gaussian_count",
                                topology_diagnostics[
                                    "detail_lineage_alive_count"
                                ],
                                iteration,
                            )
                            tensorboard_writer.add_scalar(
                                "detail/mean_score",
                                score_result.statistics[
                                    "detail_mean_score"
                                ],
                                iteration,
                            )
                            tensorboard_writer.add_scalar(
                                "detail/cap_reached",
                                int(topology_diagnostics["cap_reached"]),
                                iteration,
                            )
                            _add_tensorboard_histogram(
                                tensorboard_writer,
                                "detail/view_hits",
                                score_result.detail_view_hits,
                                iteration,
                            )
                            _add_tensorboard_histogram(
                                tensorboard_writer,
                                "detail/visible_view_count",
                                score_result.visible_view_count,
                                iteration,
                            )
                            _add_tensorboard_histogram(
                                tensorboard_writer,
                                "detail/support_ratio",
                                score_result.detail_support_ratio,
                                iteration,
                            )
                            scalar_diagnostics = (
                                (
                                    "detail/clone_overlap_count",
                                    "normal_detail_clone_overlap_count",
                                ),
                                (
                                    "detail/split_overlap_count",
                                    "normal_detail_split_overlap_count",
                                ),
                                (
                                    "detail/selected_detail_only_clone_count",
                                    "selected_detail_only_clone_count",
                                ),
                                (
                                    "detail/selected_detail_only_split_count",
                                    "selected_detail_only_split_count",
                                ),
                                (
                                    "detail/extra_children_count",
                                    "detail_extra_children_count",
                                ),
                                (
                                    "detail/extra_net_gaussian_count",
                                    "detail_extra_net_gaussian_count",
                                ),
                                (
                                    "detail/region_alive_count",
                                    "detail_region_alive_count",
                                ),
                                (
                                    "detail/extra_origin_alive_count",
                                    "detail_extra_origin_alive_count",
                                ),
                                (
                                    "detail/lineage_alive_count",
                                    "detail_lineage_alive_count",
                                ),
                            )
                            for tag, diagnostic_name in scalar_diagnostics:
                                tensorboard_writer.add_scalar(
                                    tag,
                                    topology_diagnostics[diagnostic_name],
                                    iteration,
                                )
                            tensorboard_writer.add_scalar(
                                "detail/scoring_time_ms",
                                detail_scoring_time_ms,
                                iteration,
                            )
                    elif event == "opacity_reset":
                        gaussians.reset_opacity()
                    elif event == "final_vcp":
                        scoring_cameras = sample_fastgs_scoring_cameras(
                            active_cameras.copy()
                        )
                        _, pruning_score = compute_gaussian_score_fastgs(
                            scoring_cameras,
                            gaussians,
                            pipe,
                            bg,
                            opt,
                        )
                        before_final_vcp = gaussians.get_xyz.shape[0]
                        gaussians.final_prune_fastgs(
                            min_opacity=0.1,
                            pruning_score=pruning_score,
                        )
                        topology_diagnostics[
                            "final_vcp_removed_count"
                        ] = before_final_vcp - gaussians.get_xyz.shape[0]
                        print(
                            "[Detail final VCP] global={} hr_iter={} "
                            "final_vcp_removed_count={} remaining={}".format(
                                iteration,
                                clock.hr_iteration,
                                topology_diagnostics[
                                    "final_vcp_removed_count"
                                ],
                                gaussians.get_xyz.shape[0],
                            )
                        )
                    else:
                        raise RuntimeError(
                            "Unknown HR topology event: {}".format(event)
                        )

        count_topology_events = [
            event
            for event in events
            if event in ("densify", "detail_densify", "final_vcp")
        ]
        if gaussian_count_log and (
            iteration % gaussian_count_interval == 0
            or count_topology_events
            or transition.reset_hr_statistics
            or iteration == opt.iterations
        ):
            if count_topology_events:
                count_event = "+".join(count_topology_events)
            elif transition.reset_hr_statistics:
                count_event = "stage_switch"
            elif iteration == opt.iterations:
                count_event = "final"
            else:
                count_event = "sample"
            phase_iteration = (
                iteration
                if clock.phase == "lr"
                else clock.hr_iteration
            )
            append_count_log(
                gaussian_count_log,
                iteration,
                clock.phase,
                phase_iteration,
                gaussians.get_xyz.shape[0],
                count_event,
            )

        with torch.no_grad():
            optimizer_iteration = optimizer_schedule_iteration(clock)
            if iteration < opt.iterations:
                gaussians.optimizer_step(optimizer_iteration)

        iteration_end.record()
        torch.cuda.synchronize()
        iteration_time_ms = iteration_start.elapsed_time(iteration_end)
        phase_time_ms[clock.phase] += iteration_time_ms

        if iteration == opt.iterations:
            release_frequency_target_cache(
                frequency_target_cache,
                "before_final_outputs",
            )

        if not gaussian_count_only and iteration in saving_iterations:
            print(
                "\n[ITER {}] Saving Gaussians after topology and optimizer".format(
                    iteration
                )
            )
            scene.save(iteration)
            detail_metadata_path = save_detail_metadata_sidecar(
                scene.model_path,
                iteration,
                gaussians,
                switch_iteration,
            )
            print(
                "[ITER {}] Saved detail metadata {}".format(
                    iteration,
                    detail_metadata_path,
                )
            )

        debug_errors = []
        if (
            not gaussian_count_only
            and
            clock.phase == "hr"
            and clock.hr_iteration % detail_config.debug_interval == 0
        ):
            try:
                debug_statistics = save_detail_debug_bundle(
                    scene.model_path,
                    clock.hr_iteration,
                    fixed_debug_camera,
                    gaussians,
                    pipe,
                    bg,
                    opt,
                    detail_config,
                )
                print(
                    "[Detail debug] global={} hr_iter={} "
                    "detail_ratio={:.6f}".format(
                        iteration,
                        clock.hr_iteration,
                        debug_statistics["detail_ratio"],
                    )
                )
            except Exception as error:
                debug_errors.append(
                    "detail {}: {}".format(type(error).__name__, error)
                )
                torch.cuda.empty_cache()
                print(
                    "[Detail debug] global={} hr_iter={} failed: {}".format(
                        iteration,
                        clock.hr_iteration,
                        error,
                    )
                )

        if (
            not gaussian_count_only
            and
            clock.phase == "hr"
            and clock.hr_iteration % 1_000 == 0
        ):
            try:
                frequency_debug_statistics = save_frequency_debug_bundle(
                    scene.model_path,
                    clock.hr_iteration,
                    fixed_debug_camera,
                    gaussians,
                    pipe,
                    bg,
                    opt,
                    frequency_target_cache,
                    frequency_config,
                )
                print(
                    "[Frequency debug] global={} hr_iter={} "
                    "mask_ratio={:.6f} lr_psnr={:.6f}".format(
                        iteration,
                        clock.hr_iteration,
                        frequency_debug_statistics["hf_mask_ratio"],
                        frequency_debug_statistics["lr_psnr"],
                    )
                )
            except Exception as error:
                debug_errors.append(
                    "frequency {}: {}".format(type(error).__name__, error)
                )
                torch.cuda.empty_cache()
                print(
                    "[Frequency debug] global={} hr_iter={} failed: {}".format(
                        iteration,
                        clock.hr_iteration,
                        error,
                    )
                )

        if (
            not gaussian_count_only
            and
            iteration == opt.iterations
            and iteration in checkpoint_iterations
        ):
            release_frequency_target_cache(
                frequency_target_cache,
                "before_final_checkpoint",
            )

        if not gaussian_count_only and iteration in checkpoint_iterations:
            checkpoint_path = os.path.join(
                scene.model_path,
                "chkpnt{}.pth".format(iteration),
            )
            checkpoint_payload = build_checkpoint_payload(
                gaussians,
                opt,
                state,
                stage_config,
                manifest_hash,
                capture_runtime_state(
                    lr_cameras,
                    hr_cameras,
                    viewpoint_stack,
                    bg,
                ),
                detail_config,
                frequency_config,
                max_gaussians,
                switch_iteration,
            )
            save_checkpoint_atomic(
                checkpoint_path,
                checkpoint_payload,
            )
            print(
                "[ITER {}] Saved recoverable checkpoint {}".format(
                    iteration,
                    checkpoint_path,
                )
            )
        if debug_errors:
            raise RuntimeError(
                "Required HR debug snapshot failed at HR iteration {}: "
                "{}".format(
                    clock.hr_iteration,
                    "; ".join(debug_errors),
                )
            )

        ema_loss_for_log = (
            0.4 * loss.item() + 0.6 * ema_loss_for_log
        )
        progress_bar.update(1)
        should_log = (
            iteration % 100 == 0
            or iteration == switch_iteration
            or iteration == switch_iteration + 1
            or iteration == opt.iterations
        )
        frequency_log_values = None
        if should_log and frequency_result is not None:
            frequency_log_values = {
                "raw_lr": float(frequency_result.raw_lr.item()),
                "raw_hf": float(frequency_result.raw_hf.item()),
                "raw_rgb": float(frequency_result.raw_rgb.item()),
                "weighted_lr": float(frequency_result.weighted_lr.item()),
                "weighted_hf": float(frequency_result.weighted_hf.item()),
                "weighted_rgb": float(frequency_result.weighted_rgb.item()),
                "hf_mask_ratio": frequency_result.mask_ratio,
                "target_residual_mean": (
                    frequency_result.target_residual_mean
                ),
                "render_residual_mean": float(
                    frequency_result.render_residual_mean.item()
                ),
                "lr_psnr": float(frequency_result.lr_psnr.item()),
            }
        if iteration % 10 == 0 or should_log:
            progress_bar.set_postfix(
                {
                    "phase": clock.phase,
                    "loss": "{:.7f}".format(ema_loss_for_log),
                }
            )
        if should_log:
            hr_label = (
                "-" if clock.hr_iteration is None else clock.hr_iteration
            )
            print(
                "[Two-stage] global={} phase={} hr_iter={} size={}x{} "
                "loss={:.7f} l1={:.7f} dssim={:.7f} gaussians={} "
                "topology={}".format(
                    iteration,
                    clock.phase,
                    hr_label,
                    viewpoint_camera.image_width,
                    viewpoint_camera.image_height,
                    loss.item(),
                    loss_l1.item(),
                    loss_dssim.item(),
                    gaussians._xyz.shape[0],
                    ",".join(events) if events else "-",
                )
            )
            if frequency_log_values is not None:
                print(
                    "[Frequency] global={} hr_iter={} alpha={:.6f} "
                    "raw_lr={:.7f} raw_hf={:.7f} raw_rgb={:.7f} "
                    "weighted_lr={:.7f} weighted_hf={:.7f} "
                    "weighted_rgb={:.7f} mask_ratio={:.6f} "
                    "target_residual_mean={:.7f} "
                    "render_residual_mean={:.7f} lr_psnr={:.6f}".format(
                        iteration,
                        clock.hr_iteration,
                        frequency_result.alpha,
                        frequency_log_values["raw_lr"],
                        frequency_log_values["raw_hf"],
                        frequency_log_values["raw_rgb"],
                        frequency_log_values["weighted_lr"],
                        frequency_log_values["weighted_hf"],
                        frequency_log_values["weighted_rgb"],
                        frequency_log_values["hf_mask_ratio"],
                        frequency_log_values["target_residual_mean"],
                        frequency_log_values["render_residual_mean"],
                        frequency_log_values["lr_psnr"],
                    )
                )
        if tensorboard_writer and should_log:
            tensorboard_writer.add_scalar(
                "train/loss_total",
                loss.item(),
                iteration,
            )
            tensorboard_writer.add_scalar(
                "train/loss_l1",
                loss_l1.item(),
                iteration,
            )
            tensorboard_writer.add_scalar(
                "train/loss_dssim",
                loss_dssim.item(),
                iteration,
            )
            tensorboard_writer.add_scalar(
                "train/gaussian_count",
                gaussians._xyz.shape[0],
                iteration,
            )
            if frequency_log_values is not None:
                frequency_scalars = (
                    ("loss/raw_lr", frequency_log_values["raw_lr"]),
                    ("loss/raw_hf", frequency_log_values["raw_hf"]),
                    ("loss/raw_rgb", frequency_log_values["raw_rgb"]),
                    ("loss/weighted_lr", frequency_log_values["weighted_lr"]),
                    ("loss/weighted_hf", frequency_log_values["weighted_hf"]),
                    ("loss/weighted_rgb", frequency_log_values["weighted_rgb"]),
                    (
                        "frequency/hf_mask_ratio",
                        frequency_log_values["hf_mask_ratio"],
                    ),
                    (
                        "frequency/target_residual_mean",
                        frequency_log_values["target_residual_mean"],
                    ),
                    (
                        "frequency/render_residual_mean",
                        frequency_log_values["render_residual_mean"],
                    ),
                    ("frequency/lr_psnr", frequency_log_values["lr_psnr"]),
                )
                for tag, value in frequency_scalars:
                    tensorboard_writer.add_scalar(tag, value, iteration)

    progress_bar.close()
    if tensorboard_writer:
        tensorboard_writer.close()
    total_time_seconds = (
        phase_time_ms["lr"] + phase_time_ms["hr"]
    ) / 1_000.0
    print("Gaussian number: {}".format(gaussians._xyz.shape[0]))
    print(
        "Training time: LR={:.3f}s HR={:.3f}s total={:.3f}s".format(
            phase_time_ms["lr"] / 1_000.0,
            phase_time_ms["hr"] / 1_000.0,
            total_time_seconds,
        )
    )
    detail_scoring_share = (
        0.0
        if phase_time_ms["hr"] <= 0.0
        else 100.0
        * detail_scoring_time_total_ms
        / phase_time_ms["hr"]
    )
    print(
        "Detail scoring time: {:.3f}s ({:.2f}% of HR time)".format(
            detail_scoring_time_total_ms / 1_000.0,
            detail_scoring_share,
        )
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024.0 ** 2)
        print(
            "Peak CUDA memory: allocated={:.1f} MiB reserved={:.1f} MiB".format(
                peak_allocated_mib,
                peak_reserved_mib,
            )
        )
    print("Dataset manifest SHA-256: {}".format(manifest_hash))


def _normalized_iterations(values, switch_iteration, total_iterations):
    normalized = set(values)
    normalized.add(switch_iteration)
    normalized.add(total_iterations)
    invalid = sorted(
        value
        for value in normalized
        if value < 1 or value > total_iterations
    )
    if invalid:
        raise ValueError(
            "Save/checkpoint iterations are out of range: {}".format(
                invalid
            )
        )
    return sorted(normalized)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Experiment 4-A two-stage FastGS frequency-decoupled supervision"
        )
    )
    model_parameters = ModelParams(parser)
    optimization_parameters = OptimizationParams(parser)
    pipeline_parameters = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        default=False,
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--websockets",
        action="store_true",
        default=False,
    )
    parser.add_argument("--switch_iteration", type=int, default=8_000)
    parser.add_argument("--lr_image_dir", type=str, default="images_8")
    parser.add_argument("--hr_image_dir", type=str, default="images_sr_x4")
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--detail_hf_quantile", type=float, default=0.70)
    parser.add_argument("--detail_error_quantile", type=float, default=0.70)
    parser.add_argument("--detail_min_visible_views", type=int, default=3)
    parser.add_argument("--detail_min_total_pixels", type=int, default=8)
    parser.add_argument("--detail_min_views", type=int, default=2)
    parser.add_argument("--detail_min_pixels", type=int, default=3)
    parser.add_argument(
        "--detail_min_overlap_ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--detail_min_support_ratio",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--detail_min_average_overlap_ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument("--detail_grad_scale", type=float, default=0.80)
    parser.add_argument(
        "--detail_importance_threshold",
        type=int,
        default=4,
    )
    parser.add_argument("--detail_dilation", type=int, default=1)
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument("--freq_lr_weight", type=float, default=1.0)
    parser.add_argument("--freq_hf_weight", type=float, default=0.5)
    parser.add_argument("--freq_rgb_weight", type=float, default=0.1)
    parser.add_argument("--freq_transition_iters", type=int, default=1_000)
    parser.add_argument("--freq_hf_quantile", type=float, default=0.70)
    parser.add_argument("--freq_hf_abs_thresh", type=float, default=0.01)
    parser.add_argument("--freq_hf_min_pixels", type=int, default=256)
    parser.add_argument("--freq_hf_border_ignore", type=int, default=2)
    parser.add_argument("--freq_charbonnier_epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=[8_000, 9_000, 12_000, 15_000, 18_000, 24_000, 30_000],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[8_000, 9_000, 12_000, 15_000, 18_000, 24_000, 30_000],
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--sparse_view_manifest", type=str, default=None)
    parser.add_argument("--gaussian_count_log", type=str, default=None)
    parser.add_argument("--gaussian_count_interval", type=int, default=100)
    parser.add_argument("--gaussian_count_only", action="store_true")
    arguments = parser.parse_args(sys.argv[1:])
    validate_count_interval(arguments.gaussian_count_interval)
    detail_config = DetailDensifyConfig(
        high_frequency_quantile=arguments.detail_hf_quantile,
        error_quantile=arguments.detail_error_quantile,
        dilation_radius=arguments.detail_dilation,
        min_visible_views=arguments.detail_min_visible_views,
        min_total_pixels_per_view=arguments.detail_min_total_pixels,
        min_detail_views=arguments.detail_min_views,
        min_detail_pixels_per_view=arguments.detail_min_pixels,
        min_detail_overlap_ratio=arguments.detail_min_overlap_ratio,
        min_detail_support_ratio=arguments.detail_min_support_ratio,
        min_average_detail_overlap_ratio=(
            arguments.detail_min_average_overlap_ratio
        ),
        detail_gradient_scale=arguments.detail_grad_scale,
        detail_importance_threshold=(
            arguments.detail_importance_threshold
        ),
    )
    frequency_config = FrequencyLossConfig(
        scale=arguments.sr_scale,
        lr_weight=arguments.freq_lr_weight,
        hf_weight=arguments.freq_hf_weight,
        rgb_weight=arguments.freq_rgb_weight,
        transition_iterations=arguments.freq_transition_iters,
        hf_quantile=arguments.freq_hf_quantile,
        hf_absolute_threshold=arguments.freq_hf_abs_thresh,
        hf_min_pixels=arguments.freq_hf_min_pixels,
        hf_border_ignore=arguments.freq_hf_border_ignore,
        charbonnier_epsilon=arguments.freq_charbonnier_epsilon,
    )

    arguments.save_iterations = _normalized_iterations(
        arguments.save_iterations,
        arguments.switch_iteration,
        arguments.iterations,
    )
    arguments.checkpoint_iterations = _normalized_iterations(
        arguments.checkpoint_iterations,
        arguments.switch_iteration,
        arguments.iterations,
    )
    print("Optimizing {}".format(arguments.model_path))

    safe_state(arguments.quiet)
    if arguments.websockets:
        network_gui_ws.init(arguments.ip, arguments.port)
    torch.autograd.set_detect_anomaly(arguments.detect_anomaly)

    training_two_stage_frequency(
        model_parameters.extract(arguments),
        optimization_parameters.extract(arguments),
        pipeline_parameters.extract(arguments),
        arguments.save_iterations,
        arguments.checkpoint_iterations,
        arguments.start_checkpoint,
        arguments.debug_from,
        arguments.websockets,
        arguments.switch_iteration,
        arguments.lr_image_dir,
        arguments.hr_image_dir,
        arguments.sr_scale,
        detail_config,
        frequency_config,
        arguments.max_gaussians,
        arguments.sparse_view_manifest,
        arguments.gaussian_count_log,
        arguments.gaussian_count_interval,
        arguments.gaussian_count_only,
    )
    print("\nTraining complete.")
