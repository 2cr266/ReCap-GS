"""Memory-safe frequency target cache for 4-A best-of-five runs.

The original 4-A implementation remains untouched.  This variant only adds an
explicit release hook used after the final training update and before large
output files are serialized.
"""

from utils.frequency_loss_utils import (
    FrequencyLossConfig,
    FrequencyTarget,
    FrequencyTargetCache as _FrequencyTargetCache,
    compute_frequency_loss,
)


class FrequencyTargetCache(_FrequencyTargetCache):
    """Original 4-A targets without duplicating the debug-only base tensor.

    ``target_base`` is not used by the 4-A loss.  The original cache stores it
    next to ``target_residual`` even though the HR image and residual already
    determine the base exactly.  Alias the field to the residual tensor here;
    the dedicated trainer reconstructs the base only when writing debug PNGs.
    All tensors that affect optimization remain the original FP32 tensors.
    """

    def get(self, image_name, target_lr, target_sr):
        entry = super().get(image_name, target_lr, target_sr)
        if entry.target_base.data_ptr() == entry.target_residual.data_ptr():
            return entry
        compact_entry = FrequencyTarget(
            target_base=entry.target_residual,
            target_residual=entry.target_residual,
            hf_mask=entry.hf_mask,
            mask_pixel_count=entry.mask_pixel_count,
            mask_ratio=entry.mask_ratio,
            target_residual_mean=entry.target_residual_mean,
        )
        self._entries[image_name] = compact_entry
        return compact_entry

    def clear(self):
        released_entries = len(self._entries)
        self._entries.clear()
        self._sources.clear()
        return released_entries


__all__ = [
    "FrequencyLossConfig",
    "FrequencyTarget",
    "FrequencyTargetCache",
    "compute_frequency_loss",
]
