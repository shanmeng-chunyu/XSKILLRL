"""Utilities for mixed-benchmark preparation and RL-facing dataset export."""

from .benchmark_protocol import (
    allocate_global_val_split,
    expand_records_sqrt_size,
    read_json,
    stratified_train_test_split,
    write_json,
    write_split_bundle,
)

from .dataset import (
    load_records_from_spec,
    load_records_grouped_by_benchmark,
    samples_to_grouped_records,
)

__all__ = [
    "allocate_global_val_split",
    "expand_records_sqrt_size",
    "load_records_from_spec",
    "load_records_grouped_by_benchmark",
    "read_json",
    "samples_to_grouped_records",
    "stratified_train_test_split",
    "write_json",
    "write_split_bundle",
]
