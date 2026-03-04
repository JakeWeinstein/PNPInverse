"""Surrogate -- POD/RBF surrogate model for BV kinetics inference.

Public API:

    from Surrogate import (
        ParameterBounds,
        generate_lhs_samples,
        generate_multi_region_lhs_samples,
        SurrogateConfig,
        BVSurrogateModel,
        generate_training_data_single,
        generate_training_dataset,
        SurrogateObjective,
        AlphaOnlySurrogateObjective,
        ReactionBlockSurrogateObjective,
        validate_surrogate,
        print_validation_report,
        save_surrogate,
        load_surrogate,
        BCDConfig,
        BCDResult,
        run_block_coordinate_descent,
        MultiStartConfig,
        MultiStartResult,
        run_multistart_inference,
        CascadeConfig,
        CascadeResult,
        run_cascade_inference,
    )
"""

from Surrogate.sampling import ParameterBounds, generate_lhs_samples, generate_multi_region_lhs_samples
from Surrogate.surrogate_model import SurrogateConfig, BVSurrogateModel
from Surrogate.training import generate_training_data_single, generate_training_dataset
from Surrogate.objectives import SurrogateObjective, AlphaOnlySurrogateObjective, ReactionBlockSurrogateObjective
from Surrogate.validation import validate_surrogate, print_validation_report
from Surrogate.io import save_surrogate, load_surrogate
from Surrogate.bcd import BCDConfig, BCDResult, run_block_coordinate_descent
from Surrogate.multistart import MultiStartConfig, MultiStartResult, run_multistart_inference
from Surrogate.cascade import CascadeConfig, CascadeResult, run_cascade_inference

__all__ = [
    "ParameterBounds",
    "generate_lhs_samples",
    "generate_multi_region_lhs_samples",
    "SurrogateConfig",
    "BVSurrogateModel",
    "generate_training_data_single",
    "generate_training_dataset",
    "SurrogateObjective",
    "AlphaOnlySurrogateObjective",
    "ReactionBlockSurrogateObjective",
    "validate_surrogate",
    "print_validation_report",
    "save_surrogate",
    "load_surrogate",
    "BCDConfig",
    "BCDResult",
    "run_block_coordinate_descent",
    "MultiStartConfig",
    "MultiStartResult",
    "run_multistart_inference",
    "CascadeConfig",
    "CascadeResult",
    "run_cascade_inference",
]
