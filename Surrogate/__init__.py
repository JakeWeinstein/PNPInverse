"""Surrogate -- POD/RBF and NN surrogate models for BV kinetics inference.

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
        SubsetSurrogateObjective,
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
        EnsembleMeanWrapper,
        load_nn_ensemble,
    )
"""

from Surrogate.sampling import ParameterBounds, generate_lhs_samples, generate_multi_region_lhs_samples
from Surrogate.surrogate_model import SurrogateConfig, BVSurrogateModel
from Surrogate.training import generate_training_data_single, generate_training_dataset
from Surrogate.objectives import (
    SurrogateObjective,
    AlphaOnlySurrogateObjective,
    ReactionBlockSurrogateObjective,
    SubsetSurrogateObjective,
)
from Surrogate.validation import validate_surrogate, print_validation_report
from Surrogate.io import save_surrogate, load_surrogate
from Surrogate.bcd import BCDConfig, BCDResult, run_block_coordinate_descent
from Surrogate.multistart import MultiStartConfig, MultiStartResult, run_multistart_inference
from Surrogate.cascade import CascadeConfig, CascadeResult, run_cascade_inference
from Surrogate.ensemble import EnsembleMeanWrapper, load_nn_ensemble

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
    "SubsetSurrogateObjective",
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
    "EnsembleMeanWrapper",
    "load_nn_ensemble",
]
