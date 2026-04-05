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
        GPSurrogateModel,
        load_gp_surrogate,
        PCEConfig,
        PCESurrogateModel,
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
from Surrogate.gp_model import GPSurrogateModel, load_gp_surrogate
from Surrogate.pce_model import PCEConfig, PCESurrogateModel
from Surrogate.ismo import (
    AcquisitionStrategy,
    ISMOConfig,
    ISMOIteration,
    ISMOResult,
    run_ismo,
)
from Surrogate.acquisition import AcquisitionConfig, AcquisitionResult, select_new_samples
from Surrogate.ismo_retrain import (
    ISMORetrainConfig,
    ISMORetrainResult,
    MergedData,
    merge_training_data,
    retrain_surrogate,
)
from Surrogate.ismo_pde_eval import (
    PDESolverBundle,
    PDEEvalResult,
    SurrogatePDEComparison,
    AugmentedDataset,
    QualityReport,
    evaluate_candidates_with_pde,
    compare_surrogate_vs_pde,
    integrate_new_data,
    check_pde_quality,
)
from Surrogate.ismo_convergence import (
    ISMOConvergenceCriteria,
    ISMOConvergenceChecker,
    ISMODiagnosticRecord,
)

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
    "GPSurrogateModel",
    "load_gp_surrogate",
    "PCEConfig",
    "PCESurrogateModel",
    # ISMO core (4-01)
    "AcquisitionStrategy",
    "ISMOConfig",
    "ISMOIteration",
    "ISMOResult",
    "run_ismo",
    # Acquisition (4-02)
    "AcquisitionConfig",
    "AcquisitionResult",
    "select_new_samples",
    # Retraining (4-03)
    "ISMORetrainConfig",
    "ISMORetrainResult",
    "MergedData",
    "merge_training_data",
    "retrain_surrogate",
    # PDE eval (4-04)
    "PDESolverBundle",
    "PDEEvalResult",
    "SurrogatePDEComparison",
    "AugmentedDataset",
    "QualityReport",
    "evaluate_candidates_with_pde",
    "compare_surrogate_vs_pde",
    "integrate_new_data",
    "check_pde_quality",
    # Convergence (4-05)
    "ISMOConvergenceCriteria",
    "ISMOConvergenceChecker",
    "ISMODiagnosticRecord",
]
