from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

EASYTSAD_SCHEMAS = ("naive", "all_in_one", "zero_shot")
EASYTSAD_REVISION = "e7be92374a4543d7c52d6e34ef7047f4d6776be0"
BEST_UNLABELED_VALIDATION = "best_unlabeled_validation"

ArtifactType = Literal[
    "torch_state_dict",
    "fitted_estimator",
    "replay_recipe",
    "tensorflow_checkpoint",
]
SelectionStatus = Literal["available", "unavailable", "not_applicable"]


@dataclass(frozen=True)
class SampleSpec:
    window_parameter: str | None = None
    step_parameter: str | None = None
    horizon: int = 0

    def count(self, length: int, method: object) -> int:
        if self.window_parameter is None:
            return length
        window = int(getattr(method, self.window_parameter))
        step = (
            int(getattr(method, self.step_parameter))
            if self.step_parameter is not None
            else 1
        )
        return max(0, (length - window - self.horizon) // step + 1)


@dataclass(frozen=True)
class SchemaCapability:
    fit_scope: str | None
    selection_objective: str | None = None
    selection_policy: str | None = None
    unavailable_reason: str | None = None

    @property
    def supported(self) -> bool:
        return self.fit_scope is not None


@dataclass(frozen=True)
class AlgorithmSpec:
    method_name: str
    module: str
    requirements: tuple[str, ...]
    schemas: Mapping[str, SchemaCapability]
    artifact_type: ArtifactType
    sample_spec: SampleSpec
    artifact_attribute: str | None = None
    python_requires: str = ">=3.9,<3.13"
    shared_runtime: str | None = None
    agent_available: bool = True
    searchable_parameters: tuple[str, ...] | None = None
    supported_datasets: tuple[str, ...] | None = None

    def supports_dataset(self, dataset: str) -> bool:
        return self.supported_datasets is None or dataset in self.supported_datasets

    def capability(self, schema: str) -> SchemaCapability:
        if schema not in EASYTSAD_SCHEMAS:
            raise ValueError(f"unsupported EasyTSAD schema: {schema}")
        return self.schemas[schema]

    def validate_parameters(self, parameters: Mapping[str, object], defaults: Mapping[str, object]) -> None:
        from .configuration import validate_parameter_type

        if set(parameters) != set(defaults):
            raise ValueError(f"{self.method_name} parameters must match the declared Method names")
        for name, value in parameters.items():
            try:
                validate_parameter_type(defaults[name], value, name)
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
        if self.searchable_parameters is None:
            return
        for name, value in parameters.items():
            if name not in self.searchable_parameters and (type(value) is not type(defaults[name]) or value != defaults[name]):
                raise ValueError(f"{self.method_name}.{name} is frozen at its native default")
        if self.method_name in ("LSTMADalpha", "LSTMADbeta"):
            for name in ("window_size", "hidden_dim"):
                if type(parameters[name]) is not int or parameters[name] < 1:
                    raise ValueError(f"{self.method_name}.{name} must be a positive integer")
            lr = parameters["lr"]
            if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
                raise ValueError(f"{self.method_name}.lr must be a finite non-negative number")
        elif self.method_name == "FITS":
            for name in ("out_len", "cut_freq"):
                if type(parameters[name]) is not int or parameters[name] < 1:
                    raise ValueError(f"FITS.{name} must be a positive integer")
            width, inputs, cutoff = parameters["out_len"], parameters["input_len"], parameters["cut_freq"]
            if width % inputs or width % 2:
                raise ValueError("FITS out_len must be even and divisible by frozen input_len")
            if cutoff > inputs // 2 + 1 or cutoff * (width // inputs) > width // 2 + 1:
                raise ValueError("FITS cut_freq exceeds the input/output real-FFT frequency support")
            lr = parameters["lr"]
            if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
                raise ValueError("FITS.lr must be a finite non-negative number")
        elif self.method_name == "DCdetector":
            for name in ("win_size", "d_model"):
                if type(parameters[name]) is not int or parameters[name] < 2:
                    raise ValueError(f"DCdetector.{name} must be an integer >= 2")
            if parameters["win_size"] > 5000 or any(parameters["win_size"] % size for size in parameters["patch_size"]):
                raise ValueError("DCdetector win_size must divide into every frozen patch size and fit positional length 5000")
            if parameters["d_model"] % 2:
                raise ValueError("DCdetector d_model must be even for native positional embedding")
            lr = parameters["learning_rate"]
            if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
                raise ValueError("DCdetector.learning_rate must be a finite non-negative number")
        elif self.method_name == "FCVAE":
            width = parameters["window"]
            if type(width) is not int or width < max(8, parameters["kernel_size"]) or width % 2:
                raise ValueError("FCVAE window must be even, at least 8 and cover the frozen local FFT kernel")
            if type(parameters["latent_dim"]) is not int or parameters["latent_dim"] < 1:
                raise ValueError("FCVAE latent_dim must be a positive integer")
            lr = parameters["learning_rate"]
            if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
                raise ValueError("FCVAE.learning_rate must be a finite non-negative number")

    def score_support(self, length: int, parameters: Mapping[str, object]) -> dict[str, int] | None:
        if self.method_name == "IsolationForest":
            if length < 1:
                raise ValueError("IsolationForest requires at least one test observation")
            return {"start": 0, "stop": length, "count": length,
                    "window_samples": length, "native_tail_zeros": 0}
        if self.method_name == "FCVAE":
            width = int(parameters["window"])
            if length < width:
                raise ValueError(f"FCVAE test length {length} requires window={width}")
            return {"start": width - 1, "stop": length, "count": length - width + 1,
                    "window_samples": length - width + 1, "native_tail_zeros": 0}
        if self.method_name == "DCdetector":
            width = int(parameters["win_size"])
            if length < width:
                raise ValueError(f"DCdetector test length {length} requires win_size={width}")
            return {"start": length % width, "stop": length, "count": (length // width) * width,
                    "window_samples": length // width, "native_tail_zeros": 0}
        if self.method_name == "FITS":
            width = int(parameters["out_len"])
            if length < width:
                raise ValueError(f"FITS test length {length} requires out_len={width}")
            return {"start": width - 1, "stop": length, "count": length - width + 1,
                    "window_samples": length - width + 1, "native_tail_zeros": 0}
        if self.method_name not in ("LSTMADalpha", "LSTMADbeta"):
            return None
        window, horizon = int(parameters["window_size"]), int(parameters["pred_len"])
        minimum = window + 2 * horizon
        if length < minimum:
            raise ValueError(f"{self.method_name} test length {length} requires at least window_size + 2*pred_len = {minimum}; shorter input gives invalid or only-padding scores")
        return {"start": window + horizon, "stop": length, "count": length - window - horizon,
                "window_samples": length - window - horizon + 1, "native_tail_zeros": horizon - 1}


TRAINING_FREE = SchemaCapability(
    fit_scope="training_free",
)
PER_CURVE_FIT = SchemaCapability(
    fit_scope="per_curve",
)
NO_SHARED_FIT = SchemaCapability(
    fit_scope=None,
    unavailable_reason="the official Method does not fit shared state",
)


_METHOD_CATALOG = {
    "IsolationForest": AlgorithmSpec(
        method_name="IsolationForest",
        module="timesigma.algorithms.isolation_forest",
        requirements=("numpy", "sklearn", "joblib"),
        schemas={schema: SchemaCapability(
            fit_scope="per_curve" if schema == "naive" else "shared_train",
        ) for schema in EASYTSAD_SCHEMAS},
        artifact_type="fitted_estimator",
        artifact_attribute="estimator",
        sample_spec=SampleSpec(),
        python_requires=">=3.11,<3.13",
        agent_available=True,
        searchable_parameters=(),
    ),
    "FCVAE": AlgorithmSpec(
        method_name="FCVAE",
        module="EasyTSAD.Methods.FCVAE",
        requirements=("numpy", "torch", "torchinfo", "tqdm"),
        schemas={schema: SchemaCapability(
            fit_scope="per_curve" if schema == "naive" else "shared_train",
            selection_objective="native_cvae_validation_loss_plus_one" if schema == "naive" else "native_augmented_cvae_validation_loss_plus_one",
            selection_policy=BEST_UNLABELED_VALIDATION,
        ) for schema in EASYTSAD_SCHEMAS},
        artifact_type="torch_state_dict",
        sample_spec=SampleSpec(window_parameter="window"),
        agent_available=True,
        searchable_parameters=("window", "learning_rate", "latent_dim"),
    ),
    "LSTMADalpha": AlgorithmSpec(
        method_name="LSTMADalpha",
        module="EasyTSAD.Methods.LSTMADalpha",
        requirements=("numpy", "torch", "torchinfo", "tqdm"),
        schemas={schema: SchemaCapability(
            fit_scope="per_curve" if schema == "naive" else "shared_train",
            selection_objective="forecasting_mse",
            selection_policy=BEST_UNLABELED_VALIDATION,
        ) for schema in EASYTSAD_SCHEMAS},
        artifact_type="torch_state_dict",
        sample_spec=SampleSpec(window_parameter="window_size", horizon=3),
        agent_available=True,
        searchable_parameters=("window_size", "lr", "hidden_dim"),
    ),
    "LSTMADbeta": AlgorithmSpec(
        method_name="LSTMADbeta",
        module="EasyTSAD.Methods.LSTMADbeta",
        requirements=("numpy", "torch", "torchinfo", "tqdm"),
        schemas={schema: SchemaCapability(
            fit_scope="per_curve" if schema == "naive" else "shared_train",
            selection_objective="native_train_plus_validation_mse" if schema == "naive" else "forecasting_mse",
            selection_policy=BEST_UNLABELED_VALIDATION,
        ) for schema in EASYTSAD_SCHEMAS},
        artifact_type="torch_state_dict",
        sample_spec=SampleSpec(window_parameter="window_size", horizon=3),
        agent_available=True,
        searchable_parameters=("window_size", "lr", "hidden_dim"),
    ),
    "FITS": AlgorithmSpec(
        method_name="FITS",
        module="EasyTSAD.Methods.FITS",
        requirements=("numpy", "torch", "torchinfo", "tqdm"),
        schemas={schema: SchemaCapability(
            fit_scope="per_curve" if schema == "naive" else "shared_train",
            selection_objective="reconstruction_mse",
            selection_policy=BEST_UNLABELED_VALIDATION,
        ) for schema in EASYTSAD_SCHEMAS},
        artifact_type="torch_state_dict",
        sample_spec=SampleSpec(window_parameter="out_len"),
        agent_available=True,
        searchable_parameters=("out_len", "cut_freq", "lr"),
    ),
    "DCdetector": AlgorithmSpec(
        method_name="DCdetector",
        module="EasyTSAD.Methods.DCdetector",
        requirements=("numpy", "torch", "torchinfo", "einops"),
        schemas={schema: SchemaCapability(
            fit_scope="per_curve" if schema == "naive" else "shared_train",
            selection_objective="association_discrepancy",
            selection_policy=BEST_UNLABELED_VALIDATION,
        ) for schema in EASYTSAD_SCHEMAS},
        artifact_type="torch_state_dict",
        sample_spec=SampleSpec(window_parameter="win_size", step_parameter="step"),
        agent_available=True,
        searchable_parameters=("win_size", "learning_rate", "d_model"),
    ),
    "MatrixProfile": AlgorithmSpec(
        method_name="MatrixProfile",
        module="EasyTSAD.Methods.MatrixProfile",
        requirements=("matrixprofile", "numpy"),
        schemas={schema: TRAINING_FREE for schema in EASYTSAD_SCHEMAS},
        artifact_type="replay_recipe",
        sample_spec=SampleSpec(),
        python_requires=">=3.9,<3.10",
        shared_runtime="easytsad-python39",
    ),
    "OCSVM": AlgorithmSpec(
        method_name="OCSVM",
        module="EasyTSAD.Methods.OCSVM",
        requirements=("numpy", "scikit-learn"),
        schemas={
            "naive": PER_CURVE_FIT,
            "all_in_one": NO_SHARED_FIT,
            "zero_shot": NO_SHARED_FIT,
        },
        artifact_type="fitted_estimator",
        artifact_attribute="ocsvm",
        sample_spec=SampleSpec(),
    ),
    "LOF": AlgorithmSpec(
        method_name="LOF",
        module="EasyTSAD.Methods.LOF",
        requirements=("numpy", "scikit-learn"),
        schemas={
            "naive": PER_CURVE_FIT,
            "all_in_one": NO_SHARED_FIT,
            "zero_shot": NO_SHARED_FIT,
        },
        artifact_type="fitted_estimator",
        artifact_attribute="lof",
        sample_spec=SampleSpec(),
    ),
    "AR": AlgorithmSpec(
        method_name="AR",
        module="EasyTSAD.Methods.AR",
        requirements=("numpy", "torch", "torchinfo", "tqdm"),
        schemas={
            "naive": SchemaCapability(
                fit_scope="per_curve",
                selection_objective="forecasting_mse",
                selection_policy=BEST_UNLABELED_VALIDATION,
            ),
            "all_in_one": SchemaCapability(
                fit_scope="shared_train",
                selection_objective="forecasting_mse",
                selection_policy=BEST_UNLABELED_VALIDATION,
            ),
            "zero_shot": SchemaCapability(
                fit_scope="shared_train",
                selection_objective="forecasting_mse",
                selection_policy=BEST_UNLABELED_VALIDATION,
            ),
        },
        artifact_type="torch_state_dict",
        sample_spec=SampleSpec(window_parameter="p", horizon=1),
    ),
    "AE": AlgorithmSpec(
        method_name="AE",
        module="EasyTSAD.Methods.AE",
        requirements=("numpy", "torch", "torchinfo", "tqdm"),
        schemas={
            "naive": SchemaCapability(
                fit_scope="per_curve",
                selection_objective="reconstruction_mse",
                selection_policy=BEST_UNLABELED_VALIDATION,
            ),
            "all_in_one": SchemaCapability(
                fit_scope="shared_train",
                selection_objective="reconstruction_mse",
                selection_policy=BEST_UNLABELED_VALIDATION,
            ),
            "zero_shot": SchemaCapability(
                fit_scope="shared_train",
                selection_objective="reconstruction_mse",
                selection_policy=BEST_UNLABELED_VALIDATION,
            ),
        },
        artifact_type="torch_state_dict",
        sample_spec=SampleSpec(window_parameter="p"),
    ),
    "AnomalyTransformer": AlgorithmSpec(
        method_name="AnomalyTransformer",
        module="EasyTSAD.Methods.AnomalyTransformer",
        requirements=("numpy", "torch", "torchinfo", "tqdm"),
        schemas={
            "naive": SchemaCapability(
                fit_scope="per_curve",
                selection_objective="reconstruction_association_loss",
                selection_policy=BEST_UNLABELED_VALIDATION,
            ),
            "all_in_one": SchemaCapability(
                fit_scope="shared_train",
                selection_objective="reconstruction_association_loss",
                selection_policy=BEST_UNLABELED_VALIDATION,
            ),
            "zero_shot": SchemaCapability(
                fit_scope="shared_train",
                selection_objective="reconstruction_association_loss",
                selection_policy=BEST_UNLABELED_VALIDATION,
            ),
        },
        artifact_type="torch_state_dict",
        sample_spec=SampleSpec(
            window_parameter="win_size", step_parameter="step"
        ),
    ),
}
METHOD_CATALOG: Mapping[str, AlgorithmSpec] = _METHOD_CATALOG


def get_method(method_name: str) -> AlgorithmSpec:
    return METHOD_CATALOG[method_name]
