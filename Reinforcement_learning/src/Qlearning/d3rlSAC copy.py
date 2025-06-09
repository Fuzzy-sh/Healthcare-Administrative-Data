from abc import ABCMeta, abstractmethod
from typing import (
    Union,
    Sequence,
    cast,
    TypeVar,
    Any,
    Protocol,
    Callable,
    runtime_checkable,
    Mapping,
    BinaryIO,
    Generic,
    Iterable,
    Optional,
    Iterator,
    NoReturn,
    Generator,
)
from torch import nn
from typing import NamedTuple, Optional
import torch
import torch.nn.functional as F
import dataclasses
from dataclasses import dataclass, field
import numpy.typing as npt
import numpy as np
from dataclasses_json import config, dataclass_json
import gymnasium
import gym
from gym.spaces import Box, Discrete
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from enum import Enum, IntEnum
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
import io
from d3rlpy.dataset import ReplayBuffer, ReplayBufferBase
from d3rlpy.constants import (
    LoggingStrategy,
    ActionSpace,
    IMPL_NOT_INITIALIZED_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
)
from typing_extensions import Self
from torch.optim.lr_scheduler import LRScheduler
import structlog
import collections

from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop
import json
import os
from torch.distributions import Categorical
import math

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime

from torch.cuda import CUDAGraph
from tqdm.auto import tqdm, trange


# Types
##################################################################

NDArray = npt.NDArray[Any]
TorchObservation = Union[torch.Tensor, Sequence[torch.Tensor]]
ObservationSequence = Union[NDArray, Sequence[NDArray]]
Float32NDArray = npt.NDArray[np.float32]
DType = npt.DTypeLike
Observation = Union[NDArray, Sequence[NDArray]]
Shape = Union[Sequence[int], Sequence[Sequence[int]]]
Int32NDArray = npt.NDArray[np.int32]

GymEnv = Union[gym.Env[Any, Any], gymnasium.Env[Any, Any]]


@runtime_checkable
class OptimizerWrapperProto(Protocol):
    @property
    def optim(self) -> Optimizer:
        raise NotImplementedError

    def state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError


# Version
##################################################################
__version__ = "2.8.1"


# Constants
##################################################################

# class ActionSpace(Enum):
#     CONTINUOUS = 1
#     DISCRETE = 2
#     BOTH = 3


# IMPL_NOT_INITIALIZED_ERROR = (
#     "The neural network parameters are not "
#     "initialized. Pleaes call build_with_dataset, "
#     "build_with_env, or directly call fit or "
#     "fit_online method."
# )


# class LoggingStrategy(Enum):
#     STEPS = "steps"
#     EPOCH = "epoch"


# DISCRETE_ACTION_SPACE_MISMATCH_ERROR = (
#     "The action-space of the given dataset is not compatible with the"
#     " algorithm. Please use discrete action-space algorithms. The algorithms"
#     " list is available below.\n"
#     f"https://d3rlpy.readthedocs.io/en/v{__version__}/references/algos.html"
# )

# CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR = (
#     "The action-space of the given dataset is not compatible with the"
#     " algorithm. Please use continuous action-space algorithms. The algorithm"
#     " list is available below.\n"
#     f"https://d3rlpy.readthedocs.io/en/v{__version__}/references/algos.html"
# )


# ittertool
##################################################################
T = TypeVar("T")


def last_flag(iterator: Iterable[T]) -> Iterator[tuple[bool, T]]:
    items = list(iterator)
    for i, item in enumerate(items):
        yield i == len(items) - 1, item


def first_flag(iterator: Iterable[T]) -> Iterator[tuple[bool, T]]:
    items = list(iterator)
    for i, item in enumerate(items):
        yield i == 0, item


# Logger
##################################################################


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
)


LOG: structlog.BoundLogger = structlog.get_logger(__name__)


class ModuleProtocol(Protocol):
    def get_torch_modules(self) -> dict[str, nn.Module]: ...
    def get_gradients(self) -> Iterator[tuple[str, Float32NDArray]]: ...


class ImplProtocol(Protocol):
    @property
    def modules(self) -> ModuleProtocol:
        raise NotImplementedError


class AlgProtocol(Protocol):
    @property
    def impl(self) -> Optional[ImplProtocol]:
        raise NotImplementedError


class SaveProtocol(Protocol):
    def save(self, fname: str) -> None: ...


class LoggerAdapter(Protocol):
    r"""Interface of LoggerAdapter."""

    def write_params(self, params: dict[str, Any]) -> None:
        r"""Writes hyperparameters.

        Args:
            params: Dictionary of hyperparameters.
        """

    def before_write_metric(self, epoch: int, step: int) -> None:
        r"""Callback executed before write_metric method.

        Args:
            epoch: Epoch.
            step: Training step.
        """

    def write_metric(self, epoch: int, step: int, name: str, value: float) -> None:
        r"""Writes metric.

        Args:
            epoch: Epoch.
            step: Training step.
            name: Metric name.
            value: Metric value.
        """

    def after_write_metric(self, epoch: int, step: int) -> None:
        r"""Callback executed after write_metric method.

        Args:
            epoch: Epoch.
            step: Training step.
        """

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        r"""Saves models.

        Args:
            epoch: Epoch.
            algo: Algorithm that provides ``save`` method.
        """

    def close(self) -> None:
        r"""Closes this LoggerAdapter."""

    def watch_model(
        self,
        epoch: int,
        step: int,
    ) -> None:
        r"""Watch model parameters / gradients during training.

        Args:
            epoch: Epoch.
            step: Training step.
        """


class LoggerAdapterFactory(Protocol):
    r"""Interface of LoggerAdapterFactory."""

    def create(
        self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> LoggerAdapter:
        r"""Creates LoggerAdapter.

        This method instantiates ``LoggerAdapter`` with a given
        ``experiment_name``.
        This method is usually called at the beginning of training.

        Args:
            algo: Algorithm.
            experiment_name: Experiment name.
            steps_per_epoch: Number of steps per epoch.
        """
        raise NotImplementedError


class D3RLPyLogger:
    _algo: AlgProtocol
    _adapter: LoggerAdapter
    _experiment_name: str
    _metrics_buffer: defaultdict[str, list[float]]

    def __init__(
        self,
        algo: AlgProtocol,
        adapter_factory: LoggerAdapterFactory,
        experiment_name: str,
        n_steps_per_epoch: int,
        with_timestamp: bool = True,
    ):
        if with_timestamp:
            date = datetime.now().strftime("%Y%m%d%H%M%S")
            self._experiment_name = experiment_name + "_" + date
        else:
            self._experiment_name = experiment_name
        self._algo = algo
        self._adapter = adapter_factory.create(
            algo, self._experiment_name, n_steps_per_epoch
        )
        self._metrics_buffer = defaultdict(list)

    def add_params(self, params: dict[str, Any]) -> None:
        self._adapter.write_params(params)
        LOG.info("Parameters", params=params)

    def add_metric(self, name: str, value: float) -> None:
        self._metrics_buffer[name].append(value)

    def commit(self, epoch: int, step: int) -> dict[str, float]:
        self._adapter.before_write_metric(epoch, step)

        metrics = {}
        for name, buffer in self._metrics_buffer.items():
            metric = sum(buffer) / len(buffer)
            self._adapter.write_metric(epoch, step, name, metric)
            metrics[name] = metric

        LOG.info(
            f"{self._experiment_name}: epoch={epoch} step={step}",
            epoch=epoch,
            step=step,
            metrics=metrics,
        )

        self._adapter.after_write_metric(epoch, step)

        # save model parameter metrics
        self._adapter.watch_model(epoch, step)

        # initialize metrics buffer
        self._metrics_buffer.clear()
        return metrics

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        self._adapter.save_model(epoch, algo)

    def close(self) -> None:
        self._adapter.close()

    @contextmanager
    def measure_time(self, name: str) -> Iterator[None]:
        name = "time_" + name
        start = time.time()
        try:
            yield
        finally:
            self.add_metric(name, time.time() - start)

    @property
    def adapter(self) -> LoggerAdapter:
        return self._adapter


# Logging file adapter
##################################################################


# default json encoder for numpy objects
def default_json_encoder(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (Enum, IntEnum)):
        return obj.value
    raise ValueError(f"invalid object type: {type(obj)}")


class FileAdapter(LoggerAdapter):
    r"""FileAdapter class.

    This class saves metrics as CSV files, hyperparameters as json file and
    models as d3 files.

    Args:
        algo: Algorithm.
        logdir (str): Log directory.
    """

    _algo: AlgProtocol
    _logdir: str
    _is_model_watched: bool

    def __init__(self, algo: AlgProtocol, logdir: str):
        self._algo = algo
        self._logdir = logdir
        self._is_model_watched = False
        if not os.path.exists(self._logdir):
            os.makedirs(self._logdir)
            LOG.info(f"Directory is created at {self._logdir}")

    def write_params(self, params: dict[str, Any]) -> None:
        # save dictionary as json file
        params_path = os.path.join(self._logdir, "params.json")
        with open(params_path, "w") as f:
            json_str = json.dumps(params, default=default_json_encoder, indent=2)
            f.write(json_str)

    def before_write_metric(self, epoch: int, step: int) -> None:
        pass

    def write_metric(self, epoch: int, step: int, name: str, value: float) -> None:
        path = os.path.join(self._logdir, f"{name}.csv")
        with open(path, "a") as f:
            print(f"{epoch},{step},{value}", file=f)

    def after_write_metric(self, epoch: int, step: int) -> None:
        pass

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        # save entire model
        model_path = os.path.join(self._logdir, f"model_{epoch}.d3")
        algo.save(model_path)
        LOG.info(f"Model parameters are saved to {model_path}")

    def close(self) -> None:
        pass

    @property
    def logdir(self) -> str:
        return self._logdir

    def watch_model(
        self,
        epoch: int,
        step: int,
    ) -> None:
        assert self._algo.impl

        # write header at the first call
        if not self._is_model_watched:
            self._is_model_watched = True
            for name, grad in self._algo.impl.modules.get_gradients():
                path = os.path.join(self._logdir, f"{name}_grad.csv")
                with open(path, "w") as f:
                    print(
                        ",".join(["epoch", "step", "min", "max", "mean", "std"]),
                        file=f,
                    )

        for name, grad in self._algo.impl.modules.get_gradients():
            path = os.path.join(self._logdir, f"{name}_grad.csv")
            with open(path, "a") as f:
                min_grad = grad.min()
                max_grad = grad.max()
                mean = grad.mean()
                std = grad.std()
                print(
                    f"{epoch},{step},{min_grad},{max_grad},{mean},{std}",
                    file=f,
                )


class FileAdapterFactory(LoggerAdapterFactory):
    r"""FileAdapterFactory class.

    This class instantiates ``FileAdapter`` object.
    Log directory will be created at ``<root_dir>/<experiment_name>``.

    Args:
        root_dir (str): Top-level log directory.
    """

    _root_dir: str

    def __init__(self, root_dir: str = "d3rlpy_logs"):
        self._root_dir = root_dir

    def create(
        self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> FileAdapter:
        logdir = os.path.join(self._root_dir, experiment_name)
        return FileAdapter(algo, logdir)


# Config Serialization
#################################################################

TConfig = TypeVar("TConfig", bound="SerializableConfig")
TDynamicConfig = TypeVar("TDynamicConfig", bound="DynamicConfig")


@dataclass_json
@dataclasses.dataclass()
class SerializableConfig:
    def serialize(self) -> str:
        return self.to_json()  # type: ignore

    def serialize_to_dict(self) -> dict[str, Any]:
        return self.to_dict()  # type: ignore

    @classmethod
    def deserialize(cls: type[TConfig], serialized_config: str) -> TConfig:
        return cls.from_json(serialized_config)  # type: ignore

    @classmethod
    def deserialize_from_dict(
        cls: type[TConfig], dict_config: dict[str, Any]
    ) -> TConfig:
        return cls.from_dict(dict_config)  # type: ignore

    @classmethod
    def deserialize_from_file(cls: type[TConfig], path: str) -> TConfig:
        with open(path, "r") as f:
            return cls.deserialize(f.read())


class DynamicConfig(SerializableConfig):
    @staticmethod
    def get_type() -> str:
        raise NotImplementedError


def _numpy_decoder(v: Sequence[float]) -> NDArray:
    return np.array(v)


# setup numpy encoder/decoder
def _numpy_encoder(v: NDArray) -> Sequence[float]:
    return v.tolist()  # type: ignore


def make_optional_numpy_field() -> Optional[NDArray]:
    return dataclasses.field(
        metadata=config(encoder=_numpy_encoder, decoder=_numpy_decoder),
        default=None,
    )


@dataclasses.dataclass(frozen=True)
class ConfigMetadata:
    base_cls: type[DynamicConfig]
    encoder: Callable[[DynamicConfig], dict[str, Any]]
    decoder: Callable[[dict[str, Any]], Union[DynamicConfig, Optional[DynamicConfig]]]
    config_list: dict[str, type[DynamicConfig]]

    def add_config(self, name: str, new_config: type[DynamicConfig]) -> None:
        assert name not in self.config_list, f"{name} is already registered"
        self.config_list[name] = new_config


CONFIG_STORAGE: dict[type[DynamicConfig], ConfigMetadata] = {}


def generate_optional_config_generation(
    base_cls: type[TDynamicConfig],
) -> tuple[
    Callable[[type[TDynamicConfig]], None],
    Callable[[], Optional[TDynamicConfig]],
]:
    CONFIG_LIST: dict[str, type[TDynamicConfig]] = {}

    def register_config(cls: type[TDynamicConfig]) -> None:
        assert issubclass(cls, base_cls)
        type_name = cls.get_type()
        is_registered = type_name in CONFIG_LIST
        assert not is_registered, f"{type_name} seems to be already registered"
        CONFIG_LIST[type_name] = cls

    def _encoder(orig_config: Optional[TDynamicConfig]) -> dict[str, Any]:
        if orig_config is None:
            return {"type": "none", "params": {}}
        return {
            "type": orig_config.get_type(),
            "params": orig_config.serialize_to_dict(),
        }

    def _decoder(dict_config: dict[str, Any]) -> Optional[TDynamicConfig]:
        name = dict_config["type"]
        params = dict_config["params"]
        if name == "none":
            return None
        return CONFIG_LIST[name].deserialize_from_dict(params)

    config_metadata = ConfigMetadata(
        base_cls=base_cls,
        encoder=_encoder,  # type: ignore
        decoder=_decoder,
        config_list=CONFIG_LIST,  # type: ignore
    )
    CONFIG_STORAGE[base_cls] = config_metadata

    def make_field() -> Optional[TDynamicConfig]:
        return dataclasses.field(
            metadata=config(encoder=_encoder, decoder=_decoder),
            default=None,
        )

    return register_config, make_field


def generate_list_config_field(
    base_cls: type[TDynamicConfig],
) -> Callable[[], Sequence[TDynamicConfig]]:
    assert base_cls in CONFIG_STORAGE

    config_metadata = CONFIG_STORAGE[base_cls]

    def _encoder(
        orig_config: Sequence[TDynamicConfig],
    ) -> Sequence[dict[str, Any]]:
        return [config_metadata.encoder(config) for config in orig_config]

    def _decoder(
        dict_config: Sequence[dict[str, Any]],
    ) -> Sequence[TDynamicConfig]:
        configs = [config_metadata.decoder(config) for config in dict_config]
        return configs  # type: ignore

    def make_field() -> Sequence[TDynamicConfig]:
        return dataclasses.field(
            metadata=config(encoder=_encoder, decoder=_decoder),
            default_factory=list,
        )

    return make_field


def generate_config_registration(
    base_cls: type[TDynamicConfig],
    default_factory: Optional[Callable[[], TDynamicConfig]] = None,
) -> tuple[Callable[[type[TDynamicConfig]], None], Callable[[], TDynamicConfig]]:
    CONFIG_LIST: dict[str, type[TDynamicConfig]] = {}

    def register_config(cls: type[TDynamicConfig]) -> None:
        assert issubclass(cls, base_cls)
        type_name = cls.get_type()
        is_registered = type_name in CONFIG_LIST
        assert not is_registered, f"{type_name} seems to be already registered"
        CONFIG_LIST[type_name] = cls

    def _encoder(orig_config: TDynamicConfig) -> dict[str, Any]:
        return {
            "type": orig_config.get_type(),
            "params": orig_config.serialize_to_dict(),
        }

    def _decoder(dict_config: dict[str, Any]) -> TDynamicConfig:
        name = dict_config["type"]
        params = dict_config["params"]
        return CONFIG_LIST[name].deserialize_from_dict(params)

    config_metadata = ConfigMetadata(
        base_cls=base_cls,
        encoder=_encoder,  # type: ignore
        decoder=_decoder,
        config_list=CONFIG_LIST,  # type: ignore
    )
    CONFIG_STORAGE[base_cls] = config_metadata

    if default_factory is None:

        def make_field() -> TDynamicConfig:
            field = cast(
                TDynamicConfig,
                dataclasses.field(metadata=config(encoder=_encoder, decoder=_decoder)),
            )
            return field

    else:

        def make_field() -> TDynamicConfig:
            return dataclasses.field(
                metadata=config(encoder=_encoder, decoder=_decoder),
                default_factory=default_factory,
            )

    return register_config, make_field


# Data calss utils
#####################################################################
def asdict_without_copy(obj: Any) -> dict[str, Any]:
    assert dataclasses.is_dataclass(obj)
    fields = dataclasses.fields(obj)
    return {field.name: getattr(obj, field.name) for field in fields}


# Dataset
#####################################################################
# utils
def is_tuple_shape(shape: Shape) -> bool:
    return isinstance(shape[0], (list, tuple))


def cast_flat_shape(shape: Shape) -> Sequence[int]:
    assert not is_tuple_shape(shape)
    return shape  # type: ignore


def get_shape_from_observation(observation: Observation) -> Shape:
    if isinstance(observation, np.ndarray):
        return observation.shape
    elif isinstance(observation, (list, tuple)):
        return [obs.shape for obs in observation]
    else:
        raise ValueError(f"invalid observation type: {type(observation)}")


def get_dtype_from_observation(
    observation: Observation,
) -> Union[DType, Sequence[DType]]:
    if isinstance(observation, np.ndarray):
        return observation.dtype
    elif isinstance(observation, (list, tuple)):
        return [obs.dtype for obs in observation]
    else:
        raise ValueError(f"invalid observation type: {type(observation)}")


_TDType = TypeVar("_TDType", bound=Any)


def cast_recursively(
    array: Union[NDArray, Sequence[NDArray]], dtype: type[_TDType]
) -> Union[npt.NDArray[_TDType], Sequence[npt.NDArray[_TDType]]]:
    if isinstance(array, (list, tuple)):
        return [array[i].astype(dtype) for i in range(len(array))]
    elif isinstance(array, np.ndarray):
        return array.astype(dtype)
    else:
        raise ValueError(f"invalid array type: {type(array)}")


def stack_observations(observations: Sequence[Observation]) -> Observation:
    if isinstance(observations[0], (list, tuple)):
        obs_kinds = len(observations[0])
        return [
            np.stack([obs[i] for obs in observations], axis=0) for i in range(obs_kinds)
        ]
    elif isinstance(observations[0], np.ndarray):
        return np.stack(observations, axis=0)
    else:
        raise ValueError(f"invalid observation type: {type(observations[0])}")


def detect_action_size_from_env(env: GymEnv) -> int:
    if isinstance(env.action_space, (Discrete, GymnasiumDiscrete)):
        action_size = env.action_space.n
    elif isinstance(env.action_space, (Box, GymnasiumBox)):
        action_size = env.action_space.shape[0]
    else:
        raise ValueError(f"Unsupported action_space: {type(env.action_space)}")
    return int(action_size)


def check_dtype(array: Union[NDArray, Sequence[NDArray]], dtype: DType) -> bool:
    if isinstance(array, (list, tuple)):
        return all(v.dtype == dtype for v in array)
    elif isinstance(array, np.ndarray):
        return array.dtype == dtype
    else:
        raise ValueError(f"invalid array type: {type(array)}")


def check_non_1d_array(array: Union[NDArray, Sequence[NDArray]]) -> bool:
    if isinstance(array, (list, tuple)):
        return all(v.ndim > 1 for v in array)
    elif isinstance(array, np.ndarray):
        return array.ndim > 1
    else:
        raise ValueError(f"invalid array type: {type(array)}")


#  dataset components
##################################################################
@dataclasses.dataclass(frozen=True)
class Signature:
    r"""Signature of arrays.

    Args:
        dtype: List of numpy data types.
        shape: List of array shapes.
    """

    dtype: Sequence[DType]
    shape: Sequence[Sequence[int]]

    def sample(self) -> Sequence[NDArray]:
        r"""Returns sampled arrays.

        Returns:
            List of arrays based on dtypes and shapes.
        """
        return [
            np.random.random(shape).astype(dtype)
            for shape, dtype in zip(self.shape, self.dtype)
        ]


def get_shape_from_observation_sequence(
    observations: ObservationSequence,
) -> Shape:
    if isinstance(observations, np.ndarray):
        return observations.shape[1:]
    elif isinstance(observations, (list, tuple)):
        return [obs.shape[1:] for obs in observations]
    else:
        raise ValueError(f"invalid observation type: {type(observations)}")


def get_dtype_from_observation_sequence(
    observations: ObservationSequence,
) -> Union[DType, Sequence[DType]]:
    if isinstance(observations, np.ndarray):
        return observations.dtype
    elif isinstance(observations, (list, tuple)):
        return [obs.dtype for obs in observations]
    else:
        raise ValueError(f"invalid observation type: {type(observations)}")


@dataclasses.dataclass(frozen=True)
class PartialTrajectory:
    r"""Partial trajectory.

    Args:
        observations: Sequence of observations.
        actions: Sequence of actions.
        rewards: Sequence of rewards.
        returns_to_go: Sequence of remaining returns.
        terminals: Sequence of terminal flags.
        timesteps: Sequence of timesteps.
        masks: Sequence of masks that represent padding.
        length: Sequence length.
    """

    observations: ObservationSequence  # (L, ...)
    actions: NDArray  # (L, ...)
    rewards: Float32NDArray  # (L, 1)
    returns_to_go: Float32NDArray  # (L, 1)
    terminals: Float32NDArray  # (L, 1)
    timesteps: Int32NDArray  # (L,)
    masks: Float32NDArray  # (L,)
    length: int

    @property
    def observation_signature(self) -> Signature:
        r"""Returns observation sigunature.

        Returns:
            Observation signature.
        """
        shape = get_shape_from_observation_sequence(self.observations)
        dtype = get_dtype_from_observation_sequence(self.observations)
        if isinstance(self.observations, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        r"""Returns action signature.

        Returns:
            Action signature.
        """
        return Signature(
            dtype=[self.actions.dtype],
            shape=[self.actions.shape[1:]],
        )

    @property
    def reward_signature(self) -> Signature:
        r"""Returns reward signature.

        Returns:
            Reward signature.
        """
        return Signature(
            dtype=[self.rewards.dtype],
            shape=[self.rewards.shape[1:]],
        )

    def get_transition_count(self) -> int:
        """Returns number of transitions.

        Returns:
            Number of transitions.
        """
        return self.length if bool(self.terminals[-1]) else self.length - 1

    def __len__(self) -> int:
        return self.length


class EpisodeBase(Protocol):
    r"""Episode interface.

    ``Episode`` represens an entire episode.
    """

    @property
    def observations(self) -> ObservationSequence:
        r"""Returns sequence of observations.

        Returns:
            Sequence of observations.
        """
        raise NotImplementedError

    @property
    def actions(self) -> NDArray:
        r"""Returns sequence of actions.

        Returns:
            Sequence of actions.
        """
        raise NotImplementedError

    @property
    def rewards(self) -> Float32NDArray:
        r"""Returns sequence of rewards.

        Returns:
            Sequence of rewards.
        """
        raise NotImplementedError

    @property
    def terminated(self) -> bool:
        r"""Returns environment terminal flag.

        This flag becomes true when this episode is terminated. For timeout,
        this flag stays false.

        Returns:
            Terminal flag.
        """
        raise NotImplementedError

    @property
    def observation_signature(self) -> Signature:
        r"""Returns observation signature.

        Returns:
            Observation signature.
        """
        raise NotImplementedError

    @property
    def action_signature(self) -> Signature:
        r"""Returns action signature.

        Returns:
            Action signature.
        """
        raise NotImplementedError

    @property
    def reward_signature(self) -> Signature:
        r"""Returns reward signature.

        Returns:
            Reward signature.
        """
        raise NotImplementedError

    def size(self) -> int:
        r"""Returns length of an episode.

        Returns:
            Episode length.
        """
        raise NotImplementedError

    def compute_return(self) -> float:
        r"""Computes total episode return.

        Returns:
            Total episode return.
        """
        raise NotImplementedError

    def serialize(self) -> dict[str, Any]:
        r"""Returns serized episode data.

        Returns:
            Serialized episode data.
        """
        raise NotImplementedError

    @classmethod
    def deserialize(cls, serializedData: dict[str, Any]) -> "EpisodeBase":
        r"""Constructs episode from serialized data.

        This is an inverse operation of ``serialize`` method.

        Args:
            serializedData: Serialized episode data.

        Returns:
            Episode object.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def transition_count(self) -> int:
        r"""Returns the number of transitions.

        Returns:
            Number of transitions.
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class DatasetInfo:
    r"""Dataset information.

    Args:
        observation_signature: Observation signature.
        action_signature: Action signature.
        reward_signature: Reward signature.
        action_space: Action space type.
        action_size: Size of action-space. For continuous action-space,
            this represents dimension of action vectors. For discrete
            action-space, this represents the number of discrete actions.
    """

    observation_signature: Signature
    action_signature: Signature
    reward_signature: Signature
    action_space: ActionSpace
    action_size: int


#  dataset Transition picker
##################################################################
#


@dataclasses.dataclass(frozen=True)
class Transition:
    r"""Transition tuple.

    Args:
        observation: Observation.
        action: Action
        reward: Reward. This could be a multi-step discounted return.
        next_observation: Observation at next timestep. This could be
            observation at multi-step ahead.
        next_action: Action at next timestep. This could be action at
            multi-step ahead.
        terminal: Flag of environment termination.
        interval: Timesteps between ``observation`` and ``next_observation``.
        rewards_to_go: Remaining rewards till the end of an episode, which is
            used to compute returns_to_go.
    """

    observation: Observation  # (...)
    action: NDArray  # (...)
    reward: Float32NDArray  # (1,)
    next_observation: Observation  # (...)
    next_action: NDArray  # (...)
    terminal: float
    interval: int
    rewards_to_go: Float32NDArray  # (L, 1)

    @property
    def observation_signature(self) -> Signature:
        r"""Returns observation sigunature.

        Returns:
            Observation signature.
        """
        shape = get_shape_from_observation(self.observation)
        dtype = get_dtype_from_observation(self.observation)
        if isinstance(self.observation, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        r"""Returns action signature.

        Returns:
            Action signature.
        """
        return Signature(
            dtype=[self.action.dtype],
            shape=[self.action.shape],
        )

    @property
    def reward_signature(self) -> Signature:
        r"""Returns reward signature.

        Returns:
            Reward signature.
        """
        return Signature(
            dtype=[self.reward.dtype],
            shape=[self.reward.shape],
        )


class TransitionPickerProtocol(Protocol):
    r"""Interface of TransitionPicker."""

    def __call__(self, episode: EpisodeBase, index: int) -> Transition:
        r"""Returns transition specified by ``index``.

        Args:
            episode: Episode.
            index: Index at the target transition.

        Returns:
            Transition.
        """
        raise NotImplementedError


class TrajectorySlicerProtocol(Protocol):
    r"""Interface of TrajectorySlicer."""

    def __call__(
        self, episode: EpisodeBase, end_index: int, size: int
    ) -> PartialTrajectory:
        r"""Slice trajectory.

        This method returns a partial trajectory from ``t=end_index-size`` to
        ``t=end_index``. If ``end_index-size`` is smaller than 0, those parts
        will be padded by zeros.

        Args:
            episode: Episode.
            end_index: Index at the end of the sliced trajectory.
            size: Length of the sliced trajectory.

        Returns:
            Sliced trajectory.
        """
        raise NotImplementedError


#  dataset Mini batch
##################################################################
#


@dataclasses.dataclass(frozen=True)
class TransitionMiniBatch:
    r"""Mini-batch of transitions.

    Args:
        observations: Batched observations.
        actions: Batched actions.
        rewards: Batched rewards.
        next_observations: Batched next observations.
        returns_to_go: Batched returns-to-go.
        terminals: Batched environment terminal flags.
        intervals: Batched timesteps between observations and next
            observations.
        transitions: List of transitions.
    """

    observations: Union[Float32NDArray, Sequence[Float32NDArray]]  # (B, ...)
    actions: Float32NDArray  # (B, ...)
    rewards: Float32NDArray  # (B, 1)
    next_observations: Union[Float32NDArray, Sequence[Float32NDArray]]  # (B, ...)
    next_actions: Float32NDArray  # (B, ...)
    terminals: Float32NDArray  # (B, 1)
    intervals: Float32NDArray  # (B, 1)
    transitions: Sequence[Transition]

    def __post_init__(self) -> None:
        assert check_non_1d_array(self.observations)
        assert check_dtype(self.observations, np.float32)
        assert check_non_1d_array(self.actions)
        assert check_dtype(self.actions, np.float32)
        assert check_non_1d_array(self.next_actions)
        assert check_dtype(self.next_actions, np.float32)
        assert check_non_1d_array(self.rewards)
        assert check_dtype(self.rewards, np.float32)
        assert check_non_1d_array(self.next_observations)
        assert check_dtype(self.next_observations, np.float32)
        assert check_non_1d_array(self.terminals)
        assert check_dtype(self.terminals, np.float32)
        assert check_non_1d_array(self.intervals)
        assert check_dtype(self.intervals, np.float32)

    @classmethod
    def from_transitions(
        cls, transitions: Sequence[Transition]
    ) -> "TransitionMiniBatch":
        r"""Constructs mini-batch from list of transitions.

        Args:
            transitions: List of transitions.

        Returns:
            Mini-batch.
        """
        observations = stack_observations(
            [transition.observation for transition in transitions]
        )
        actions = np.stack([transition.action for transition in transitions], axis=0)
        rewards = np.stack([transition.reward for transition in transitions], axis=0)
        next_observations = stack_observations(
            [transition.next_observation for transition in transitions]
        )
        next_actions = np.stack(
            [transition.next_action for transition in transitions], axis=0
        )
        terminals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )
        intervals = np.reshape(
            np.array([transition.interval for transition in transitions]),
            [-1, 1],
        )
        return TransitionMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            next_observations=cast_recursively(next_observations, np.float32),
            next_actions=cast_recursively(next_actions, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            intervals=cast_recursively(intervals, np.float32),
            transitions=transitions,
        )

    @property
    def observation_shape(self) -> Shape:
        r"""Returns observation shape.

        Returns:
            Observation shape.
        """
        return get_shape_from_observation_sequence(self.observations)

    @property
    def action_shape(self) -> Sequence[int]:
        r"""Returns action shape.

        Returns:
            Action shape.
        """
        return self.actions.shape[1:]

    @property
    def reward_shape(self) -> Sequence[int]:
        r"""Returns reward shape.

        Returns:
            Reward shape.
        """
        return self.rewards.shape[1:]

    def __len__(self) -> int:
        return int(self.actions.shape[0])


@dataclasses.dataclass(frozen=True)
class TrajectoryMiniBatch:
    r"""Mini-batch of trajectories.

    Args:
        observations: Batched sequence of observations.
        actions: Batched sequence of actions.
        rewards: Batched sequence of rewards.
        returns_to_go: Batched sequence of returns-to-go.
        terminals: Batched sequence of environment terminal flags.
        timesteps: Batched sequence of environment timesteps.
        masks: Batched masks that represent padding.
        length: Length of trajectories.
    """

    observations: Union[Float32NDArray, Sequence[Float32NDArray]]  # (B, L, ...)
    actions: Float32NDArray  # (B, L, ...)
    rewards: Float32NDArray  # (B, L, 1)
    returns_to_go: Float32NDArray  # (B, L, 1)
    terminals: Float32NDArray  # (B, L, 1)
    timesteps: Float32NDArray  # (B, L)
    masks: Float32NDArray  # (B, L)
    length: int

    def __post_init__(self) -> None:
        assert check_dtype(self.observations, np.float32)
        assert check_dtype(self.actions, np.float32)
        assert check_dtype(self.rewards, np.float32)
        assert check_dtype(self.returns_to_go, np.float32)
        assert check_dtype(self.terminals, np.float32)
        assert check_dtype(self.timesteps, np.float32)
        assert check_dtype(self.masks, np.float32)

    @classmethod
    def from_partial_trajectories(
        cls, trajectories: Sequence[PartialTrajectory]
    ) -> "TrajectoryMiniBatch":
        r"""Constructs mini-batch from list of trajectories.

        Args:
            trajectories: List of trajectories.

        Returns:
            Mini-batch of trajectories.
        """
        observations = stack_observations([traj.observations for traj in trajectories])
        actions = np.stack([traj.actions for traj in trajectories], axis=0)
        rewards = np.stack([traj.rewards for traj in trajectories], axis=0)
        returns_to_go = np.stack([traj.returns_to_go for traj in trajectories], axis=0)
        terminals = np.stack([traj.terminals for traj in trajectories], axis=0)
        timesteps = np.stack([traj.timesteps for traj in trajectories], axis=0)
        masks = np.stack([traj.masks for traj in trajectories], axis=0)
        return TrajectoryMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            returns_to_go=cast_recursively(returns_to_go, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            timesteps=cast_recursively(timesteps, np.float32),
            masks=cast_recursively(masks, np.float32),
            length=trajectories[0].length,
        )

    @property
    def observation_shape(self) -> Shape:
        r"""Returns observation shape.

        Returns:
            Observation shape.
        """
        return get_shape_from_observation_sequence(self.observations)

    @property
    def action_shape(self) -> Sequence[int]:
        r"""Returns action shape.

        Returns:
            Action shape.
        """
        return self.actions.shape[1:]

    @property
    def reward_shape(self) -> Sequence[int]:
        r"""Returns reward shape.

        Returns:
            Reward shape.
        """
        return self.rewards.shape[1:]

    def __len__(self) -> int:
        return int(self.actions.shape[0])


# dataset writer
##################################################################
class WriterPreprocessProtocol(Protocol):
    r"""Interface of WriterPreprocess."""

    def process_observation(self, observation: Observation) -> Observation:
        r"""Processes observation.

        Args:
            observation: Observation.

        Returns:
            Processed observation.
        """
        raise NotImplementedError

    def process_action(self, action: NDArray) -> NDArray:
        r"""Processes action.

        Args:
            action: Action.

        Returns:
            Processed action.
        """
        raise NotImplementedError

    def process_reward(self, reward: NDArray) -> NDArray:
        r"""Processes reward.

        Args:
            reward: Reward.

        Returns:
            Processed reward.
        """
        raise NotImplementedError


# Preprocessing
#####################################################################
# base
class Scaler(DynamicConfig, metaclass=ABCMeta):
    @abstractmethod
    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            episodes: List of episodes.
            transition_picker: Transition picker to process mini-batch.
        """
        raise NotImplementedError


def add_leading_dims(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert x.ndim <= target.ndim
    dim_diff = target.ndim - x.ndim
    assert x.shape == target.shape[dim_diff:]
    return torch.reshape(x, [1] * dim_diff + list(x.shape))


def add_leading_dims_numpy(x: NDArray, target: NDArray) -> NDArray:
    assert x.ndim <= target.ndim
    dim_diff = target.ndim - x.ndim
    assert x.shape == target.shape[dim_diff:]
    return np.reshape(x, [1] * dim_diff + list(x.shape))


def save_config(
    algo: "LearnableBase[ImplBase, LearnableConfig]", logger: D3RLPyLogger
) -> None:
    assert algo.impl
    config = LearnableConfigWithShape(
        observation_shape=algo.impl.observation_shape,
        action_size=algo.impl.action_size,
        config=algo.config,
    )
    logger.add_params(config.serialize_to_dict())


# scaler
class ObservationScaler(Scaler):
    pass


class ActionScaler(Scaler):
    pass


class RewardScaler(Scaler):
    def fit_with_env(self, env: GymEnv) -> None:
        pass


@dataclasses.dataclass()
class ConstantShiftRewardScaler(RewardScaler):
    r"""Reward shift preprocessing.

    .. math::

        r' = r + c

    You need to initialize manually.

    .. code-block:: python

        from d3rlpy.preprocessing import ConstantShiftRewardScaler
        from d3rlpy.algos import CQLConfig

        reward_scaler = ConstantShiftRewardScaler(shift=-1.0)
        cql = CQLConfig(reward_scaler=reward_scaler).create()

    References:
        * `Kostrikov et al., Offline Reinforcement Learning with Implicit
          Q-Learning. <https://arxiv.org/abs/2110.06169>`_

    Args:
        shift (float): Constant shift value
        multiplier (float): Constant multiplication value.
        multiply_first (bool): Flag to multiply rewards and then shift.
    """

    shift: float
    multiplier: float = 1.0
    multiply_first: bool = False

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        pass

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiply_first:
            return x * self.multiplier + self.shift
        else:
            return (self.shift + x) * self.multiplier

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiply_first:
            return (x - self.shift) / self.multiplier
        else:
            return x / self.multiplier - self.shift

    def transform_numpy(self, x: NDArray) -> NDArray:
        if self.multiply_first:
            return x * self.multiplier + self.shift
        else:
            return (self.shift + x) * self.multiplier

    def reverse_transform_numpy(self, x: NDArray) -> NDArray:
        if self.multiply_first:
            return (x - self.shift) / self.multiplier
        else:
            return x / self.multiplier - self.shift

    @staticmethod
    def get_type() -> str:
        return "shift"

    @property
    def built(self) -> bool:
        return True


(
    register_reward_scaler,
    make_reward_scaler_field,
) = generate_optional_config_generation(
    RewardScaler  # type: ignore
)


@dataclasses.dataclass()
class MinMaxActionScaler(ActionScaler):
    r"""Min-Max normalization action preprocessing.

    Actions will be normalized in range ``[-1.0, 1.0]``.

    .. math::

        a' = (a - \min{a}) / (\max{a} - \min{a}) * 2 - 1

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxActionScaler
        from d3rlpy.algos import CQLConfig

        # normalize based on datasets or environments
        cql = CQLConfig(action_scaler=MinMaxActionScaler()).create()

        # manually initialize
        minimum = actions.min(axis=0)
        maximum = actions.max(axis=0)
        action_scaler = MinMaxActionScaler(minimum=minimum, maximum=maximum)
        cql = CQLConfig(action_scaler=action_scaler).create()

    Args:
        minimum (numpy.ndarray): Minimum values at each entry.
        maximum (numpy.ndarray): Maximum values at each entry.
    """

    minimum: Optional[NDArray] = make_optional_numpy_field()
    maximum: Optional[NDArray] = make_optional_numpy_field()

    def __post_init__(self) -> None:
        if self.minimum is not None:
            self.minimum = np.asarray(self.minimum)
        if self.maximum is not None:
            self.maximum = np.asarray(self.maximum)
        self._torch_minimum: Optional[torch.Tensor] = None
        self._torch_maximum: Optional[torch.Tensor] = None

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        assert not self.built
        minimum = np.zeros(episodes[0].action_signature.shape[0])
        maximum = np.zeros(episodes[0].action_signature.shape[0])
        for i, episode in enumerate(episodes):
            for j in range(episode.transition_count):
                transition = transition_picker(episode, j)
                if i == 0 and j == 0:
                    minimum = transition.action
                    maximum = transition.action
                else:
                    minimum = np.minimum(minimum, transition.action)
                    maximum = np.maximum(maximum, transition.action)
        self.minimum = minimum
        self.maximum = maximum

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        assert not self.built
        minimum = np.zeros(episodes[0].action_signature.shape[0])
        maximum = np.zeros(episodes[0].action_signature.shape[0])
        for i, episode in enumerate(episodes):
            traj = trajectory_slicer(episode, episode.size() - 1, episode.size())
            actions = np.asarray(traj.actions)
            min_action = np.min(actions, axis=0)
            max_action = np.max(actions, axis=0)
            if i == 0:
                minimum = min_action
                maximum = max_action
            else:
                minimum = np.minimum(minimum, min_action)
                maximum = np.maximum(maximum, max_action)
        self.minimum = minimum
        self.maximum = maximum

    def fit_with_env(self, env: GymEnv) -> None:
        assert not self.built
        assert isinstance(env.action_space, (Box, GymnasiumBox))
        low = np.asarray(env.action_space.low)
        high = np.asarray(env.action_space.high)
        self.minimum = low
        self.maximum = high

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        if self._torch_minimum is None or self._torch_maximum is None:
            self._set_torch_value(x.device)
        assert self._torch_minimum is not None and self._torch_maximum is not None
        minimum = add_leading_dims(self._torch_minimum, target=x)
        maximum = add_leading_dims(self._torch_maximum, target=x)
        # transform action into [-1.0, 1.0]
        return ((x - minimum) / (maximum - minimum)) * 2.0 - 1.0

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        if self._torch_minimum is None or self._torch_maximum is None:
            self._set_torch_value(x.device)
        assert self._torch_minimum is not None and self._torch_maximum is not None
        minimum = add_leading_dims(self._torch_minimum, target=x)
        maximum = add_leading_dims(self._torch_maximum, target=x)
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((x + 1.0) / 2.0)) + minimum

    def transform_numpy(self, x: NDArray) -> NDArray:
        assert self.built
        assert self.maximum is not None and self.minimum is not None
        minimum = add_leading_dims_numpy(self.minimum, target=x)
        maximum = add_leading_dims_numpy(self.maximum, target=x)
        # transform action into [-1.0, 1.0]
        ret = ((x - minimum) / (maximum - minimum)) * 2.0 - 1.0
        return ret  # type: ignore

    def reverse_transform_numpy(self, x: NDArray) -> NDArray:
        assert self.built
        assert self.maximum is not None and self.minimum is not None
        minimum = add_leading_dims_numpy(self.minimum, target=x)
        maximum = add_leading_dims_numpy(self.maximum, target=x)
        # transform action from [-1.0, 1.0]
        ret = ((maximum - minimum) * ((x + 1.0) / 2.0)) + minimum
        return ret  # type: ignore

    def _set_torch_value(self, device: torch.device) -> None:
        self._torch_minimum = torch.tensor(
            self.minimum, dtype=torch.float32, device=device
        )
        self._torch_maximum = torch.tensor(
            self.maximum, dtype=torch.float32, device=device
        )

    @staticmethod
    def get_type() -> str:
        return "min_max"

    @property
    def built(self) -> bool:
        return self.minimum is not None and self.maximum is not None


(
    register_action_scaler,
    make_action_scaler_field,
) = generate_optional_config_generation(
    ActionScaler  # type: ignore
)


@dataclasses.dataclass()
class StandardObservationScaler(ObservationScaler):
    r"""Standardization preprocessing.

    .. math::

        x' = (x - \mu) / \sigma

    .. code-block:: python

        from d3rlpy.preprocessing import StandardObservationScaler
        from d3rlpy.algos import CQLConfig

        # normalize based on datasets
        cql = CQLConfig(observation_scaler=StandardObservationScaler()).create()

        # manually initialize
        mean = observations.mean(axis=0)
        std = observations.std(axis=0)
        observation_scaler = StandardObservationScaler(mean=mean, std=std)
        cql = CQLConfig(observation_scaler=observation_scaler).create()

    Args:
        mean (numpy.ndarray): Mean values at each entry.
        std (numpy.ndarray): Standard deviation at each entry.
        eps (float): Small constant value to avoid zero-division.
    """

    mean: Optional[NDArray] = make_optional_numpy_field()
    std: Optional[NDArray] = make_optional_numpy_field()
    eps: float = 1e-3

    def __post_init__(self) -> None:
        if self.mean is not None:
            self.mean = np.asarray(self.mean)
        if self.std is not None:
            self.std = np.asarray(self.std)
        self._torch_mean: Optional[torch.Tensor] = None
        self._torch_std: Optional[torch.Tensor] = None

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        assert not self.built
        # compute mean
        total_sum = np.zeros(episodes[0].observation_signature.shape[0])
        total_count = 0
        for episode in episodes:
            for i in range(episode.transition_count):
                transition = transition_picker(episode, i)
                total_sum += transition.observation
            total_count += episode.transition_count
        mean = total_sum / total_count

        # compute stdandard deviation
        total_sqsum = np.zeros(episodes[0].observation_signature.shape[0])
        for episode in episodes:
            for i in range(episode.transition_count):
                transition = transition_picker(episode, i)
                total_sqsum += (transition.observation - mean) ** 2
        std = np.sqrt(total_sqsum / total_count)

        self.mean = mean
        self.std = std

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        assert not self.built
        # compute mean
        total_sum = np.zeros(episodes[0].observation_signature.shape[0])
        total_count = 0
        for episode in episodes:
            traj = trajectory_slicer(episode, episode.size() - 1, episode.size())
            total_sum += np.sum(traj.observations, axis=0)
            total_count += episode.size()
        mean = total_sum / total_count

        # compute stdandard deviation
        total_sqsum = np.zeros(episodes[0].observation_signature.shape[0])
        expanded_mean = mean.reshape((1,) + mean.shape)
        for episode in episodes:
            traj = trajectory_slicer(episode, episode.size() - 1, episode.size())
            observations = np.asarray(traj.observations)
            total_sqsum += np.sum((observations - expanded_mean) ** 2, axis=0)
        std = np.sqrt(total_sqsum / total_count)

        self.mean = mean
        self.std = std

    # def fit_with_env(self, env: GymEnv) -> None:
    #     raise NotImplementedError(
    #         "standard scaler does not support fit_with_env."
    #     )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        if self._torch_mean is None or self._torch_std is None:
            self._set_torch_value(x.device)
        assert self._torch_mean is not None and self._torch_std is not None
        mean = add_leading_dims(self._torch_mean, target=x)
        std = add_leading_dims(self._torch_std, target=x)
        return (x - mean) / (std + self.eps)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        if self._torch_mean is None or self._torch_std is None:
            self._set_torch_value(x.device)
        assert self._torch_mean is not None and self._torch_std is not None
        mean = add_leading_dims(self._torch_mean, target=x)
        std = add_leading_dims(self._torch_std, target=x)
        return ((std + self.eps) * x) + mean

    def transform_numpy(self, x: NDArray) -> NDArray:
        assert self.built
        assert self.mean is not None and self.std is not None
        mean = add_leading_dims_numpy(self.mean, target=x)
        std = add_leading_dims_numpy(self.std, target=x)
        ret = (x - mean) / (std + self.eps)
        return ret  # type: ignore

    def reverse_transform_numpy(self, x: NDArray) -> NDArray:
        assert self.built
        assert self.mean is not None and self.std is not None
        mean = add_leading_dims_numpy(self.mean, target=x)
        std = add_leading_dims_numpy(self.std, target=x)
        return ((std + self.eps) * x) + mean

    def _set_torch_value(self, device: torch.device) -> None:
        self._torch_mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        self._torch_std = torch.tensor(self.std, dtype=torch.float32, device=device)

    @staticmethod
    def get_type() -> str:
        return "standard"

    @property
    def built(self) -> bool:
        return self.mean is not None and self.std is not None


(
    register_observation_scaler,
    make_observation_scaler_field,
) = generate_optional_config_generation(
    ObservationScaler  # type: ignore
)

observation_scaler_list_field = generate_list_config_field(
    ObservationScaler  # type: ignore
)


# torch utility
#######################################################################

_TModule = TypeVar("_TModule", bound=nn.Module)


class Checkpointer:
    _modules: dict[str, Union[nn.Module, OptimizerWrapperProto]]
    _device: str

    def __init__(
        self,
        modules: dict[str, Union[nn.Module, OptimizerWrapperProto]],
        device: str,
    ):
        self._modules = modules
        self._device = device

    def save(self, f: BinaryIO) -> None:
        # unwrap DDP
        modules = {
            k: unwrap_ddp_model(v) if isinstance(v, nn.Module) else v
            for k, v in self._modules.items()
        }
        states = {k: v.state_dict() for k, v in modules.items()}
        torch.save(states, f)

    def load(self, f: BinaryIO) -> None:
        chkpt = torch.load(f, map_location=map_location(self._device))
        for k, v in self._modules.items():
            v.load_state_dict(chkpt[k])

    @property
    def modules(self) -> dict[str, Union[nn.Module, OptimizerWrapperProto]]:
        return self._modules


@dataclasses.dataclass(frozen=True)
class Modules:
    def create_checkpointer(self, device: str) -> Checkpointer:
        modules = {
            k: v
            for k, v in asdict_without_copy(self).items()
            if isinstance(v, (nn.Module, OptimizerWrapperProto))
        }
        return Checkpointer(modules=modules, device=device)

    def freeze(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, nn.Module):
                for p in v.parameters():
                    p.requires_grad = False

    def unfreeze(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, nn.Module):
                for p in v.parameters():
                    p.requires_grad = True

    def set_eval(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, nn.Module) and v.training:
                v.eval()

    def set_train(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, nn.Module) and not v.training:
                v.train()

    def reset_optimizer_states(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, OptimizerWrapperProto):
                v.optim.state = collections.defaultdict(dict)

    def get_torch_modules(self) -> dict[str, nn.Module]:
        torch_modules: dict[str, nn.Module] = {}
        for k, v in asdict_without_copy(self).items():
            if isinstance(v, nn.Module):
                torch_modules[k] = v
        return torch_modules

    def get_gradients(self) -> Iterator[tuple[str, Float32NDArray]]:
        for module_name, module in self.get_torch_modules().items():
            for name, parameter in module.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    yield (
                        f"{module_name}.{name}",
                        parameter.grad.cpu().detach().numpy(),
                    )


def map_location(device: str) -> Any:
    if "cuda" in device:
        _, index = device.split(":")
        return lambda storage, loc: storage.cuda(int(index))
    if "cpu" in device:
        return "cpu"
    raise ValueError(f"invalid device={device}")


def unwrap_ddp_model(model: _TModule) -> _TModule:
    if isinstance(model, DDP):
        model = model.module
    if isinstance(model, nn.ModuleList):
        module_list = nn.ModuleList()
        for v in model:
            module_list.append(unwrap_ddp_model(v))
        model = module_list
    return model


class Checkpointer:
    _modules: dict[str, Union[nn.Module, OptimizerWrapperProto]]
    _device: str

    def __init__(
        self,
        modules: dict[str, Union[nn.Module, OptimizerWrapperProto]],
        device: str,
    ):
        self._modules = modules
        self._device = device

    def save(self, f: BinaryIO) -> None:
        # unwrap DDP
        modules = {
            k: unwrap_ddp_model(v) if isinstance(v, nn.Module) else v
            for k, v in self._modules.items()
        }
        states = {k: v.state_dict() for k, v in modules.items()}
        torch.save(states, f)

    def load(self, f: BinaryIO) -> None:
        chkpt = torch.load(f, map_location=map_location(self._device))
        for k, v in self._modules.items():
            v.load_state_dict(chkpt[k])

    @property
    def modules(self) -> dict[str, Union[nn.Module, OptimizerWrapperProto]]:
        return self._modules


TCallable = TypeVar("TCallable")


def train_api(f: TCallable) -> TCallable:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        assert hasattr(self, "modules")
        assert isinstance(self.modules, Modules)
        self.modules.set_train()
        return f(self, *args, **kwargs)  # type: ignore

    return wrapper  # type: ignore


def get_device(x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> str:
    if isinstance(x, torch.Tensor):
        return str(x.device)
    else:
        return str(x[0].device)


def get_batch_size(x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.shape[0])
    else:
        return int(x[0].shape[0])


_TModule = TypeVar("_TModule", bound=nn.Module)


def wrap_model_by_ddp(model: _TModule) -> _TModule:
    device_id = next(model.parameters()).device.index
    return DDP(model, device_ids=[device_id] if device_id else None)  # type: ignore


class QFunctionOutput(NamedTuple):
    q_value: torch.Tensor
    quantiles: Optional[torch.Tensor]
    taus: Optional[torch.Tensor]


class Encoder(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation) -> torch.Tensor:
        pass

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        return super().__call__(x)


class EncoderWithAction(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        return super().__call__(x, action)


class DiscreteQFunction(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation) -> QFunctionOutput:
        pass

    def __call__(self, x: TorchObservation) -> QFunctionOutput:
        return super().__call__(x)  # type: ignore

    @property
    @abstractmethod
    def encoder(self) -> Encoder:
        pass


class DiscreteQFunctionForwarder(metaclass=ABCMeta):
    @abstractmethod
    def compute_expected_q(self, x: TorchObservation) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(
        self, x: TorchObservation, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def set_q_func(self, q_func: DiscreteQFunction) -> None:
        pass


class DiscreteEnsembleQFunctionForwarder:
    _forwarders: Sequence[DiscreteQFunctionForwarder]
    _action_size: int

    def __init__(
        self, forwarders: Sequence[DiscreteQFunctionForwarder], action_size: int
    ):
        self._forwarders = forwarders
        self._action_size = action_size

    def compute_expected_q(
        self, x: TorchObservation, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for forwarder in self._forwarders:
            value = forwarder.compute_expected_q(x)
            values.append(
                value.view(
                    1,
                    (
                        x[0].shape[0]
                        if isinstance(x, (list, tuple))
                        else x.shape[0]  # type: ignore
                    ),
                    self._action_size,
                )
            )
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return compute_ensemble_q_function_error(
            forwarders=self._forwarders,
            observations=observations,
            actions=actions,
            rewards=rewards,
            target=target,
            terminals=terminals,
            gamma=gamma,
            masks=masks,
        )

    def compute_target(
        self,
        x: TorchObservation,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return compute_ensemble_q_function_target(
            forwarders=self._forwarders,
            action_size=self._action_size,
            x=x,
            action=action,
            reduction=reduction,
            lam=lam,
        )

    def compute_uncertainty(
        self,
        x: TorchObservation,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns a per-sample uncertainty measure for (x, action).
        """
        # print ("compute_uncertainty")
        std = x.std(dim=1).mean()
        penalty = torch.exp(-std)
        return penalty

    @property
    def forwarders(self) -> Sequence[DiscreteQFunctionForwarder]:
        return self._forwarders


class _DiscreteQFunctionProtocol(Protocol):
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder


class DiscreteQFunctionMixin:
    def inner_predict_value(
        self: _DiscreteQFunctionProtocol,
        x: TorchObservation,
        action: torch.Tensor,
    ) -> torch.Tensor:
        values = self._q_func_forwarder.compute_expected_q(x, reduction="mean")
        flat_action = action.reshape(-1)
        return values[torch.arange(0, values.size(0)), flat_action].reshape(-1)


DeviceArg = Optional[Union[bool, int, str]]
TImpl_co = TypeVar("TImpl_co", bound="ImplBase", covariant=True)
TConfig_co = TypeVar("TConfig_co", bound="LearnableConfig", covariant=True)


class ImplBase(metaclass=ABCMeta):
    _observation_shape: Shape
    _action_size: int
    _modules: Modules
    _checkpointer: Checkpointer
    _device: str

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: Modules,
        device: str,
    ):
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._modules = modules
        self._checkpointer = modules.create_checkpointer(device)
        self._device = device

    def save_model(self, f: BinaryIO) -> None:
        self._checkpointer.save(f)

    def load_model(self, f: BinaryIO) -> None:
        self._checkpointer.load(f)

    @property
    def observation_shape(self) -> Shape:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def modules(self) -> Modules:
        return self._modules


@dataclasses.dataclass()
class LearnableConfig(DynamicConfig):
    batch_size: int = 256
    gamma: float = 0.99
    observation_scaler: Optional[ObservationScaler] = make_observation_scaler_field()
    action_scaler: Optional[ActionScaler] = make_action_scaler_field()
    reward_scaler: Optional[RewardScaler] = make_reward_scaler_field()
    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "LearnableBase[ImplBase, LearnableConfig]":
        r"""Returns algorithm object.

        Args:
            device (Union[int, str, bool]): device option. If the value is
                boolean and True, ``cuda:0`` will be used. If the value is
                integer, ``cuda:<device>`` will be used. If the value is string
                in torch device style, the specified device will be used.
            enable_ddp (bool): Flag to wrap models with DDP to enable Data
                Distributed Parallel training.

        Returns:
            algorithm object.
        """
        raise NotImplementedError


register_learnable, make_learnable_field = generate_config_registration(LearnableConfig)


@dataclasses.dataclass()
class LearnableConfigWithShape(DynamicConfig):
    observation_shape: Shape
    action_size: int
    config: LearnableConfig = make_learnable_field()

    def create(
        self, device: DeviceArg = False
    ) -> "LearnableBase[ImplBase, LearnableConfig]":
        algo = self.config.create(device)
        algo.create_impl(self.observation_shape, self.action_size)
        return algo


def save_config(
    algo: "LearnableBase[ImplBase, LearnableConfig]", logger: D3RLPyLogger
) -> None:
    assert algo.impl
    config = LearnableConfigWithShape(
        observation_shape=algo.impl.observation_shape,
        action_size=algo.impl.action_size,
        config=algo.config,
    )
    logger.add_params(config.serialize_to_dict())


def _process_device(value: DeviceArg) -> str:
    """Checks value and returns PyTorch target device.

    Returns:
        str: target device.
    """
    # isinstance cannot tell difference between bool and int
    if isinstance(value, bool):
        return "cuda:0" if value else "cpu:0"
    if isinstance(value, int):
        return f"cuda:{value}"
    if isinstance(value, str):
        return value
    if value is None:
        return "cpu:0"
    raise ValueError("This argument must be bool, int or str.")


def dump_learnable(
    algo: "LearnableBase[ImplBase, LearnableConfig]", fname: str
) -> None:
    assert algo.impl
    with open(fname, "wb") as f:
        torch_bytes = io.BytesIO()
        algo.impl.save_model(torch_bytes)
        config = LearnableConfigWithShape(
            observation_shape=algo.impl.observation_shape,
            action_size=algo.impl.action_size,
            config=algo.config,
        )
        obj = {
            "torch": torch_bytes.getvalue(),
            "config": config.serialize(),
            "version": __version__,
        }
        pickle.dump(obj, f)


def load_learnable(
    fname: str, device: DeviceArg = None
) -> "LearnableBase[ImplBase, LearnableConfig]":
    with open(fname, "rb") as f:
        obj = pickle.load(f)
        if obj["version"] != __version__:
            LOG.warning(
                "There might be incompatibility because of version mismatch.",
                current_version=__version__,
                saved_version=obj["version"],
            )
        config = LearnableConfigWithShape.deserialize(obj["config"])
        algo = config.create(device)
        assert algo.impl
        algo.impl.load_model(io.BytesIO(obj["torch"]))
    return algo


class LearnableBase(Generic[TImpl_co, TConfig_co], metaclass=ABCMeta):
    _config: TConfig_co
    _device: str
    _enable_ddp: bool
    _impl: Optional[TImpl_co]
    _grad_step: int

    def __init__(
        self,
        config: TConfig_co,
        device: DeviceArg,
        enable_ddp: bool,
        impl: Optional[TImpl_co] = None,
    ):
        if self.get_action_type() == ActionSpace.DISCRETE:
            assert config.action_scaler is None, (
                "action_scaler cannot be used with discrete action-space " "algorithms."
            )
        self._config = config
        self._device = _process_device(device)
        self._enable_ddp = enable_ddp
        self._impl = impl
        self._grad_step = 0

    def save(self, fname: str) -> None:
        """Saves paired data of neural network parameters and serialized config.

        .. code-block:: python

            algo.save('model.d3')

            # reconstruct everything
            algo2 = d3rlpy.load_learnable("model.d3", device="cuda:0")

        Args:
            fname: destination file path.
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        dump_learnable(self, fname)

    def save_model(self, fname: str) -> None:
        """Saves neural network parameters.

        .. code-block:: python

            algo.save_model('model.pt')

        Args:
            fname: destination file path.
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        with open(fname, "wb") as f:
            self._impl.save_model(f)

    def load_model(self, fname: str) -> None:
        """Load neural network parameters.

        .. code-block:: python

            algo.load_model('model.pt')

        Args:
            fname: source file path.
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        with open(fname, "rb") as f:
            self._impl.load_model(f)

    @classmethod
    def from_json(cls: type[Self], fname: str, device: DeviceArg = False) -> Self:
        r"""Construct algorithm from params.json file.

        .. code-block:: python

            from d3rlpy.algos import CQL

            cql = CQL.from_json("<path-to-json>", device='cuda:0')

        Args:
            fname: path to params.json
            device (Union[int, str, bool]): device option. If the value is
                boolean and True, ``cuda:0`` will be used. If the value is
                integer, ``cuda:<device>`` will be used. If the value is string
                in torch device style, the specified device will be used.

        Returns:
            algorithm object.
        """
        config = LearnableConfigWithShape.deserialize_from_file(fname)
        return config.create(device)  # type: ignore

    def create_impl(self, observation_shape: Shape, action_size: int) -> None:
        """Instantiate implementation objects with the dataset shapes.

        This method will be used internally when `fit` method is called.

        Args:
            observation_shape: observation shape.
            action_size: dimension of action-space.
        """
        if self._impl:
            LOG.warn("Parameters will be reinitialized.")
        self.inner_create_impl(observation_shape, action_size)

    @abstractmethod
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        pass

    def build_with_dataset(self, dataset: ReplayBuffer) -> None:
        """Instantiate implementation object with ReplayBuffer object.

        Args:
            dataset: dataset.
        """
        observation_shape = dataset.sample_transition().observation_signature.shape[0]
        self.create_impl(observation_shape, dataset.dataset_info.action_size)

    def build_with_env(self, env: GymEnv) -> None:
        """Instantiate implementation object with OpenAI Gym object.

        Args:
            env: gym-like environment.
        """
        assert isinstance(
            env.observation_space, (Box, GymnasiumBox)
        ), f"Unsupported observation space: {type(env.observation_space)}"
        observation_shape = env.observation_space.shape
        action_size = detect_action_size_from_env(env)
        self.create_impl(observation_shape, action_size)

    def get_action_type(self) -> ActionSpace:
        """Returns action type (continuous or discrete).

        Returns:
            action type.
        """
        raise NotImplementedError

    @property
    def config(self) -> TConfig_co:
        """Config.

        Returns:
            LearnableConfig: config.
        """
        return self._config

    @property
    def compiled(self) -> bool:
        """Compiled flag.

        This represents if computational graph is optimized with CudaGraph and
        torch.compile.

        Returns:
            bool: True if compiled.
        """
        return self._config.compile_graph and "cuda" in self._device

    @property
    def batch_size(self) -> int:
        """Batch size to train.

        Returns:
            int: batch size.
        """
        return self._config.batch_size

    @property
    def gamma(self) -> float:
        """Discount factor.

        Returns:
            float: discount factor.
        """
        return self._config.gamma

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]:
        """Preprocessing observation scaler.

        Returns:
            Optional[ObservationScaler]: preprocessing observation scaler.
        """
        return self._config.observation_scaler

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        """Preprocessing action scaler.

        Returns:
            Optional[ActionScaler]: preprocessing action scaler.
        """
        return self._config.action_scaler

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        """Preprocessing reward scaler.

        Returns:
            Optional[RewardScaler]: preprocessing reward scaler.
        """
        return self._config.reward_scaler

    @property
    def impl(self) -> Optional[TImpl_co]:
        """Implementation object.

        Returns:
            Optional[ImplBase]: implementation object.
        """
        return self._impl

    @property
    def observation_shape(self) -> Optional[Shape]:
        """Observation shape.

        Returns:
            Optional[Sequence[int]]: observation shape.
        """
        if self._impl:
            return self._impl.observation_shape
        return None

    @property
    def action_size(self) -> Optional[int]:
        """Action size.

        Returns:
            Optional[int]: action size.
        """
        if self._impl:
            return self._impl.action_size
        return None

    @property
    def grad_step(self) -> int:
        """Total gradient step counter.

        This value will keep counting after ``fit`` and ``fit_online``
        methods finish.

        Returns:
            total gradient step counter.
        """
        return self._grad_step

    def set_grad_step(self, grad_step: int) -> None:
        """Set total gradient step counter.

        This method can be used to restart the middle of training with an
        arbitrary gradient step counter, which has effects on periodic
        functions such as the target update.

        Args:
            grad_step: total gradient step counter.
        """
        self._grad_step = grad_step


def build_scalers_with_env(
    algo: LearnableBase[Any, Any],
    env: GymEnv,
) -> None:
    # initialize observation scaler
    if algo.observation_scaler and not algo.observation_scaler.built:
        LOG.debug(
            "Fitting observation scaler...",
            observation_scaler=algo.observation_scaler.get_type(),
        )
        algo.observation_scaler.fit_with_env(env)

    # initialize action scaler
    if algo.action_scaler and not algo.action_scaler.built:
        LOG.debug(
            "Fitting action scaler...",
            action_scler=algo.action_scaler.get_type(),
        )
        algo.action_scaler.fit_with_env(env)


# base
#########################################################################
DeviceArg = Optional[Union[bool, int, str]]
TImpl_co = TypeVar("TImpl_co", bound="ImplBase", covariant=True)
TConfig_co = TypeVar("TConfig_co", bound="LearnableConfig", covariant=True)


def _process_device(value: DeviceArg) -> str:
    """Checks value and returns PyTorch target device.

    Returns:
        str: target device.
    """
    # isinstance cannot tell difference between bool and int
    if isinstance(value, bool):
        return "cuda:0" if value else "cpu:0"
    if isinstance(value, int):
        return f"cuda:{value}"
    if isinstance(value, str):
        return value
    if value is None:
        return "cpu:0"
    raise ValueError("This argument must be bool, int or str.")


@dataclasses.dataclass()
class LearnableConfig(DynamicConfig):
    batch_size: int = 256
    gamma: float = 0.99
    observation_scaler: Optional[ObservationScaler] = make_observation_scaler_field()
    action_scaler: Optional[ActionScaler] = make_action_scaler_field()
    reward_scaler: Optional[RewardScaler] = make_reward_scaler_field()
    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "LearnableBase[ImplBase, LearnableConfig]":
        r"""Returns algorithm object.

        Args:
            device (Union[int, str, bool]): device option. If the value is
                boolean and True, ``cuda:0`` will be used. If the value is
                integer, ``cuda:<device>`` will be used. If the value is string
                in torch device style, the specified device will be used.
            enable_ddp (bool): Flag to wrap models with DDP to enable Data
                Distributed Parallel training.

        Returns:
            algorithm object.
        """
        raise NotImplementedError


register_learnable, make_learnable_field = generate_config_registration(LearnableConfig)


@dataclasses.dataclass()
class LearnableConfigWithShape(DynamicConfig):
    observation_shape: Shape
    action_size: int
    config: LearnableConfig = make_learnable_field()

    def create(
        self, device: DeviceArg = False
    ) -> "LearnableBase[ImplBase, LearnableConfig]":
        algo = self.config.create(device)
        algo.create_impl(self.observation_shape, self.action_size)
        return algo


def dump_learnable(
    algo: "LearnableBase[ImplBase, LearnableConfig]", fname: str
) -> None:
    assert algo.impl
    with open(fname, "wb") as f:
        torch_bytes = io.BytesIO()
        algo.impl.save_model(torch_bytes)
        config = LearnableConfigWithShape(
            observation_shape=algo.impl.observation_shape,
            action_size=algo.impl.action_size,
            config=algo.config,
        )
        obj = {
            "torch": torch_bytes.getvalue(),
            "config": config.serialize(),
            "version": __version__,
        }
        pickle.dump(obj, f)


class LearnableBase(Generic[TImpl_co, TConfig_co], metaclass=ABCMeta):
    _config: TConfig_co
    _device: str
    _enable_ddp: bool
    _impl: Optional[TImpl_co]
    _grad_step: int

    def __init__(
        self,
        config: TConfig_co,
        device: DeviceArg,
        enable_ddp: bool,
        impl: Optional[TImpl_co] = None,
    ):
        if self.get_action_type() == ActionSpace.DISCRETE:
            assert config.action_scaler is None, (
                "action_scaler cannot be used with discrete action-space " "algorithms."
            )
        self._config = config
        self._device = _process_device(device)
        self._enable_ddp = enable_ddp
        self._impl = impl
        self._grad_step = 0

    def save(self, fname: str) -> None:
        """Saves paired data of neural network parameters and serialized config.

        .. code-block:: python

            algo.save('model.d3')

            # reconstruct everything
            algo2 = d3rlpy.load_learnable("model.d3", device="cuda:0")

        Args:
            fname: destination file path.
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        dump_learnable(self, fname)

    def save_model(self, fname: str) -> None:
        """Saves neural network parameters.

        .. code-block:: python

            algo.save_model('model.pt')

        Args:
            fname: destination file path.
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        with open(fname, "wb") as f:
            self._impl.save_model(f)

    def load_model(self, fname: str) -> None:
        """Load neural network parameters.

        .. code-block:: python

            algo.load_model('model.pt')

        Args:
            fname: source file path.
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        with open(fname, "rb") as f:
            self._impl.load_model(f)

    @classmethod
    def from_json(cls: type[Self], fname: str, device: DeviceArg = False) -> Self:
        r"""Construct algorithm from params.json file.

        .. code-block:: python

            from d3rlpy.algos import CQL

            cql = CQL.from_json("<path-to-json>", device='cuda:0')

        Args:
            fname: path to params.json
            device (Union[int, str, bool]): device option. If the value is
                boolean and True, ``cuda:0`` will be used. If the value is
                integer, ``cuda:<device>`` will be used. If the value is string
                in torch device style, the specified device will be used.

        Returns:
            algorithm object.
        """
        config = LearnableConfigWithShape.deserialize_from_file(fname)
        return config.create(device)  # type: ignore

    def create_impl(self, observation_shape: Shape, action_size: int) -> None:
        """Instantiate implementation objects with the dataset shapes.

        This method will be used internally when `fit` method is called.

        Args:
            observation_shape: observation shape.
            action_size: dimension of action-space.
        """
        if self._impl:
            LOG.warn("Parameters will be reinitialized.")
        self.inner_create_impl(observation_shape, action_size)

    @abstractmethod
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        pass

    def build_with_dataset(self, dataset: ReplayBuffer) -> None:
        """Instantiate implementation object with ReplayBuffer object.

        Args:
            dataset: dataset.
        """
        observation_shape = dataset.sample_transition().observation_signature.shape[0]
        self.create_impl(observation_shape, dataset.dataset_info.action_size)

    def build_with_env(self, env: GymEnv) -> None:
        """Instantiate implementation object with OpenAI Gym object.

        Args:
            env: gym-like environment.
        """
        assert isinstance(
            env.observation_space, (Box, GymnasiumBox)
        ), f"Unsupported observation space: {type(env.observation_space)}"
        observation_shape = env.observation_space.shape
        action_size = detect_action_size_from_env(env)
        self.create_impl(observation_shape, action_size)

    def get_action_type(self) -> ActionSpace:
        """Returns action type (continuous or discrete).

        Returns:
            action type.
        """
        raise NotImplementedError

    @property
    def config(self) -> TConfig_co:
        """Config.

        Returns:
            LearnableConfig: config.
        """
        return self._config

    @property
    def compiled(self) -> bool:
        """Compiled flag.

        This represents if computational graph is optimized with CudaGraph and
        torch.compile.

        Returns:
            bool: True if compiled.
        """
        return self._config.compile_graph and "cuda" in self._device

    @property
    def batch_size(self) -> int:
        """Batch size to train.

        Returns:
            int: batch size.
        """
        return self._config.batch_size

    @property
    def gamma(self) -> float:
        """Discount factor.

        Returns:
            float: discount factor.
        """
        return self._config.gamma

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]:
        """Preprocessing observation scaler.

        Returns:
            Optional[ObservationScaler]: preprocessing observation scaler.
        """
        return self._config.observation_scaler

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        """Preprocessing action scaler.

        Returns:
            Optional[ActionScaler]: preprocessing action scaler.
        """
        return self._config.action_scaler

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        """Preprocessing reward scaler.

        Returns:
            Optional[RewardScaler]: preprocessing reward scaler.
        """
        return self._config.reward_scaler

    @property
    def impl(self) -> Optional[TImpl_co]:
        """Implementation object.

        Returns:
            Optional[ImplBase]: implementation object.
        """
        return self._impl

    @property
    def observation_shape(self) -> Optional[Shape]:
        """Observation shape.

        Returns:
            Optional[Sequence[int]]: observation shape.
        """
        if self._impl:
            return self._impl.observation_shape
        return None

    @property
    def action_size(self) -> Optional[int]:
        """Action size.

        Returns:
            Optional[int]: action size.
        """
        if self._impl:
            return self._impl.action_size
        return None

    @property
    def grad_step(self) -> int:
        """Total gradient step counter.

        This value will keep counting after ``fit`` and ``fit_online``
        methods finish.

        Returns:
            total gradient step counter.
        """
        return self._grad_step

    def set_grad_step(self, grad_step: int) -> None:
        """Set total gradient step counter.

        This method can be used to restart the middle of training with an
        arbitrary gradient step counter, which has effects on periodic
        functions such as the target update.

        Args:
            grad_step: total gradient step counter.
        """
        self._grad_step = grad_step


class ImplBase(metaclass=ABCMeta):
    _observation_shape: Shape
    _action_size: int
    _modules: Modules
    _checkpointer: Checkpointer
    _device: str

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: Modules,
        device: str,
    ):
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._modules = modules
        self._checkpointer = modules.create_checkpointer(device)
        self._device = device

    def save_model(self, f: BinaryIO) -> None:
        self._checkpointer.save(f)

    def load_model(self, f: BinaryIO) -> None:
        self._checkpointer.load(f)

    @property
    def observation_shape(self) -> Shape:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def modules(self) -> Modules:
        return self._modules


@dataclasses.dataclass()
class LearnableConfig(DynamicConfig):
    batch_size: int = 256
    gamma: float = 0.99
    observation_scaler: Optional[ObservationScaler] = make_observation_scaler_field()
    action_scaler: Optional[ActionScaler] = make_action_scaler_field()
    reward_scaler: Optional[RewardScaler] = make_reward_scaler_field()
    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "LearnableBase[ImplBase, LearnableConfig]":
        r"""Returns algorithm object.

        Args:
            device (Union[int, str, bool]): device option. If the value is
                boolean and True, ``cuda:0`` will be used. If the value is
                integer, ``cuda:<device>`` will be used. If the value is string
                in torch device style, the specified device will be used.
            enable_ddp (bool): Flag to wrap models with DDP to enable Data
                Distributed Parallel training.

        Returns:
            algorithm object.
        """
        raise NotImplementedError


register_learnable, make_learnable_field = generate_config_registration(LearnableConfig)


# algo utility
#######################################################################
def assert_action_space_with_dataset(
    algo: LearnableBase[Any, Any], dataset_info: DatasetInfo
) -> None:

    # from == changed to == to fix the error as we are using the buffer from d3rlpy.alpgos
    if algo.get_action_type() == ActionSpace.BOTH:
        pass
    elif dataset_info.action_space == ActionSpace.DISCRETE:

        assert (
            algo.get_action_type() is ActionSpace.DISCRETE
        ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
    else:
        assert (
            algo.get_action_type() == ActionSpace.CONTINUOUS
        ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR


def build_scalers_with_transition_picker(
    algo: LearnableBase[Any, Any], dataset: ReplayBufferBase
) -> None:
    # initialize observation scaler
    if algo.observation_scaler and not algo.observation_scaler.built:
        LOG.debug(
            "Fitting observation scaler...",
            observation_scaler=algo.observation_scaler.get_type(),
        )
        algo.observation_scaler.fit_with_transition_picker(
            dataset.episodes, dataset.transition_picker
        )

    # initialize action scaler
    if algo.action_scaler and not algo.action_scaler.built:
        LOG.debug(
            "Fitting action scaler...",
            action_scaler=algo.action_scaler.get_type(),
        )
        algo.action_scaler.fit_with_transition_picker(
            dataset.episodes, dataset.transition_picker
        )

    # initialize reward scaler
    if algo.reward_scaler and not algo.reward_scaler.built:
        LOG.debug(
            "Fitting reward scaler...",
            reward_scaler=algo.reward_scaler.get_type(),
        )
        algo.reward_scaler.fit_with_transition_picker(
            dataset.episodes, dataset.transition_picker
        )


def assert_action_space_with_env(algo: LearnableBase[Any, Any], env: GymEnv) -> None:

    if isinstance(env.action_space, (Box, GymnasiumBox)):
        assert (
            algo.get_action_type() == ActionSpace.CONTINUOUS
        ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR
    elif isinstance(env.action_space, (Discrete, GymnasiumDiscrete)):
        assert (
            algo.get_action_type() == ActionSpace.DISCRETE
        ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
    else:
        action_space = type(env.action_space)
        raise ValueError(f"The action-space is not supported: {action_space}")


# utilities
##################################################################


def compute_huber_loss(
    y: torch.Tensor, target: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    diff = target - y
    cond = diff.detach().abs() < beta
    return torch.where(cond, 0.5 * diff**2, beta * (diff.abs() - 0.5 * beta))


def compute_reduce(value: torch.Tensor, reduction_type: str) -> torch.Tensor:
    if reduction_type == "mean":
        return value.mean()
    elif reduction_type == "sum":
        return value.sum()
    elif reduction_type == "none":
        return value.view(-1, 1)
    raise ValueError("invalid reduction type.")


def pick_value_by_action(
    values: torch.Tensor, action: torch.Tensor, keepdim: bool = False
) -> torch.Tensor:
    assert values.ndim == 2
    action_size = values.shape[1]
    one_hot = F.one_hot(action.view(-1), num_classes=action_size)
    masked_values = values * cast(torch.Tensor, one_hot.float())
    return masked_values.sum(dim=1, keepdim=keepdim)


# torch utility
#########################################################


class Swish(nn.Module):  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GEGLU(nn.Module):  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


@dataclasses.dataclass(frozen=True)
class TorchMiniBatch:
    observations: TorchObservation
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: TorchObservation
    next_actions: torch.Tensor
    returns_to_go: torch.Tensor
    terminals: torch.Tensor
    intervals: torch.Tensor
    device: str
    numpy_batch: Optional[TransitionMiniBatch] = None

    @classmethod
    def from_batch(
        cls,
        batch: TransitionMiniBatch,
        gamma: float,
        compute_returns_to_go: bool,
        device: str,
        observation_scaler: Optional[ObservationScaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ) -> "TorchMiniBatch":
        # convert numpy array to torch tensor
        observations = convert_to_torch_recursively(batch.observations, device)
        actions = convert_to_torch(batch.actions, device)
        next_actions = convert_to_torch(batch.next_actions, device)
        rewards = convert_to_torch(batch.rewards, device)
        next_observations = convert_to_torch_recursively(
            batch.next_observations, device
        )
        terminals = convert_to_torch(batch.terminals, device)
        intervals = convert_to_torch(batch.intervals, device)

        if compute_returns_to_go:
            returns_to_go = convert_to_torch(
                np.array(
                    [
                        _compute_return_to_go(
                            gamma=gamma,
                            rewards_to_go=transition.rewards_to_go,
                            reward_scaler=reward_scaler,
                        )
                        for transition in batch.transitions
                    ]
                ),
                device,
            )
        else:
            returns_to_go = torch.zeros_like(rewards)

        # apply scaler
        if observation_scaler:
            observations = observation_scaler.transform(observations)
            next_observations = observation_scaler.transform(next_observations)
        if action_scaler:
            actions = action_scaler.transform(actions)
            next_actions = action_scaler.transform(next_actions)
        if reward_scaler:
            rewards = reward_scaler.transform(rewards)

        return TorchMiniBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_actions=next_actions,
            returns_to_go=returns_to_go,
            terminals=terminals,
            intervals=intervals,
            device=device,
            numpy_batch=batch,
        )

    def copy_(self, src: Self) -> None:
        assert self.device == src.device, "incompatible device"
        copy_recursively(src.observations, self.observations)
        self.actions.copy_(src.actions)
        self.rewards.copy_(src.rewards)
        copy_recursively(src.next_observations, self.next_observations)
        self.next_actions.copy_(src.next_actions)
        self.returns_to_go.copy_(src.returns_to_go)
        self.terminals.copy_(src.terminals)
        self.intervals.copy_(src.intervals)


@dataclasses.dataclass(frozen=True)
class TorchTrajectoryMiniBatch:
    observations: TorchObservation  # (B, L, ...)
    actions: torch.Tensor  # (B, L, ...)
    rewards: torch.Tensor  # (B, L, 1)
    returns_to_go: torch.Tensor  # (B, L, 1)
    terminals: torch.Tensor  # (B, L, 1)
    timesteps: torch.Tensor  # (B, L, 1)
    masks: torch.Tensor  # (B, L)
    device: str
    numpy_batch: Optional[TrajectoryMiniBatch] = None

    @classmethod
    def from_batch(
        cls,
        batch: TrajectoryMiniBatch,
        device: str,
        observation_scaler: Optional[ObservationScaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ) -> "TorchTrajectoryMiniBatch":
        # convert numpy array to torch tensor
        observations = convert_to_torch_recursively(batch.observations, device)
        actions = convert_to_torch(batch.actions, device)
        rewards = convert_to_torch(batch.rewards, device)
        returns_to_go = convert_to_torch(batch.returns_to_go, device)
        terminals = convert_to_torch(batch.terminals, device)
        timesteps = convert_to_torch(batch.timesteps, device).long()
        masks = convert_to_torch(batch.masks, device)

        # apply scaler
        if observation_scaler:
            observations = observation_scaler.transform(observations)
        if action_scaler:
            actions = action_scaler.transform(actions)
        if reward_scaler:
            rewards = reward_scaler.transform(rewards)
            # NOTE: some operations might be incompatible with returns
            returns_to_go = reward_scaler.transform(returns_to_go)

        return TorchTrajectoryMiniBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            terminals=terminals,
            timesteps=timesteps,
            masks=masks,
            device=device,
            numpy_batch=batch,
        )

    def copy_(self, src: Self) -> None:
        assert self.device == src.device, "incompatible device"
        copy_recursively(src.observations, self.observations)
        self.actions.copy_(src.actions)
        self.rewards.copy_(src.rewards)
        self.returns_to_go.copy_(src.returns_to_go)
        self.terminals.copy_(src.terminals)
        self.timesteps.copy_(src.timesteps)
        self.masks.copy_(src.masks)

    def to_transition_batch(self) -> tuple[TorchMiniBatch, torch.Tensor]:
        if isinstance(self.observations, torch.Tensor):
            observations = self.observations[:, :-1].reshape(
                -1, *self.observations.shape[2:]
            )
            next_observations = self.observations[:, 1:].reshape(
                -1, *self.observations.shape[2:]
            )
        else:
            observations = [
                obs[:, :-1].reshape(-1, *obs.shape[2:]) for obs in self.observations
            ]
            next_observations = [
                obs[:, 1:].reshape(-1, *obs.shape[2:]) for obs in self.observations
            ]
        actions = self.actions[:, :-1].reshape(-1, *self.actions.shape[2:])
        rewards = self.rewards[:, :-1].reshape(-1, 1)
        terminals = self.terminals[:, :-1].reshape(-1, 1)
        next_actions = self.actions[:, 1:].reshape(-1, *self.actions.shape[2:])
        returns_to_go = self.returns_to_go[:, :-1].reshape(-1, 1)
        intervals = torch.ones_like(rewards)
        masks = self.masks[:, :-1].reshape(-1, 1)
        batch = TorchMiniBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_actions=next_actions,
            returns_to_go=returns_to_go,
            terminals=terminals,
            intervals=intervals,
            device=self.device,
        )
        return batch, masks


# from d3rlpy.torch_utility import TorchTrajectoryMiniBatch, TorchMiniBatch
BatchT_contra = TypeVar(
    "BatchT_contra",
    bound=Union[TorchMiniBatch, TorchTrajectoryMiniBatch],
    contravariant=True,
)
RetT_co = TypeVar("RetT_co", covariant=True)


class CudaGraphFunc(Generic[BatchT_contra, RetT_co], Protocol):
    def __call__(self, batch: BatchT_contra) -> RetT_co: ...


class CudaGraphWrapper(Generic[BatchT_contra, RetT_co]):
    _func: CudaGraphFunc[BatchT_contra, RetT_co]
    _input: TorchTrajectoryMiniBatch
    _graph: Optional[CUDAGraph]
    _inpt: Optional[BatchT_contra]
    _out: Optional[RetT_co]

    def __init__(
        self,
        func: CudaGraphFunc[BatchT_contra, RetT_co],
        warmup_steps: int = 3,
        compile_func: bool = True,
    ):
        self._func = torch.compile(func) if compile_func else func
        self._step = 0
        self._graph = None
        self._inpt = None
        self._out = None
        self._warmup_steps = warmup_steps
        self._warmup_stream = torch.cuda.Stream()

    def __call__(self, batch: BatchT_contra) -> RetT_co:
        if self._step < self._warmup_steps:  # warmup
            self._warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._warmup_stream):
                out = self._func(batch)
            torch.cuda.current_stream().wait_stream(self._warmup_stream)
        if self._step == self._warmup_steps - 1:  # build graph
            self._graph = torch.cuda.CUDAGraph()
            self._inpt = batch
            with torch.cuda.graph(self._graph):
                self._out = self._func(self._inpt)
        if self._step >= self._warmup_steps:  # reuse cuda graph
            assert self._inpt
            assert self._out is not None
            assert self._graph
            with torch.no_grad():
                self._inpt.copy_(batch)  # type: ignore
            self._graph.replay()
            out = self._out
        self._step += 1
        return out


# Model
##################################################################
# utils


def create_activation(activation_type: str) -> nn.Module:
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "swish":
        return Swish()
    elif activation_type == "none":
        return nn.Identity()
    elif activation_type == "geglu":
        return GEGLU()
    raise ValueError("invalid activation_type.")


# Encoder
##################################################################


class Encoder(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation) -> torch.Tensor:
        pass

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        return super().__call__(x)


class EncoderWithAction(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        return super().__call__(x, action)


class EncoderFactory(DynamicConfig):
    def create(self, observation_shape: Shape) -> Encoder:
        """Returns PyTorch's state enocder module.

        Args:
            observation_shape: observation shape.

        Returns:
            an enocder object.
        """
        raise NotImplementedError

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> EncoderWithAction:
        """Returns PyTorch's state-action enocder module.

        Args:
            observation_shape: observation shape.
            action_size: action size. If None, the encoder does not take
                action as input.
            discrete_action: flag if action-space is discrete.

        Returns:
            an enocder object.
        """
        raise NotImplementedError


class VectorEncoderWithAction(EncoderWithAction):
    _layers: nn.Module
    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
        exclude_last_activation: bool = False,
        last_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self._action_size = action_size
        self._discrete_action = discrete_action

        if hidden_units is None:
            hidden_units = [256, 256]

        layers = []
        in_units = [observation_shape[0] + action_size] + list(hidden_units[:-1])
        for is_last, (in_unit, out_unit) in last_flag(zip(in_units, hidden_units)):
            layers.append(nn.Linear(in_unit, out_unit))
            if not is_last or not exclude_last_activation:
                if is_last and last_activation:
                    layers.append(last_activation)
                else:
                    layers.append(activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_unit))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_unit))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
        self._layers = nn.Sequential(*layers)

    def forward(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self._action_size
            ).float()
        x = torch.cat([x, action], dim=1)
        return self._layers(x)


class PixelEncoder(Encoder):
    _cnn_layers: nn.Module
    _last_layers: nn.Module

    def __init__(
        self,
        observation_shape: Sequence[int],
        filters: Optional[list[list[int]]] = None,
        feature_size: int = 512,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = False,
        activation: nn.Module = nn.ReLU(),
        exclude_last_activation: bool = False,
        last_activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        # default architecture is based on Nature DQN paper.
        if filters is None:
            filters = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
        if feature_size is None:
            feature_size = 512

        # convolutional layers
        cnn_layers = []
        in_channels = [observation_shape[0]] + [f[0] for f in filters[:-1]]
        for in_channel, f in zip(in_channels, filters):
            out_channel, kernel_size, stride = f
            conv = nn.Conv2d(
                in_channel, out_channel, kernel_size=kernel_size, stride=stride
            )
            cnn_layers.append(conv)
            cnn_layers.append(activation)

            # use batch normalization layer
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm2d(out_channel))

            # use dropout layer
            if dropout_rate is not None:
                cnn_layers.append(nn.Dropout2d(dropout_rate))
        self._cnn_layers = nn.Sequential(*cnn_layers)

        # compute output shape of CNN layers
        x = torch.rand((1,) + tuple(observation_shape))
        with torch.no_grad():
            cnn_output_size = self._cnn_layers(x).view(1, -1).shape[1]

        # last dense layer
        layers: list[nn.Module] = []
        layers.append(nn.Linear(cnn_output_size, feature_size))
        if not exclude_last_activation:
            layers.append(last_activation if last_activation else activation)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(feature_size))
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))

        self._last_layers = nn.Sequential(*layers)

    def forward(self, x: TorchObservation) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        h = self._cnn_layers(x)
        return self._last_layers(h.reshape(x.shape[0], -1))


class PixelEncoderWithAction(EncoderWithAction):
    _cnn_layers: nn.Module
    _last_layers: nn.Module
    _discrete_action: bool
    _action_size: int

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        filters: Optional[list[list[int]]] = None,
        feature_size: int = 512,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = False,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
        exclude_last_activation: bool = False,
        last_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self._discrete_action = discrete_action
        self._action_size = action_size

        # default architecture is based on Nature DQN paper.
        if filters is None:
            filters = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
        if feature_size is None:
            feature_size = 512

        # convolutional layers
        cnn_layers = []
        in_channels = [observation_shape[0]] + [f[0] for f in filters[:-1]]
        for in_channel, f in zip(in_channels, filters):
            out_channel, kernel_size, stride = f
            conv = nn.Conv2d(
                in_channel, out_channel, kernel_size=kernel_size, stride=stride
            )
            cnn_layers.append(conv)
            cnn_layers.append(activation)

            # use batch normalization layer
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm2d(out_channel))

            # use dropout layer
            if dropout_rate is not None:
                cnn_layers.append(nn.Dropout2d(dropout_rate))
        self._cnn_layers = nn.Sequential(*cnn_layers)

        # compute output shape of CNN layers
        x = torch.rand((1,) + tuple(observation_shape))
        with torch.no_grad():
            cnn_output_size = self._cnn_layers(x).view(1, -1).shape[1]

        # last dense layer
        layers: list[nn.Module] = []
        layers.append(nn.Linear(cnn_output_size + action_size, feature_size))
        if not exclude_last_activation:
            layers.append(last_activation if last_activation else activation)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(feature_size))
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))
        self._last_layers = nn.Sequential(*layers)

    def forward(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        h = self._cnn_layers(x)

        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self._action_size
            ).float()

        # cocat feature and action
        h = torch.cat([h.reshape(h.shape[0], -1), action], dim=1)

        return self._last_layers(h)


@dataclass()
class PixelEncoderFactory(EncoderFactory):
    """Pixel encoder factory class.

    This is the default encoder factory for image observation.

    Args:
        filters (list): List of tuples consisting with
            ``(filter_size, kernel_size, stride)``. If None,
            ``Nature DQN``-based architecture is used.
        feature_size (int): Last linear layer size.
        activation (str): Activation function name.
        use_batch_norm (bool): Flag to insert batch normalization layers.
        dropout_rate (float): Dropout probability.
        exclude_last_activation (bool): Flag to exclude activation function at
            the last layer.
        last_activation (str): Activation function name for the last layer.
    """

    filters: list[list[int]] = field(
        default_factory=lambda: [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
    )
    feature_size: int = 512
    activation: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: Optional[float] = None
    exclude_last_activation: bool = False
    last_activation: Optional[str] = None

    def create(self, observation_shape: Shape) -> PixelEncoder:
        assert len(observation_shape) == 3
        return PixelEncoder(
            observation_shape=cast_flat_shape(observation_shape),
            filters=self.filters,
            feature_size=self.feature_size,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            activation=create_activation(self.activation),
            exclude_last_activation=self.exclude_last_activation,
            last_activation=(
                create_activation(self.last_activation)
                if self.last_activation
                else None
            ),
        )

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> PixelEncoderWithAction:
        assert len(observation_shape) == 3
        return PixelEncoderWithAction(
            observation_shape=cast_flat_shape(observation_shape),
            action_size=action_size,
            filters=self.filters,
            feature_size=self.feature_size,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            discrete_action=discrete_action,
            activation=create_activation(self.activation),
            exclude_last_activation=self.exclude_last_activation,
            last_activation=(
                create_activation(self.last_activation)
                if self.last_activation
                else None
            ),
        )

    @staticmethod
    def get_type() -> str:
        return "pixel"


class VectorEncoder(Encoder):
    _layers: nn.Module

    def __init__(
        self,
        observation_shape: Sequence[int],
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
        activation: nn.Module = nn.ReLU(),
        exclude_last_activation: bool = False,
        last_activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        if hidden_units is None:
            hidden_units = [256, 256]

        layers = []
        in_units = [observation_shape[0]] + list(hidden_units[:-1])
        for is_last, (in_unit, out_unit) in last_flag(zip(in_units, hidden_units)):
            layers.append(nn.Linear(in_unit, out_unit))
            if not is_last or not exclude_last_activation:
                if is_last and last_activation:
                    layers.append(last_activation)
                else:
                    layers.append(activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_unit))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_unit))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
        self._layers = nn.Sequential(*layers)

    def forward(self, x: TorchObservation) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self._layers(x)


class VectorEncoderWithAction(EncoderWithAction):
    _layers: nn.Module
    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
        exclude_last_activation: bool = False,
        last_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self._action_size = action_size
        self._discrete_action = discrete_action

        if hidden_units is None:
            hidden_units = [256, 256]

        layers = []
        in_units = [observation_shape[0] + action_size] + list(hidden_units[:-1])
        for is_last, (in_unit, out_unit) in last_flag(zip(in_units, hidden_units)):
            layers.append(nn.Linear(in_unit, out_unit))
            if not is_last or not exclude_last_activation:
                if is_last and last_activation:
                    layers.append(last_activation)
                else:
                    layers.append(activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_unit))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_unit))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
        self._layers = nn.Sequential(*layers)

    def forward(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self._action_size
            ).float()
        x = torch.cat([x, action], dim=1)
        return self._layers(x)


@dataclass()
class VectorEncoderFactory(EncoderFactory):
    """Vector encoder factory class.

    This is the default encoder factory for vector observation.

    Args:
        hidden_units (list): List of hidden unit sizes. If ``None``, the
            standard architecture with ``[256, 256]`` is used.
        activation (str): activation function name.
        use_batch_norm (bool): Flag to insert batch normalization layers.
        use_layer_norm (bool): Flag to insert layer normalization layers.
        dropout_rate (float): Dropout probability.
        exclude_last_activation (bool): Flag to exclude activation function at
            the last layer.
        last_activation (str): Activation function name for the last layer.
    """

    hidden_units: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    exclude_last_activation: bool = False
    last_activation: Optional[str] = None

    def create(self, observation_shape: Shape) -> VectorEncoder:
        assert len(observation_shape) == 1
        return VectorEncoder(
            observation_shape=cast_flat_shape(observation_shape),
            hidden_units=self.hidden_units,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            activation=create_activation(self.activation),
            exclude_last_activation=self.exclude_last_activation,
            last_activation=(
                create_activation(self.last_activation)
                if self.last_activation
                else None
            ),
        )

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> VectorEncoderWithAction:
        assert len(observation_shape) == 1
        return VectorEncoderWithAction(
            observation_shape=cast_flat_shape(observation_shape),
            action_size=action_size,
            hidden_units=self.hidden_units,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            discrete_action=discrete_action,
            activation=create_activation(self.activation),
            exclude_last_activation=self.exclude_last_activation,
            last_activation=(
                create_activation(self.last_activation)
                if self.last_activation
                else None
            ),
        )

    @staticmethod
    def get_type() -> str:
        return "vector"


@dataclass()
class DefaultEncoderFactory(EncoderFactory):
    """Default encoder factory class.

    This encoder factory returns an encoder based on observation shape.

    Args:
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        dropout_rate (float): dropout probability.
    """

    activation: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: Optional[float] = None

    def create(self, observation_shape: Shape) -> Encoder:
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
            )
        else:
            factory = VectorEncoderFactory(
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
            )
        return factory.create(observation_shape)

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> EncoderWithAction:
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
            )
        else:
            factory = VectorEncoderFactory(
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
            )
        return factory.create_with_action(
            observation_shape, action_size, discrete_action
        )

    @staticmethod
    def get_type() -> str:
        return "default"


class EncoderWithAction(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        return super().__call__(x, action)


register_encoder_factory, make_encoder_field = generate_config_registration(
    EncoderFactory, lambda: DefaultEncoderFactory()
)


def compute_output_size(input_shapes: Sequence[Shape], encoder: nn.Module) -> int:
    device = next(encoder.parameters()).device
    with torch.no_grad():
        inputs = []
        for shape in input_shapes:
            if isinstance(shape[0], (list, tuple)):
                inputs.append([torch.rand(2, *s, device=device) for s in shape])
            else:
                inputs.append(torch.rand(2, *shape, device=device))
        y = encoder(*inputs)
    return int(y.shape[1])


# QFunction
#########################################################################


class QFunctionOutput(NamedTuple):
    q_value: torch.Tensor
    quantiles: Optional[torch.Tensor]
    taus: Optional[torch.Tensor]


class DiscreteQFunction(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation) -> QFunctionOutput:
        pass

    def __call__(self, x: TorchObservation) -> QFunctionOutput:
        return super().__call__(x)  # type: ignore

    @property
    @abstractmethod
    def encoder(self) -> Encoder:
        pass


class DiscreteQFunctionForwarder(metaclass=ABCMeta):
    @abstractmethod
    def compute_expected_q(self, x: TorchObservation) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(
        self, x: TorchObservation, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def set_q_func(self, q_func: DiscreteQFunction) -> None:
        pass


# DiscreteQFunction (Mean)
#############################################################################


class DiscreteMeanQFunction(DiscreteQFunction):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: TorchObservation) -> QFunctionOutput:
        return QFunctionOutput(
            q_value=self._fc(self._encoder(x)),
            quantiles=None,
            taus=None,
        )

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousQFunction(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation, action: torch.Tensor) -> QFunctionOutput:
        pass

    def __call__(self, x: TorchObservation, action: torch.Tensor) -> QFunctionOutput:
        return super().__call__(x, action)  # type: ignore

    @property
    @abstractmethod
    def encoder(self) -> EncoderWithAction:
        pass


class ContinuousQFunctionForwarder(metaclass=ABCMeta):
    @abstractmethod
    def compute_expected_q(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def set_q_func(self, q_func: ContinuousQFunction) -> None:
        pass


class DiscreteMeanQFunctionForwarder(DiscreteQFunctionForwarder):
    _q_func: DiscreteMeanQFunction
    _action_size: int

    def __init__(self, q_func: DiscreteMeanQFunction, action_size: int):
        self._q_func = q_func
        self._action_size = action_size

    def compute_expected_q(self, x: TorchObservation) -> torch.Tensor:
        return self._q_func(x).q_value

    # def compute_expected_q(self, x: TorchObservation, penalty:bool =False) -> torch.Tensor:
    #     expected_q_std = torch.std(extected_q, dim=1, keepdim= True)
    #     expected_q_penalty = torch.exp(-expected_q_std)
    #     extected_q = self.compute_expected_q_(x) if not penalty else self.compute_expected_q_(x) - torch.abs(expected_q_penalty * 0.0001)

    #     return extected_q #- torch.abs(penalty * 0.0001)

    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        try:
            one_hot = F.one_hot(actions.view(-1), num_classes=self._action_size)
        except RuntimeError:
            print(actions)

        value = (self._q_func(observations).q_value * one_hot.float()).sum(
            dim=1, keepdim=True
        )

        y = rewards + gamma * target * (1 - terminals)

        loss = compute_huber_loss(value, y)
        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: TorchObservation, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if action is None:
            return self._q_func(x).q_value
        return pick_value_by_action(self._q_func(x).q_value, action, keepdim=True)

    def set_q_func(self, q_func: DiscreteQFunction) -> None:
        self._q_func = q_func


# LRScheduler
# ###################################################


@dataclasses.dataclass()
class LRSchedulerFactory(DynamicConfig):
    """A factory class that creates a learning rate scheduler a lazy way."""

    def create(self, optim: Optimizer) -> LRScheduler:
        """Returns a learning rate scheduler object.

        Args:
            optim: PyTorch optimizer.

        Returns:
            Learning rate scheduler.
        """
        raise NotImplementedError


register_lr_scheduler_factory, make_lr_scheduler_field = (
    generate_optional_config_generation(
        LRSchedulerFactory,
    )
)
# optimizer
##################################################################


def _get_parameters_from_named_modules(
    named_modules: Iterable[tuple[str, nn.Module]],
) -> Sequence[nn.Parameter]:
    # retrieve unique set of parameters
    params_dict = {}
    for _, module in named_modules:
        for param in module.parameters():
            if param not in params_dict:
                params_dict[param] = param
    return list(params_dict.values())


class OptimizerWrapper:
    """OptimizerWrapper class.

    This class wraps PyTorch optimizer to add additional steps such as gradient
    clipping.

    Args:
        params: List of torch parameters.
        optim: PyTorch optimizer.
        compiled: Flag to be True if CudaGraph and torch.compile are applied.
        clip_grad_norm: Maximum norm value of gradients to clip.
    """

    _params: Sequence[nn.Parameter]
    _optim: Optimizer
    _compiled: bool
    _clip_grad_norm: Optional[float]
    _lr_scheduler: Optional[LRScheduler]

    def __init__(
        self,
        params: Sequence[nn.Parameter],
        optim: Optimizer,
        compiled: bool,
        clip_grad_norm: Optional[float] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        self._params = params
        self._optim = optim
        self._compiled = compiled
        self._clip_grad_norm = clip_grad_norm
        self._lr_scheduler = lr_scheduler

    def zero_grad(self) -> None:
        self._optim.zero_grad(set_to_none=self._compiled)

    def step(self) -> None:
        """Updates parameters.

        Args:
            grad_step: Total gradient step. This can be used for learning rate
                schedulers.
        """
        # clip gradients
        if self._clip_grad_norm:
            nn.utils.clip_grad_norm_(self._params, max_norm=self._clip_grad_norm)

        # update parameters
        self._optim.step()

        # schedule learning rate
        if self._lr_scheduler:
            self._lr_scheduler.step()

    @property
    def optim(self) -> Optimizer:
        return self._optim

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "optim": self._optim.state_dict(),
            "lr_scheduler": (
                self._lr_scheduler.state_dict() if self._lr_scheduler else None
            ),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if "optim" in state_dict:
            self._optim.load_state_dict(state_dict["optim"])
        else:
            LOG.warning("Skip loading optimizer state.")
        if self._lr_scheduler:
            if "lr_scheduler" in state_dict:
                self._lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            else:
                LOG.warning("Skip loading lr scheduler state.")


@dataclasses.dataclass()
class OptimizerFactory(DynamicConfig):
    """A factory class that creates an optimizer object in a lazy way.

    The optimizers in algorithms can be configured through this factory class.
    """

    clip_grad_norm: Optional[float] = None
    lr_scheduler_factory: Optional[LRSchedulerFactory] = make_lr_scheduler_field()

    def create(
        self,
        named_modules: Iterable[tuple[str, nn.Module]],
        lr: float,
        compiled: bool,
    ) -> OptimizerWrapper:
        """Returns an optimizer object.

        Args:
            named_modules (list): List of tuples of module names and modules.
            lr (float): Learning rate.
            compiled (bool): Flag to be True if CudaGraph and torch.compile are
                applied.

        Returns:
            OptimizerWrapper object.
        """
        named_modules = list(named_modules)
        params = _get_parameters_from_named_modules(named_modules)
        optim = self.create_optimizer(named_modules, lr)
        return OptimizerWrapper(
            params=params,
            optim=optim,
            compiled=compiled,
            clip_grad_norm=self.clip_grad_norm,
            lr_scheduler=(
                self.lr_scheduler_factory.create(optim)
                if self.lr_scheduler_factory
                else None
            ),
        )

    def create_optimizer(
        self, named_modules: Iterable[tuple[str, nn.Module]], lr: float
    ) -> Optimizer:
        raise NotImplementedError


@dataclasses.dataclass()
class AdamFactory(OptimizerFactory):
    """An alias for Adam optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import AdamFactory

        factory = AdamFactory(weight_decay=1e-4)

    Args:
        clip_grad_norm: Maximum norm value of gradients to clip.
        lr_scheduler_factory: LRSchedulerFactory.
        betas: coefficients used for computing running averages of
            gradient and its square.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        amsgrad: flag to use the AMSGrad variant of this algorithm.
    """

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

    def create_optimizer(
        self, named_modules: Iterable[tuple[str, nn.Module]], lr: float
    ) -> Adam:
        return Adam(
            params=_get_parameters_from_named_modules(named_modules),
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

    @staticmethod
    def get_type() -> str:
        return "adam"


register_optimizer_factory, make_optimizer_field = generate_config_registration(
    OptimizerFactory, lambda: AdamFactory()
)

# Model Q function
##################################################################


@dataclasses.dataclass()
class QFunctionFactory(DynamicConfig):
    share_encoder: bool = False

    def create_discrete(
        self, encoder: Encoder, hidden_size: int, action_size: int
    ) -> tuple[DiscreteQFunction, DiscreteQFunctionForwarder]:
        """Returns PyTorch's Q function module.

        Args:
            encoder: Encoder that processes the observation to
                obtain feature representations.
            hidden_size: Dimension of encoder output.
            action_size: Dimension of discrete action-space.

        Returns:
            Tuple of discrete Q function and its forwarder.
        """
        raise NotImplementedError

    # def create_continuous(
    #     self, encoder: EncoderWithAction, hidden_size: int
    # ) -> tuple[ContinuousQFunction, ContinuousQFunctionForwarder]:
    #     """Returns PyTorch's Q function module.

    #     Args:
    #         encoder: Encoder module that processes the observation and
    #             action to obtain feature representations.
    #         hidden_size: Dimension of encoder output.

    #     Returns:
    #         Tuple of continuous Q function and its forwarder.
    #     """
    #     raise NotImplementedError

    @staticmethod
    def get_type() -> str:
        """Returns Q function type.

        Returns:
            Q function type.
        """
        raise NotImplementedError


@dataclasses.dataclass()
class MeanQFunctionFactory(QFunctionFactory):
    """Standard Q function factory class.

    This is the standard Q function factory class.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_

    Args:
        share_encoder (bool): flag to share encoder over multiple Q functions.
    """

    def create_discrete(
        self,
        encoder: Encoder,
        hidden_size: int,
        action_size: int,
    ) -> tuple[DiscreteMeanQFunction, DiscreteMeanQFunctionForwarder]:
        q_func = DiscreteMeanQFunction(encoder, hidden_size, action_size)
        forwarder = DiscreteMeanQFunctionForwarder(q_func, action_size)
        return q_func, forwarder

    # def create_continuous(
    #     self,
    #     encoder: EncoderWithAction,
    #     hidden_size: int,
    # ) -> tuple[ContinuousMeanQFunction, ContinuousMeanQFunctionForwarder]:
    #     q_func = ContinuousMeanQFunction(encoder, hidden_size)
    #     forwarder = ContinuousMeanQFunctionForwarder(q_func)
    #     return q_func, forwarder

    @staticmethod
    def get_type() -> str:
        return "mean"


register_q_func_factory, make_q_func_field = generate_config_registration(
    QFunctionFactory, lambda: MeanQFunctionFactory()
)


# Torch policies
###################################################################


class ActionOutput(NamedTuple):
    mu: torch.Tensor
    squashed_mu: torch.Tensor
    logstd: Optional[torch.Tensor]

    def copy_(self, src: "ActionOutput") -> None:
        self.mu.copy_(src.mu)
        self.squashed_mu.copy_(src.squashed_mu)
        if self.logstd:
            assert src.logstd is not None
            self.logstd.copy_(src.logstd)


class Policy(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation, *args: Any) -> ActionOutput:
        pass

    def __call__(self, x: TorchObservation, *args: Any) -> ActionOutput:
        return super().__call__(x, *args)  # type: ignore


class CategoricalPolicy(nn.Module):  # type: ignore
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: TorchObservation) -> Categorical:
        return Categorical(logits=self._fc(self._encoder(x)))

    def __call__(self, x: TorchObservation) -> Categorical:
        return super().__call__(x)


# Torch parmaters
###################################################################
class Parameter(nn.Module):  # type: ignore
    _parameter: nn.Parameter

    def __init__(self, data: torch.Tensor):
        super().__init__()
        self._parameter = nn.Parameter(data)

    def forward(self) -> NoReturn:
        raise NotImplementedError(
            "Parameter does not support __call__. Use parameter property " "instead."
        )

    def __call__(self) -> NoReturn:
        raise NotImplementedError(
            "Parameter does not support __call__. Use parameter property " "instead."
        )


def get_parameter(parameter: Parameter) -> nn.Parameter:
    return next(parameter.parameters())


# Torch util
####################################################################

_T = TypeVar("_T", bound=Union[torch.Tensor, Sequence[torch.Tensor]])


def copy_recursively(src: _T, dst: _T) -> None:
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        dst.copy_(src)
    elif isinstance(src, (list, tuple)) and isinstance(dst, (list, tuple)):
        for s, d in zip(src, dst):
            d.copy_(s)
    else:
        raise ValueError(f"invalid inpu types: src={type(src)}, dst={type(dst)}")


def convert_to_torch(array: NDArray, device: str) -> torch.Tensor:
    dtype = torch.uint8 if array.dtype == np.uint8 else torch.float32
    tensor = torch.tensor(data=array, dtype=dtype, device=device)
    return tensor.float()


def convert_to_torch_recursively(
    array: Union[NDArray, Sequence[NDArray]], device: str
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    if isinstance(array, (list, tuple)):
        return [convert_to_torch(data, device) for data in array]
    elif isinstance(array, np.ndarray):
        return convert_to_torch(array, device)
    else:
        raise ValueError(f"invalid array type: {type(array)}")


def _compute_return_to_go(
    gamma: float,
    rewards_to_go: Float32NDArray,
    reward_scaler: Optional[RewardScaler],
) -> Float32NDArray:
    rewards = (
        reward_scaler.transform_numpy(rewards_to_go) if reward_scaler else rewards_to_go
    )
    cum_gammas: Float32NDArray = np.array(
        np.expand_dims(gamma ** np.arange(rewards.shape[0]), axis=1),
        dtype=np.float32,
    )
    return np.sum(cum_gammas * rewards, axis=0)  # type: ignore


@dataclasses.dataclass(frozen=True)
class TorchMiniBatch:
    observations: TorchObservation
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: TorchObservation
    next_actions: torch.Tensor
    returns_to_go: torch.Tensor
    terminals: torch.Tensor
    intervals: torch.Tensor
    device: str
    numpy_batch: Optional[TransitionMiniBatch] = None

    @classmethod
    def from_batch(
        cls,
        batch: TransitionMiniBatch,
        gamma: float,
        compute_returns_to_go: bool,
        device: str,
        observation_scaler: Optional[ObservationScaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ) -> "TorchMiniBatch":
        # convert numpy array to torch tensor
        observations = convert_to_torch_recursively(batch.observations, device)
        actions = convert_to_torch(batch.actions, device)
        next_actions = convert_to_torch(batch.next_actions, device)
        rewards = convert_to_torch(batch.rewards, device)
        next_observations = convert_to_torch_recursively(
            batch.next_observations, device
        )
        terminals = convert_to_torch(batch.terminals, device)
        intervals = convert_to_torch(batch.intervals, device)

        if compute_returns_to_go:
            returns_to_go = convert_to_torch(
                np.array(
                    [
                        _compute_return_to_go(
                            gamma=gamma,
                            rewards_to_go=transition.rewards_to_go,
                            reward_scaler=reward_scaler,
                        )
                        for transition in batch.transitions
                    ]
                ),
                device,
            )
        else:
            returns_to_go = torch.zeros_like(rewards)

        # apply scaler
        if observation_scaler:
            observations = observation_scaler.transform(observations)
            next_observations = observation_scaler.transform(next_observations)
        if action_scaler:
            actions = action_scaler.transform(actions)
            next_actions = action_scaler.transform(next_actions)
        if reward_scaler:
            rewards = reward_scaler.transform(rewards)

        return TorchMiniBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_actions=next_actions,
            returns_to_go=returns_to_go,
            terminals=terminals,
            intervals=intervals,
            device=device,
            numpy_batch=batch,
        )

    def copy_(self, src: Self) -> None:
        assert self.device == src.device, "incompatible device"
        copy_recursively(src.observations, self.observations)
        self.actions.copy_(src.actions)
        self.rewards.copy_(src.rewards)
        copy_recursively(src.next_observations, self.next_observations)
        self.next_actions.copy_(src.next_actions)
        self.returns_to_go.copy_(src.returns_to_go)
        self.terminals.copy_(src.terminals)
        self.intervals.copy_(src.intervals)


def eval_api(f: TCallable) -> TCallable:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        assert hasattr(self, "modules")
        assert isinstance(self.modules, Modules)
        self.modules.set_eval()
        return f(self, *args, **kwargs)  # type: ignore

    return wrapper  # type: ignore


def hard_sync(targ_model: nn.Module, model: nn.Module) -> None:
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.copy_(p.data)


def sync_optimizer_state(targ_optim: Optimizer, optim: Optimizer) -> None:
    # source optimizer state
    state = optim.state_dict()["state"]
    # destination optimizer param_groups
    param_groups = targ_optim.state_dict()["param_groups"]
    # update only state
    targ_optim.load_state_dict({"state": state, "param_groups": param_groups})


# Interface


####################################################################
class QLearningAlgoProtocol(Protocol):
    def predict(self, x: Observation) -> NDArray: ...

    def predict_value(self, x: Observation, action: NDArray) -> NDArray: ...

    def sample_action(self, x: Observation) -> NDArray: ...

    @property
    def gamma(self) -> float: ...

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]: ...

    @property
    def action_scaler(self) -> Optional[ActionScaler]: ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]: ...

    @property
    def action_size(self) -> Optional[int]: ...


# Metrics evaluator
####################################################################
class EvaluatorProtocol(Protocol):
    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBufferBase,
    ) -> float:
        """Computes metrics.

        Args:
            algo: Q-learning algorithm.
            dataset: ReplayBuffer.

        Returns:
            Computed metrics.
        """
        raise NotImplementedError


# explorer

####################################################################


class Explorer(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, algo: QLearningAlgoProtocol, x: Observation, step: int) -> NDArray:
        pass


# buffers
####################################################################


class BufferProtocol(Protocol):
    r"""Interface of Buffer."""

    def append(self, episode: EpisodeBase, index: int) -> None:
        r"""Adds transition to buffer.

        Args:
            episode: Episode object.
            index: Transition index.
        """
        raise NotImplementedError

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        r"""Returns list of episodes.

        Returns:
            List of saved episodes.
        """
        raise NotImplementedError

    @property
    def transition_count(self) -> int:
        r"""Returns the number of transitions.

        Returns:
            Number of transitions.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[EpisodeBase, int]:
        raise NotImplementedError


class FIFOBuffer(BufferProtocol):
    r"""FIFO buffer.

    Args:
        limit (int): buffer capacity.
    """

    _transitions: deque[tuple[EpisodeBase, int]]
    _episodes: list[EpisodeBase]
    _limit: int

    def __init__(self, limit: int):
        self._limit = limit
        self._transitions = deque(maxlen=limit)
        self._episodes = []

    def append(self, episode: EpisodeBase, index: int) -> None:
        if len(self._transitions) == self._limit:
            # check if dropped transition is the last transition
            if self._transitions[0][0] is not self._transitions[1][0]:
                self._episodes.pop(0)
        self._transitions.append((episode, index))
        if not self._episodes or episode is not self._episodes[-1]:
            self._episodes.append(episode)

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return self._episodes

    @property
    def transition_count(self) -> int:
        return len(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(self, index: int) -> tuple[EpisodeBase, int]:
        return self._transitions[index]


# Replay buffer
####################################################################
def create_fifo_replay_buffer(
    limit: int,
    episodes: Optional[Sequence[EpisodeBase]] = None,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    env: Optional[GymEnv] = None,
    write_at_termination: bool = False,
) -> ReplayBuffer:
    """Builds FIFO replay buffer.

    This function is a shortcut alias to build replay buffer with
    ``FIFOBuffer``.

    Args:
        limit: Maximum capacity of FIFO buffer.
        episodes: List of episodes to initialize replay buffer.
        transition_picker:
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer:
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        writer_preprocessor:
            Writer preprocessor implementation. If ``None`` is given,
            ``BasicWriterPreprocess`` is used by default.
        env: Gym environment to extract shapes of observations and action.
        write_at_termination (bool): Flag to write experiences to the buffer at
            the end of an episode all at once.

    Returns:
        Replay buffer.
    """
    buffer = FIFOBuffer(limit)
    return ReplayBuffer(
        buffer,
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
        writer_preprocessor=writer_preprocessor,
        env=env,
        write_at_termination=write_at_termination,
    )


# Qlearning in base
####################################################################


def evaluate_qlearning_with_environment(
    algo: QLearningAlgoProtocol,
    env: GymEnv,
    n_trials: int = 10,
    epsilon: float = 0.0,
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    episode_rewards = []
    for _ in range(n_trials):
        observation, _ = env.reset()
        episode_reward = 0.0

        while True:
            # take action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                if isinstance(observation, np.ndarray):
                    observation = np.expand_dims(observation, axis=0)
                elif isinstance(observation, (tuple, list)):
                    observation = [np.expand_dims(o, axis=0) for o in observation]
                else:
                    raise ValueError(
                        f"Unsupported observation type: {type(observation)}"
                    )
                action = algo.predict(observation)[0]

            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)

            if done or truncated:
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))


class QLearningAlgoImplBase(ImplBase):
    @train_api
    def update(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
        return self.inner_update(batch, grad_step)

    @abstractmethod
    def inner_update(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
        pass

    @eval_api
    def predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    @abstractmethod
    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        pass

    @eval_api
    def sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_sample_action(x)

    @abstractmethod
    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        pass

    @eval_api
    def predict_value(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_value(x, action)

    @abstractmethod
    def inner_predict_value(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        pass

    @property
    def policy(self) -> Policy:
        raise NotImplementedError

    def copy_policy_from(self, impl: "QLearningAlgoImplBase") -> None:
        if not isinstance(impl.policy, type(self.policy)):
            raise ValueError(
                f"Invalid policy type: expected={type(self.policy)},"
                f"actual={type(impl.policy)}"
            )
        hard_sync(self.policy, impl.policy)

    @property
    def policy_optim(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def copy_policy_optim_from(self, impl: "QLearningAlgoImplBase") -> None:
        if not isinstance(impl.policy_optim, type(self.policy_optim)):
            raise ValueError(
                "Invalid policy optimizer type: "
                f"expected={type(self.policy_optim)},"
                f"actual={type(impl.policy_optim)}"
            )
        sync_optimizer_state(self.policy_optim, impl.policy_optim)

    @property
    def q_function(self) -> nn.ModuleList:
        raise NotImplementedError

    def copy_q_function_from(self, impl: "QLearningAlgoImplBase") -> None:
        q_func = self.q_function[0]
        if not isinstance(impl.q_function[0], type(q_func)):
            raise ValueError(
                f"Invalid Q-function type: expected={type(q_func)},"
                f"actual={type(impl.q_function[0])}"
            )
        hard_sync(self.q_function, impl.q_function)

    @property
    def q_function_optim(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def copy_q_function_optim_from(self, impl: "QLearningAlgoImplBase") -> None:
        if not isinstance(impl.q_function_optim, type(self.q_function_optim)):
            raise ValueError(
                "Invalid Q-function optimizer type: "
                f"expected={type(self.q_function_optim)}",
                f"actual={type(impl.q_function_optim)}",
            )
        sync_optimizer_state(self.q_function_optim, impl.q_function_optim)

    def reset_optimizer_states(self) -> None:
        self.modules.reset_optimizer_states()


TQLearningImpl = TypeVar("TQLearningImpl", bound=QLearningAlgoImplBase)
TQLearningConfig = TypeVar("TQLearningConfig", bound=LearnableConfig)


class QLearningAlgoBase(
    Generic[TQLearningImpl, TQLearningConfig],
    LearnableBase[TQLearningImpl, TQLearningConfig],
):
    def save_policy(self, fname: str) -> None:
        """Save the greedy-policy computational graph as TorchScript or ONNX.

        The format will be automatically detected by the file name.

        .. code-block:: python

            # save as TorchScript
            algo.save_policy('policy.pt')

            # save as ONNX
            algo.save_policy('policy.onnx')

        The artifacts saved with this method will work without d3rlpy.
        This method is especially useful to deploy the learned policy to
        production environments or embedding systems.

        See also

            * https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html (for Python).
            * https://pytorch.org/tutorials/advanced/cpp_export.html (for C++).
            * https://onnx.ai (for ONNX)

        Visit https://d3rlpy.readthedocs.io/en/stable/tutorials/after_training_policies.html#export-policies-as-torchscript for the further usage.

        Args:
            fname: Destination file path.
        """  # noqa: E501
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        if is_tuple_shape(self._impl.observation_shape):
            dummy_x = [
                torch.rand(1, *shape, device=self._device)
                for shape in self._impl.observation_shape
            ]
            num_inputs = len(self._impl.observation_shape)
        else:
            dummy_x = torch.rand(1, *self._impl.observation_shape, device=self._device)
            num_inputs = 1

        # workaround until version 1.6
        self._impl.modules.freeze()

        # local function to select best actions
        def _func(*x: Sequence[torch.Tensor]) -> torch.Tensor:
            assert self._impl

            observation: TorchObservation = x
            if len(observation) == 1:
                observation = observation[0]

            if self._config.observation_scaler:
                observation = self._config.observation_scaler.transform(observation)

            action = self._impl.predict_best_action(observation)

            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

            return action

        traced_script = torch.jit.trace(_func, dummy_x, check_trace=False)

        if fname.endswith(".onnx"):
            # currently, PyTorch cannot directly export function as ONNX.
            torch.onnx.export(
                traced_script,
                dummy_x,
                fname,
                export_params=True,
                opset_version=11,
                input_names=[f"input_{i}" for i in range(num_inputs)],
                output_names=["output_0"],
            )
        elif fname.endswith(".pt"):
            traced_script.save(fname)
        else:
            raise ValueError(
                f"invalid format type: {fname}."
                " .pt and .onnx extensions are currently supported."
            )

        # workaround until version 1.6
        self._impl.modules.unfreeze()

    def predict(self, x: Observation) -> NDArray:
        """Returns greedy actions.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            actions = algo.predict(x)
            # actions.shape == (100, action size) for continuous control
            # actions.shape == (100,) for discrete control

        Args:
            x: Observations

        Returns:
            Greedy actions
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        torch_x = convert_to_torch_recursively(x, self._device)

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            action = self._impl.predict_best_action(torch_x)

            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

        return action.cpu().detach().numpy()  # type: ignore

    def predict_value(self, x: Observation, action: NDArray) -> NDArray:
        """Returns predicted action-values.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            # for continuous control
            # 100 actions with shape of (2,)
            actions = np.random.random((100, 2))

            # for discrete control
            # 100 actions in integer values
            actions = np.random.randint(2, size=100)

            values = algo.predict_value(x, actions)
            # values.shape == (100,)

        Args:
            x: Observations
            action: Actions

        Returns:
            Predicted action-values
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        torch_x = convert_to_torch_recursively(x, self._device)

        torch_action = convert_to_torch(action, self._device)

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            if self.get_action_type() == ActionSpace.CONTINUOUS:
                if self._config.action_scaler:
                    torch_action = self._config.action_scaler.transform(torch_action)
            elif self.get_action_type() == ActionSpace.DISCRETE:
                torch_action = torch_action.long()
            else:
                raise ValueError("invalid action type")

            value = self._impl.predict_value(torch_x, torch_action)

        return value.cpu().detach().numpy()  # type: ignore

    def sample_action(self, x: Observation) -> NDArray:
        """Returns sampled actions.

        The sampled actions are identical to the output of `predict` method if
        the policy is deterministic.

        Args:
            x: Observations.

        Returns:
            Sampled actions.
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        torch_x = convert_to_torch_recursively(x, self._device)

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            action = self._impl.sample_action(torch_x)

            # transform action back to the original range
            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

        return action.cpu().detach().numpy()  # type: ignore

    def fit(
        self,
        dataset: ReplayBufferBase,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logging_steps: int = 500,
        logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> list[tuple[int, dict[str, float]]]:
        """Trains with given dataset.

        .. code-block:: python

            algo.fit(episodes, n_steps=1000000)

        Args:
            dataset: ReplayBuffer object.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logging_steps: Number of steps to log metrics. This will be ignored
                if logging_strategy is EPOCH.
            logging_strategy: Logging strategy to use.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            save_interval: Interval to save parameters.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            List of result tuples (epoch, metrics) per epoch.
        """
        results = list(
            self.fitter(
                dataset=dataset,
                n_steps=n_steps,
                n_steps_per_epoch=n_steps_per_epoch,
                experiment_name=experiment_name,
                with_timestamp=with_timestamp,
                logging_steps=logging_steps,
                logging_strategy=logging_strategy,
                logger_adapter=logger_adapter,
                show_progress=show_progress,
                save_interval=save_interval,
                evaluators=evaluators,
                callback=callback,
                epoch_callback=epoch_callback,
            )
        )
        return results

    def fitter(
        self,
        dataset: ReplayBufferBase,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        logging_steps: int = 500,
        logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> Generator[tuple[int, dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
        iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: Offline dataset to train.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logging_steps: Number of steps to log metrics. This will be ignored
                if logging_strategy is EPOCH.
            logging_strategy: Logging strategy to use.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            save_interval: Interval to save parameters.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            Iterator yielding current epoch and metrics dict.
        """
        LOG.info("dataset info", dataset_info=dataset.dataset_info)

        # check action space
        assert_action_space_with_dataset(self, dataset.dataset_info)

        # initialize scalers
        build_scalers_with_transition_picker(self, dataset)

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            action_size = dataset.dataset_info.action_size
            observation_shape = dataset.sample_transition().observation_signature.shape

            if len(observation_shape) == 1:
                observation_shape = observation_shape[0]  # type: ignore
            self.create_impl(observation_shape, action_size)
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__
        logger = D3RLPyLogger(
            algo=self,
            adapter_factory=logger_adapter,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            with_timestamp=with_timestamp,
        )

        # save hyperparameters
        save_config(self, logger)

        # training loop
        n_epochs = n_steps // n_steps_per_epoch
        total_step = 0
        for epoch in range(1, n_epochs + 1):
            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(n_steps_per_epoch),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            for itr in range_gen:
                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):

                        batch = dataset.sample_transition_batch(self._config.batch_size)

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {k: np.mean(v) for k, v in epoch_loss.items()}
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                if (
                    logging_strategy == LoggingStrategy.STEPS
                    and total_step % logging_steps == 0
                ):
                    metrics = logger.commit(epoch, total_step)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # call epoch_callback if given
            if epoch_callback:
                epoch_callback(self, epoch, total_step)

            if evaluators:
                for name, evaluator in evaluators.items():
                    test_score = evaluator(self, dataset)
                    logger.add_metric(name, test_score)

            # save metrics
            if logging_strategy == LoggingStrategy.EPOCH:
                metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        logger.close()

    def fit_online(
        self,
        env: GymEnv,
        buffer: Optional[ReplayBufferBase] = None,
        explorer: Optional[Explorer] = None,
        n_steps: int = 1000000,
        n_steps_per_epoch: int = 10000,
        update_interval: int = 1,
        n_updates: int = 1,
        update_start_step: int = 0,
        random_steps: int = 0,
        eval_env: Optional[GymEnv] = None,
        eval_epsilon: float = 0.0,
        eval_n_trials: int = 10,
        save_interval: int = 1,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logging_steps: int = 500,
        logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> None:
        """Start training loop of online deep reinforcement learning.

        Args:
            env: Gym-like environment.
            buffer : Replay buffer.
            explorer: Action explorer.
            n_steps: Number of total steps to train.
            n_steps_per_epoch: Number of steps per epoch.
            update_interval: Number of steps per update.
            n_updates: Number of gradient steps at a time. The combination of
                ``update_interval`` and ``n_updates`` controls Update-To-Data
                (UTD) ratio.
            update_start_step: Steps before starting updates.
            random_steps: Steps for the initial random explortion.
            eval_env: Gym-like environment. If None, evaluation is skipped.
            eval_epsilon: :math:`\\epsilon`-greedy factor during evaluation.
            save_interval: Number of epochs before saving models.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be ``{class name}_online_{timestamp}``.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logging_steps: Number of steps to log metrics. This will be ignored
                if logging_strategy is EPOCH.
            logging_strategy: Logging strategy to use.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called at the end of epochs.
        """

        # create default replay buffer
        if buffer is None:
            buffer = create_fifo_replay_buffer(1000000, env=env)

        # check action-space
        assert_action_space_with_env(self, env)

        # initialize algorithm parameters
        build_scalers_with_env(self, env)

        # setup algorithm
        if self.impl is None:
            LOG.debug("Building model...")
            self.build_with_env(env)
            LOG.debug("Model has been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__ + "_online"
        logger = D3RLPyLogger(
            algo=self,
            adapter_factory=logger_adapter,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            with_timestamp=with_timestamp,
        )

        # save hyperparameters
        save_config(self, logger)

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        # start training loop
        observation, _ = env.reset()
        rollout_return = 0.0
        for total_step in xrange(1, n_steps + 1):
            with logger.measure_time("step"):
                # sample exploration action
                with logger.measure_time("inference"):
                    if total_step < random_steps:
                        action = env.action_space.sample()
                    elif explorer:
                        x = observation.reshape((1,) + observation.shape)
                        action = explorer.sample(self, x, total_step)[0]
                    else:
                        action = self.sample_action(
                            np.expand_dims(observation, axis=0)
                        )[0]

                # step environment
                with logger.measure_time("environment_step"):
                    (
                        next_observation,
                        reward,
                        terminal,
                        truncated,
                        _,
                    ) = env.step(action)
                    rollout_return += float(reward)

                clip_episode = terminal or truncated

                # store observation
                buffer.append(observation, action, float(reward))

                # reset if terminated
                if clip_episode:
                    buffer.clip_episode(terminal)
                    observation, _ = env.reset()
                    logger.add_metric("rollout_return", rollout_return)
                    rollout_return = 0.0
                else:
                    observation = next_observation

                # psuedo epoch count
                epoch = total_step // n_steps_per_epoch

                if (
                    total_step > update_start_step
                    and buffer.transition_count > self.batch_size
                ):
                    if total_step % update_interval == 0:
                        for _ in range(n_updates):  # controls UTD ratio
                            # sample mini-batch
                            with logger.measure_time("sample_batch"):
                                batch = buffer.sample_transition_batch(self.batch_size)

                            # update parameters
                            with logger.measure_time("algorithm_update"):
                                loss = self.update(batch)

                            # record metrics
                            for name, val in loss.items():
                                logger.add_metric(name, val)

                        if (
                            logging_strategy == LoggingStrategy.STEPS
                            and total_step % logging_steps == 0
                        ):
                            logger.commit(epoch, total_step)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            if epoch > 0 and total_step % n_steps_per_epoch == 0:
                # evaluation
                if eval_env:
                    eval_score = evaluate_qlearning_with_environment(
                        self,
                        eval_env,
                        n_trials=eval_n_trials,
                        epsilon=eval_epsilon,
                    )
                    logger.add_metric("evaluation", eval_score)

                if epoch % save_interval == 0:
                    logger.save_model(total_step, self)

                # save metrics
                if logging_strategy == LoggingStrategy.EPOCH:
                    logger.commit(epoch, total_step)

        # clip the last episode
        buffer.clip_episode(False)

        # close logger
        logger.close()

    def collect(
        self,
        env: GymEnv,
        buffer: Optional[ReplayBufferBase] = None,
        explorer: Optional[Explorer] = None,
        deterministic: bool = False,
        n_steps: int = 1000000,
        show_progress: bool = True,
    ) -> ReplayBufferBase:
        """Collects data via interaction with environment.

        If ``buffer`` is not given, ``ReplayBuffer`` will be internally created.

        Args:
            env: Fym-like environment.
            buffer: Replay buffer.
            explorer: Action explorer.
            deterministic: Flag to collect data with the greedy policy.
            n_steps: Number of total steps to train.
            show_progress: Flag to show progress bar for iterations.

        Returns:
            Replay buffer with the collected data.
        """
        # create default replay buffer
        if buffer is None:
            buffer = create_fifo_replay_buffer(1000000, env=env)

        # check action-space
        assert_action_space_with_env(self, env)

        # initialize algorithm parameters
        build_scalers_with_env(self, env)

        # setup algorithm
        if self.impl is None:
            LOG.debug("Building model...")
            self.build_with_env(env)
            LOG.debug("Model has been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        # start training loop
        observation, _ = env.reset()
        for total_step in xrange(1, n_steps + 1):
            # sample exploration action
            if deterministic:
                action = self.predict(np.expand_dims(observation, axis=0))[0]
            else:
                if explorer:
                    x = observation.reshape((1,) + observation.shape)
                    action = explorer.sample(self, x, total_step)[0]
                else:
                    action = self.sample_action(np.expand_dims(observation, axis=0))[0]

            # step environment
            next_observation, reward, terminal, truncated, _ = env.step(action)

            clip_episode = terminal or truncated

            # store observation
            buffer.append(observation, action, float(reward))

            # reset if terminated
            if clip_episode:
                buffer.clip_episode(terminal)
                observation, _ = env.reset()
            else:
                observation = next_observation

        # clip the last episode
        buffer.clip_episode(False)

        return buffer

    def update(self, batch: TransitionMiniBatch) -> dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: Mini-batch data.

        Returns:
            Dictionary of metrics.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        torch_batch = TorchMiniBatch.from_batch(
            batch=batch,
            gamma=self._config.gamma,
            compute_returns_to_go=self.need_returns_to_go,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )
        loss = self._impl.update(torch_batch, self._grad_step)
        self._grad_step += 1
        return loss

    @property
    def need_returns_to_go(self) -> bool:
        return False

    def copy_policy_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
        """Copies policy parameters from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithm
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_policy_from(cql)

        Args:
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_policy_from(algo.impl)

    def copy_policy_optim_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
        """Copies policy optimizer states from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithm
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_policy_optim_from(cql)

        Args:
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_policy_optim_from(algo.impl)

    def copy_q_function_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
        """Copies Q-function parameters from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithmn
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_q_function_from(cql)

        Args:
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_q_function_from(algo.impl)

    def copy_q_function_optim_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
        """Copies Q-function optimizer states from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithm
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_policy_optim_from(cql)

        Args:
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_q_function_optim_from(algo.impl)

    def reset_optimizer_states(self) -> None:
        """Resets optimizer states.

        This is especially useful when fine-tuning policies with setting inital
        optimizer states.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        self._impl.reset_optimizer_states()


# SAC IMplementation
###################################################################


# torch Q function ensemble
###################################################################


def _reduce_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    if reduction == "min":
        return y.min(dim=dim).values
    elif reduction == "max":
        return y.max(dim=dim).values
    elif reduction == "mean":
        return y.mean(dim=dim)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def _gather_quantiles_by_indices(
    y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # TODO: implement this in general case
    if y.dim() == 3:
        # (N, batch, n_quantiles) -> (batch, n_quantiles)
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif y.dim() == 4:
        # (N, batch, action, n_quantiles) -> (batch, action, N, n_quantiles)
        transposed_y = y.transpose(0, 1).transpose(1, 2)
        # (batch, action, N, n_quantiles) -> (batch * action, N, n_quantiles)
        flat_y = transposed_y.reshape(-1, y.shape[0], y.shape[3])
        head_indices = torch.arange(y.shape[1] * y.shape[2])
        # (batch * action, N, n_quantiles) -> (batch * action, n_quantiles)
        gathered_y = flat_y[head_indices, indices.view(-1)]
        # (batch * action, n_quantiles) -> (batch, action, n_quantiles)
        return gathered_y.view(y.shape[1], y.shape[2], -1)
    raise ValueError


def _reduce_quantile_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    # reduction beased on expectation
    mean = y.mean(dim=-1)
    if reduction == "min":
        indices = mean.min(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "max":
        indices = mean.max(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        min_indices = mean.min(dim=dim).indices
        max_indices = mean.max(dim=dim).indices
        min_values = _gather_quantiles_by_indices(y, min_indices)
        max_values = _gather_quantiles_by_indices(y, max_indices)
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def compute_ensemble_q_function_target(
    forwarders: Union[
        Sequence[DiscreteQFunctionForwarder],
        Sequence[ContinuousQFunctionForwarder],
    ],
    action_size: int,
    x: TorchObservation,
    action: Optional[torch.Tensor] = None,
    reduction: str = "min",
    lam: float = 0.75,
) -> torch.Tensor:
    batch_size = get_batch_size(x)
    values_list: list[torch.Tensor] = []
    for forwarder in forwarders:
        if isinstance(forwarder, ContinuousQFunctionForwarder):
            assert action is not None
            target = forwarder.compute_target(x, action)
        else:
            target = forwarder.compute_target(x, action)
        values_list.append(target.reshape(1, batch_size, -1))

    values = torch.cat(values_list, dim=0)

    if action is None:
        # mean Q function
        if values.shape[2] == action_size:
            return _reduce_ensemble(values, reduction)
        # distributional Q function
        n_q_funcs = values.shape[0]
        values = values.view(n_q_funcs, batch_size, action_size, -1)
        return _reduce_quantile_ensemble(values, reduction)

    if values.shape[2] == 1:
        return _reduce_ensemble(values, reduction, lam=lam)

    return _reduce_quantile_ensemble(values, reduction, lam=lam)


def compute_ensemble_q_function_error(
    forwarders: Union[
        Sequence[DiscreteQFunctionForwarder],
        Sequence[ContinuousQFunctionForwarder],
    ],
    observations: TorchObservation,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    target: torch.Tensor,
    terminals: torch.Tensor,
    gamma: Union[float, torch.Tensor] = 0.99,
    masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert target.ndim == 2
    td_sum = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=get_device(observations),
    )
    for forwarder in forwarders:
        loss = forwarder.compute_error(
            observations=observations,
            actions=actions,
            rewards=rewards,
            target=target,
            terminals=terminals,
            gamma=gamma,
            reduction="none",
        )
        if masks is not None:
            loss = loss * masks
        td_sum += loss.mean()
    return td_sum


# (New) Ensemble Q function forwarder
###################################################################
def compute_ensemble_q_function_uncertainty(
    forwarders: Union[
        Sequence[DiscreteQFunctionForwarder],
        Sequence[ContinuousQFunctionForwarder],
    ],
    action_size: int,
    x: TorchObservation,
    action: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes an uncertainty measure (e.g. std) across the ensemble of Q-functions.
    Returns a tensor of shape [batch_size, 1] for each sample's uncertainty.
    """
    batch_size = get_batch_size(x)
    values_list: list[torch.Tensor] = []

    # 1) Forward pass for each Q-func in the ensemble
    #    (Just like compute_ensemble_q_function_target)
    for forwarder in forwarders:
        if isinstance(forwarder, ContinuousQFunctionForwarder):
            assert action is not None, "action must be provided for continuous Q"
            out = forwarder.compute_target(x, action)
        else:
            out = forwarder.compute_target(x, action)

        # Reshape to: [1, batch_size, -1]
        # so we can later concat along dim=0 for each ensemble member
        values_list.append(out.reshape(1, batch_size, -1))

    # shape = [n_ensembles, batch_size, some_dim]
    values = torch.cat(values_list, dim=0)

    # 2) If we have discrete Q-values, shape is [n_ensembles, batch_size, action_size]
    #    We'll compute the "average across actions" first, then std across ensembles
    if values.shape[2] == action_size:
        # shape = [n_ensembles, batch_size, action_size]
        values_mean = values.mean(dim=2)  # shape [n_ensembles, batch_size]
        values_std = values_mean.std(dim=0)  # shape [batch_size]
        return values_std.unsqueeze(1)  # shape [batch_size, 1]

    # 3) If it's a distributional Q function (quantile-based),
    #    you may need different logic. For a minimal example, you could do:
    else:
        # e.g., shape = [n_ensembles, batch_size, action_size * n_quantiles]
        # or shape [n_ensembles, batch_size, action_size, n_quantiles].
        #
        # Just flatten and compute std across ensemble dimension.
        # The example below flattens the trailing dimensions into one.
        # Adjust as needed for your distributional approach.
        flattened = values.view(values.shape[0], values.shape[1], -1)
        values_mean = flattened.mean(dim=2)  # [n_ensembles, batch_size]
        values_std = values_mean.std(dim=0)  # [batch_size]
        return values_std.unsqueeze(1)


# model builder
###################################################################


class SelfAttention(nn.Module):
    """Self-Attention Layer for reinforcement learning state encoding."""

    def __init__(self, input_dim: int, num_heads: int = 2):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        """
        Forward pass for self-attention.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            torch.Tensor: Attention-weighted output.
        """
        attn_output, _ = self.attention(x, x, x)  # Self-attention (Query=Key=Value)
        return attn_output


def create_discrete_q_function(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    device: str,
    enable_ddp: bool,
    # (New) Add attention layer to encoder
    use_attention: bool = False,  # New flag for enabling attention
    num_attention_heads: int = 2,  # Number of attention heads
    n_ensembles: int = 1,
) -> tuple[nn.ModuleList, DiscreteEnsembleQFunctionForwarder]:
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create(observation_shape)
        hidden_size = compute_output_size([observation_shape], encoder)
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

        # (New) Add attention layer to encoder
        if use_attention:
            attention_layer = SelfAttention(hidden_size, num_attention_heads)
            encoder = nn.Sequential(encoder, attention_layer)

    q_funcs = []
    forwarders = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create(observation_shape)
            hidden_size = compute_output_size([observation_shape], encoder)

            # (New) Add attention layer to encoder
            if use_attention:
                attention_layer = SelfAttention(hidden_size, num_attention_heads)
                encoder = nn.Sequential(encoder, attention_layer)

        q_func, forwarder = q_func_factory.create_discrete(
            encoder, hidden_size, action_size
        )
        q_func.to(device)
        if enable_ddp:
            q_func = wrap_model_by_ddp(q_func)
            forwarder.set_q_func(q_func)

        q_funcs.append(q_func)
        forwarders.append(forwarder)

    q_func_modules = nn.ModuleList(q_funcs)
    ensemble_forwarder = DiscreteEnsembleQFunctionForwarder(forwarders, action_size)
    return q_func_modules, ensemble_forwarder


def create_categorical_policy(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    device: str,
    enable_ddp: bool,
) -> CategoricalPolicy:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)
    policy = CategoricalPolicy(
        encoder=encoder, hidden_size=hidden_size, action_size=action_size
    )
    policy.to(device)
    if enable_ddp:
        policy = wrap_model_by_ddp(policy)
    return policy


def create_parameter(
    shape: Sequence[int], initial_value: float, device: str, enable_ddp: bool
) -> Parameter:
    data = torch.full(shape, initial_value, dtype=torch.float32)
    parameter = Parameter(data)
    parameter.to(device)
    if enable_ddp:
        parameter = wrap_model_by_ddp(parameter)
    return parameter


# SAC
###################################################################
@dataclasses.dataclass(frozen=True)
class TorchMiniBatchExtended(TorchMiniBatch):
    """Extended Minibatch to include both levels of actions."""

    def __init__(
        self,
        batch: TransitionMiniBatch,
        gamma: float,
        compute_returns_to_go: bool,
        device: str,
        observation_scaler: Optional[ObservationScaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ):
        super().__init__(
            batch,
            gamma,
            compute_returns_to_go,
            device,
            observation_scaler,
            action_scaler,
            reward_scaler,
        )
        # Extract action_level_1 and action_level_2 from the dataset
        # Extract action_level_2 (full 8-sized action)
        self.action_level_2 = torch.tensor(
            batch.actions, dtype=torch.float32, device=device
        )  # (batch_size, 8)

        # Determine action_level_1 (treatment decision)
        self.action_level_1 = (
            self.action_level_2.sum(dim=1) > 0
        ).long()  # 1 if any action > 0, else 0

    def filter_batch(self, selected_indices: torch.Tensor) -> "TorchMiniBatchExtended":
        """Filters the batch based on selected indices and returns a new filtered batch."""
        return TorchMiniBatchExtended(
            observations=self.observations[selected_indices],
            actions=self.actions[selected_indices],
            rewards=self.rewards[selected_indices],
            next_observations=self.next_observations[selected_indices],
            terminals=self.terminals[selected_indices],
            intervals=self.intervals[selected_indices],
        )


@dataclasses.dataclass(frozen=True)
class DiscreteSACModules(Modules):
    """SAC modules for policy, Q-functions, and optimizers."""

    policy: CategoricalPolicy
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    log_temp: Optional[Parameter]
    actor_optim: OptimizerWrapper
    critic_optim: OptimizerWrapper
    temp_optim: Optional[OptimizerWrapper]


class DiscreteSACImpl(DiscreteQFunctionMixin, QLearningAlgoImplBase):
    _modules: DiscreteSACModules
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _target_update_interval: int
    _compute_critic_grad: Callable[[TorchMiniBatch], dict[str, torch.Tensor]]
    _compute_actor_grad: Callable[[TorchMiniBatch], dict[str, torch.Tensor]]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteSACModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        target_update_interval: int,
        gamma: float,
        compiled: bool,
        device: str,
        # adding Sac 1
        sac1: Optional["DiscreteSACImpl"],  # SAC1 instance  (NEW)
        use_uncertainty_critic_loss: bool = False,  # (NEW)
        uncertainty_critic_weight: float = 1.0,  # (NEW)
        use_uncertainty_actor_loss: bool = False,  # (NEW)
        uncertainty_actor_weight: float = 1.0,  # (NEW)
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._gamma = gamma
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._target_update_interval = target_update_interval
        self.sac1 = sac1  # Store SAC1 instance  (NEW)
        self.use_uncertainty_critic_loss = use_uncertainty_critic_loss  # (NEW)
        self.uncertainty_critic_weight = uncertainty_critic_weight  # (NEW)
        self.use_uncertainty_actor_loss = use_uncertainty_actor_loss  # (NEW)
        self.uncertainty_actor_weight = uncertainty_actor_weight
        self._compute_critic_grad = (
            CudaGraphWrapper(self.compute_critic_grad)
            if compiled
            else self.compute_critic_grad
        )
        self._compute_actor_grad = (
            CudaGraphWrapper(self.compute_actor_grad)
            if compiled
            else self.compute_actor_grad
        )
        hard_sync(modules.targ_q_funcs, modules.q_funcs)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        """Computes target Q-values. If SAC1 is used, filters batch accordingly."""
        # print("compute_target")
        with torch.no_grad():
            modified_batch = batch  # Default to original batch
            if self.sac1 is not None and self.sac1._impl is not None:

                # Get SAC1's decision logits & actions
                # sac1_dist = self.sac1._modules.policy(batch.observations)
                sac1_dist = self.sac1._impl.policy(batch.observations)
                sac1_logits = sac1_dist.logits
                sac1_output = sac1_dist.sample()

                # Store SAC1 action as additional input
                # batch.sac1_output = sac1_output.unsqueeze(-1)
                treatment_ = sac1_logits[:, 1]
                treatment_prob = torch.exp(treatment_) / torch.exp(treatment_).sum()
                mean_probability = treatment_prob.mean()
                high_treatment_mask = (
                    treatment_prob >= mean_probability
                )  # High confidence in treatment
                # Compute mean of SAC1 logits
                # mean_logits = sac1_logits.mean(dim=1, keepdim=True)
                # selected_indices = (sac1_logits > mean_logits).squeeze()

                # Filter batch (SAC2 only sees data where SAC1 > mean decision)

                modified_batch = TorchMiniBatch(
                    observations=batch.observations[high_treatment_mask],
                    actions=batch.actions[high_treatment_mask],
                    rewards=batch.rewards[high_treatment_mask],
                    next_observations=batch.next_observations[high_treatment_mask],
                    terminals=batch.terminals[high_treatment_mask],
                    intervals=batch.intervals[high_treatment_mask],
                    next_actions=batch.next_actions[high_treatment_mask],
                    returns_to_go=batch.returns_to_go[high_treatment_mask],
                    device=batch.device,
                )
                batch = modified_batch  # Use modified batch for SAC2
                # print("batch is modified")

            # Continue standard SAC2 computation
            dist = self._modules.policy(batch.observations)

            log_probs = dist.logits
            probs = dist.probs
            entropy = get_parameter(self._modules.log_temp).exp() * log_probs

            target = self._targ_q_func_forwarder.compute_target(batch.observations)

            return (probs * (target - entropy)).sum(dim=1, keepdim=True), modified_batch

    def inner_update(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
        metrics = {}
        # print ("inner_update")
        # metrics.update(self.update_critic(batch))
        # metrics.update(self.update_actor(batch))
        # if grad_step % self._target_update_interval == 0:
        #     self.update_target()
        # return metrics
        # Compute and store critic loss for SAC2
        sac2_critic_loss = self.update_critic(batch)
        metrics.update({"sac2_critic_loss": sac2_critic_loss["critic_loss"]})

        # Compute and store actor loss for SAC2
        sac2_actor_loss = self.update_actor(batch)
        metrics.update({"sac2_actor_loss": sac2_actor_loss["actor_loss"]})

        # If SAC1 exists, compute its losses separately
        if self.sac1 is not None:
            sac1_critic_loss = self.sac1._impl.update_critic(batch)
            sac1_actor_loss = self.sac1._impl.update_actor(batch)

            metrics.update(
                {
                    "sac1_critic_loss": sac1_critic_loss["critic_loss"],
                    "sac1_actor_loss": sac1_actor_loss["actor_loss"],
                }
            )

        # Sync target networks at the specified interval
        if grad_step % self._target_update_interval == 0:
            self.update_target()

        return metrics

    def compute_critic_grad(self, batch: TorchMiniBatch) -> dict[str, torch.Tensor]:
        # print ("compute_critic_grad")
        self._modules.critic_optim.zero_grad()
        q_tpn, modified_batch = self.compute_target(batch)
        loss = self.compute_critic_loss(modified_batch, q_tpn)
        loss.backward()
        return {"loss": loss}

    def update_critic(self, batch: TorchMiniBatch) -> dict[str, float]:
        # print ("update_critic")
        loss = self._compute_critic_grad(batch)
        self._modules.critic_optim.step()
        return {"critic_loss": float(loss["loss"].cpu().detach().numpy())}

    def compute_critic_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        base_loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        # print ("compute_critic_loss")
        # 2) If the config says to use uncertainty loss, compute the penalty
        if self.use_uncertainty_critic_loss:
            # Example: compute Q-value std across ensemble heads
            penalty = self._q_func_forwarder.compute_uncertainty(batch.observations)
            penalty = self.uncertainty_critic_weight * penalty
            total_loss = base_loss + penalty
            # print ("compute_uncertainty")

            return total_loss

        return base_loss

        # return self._q_func_forwarder.compute_error(
        #     observations=batch.observations,
        #     actions=batch.actions.long(),
        #     rewards=batch.rewards,
        #     target=q_tpn,
        #     terminals=batch.terminals,
        #     gamma=self._gamma**batch.intervals,
        # )

    def compute_actor_grad(self, batch: TorchMiniBatch) -> dict[str, torch.Tensor]:
        # print ("compute_actor_grad")
        self._modules.actor_optim.zero_grad()
        loss = self.compute_actor_loss(batch)
        loss["loss"].backward()
        return loss

    def update_actor(self, batch: TorchMiniBatch) -> dict[str, float]:
        # print ("update_actor")
        # Q function should be inference mode for stability
        self._modules.q_funcs.eval()
        loss = self._compute_actor_grad(batch)
        self._modules.actor_optim.step()
        return {"actor_loss": float(loss["loss"].cpu().detach().numpy())}

    def compute_actor_loss(self, batch: TorchMiniBatch) -> dict[str, torch.Tensor]:
        # print ("compute_actor_loss")
        with torch.no_grad():
            q_t = self._q_func_forwarder.compute_expected_q(
                batch.observations, reduction="min"
            )
        dist = self._modules.policy(batch.observations)

        loss = {}
        if self._modules.temp_optim:
            loss.update(self.update_temp(dist))

        log_probs = dist.logits
        probs = dist.probs
        if self._modules.log_temp is None:
            temp = torch.zeros_like(log_probs)
        else:
            temp = get_parameter(self._modules.log_temp).exp()
        entropy = temp * log_probs
        loss["loss"] = (probs * (entropy - q_t)).sum(dim=1).mean()

        if self.use_uncertainty_actor_loss:
            # Example: compute Q-value std across ensemble heads
            penalty = self._q_func_forwarder.compute_uncertainty(batch.observations)
            penalty = self.uncertainty_actor_weight * penalty
            loss["loss"] += penalty
            # print ("compute_uncertainty")

        return loss

    def update_temp(self, dist: Categorical) -> dict[str, torch.Tensor]:
        assert self._modules.temp_optim
        assert self._modules.log_temp is not None
        self._modules.temp_optim.zero_grad()
        # print ("update_temp")

        with torch.no_grad():
            log_probs = F.log_softmax(dist.logits, dim=1)
            probs = dist.probs
            expct_log_probs = (probs * log_probs).sum(dim=1, keepdim=True)
            entropy_target = 0.98 * (-math.log(1 / self.action_size))
            targ_temp = expct_log_probs + entropy_target

        loss = -(get_parameter(self._modules.log_temp).exp() * targ_temp).mean()

        loss.backward()
        self._modules.temp_optim.step()

        # current temperature value
        log_temp = get_parameter(self._modules.log_temp)

        return {"temp_loss": loss, "temp": log_temp.exp()[0][0]}

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        dist = self._modules.policy(x)
        # print ("inner_predict_best_action")
        return dist.probs.argmax(dim=1)

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = self._modules.policy(x)
        # print ("inner_sample_action")
        return dist.sample()

    def update_target(self) -> None:
        # print ("update_target")
        hard_sync(self._modules.targ_q_funcs, self._modules.q_funcs)

    @property
    def policy(self) -> Policy:
        return self._modules.policy

    @property
    def policy_optim(self) -> Optimizer:
        return self._modules.actor_optim.optim

    @property
    def q_function(self) -> nn.ModuleList:
        return self._modules.q_funcs

    @property
    def q_function_optim(self) -> Optimizer:
        return self._modules.critic_optim.optim


@dataclasses.dataclass(frozen=True)
class DQNLoss:
    loss: torch.Tensor


@dataclasses.dataclass(frozen=True)
class DiscreteCQLLoss(DQNLoss):
    td_loss: torch.Tensor
    conservative_loss: torch.Tensor


@dataclasses.dataclass()
class DiscreteSACConfig_H(LearnableConfig):
    r"""Config of Soft Actor-Critic algorithm for discrete action-space.

    This discrete version of SAC is built based on continuous version of SAC
    with additional modifications.

    The target state-value is calculated as expectation of all action-values.

    .. math::

        V(s_t) = \pi_\phi (s_t)^T [Q_\theta(s_t) - \alpha \log (\pi_\phi (s_t))]

    Similarly, the objective function for the temperature parameter is as
    follows.

    .. math::

        J(\alpha) = \pi_\phi (s_t)^T [-\alpha (\log(\pi_\phi (s_t)) + H)]

    Finally, the objective function for the policy function is as follows.

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D}
            [\pi_\phi(s_t)^T [\alpha \log(\pi_\phi(s_t)) - Q_\theta(s_t)]]

    References:
        * `Christodoulou, Soft Actor-Critic for Discrete Action Settings.
          <https://arxiv.org/abs/1910.07207>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        temp_learning_rate (float): Learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        temp_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        n_critics (int): Number of Q functions for ensemble.
        initial_temperature (float): Initial temperature value.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
        sac1_config (Optional[DiscreteSACConfig_H]): Config for SAC1 (if hierarchical).
        use_sac1_filtering (bool): Whether SAC1 output should filter data for SAC2.
        sac1_action_size (int): Action size for SAC1.
        sac2_action_size (int): Action size for SAC2.

    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    temp_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 64
    gamma: float = 0.99
    n_critics: int = 2
    initial_temperature: float = 1.0
    target_update_interval: int = 8000

    # Hierarchical RL (NEW)
    sac1_config: Optional["DiscreteSACConfig_H"] = None
    use_sac1_filtering: bool = False
    sac1_action_size: int = 2  # SAC1 action size
    sac2_action_size: int = 8  # SAC2 action size

    # Custom encoder networks (Critic and Actor)
    critic_encoder_factory: Optional[Callable[[int, int], nn.Module]] = None
    actor_encoder_factory: Optional[Callable[[int, int], nn.Module]] = None
    use_attention: bool = False  # Enable attention mechanism
    num_attention_heads: int = 4  # Number of attention heads
    use_uncertainty_critic_loss: bool = False  # Use uncertainty penalty in critic loss
    uncertainty_critic_weight: float = 0.001  # Uncertainty penalty weight
    use_uncertainty_actor_loss: bool = False  # Use uncertainty penalty in actor loss
    uncertainty_actor_weight: float = 0.001

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DiscreteSAC":
        """Creates a Discrete SAC model. If SAC1 is enabled, initializes it."""
        sac1 = (
            self.sac1_config.create(
                device,
                enable_ddp,
            )
            if self.sac1_config
            else None
        )
        return DiscreteSAC(
            self,
            device,
            enable_ddp,
            sac1=sac1,
            use_sac1_filtering=self.use_sac1_filtering,
            num_attention_heads=self.num_attention_heads,
            use_attention=self.use_attention,
            use_uncertainty_critic_loss=self.use_uncertainty_critic_loss,
            uncertainty_critic_weight=self.uncertainty_critic_weight,
            use_uncertainty_actor_loss=self.use_uncertainty_actor_loss,
            uncertainty_actor_weight=self.uncertainty_actor_weight,
        )

    @staticmethod
    def get_type() -> str:
        return (
            "hierarchical_discrete_sac"
            if DiscreteSACConfig_H.sac1_config
            else "discrete_sac"
        )


class DiscreteSAC(QLearningAlgoBase[DiscreteSACImpl, DiscreteSACConfig_H]):
    def __init__(
        self,
        config: DiscreteSACConfig_H,
        device: DeviceArg = False,
        enable_ddp: bool = False,
        sac1: Optional["DiscreteSACImpl"] = None,  # SAC1 instance  (NEW)
        use_sac1_filtering: bool = False,  # SAC1 filtering flag  (NEW)
        use_attention: bool = False,  # Enable attention mechanism
        num_attention_heads: int = 4,  # Number of attention heads
        use_uncertainty_critic_loss: bool = False,  # Use uncertainty penalty in critic loss
        uncertainty_critic_weight: float = 0.001,  # Uncertainty penalty weight
        use_uncertainty_actor_loss: bool = False,  # Use uncertainty penalty in actor loss
        uncertainty_actor_weight: float = 0.001,
    ):
        super().__init__(config, device, enable_ddp)
        self.sac1 = sac1  # Store SAC1 instance  (NEW)
        self.use_sac1_filtering = use_sac1_filtering  # Store SAC1 filtering flag  (NEW)
        self.use_attention = use_attention  # Store attention mechanism flag
        self.num_attention_heads = (
            num_attention_heads  # Store number of attention heads
        )
        self.use_uncertainty_critic_loss = (
            use_uncertainty_critic_loss  # Store uncertainty flag
        )
        self.uncertainty_critic_weight = (
            uncertainty_critic_weight  # Store uncertainty penalty weight
        )
        self.use_uncertainty_actor_loss = (
            use_uncertainty_actor_loss  # Store uncertainty flag
        )
        self.uncertainty_actor_weight = (
            uncertainty_actor_weight  # Store uncertainty penalty weight
        )

    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        """
        If SAC1 exists, its output is added to SAC2's input space.
        """
        #  Build SAC1 if it exists but is not initialized
        if self.sac1 and self.sac1._impl is None:
            print("SAC1 is being initialized...")
            self.sac1.create_impl(observation_shape, self._config.sac1_action_size)
            assert (
                self.sac1._impl is not None
            ), "ERROR: SAC1 _impl is None after creation!"

        # Use the provided Q-Network (Critic) or default
        critic_encoder = self._config.critic_encoder_factory
        # Use the provided Policy Network (Actor) or default

        actor_encoder = self._config.actor_encoder_factory

        # Create SAC2 Q-function, policy, and temperature parameter
        q_funcs, q_func_forwarder = create_discrete_q_function(
            observation_shape,
            self._config.sac2_action_size,
            critic_encoder,
            # self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
            use_attention=self.use_attention,  # Enable attention mechanism
            num_attention_heads=self.num_attention_heads,  # Use 4 attention heads
        )
        targ_q_funcs, targ_q_func_forwarder = create_discrete_q_function(
            observation_shape,
            self._config.sac2_action_size,
            critic_encoder,
            # self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
            use_attention=self.use_attention,  # Enable attention mechanism
            num_attention_heads=self.num_attention_heads,  # Use 4 attention heads
        )
        policy = create_categorical_policy(
            observation_shape,
            self._config.sac2_action_size,
            actor_encoder,
            # self._config.actor_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        if self._config.initial_temperature > 0:
            log_temp = create_parameter(
                (1, 1),
                math.log(self._config.initial_temperature),
                device=self._device,
                enable_ddp=self._enable_ddp,
            )
        else:
            log_temp = None

        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(),
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )
        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )
        if self._config.temp_learning_rate > 0:
            assert log_temp is not None
            temp_optim = self._config.temp_optim_factory.create(
                log_temp.named_modules(),
                lr=self._config.temp_learning_rate,
                compiled=self.compiled,
            )
        else:
            temp_optim = None

        # Store SAC Modeuls
        modules = DiscreteSACModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
        )

        # Create SAC Implementation (SAC2)
        self._impl = DiscreteSACImpl(
            observation_shape=observation_shape,
            action_size=self._config.sac2_action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            target_update_interval=self._config.target_update_interval,
            gamma=self._config.gamma,
            compiled=self.compiled,
            device=self._device,
            sac1=self.sac1,  # Pass SAC1 instance  (NEW)
            use_uncertainty_actor_loss=self.use_uncertainty_actor_loss,  # Use uncertainty penalty in actor loss
            uncertainty_actor_weight=self.uncertainty_actor_weight,  # Uncertainty penalty weight
            use_uncertainty_critic_loss=self.use_uncertainty_critic_loss,  # Use uncertainty penalty in critic loss
            uncertainty_critic_weight=self.uncertainty_critic_weight,  # Uncertainty penalty weight
        )

    def update(self, batch):
        return super().update(batch)

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE
