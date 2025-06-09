import dataclasses
import math
from typing import Optional

from d3rlpy.base import DeviceArg, LearnableConfig, register_learnable
from d3rlpy.constants import ActionSpace
from d3rlpy.models.builders import (
    create_categorical_policy,
    create_continuous_q_function,
    create_discrete_q_function,
    create_normal_policy,
    create_parameter,
)
from d3rlpy.models.encoders import EncoderFactory, make_encoder_field
from d3rlpy.models.q_functions import QFunctionFactory, make_q_func_field
from d3rlpy.optimizers.optimizers import OptimizerFactory, make_optimizer_field
from d3rlpy.types import Shape
from d3rlpy.algos.qlearning.base import QLearningAlgoBase
from d3rlpy.algos.qlearning.torch.sac_impl import (
    DiscreteSACImpl,
    SACConfig,
    SACImpl,
    SACModules,
    DiscreteSACConfig,
    DiscreteSACModules,
    SACImpl,
    SACModules,
)

__all__ = ["SACConfig", "SAC", "DiscreteSACConfig", "DiscreteSAC"]






@dataclasses.dataclass()
class DiscreteSACConfig(LearnableConfig):


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

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DiscreteSAC":
        return DiscreteSAC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "discrete_sac"


class DiscreteSAC(QLearningAlgoBase[DiscreteSACImpl, DiscreteSACConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        q_funcs, q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        policy = create_categorical_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
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

        modules = DiscreteSACModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
        )

        self._impl = DiscreteSACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            target_update_interval=self._config.target_update_interval,
            gamma=self._config.gamma,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE



register_learnable(DiscreteSACConfig)