from typing import List, Literal, Optional, Tuple, TypedDict
from cyberbattle._env import cyberbattle_env
from cyberbattle.agents.baseline.agent_wrapper import AgentWrapper
from cyberbattle.agents.baseline.learner import Learner
import openai

OpenAIRole = Literal["system", "user", "assistant"]
OpenAIModel = Literal["gpt-4", "gpt-3.5-turbo", "babbage-002", "davinci-002", "text-davinci-003", "text-davinci-002", "davinci", "curie", "babbage", "ada"]


class OpenAIMessage(TypedDict):
    """A message used by the completion API"""
    role: OpenAIRole
    content: str


class OpenAILearner(Learner):
    """The OpenAI Learner attempts to use OpenAI's models to conversationally choose actions in the environment"""

    _system_message = OpenAIMessage(
        role="system",
        content="""You are a system designed to navigate a network.
        Given the state of the network, your job is to return a valid command to run.
        There are three commands you can run:

        The first command is the run_attack(node_id, vulnerability_id) command.
        It takes in a target node to run a local attack and the type of attack to run.

        The second command is the run_remote_attack(node_id, target_node_id, vulnerability_id) command.
        It takes in the node id of the node to launch the attack from, the target node to run the attack against, and the type of attack.

        The final command is the connect_and_infect(node_id, target_node_id, port_name, credentials) command.
        It takes in the node id of a node connected to the target node, the node id of the target node, the port name of the target port on the target node, and the credential to use for the connection.

        Give your response by providing the command_type followed by its parameters in parentheses.
        The following are examples of valid responses:
        run_attack(1, 2, eternal blue)
        run_remote_attack(2, 1, log4j)
        connect_and_infect(5, 1, ssh, bob)"""
    )

    def __init__(self, api_key: str, model: OpenAIModel) -> None:
        openai.api_key = api_key
        self._model = model

    def explore(self, wrapped_env: AgentWrapper) -> Tuple[str, cyberbattle_env.Action, object]:
        raise NotImplementedError

    def exploit(self, wrapped_env: AgentWrapper, observation: cyberbattle_env.Observation) -> Tuple[str, Optional[cyberbattle_env.Action], object]:
        raise NotImplementedError

    def on_step(self, wrapped_env: AgentWrapper, observation, reward, done, info, action_metadata) -> None:
        return None

    def _prompt_environment_info(self, env_info: str) -> List[OpenAIMessage]:
        return [
            self._system_message,
            OpenAIMessage(role="user", content=env_info)
        ]

    def _get_action_choices(self, env_info: str) -> List[str]:
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=self._prompt_environment_info(env_info)
        )
        return list(map(lambda x: x["message"]["content"], response["choices"]))
