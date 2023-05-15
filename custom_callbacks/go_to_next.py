from stable_baselines3.common.callbacks import BaseCallback

class GoToNextCallback(BaseCallback):

    def __init__(self, verbose=0, ):
        super().__init__(verbose)
        self.next = False

    # def _on_rollout_end(self) -> None:
    #     print("Rollout ended")
    #     self.training_env.env_method('mavlink_mission_set_current')

    def _on_step(self) -> bool:
        print(self.training_env.env_method('get_terminal'))

    

