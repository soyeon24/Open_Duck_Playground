"""Training entry point for the get-up task.

Kept separate from `runner.py` on purpose: that file is shared with the walking
runs and editing it risks colliding with a job already in flight. This adds the
standup env without touching it.

    uv run playground/open_duck_mini_v2/standup_runner.py \
        --output_dir checkpoints_standup \
        --num_timesteps 150000000

ONNX files are exported next to the checkpoints on every save, exactly as in the
walking runner, so intermediate policies can be pulled down and viewed while the
job is still running.
"""

import argparse

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import standup


class OpenDuckMiniV2StandupRunner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        self.env_config = standup.default_config()
        self.env = standup.Standup()
        self.eval_env = standup.Standup()
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(self.env.observation_size["state"][0])
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini V2 standup runner")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_standup",
        help="Where to save the checkpoints",
    )
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument(
        "--task",
        type=str,
        default="standup",
        help="Ignored; the standup scene is always used. Present so that the "
        "BaseRunner argument handling matches runner.py.",
    )
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    args = parser.parse_args()

    OpenDuckMiniV2StandupRunner(args).train()


if __name__ == "__main__":
    main()
