import sys

from areal import PPOTrainer, CurriculumPPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.dynamic_filter import filter_always_fail_pass


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split=config.train_dataset.split,
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )

    valid_dataset = None
    eval_workflow = None
    eval_workflow_kwargs = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            split=config.valid_dataset.split,
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )

        eval_workflow = "areal.workflow.rlvr.RLVRWorkflow"
        eval_workflow_kwargs = workflow_kwargs.copy()
        eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    if config.dynamic_hint is None:
        print("Using PPOTrainer.")
        with PPOTrainer(
            config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        ) as trainer:
            trainer.train(
                workflow="areal.workflow.rlvr.RLVRWorkflow",
                workflow_kwargs=workflow_kwargs,
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
                dynamic_filter_fn=filter_always_fail_pass,
            )
    else:
        print("Using CurriculumPPOTrainer with dynamic hint generation.")
        hint_percentage = dict()
        workflow_kwargs["hint_percentage"] = hint_percentage
        print(workflow_kwargs)

        with CurriculumPPOTrainer(
            config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        ) as trainer:
            trainer.train(
                workflow="areal.workflow.dynamic_hint_rlvr.DynamicHintRLVRWorkflow",
                workflow_kwargs=workflow_kwargs,
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
                dynamic_filter_fn=filter_always_fail_pass,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
