import os
import sys

from areal.api import AllocationMode
from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config, vLLMConfig
from areal.dataset import get_custom_dataset
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler
from areal.utils import logging, seeding
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.printing import tabulate_stats
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("CodeEval")


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    logging.setup_file_logging(f"{config.cluster.fileroot}/eval.log")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    seeding.set_random_seed(config.seed, key="eval")

    allocation_mode = AllocationMode.from_str(config.allocation_mode)

    # Initialize scheduler
    cfg = config.scheduler
    if cfg.type == "local":
        scheduler = LocalScheduler(exp_config=config)
    elif cfg.type == "ray":
        scheduler = RayScheduler(exp_config=config)
    elif cfg.type == "slurm":
        scheduler = SlurmScheduler(exp_config=config)
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.type}")

    # Load evaluation dataset
    valid_dataset = get_custom_dataset(
        split=config.valid_dataset.split, dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.valid_dataset,
    )
    assert len(valid_dataloader) > 0

    # Initialize RolloutController
    config.rollout.max_head_offpolicyness = int(1e12)

    if allocation_mode.gen_backend == "sglang":
        engine_cls = RemoteSGLangEngine
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=allocation_mode.gen.tp_size,
            base_gpu_id=0,
        )
    elif allocation_mode.gen_backend == "vllm":
        engine_cls = RemotevLLMEngine
        server_args = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=allocation_mode.gen.tp_size,
            pp_size=allocation_mode.gen.pp_size,
        )
    else:
        raise ValueError(f"Invalid backend: {allocation_mode.gen_backend}")

    eval_rollout = engine_cls.as_controller(config.rollout, scheduler)

    try:
        eval_rollout.initialize(
            role="eval-rollout",
            alloc_mode=allocation_mode,
            server_args=server_args,
        )

        # Create evaluation workflow
        workflow = "areal.workflow.rlvr.RLVRWorkflow"
        workflow_kwargs = dict(
            reward_fn="areal.reward.opencode.opencode_reward_fn",
            gconfig=config.gconfig,
            tokenizer=config.tokenizer_path,
            enable_thinking=False,
        )

        rollout_dir = os.path.join(
            StatsLogger.get_log_path(
                experiment_name=config.experiment_name,
                trial_name=config.trial_name,
                fileroot=config.cluster.fileroot,
            ),
            "rollout",
            "0"
        )

        # Submit all evaluation tasks
        cnt = 0
        run_cnt = 0
        for data in valid_dataloader:
            for item in data:
                if os.path.isfile(f"{rollout_dir}/{cnt}.jsonl"):
                    logger.info(f"Skipping {cnt}")
                else:
                    eval_rollout.submit(
                        item,
                        workflow=workflow,
                        workflow_kwargs=workflow_kwargs,
                        group_size=config.gconfig.n_samples,
                        task_id=cnt,
                    )
                    run_cnt += 1
                cnt += 1
        print(len(valid_dataset))
        print(len(valid_dataloader))
        print(cnt, run_cnt)
        eval_rollout.wait(run_cnt, timeout=None)
        eval_stats = eval_rollout.export_stats()

        # Print and log results
        logger.info(f"Evaluation Results: {tabulate_stats(eval_stats)}")
    finally:
        eval_rollout.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
