import atexit
import torch.distributed as dist
import torch
import time
import torch.multiprocessing as mp
import socket
import uuid

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils.loader import resolve_checkpoint_path
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    try:
        model_runner = ModelRunner(config, rank, event)
        model_runner.loop()
    finally:
        # ModelRunner.exit normally destroys this group. This fallback covers
        # constructor failures and a parent that exits during initialization.
        if dist.is_initialized():
            dist.destroy_process_group()


def resolve_checkpoint_once(config: dict) -> str | None:
    """Resolve a remote checkpoint before tensor-parallel workers are spawned."""
    model_name_or_path = config.get("model_name_or_path")
    if not model_name_or_path:
        return None
    if "checkpoint_path" not in config:
        config["checkpoint_path"] = resolve_checkpoint_path(model_name_or_path)
    return config["checkpoint_path"]


class LLMEngine:
    _WORKER_JOIN_TIMEOUT_SECONDS = 5

    def __init__(self, config: dict):
        self.config = dict(config)
        self.processes = []
        self.events = []
        self.model_runner = None
        world_size = self.config.get("world_size", 1)
        if world_size <= 0:
            raise ValueError("world_size must be greater than 0")
        if world_size > torch.cuda.device_count():
            raise ValueError(
                f"world_size ({world_size}) exceeds available CUDA devices "
                f"({torch.cuda.device_count()})"
            )
        max_position = self.config.get(
            "max_position", self.config.get("max_position_embeddings")
        )
        if max_position is not None and max_position < self.config["max_model_length"]:
            raise ValueError(
                f"Rotary embedding capacity ({max_position}) is smaller than "
                f"max_model_length ({self.config['max_model_length']})"
            )
        if "distributed_init_method" not in self.config:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            self.config["distributed_init_method"] = f"tcp://127.0.0.1:{port}"
        if world_size > 1:
            self.config.setdefault(
                "shared_memory_name", f"myvllm-{uuid.uuid4().hex}"
            )

        try:
            # Resolve or download the checkpoint exactly once in the parent.
            # Every TP rank receives the resulting local directory.
            resolve_checkpoint_once(self.config)

            ctx = mp.get_context("spawn")
            for i in range(1, world_size):
                event = ctx.Event()
                process = ctx.Process(
                    target=worker_process,
                    args=(self.config, i, event),
                )
                self.events.append(event)
                self.processes.append(process)
                process.start()
            # Start the engine only on the master thread with rank = 0.
            self.model_runner = ModelRunner(
                self.config, rank=0, event=self.events
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.get("model_name_or_path", "gpt2")
            )
            configured_eos = self.tokenizer.eos_token_id
            if configured_eos is None:
                configured_eos = self.config.get("eos")
            if configured_eos is None:
                raise ValueError("Tokenizer/config must provide an EOS token ID")
            self.config["eos"] = configured_eos

            # Scheduler initialization follows the distributed rendezvous.
            self.scheduler = Scheduler(
                max_num_sequences=self.config.get("max_num_sequences", 16),
                max_num_batched_tokens=self.config.get(
                    "max_num_batched_tokens", 1024
                ),
                max_cached_blocks=self.config.get("max_cached_blocks", 1024),
                block_size=self.config.get("block_size", 256),
                eos=configured_eos,
                max_model_length=self.config["max_model_length"],
            )
        except BaseException:
            self._abort_initialization()
            raise

        atexit.register(self.exit)

    @staticmethod
    def _destroy_process_group() -> None:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                # Cleanup must continue so orphaned workers are still killed.
                pass

    def _stop_workers(self, terminate_first: bool) -> None:
        processes = getattr(self, "processes", [])
        if terminate_first:
            for process in processes:
                if process.is_alive():
                    try:
                        process.terminate()
                    except BaseException:
                        pass

        for process in processes:
            try:
                process.join(timeout=self._WORKER_JOIN_TIMEOUT_SECONDS)
            except BaseException:
                pass

        # A worker can be stuck inside NCCL/CUDA and ignore SIGTERM. Escalate
        # only for workers that did not exit within the grace period.
        for process in processes:
            if process.is_alive():
                try:
                    process.kill()
                    process.join(timeout=self._WORKER_JOIN_TIMEOUT_SECONDS)
                except BaseException:
                    pass

    def _abort_initialization(self) -> None:
        model_runner = getattr(self, "model_runner", None)
        self.model_runner = None
        if model_runner is not None:
            try:
                model_runner.call("exit")
            except BaseException:
                pass
        self._stop_workers(terminate_first=True)
        self._destroy_process_group()

    def exit(self):
        model_runner = getattr(self, "model_runner", None)
        self.model_runner = None
        if model_runner is not None:
            try:
                model_runner.call("exit")
            except BaseException:
                pass
        self._stop_workers(terminate_first=False)
        self._destroy_process_group()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if not scheduled_sequences:
            return [], 0, is_prefill
        # run the model
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        # Move outputs to CPU and convert them to a list
        if outputs is not None:
            outputs = outputs.cpu().tolist()
        num_processed_tokens = (
            sum(len(seq) - seq.num_cached_tokens for seq in scheduled_sequences)
            if is_prefill else len(scheduled_sequences)
        )
        # postprocess the outputs
        self.scheduler.postprocess(scheduled_sequences, outputs)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]

        return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("Prompt must contain at least one token")
        self.scheduler.add_sequence(Sequence(token_ids=token_ids, block_size=self.config['block_size'],sampling_params=sampling_params))

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> dict[str, list]:
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {
            'text': [
                self.tokenizer.decode(tokens, skip_special_tokens=True)
                for tokens in generated_tokens
            ],
            'token_ids': generated_tokens,
        }
        return output
