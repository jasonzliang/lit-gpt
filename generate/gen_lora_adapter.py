# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Block, Config, merge_lora_weights
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, gptq_quantization, lazy_load
from scripts.prepare_alpaca import generate_prompt

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from evalplus.data import write_jsonl

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False


def generate_code_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""
    # if example["input"]:
    #     return (
    #         "Below is an instruction that describes a task, paired with an input that provides further context. "
    #         "Write a response that appropriately completes the request.\n\n"
    #         f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
    #     )
    # return (
    #     "Below is a function definition and comment that describes a task. "
    #     "Implement and complete the function so that it can successfully complete the task.\n\n"
    #     f"### Instruction:\n{example['instruction']}\n\n### Response:"
    # )
    return example['instruction']


def filter_by_stopwords(decoded_string):
    stop_tokens = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


def postprocess_generation(generation, prompt):
    generation = generation[len(prompt):]
    return prompt + filter_by_stopwords(generation)


def generate_eval_results(
    lora_path: Path = Path("out/lora/alpaca_codellama7b/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/codellama/CodeLlama-7b-Python-hf"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    max_new_tokens: int = 512,
    top_k: Optional[int] = 200,
    temperature: float = 0.2,
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = None,
    humaneval: bool = True,
    use_lora: bool = False
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if devices > 1:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training."
                " Please set devices=1 when using the --quantize flag."
            )
        if quantize.startswith("bnb."):
            if "mixed" in precision:
                raise ValueError("Quantization and mixed precision is not supported.")
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16,
                "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(quantize[4:], dtype)
            precision = None

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)

    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy,
        plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    if use_lora:
        config = Config.from_json(
            checkpoint_dir / "lit_config.json",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            to_query=lora_query,
            to_key=lora_key,
            to_value=lora_value,
            to_projection=lora_projection,
            to_mlp=lora_mlp,
            to_head=lora_head,
        )
    else:
        config = Config.from_json(checkpoint_dir / "lit_config.json")

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    L.seed_everything(None)
    tokenizer = Tokenizer(checkpoint_dir)

    fabric.print(f"Loading model \
        {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(
        empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: \
        {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        # model.max_seq_length = max_new_tokens * 10
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    if use_lora:
        lora_checkpoint = lazy_load(lora_path)
        checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(f"Time to load the model weights: \
        {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    if use_lora:
        merge_lora_weights(model)
    model = fabric.setup(model)

    # samples = [
    #     dict(task_id=task_id, solution=main(prompt=problem["prompt"],temperature=temp))
    #     for task_id, problem in get_[human_eval|mbpp]_plus().items()
    # ]
    # write_jsonl("samples.jsonl", samples)
    if humaneval:
        problems = get_human_eval_plus()
    else:
        problems = get_mbpp_plus()

    fabric.print("Total prompts: %s" % len(problems.items()))
    results = []
    for task_id, problem in problems.items():
        sample = {"instruction": problem['prompt'],
            "input": problem['base_input']}
        # print(problem['prompt']); exit()
        prompt = generate_code_prompt(sample)
        fabric.print("\n\n#### Task ID: %s, Prompt:\n%s" % (task_id, prompt))

        encoded = tokenizer.encode(prompt, device=fabric.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens
        model.max_seq_length = max_returned_tokens

        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens,
            temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0

        # for block in model.transformer.h:
        #     block.attn.kv_cache.reset_parameters()

        output = tokenizer.decode(y)
        output = postprocess_generation(output, prompt).strip()
        # output = output.split("### Response:")[1].strip()
        fabric.print("#### LLM Output:\n%s" % output)

        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"\n\nTime for inference: {t:.02f} sec total, \
            {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)

        results.append({'task_id': task_id, 'solution': output})
        # break

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: \
            {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)

    eval_name = "humaneval" if humaneval else "mbpp"
    result_dir  = os.path.join(os.path.dirname(str(lora_path)),
        "results-%s_lora-%s_temp-%.2f" % (eval_name, use_lora, temperature))
    # write_jsonl(result_file, results)

    def write_to_dir(result_dir, results):
        for result_dict in results:
            task_id_dir = os.path.join(result_dir,
                result_dict['task_id'].replace("/", "_"))
            os.makedirs(task_id_dir, exist_ok=True)
            result_file = os.path.join(task_id_dir, "0.py")
            with open(result_file, 'w') as f:
                f.write(result_dict['solution'])
        os.system("evalplus.evaluate --dataset %s --samples %s"
            % (eval_name, result_dir))

    write_to_dir(result_dir, results)


def main(
    prompt: str = "What food do llamas eat?",
    input: str = "",
    lora_path: Path = Path("out/lora/alpaca/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    max_new_tokens: int = 100,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    precision: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(
        checkpoint_dir / "lit_config.json",
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    tokenizer = Tokenizer(checkpoint_dir)
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    lora_checkpoint = lazy_load(lora_path)
    checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    merge_lora_weights(model)
    model = fabric.setup(model)

    L.seed_everything(1234)
    t0 = time.perf_counter()
    y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    t = time.perf_counter() - t0

    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()
    fabric.print(output)

    tokens_generated = y.size(0) - prompt_length
    fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    # CLI(main)

    for temp in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        generate_eval_results(
            checkpoint_dir=Path("checkpoints/codellama/CodeLlama-7b-Python-hf"),
            lora_path=Path("out/lora/alpaca_codellama7b/lit_model_lora_finetuned.pth"),
            use_lora=True,
            temperature=temp)

    # generate_eval_results(
    #     checkpoint_dir=Path("checkpoints/stabilityai/stablelm-tuned-alpha-3b"),
    #     lora_path=Path("out/lora/stablelmtuned3b/lit_model_lora_finetuned.pth"),
    #     use_lora=False)

    # generate_eval_results(
    #     checkpoint_dir=Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    #     lora_path=Path("out/lora/alpaca_stablelmbase3b/lit_model_lora_finetuned.pth"),
    #     use_lora=False,
    #     humaneval=False)
    # generate_eval_results(
    #     checkpoint_dir=Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    #     lora_path=Path("out/lora/alpaca_stablelmbase3b/lit_model_lora_finetuned.pth"),
    #     use_lora=True,
    #     humaneval=False)

    # generate_eval_results(
    #     lora_path=Path("out/lora/alpaca_codellama7b/lit_model_lora_finetuned.pth"),
    #     checkpoint_dir=Path("checkpoints/codellama/CodeLlama-7b-Python-hf"),
    #     use_lora=False)
    # generate_eval_results(
    #     checkpoint_dir=Path("checkpoints/codellama/CodeLlama-7b-Python-hf"),
    #     lora_path=Path("out/lora/alpaca_codellama7b/lit_model_lora_finetuned.pth"),
    #     use_lora=True)

    # generate_eval_results(
    #     checkpoint_dir=Path("checkpoints/codellama/CodeLlama-7b-Python-hf"),
    #     lora_path=Path("out/lora/codealpaca_codellama7b/lit_model_lora_finetuned.pth"),
    #     use_lora=False)
    # generate_eval_results(
    #     checkpoint_dir=Path("checkpoints/codellama/CodeLlama-7b-Python-hf"),
    #     lora_path=Path("out/lora/codealpaca_codellama7b/lit_model_lora_finetuned.pth"),
    #     use_lora=True)
