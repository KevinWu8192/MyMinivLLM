import sys
from pathlib import Path

from transformers import AutoTokenizer

# Add src to Python path when this file is run directly from the repository.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams


# Model fields mirror Qwen/Qwen3-32B's Hugging Face config.json.
config = {
    'max_num_sequences': 16,
    'max_num_batched_tokens': 4096,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 2,
    'model_name_or_path': 'Qwen/Qwen3-32B',
    'enforce_eager': False,
    'dtype': 'bfloat16',
    'vocab_size': 151936,
    'hidden_size': 5120,
    'num_heads': 64,
    'head_dim': 128,
    'num_kv_heads': 8,
    'intermediate_size': 25600,
    'num_layers': 64,
    'tie_word_embeddings': False,
    'base': 1000000,
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 40960,
    'ffn_bias': False,
    'max_model_length': 40960,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,
    'use_large_scale_attention': True,
}


def main():
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
    llm = LLM(config=config)
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=256,
        max_model_length=config['max_model_length'],
    )
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Introduce yourself."}],
            tokenize=False,
            add_generation_prompt=True,
        )
    ]
    outputs = llm.generate(prompts, sampling_params)
    print(outputs['text'][0])


if __name__ == "__main__":
    main()
