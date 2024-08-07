from abc import ABC, abstractmethod
import json
import logging
import os
from pathlib import Path
from typing import Type, Union

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from transformers.generation.utils import GenerationConfig

try:
    from mistral_common.tokens.tokenizers.base import Tokenizer
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_inference.generate import generate, generate_mamba
    from mistral_inference.mamba import Mamba
    from mistral_inference.transformer import Transformer
except ImportError as e:
    logging.warning('Try to install mistral_inference!')
    print(e)

from ._registry import register_model

__all__ = [
    'tnl3', 'tnl3_tf32', 'tnl', 'llama3', 'llama', 'baichuan', 'baichuan2', 'qwen', 'qwen1_5', 'bloom', 'pythia',
    'mistral', 'mixtral', 'mamba', 'mpt', 'jamba', 'recurrentgemma', 'mistral_inf', 'LLModel', 'HuggingFaceModel'
]

# detect transformers version
logging.info(f'transformers version: {transformers.__version__}')


class LLModel(ABC):

    @abstractmethod
    def get_tokenizer(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def chat(self):
        pass


class HuggingFaceModel(LLModel):

    def __init__(self, tokenizer, model):
        model.eval()

        self.tok = tokenizer
        self.model = model

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tok

    def get_model(self) -> AutoModelForCausalLM:
        return self.model

    def chat(self, messages, config):
        inputs = self.tok(messages, return_tensors='pt').to('cuda')
        outputs = self.model.generate(inputs.input_ids, do_sample=True, **config)
        try:
            response = self.tok.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        except Exception as e:
            logging.error(e)
            response = ''

        return response


class HuggingFaceModelWrap(HuggingFaceModel):

    def __init__(self, repo_or_path, tok_configs, model_configs):

        if os.path.exists(f'{repo_or_path}/device_map.json'):
            dev_map = json.load(open(f'{repo_or_path}/device_map.json'))
        else:
            dev_map = 'auto'

        print('dev_map', dev_map)

        tok: AutoTokenizer = AutoTokenizer.from_pretrained(repo_or_path,
                                                           trust_remote_code=True,
                                                           add_bos_token=False,
                                                           padding_side='left',
                                                           **tok_configs)
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(repo_or_path,
                                                                           trust_remote_code=True,
                                                                           device_map=dev_map,
                                                                           **model_configs)

        model_vocab_size = model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tok)
        logging.info(f'Vocab of the base model: {model_vocab_size}')
        logging.info(f'Vocab of the tokenizer: {tokenzier_vocab_size}')

        super().__init__(tok, model)


class MistralModel(LLModel):

    def __init__(self, tokenizer, model):
        self.tok = tokenizer
        self.model = model

        self.generate_fn = generate if isinstance(model, Transformer) else generate_mamba

    def get_tokenizer(self):
        return self.tok

    def get_model(self):
        return self.model

    def chat(self, messages, config):
        tokens = self.tok.encode(messages, bos=True, eos=False)

        generated_tokens, _ = self.generate_fn(  # type: ignore[operator]
            [tokens],
            self.model,
            max_tokens=config['max_new_tokens'],
            temperature=config['temperature'],
            eos_id=self.tok.eos_id,
        )

        try:
            response = self.tok.decode(generated_tokens[0])
        except Exception as e:
            logging.error(e)
            response = ''

        return response


class MistralModelWrap(MistralModel):

    def __init__(self, path):
        mistral_tokenizer: MistralTokenizer = self.load_tokenizer(Path(path))
        tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

        model_cls = self.get_model_cls(path)
        model = model_cls.from_folder(Path(path), max_batch_size=1, num_pipeline_ranks=1)

        super().__init__(tokenizer, model)

    @staticmethod
    def get_model_cls(model_path: str) -> Union[Type[Mamba], Type[Transformer]]:
        with open(Path(model_path) / 'params.json', 'r') as f:
            args_dict = json.load(f)

        return {
            'mamba': Mamba,
            'transformer': Transformer
        }[args_dict.get('model_type', 'transformer')]  # type: ignore[return-value]

    @staticmethod
    def load_tokenizer(model_path: Path) -> MistralTokenizer:
        tokenizer = [f for f in os.listdir(Path(model_path)) if f.startswith('tokenizer.model')]
        assert (
            len(tokenizer) > 0
        ), f"No tokenizer found in {model_path}, make sure to place a `tokenizer.model.[v1,v2,v3]` file in {model_path}."
        assert (
            len(tokenizer) == 1
        ), f"Multiple tokenizers {', '.join(tokenizer)} found in `model_path`, make sure to only have one tokenizer"

        mistral_tokenizer = MistralTokenizer.from_file(str(model_path / tokenizer[0]))

        return mistral_tokenizer


@register_model
def mistral_inf(repo_or_path) -> MistralModel:
    return MistralModelWrap(path=repo_or_path)


@register_model
def tnl3(repo_or_path) -> HuggingFaceModel:
    try:
        import tiktoken
    except Exception as e:
        logging.error('Try to pip install tiktoken!')
        raise e

    tok_configs = {'use_fast': True}
    model_configs = {'torch_dtype': torch.bfloat16}

    return HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)


@register_model
def tnl3_tf32(repo_or_path) -> HuggingFaceModel:
    try:
        import tiktoken
    except Exception as e:
        logging.error('Try to pip install tiktoken!')
        raise e

    tok_configs = {'use_fast': True}
    model_configs = {'torch_dtype': torch.float32}

    return HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)


@register_model
def tnl(repo_or_path) -> HuggingFaceModel:
    tok_configs = {'use_fast': True}
    model_configs = {'torch_dtype': torch.bfloat16}

    model = HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)

    model.tok.pad_token_id = 0
    model.model.config.pad_token_id = 0

    return model


@register_model
def mistral(repo_or_path) -> HuggingFaceModel:
    tok_configs = {'use_fast': True}
    model_configs = {'torch_dtype': torch.bfloat16}

    return HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)


@register_model
def mixtral(repo_or_path) -> HuggingFaceModel:
    tok_configs = {'use_fast': True}
    model_configs = {'torch_dtype': torch.bfloat16}

    return HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)


@register_model
def llama3(repo_or_path) -> HuggingFaceModel:
    try:
        import tiktoken
    except Exception as e:
        logging.error('Try to pip install tiktoken!')
        raise e

    tok_configs = {}
    model_configs = {'torch_dtype': torch.bfloat16}

    return HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)


@register_model
def llama(repo_or_path) -> HuggingFaceModel:

    tok = LlamaTokenizer.from_pretrained(repo_or_path,
                                         padding_side='left',
                                         add_bos_token=False,
                                         use_fast=True,
                                         legacy=False)
    if 'llama2' or 'LLaMA2' or 'llama-2' or 'LLaMA-2' in repo_or_path:
        config = AutoConfig.from_pretrained(repo_or_path, pretraining_tp=0)
    else:
        config = AutoConfig.from_pretrained(repo_or_path)

    model = LlamaForCausalLM.from_pretrained(repo_or_path, config=config, torch_dtype=torch.bfloat16, device_map='auto')

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tok)
    logging.info(f'Vocab of the base model: {model_vocab_size}')
    logging.info(f'Vocab of the tokenizer: {tokenzier_vocab_size}')

    tok.pad_token = tok.eos_token
    model.config.end_token_id = tok.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    return HuggingFaceModel(tok, model)


@register_model
def baichuan(repo_or_path) -> HuggingFaceModel:
    model_configs = {'torch_dtype': torch.float16}
    model = HuggingFaceModelWrap(repo_or_path, {}, model_configs)

    try:
        model.model.generation_config = GenerationConfig.from_pretrained(repo_or_path)
    except Exception:
        model.tok.pad_token_id = 0
        model.model.config.pad_token_id = 0

    return model


@register_model
def baichuan2(repo_or_path) -> HuggingFaceModel:
    tok_configs = {'use_fast': False}
    model_configs = {'torch_dtype': torch.bfloat16}
    model = HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)

    try:
        model.model.generation_config = GenerationConfig.from_pretrained(repo_or_path)
    except Exception:
        model.tok.pad_token_id = 0
        model.model.config.pad_token_id = 0

    return model


@register_model
def qwen(repo_or_path) -> HuggingFaceModel:
    tok_configs = {}
    model_configs = {'bf16': True}

    model = HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)
    try:
        model.model.generation_config = GenerationConfig.from_pretrained(repo_or_path, trust_remote_code=True)
    except Exception as e:
        print(e)

    return HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)


@register_model
def qwen1_5(repo_or_path) -> HuggingFaceModel:
    tok_configs = {}
    model_configs = {'bf16': True}

    model = HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)
    try:
        model.model.generation_config = GenerationConfig.from_pretrained(repo_or_path, trust_remote_code=True)
    except Exception as e:
        print(e)

    return HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)


@register_model
def bloom(repo_or_path) -> HuggingFaceModel:
    tok = AutoTokenizer.from_pretrained(repo_or_path, trust_remote_code=True, add_bos_token=False, padding_side='left')

    model = BloomForCausalLM.from_pretrained(repo_or_path, device_map='auto')

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tok)
    logging.info(f'Vocab of the base model: {model_vocab_size}')
    logging.info(f'Vocab of the tokenizer: {tokenzier_vocab_size}')

    return HuggingFaceModel(tok, model)


@register_model
def pythia(repo_or_path) -> HuggingFaceModel:
    tok = AutoTokenizer.from_pretrained(repo_or_path, trust_remote_code=True, add_bos_token=False, padding_side='left')

    model = GPTNeoXForCausalLM.from_pretrained(repo_or_path, device_map='auto')

    tok.eos_token_id = 0
    model.config.eos_token_id = 0
    tok.pad_token_id = 1
    model.config.pad_token_id = 1

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tok)
    logging.info(f'Vocab of the base model: {model_vocab_size}')
    logging.info(f'Vocab of the tokenizer: {tokenzier_vocab_size}')

    return HuggingFaceModel(tok, model)


class MambaModel(LLModel):

    def __init__(self, repo_or_path):
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        except Exception as e:
            print('Please try to install mamba!')
            raise e

        self.device = 'cuda'

        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        model = MambaLMHeadModel.from_pretrained(repo_or_path, device=self.device, dtype=torch.float16)
        model.eval()

        self.tok = tokenizer
        self.model = model

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tok

    def get_model(self) -> AutoModelForCausalLM:
        return self.model

    def chat(self, messages, config={}):
        input_ids = self.tok(messages, return_tensors='pt').input_ids.to(device=self.device)
        max_length = input_ids.shape[1] + config['max_new_tokens']
        del config['max_new_tokens']

        outputs = self.model.generate(input_ids,
                                      max_length=max_length,
                                      cg=True,
                                      enable_timing=False,
                                      top_p=0.8,
                                      **config)
        try:
            response = self.tok.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        except Exception as e:
            logging.error(e)
            response = ''

        return response


@register_model
def mamba(repo_or_path) -> LLModel:
    return MambaModel(repo_or_path)


@register_model
def jamba(repo_or_path) -> HuggingFaceModel:
    try:
        from mamba_ssm import Mamba
    except Exception as e:
        logging.error('Try to pip install mamba_ssm!')
        raise e

    tok_configs = {'use_fast': True}
    model_configs = {'torch_dtype': torch.bfloat16}

    return HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)


@register_model
def recurrentgemma(repo_or_path) -> HuggingFaceModel:

    tok_configs = {}
    model_configs = {'torch_dtype': torch.bfloat16}

    return HuggingFaceModelWrap(repo_or_path, tok_configs, model_configs)


@register_model
def mpt(repo_or_path) -> HuggingFaceModel:
    # tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', add_bos_token=False, padding_side='left')
    tok = AutoTokenizer.from_pretrained('/cpfs01/user/shenxuyang/LLM/models-hf/gpt-neox-20b',
                                        trust_remote_code=True,
                                        add_bos_token=False,
                                        padding_side='left')

    config = AutoConfig.from_pretrained(repo_or_path, trust_remote_code=True)
    config.max_seq_len = 12000  # (input + output) tokens can now be up to 4096

    model = AutoModelForCausalLM.from_pretrained(repo_or_path, config=config, trust_remote_code=True, device_map='auto')

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tok)
    logging.info(f'Vocab of the base model: {model_vocab_size}')
    logging.info(f'Vocab of the tokenizer: {tokenzier_vocab_size}')

    return HuggingFaceModel(tok, model)
