from collections import defaultdict
import sys
from typing import Any, Callable, Dict, Optional, Set, Tuple
import warnings

__all__ = ['split_model_name_tag', 'get_arch_name', 'register_model', 'is_model', 'model_entrypoint']

_model_entrypoints: Dict[str, Callable[..., Any]] = {}  # mapping of model names to architecture entrypoint fns
_module_to_models: Dict[str, Set[str]] = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module: Dict[str, str] = {}  # mapping of model names to module names


def split_model_name_tag(model_name: str, no_tag: str = '') -> Tuple[str, str]:
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def get_arch_name(model_name: str) -> str:
    return split_model_name_tag(model_name)[0]


def register_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]  # type: ignore

    # add entries to registry dict/sets
    if model_name in _model_entrypoints:
        warnings.warn(
            f'Overwriting {model_name} in registry with {fn.__module__}.{model_name}. This is because the name being '
            'registered conflicts with an existing name. Please check if this is not expected.',
            stacklevel=2,
        )
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)

    return fn


def is_model(model_name: str) -> bool:
    """ Check if a model name exists
    """
    arch_name = get_arch_name(model_name)
    return arch_name in _model_entrypoints


def model_entrypoint(model_name: str, module_filter: Optional[str] = None) -> Callable[..., Any]:
    """Fetch a model entrypoint for specified model name
    """
    arch_name = get_arch_name(model_name)
    if module_filter and arch_name not in _module_to_models.get(module_filter, {}):
        raise RuntimeError(f'Model ({model_name} not found in module {module_filter}.')
    return _model_entrypoints[arch_name]
