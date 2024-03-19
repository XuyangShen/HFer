from ._registry import is_model, model_entrypoint

__all__ = ['create_model']


def create_model(
    model_name: str,
    repo_or_path: str,
    **kwargs,
):
    """Create a model.

    Lookup model's entrypoint function and pass relevant args to create a new model.

    Args: 
        model_name: Name of model
        repo_or_path: Path/Url of model to instantiate.
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if not is_model(model_name):
        raise RuntimeError(f'Unknown model {model_name}')

    create_fn = model_entrypoint(model_name)
    model = create_fn(repo_or_path=repo_or_path, **kwargs)

    return model
