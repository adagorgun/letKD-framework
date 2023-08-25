from importlib import import_module


def factory(criterion_name, *args, **kwargs):
    criterion_module = import_module(
        '.criterions.' + criterion_name, package='distillation')
    create_criterion = getattr(criterion_module, 'create_criterion')
    return create_criterion(*args, **kwargs)
