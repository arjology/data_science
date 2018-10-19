def optmap(func, *args):
    """Map values using the provided function or functor if non-None, otherwise leave it None."""
    values = [None if arg is None else func(arg) for arg in args]
    return values[0] if len(args) == 1 else values
