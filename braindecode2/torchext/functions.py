def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """ Prevents log(0) by using max of eps and given x."""
    return torch.log(torch.clamp(x, min=eps))
