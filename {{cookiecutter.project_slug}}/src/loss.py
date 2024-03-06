import torch


def _log_loss(output, target, min_ret):
    log_output = torch.log(min_ret + output)
    log_target = torch.log(min_ret + target)

    return log_output, log_target


def _dnu_weight(dnu):
    weight = torch.sqrt(dnu) / 10.0
    weight = torch.clamp(weight, min=0.1, max=1.0)
    return weight


def get_weight(dnu, config):
    base_weight = torch.ones_like(dnu, dtype=torch.float)
    if config.hyperparameters.dnu_weight:
        base_weight *= _dnu_weight(dnu)

    return base_weight


def msle_loss(output, target, parameters, config):
    dnu = parameters
    weight = get_weight(dnu, config)
    min_ret = 1 / dnu

    output, target = _log_loss(output, target, min_ret)

    loss = (output - target) ** 2
    loss *= weight
    return loss.mean()
