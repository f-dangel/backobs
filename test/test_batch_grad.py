"""Test individual gradients."""

import argparse

import torch

from backpack import backpack, extensions
from deepobs.config import set_data_dir
from test_forward import forward_pass, set_up_problem, tproblem_cls_from_str


def autograd_individual_gradients(X, y, forward_fn, parameters):
    """Compute individual gradients with a for-loop using autograd.

    Loop over data (xₙ, yₙ) and compute ∇ℓ(xₙ, yₙ) with respect to `parameters`,
    where ℓ is the forward function.

    Note:
        Individual gradients only make sense, if the summands in the loss
        depend on a single datum (xₙ, yₙ). 

    Args:
        X (torch.Tensor): `(N, *)` batch of input data.
        y (torch.Tensor): `(N, ∘)` batch of input labels.
        forward_func (callable): Function that computes the (individual) loss. Must have
            signature `loss = forward(X, y)` and return a scalar tensor `loss`.
        parameters (list): List of parameters, used `forward_fn` to compute the loss,
            that `require_grad` (and w.r.t. which gradients will be computed).

    Returns:
        list: Individual gradients for every parameter in `parameters`, arranged in the
            same order. Every item is of same shape as the associated parameter, with
            an additional leading dimension of size `N` (gradient for each sample).
    """
    N = X.shape[0]

    individual_gradients = [torch.zeros(N, *p.shape).to(X.device) for p in parameters]

    for n in range(N):
        x_n = X[n].unsqueeze(0)
        y_n = y[n].unsqueeze(0)

        l_n = forward_fn(x_n, y_n)
        g_n = torch.autograd.grad(l_n, parameters)

        for param_idx, g in enumerate(g_n):
            individual_gradients[param_idx][n] = g

    return individual_gradients


def backpack_individual_gradients(X, y, forward_fn, parameters):
    """Compute individual gradients with BackPACK.

    Note:
        The model used in the forward pass must already be extended with BackPACK.

    Args:
        X (torch.Tensor): `(N, *)` batch of input data.
        y (torch.Tensor): `(N, ∘)` batch of input labels.
        forward_fn (callable): Function that computes the (individual) loss. Must have
            signature `loss = forward(X, y)` and return a scalar tensor `loss`.
        parameters (list): List of parameters, used `forward_fn` to compute the loss,
            that `require_grad` (and w.r.t. which gradients will be computed).

    Returns:
        [torch.Tensor]: Individual gradients for samples in the mini-batch
            with respect to the model parameters. Arranged in the same order
           as `model.parameters()`.
    """
    loss = forward_fn(X, y)

    with backpack(extensions.BatchGrad()):
        loss.backward()

    individual_gradients = [p.grad_batch for p in parameters]

    return individual_gradients


def individual_gradients(tproblem, use_backpack):
    X, y = tproblem._get_next_batch()
    X = X.to(tproblem._device)
    y = y.to(tproblem._device)

    reduction = "mean"
    assert tproblem.phase == "train"

    def forward_fn(X, y):
        return tproblem.loss_function(reduction=reduction)(tproblem.net(X), y)

    individual_grad_fn = (
        backpack_individual_gradients if use_backpack else autograd_individual_gradients
    )

    return individual_grad_fn(
        X, y, forward_fn, [p for p in tproblem.net.parameters() if p.requires_grad]
    )


def scale(tensors, by):
    return [by * t for t in tensors]


def individual_gradients_correct(tproblem_cls, batch_size, seed=0, verbose=True):
    try:
        use_backpack = True
        tproblem = set_up_problem(
            tproblem_cls, batch_size, seed=seed, extend=use_backpack
        )
        backpack_batch_grad = individual_gradients(tproblem, use_backpack=use_backpack)
        # because "mean" reduction is used in al DeepOBS problems, we need to rescale
        backpack_batch_grad = scale(backpack_batch_grad, tproblem._batch_size)

        use_backpack = False
        tproblem = set_up_problem(
            tproblem_cls, batch_size, seed=seed, extend=use_backpack
        )
        autograd_batch_grad = individual_gradients(tproblem, use_backpack=use_backpack)

        same_param_wise = [
            torch.allclose(g1, g2, atol=1e-5, rtol=1e-5)
            for g1, g2 in zip(backpack_batch_grad, autograd_batch_grad)
        ]
        same = all(same_param_wise)

        if verbose:
            name = tproblem_cls.__name__
            same_symbol = "✓" if same else "❌"
            print(
                "{} [{}, individual gradients] Same? {}".format(
                    same_symbol, name, same_param_wise
                )
            )

        return same

    except Exception as e:
        if verbose:
            name = tproblem_cls.__name__
            fail = "❌"
            print(
                "{} [{}, individual gradients] Raised exception: {}".format(
                    fail, name, e
                )
            )
        return False

    pass


if __name__ == "__main__":
    set_data_dir("~/tmp/data_deepobs")

    parser = argparse.ArgumentParser(
        description="Compare individual gradients on DeepOBS problems"
    )

    parser.add_argument(
        "tproblem_cls", type=str, help="Name of the DeepOBS testproblem",
    )
    parser.add_argument(
        "--batch_size", type=int, default=3, help="Batch size",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output to command line",
    )

    kwargs = vars(parser.parse_args())
    kwargs["tproblem_cls"] = tproblem_cls_from_str(kwargs["tproblem_cls"])

    individual_gradients_correct(**kwargs)
