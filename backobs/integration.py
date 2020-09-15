"""Integrate BackPACK into DeepOBS problems."""

import contextlib
import copy
import types

import torch

from backobs.utils import SUPPORTED, has_no_accuracy
from backpack import extend as backpack_extend
from deepobs.pytorch.testproblems.testproblem import TestProblem


def extend(tproblem: TestProblem, debug=False):
    """Add BackPACK functionality to a DeepOBS test problem.

    Note:
        Only a subset of the DeepOBS problems can be supported:
        - The computational graph structure for variational autoencoder problems differs
        from the assumptions in BackPACK.
        - For problems with batch normalization, the concept of many BackPACK
          quantities, such as individual gradients, is not defined.
        - Natural Language processing problems/RNNs are excluded, as they can
          not be handled with BackPACK (yet).
        - ℓ₂ regularization is not supported.

    Args:
        tproblem (TestProblem): DeepOBS testproblem, which has already been set up.
        debug (bool): Activate debugging mode of BackPACK's `extend` function.

    Raises:
        NotImplementedError: If BackPACK does not support the DeepOBS TestProblem
            or the test problem has ℓ₂ regularization.
        ValueError: If the testproblem has already been extended.

    Returns:
        TestProblem: extended testproblem.
    """
    if not isinstance(tproblem, SUPPORTED):
        raise NotImplementedError(f"Unsupported problem: {tproblem.__class__.__name__}")
    if not (tproblem._l2_reg is None or tproblem._l2_reg == 0.0):
        raise NotImplementedError(
            f"No support for ℓ₂ regularization (got l2_reg={tproblem._l2_reg})"
        )

    original_loss_function_savefield = "_old_loss_function"

    already_extended = hasattr(tproblem, original_loss_function_savefield)
    if already_extended:
        raise ValueError("Test problem is already extended")

    backpack_extend(tproblem.net, debug=debug)

    setattr(tproblem, original_loss_function_savefield, tproblem.loss_function)

    def new_loss_function(reduction="mean"):
        """Loss function of original DeepOBS problem, extended by BackPACK.

        Args:
            reduction (str): Reduction of individual losses, 'mean', 'sum' or 'none'.

        Returns:
            torch.nn.Module: Module used to compute the loss from the network
                prediction.
        """
        original_loss_function = copy.deepcopy(
            getattr(tproblem, original_loss_function_savefield)
        )
        return backpack_extend(original_loss_function(reduction=reduction), debug=debug)

    tproblem.loss_function = new_loss_function

    return tproblem


def extend_with_access_unreduced_loss(
    tproblem: TestProblem, savefield="_unreduced_loss", detach=False, debug=False
):
    """Same as `extend`, modifies loss computation to provide access to unreduced loss.

    Args:
        tproblem (TestProblem): DeepOBS testproblem, which has already been set up.
        savefield (str): Name of attribute through which individual loss can be
            accessed.
        detach (bool): Detach the unreduced loss.
        debug (bool): Activate debugging mode of BackPACK's `extend` function.

    Returns:
        TestProblem: Extended DeepOBS testproblem with overwritten forward pass,
            such that the unreduced loss can be accessed via the mini-batch loss.
    """
    tproblem = extend(tproblem, debug=debug)
    tproblem = _add_access_unreduced_loss(tproblem, savefield=savefield, detach=detach)

    return tproblem


def _add_access_unreduced_loss(
    tproblem: TestProblem, savefield="_unreduced_loss", detach=False
):
    """Provide access to unreduced losses when evaluating the reduced loss.

    Overwrites the `get_batch_loss_and_accuracy_func` of a testproblem. The function
    returned from the latter is a callable which evaluates to a tuple of the mini-batch
    loss and accuracy. The mini-batch loss will contain an attribute under `savefield`
    that contains the unreduced loss.

    Note:
        Must be called before extending the testproblem with BackPACK.

    Args:
        tproblem (TestProblem): DeepOBS testproblem, which has already been set up.
        savefield (str): Name of attribute through which individual loss can
            be accessed.
        detach (bool): Detach the unreduced loss.

    Details:
        - Adding a function to an instance: https://stackoverflow.com/a/8961717

    Returns:
        TestProblem: DeepOBS testproblem with overwritten forward pass, such that
            the unreduced loss can be accessed via the mini-batch loss.
    """
    original_get_batch_loss_and_accuracy_func_savefield = (
        "_old_get_batch_loss_and_accuracy_func"
    )

    setattr(
        tproblem,
        original_get_batch_loss_and_accuracy_func_savefield,
        tproblem.get_batch_loss_and_accuracy_func,
    )

    def new_get_batch_loss_and_accuracy_func(
        self, reduction="mean", add_regularization_if_available=True
    ):
        """Return callable to evaluate loss and accuracy on the current mini-batch.

        Args:
            self (deepobs.testproblems.TestProblem): Test problem of the callable.
            reduction (str): Reduction of individual losses, 'mean', 'sum' or 'none'.
            add_regularization_if_available (bool): Add regularization to the
                empirical risk.

        Returns:
            callable: Function to evaluate loss and accuracy. Loss has an attribute
                to access the unreduced loss.
        """
        inputs, labels = self._get_next_batch()
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)

        def forward_func():
            """Compute loss and accuracy. Provide access to unreduced loss.

            Returns:
                torch.Tensor: Mini-batch loss with an additional attribute that
                   contains the unreduced loss.
                float: Mini-batch accuracy. 0 for regression tasks.
            """
            # evaluation phases don't require gradients
            if self.phase in ["train_eval", "test", "valid"]:
                grad_ctx = torch.no_grad
            else:
                grad_ctx = contextlib.nullcontext

            with grad_ctx():
                outputs = self.net(inputs)
                unreduced_loss = self.loss_function(reduction="none")(outputs, labels)
                if detach:
                    unreduced_loss = unreduced_loss.detach()
                loss = self.loss_function(reduction=reduction)(outputs, labels)

            if has_no_accuracy(self):
                accuracy = 0.0
            else:
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()

                accuracy = correct / total

            if add_regularization_if_available:
                regularizer_loss = self.get_regularization_loss()
            else:
                regularizer_loss = torch.tensor(0.0, device=torch.device(self._device))

            result = loss + regularizer_loss

            # make unreduced loss accessible
            setattr(result, savefield, unreduced_loss)

            return result, accuracy

        return forward_func

    tproblem.get_batch_loss_and_accuracy_func = types.MethodType(
        new_get_batch_loss_and_accuracy_func, tproblem
    )

    return tproblem
