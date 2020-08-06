"""
Reduced and unreduced forward pass using only one forward throught the model.
"""

import types

import torch

from backobs.integration import integrate_backpack
from backpack import backpack, extensions
from deepobs.config import set_data_dir
from deepobs.pytorch.testproblems import (fmnist_2c2d, mnist_logreg,
                                          quadratic_deep)
from test_forward import forward_pass, set_deepobs_seed, set_up_problem


def has_no_accuracy(tproblem):
    """Return whether accuracy is defined for a DeepOBS testproblem task."""
    regression_tproblems = (quadratic_deep,)
    return isinstance(tproblem, regression_tproblems)


def hotfix_get_batch_loss_and_accuracy_func(
    self, reduction="mean", add_regularization_if_available=True
):

    inputs, labels = self._get_next_batch()
    inputs = inputs.to(self._device)
    labels = labels.to(self._device)

    def forward_func():
        """Attach individual losses in `._deepobs_unreduced_loss`."""
        savefield = "_deepobs_unreduced_loss"

        correct = 0.0
        total = 0.0

        # in evaluation phase is no gradient needed
        if self.phase in ["train_eval", "test", "valid"]:
            with torch.no_grad():
                outputs = self.net(inputs)
                loss = self.loss_function(reduction=reduction)(outputs, labels)
        else:
            outputs = self.net(inputs)
            loss = self.loss_function(reduction=reduction)(outputs, labels)

        # hotfix: compute unreduced loss
        with torch.no_grad():
            unreduced = self.loss_function(reduction="none")(outputs, labels)

        if has_no_accuracy(self):
            accuracy = 0
        else:
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = correct / total

        if add_regularization_if_available:
            regularizer_loss = self.get_regularization_loss()
        else:
            regularizer_loss = torch.tensor(0.0, device=torch.device(self._device))

        result = loss + regularizer_loss

        # hotfix: append unreduced loss
        setattr(result, savefield, unreduced)

        return result, accuracy

    return forward_func


def integrate_individual_loss(tproblem):
    """Modify loss evaluation such that individual losses are attached.

    Details on adding a function to an instance:
    https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance

    Note:
        If combined with BackPACK, the testproblem must have been extended before.
    """
    tproblem._old_get_batch_loss_and_accuracy_func = (
        tproblem.get_batch_loss_and_accuracy_func
    )
    tproblem.get_batch_loss_and_accuracy_func = types.MethodType(
        hotfix_get_batch_loss_and_accuracy_func, tproblem
    )
    return tproblem


if __name__ == "__main__":
    use_backpack = False
    set_data_dir("~/tmp/data_deepobs")

    batch_size = 20

    tp_classes = [
        mnist_logreg,
        fmnist_2c2d,
        quadratic_deep,
    ]

    for tp_cls in tp_classes:
        for use_backpack in [
            False,
            True,
        ]:

            losses = []
            accuracies = []

            for add_individual_loss in [
                False,
                True,
            ]:
                tp = set_up_problem(tp_cls, batch_size, seed=0, extend=use_backpack)

                if add_individual_loss:
                    tp = integrate_individual_loss(tp)

                loss, acc = tp.get_batch_loss_and_accuracy(
                    add_regularization_if_available=False
                )

                if add_individual_loss:
                    print(
                        "Individual loss shape: {}".format(
                            loss._deepobs_unreduced_loss.shape
                        )
                    )

                if use_backpack:
                    with backpack(extensions.BatchGrad()):
                        loss.backward()

                losses.append(loss.item())
                accuracies.append(acc)

            same_loss = losses[0] == losses[1]
            same_acc = accuracies[0] == accuracies[1]

            same = same_loss and same_acc
            same_symbol = "✓" if same else "❌"

            print(
                "{} [{}, BackPACK: {}] losses: {}, accuracies: {}".format(
                    same_symbol, tp_cls.__name__, use_backpack, losses, accuracies,
                )
            )
