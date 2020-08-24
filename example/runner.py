import warnings

import numpy
import torch

from backobs.integration import extend
from backpack import backpack
from deepobs.pytorch.runners.runner import PTRunner


class BackpackRunner(PTRunner):
    """Runner that supports BackPACK."""

    def __init__(self, optimizer_class, hyperparameter_names, extensions_fn):
        """
        Args:
            extensions_fn (callable): Function that returns a list of BackPACK
                extensions that should computed during the backward pass.
        """
        super().__init__(optimizer_class, hyperparameter_names)
        self._extensions_fn = extensions_fn

    def training(  # noqa: C901
        self,
        tproblem,
        hyperparams,
        num_epochs,
        print_train_iter,
        train_log_interval,
        tb_log,
        tb_log_dir,
    ):
        # [backobs] Integrate BackPACK
        tproblem = extend(tproblem)

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn(
                    "Not possible to use tensorboard for pytorch. Reason: " + e.msg,
                    RuntimeWarning,
                )
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(
                epoch_count,
                num_epochs,
                tproblem,
                train_losses,
                valid_losses,
                test_losses,
                train_accuracies,
                valid_accuracies,
                test_accuracies,
            )

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            # Training #

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()
                    batch_loss, _ = tproblem.get_batch_loss_and_accuracy()

                    # [backobs] Use BackPACK in backward pass
                    extensions = self._extensions_fn()
                    with backpack(*extensions):
                        batch_loss.backward()

                    # [backobs] This block is for demonstration and can be removed
                    for ext in extensions:
                        print(f"[backobs] Extension: {ext.__class__.__name__}")
                        for num, param in enumerate(tproblem.net.parameters()):

                            quantity = getattr(param, ext.savefield)
                            if isinstance(quantity, list):
                                quantity_print = [tuple(q.shape) for q in quantity]
                            elif isinstance(quantity, torch.Tensor):
                                quantity_print = tuple(quantity.shape)

                            print(f"\tParameter {num}: Shape {quantity_print}")

                    # [backobs] This block is for demonstration and can be removed
                    if batch_count == 3:
                        warnings.warn(
                            "[backobs] This demo performs only 3 steps per epoch."
                        )
                        raise StopIteration

                    opt.step()

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print(
                                "Epoch {0:d}, step {1:d}: loss {2:g}".format(
                                    epoch_count, batch_count, batch_loss
                                )
                            )
                        if tb_log:
                            summary_writer.add_scalar(
                                "loss", batch_loss.item(), global_step
                            )

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not numpy.isfinite(batch_loss.item()):
                self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses,
                )
                break
            else:
                continue

        if tb_log:
            summary_writer.close()
        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "valid_accuracies": valid_accuracies,
            "test_accuracies": test_accuracies,
        }

        return output
