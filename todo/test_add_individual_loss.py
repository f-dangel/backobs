"""
Reduced and unreduced forward pass using only one forward throught the model.
"""

from test_forward import set_up_problem

from backobs.integration import integrate_individual_loss
from backpack import backpack, extensions
from deepobs.config import set_data_dir
from deepobs.pytorch.testproblems import fmnist_2c2d, mnist_logreg, quadratic_deep

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
