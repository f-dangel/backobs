.PHONY: help
.PHONY: test, test-test, test-example

.DEFAULT: help

help:
	@echo "test"
	@echo "        Run integration tests"

test:
	@make test-example
	@make test-test

test-example:
	@python example/supported.py
	@python example/extend.py
	@python example/extend_with_access_unreduced_loss.py

	@echo "BackPACK runner with SGD on mnist_logreg"
	@python example/run.py mnist_logreg

	@echo "BackPACK runner with SGD on fmnist_2c2d"
	@python example/run.py fmnist_2c2d

	@echo "BackPACK runner with SGD on cifar10_3c3d"
	@python example/run.py cifar10_3c3d --l2_reg 0.0

	@echo "BackPACK runner with SGD on cifar100_allcnnc"
	@python example/run.py cifar100_allcnnc --l2_reg 0.0

test-test:
	@pytest -vx --cov=backobs test
