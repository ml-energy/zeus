# Contributing to Zeus

## Welcome!

We're excited that you're considering contributing to Zeus! Your contributions are vital to making Zeus getting better and better.

## Dual Purpose

Zeus serves two main goals:
1. **Research Artifact**: A platform for disseminating cutting-edge research in the field of machine learning energy (i.e., [ML.ENERGY](https://ml.energy)).
2. **Practical Tool**: We aim to make Zeus highly usable in real-world scenarios.

## Code of Conduct

Everyone taking part in Zeus is expected to follow our [Code of Conduct](/CODE_OF_CONDUCT.md), which is the Linux Foundation Projects Code of Conduct.

## Governance

Zeus is consensus-seeking: maintainers review and merge changes, and the project lead breaks ties when consensus is not reached.
[GOVERNANCE.md](/GOVERNANCE.md) describes the decision-making process in full, and [MAINTAINERS.md](/MAINTAINERS.md) lists the current maintainers.

If you would like to take on more responsibility in the project, [GOVERNANCE.md](/GOVERNANCE.md#becoming-a-maintainer) explains how contributors become maintainers.

## Ways to Contribute

- **Bug Reports and Fixes**: We use GitHub Issues for bug tracking.
- **New Features**: You can submit feature proposals via GitHub Issues.
- **Documentation**: Enhancing the README and documentation (both inline with code or under [`/docs`](/docs)) is also welcome.

## AI Usage

Contributors are welcome to use any kind of AI tools to assist with their contributions.

However, [*meat proxies*](https://gruhn.me/blog/2026-08-03/) are not welcome in this project.
A meat proxy is someone who simply acts as a direct (or virtually direct) relay between an AI and the project: one who pipes project communication into an AI tool and then pipes the AI's output back to the project while adding no or little value themselves.
While we have maintainers who review incoming contributions, open-source is still primarily trust-based: we trust that the human behind the contribution is capable of understanding relevant parts of the project and their own contributions, and of vetting them for quality and correctness before submitting them to the project.
Meat proxies fail to establish that trust.
You will be warned explicitly if you are suspected of being a meat proxy, and **continued behaviors can result in being blocked**.
This policy applies to all project interactions, including issues, PRs, reviews/comments, and discussions, and to everyone including the maintainers.


## Process

1. **Fork & Clone**: Fork the repository and clone it to your local machine.
1. **Create an Issue**: Discuss your proposed changes via a new GitHub Issue.
1. **Branch**: Create a new branch for your feature or fix.
1. **Dev dependencies**: `uv sync` will install everything you need into `.venv`.
1. **Code and Test**: Write code and make sure to add tests. `pytest` should successfully terminate including the new tests you wrote.
1. **Format and Lint**: Run `bash scripts/lint.sh` and make sure it runs without complaining.
1. **Check documentation**: Run `bash scripts/preview_docs.sh` to build and spin up a local instance of the documentation. In particular, check whether your docstrings are correctly rendered in the Source Code Reference section.
1. **Pull Request**: Open a PR to the main repository. Ensure all CI checks pass before requesting a review.

## Coding Standards

- We want to support Python 3.10 and later.
- The formatting and linting script [`scripts/lint.sh`](/scripts/lint.sh) should pass for PRs to be merged.
- Strictly type-annotate all code.
- Tests should accompany new features and be placed in the `tests/` directory. Tests should not require GPUs to run.
- Changes, whenever appropriate, should be accompanied by documentation changes in the `docs/` directory.

## License

By contributing to Zeus, you agree to license your contributions under our [Apache-2.0 License](/LICENSE).
