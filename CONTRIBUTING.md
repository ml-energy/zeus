# Contributing to Zeus

## Welcome!

We're excited that you're considering contributing to Zeus! Your contributions are vital to making Zeus getting better and better.

## Dual Purpose

Zeus serves two main goals:
1. **Research Artifact**: A platform for disseminating cutting-edge research in the field of machine learning energy (i.e., [ML.ENERGY](https://ml.energy)).
2. **Practical Tool**: We aim to make Zeus highly usable in real-world scenarios.

## Governance

Zeus is guided by the **Benevolent Dictator For Life (BDFL)** model, where the project founder, [Jae-Won Chung](https://github.com/jaywonchung), holds the primary decision-making authority. This model ensures consistent vision and leadership for the project.

### Community Input

- Community feedback is highly valued and can be provided through GitHub issues and discussions.
- The BDFL can also be reached over email.

### Decision-Making Process

- Efforts are to be made to reach consensus for all decisions, including but not limited to project direction, code acceptance, and feature inclusion.
- If unresolved, the BDFL has the final say.
- Routine decisions may be delegated to specific maintainers.

## Ways to Contribute

- **Bug Reports and Fixes**: We use GitHub Issues for bug tracking.
- **New Features**: You can submit feature proposals via GitHub Issues.
- **Documentation**: Enhancing the README and documentation (both inline with code or under [`/docs`](/docs)) is also welcome.

## Process

1. **Fork & Clone**: Fork the repository and clone it to your local machine.
1. **Create an Issue**: Discuss your proposed changes via a new GitHub Issue.
1. **Branch**: Create a new branch for your feature or fix.
1. **Dev dependencies**: `pip install -e '.[dev]'` will be helpful.
1. **Code and Test**: Write code and make sure to add tests. `pytest` should successfully terminate including the new tests you wrote.
1. **Format and Lint**: Run `bash scripts/lint.sh` and make sure it runs without complaining.
1. **Check documentation**: Run `bash scripts/preview_docs.sh` to build and spin up a local instance of the documentation. In particular, check whether your docstrings are correctly rendered in the Source Code Reference section.
1. **Pull Request**: Open a PR to the main repository. Ensure all CI checks pass before requesting a review.

## Coding Standards

- We want to support Python 3.9 and later.
- The formatting and linting script [`scripts/lint.sh`](/scripts/lint.sh) should pass for PRs to be merged.
- Strictly type-annotate all code.
- Tests should accompany new features and be placed in the `tests/` directory. Tests should not require GPUs to run.
- Changes, whenever appropriate, should be accompanied by documentation changes in the `docs/` directory.

## License

By contributing to Zeus, you agree to license your contributions under our [Apache-2.0 License](/LICENSE).
