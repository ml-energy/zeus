# Contributing to Zeus

## Welcome!

We're excited that you're considering contributing to Zeus! Your contributions are vital for keeping Zeus innovative, robust, and user-friendly.

## What to Know Before You Contribute

### Dual Purpose

Zeus serves two main goals:
1. **Research Artifact**: A platform for disseminating cutting-edge research in the field of machine learning energy (i.e., [ML.ENERGY](https://ml.energy)).
2. **Practical Tool**: We aim to make Zeus highly usable in real-world scenarios.

### Governance

Zeus follows a **BDFL (Benevolent Dictator For Life)** governance model. The project founder (Jae-Won Chung) retains decisive authority to ensure alignment with the goals of Zeus.

## Ways to Contribute

- **Bug Reports and Fixes**: We use GitHub Issues for bug tracking.
- **New Features**: You can submit feature proposals via GitHub Issues.
- **Documentation**: Enhancing the README and documentation (both inline with code or under [`/docs`](/docs)) is also welcome.

## Process

1. **Fork & Clone**: Fork the repository and clone it to your local machine.
2. **Create an Issue**: Discuss your proposed changes via a new GitHub Issue.
3. **Branch**: Create a new branch for your feature or fix.
4. **Code and Test**: Write code and make sure to add tests. `pytest` should successfully terminate including the new tests you wrote.
5. **Format and Lint**: Run `bash scripts/lint.sh` and make sure it runs without complaining.
6. **Pull Request**: Open a PR to the main repository. Ensure all CI checks pass before requesting a review.

## Coding Standards

- Python >= 3.9
- Run [`scripts/lint.sh`](/scripts/lint.sh) to format and lint code.
- Strictly type-annotate all code.
- Tests should accompany new features and be placed in the `tests/` directory.

## Review Process

Your PR will undergo review by the maintainers. Approval is based on its alignment with Zeus's dual goals and overall vision.

## License

By contributing to Zeus, you agree to license your contributions under our [Apache-2.0 License](/LICENSE).
