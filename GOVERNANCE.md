# Zeus Governance

This document describes how the Zeus project is run: who makes decisions, how they are made, and how people take on more responsibility over time.

Zeus is part of [The ML.ENERGY Initiative](https://ml.energy) and is a [PyTorch Ecosystem](https://landscape.pytorch.org/) project.

## Roles

### Contributors

Anyone who participates in the project is a contributor.
That includes opening pull requests, filing issues, reviewing code, improving documentation, and helping others in discussions.
There is no application process and no minimum level of activity.

All contributions are licensed under the project's [Apache-2.0 License](/LICENSE).

### Maintainers

Maintainers have write access to the repository and are responsible for:

- Reviewing and merging pull requests
- Triaging issues
- Shaping the technical direction of the project
- Cutting releases
- Upholding the [Code of Conduct](/CODE_OF_CONDUCT.md)

The current maintainers are listed in [MAINTAINERS.md](/MAINTAINERS.md), and code ownership is recorded in [`.github/CODEOWNERS`](/.github/CODEOWNERS).

### Project Lead

The project founder, [Jae-Won Chung](https://github.com/jaywonchung), serves as project lead.
The lead's role is to break ties, not to make routine decisions.
Day-to-day authority rests with the maintainers.

## Decision-Making

Zeus is consensus-seeking.
Most decisions never need a formal process: someone opens an issue or pull request, people discuss it, and the result is clear.

**Routine changes.**
Any maintainer may merge a pull request once it has been approved by a maintainer and CI passes.
Maintainers should not merge their own substantial changes without a second pair of eyes, though trivial fixes and release chores are fine to self-merge.

**Substantial changes.**
New optimizers, new device backends, breaking API changes, and anything that reshapes how Zeus is used should start as an issue so the design can be discussed before code is written.
These need agreement among the maintainers, not just one approval.

**When consensus fails.**
If the maintainers cannot converge, the project lead decides.
This is intended to be rare, and the reasoning is recorded in the relevant issue or pull request.

## Becoming a Maintainer

Maintainers are drawn from contributors who have shown sustained, high-quality involvement.
What we look for:

- A track record of merged contributions over several months, not a single large pull request
- Good judgment in code review, including on other people's work
- Familiarity with the codebase beyond the area they first contributed to
- Constructive participation in issues and discussions

The process:

1. An existing maintainer nominates the contributor by opening a pull request that adds them to [MAINTAINERS.md](/MAINTAINERS.md) and [`.github/CODEOWNERS`](/.github/CODEOWNERS).
2. The current maintainers discuss it in the pull request.
3. If the maintainers agree and the nominee accepts, the pull request is merged and write access is granted.
4. If the maintainers do not converge, the project lead decides.

Anyone may suggest a candidate to a maintainer, including the candidate themselves.

## Stepping Down

Maintainers who no longer have time may step down by opening a pull request that moves them to the emeritus section of [MAINTAINERS.md](/MAINTAINERS.md).
Maintainers who have been unreachable for an extended period may be moved to emeritus by the remaining maintainers.
Emeritus maintainers are welcome to return through the normal nomination process.

## Changing This Document

Changes to this document are proposed as pull requests and follow the same process as substantial changes above.

## Code of Conduct

Everyone participating in the project is expected to follow the [Code of Conduct](/CODE_OF_CONDUCT.md).
