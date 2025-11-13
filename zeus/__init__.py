"""Zeus is a framework for deep learning energy measurement and optimization.

- [`device`][zeus.device]: Abstraction layer over compute devices
- [`monitor`][zeus.monitor]: Programmatic power and energy measurement tools
- [`optimizer`][zeus.optimizer]: A collection of optimizers for time and energy
- [`utils`][zeus.utils]: Utility functions and classes
- [`callback`][zeus.callback]: Callback definition
- [`exception`][zeus.exception]: Base exception class definition
- [`metric`][zeus.metric]: Tools for defining and tracking power and energy-related metrics
- [`show_env`][zeus.show_env]: Command line tool for install verification and device detection
- [`_legacy`][zeus._legacy.policy]: Legacy code mostly to keep our papers reproducible
"""

import logging

__version__ = "0.13.1"

# Add NullHandler to prevent "No handler found" warnings when Zeus is used as a library.
# Applications using Zeus should configure logging via logging.basicConfig() or by
# setting up handlers on the root logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
