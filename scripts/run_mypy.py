#!/usr/bin/env python
# Type-check pytensor_ml and fail on any error. The codebase is kept mypy-clean -- there is no
# allowlist of expected failures. Fix new errors, or suppress a genuinely irreducible upstream-stub
# issue at its line with a targeted `# type: ignore[errorcode]` (warn_unused_ignores keeps those
# honest: if the upstream stub is fixed, the now-unused ignore errors and flags itself for removal).

import subprocess
import sys

if __name__ == "__main__":
    result = subprocess.run(["mypy", "--disable-error-code", "annotation-unchecked", "pytensor_ml"])
    sys.exit(result.returncode)
