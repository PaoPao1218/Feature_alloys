"""Command-line entry point for OH adsorption descriptors.

Feature-extraction method adapted from Wang et al., npj Computational Materials,
11, 111 (2025), https://doi.org/10.1038/s41524-025-01607-4.
"""

from feature_alloys import run


if __name__ == "__main__":
    raise SystemExit(run(default_mode="oh"))
