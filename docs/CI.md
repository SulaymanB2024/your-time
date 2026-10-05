# Hosted test workflow

Local verification is available through `make check`. The approved workflow is
`.github/workflows/checks.yml`; it runs the same checks on pushes to `main` and
pull requests. A successful local run is separate from a successful hosted run.
Source publication requires the GitHub login to have the `workflow` scope.

Standard GitHub-hosted runners are free for this public repository. This uses
a standard `macos-14` runner, not a paid larger runner or the user's Mac:
[GitHub runner documentation](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

Workflow configuration:

```yaml
name: Source checks
on:
  push:
    branches: [main]
  pull_request:
permissions:
  contents: read
jobs:
  checks:
    runs-on: macos-14
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: '3.11'
      - run: python -m pip install uv==0.11.19
      - run: uv sync --locked
      - run: make check
```

The workflow has a read-only token, no stored secrets, no privileged
`pull_request_target` trigger, no deploy/publish step, and no artifact upload.
Only public source and synthetic fixtures would reach the runner. Dependency
advisory scanning and macOS permission readback remain separate checks.
