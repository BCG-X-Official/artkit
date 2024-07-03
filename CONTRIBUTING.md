# Contributing

We welcome and appreciate contributions to ARTKIT!

### Ways to contribute

There are many ways to contribute, including:

- Create issues for bugs or feature requests
- Participate in community discussions
- Address an open [issue](https://github.com/BCG-X-Official/artkit/issues)
- Create tutorials
- Improve documentation
- Submit pull requests

Known opportunities:

- Add new model connectors
- Add or improve unit tests

We especially encourage contributions that integrate additional model providers and enhance our documentation.

### How to contribute

All contributions must be reviewed and merged by a member of the core ARTKIT team.

For detailed guidance on how to contribute to ARTKIT, please see the [Contributor Guidelines](https://bcg-x-official.github.io/artkit/contributor_guide/how_to_contribute.html).

For major contributions, reach out to the ARTKIT team in advance (ARTKIT@bcg.com).

---

## Setup

### Pre-requisites

The basic requirements for developing this library are:

- [Python](https://www.python.org/downloads/) 3.10 or later
- [git](https://git-scm.com/downloads) for cloning and contributing to the library
- [pip](https://pip.pypa.io/en/stable/installation/) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for installing and managing Python libraries

We recommend using an IDE such as [VS Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/).

### Fork and clone git repository

Fork the public repo and create your branch (e.g. `feat/`, `doc/`, etc.) from the default branch. Synchronize your forked repo/branch with the upstream occasionally.

Clone your fork using HTTPS:

```
git clone https://github.com/<username>/artkit.git
```

or SSH:

```
git clone git@github.com:<username>/artkit.git
```

### Configure environment variables

API credentials should be defined as environment variables in a `.env` file in the project root. This file is listed in `.gitignore` to [keep your credentials secure](https://blog.gitguardian.com/secrets-api-management/). ARTKIT includes a template file called `.env_example` in the project root with placeholder credentials for supported APIs. Modify this file with your credentials and rename it to `.env`.

To load environment variables in a script or notebook, use:

```
from dotenv import load_dotenv
load_dotenv()
```

To encourage secure storage of credentials, ARTKIT classes which make API requests do not accept API credentials directly, but instead require environmental variables to be defined.  
For example, if your OpenAI API key is stored in an environment variable called `OPENAI_API_KEY`, then you can then initialize objects that interact with OpenAI API like this:

```
from artkit.api import OpenAIChat

client = OpenAIChat(model_id="gpt-3.5", api_key_env="OPENAI_API_KEY")
```

Note that the `api_key_env` variable accepts the name of the environment variable as a string instead of directly accepting an API key as a parameter. This pattern reduces the risk of accidental exposure of API keys in code repositories, since the key is not stored as a Python object which can be printed.

### Create virtual environment

From the project root, create a dedicated environment for your project using, e.g., `venv`:

```
python -m venv artkit-env
```

and activate it on MacOS or Unix with:

```
source artkit-env/bin/activate
```

or on Windows with:

```
.\artkit\scripts\Activate
```

### Enable import of local ARTKIT modules

We recommend installing the project locally in developer mode to enable importing local ARTKIT modules in
scripts or notebooks as if the library is installed, but with local changes immediately reflected.

To install ARTKIT in developer mode, run the following from your project root:

```
pip install -e ".[dev]"
```

There are optional dependencies for the compatible LLM providers (anthropic, google, groq, huggingface, openai as of June 19, 2024) which can be installed collectively via "dev" or individually as desired:

```
pip install -e ".[anthropic, google-generativeai, groq, huggingface_hub, openai]"
```

Note: The LLM providers are optional to run and use ARTKIT, but the unit tests will fail without installing them

As an alternative approach, you can add the folder `artkit/src` to your `PYTHONPATH`, and this will
enable importing local ARTKIT modules into scripts or notebooks.

### Install run dependencies

#### GraphViz

[GraphViz](https://graphviz.org/) is required for generating pipeline flow diagrams, which is necessary for ARTKIT's full functionality. Install the library and ensure it is in your system's PATH variable:

- For MacOS and Linux users, simple instructions provided on [GraphViz Downloads](https://www.graphviz.org/download/) should automatically add GraphViz to your path
- Windows users may need to manually add GraphViz to your PATH (see [Simplified Windows installation procedure](https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224))
- Run `dot -V` in Terminal or Command Prompt to verify installation

### Set up local development environment

#### Pandoc

Pandoc is required to render Jupyter Notebooks for building the sphinx documentation.

- MacOS users can install Pandoc with `brew install pandoc`
- Windows and Linux users should follow the [Pandoc installation](https://pandoc.org/installing.html) instructions for their system

#### Pre-commit hooks

[Pre-commit hooks](https://pre-commit.com/) are strongly encouraged to enforce uniform coding standards across all contributors:

First, install pre-commit within your virtual environment
```
pip install pre-commit
```

Next, run pre-commit's install command to create a Git hook script configured by `.pre-commit-config.yaml`.
```
pre-commit install
```

To execute the pre-commit hooks on demand, use `pre-commit run` from the command line.

### Run unit tests

This project uses [pytest](https://docs.pytest.org/en/8.0.x/) to support functional testing. To run the test suite:

```
pytest
```

To maintain high standard for test coverage, the testing pipeline is configured to require at least 90% test coverage of the codebase, otherwise `pytest` will exit with a failure status.

### Build documentation

Ensure the src directory does not contain any environment-specific folders or files which can be interpreted as packages. If you do, you will get the following error during the sphinx build: `assert len(PACKAGE_NAMES) == 1, "only one package per Sphinx build is supported"`. We have added logic to ignore any directory that ends with `.egg-info`, which is generated when you pip install.

To build the [sphinx](https://www.sphinx-doc.org/en/master/) documentation, navigate to the `sphinx` directory and run:

```
./make.py html
```

To view the documentation, open `sphinx/build/html/index.html` in the web browser of your choice.

## Contribution Guidelines

Visit the [Contributor Guide](https://bcg-x-official.github.io/artkit/contributor_guide/index.html) for detailed standards, best practices, and processes for contributors.
