# Gemini Agent Instructions for QuantumSim

This document provides guidelines for the Gemini AI agent to ensure its contributions are consistent with the project's standards and conventions.

## 1. Project Overview

QuantumSim is a Python library for high-fidelity quantum dynamics simulation and control pulse design. It focuses on modularity and composability, with a clear separation between physical models (`hamiltonian.py`), control pulse definitions (`pulses.py`), and the simulation engine (`simulator.py`). The library is built on `qutip`, `numpy`, and `matplotlib`.

## 2. Environment Setup

**CRITICAL:** All development, including running Python scripts, installing dependencies, and executing tests, MUST be performed within the Conda virtual environment named `qutip-env`.

- **Running Commands:** To run any command within this environment, chain it to the activation command with `&&`. For example:
    - `conda activate qutip-env && python examples/template.py`
    - `conda activate qutip-env && pip install <new_package>`
    - `conda activate qutip-env && pytest`
- **Environment Not Found:** If the `qutip-env` environment cannot be found or activated, you MUST stop immediately and report this issue. Do not attempt to proceed.

## 3. Development Conventions

### Code Style & Formatting
- **PEP 8:** Adhere strictly to PEP 8 guidelines for all Python code.
- **Existing Style:** Critically, always match the style of the surrounding code. Pay attention to variable naming conventions (e.g., `snake_case`), class names (`PascalCase`), and the use of whitespace.
- **Comments:** Add comments sparingly. Focus on *why* something is done, not *what* is done. The code should be self-documenting. Use Chinese for comments as is the existing convention.
- **Line Length:** Keep lines under 120 characters.

### Typing
- **Mandatory Type Hinting:** All new functions and methods MUST include type hints for all arguments and return values. The project uses the `typing` module extensively.

### Docstrings
- **Google Style:** Use Google-style docstrings for all public modules, classes, and functions. Docstrings should be clear and concise, explaining the purpose, arguments, and return values.
- **Language:** Write docstrings in Chinese, following the convention in the existing code.

### Imports
- **Organization:** Group imports in the following order:
    1. Standard library imports (e.g., `os`, `sys`).
    2. Third-party library imports (e.g., `numpy`, `qutip`).
    3. Local application imports (e.g., `from .hamiltonian import ...`).
- **Relative Imports:** Within the `src/quantum_sim` package, use relative imports (`.`) to refer to other modules within the package.

### Dependencies
- Any new third-party dependencies must be added to both `requirements.txt` and the `install_requires` list in `setup.py`.

### Plotting & Visualization (matplotlib)
- **Use English for Labels:** All plot titles, labels, and legends must be in English to avoid potential encoding issues with non-ASCII characters.
- **Use Raw Strings for Formulas:** When using LaTeX for mathematical formulas in annotations, always use raw strings (e.g., `r'$|0\rangle$'`) to ensure they are rendered correctly.

## 4. Testing

- **Framework:** Use `pytest` for all tests.
- **Location:** All test files should be placed in a `tests/` directory at the project root. Test files should mirror the structure of the `src/` directory (e.g., tests for `src/quantum_sim/pulses.py` should be in `tests/test_pulses.py`).
- **Coverage:** All new features or bug fixes must be accompanied by corresponding tests to ensure correctness and prevent regressions.

## 5. Committing Changes

- **Commit Messages:** Use the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. The basic format is:
  ```
  <type>[optional scope]: <description>
  
  [optional body]
  
  [optional footer]
  ```
  - **Common types:** `feat` (new feature), `fix` (bug fix), `docs` (documentation), `style` (formatting), `refactor`, `test`, `chore` (build/tool changes).
- **Granularity:** Commits should be atomic and represent a single logical change.
- **Commit Workflow:** To avoid shell quoting issues with complex, multi-line commit messages, MUST use the following workflow:
  
  1.  MUST Write the full commit message to a temporary file (e.g., `COMMIT_MSG.tmp`) using the `write_file` tool.
  2.  Use `git commit -F COMMIT_MSG.tmp` to create the commit.
  3.  Delete the temporary file after the commit is successful.

### Notebook Output Cleaning (pre-commit)

This project uses a `pre-commit` hook to automatically clean the output from Jupyter Notebooks (`.ipynb` files) before they are committed. This ensures that the repository remains clean and avoids large, unnecessary diffs caused by cell outputs.

- **Tool:** The cleaning is performed by `nbstripout`.
- **Configuration:** The hook is defined in `.pre-commit-config.yaml` and the dependencies are listed in `requirements.txt`.

**Workflow:**

This process is mostly automatic, but it requires understanding a specific behavior:

1.  When you commit a notebook that has outputs, the `pre-commit` hook will run, strip the outputs, and then **abort the commit**.
2.  You will see a `Failed` message from the hook, indicating that it modified files. **This is expected.**
3.  To complete the commit, you must **stage the newly cleaned files** (`git add .`) and then **run the `git commit` command again**.
4.  On the second attempt, the files will be clean, the hook will pass without doing anything, and the commit will succeed.

## 6. General Workflow

When asked to perform a task (e.g., add a feature, fix a bug):

1.  **Understand:** Analyze the request and the relevant codebase. Use `read_file` and `glob` to explore the project.
2.  **Plan:** Formulate a clear plan. If the change is significant, communicate the plan before implementing.
3.  **Implement:** Write the code, strictly adhering to the conventions in this document.
4.  **Test:** Write or update tests in the `tests/` directory. Run the full test suite to ensure no regressions were introduced.
5.  **Verify:** Run any necessary static analysis or linting tools.
6.  **Commit:** Once verified, prepare a commit with a well-written, conventional commit message.