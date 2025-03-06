.. _contributor-guide-testing:

Testing
=======

Effective unit testing is a cornerstone of stable and reliable software. ARTKIT uses `pytest <https://docs.pytest.org/en/8.0.x/>`_,
a powerful and flexible testing framework for Python. This section provides guidelines on writing unit tests with ``pytest`` to ensure 
your contributions align with our standards.


Getting Started with Pytest
---------------------------

Pytest discovers tests in your project automatically, without the need for boilerplate code. It encourages the use of plain assert 
statements and is compatible with the built-in ``unittest`` framework.

If you followed the :ref:`setup guide <contributor-guide-setup>` then ``pytest`` should already be installed in your development environment.
The test suite can be run from the project root:

::

   pytest

.. attention::
    To maintain high standard for test coverage, the testing pipeline is configured to require at least 90% 
    test coverage of the codebase, otherwise ``pytest`` will exit with a failure status.


Structure of a Test Module
--------------------------

A test module is a Python file starting with ``test_``, containing test functions that also start with ``test_``. Here's an example structure:

.. code-block:: python

   def test_example():
       assert True

ARTKIT tests are organized in the ``test`` folder at the project root. Please review the folder structure and add your tests in an appropriate sub-directory.


Guidelines for Writing Tests
----------------------------

1. **Test One Thing per Function**: Each test function should focus on testing a single behavior or aspect of a function. This makes tests easier to understand and debug.

2. **Use Descriptive Test Names**: The function names of your tests should be descriptive enough to understand the purpose of the test. For example, ``test_sort_empty_list_returns_empty_list()`` is clear and informative.

3. **Arrange-Act-Assert Pattern**: Structure your tests with setup (arrange), invocation (act), and assertion (assert) stages clearly separated. This pattern keeps tests organized and easy to follow.

4. **Leverage Fixtures for Setup and Teardown**: ``pytest`` fixtures are functions run by pytest before (and sometimes after) the actual test functions. They are a powerful feature for setup and teardown operations. Use fixtures to manage resources like database connections or temporary files.

.. code-block:: python

   @pytest.fixture
   def input_value():
       return 39

   def test_divisible_by_3(input_value):
       assert input_value % 3 == 0

5. **Parametrize Tests**: Use ``pytest.mark.parametrize`` to run a test function multiple times with different arguments. This is powerful for testing a function over a range of inputs.

.. code-block:: python

   @pytest.mark.parametrize("test_input,expected", [(3, True), (4, False)])
   def test_is_odd(test_input, expected):
       assert is_odd(test_input) == expected

6. **Cover Edge Cases**: Ensure your tests cover edge cases, not just the typical or expected use cases. This includes testing with empty inputs, invalid data, or extreme values.

7. **Run Tests Frequently**: Run your tests frequently during development to catch regressions or unintended changes early.
