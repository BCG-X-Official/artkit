.. _contributor-guide-git:

How to contribute
=================

This section covers various aspects of contributing to ARTKIT, including commits, pull requests (PRs), and issues management.


General Git Process
-------------------


Contributing to the ARTKIT project is a rewarding experience. Here are the steps and best practices to follow:

1. **Open an Issue**: Before starting work on a new feature or a fix, please `create an issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`_ to discuss it with the core ARTKIT team. This step ensures that your efforts align with the project's direction and that your contribution is likely to be accepted.

2. **Feedback and Confirmation**: Wait for feedback from the core ARTKIT team on your issue. Once the team confirms that your proposed contribution is in line with the project's goals and likely to be accepted, you can proceed to develop your code.

3. **Develop Your Code**: With the green light from the core team, start developing your solution. Please take time to familiarize yourself with the codebase and strive to write code which aligns with the style and conventions of the project.

4. **Write Documentation**: Update the documentation to reflect any changes in functionality or usage introduced by your contribution. Follow the guidelines provided in the :ref:`Documentation <contributor-guide-documentation>` section to add documentation that conforms with our standards and rebuild the documentation to verify the changes. 

5. **Test Your Changes**: Make a concerted effort to test your code thoroughly. Follow the guidelines provided in the :ref:`Testing <contributor-guide-testing>` section to create tests, run the testing suite, and verify that your submission meets our requirements.

6. **Create a Pull Request**: Once your code is ready and tested, submit it through a `pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_ (PR). Ensure your PR follows the detailed guidelines outlined below.

Following these processes not only helps maintain the quality and coherence of the project but also streamlines the contribution and review process.


Creating Issues
----------------

Before `create a new issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`_, please do the following:

1. **Search Existing Issues**: Check ARTKIT's `issue tracker <https://github.com/BCG-X-Official/artkit/issues>`_ to see if the issue has already been reported. Duplicate issues clutter the tracker and dilute attention.

2. **Use a Clear Title**: Provide a descriptive title that summarizes the issue succinctly.

3. **Provide Detailed Information**: Include as much information as possible:

   - Describe the issue, the expected behavior, and the actual behavior.
   - Mention steps to reproduce the problem.
   - Include error messages, relevant logs, and screenshots if applicable.
   - Specify the version of the project and any relevant dependencies.


Commit Messages
---------------

A well-crafted git commit message communicates context about a change to fellow developers and future selves:

1. **Make Atomic Commits**: Each commit should represent a single logical change.

2. **Write Meaningful Commit Messages**:

   - Begin with a short summary (50 characters or less).
   - Follow with a blank line, then a more detailed explanation if necessary.
   - Use bullet points for multiple points.
   - Reference related issues or PRs.

3. **Present Tense and Imperative Mood**: Write "Add feature" not "Added feature".

4. **Capitalize the Subject Line** and **Reference Issues** liberally.

Example Commit Message
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: none

   Add a new LLM classes for integrating with NewService LLMs
   
   - Add `base`, `NewServiceLLM`, `LocalNewServiceLLM`, and `AsynchNewServiceLLM` classes in `src/artkit/llm/newservice`
   - Update `.env_example` with mock API credentials for NewService
   - Add NewService to list of out-of-box supported services in the `README` 
   - Resolves issue #42


Pull Request (PR) Guidelines
----------------------------

1. **Fork and Clone the Repository**: Start by forking the project and cloning your fork.

2. **Create a New Branch**: Work on a new branch specific to the feature or fix.
   - Naming convention: 
      - Features: ``api/<name>``
      - Bugs: ``fix/<name>``
      - Documentation: ``doc/<name>``
      - Tests: ``test/<name>``
      - Build pipelines: ``build/<name>``
   - Branch name should be separated with "-", example: ``api/new-feature``.

3. **Adhere to Commit Guidelines**: Ensure your commits follow the above guidelines.
   - Prefix your commits with ``API``, ``FIX``, ``DOC``, ``TEST``, ``BUILD``.

4. **Write a Clear PR Title and Description**: When opening your PR, provide a detailed description of the changes and motivations.
   - Prefix the PR title with: ``API``, ``FIX``, ``DOC``, ``TEST``, ``BUILD``.

5. **Keep PRs Small and Focused**: This makes PRs easier to review and merge.

6. **Request Reviews** and **Stay Engaged**: Be responsive to feedback.

Example PR Description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: none

   Enhance Search Functionality
   
   This PR introduces improvements to search functionality:
   
   - New search algorithm increases accuracy by 30%.
   - Ability to filter search results by date and relevance.
   - Optimized search query performance for large datasets.
   
   Resolves #123, Related to #456
