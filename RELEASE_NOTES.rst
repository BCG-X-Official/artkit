Release Notes
=============

*artkit* 1.0.10
--------------
This release adds access to Azure OpenAI models and adds missing caches for documentation notebooks.

- API: Added :class:`.AzureOpenAIChat` to grant users access to models deployed on Azure OpenAI.
- DOC: Added missing caches in the Connecting to GenAI Models documentation notebook.
- DOC: Fixed some outdated and unclear instructions in the documentation.

*artkit* 1.0.9
--------------


*artkit* 1.0.8
--------------

This is a bugfix release containing updates for a failing unit test and enhancements to historical release notes.

- TEST: Fix `test_huggingface_retry <https://github.com/BCG-X-Official/artkit/blob/1.0.x/test/artkit_test/model/llm/huggingface_tests/test_hugging_face.py>`_ where the session was not mocked correctly
- DOC: Retroactively updated release notes for more consistent quality and detail e.g. hyperlinking class definitions 

*artkit* 1.0.7
--------------

This release adds access to Google's Vertex AI model and fixes the links in the documentation.

- API: Added :class:`.VertexAIChat` to grant users access to Gemini models deployed on Google Vertex AI.
- DOC: Fix broken links on the sphinx homepage

*artkit* 1.0.6
--------------

This release expands ARTKIT's connectivity to include any GenAI application with an exposed HTTP endpoint enabling evaluations of virtually any custom target system.

- API: Added the :class:`.HTTPXChatConnector` class which requires further customization by a user for implementation but considerably expands ARTKIT's connectivity with custom target systems.
- DOC: Added "Calling custom endpoints via HTTP" section in :doc:`user_guide/advanced_tutorials/creating_new_model_classes` to guide implementation of the HTTPX connector

*artkit* 1.0.5
--------------

This release adds access to the Titan diffusion model in AWS bedrock and improves the documentation.

- BUILD: Add prefix to veracode scan number
- DOC: Add clarifying documentation including links, badges, and extra details on dev dependencies 
- API: Add :class:`.TitanBedrockChat` to enable users to access Titan diffusion models.

*artkit* 1.0.4
--------------

This release provides direct ARTKIT access to AWS Bedrock models.

- API: Added :class:`.TitanBedrockDiffusionModel` to enable users to access models deployed on AWS Bedrock.

*artkit* 1.0.3
--------------

This release exposes key base classes so that users can access them for typing purposes.

- API: Key base classes are now exposed through the :mod:`artkit.api` module:
  :class:`.ChatModel`, :class:`.CompletionModel`, :class:`.DiffusionModel`, and
  :class:`.VisionModel`. These classes are frequently used in type hints, and this
  change makes it easier to import them without having to know the exact module
  they are defined in.

*artkit* 1.0.2
--------------

This is a maintenance release that introduces backward compatibility with Python 3.10
along with a few minor API improvements.

- BUILD: *artkit* can now be used with Python versions back to 3.10, allowing use with
  popular services such as Google Colab.
- API: Method :meth:`~.CachedGenAIModel.clear_cache` can now clear cache entries
  after a given create or access time using the new arguments ``created_after`` and
  ``accessed_after``.
- DOC: Minor documentation cleanups.


*artkit* 1.0.1
--------------

- FIX: :class:`.CachedDiffusionModel` and :class:`.CachedVisionModel` are now also
  available through the :mod:`artkit.api` module. Bot classes had been defined in the
  :mod:`artkit.diffusion.base` and :mod:`artkit.vision.base` modules, respectively,
  even though they are not abstract base classes. The fix moves both classes one level
  up to the :mod:`artkit.diffusion` and :mod:`artkit.vision` modules, which also exposes
  then through the :mod:`artkit.api` module.


*artkit* 1.0.0
--------------

Initial release of *artkit*.