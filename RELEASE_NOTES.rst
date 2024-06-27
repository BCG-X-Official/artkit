Release Notes
=============

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