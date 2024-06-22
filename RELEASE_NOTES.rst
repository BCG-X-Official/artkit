Release Notes
=============

*artkit* 1.0.1
--------------

- FIX: :class:`.CachedDiffusionModel` and :class`CachedVisionModel` are now also
  available through the :mod:`artkit.api` module. Bot classes had been defined in the
  :mod:`artkit.diffusion.base` and :mod:`artkit.vision.base` modules, respectively,
  even though they are not abstract base classes. The fix moves both classes one level
  up to the :mod:`artkit.diffusion` and :mod:`artkit.vision` modules, which also exposes
  then through the :mod:`artkit.api` module.


*artkit* 1.0.0
--------------

Initial release of *artkit*.