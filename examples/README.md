# Reproducible visual fixture

`coffee.png` is the **coffee** image from scikit-image, photographed by **Rachel Michetti**, courtesy of Pikolo Espresso Bar. scikit-image documents it as **CC0 / no copyright restrictions**.

- License/source documentation: https://scikit-image.org/docs/stable/api/skimage.data#skimage.data.coffee
- Source file: https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/coffee.png
- SHA-256: `cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7`

`make_fixture.py` creates eight seconds of stationary, translated, blank, duplicated, and returned imagery. This is a **controlled image-transformation fixture**, not a real camera recording or a dataset of human communication. No private recordings are included.

`queries.tsv` contains a minimum observed media time in milliseconds, a TAB, and a finite-grammar query. It triggers on the first processed frame reaching that time, so the trigger can be late under overload. It is not an audio-alignment benchmark. `previous cup` means the newest older retained cup observation.
