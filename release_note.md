### UltraFix

On-device 4K high-quality anime image generation

- **UltraFix** — a new tiled-diffusion repair pass for upscaled images. After upscaling a result, tap UltraFix to re-run the model over the image tile by tile and restore real fine detail at high resolution, instead of just a smoother enlargement. **Available only on SDXL DMD2 models** (it runs at CFG 1 / few steps). It has its own Steps / Denoise controls (defaults 10 / 4): keep denoise-steps ÷ steps ≤ 0.5. **Use it only on anime models** — on realistic models it tends to duplicate subjects. Use it as a finishing pass on images you like, and run it again on the result to build up more detail. (Based on the PixelRush algorithm.)

### Generation

- The back gesture during generation now interrupts the run instead of exiting the backend — it stops within about one step and frees the NPU, so you can start again right away.
- Improved inpaint blending when compositing the result back into the original image, removing the faint square seam.

### History & Results

- Added Favorite for generation results, with a favorited / not-favorited filter in history.
- Added Upscale and UltraFix buttons to history images.
- The result page now shows the most recent image when the current model has history, instead of an empty state.
- Added a global history screen covering every model — including models you've deleted.
- Deleting a model now keeps its history by default, with an optional checkbox to also delete the history data.
- History now loads as a paged query, so large histories stay smooth.

### Models

- Support pinning models to the top and renaming custom models (history, parameters and pin state all move with the new name).

### App & Settings

- Added history import / export — back up your whole history (images + parameters) to a zip and restore it later or on another device.
- Added a manual Clean Temp Files action to clear temporary files that in some cases can't be cleaned up automatically (e.g. an interrupted download or extraction).

### Fixes

- Fixed blurry preview thumbnails on the result page.
- Fixed the scroll-position memory of the recent-results strip on the result page.
- Fixed non-square images not exiting preview mode when tapping outside the image.
- Fixed a crash on some devices when the upscaler page rendered an overly large image.
- Fixed the upscaler download not following a changed download source, and the upscaler existence check; downloads are now integrity-checked.
- Fixed a "job cancelled" error that sometimes appeared even though the model downloaded successfully.
- Fixed backend lifecycle issues during model exit / load that had existed since the MD3E UI update.
- Other i18n and performance refinements.

Variants :

- LocalDream_xxx_with_filter: Block NSFW Results. Same as Google Play version.
- LocalDream_xxx: No filter.

Co-Authored-By: Claude Fable 5 & Claude Opus 4.8
