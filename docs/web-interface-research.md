# Video face swap interface: product research

Reviewed October 2026. These are implementation references, not model-quality
comparisons.

| Product | Observed workflow | Useful lesson for this interface |
| --- | --- | --- |
| [HeyGen Face Swap](https://help.heygen.com/en/articles/7951446-how-to-use-face-swap-in-heygen) | Select original video or image, choose a replacement face, generate, then review. Their guidance asks for a clear, well-lit, front-facing reference. | Keep the main path short and tell users what makes a usable reference image. |
| [FaceFusion preview](https://github.com/facefusion/facefusion/blob/master/facefusion/uis/components/preview.py) | Renders a chosen video frame and updates that preview when relevant controls change. It also has a frame-by-frame comparison mode. | Let users test a chosen frame before spending time on a full video render. Re-render only when asked, since model inference can be slow. |
| [FaceFusion options](https://github.com/facefusion/facefusion/blob/master/facefusion/uis/components/face_swapper_options.py) and [mask controls](https://docs.facefusion.io/usage/cli-arguments/face-masker) | Exposes swapper weight for supported models, plus separate masking controls such as blur and padding. | Label this project's control **Effect blend**. It uses this repository's existing final-frame opacity blend; it is not FaceFusion's model weight or mask blur. |
| [Premiere comparison view](https://helpx.adobe.com/premiere/desktop/get-started/source-and-program-monitor-adjustments/compare-edits-to-clips-in-the-program-monitor.html) | Offers before/after comparison of the same frame, including a draggable split view. | Show the original and swapped *same frame* with a Result/Compare toggle and reveal slider. |
| [Akool Character Swap API](https://docs.akool.com/ai-tools-suite/character-swap/create) | Treats rendering as a queued task with processing, complete, and failed states. | Keep rendering asynchronous and report its state instead of blocking the browser request. |

## Applied here

1. Upload the video and face once, then use the staged files for any frame
   previews and the full render.
2. Choose the preview time and effect blend. The preview runs the repository's
   image pipeline on an extracted video frame with the same `--opacity` value
   used for full video processing.
3. Compare original and swapped frames visually before rendering. Changing the
   blend or frame time marks the old preview stale.
4. Keep the output video available for in-page review and download. The preview
   is one frame, so it cannot promise that every pose and occlusion in the
   finished video will look the same.

The repository currently swaps the default selected target face. Face selection
for clips containing several people, masking controls, and measured frame
progress are possible later extensions, but they require additional model and
pipeline work beyond an opacity control.
