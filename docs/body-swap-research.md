# Skin tone matching: research and implementation

The requested effect keeps the target's clothes and body shape. The old body
compositor copied source pixels across the entire person, which necessarily
copied the source outfit. Warping improvements could not fix that mismatch.

## Research

- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarkerOptions)
  provides landmarks and an optional person mask. Its
  [visibility score](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/components/containers/Landmark)
  lets us omit uncertain limbs. The person mask does **not** separate skin
  from clothing.
- Human parsing models explicitly separate body parts and garments. The
  [FASHN human parser](https://github.com/fashn-AI/fashn-human-parser) has
  classes for arms, hands, legs, feet, torso, and clothing, but its SegFormer
  model adds a heavyweight inference dependency. An
  [ATR ONNX parser](https://huggingface.co/basso4/humanparsing/blob/main/parsing_atr.onnx)
  is about 267 MB. This feature remains usable with the current dependencies.
- Colour alone is ambiguous: beige sleeves may resemble skin, while shaded
  skin can differ from the face. The implementation therefore combines a
  colour reference from cheek patches with pose-constrained limb/neck regions
  and the MediaPipe person mask. It skips a frame when the face reference is
  unavailable. Source images showing only a face use the existing face
  analyser's bounding box to obtain the reference colour.

## Implementation and limits

The processor estimates a median source and target skin colour in CIE Lab,
then shifts only candidate target skin pixels. It preserves their local light
and texture variation, and copies all other target pixels exactly. It never
warps source clothing onto the target.

This is a conservative estimate, not semantic skin segmentation. Skin-coloured
garments inside a limb region can be affected. Occluded limbs, tiny faces, and
strongly coloured lighting can cause areas to be missed or the frame to remain
unchanged. A dedicated human parser is the next option if those cases matter
more than model size and live performance.
