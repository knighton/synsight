# Synsight

## Overview

Synsight ("synesthesia" + "sight") augments your vision with superman facial recognition.  It replaces the faces in images with colorful uniquely-identifying swatches of texture and pattern that you remember and use intuitively.

The space of possible patterns is orders of magnitude richer than the highly constrained space of variations on human facial features, so the patterns are more quickly, easily, and clearly recognizable than the face imagery they replace.

Faces can't be recognized in parallel by humans particularly well.  However, other modalities like color can be (eg, "find the red object in the field of blue objects"), and even more modalities in subjects with synesthesia.  Due to the switch to more human vision parallel-friendly means such as color in the generated patterns, this system will make you superhuman at recognizing Waldo in crowds ("1 to N").

The system was originally designed to allow people with prosopagnosia to identify faces.  It does this by bypassing the specialized brain regions involved in face perception by transforming faces to visual structurally-generic swatches of texture and pattern instead.  It should make your face recognition performance comparably super ("0 to N").

## Operation

Training jointly learns four models, as described below in their respective sections.  The first three models are needed for inference (swatch recognizer can be discarded).

### Face recognizer

The face recognizer embeds faces as vectors.  The vectors encode abstract features which situate faces in the visual similarity space of human faces.  It is a convnet, trained according to triplet loss on the identities of the faces.

### Face masker

The face masker is basically a U-net which segments the faces out of images (so that face regions can be replaced by corresponding identifying swatches).

More pedantically, the generated masks give how much the corresponding pixels in the input images must be noised in order to reduce how well the face recognizer is able to recover the original face embeddings given the noised images instead, while penalizing any excess noise.  Masking pixels is lightly penalized by adding the sum of the masks that were generated, multiplied by a weight, to the loss to prevent degenerately solving the problem by masking everything.

### Face swatcher

The face swatcher replaces the face areas in images with identifying texture/pattern swatches.

This model takes as input (a) the face feature vectors describing what the original faces in the image looked like, together with (b) the images after the generated noise masks from the face masker were applied in order to erase the faces, and generates new images where the faces have been replaced by swatches.  It is trained to recover the original face embeddings from the swatches by use of the swatch recognizer.

### Swatch recognizer

The swatch recognizer is just like the face recognizer, but identifies people by their face swatches instead of their actual faces.  It is used in the training of the face swatcher.
