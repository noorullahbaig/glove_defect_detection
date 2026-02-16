# AI Prompt Bank (Glove Types + Defects)

Goal: generate a medium-scale dataset (~2.7k–3.0k images) where **one defect is present per image**.

## Global rules (use on every prompt)

- Single glove only (no extra gloves).
- Centered subject, uncluttered background.
- Photorealistic, sharp focus, high texture detail.
- No text, no watermark, no logos.

Base suffix (append to every prompt):

> Product photo, realistic lighting, high detail texture, centered subject, no text, no watermark, no logos, no extra gloves, uncluttered background.

## Variation tokens (cycle between runs)

Background:
- matte lab table
- wood tabletop
- neutral gray paper sweep
- stainless steel tray
- light concrete surface

Lighting:
- soft diffused daylight
- overhead fluorescent
- warm indoor lighting
- side-lit with mild shadows

Camera:
- top-down
- 45-degree angle
- close-up
- medium shot

## Glove type base prompts (clean)

Latex (clean):
- A single surgical latex glove lying flat, empty glove, smooth thin material with subtle highlights, {camera}, {lighting}, {background}. [base suffix]

Leather (clean):
- A single leather glove lying flat, visible leather grain and stitching seams, {camera}, {lighting}, {background}. [base suffix]

Fabric knit (clean):
- A single knit fabric winter glove lying flat, visible woven knit texture, {camera}, {lighting}, {background}. [base suffix]

## Defect clauses (append exactly one clause)

Structural:
- hole: Include a small puncture hole on one fingertip OR on the palm area; background visible through the hole.
- tear: Include an elongated tear split along a finger or across the palm; edges of the material slightly separated.
- missing_finger: One finger is missing or cut off short; the glove silhouette clearly shows the missing finger.
- extra_fingers: The glove has an extra finger-like protrusion; silhouette clearly shows an extra finger.

Surface / color:
- discoloration: Include a noticeable discoloration patch (yellowish/grayish tint shift) on the glove surface, not just a shadow.
- stain_dirty: Include a dirty stain smudge (brown/black), irregular edges, looks like grime on the surface.
- spotting: Include multiple small dark speckles across the glove surface, scattered spots.
- plastic_contamination: Include a small shiny plastic fragment or thin plastic film stuck on the glove surface.

Texture / structure:
- wrinkles_dent: Include heavy wrinkling/creased dents across the glove surface; no single long fold line dominates.
- damaged_by_fold: Include one prominent long fold crease line, like the glove was sharply folded and damaged along that fold.
- inside_out: The glove is turned inside out; seams and inner texture visible externally.

Latex-only cuff defects:
- improper_roll: The cuff looks unevenly rolled or irregularly folded, messy cuff edge.
- incomplete_beading: The cuff edge looks thin/unfinished and not properly beaded; uneven rim at the opening.

## Target counts (medium-scale)

If you are generating ~4 images per batch run:

- Clean (no defect): 250 per glove type (latex/leather/fabric) = 750 total
- Focus defects (5 labels): 80 per glove type per label = 1,200 total
  - missing_finger, extra_fingers, hole, discoloration, damaged_by_fold
- Other common defects:
  - tear, wrinkles_dent, inside_out: 40 per glove type per label = 360
  - stain_dirty, spotting, plastic_contamination: 40 per glove type per label = 360
- Latex-only cuff defects:
  - improper_roll, incomplete_beading: 60 each (latex only) = 120

Total: ~2,790 images.

## Notes convention (for labels.csv traceability)

Use notes like:

- prompt_id=latex_clean_03 variation=bg_stainless lighting_fluorescent camera_topdown defect=hole

