Style guide for the cartoon_photos niche.

Each segment becomes one scene with one action, drawn in the flat cartoon style of reference 1. Every prompt uses at least one reference from the refs list of its segment. Reference 1 is the base style and is added to most segments. Reference 6 is the main character and is added after the base style whenever the character appears. Choose the reference that matches the content of the segment, and use only reference numbers that appear in the refs list.

Reference roles:
- ref_1 (base style): a flat landscape with a bright sky, a green grass line, brown soil layers and grey rock below, with bold dark outlines and flat colour fills. Use it for any scene with ground, soil, rock, sky or water.
- ref_2 (on-frame text): a white background with one short red handwritten phrase and a small hourglass icon. Use it only when the whole segment is one short statement, for example a timer or one number, and the frame holds only that statement on a plain background. Otherwise, leave the text out of the frame.
- ref_3 (labelled diagram): a grey rock with a green top, rising blue bubbles and one handwritten label with an arrow. Use it for an explanatory diagram with one label. Use it when the segment explains how or why something works.
- ref_4 (timeline): a horizontal black arrow with evenly spaced circles, a dot at the left end, a small label under each end and a red pin above the line. Use it for time spans or a sequence of periods. Use it when the segment names a span of time or a sequence of periods.
- ref_5 (comparison): two panels side by side, divided by a dashed line, each with a title above it and a short label inside. Use it for comparing two situations. Use it when the segment compares two situations.
- ref_6 (main character): a plain round white head with a simple expression, a blue short-sleeved shirt, blue jeans and white shoes, standing straight, full body, on a white background. Use it whenever the character appears.

Reference combinations: choose exactly one of the following combinations, and list its numbers in the refs list. Base style alone: ref_1. Base style with the character: ref_1 and ref_6. Base style with the on-frame text: ref_1 and ref_2. Base style with the character and the on-frame text: ref_1, ref_2 and ref_6. The on-frame text alone: ref_2. One diagram alone: ref_3, ref_4 or ref_5. A diagram is always used alone. The character always comes with the base style.

Hero placement: the main character appears only in a scene with a place or a person's action. A frame that shows only on-frame text (reference 2 alone) and each diagram (reference 3, 4 or 5) leave the main character out. A scene with a place or a person's action adds the main character as the traveller, with the refs list [1, 6] for the scene alone, or [1, 2, 6] when on-frame text is also shown beside the scene. A segment that is only about a thing or a fact leaves the main character out as well. Whenever the main character appears, describe it only as the main character as in reference 6. In every scene with a place or a person's action, the main character is shown by default; text-only and diagram segments are the only exceptions.

Hero poses and expressions: the model chooses the pose and the expression of the main character by segment, one pose per prompt. Poses to choose from: standing upright, walking, sitting, crouching, pointing with a hand, holding an object, looking up, looking to the side, raising one hand, lowering both arms, running, leaning forward. Expressions to choose from: neutral, thoughtful, happy, surprised, frightened, tired, determined, focused. The pose and the expression come from the segment text, and the same pose is used in at most two clips in a row.

Visual style: bold dark outlines around every shape, flat colour fills with at most one simple shadow, simple readable shapes, and a limited bright palette of sky blue, green, brown, grey and orange. Diagrams sit on a plain white or pale background. Keep surfaces flat, with simple lighting drawn as shapes only.

Text on the frame: Text on the frame is the exception: most frames show only the scene. A text card always names one keyword from the segment text, 1 to 4 words, written exactly as in the subtitles, and the prompt states those words in the card. A text is either one short phrase of at most 4 words, or one object described with one word for each part, using at most 4 parts. When text is shown beside a scene, use 1 to 4 words that carry the main point of the segment, a keyword the segment itself stresses. Each shown text is one complete phrase taken whole from a single sentence of the segment. Use only one text phrase in each frame. Write the text in the on-frame language named in the task. Use handwritten capital letters, large enough to read at a glance. The default colour is black. On a dark background use white. Use red only for the most important word or number. Any other colour is allowed when it stays easy to read.

Text on the frame norm: each batch of 25 clips holds at most 5 clips with text on the frame, that is at most 1 in 5. Text on the frame stays a keyword or a short phrase of 1 to 4 words that the segment stresses. The words of each text card are written as plain words, for example: the words TIME MACHINE in red handwritten capitals.

On-frame text uses the same words as the subtitle text, keeping every name, number and spelling unchanged, and it still follows the 4-word or one-object rule above.

Crossing out: draw two red diagonal strokes over the scene, like brush strokes, to show that something is missing, wrong or a myth. Name the crossed-out element in the prompt and place it in the scene, for example a person with equipment and two red diagonal strokes over the scene.

Prompt form: describe exactly what is visible in the frame. Every object visible in the frame is described in full, including its appearance, clothing, height, age, sex, pose and colour, in the style of this niche. The main hero is described only as the main character as in ref_6, and its full appearance is taken from ref_6 alone.

Other people and animals are drawn in the base style of reference 1 alone: bold dark outlines, flat colour fills and simple shapes. Settings are described in the same style and in full detail.

Each object, character and setting in a prompt comes from the segment text alone, and it is added only when the segment text names it.

Label rule: when a segment uses references, write one tag for each reference in its refs list. The tags follow the ascending order of the refs list: <image1> is the lowest number in refs, <image2> the next, and so on. Use each tag at least once, and only those tags.

Final check before each prompt is returned: one scene, the character count follows rules.yaml, each character fully described, plain props only, describing only the scene itself, quotation marks absent, and at least one reference used.
