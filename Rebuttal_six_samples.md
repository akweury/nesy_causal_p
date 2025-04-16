
# The reviewer #4 has a new comment: 

### Regarding the "Six Samples per Pattern" setup:

I have a few questions concerning the examples used. First of all, how were these examples selected? 
I would imagine that an inappropriate selection strategy could lead to biased or 
incorrect rule learning due to spurious correlations or selection bias [1]. 
While the use of only six examples may be justified for your specific tasks, 
I would encourage a more detailed discussion on this point—particularly regarding why the full dataset is not used for rule discovery, 
especially if that choice is a key feature of GRM.

Connecting to Inductive Logic Programming (ILP) [2], 
the compactness of the resulting formula is typically controlled via the Minimum Description Length principle. 
It would be interesting to know whether, giving the full dataset for rule learning, 
a similar approach could be applied in your case, and how you view this possibility.

## On the model and data requirements:

I appreciate your explanation—especially seeing that there are potential high-stakes applications where 
the use of Gestalt principles could be valuable. 
However, I would respectfully disagree with the statement that the model only requires annotated positive and negative images. 
Based on your response, it appears that the model relies on rule learning built on top of a pre-trained object detector. 
Therefore, in a complete pipeline, 
I believe that the required inputs include annotated predicates (e.g., red_triangle) and their corresponding bounding boxes.

That said, 
I strongly encourage the authors to make both the structure of the model and its data requirements more explicit in the paper.

I look forward to hearing your thoughts on these points, and I thank you in advance for any further response.

Reviewer q8Ys

[1] Geirhos et al. (2020), Shortcut learning in deep neural networks, Nature Machine Intelligence

[2] Muggleton and De Raedt (1994), Inductive Logic Programming: Theory and Methods, The Journal of Logic Programming


---

### For the concern on the six samples per pattern setup: 

when we design the negative patterns of each task, they are designed to following only a proper subset of the rules of the positive patterns,
that is to say, it still looks like a positive pattern in some way but do not following all the positive rules. 
We design in this way in order to test the performance of reasoning model, if they are good enough for the counter-facts.




The use of 3 positive and 3 negative examples per pattern is intentional and aligns with the goal of evaluating symbolic generalization 
rather than data-driven statistical learning. 
Inspired by Inductive Logic Programming (ILP), we aim to demonstrate that a small number of structured examples 
is sufficient for discovering general logical rules when appropriate inductive biases (e.g., grouping) are available.

Regarding example selection, each set of six training samples is:

Randomly drawn from a much larger pool of automatically generated images for that pattern.
Verified to avoid label leakage and diversity collapse, 
meaning no example trivially exposes the rule (e.g., through color-only bias or group-only bias).

We agree that spurious correlations could compromise rule learning, 
as discussed in Geirhos et al. (2020), 
and have designed the synthetic generation process to vary multiple visual dimensions independently 
(e.g., shape, size, group count, color, position) precisely to avoid shortcuts. 
We will clarify this in the revised Section 4.3 and include a note in the appendix describing the random sampling strategy and 
how we avoid selection bias.



---

### Why Not Use the Full Dataset for Rule Discovery?

Using full dataset might lead to several problem, that is firstly increase the learning time, and ease the difficulty of the pattern searching.
GRMs can operate in a full-dataset setting, but we select a six examples setting to show its performance.

However, it is indeed an interesting argument that a full dataset might include more noise, 
this would open the door to introduce the probabilistic based rule candidates evaluation.

---
This is an excellent point, and we appreciate the opportunity to connect more explicitly to ILP literature such as Muggleton & De Raedt (1994). 
The compactness and simplicity of learned rules are a core strength of GRMs, 
and we are actively exploring the application of Minimum Description Length (MDL) or 
complexity penalties to guide clause selection.

Currently, we focus on few-shot learning to demonstrate the model’s capability to induce rules from minimal data, 
which is particularly relevant for applications where large labeled datasets are not available. 
However, GRMs are not limited to six examples, and the system can operate in a full-dataset setting. 
In fact, this would open the door to scoring rule candidates based on accuracy vs. complexity trade-offs, and 
we view this as a promising direction for future work. 
We will explicitly discuss this in the Discussion section of the revised paper.

---

### On Model and Data Requirements

The reasoning model takes the output from the encoder as the input, which contains the position, color, shape, size etc symbolic information of each object.

The pretrained object detector in the perception model is used to detect the object labels (or we can say the object shapes).
Its color is determined by the most frequent color in the area of that object segment, the size is the relative ratio compared to the whole image,
the position is acquired according to the image coordination. These symbolic feature is directly the input of the encoder model.

We didn't annotate any predicates nor the bounding boxes to the train image since it is synthetic scenes but it is necessary for the naturalistic data. 



We agree with your observation and appreciate the opportunity to clarify. 
It is accurate that GRMs operate on pre-extracted symbolic representations—either from:

- A pre-trained object detector (such as our perceptor module), or
- Ground-truth symbolic annotations in the synthetic experiments.

Thus, the full pipeline requires:

- For synthetic data: images with ground-truth object attributes (e.g., color, shape, group), generated programmatically.
- For real data: either ground-truth annotations or a trained perceptor that outputs predicate-level facts (e.g., has_color(obj1, red)).


The rule learner itself requires only positive/negative image labels and the symbolic object/group facts per image. 
We will revise the paper to explicitly separate:

Symbolic-level requirements for the logic learner,
- Vision-level requirements for the perception module (e.g., the training of detectors or feature extractors),
- Pipeline dependencies (e.g., the fact that the logic module depends on the perception output, but not on bounding boxes directly).

- This distinction will be clarified in the revised Section 3.1 and summarized in a new “Model and Data Requirements” subsection.



---

### For the concern on the six samples per pattern setup: 

when we design the negative patterns of each task, they are designed to following only a proper subset of the rules of the positive patterns,
that is to say, it still looks like a positive pattern in some way but do not following all the positive rules. 
We design in this way in order to test the performance of reasoning model, if they are good enough for the counter-facts.

### Why Not Use the Full Dataset for Rule Discovery?

Using full dataset might lead to several problem, that is firstly increase the learning time, and ease the difficulty of the pattern searching.
GRMs can operate in a full-dataset setting, but we select a six examples setting to show its performance.

However, it is indeed an interesting argument that a full dataset might include more noise, 
this would open the door to introduce the probabilistic based rule candidates evaluation.


### On Model and Data Requirements

The reasoning model takes the output from the encoder as the input, which contains the position, color, shape, size etc symbolic information of each object.

The pretrained object detector in the perception model is used to detect the object labels (or we can say the object shapes).
Its color is determined by the most frequent color in the area of that object segment, the size is the relative ratio compared to the whole image,
the position is acquired according to the image coordination. These symbolic feature is directly the input of the encoder model.

We didn't annotate any predicates nor the bounding boxes to the train image since it is synthetic scenes but it is necessary for the naturalistic data.

---

Thank you for the thoughtful follow-up and for raising excellent questions regarding 
the **data selection process**, **ILP connections**, and **model requirements**. 
Below we address each of your points in more detail.

---
### On the “Six Samples per Pattern” Setup
The six-sample setup (3 positive, 3 negative) is a deliberate design decision aligned with 
the goal of **evaluating symbolic generalization and counterfactual reasoning**. 
Specifically, each **negative pattern** is crafted to satisfy **only a subset of the rules** defined by the positive pattern. 
This ensures that negative examples **still exhibit plausible structure**—they may partially resemble positive samples—
but **critically violate at least one core constraint**. 
This design prevents trivial discrimination and provides a meaningful test of whether 
the reasoning system can distinguish between structurally similar but logically incompatible patterns.

We will revise the paper to include this explanation and 
emphasize that our intention is not to simplify learning, but to evaluate 
**rule discrimination under minimal supervision and subtle variation**.


---
### Why Not Use the Full Dataset for Rule Discovery?
We appreciate the connection to ILP and the Minimum Description Length (MDL) principle. 
In our current work, we chose a **minimal supervision setting** to:

1. **Highlight the reasoning capability** of GRMs, rather than pattern memorization,
2. **Reduce runtime complexity** associated with clause enumeration and evaluation,
3. Simulate low-data scenarios where reasoning is still expected to generalize.

That said, we agree that using the **full dataset for rule discovery** opens interesting directions. 
A larger dataset may introduce **semantic noise** or **outlier configurations**, 
making it more difficult to find compact, consistent rules. 
In such cases, incorporating **probabilistic rule evaluation** or **scoring based on MDL** becomes relevant and 
could improve robustness. 
We consider this a promising direction for future work and will mention this explicitly in the revised Discussion section.

---
### On Model and Data Requirements
Thank you for highlighting the need to clarify the model's data and structural requirements. 
We agree that this should be made more explicit, and we will revise **Section 3.1** accordingly.

To clarify:

- The **reasoning module** receives an **object-centric matrix** of symbolic features per image: 
color, shape, size, and position.
- These features are derived automatically via a **pre-trained perception module**, 
which does not require manual annotation in the synthetic setup.

Specifically:
- **Object segmentation** is performed by identifying 
**connected regions of uniform color** in the image. Each such region is treated as a separate object.
- The **position** of each object is computed as the centroid of its segmented region, 
avoiding the need for explicit bounding boxes.
- **Shape** is recognized using a **patch-matching memory bank**, 
comparing extracted keypoint patches against stored shape templates.
- **Color** is determined as the **dominant pixel value** within the object’s area.
- **Size** is computed as the relative area of the object compared to the total image.
 
Because of this pipeline, **no manual annotation** (e.g., bounding boxes or predicate labels) is needed for our **synthetic data**. 
However, for **naturalistic data**, a **learned object detection model** would be necessary to perform 
equivalent **segmentation** and **feature extraction**. 
We will clarify this in the revised manuscript and explicitly outline the data requirements for 
both synthetic and real-world use cases.
 

---

Thank you again for your detailed and thoughtful suggestions. 
We believe the clarifications and revisions you’ve prompted will significantly improve 
the paper’s transparency, rigor, and generalizability.


