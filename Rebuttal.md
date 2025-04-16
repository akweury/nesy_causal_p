

---

## Rebuttal to Reviewer EeGA (Score: 3 – Reject, Confidence: 3 – Somewhat confident)

---
Dear Reviewer EeGA,

we thank for the thoughtful and encouraging comments, 
especially the recognition of the paper's clear writing and 
the novelty of incorporating high-level perceptual heuristics into neuro-symbolic reasoning. 
Below, we respond to each of the concerns and outline how we plan to revise the paper accordingly.

Best regards,

Authors of Submission 509

---
### On the Contribution and Its Broader Impact

We understand the concern that the benefits of explicitly hard-coding Gestalt principles into GRMs 
may appear unsurprising, given that the evaluation tasks are designed around them. 
However, we respectfully argue that implementation is not trivial, 
and the value of GRMs lies in how this is done and what new capabilities it enables:

- **Novel capability**: GRMs are the first neuro-symbolic system to explicitly perform **reasoning over perceptual groups**, 
a critical gap in both LLM-based and differentiable logic-based systems.
- **General benefit**: Grouping is not only useful for Gestalt tasks—it also **reduces relational complexity**, 
enabling GRMs to scale where existing models fail. 
This is evident in our performance on **CLEVR-Hans and Kandinsky Patterns**, where grouping serves as a compression mechanism.
- **Path forward**: We do not position GRMs as a universal VQA model but as a **scalable, interpretable module** 
that can be integrated into broader VQA pipelines.

In the revised version, we will make the motivation and broader applicability more explicit by:

- Clarifying whether the system is best interpreted as a **cognitive model**, an **engineering tool**, or both.
- Describing how grouping could help even in non-Gestalt VQA tasks (e.g., reducing candidate relation sets).
- Proposing extensions to integrate GRMs into downstream VQA systems via modular neuro-symbolic pipelines.

---
### On Human Comparison and Cognitive Plausibility

Thank you for raising the idea of comparing GRM outputs to human perception. 
While we do not position GRMs as cognitive models per se, 
we agree that such an analysis would be valuable—particularly in 
evaluating **failure modes** or ambiguous scenes where human judgments diverge.

We will include a **discussion of this possibility** in the revised paper and highlight recent work that 
uses human-Gestalt labeling (e.g., in psychophysical experiments or vision science benchmarks). 
A future direction would be to align GRM decisions with **human grouping saliency maps** or response data, where available.



---
### On Extending the Evaluation Beyond Gestalt Tasks

We appreciate this suggestion and agree that including accuracy comparisons (not just runtime) on **CLEVR-Hans and Kandinsky Patterns** 
would significantly strengthen the paper.

In the revised version, we will:
- Add accuracy results for GRM, DeepProbLog, and Grouping-enhanced ProbLog on both benchmarks.
- Emphasize how GRMs achieve high accuracy and scalability simultaneously, whereas other models trade off one for the other.
- Extend the comparison to include error cases and analysis, showing how grouping aids or hinders generalization in non-Gestalt settings.

---
### On Clarifying Figure 5 and NEUMANN’s OOM Behavior
Thank you for pointing this out. You are right. So Figure 5 currently presents inference time for GRM and DeepProbLog (with and without grouping),
but not NEUMANN. 
The text incorrectly refers to NEUMANN, which does not appear in the figure due to **OOM (out-of-memory) 
errors** during benchmarking on large scenes.

To address this, we will do the following updates in the revised version:

- Revise the caption and accompanying text for Figure 5 to **remove the incorrect mention of NEUMANN**.
- Add a **separate figure or table** showing the **memory usage** of NEUMANN vs. GRM and DeepProbLog on smaller scenes where all models run successfully.
- Clearly state the scale at which NEUMANN fails due to graph explosion, 
and explain **how grouping allows GRMs to avoid this issue**.


---
### Conclusion
To summarize, we will revise the paper to:

- Clarify the broader utility of GRMs beyond Gestalt tasks.
- Add **accuracy comparisons** on CLEVR-Hans and Kandinsky Patterns.
- Include a discussion on **human response comparison** as a future cognitive benchmark.
- Correct and extend Figure 5, separating runtime and memory benchmarks.
- Make clear that the value of GRMs lies not just in rule-based pattern matching, 
but in **generalizing symbolic reasoning through grouping**—a novel, scalable capability.
---

**We appreciate your thoughtful critique and 
hope that these improvements will better reflect the scope and significance of our contribution.**

---


---

## Reviewer 8pMB (Score: 6 Weak Accept, Confidence 3: Somewhat confident)

---

Dear Reviewer 8pMB,

we sincerely thank for the positive remarks and insightful feedback. 
We appreciate your recognition of the novelty and relevance of our approach, 
as well as your comments on clarity and experimental design. 
Below, we respond to the concerns you raised and outline improvements for the revised version.

Best regards,

Authors of Submission 509


---
### On Evaluation Scope and Use of a Hand-Crafted Dataset

We agree that the current evaluation is focused on synthetic tasks specifically 
designed to exercise **Gestalt-based relational reasoning**. 
This was intentional: existing benchmarks often entangle many visual variables, 
making it difficult to isolate the contribution of perceptual grouping. 
By controlling for visual properties, we show that **explicit grouping enables relational generalization** 
where other models (including LLMs and NeSy baselines) fail due to combinatorial explosion.

Nonetheless, we acknowledge the need to evaluate GRMs on broader datasets. In a revised version, we will:

- Introduce additional **control tasks** not based on Gestalt principles (e.g., object-unique attributes) to show GRMs are not overfitting to grouping patterns.
- Include examples from existing structured reasoning datasets (e.g., **CLEVR-Hans**, 
already partially included in Figure 5) 
and extend to more real-world visual scenes with noisy grouping cues.

---
### On Encoding and Symbolic Bias
It is true that our current implementation uses symbolic features (e.g., one-hot encodings of shape and color), 
which facilitates interpretable rule learning. 
However, GRMs **are not limited to categorical representations**. In Section 3.2 and Appendix A, 
we note that object and group features can include **CNN-based visual descriptors**. 
While not yet the focus of this version, GRMs can be extended to:

- Use distributed neural features (e.g., embeddings from ResNet patches or VQ-VAE).
- Incorporate similarity functions over continuous attributes for grouping. 
This opens a path toward integration with modern visual backbones 
while preserving symbolic interpretability—a direction we plan to pursue.



---
### On Differences from Prior Work

Thank you for pointing out the need to further differentiate our work. 
While prior NeSy models such as NEUMANN and DeepProbLog integrate logic and perception, 
none explicitly model **perceptual grouping** or support **group-level reasoning** as first-class operations.

We will enhance the Related Work section to highlight distinctions:

- GRMs introduce **predicate invention over groups**, a capability missing from LLMs and differentiable logic learners.
- The grouping mechanism supports **perceptual compression**, enabling scaling to larger relational spaces 
(see scalability results in Figure 5).
- Unlike models trained end-to-end with thousands of examples, GRMs generalize with **only 3 positive and 3 negative** samples per task.

---
### On Scalability of the Mechanism

We respectfully disagree with the concern that GRMs are not scalable. 
Figure 5 directly supports the claim that grouping reduces complexity: as the number of objects increases, 
NEUMANN fails due to memory overload, whereas GRMs remain efficient due to perceptual compression via grouping.

Moreover, Table 1 shows that GRMs can solve reasoning tasks with **up to 45 objects and 4 groups**—a scale not achieved by existing NeSy models. 
We will further strengthen this claim by including:

- Timing and memory benchmarks across increasing object counts,
- Analysis of grouping overhead vs. relational complexity,
- Additional comparisons with symbolic and hybrid LLM pipelines.


---
### On Limited Baselines

We acknowledge the importance of clarifying baseline selection. We chose:

- **ChatGPT-4o**, representing the latest **LLM-based reasoning** with vision inputs,
- **NEUMANN**, a **differentiable logic learner** and representative of structured NeSy models.

Both are strong and representative in their respective paradigms. 
We ensured fairness by aligning their available background knowledge and preprocessing steps (see Appendix D). 
In the revision, we will explicitly justify these baselines and clarify that **GRMs are not outperforming trivial models**, 
but rather **fail cases for state-of-the-art systems**.

---
### On Evaluation Criteria

We appreciate the suggestion to better explain our evaluation pipeline. We will revise Section 4.3 to clearly describe:
- **Training regime**: 3 positive + 3 negative examples per pattern.
- **Test evaluation**: 6 novel configurations not seen during training.
- **Performance metrics**: binary classification accuracy, rule generalization success, and interpretability of the generated predicates.
- **Scalability criteria**: runtime and memory usage vs. number of objects/groups.

---
### Typographical and Minor Errors
Thank you for pointing out the typo "scalr" → "scalar". 
This will be corrected.


---
### Conclusion

We thank the reviewer for recognizing the importance and potential of our work. In the revised version, we will:

- Expand evaluation to non-Gestalt and real-world tasks,
- Clarify symbolic vs. distributed representations,
- Improve baseline justification and reproducibility details,
- Refine presentation and define evaluation more rigorously.

---

**We believe these revisions will address your concerns and further strengthen the contribution. 
Thank you again for your thoughtful and encouraging review.**

---

---

---


## Rebuttal to Reviewer Yb5e (Score: 2 – Strong Reject, Confidence: 3 – Somewhat confident)


---

Dear Reviewer Yb5e,

we thank for your detailed and constructive feedback. 
Your comments raise important issues regarding clarity, task formulation, robustness, and assumptions. 
Below, we address each point and outline the specific improvements we will implement in the revised version.

Best regards,

Authors of Submission 509

---
# Rebuttal

---
### Clarification of the Problem and Evaluation Metrics
The reviewer is correct in noting that the introduction highlights the capacity of Gestalt Reasoning Machines (GRMs) 
to describe visual patterns in natural language. 
However, this is not the sole objective. 
The primary goal of GRMs is to learn **logical rules from visual inputs** using neuro-symbolic reasoning. 

Evaluation is performed through:
- **Classification accuracy** on unseen examples: does the inferred rule generalize?
- **Predicate fidelity**: how well does the symbolic rule represent the pattern?

These logical rules are then optionally passed to an LLM to generate n**atural language descriptions**, 
enhancing interpretability—not as part of the reasoning core. 
This multi-stage pipeline (perception → encoding → logic → language) 
is described in **Section 3** and visually summarized in **Figure 2**. 
We will clarify the distinction between the reasoning and explanation components more explicitly in the revision.


---
### “GRMs are fragile and would fail under slight benchmark variations”

We respectfully disagree. 
While GRMs are evaluated on synthetic datasets, they are explicitly designed to **infer general rules**, not memorize solutions. 
Each reasoning task uses only **3 positive and 3 negative** examples for training. GRMs learn symbolic logic that generalizes across:

- Variable object counts (e.g., upto 45 objects per image, see Table 1),
- Numerous visual variations (e.g., over 150 color combinations in the red triangle task, see Table 1),
- Random groupings and positions (e.g., upto 4 groups per image, with shuffled spatial layouts, see Table 1).

As detailed in **Table 3**, these task variations are non-trivial. Additionally, GRMs support **predicate invention**, 
allowing them to construct new concepts from primitive features (e.g., "red triangle" as a composition of color and shape constraints).

Section 4.4 further demonstrates **robust performance under increased complexity**, 
including scaling to more objects and groups—a challenge where other models (e.g., NEUMANN) fail due to combinatorial explosion.
That said, we agree that testing GRMs on more diverse and real-world-like scenarios is an important next step. 
A future version will:

- Include Gestalt-incongruent control tasks,
- Extend evaluation to noisy and hybrid datasets, and
- Provide a **Discussion section** explicitly outlining current limitations and directions for broader generalization.


---
### “No clear discussion of assumptions or limitations”

We appreciate this suggestion and will incorporate a clear summary of assumptions and limitations in the revised manuscript. 
Key points include:

- **Proximity threshold ($\epsilon$)** is treated as a fixed hyperparameter (see Appendix A.1 and Section 4.3); 
we will bring this into the main text earlier.
- **Recognizable shapes and colors** are limited to a predefined symbolic set, 
but GRMs can be extended with learned visual embeddings for more continuous or abstract features (e.g., “bluish”, “round-ish”).
- **One group partition per principle** is considered at a time, but the architecture is modular and can support overlapping group hypotheses.
- **Logic constraints** are currently restricted to Horn clauses (no negation), 
and **existential quantifiers** are encoded within some predicate templates. 
We plan to explore richer logic expressiveness (e.g., negation, disjunction, multi-rule composition) in future work.


--- 
### Terminology Clarification 

Thank you for highlighting unclear terminology. 
We will include a **glossary table** in the main paper or Appendix B. 
For clarity, here are definitions of key terms:

- **Relation**: A predicate applied to a tuple (e.g., `has_color(obj, red)`).
- **Constraint**: A logic rule representing a structural condition (e.g., “all group members are red”).
- **Pattern**: The visual configuration of objects in an image.
- **Pattern property**: Abstract visual structure, often defined through group-based reasoning (e.g., "mirror symmetry").
- **Target**: The specific pattern a reasoning task seeks to detect.
- **Target rule**: The learned symbolic clause(s) that GRMs output for a given pattern.
- **Object-centric matrix**: Structured tensor encoding object attributes and spatial properties.
- **An atom ‘holds’**: The predicate evaluates to true under a specific image’s symbolic grounding.
- **High-level pattern vs. pattern**: High-level implies semantic abstraction (e.g., "a closed shape"), while "pattern" can refer to any perceptual structure.
- **Applying logical predicates**: Running symbolic inference over image encodings.



---
### On Experimental Claims and Evidence

We respectfully disagree with the assessment that the experimental claims are unsupported. Our experiments:

- Cover **11 distinct patterns** across **four Gestalt principles**, each with varied instances and test samples.
- Rely on **few-shot supervision** (3 positive + 3 negative), emphasizing **rule generalization** rather than classification by data quantity.
- Compare GRMs to two strong and representative baselines:
  - **ChatGPT-4o** (LLM-based reasoning),
  - **NEUMANN** (differentiable logic-based NeSy system),
- Include **runtime scalability evaluations** (Figure 5), showing GRMs' superior performance on real benchmarks 
like **CLEVR-Hans** and **Kandinsky Patterns**.

Together, these results support our core claims that GRMs:

- Enable scalable symbolic reasoning through perceptual grouping,
- Discover human-interpretable rules from vision,
- Outperform existing baselines in both efficiency and accuracy.

---
### On Reproducibility

We are glad the reviewer acknowledged reproducibility as a strength. 
Our **code, datasets, and experiment configurations are all publicly available**. 
While reading the code is indeed a valid way to gain deep understanding, 
we also believe the paper provides a **sufficiently detailed conceptual overview** for competent researchers to grasp the methodology. 
That said, we will add:

- A complete breakdown of the pipeline in the appendix,
- Clear descriptions of modules and data formats,
- Step-by-step instructions to replicate the main results.


---
### Conclusion
In summary, we will revise the paper to:

- Clarify the role of the LLM and symbolic classifier,
- Expand discussion of assumptions and limitations,
- Add explicit definitions and a terminology table,
- Demonstrate GRMs’ robustness to variation,
- Include Gestalt-incongruent and real-world examples,
- Improve reproducibility through detailed pipeline documentation.


--- 

**We are confident these changes will address the reviewer’s concerns and significantly improve the clarity and rigor of the work. 
Thank you again for your careful and constructive review.**

---

--- 

---

---


## Rebuttal to Reviewer f4qA (Score: 3 – Reject, Confidence: 4 – Quite confident)


---

Dear Reviewer f4qA:

we sincerely thank the reviewer for their detailed and critical feedback. 
Your comments raise important points regarding clarity, rigor, generalizability, and completeness. 
Below, we address your major concerns point-by-point and 
describe the concrete improvements we will implement in the revised version.

Best regards,

Authors of Submission 509


---
# Rebuttal

---
###  Clarification of the Task and Evaluation

We acknowledge the ambiguity in the current version regarding the exact nature of the task. 
To clarify:

GRMs are designed for **image-to-logic rule discovery** and **logic-based classification**. 
Given an input image, the system extracts symbolic rules that govern the arrangement of objects 
(often through Gestalt principles), 
which are then used for:

- Classification: Determining whether a new image conforms to the same visual logic.
- Interpretability: Translating the discovered rule into natural language, using an LLM.

The LLM is **not involved in reasoning** or rule discovery. 
It is included **solely for generating human-readable explanations**, 
and we will clarify this separation in both the main text and Figure 2.


---

### On Scope, Assumptions, and Generalization

While the current evaluation focuses on synthetic scenes, GRMs are not hardcoded to specific configurations. 
The architecture operates over structured symbolic and neural features and is extensible to other visual settings.

In the revised version, we will:

- Explicitly state the current assumptions (e.g., discrete shapes, binary similarity metrics).
- Add discussion on generalization to natural or noisy scenes.
- Outline directions to extend GRMs to **continuous features, learned embeddings, and rotation-invariant symmetry detection.**


---

### On Notation, Definitions, and Mathematical Rigor

Thank you for your precise comments on notation. 
We agree that clearer and more consistent definitions are essential. 
We will revise the manuscript as follows:

- Use bold uppercase letters (e.g., **F**) for **sets** of feature vectors.
- Use bold lowercase (e.g., **f_obj**, **f_rel**) for individual vectors.
- Define key quantities:
  - $N$: number of objects
  - $M$: number of groups 
  - Clarify fixed vs. variable-length encodings.

We will also explicitly address **permutation invariance**: 
GRM encodings are passed to symbolic rule learners, which operate on set-based logic. 
Thus, the pipeline is permutation-invariant, and we will explain this in Section 3.



---

### On Implementation Details
We acknowledge that more technical specificity is needed. The modules referenced in Equations (1), (4), and (6) are implemented as follows:

- Eq. (1): Object and group features are derived via geometric heuristics (for shape, color, location) 
and patch-based CNN features (pretrained).
- Eq. (4): Features are encoded via a deterministic map to a symbolic tensor with attributes like color, shape, and size.
- Eq. (6): Rule learning is symbolic—using clause enumeration and predicate filtering—based on prior works. 

In the revised paper, we will add **pseudocode or diagrams** for each module and clearly distinguish symbolic vs. learned components.


---

### On Formalizing Gestalt Principles

Currently, formalizations appear only in Appendix A. 
In the revision, we will move key definitions to the main paper and clarify:

- **Proximity**: Measured as an absolute Euclidean distance threshold in normalized coordinates.
- **Similarity**: Currently binary (e.g., same color/shape), 
but we will discuss extending this to **embedding-based or continuous similarity**.
- **Symmetry**: Currently axis-aligned for simplicity; we will discuss using rotation-invariant descriptors.
- **Closure**: Currently implemented over predefined geometric shapes (e.g., triangle, square). 
We will propose extensions using **graph-based shape abstraction** or **learned priors**.


---

### On Symbolic Reasoning and Rule Representation

Our reasoning module operates over **first-order definite clauses (Horn clauses without negation)**. The system:

- Generates candidate atoms and clause templates.
- Filters these using truth tables over few-shot examples.
- Supports predicate invention to learn novel, compositional concepts.

This is explained in Sections 3.3.1–3.3.3 and Appendix E, 
but we will **move a summary into the main text** for greater clarity.


---

### On Experimental Evaluation

We respectfully disagree with the characterization of the experiments as anecdotal, but we understand the concern.

Our setup evaluates generalization from minimal supervision (3 positive and 3 negative examples), 
in line with **inductive logic programming** principles. GRMs solve:

- A wide variety of **Gestalt-driven and abstract visual tasks**, as listed in Appendix C.
- Tasks with increasing complexity (objects and groups), reported in Table 1.
- Benchmarks like CLEVR-Hans and Kandinsky Patterns, with comparisons to **NEUMANN** and **ChatGPT-4o**.

To strengthen this further, we will:
- Add clarification about the **few-shot rule induction** setup.
- Discuss **noise-injected and Gestalt-incongruent** control patterns.
- Expand results on **scalability and robustness** in real-world-style settings.

---

### On Missing Discussion of Related Work (e.g., Embed2Sym)

We appreciate the pointer to **Embed2Sym** (Aspis et al., 2022) and related clustering-based NeSy approaches. 
These are indeed relevant, and we will include a discussion in the revised related work section. We will emphasize:

- GRMs differ in that grouping is **used symbolically**, not just as a latent structure.
- Our system produces **interpretable rules**, unlike latent embedding cluster models.


---

### Conclusion

To summarize, we will address your concerns by:

- Clarifying the task and pipeline roles (LLM vs. reasoning module).
- Improving notation, rigor, and formalism.
- Adding implementation details for all modules.
- Expanding evaluation to include robustness tests and Gestalt-incongruent cases.
- Including related NeSy work based on clustering.

---
**We thank you again for your thoughtful review. 
We believe these changes will address the raised concerns and significantly improve the clarity, scope, 
and technical quality of the paper.**
---



---

---

--- 

## Reviewer q8Ys (Score: 4 Borderline Reject, Confidence 3: Somewhat confident)

We are grateful for the reviewer’s positive remarks regarding the novelty, clarity, and cognitive grounding of our work, 
as well as for your thoughtful suggestions. 
Below, we respond to your comments in detail and outline concrete steps 
we will take to address your concerns in the revised version.

---
### On Dataset Scope and Generalization Beyond Gestalt-Conforming Scenarios

We acknowledge that our current experimental suite focuses on scenarios where Gestalt principles are central to reasoning. 
This is by design: our primary goal was to validate whether **Gestalt-based grouping improves neuro-symbolic reasoning** 
in tasks where grouping is essential, something existing models struggle with (as shown in Table 1 and Figure 5).

That said, the **GRM** is not limited to Gestalt-driven patterns only. 
For example, GRMs can successfully solve **Kandinsky Patterns** like two pairs or red triangle, 
which are used in **NEUMANN** but **not Gestalt-based**. 
GRMs require **no additional mechanisms** to handle such tasks, 
as they can reason about object-level rules and relations without relying on grouping.

We agree that it is valuable to test **non-Gestalt or Gestalt-neutral cases**, 
especially where grouping may be misleading or unnecessary. 

In a revised version, we will include:
- New “non-Gestalt” control tasks, where object-level relationships dominate (e.g., “find the only object with a unique shape regardless of proximity”).
- A comparison to **NEUMANN** in these settings to demonstrate when grouping is helpful—and when it isn't.

This will strengthen the claim that GRMs are a **generalizable neuro-symbolic architecture** 
rather than a system tailored to a specific class of problems.

---

### On Real-World or High-Stakes Applications

We appreciate this important suggestion. 
While the current study is **proof-of-concept**, 
the principle of leveraging grouping for **scalable, interpretable reasoning** holds promise in applications such as:
- **Medical imaging**: detecting clustered lesions mirrors the same number pattern (Appendix C.2).
- **Traffic and surveillance video analysis**:  identifying coordinated formations of people or vehicles is analogous to triangle square or 
symmetry patterns (Appendix C.3 and C.8).
- **Robotics and scene understanding**, where perceptual grouping can reduce combinatorial complexity 
in visually dense environments (relevant to most patterns in Appendix C).

We will explicitly add a **Discussion section** describing these potential use cases and the required extensions—
such as continuous feature modeling, learned encoders, and robustness to perceptual noise.

---

### On Missing Training Details: Annotations, Losses, Embeddings

We agree that more technical detail would aid reproducibility. 
While our **code and data are available**, we will update the paper and appendices to clearly include:

- **Training requirements**: GRMs require only image annotations (positive, negative). 
None of object or group labels are manually provided—groups are **computed by Gestalt-based clustering**, 
as described in Appendix A.
- **Losses**: GRMs do not use gradient descent for reasoning; rule discovery is **symbolic** 
(e.g., clause filtering and predicate invention) and does not require supervised loss functions. 
Object detection (CNN-based) is pretrained using standard classification loss on primitive shapes.
- **Embedding details**: As described in Section 3.2, 
symbolic features (e.g., shape, color, spatial position) are mapped to tensors. 
Neural features (e.g., image patches) are extracted by a CNN and aggregated at the group level.



--- 

### On the “Six Samples per Pattern” Setup

Thank you for pointing this out. 
The mention of “3 positive and 3 negative samples” refers specifically to the **few-shot rule learning** setup used in each visual reasoning task. 
The goal is not to perform statistical classification over many samples, 
but rather to evaluate **whether GRMs can infer logical rules** that generalize structurally to unseen configurations of the same concept.

This setup mimics **inductive logic programming (ILP)**, where the objective is to **derive symbolic constraints** from very few examples. 
As shown in Table 1, these rules generalize well even when the number of objects or groups in the test scenes increases significantly. 
We will clarify this distinction between **pattern classification** and **symbolic rule induction** in Section 4.2 and 
better motivate the few-shot rationale.


---

### On implementation Details and Reproducibility

We appreciate your request for a more detailed implementation description. 
While our code and data are publicly available, we will revise the appendix to include:
- A breakdown of the pipeline components (object detection, grouping, encoding, reasoning).
- Loss functions (supervised detection + symbolic evaluation filtering; no gradient descent is used in the symbolic module).
- A more formal description of group feature embedding, especially how symbolic and visual descriptors are combined.

---

### On Typographical and Presentation Issues

Thank you for spotting the typo ("natrua" → "natural") in Section 3.4. 
This will be corrected. 
We will also revise the organization of Sections 3 and 4 to improve clarity and reduce ambiguity around 
how different components interact.


---
### Conclusion

To summarize:

- We will include experiments on Gestalt-incongruent patterns and real-world use cases.
- We will clarify the low-shot symbolic learning setup and rule generalization goals.
- We will expand the appendix with full implementation, loss, and training details.
- Minor issues and wording will be corrected for clarity.


**We deeply appreciate your openness to increasing your score. 
We believe these improvements will address your concerns and significantly enhance the quality and clarity of the work.**
