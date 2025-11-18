xyyp

+
#### why do not show the results on CLEVR and RAVEN.

The GRM is built upon its parent model, αILP, and remains fully compatible with all tasks solvable by αILP. Since αILP has demonstrated success on the CLEVR dataset, GRM can equally solve CLEVR with comparable performance, as both models share the same underlying reasoning configuration.

Importantly, GRM extends αILP by introducing dual-level reasoning over both object-level and group-level symbolic structures, enabling it to discover and select the most confident rules across multiple abstraction levels.

Regarding RAVEN, this dataset primarily targets relational and analogy-based abstract reasoning, which lies outside the scope of GRM’s current focus on Gestalt-based perceptual grouping and visual reasoning.

#### Accuracy Improvement Calculation
The accuracy improvement percentages (shown in green) are computed relative to the baseline model without the group reasoning module, indicating the performance gain achieved by incorporating group-level reasoning.
improvement% = (acc_grm - acc_base)/acc_base x 100

#### Runtime Measurement Protocol
The reported runtime is measured from the start of object detection to the completion of rule induction, thereby covering the entire reasoning pipeline.

For vision–language models (VLMs), the runtime is measured from the prompt input to the generation of the final rule output, ensuring consistency across model comparisons.


j3xd

#### evaluation limited to the synthetic scenes:

The synthetic setup is used to allow precise group-truth group structures.
GRM itself is not limited to synthetic data—it operates on symbolic object representations 
and can be directly applied to natural scenes once perceptual grounding is available.

#### theoritical grounding of grouping-pair equations

The contextual affinity is designed to model Gestalt contextual grouping, where the perceived relation between two objects depends not only on their individual features (o_i,o_j) but also on the contextual region between them. The intermediate embedding o_{ij} represents this shared context (e.g., mean embedding or pooled visual features along the connecting path), aligning with perceptual grouping theory that proximity and similarity are modulated by spatial continuity.

(TODO)
Empirically, we performed an ablation removing o_{ij}^{*}, i.e., using only s_p(o_i,o_j)=\sigma(h_p(o_i,o_j)). This led to a 3–5% drop in grouping accuracy across principles, confirming that contextual information contributes significantly to stable group prediction.


#### rule search convergence guarantees and computational complexity
The rule learner performs a bounded top-k beam search over the clause space, where candidate clauses are iteratively expanded for at most T rounds. Here, k denotes the number of highest-scoring clauses retained at each step, and T represents the maximum search depth (i.e., the number of expansion rounds). Since both k and T are fixed, the search converges after at most T iterations.
The search can stop early when a high confident rule has been found.

The overall computational complexity is
\mathcal{O}(|\mathcal{C}|^{T} \cdot t_{\mathcal{C}}),
where |\mathcal{C}| is the number of candidate clauses and t_{\mathcal{C}} is the cost of evaluating a clause, which scales linearly with the number of examples. This bounded search ensures tractable computation and stable convergence to high-confidence rule sets in practice.

In practice, |\mathcal{C}| depends only on the concepts actually present in the scenes, rather than on the full vocabulary of the model’s symbolic language.


#### How sensitive the proposed freamwork to the choice of grouping thresholds s_p?

The grouping module predicts the confidence that two objects belong to the same group. 
A threshold of 0.5 is used, meaning a pair of objects is assigned to the same group if their predicted confidence exceeds 0.5.

(exp is running with other thresholds now)


#### Equation in section Appendix A is not clear: how τ = 0.99 was chosen, for instance?

The threshold τ = 0.99 is used to retain only those rules whose confidence exceeds 99%, meaning the rule is highly consistent with the training examples. In practice, this keeps only the rules that reliably distinguish positive from negative examples.

A higher τ enforces stricter rule selection—covering more positive and fewer negative cases—thus maintaining high precision and interpretability.



cs3A

Weaknesses:
## Lack of detail in 4.1 ‘Pretraining’. From my point of view, the performance of GRMs is largely dependent on the pretrained perception backbones. 
## However, the paper doesn’t include sufficient details (e.g., datasets, objectives, hyperparameters) about this.

(see below)

Unconvincing comparison between GRMs and baselines. Based on W1, actually GRMs’ perception modules are (potentially) benefited from extensive training on images in the same distribution as the evaluation. So the experimental results in the main table cannot fully support such neural-symbolic method can outperform data-driven methods, because the evaluated VLMs may not be familiar with the test images. I would wonder whether the GRMs’ performance is still superior to VLMs after they are post-trained using the same dataset.
Limited generalization potential: GRMs' framework relies on predefined predicates and simplified group patterns, and cannot be used in processing real-world images.
Questions:
Could the authors provide more reasons/evidence to support that such a neuro-symbolic method is better than large-scale data-driven training?




QuEQ

## Reviewer Question: Pretrained perception backbone details.Are attributes values and/or groups provided as training labels? 
## If yes, this limitation should be clearly stated and addressed with an end-to-end learning approach.

There are two perception backbones:

The object model structure:
The object model is a two-layer MLP: 
it flattens input patches, maps them to 128 hidden units with ReLU, then outputs class scores.

The input patches are extracted by identifying the contours of regions in the image. 
Each patch corresponds to the area enclosed by a detected contour, 
capturing the local visual information of that specific region. 
These contour-based patches are then flattened and used as input to the object model for classification or embedding. 
This approach ensures that the model processes meaningful, object-centric regions rather than arbitrary image crops.

The group model is trained in an end-to-end supervised manner.
The dataset used for training the grouping model is 
synthetically generated based on the ELVIS pattern to provide ground-truth groupings based on Gestalt principles.
The model takes the object embeddings as input and predicts the probability that two objects belong to the same group,

The group model structure:
A point encoder: a two-layer MLP with ReLU, mapping each input point to a hidden dimension.
A patch encoder: a two-layer MLP with ReLU, mapping a flattened set of encoded points (a patch) to a patch embedding.
A classifier: a two-layer MLP with ReLU, taking the concatenation of two contour embeddings and a context embedding, and outputting a single logit.
The forward pass encodes two input contours and their context, concatenates their embeddings, and passes them through the classifier to produce a score.


**Weak grouping performance. As shown in Table 4, the grouping accuracy is very low. 
Given that the main contribution of this paper is the grouping, it should propose and validate enhancements. 
First, one can question if a pure neural approach (MLP) without any inductive bias is suitable for this quite involved task.** 

The grouping task is intentionally challenging, as it requires capturing complex Gestalt principles that go beyond simple feature similarity.
The proposed grouping model shows the accuracy upto 76% and enhance the reasoning performance over ELVIS dataset in range of 11~61% across different tasks.
Thus we are able to show that with a grouping model, even with imperfect accuracy, we can significantly improve reasoning performance.
The main idea of this paper is not propose a high-end grouping model, but to show that incorporating group-level reasoning can enhance visual reasoning performance.
We design the grouping model as a simple MLP to demonstrate that even a basic neural approach can yield substantial benefits when integrated into the neuro-symbolic reasoning framework.
We believe that more sophisticated architectures (e.g., graph neural networks or transformers) could further improve grouping accuracy, which we leave for future work.


**Moreover, the group-level perception (Section 3.1) accumulates all embeddings from the global context into one embedding. 
This averaging can certainly face some capacity limit. Having a more scalable approach that allows for concatenation (e.g., a Transformer) may improve the approach.** 

Thanks for your insightful suggestion.  The mean pooling is used to provide a lightweight and permutation invariant summary of the neighborhood objects,
which keeps the grouping module stable across different number of objects. That is the main concern we had when designing the grouping model.

We agree that more expressive architectures like Transformers could capture richer global relationships and further improve grouping.
The group model can be individually upgraded by any other advanced architecture as well, we use simple MLPs to demonstrate the core idea of incorporating group-level reasoning.




## Finally, prompting foundational models (e.g., GPT-5) to perform the grouping could be considered, too.


## The timing measurements are missing the neuro-symbolic baselines without group-level information (NEUMANN). 

Give the NEUMANN baseline time measurements

However, .... We didn't show the NEUMANN baseline in the timing measurements since it doesn't achieve high accuracy in the tasks,
as shown in the Table 2. The time comparison only includes models that achieve reasonable accuracy such as InterVL3-78B and GPT-5.

## Moreover, the hardware should be specified for the different methods.
We would like to add this information in the final version.
The GRM is runable on a personal laptop with a decent GPU (e.g., NVIDIA RTX 2080), 
whereas the large VLMs (InterVL3-78B and GPT-5) require high-end servers with multiple GPUs due to their size.
In our experiments, GRM on a NVIDIA A100-SXM4-40GB with 18% (36% peak) GPU Utilization in average
We ran InterVL3-78B on 3 NVIDIA A100-SXM4-80GB with 40% average (100% peak) GPU Utilization;
GPT-5 via API, so we don't have the hardware details.


## In which real-world applications is grouping needed?
talk about the games, objects can be grouped based on their functions or spatial arrangements,









Thanks for your questions and review! We answer the questions below.

### Q1. Details about supervision of perception backbone. 

**Object Model**
The object model is a two-layer MLP:
it flattens input patches, maps them to 128 hidden units with ReLU, then outputs class scores.
The input patches are extracted by identifying the contours of regions in the image.

**Group Model** 
The group model is trained in an end-to-end supervised manner.
The dataset used for training the grouping model is synthetically generated based on the ELVIS pattern to provide ground-truth groupings based on Gestalt principles. 
The model takes the object embeddings as input and predicts the probability that two objects belong to the same group.

group model structure:
A point encoder: a two-layer MLP with ReLU, mapping each input point to a hidden dimension.
A patch encoder: a two-layer MLP with ReLU, mapping a flattened set of encoded points (a patch) to a patch embedding.
A classifier: a two-layer MLP with ReLU, taking the concatenation of two contour embeddings and a context embedding, and outputting a single logit.
The forward pass encodes two input contours and their context, concatenates their embeddings, and passes them through the classifier to produce a score.

## Q2: Weak grouping performance.

The grouping task is intentionally challenging, as it requires capturing complex Gestalt principles that go beyond simple feature similarity.
The proposed grouping model shows the accuracy upto **76%** and enhance the reasoning performance over ELVIS dataset in range of **11~61%** across different tasks.
Thus we are able to show that with a grouping model, even with imperfect accuracy, we can significantly improve reasoning performance.

The main idea of this paper is not to propose a high-end grouping model, but to show that incorporating group-level reasoning can enhance visual reasoning performance.
We design the grouping model as a simple MLP to demonstrate that even a basic neural approach can yield substantial benefits when integrated into the neuro-symbolic reasoning framework.
We believe that **more sophisticated architectures** (e.g., graph neural networks or transformers) could further improve grouping accuracy, which we leave for future work.

## Q3: Average embedding from the global context can face some capacity limit. Transformer can do a better job.

Thanks for your insightful suggestion.  The mean pooling is used to provide a lightweight and permutation invariant summary of the neighborhood objects,
so the grouping module stable across different number of objects. That is the main concern we had when designing the grouping model.

We agree that more expressive architectures like Transformers could capture richer global relationships and further improve grouping. 
We use simple MLPs to demonstrate the core idea.

## Q4: Prompting foundational models (e.g., GPT-5) to perform the grouping could be considered, too.

Using more advanced models such as GPT-5 can be a promising future work.

### Q5: The timing measurements of NEUMANN.

In Table 2, the time comparison only includes models that achieve reasonable accuracy such as InterVL3-78B and GPT-5. 
We would like to add the timing measurements for NEUMANN into the table. The full table is shown below:

| Model              | Proximity | Similarity | Closure | Symmetry | Continuity | Average |
|--------------------|-----------|------------|---------|----------|------------|---------|
| GRM (Average)      | 6.28      | 88.76      | 10.29   | 34.42    | 9.32       | 29.814  |
| GPT-5 (Average)    | 109.45    | 94.34      | 105.46  | 157.56   | 82.59      | 109.88  |
| InterVL3-78B       | 14.11     | 13.34      | 15.58   | 12.30    | 14.89      | 14.044  |
| NM                 | 4.34      | 48.45      | 6.80    | 24.48    | 6.34       | 18.082  |

### Q6: Moreover, the hardware should be specified for the different methods.

The GRM is runnable on a personal laptop with a decent GPU (e.g., a mac book pro with M2 chip),
whereas the large VLMs (InterVL3-78B and GPT-5) require high-end servers with multiple GPUs due to their size.
In our experiments,  GRM ran on a NVIDIA A100-SXM4-40GB with 18% (36% peak) GPU Utilization in average
We ran InterVL3-78B on 3 NVIDIA A100-SXM4-80GB with 40% average (100% peak) GPU Utilization
We use GPT-5 via API, not clear about the hardware details.


### Q7: Grouping applications in the real-world?
Grouping becomes essential in the multiple object reasoning scenarios.

Example 1, in household or robotic planning (“clean up the after-party room”), 
a cluster of used paper cups can be considered one group and assigned to a single action in the cleaning schedule. 

Example 2, in off-road scenes such as a forest, individual trees do not indicate where a vehicle can safely pass. 
But when viewed as a group, trees forming a line, especially with a parallel symmetry line, 
it provides essential structural cues for navigation.

We are open to further discuss and clarify any other questions!