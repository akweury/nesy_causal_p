# ✅ Manuscript Revision Checklist for GRM Paper

## 🔧 Technical Clarifications and Additions

- [ ] **Explain CNN-based visual descriptors (Section 3.2)**
  - Describe local patch extraction around keypoints
  - Explain similarity-based matching via feature memory bank
  - Emphasize robustness to occlusion and generalization beyond symbolic features

- [ ] **Modular pipeline: symbolic vs. visual features**
  - Clarify that symbolic features are used in current setup
  - Explain that the same architecture supports learned visual features
  - Note this extensibility in both text and discussion section

- [ ] **Clarify object detection and position estimation (Section 3.1)**
  - Segmentation = connected regions of color
  - Position = centroid of each region (no bounding boxes)
  - Color, size, shape extraction methods clearly described
  - Add a "Model & Data Requirements" subsection summarizing this

- [ ] **Clarify six-sample training setup (Section 4.3)**
  - Explain how negative samples are structurally similar but incomplete
  - Describe random selection from a diverse pool
  - Emphasize intentional minimal supervision for testing generalization

- [ ] **Explain why full-dataset rule learning is not used**
  - Reduces search difficulty and increases runtime
  - Mention potential future extension using MDL or probabilistic scoring
  - Connect to ILP concepts (Muggleton & De Raedt, 1994)

---

## 📊 Experiments & Evaluation

- [ ] **Add accuracy results for CLEVR-Hans and Kandinsky Patterns**
  - Include GRM, ProbLog, Grouping-enhanced ProbLog
  - Update Figure 5 or add a new results table
  - Clarify that NEUMANN is excluded due to OOM errors

- [ ] **Optional: Add memory usage comparison with NEUMANN**
  - Provide simple runtime/memory table for smaller examples

---

## 📚 Related Work and Positioning

- [ ] **Expand Related Work section**
  - Compare to DeepProbLog, NEUMANN, Embed2Sym, ILP methods
  - Highlight novelty of group-level symbolic reasoning in GRMs

---

## 📄 Writing & Structure Improvements

- [ ] Fix typo: `scalr` → `scalar` (bottom of page 4)
- [ ] Reorganize Sections 3 and 4 for clarity
  - Add "Model and Data Requirements" subsection
  - Clearly separate perception vs. reasoning components

- [ ] Add discussion of future cognitive alignment
  - Mention potential comparison of GRMs to human perceptual responses

---

## ✅ Optional Enhancements

- [ ] Add visual schematic or appendix figure
  - Show segmentation → symbolic conversion pipeline
  - Example of 3 positive / 3 negative sample configuration
