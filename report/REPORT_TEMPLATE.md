# Project Report Template (2500–3000 words)

## Front cover (required)
- Group members + IDs
- Intake code
- Module code + name
- Assignment title
- Date completed (due date)

## Table of contents

## Contribution matrix
- Member, responsibilities, dataset contribution, code modules, report sections.

## Acknowledgement

## Abstract (200–300 words)

## 1. Introduction (problem analysis)
- Glove inspection problem
- Why robustness matters (lighting/background changes)
- Scope: 3 glove types, defect types, GUI

## 2. Methods (description + justification)
### 2.1 Preprocessing
### 2.2 Glove segmentation
### 2.3 Glove type classification (3 materials)
### 2.4 Defect candidate generation
### 2.5 Defect classification (rules/features)
### 2.6 GUI design

## 3. Experimental results
- Dataset summary table
- Train/val/test split method
- Metrics: glove type confusion matrix; defect precision/recall/F1
- Robustness matrix results (lighting/background/distance/orientation)
- Add a curated sanity-check subsection (for example `data/my_test`) with:
  - expected defect from filename
  - predicted defect + score
  - pass/fail note

## 4. Discussion
- What works well and why
- Failure cases (with figures)
- Limitations and mitigation attempts
- Include practical deployment notes:
  - Streamlit startup method used in testing
  - connection-refused troubleshooting steps followed

## 5. Critical comments and future work
- More data, better segmentation, feature learning (without TF), etc.

## 6. Conclusion

## References (APA; scholarly sources only)
