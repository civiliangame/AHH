---
license: apache-2.0
language:
- en
base_model:
- openai/whisper-small.en
pipeline_tag: audio-classification
---


# Background

In the United States nearly 21M adults suffer from depression each year [1], with depression serving as the nation’s leading cause of disability [2]. 
Despite this, less than 4% of Americans receive mental health screenings from their primary care physicians during annual wellness visits.
The pandemic and public campaigns of late have made strides toward positively increasing awareness of mental health struggles, but there remains a persisting stigma around depression and other mental health conditions. 
The influence of this stigma is especially marked in older adults. People aged 65 and older are less likely than any other age group to seek mental health support. 
Older adults – for whom depression significantly increases the risk of disability and morbidity – also tend to underreport mental health symptoms [3].

In the US, this outlook becomes even more troubling when coupled with the rate at which the country’s population is aging: 1 out of every 6 people will be 60 years or over by 2030 [4]. 
As widespread and prevalent as depression is, identifying and treating depression and other mental health conditions remains challenging and there is limited objectivity in the screening processes.


# Depression–Anxiety Model (DAM)

## Model Overview

DAM is a clinical-grade, speech-based model designed to screen for signs of depression and anxiety using voice biomarkers.
To the best of our knowledge, it is the first model developed explicitly for clinical-grade mental health assessment from speech without reliance on linguistic content or transcription. A predecessor model has been peer-reviewed in the largest voice biomarker study by the Annals of Family Medicine, a leading U.S. Primary Care Journal [5].
The model operates exclusively on the acoustic properties of the speech signal, extracting depression- and anxiety-specific voice biomarkers rather than semantic or lexical information.
Numerous studies [6–8] have demonstrated that paralinguistic features – such as spectral entropy, pitch variability, fundamental frequency, and related acoustic measures – exhibit strong correlations with depression and anxiety.
Building on this body of evidence, DAM extends prior approaches by leveraging deep learning to learn fine-grained vocal biomarkers directly from the raw speech signal, yielding representations that demonstrate greater predictive power than hand-engineered paralinguistic features.
DAM analyzes spoken audio to estimate depression and anxiety severity scores which can be subsequently mapped to standardized clinical scales, such as **PHQ-9** (Patient Health Questionnaire-9) for depression and **GAD-7** (Generalized Anxiety Disorder-7) for anxiety.


## Data

The model was trained and evaluated on a large-scale speech dataset collected from approximately 35,000 individuals via phone, tablet, or web app, which corresponds to ~863 hours of speech data.
Ground-truth labels were derived from both clinician-administered and self-reported PHQ-9 and GAD-7 questionnaires, ensuring strong alignment with established clinical assessment standards.
The data consists predominantly of American English speech. However, a broad range of accents is represented, providing robustness across diverse speaking styles.

The audio data itself cannot be shared for privacy reasons. Demographic statistics, model scores, and associated metadata for each audio stream are available for threshold tuning at https://huggingface.co/datasets/KintsugiHealth/dam-dataset.


## Model Architecture

**Foundation model:** OpenAI Whisper-Small EN

**Training approach:** Fine-tuning + Multi-task learning

**Downstream tasks:** Depression and anxiety severity estimation

Whisper serves as the backbone for extracting voice biomarkers, while multi-task head is fine-tuned jointly on depression and anxiety prediction tasks to leverage shared representations across mental health conditions.

## Input Requirements

**Preferred minimum audio length:** 30 seconds of speech after Voice Activity Detector

**Input modality:** Audio only

Shorter audio samples may lead to reduced prediction accuracy.

## Output

The model outputs a dictionary of the following form `{"depression":score, "anxiety": score}`.

If `quantize=False` (see the Usage section below), the scores are returned as raw float values on arbitrary scales which correlate monotonically with PHQ-9 for depression and GAD-7 for anxiety.
For reference the cumulative distribution of scores for each label sum is plotted below.

![score_phq](https://cdn-uploads.huggingface.co/production/uploads/64b8497349bde5d94815adf7/WEL01JNdj0zOTIEDx8KJQ.png)

![score_gad](https://cdn-uploads.huggingface.co/production/uploads/64b8497349bde5d94815adf7/605obGq_k-h_E-gxuXoxh.png)

If `quantize=True` the scores are converted into integers representing the severity of depression and anxiety.

**Quantization levels for depression task:**

0 – no depression (PHQ-9 <= 9)

1 – mild to moderate depression (10 <= PHQ-9 <= 14)

2 – severe depression (PHQ-9 >= 15)


**Quantization levels for anxiety task:**

0 – no anxiety (GAD-7 <= 4)

1 – mild anxiety (5 <= GAD-7 <= 9)

2 – moderate anxiety (10 <= GAD-7 <= 14)

3 – severe anxiety (GAD-7 >= 15)

## Intended Use
* Mental health research
* Clinical decision support
* Continuous monitoring of depression and anxiety

## Limitations
* Not intended for diagnosis/self-diagnosis without clinical oversight
* Performance may degrade on speech recorded outside controlled environments or in the presence of noise
* Intended only for audio containing a single voice speaking English
  * Biases related to language, accent, or demographic representation may be present


# Usage
1. Checkout the repo:

Make sure the git LFS or XET extensions are installed so that the model checkpoint itself will be downloaded instead of a pointer to it. 

```
git clone https://huggingface.co/KintsugiHealth/dam
```

2. Install requirements:

Install the `mamba` package manager, then run

```
mamba env create -n dam -f requirements.txt -c conda-forge -c pytorch -c nvidia

```

to create the environment and

```
mamba activate dam
```

to activate it.

3. Load and run pipeline
```python
from pipeline import Pipeline

pipeline = Pipeline()
result = pipeline.run_on_file("sample.wav", quantize=True)
print(result)
```
The output will resemble a dictionary, for example {'depression': 2, 'anxiety': 3}, indicating that the analyzed audio sample exhibits voice biomarkers consistent with severe depression and severe anxiety.

## Tuning Thresholds
As mentioned in the Data section above, the raw audio data cannot be shared, but validation and test sets of model scores associated with ground truth and demographic metadata are available for threshold tuning. This way thresholds can be tuned for traditional binary classification, ternary classification with an indeterminate output, and multi-class classification of severity. Two modules are provided for this in the model code's `tuning` package, as illustrated below.

### Tuning Sensitivity, Specificity, and Indeterminate Fraction
This module implements a generalization of ROC curve analysis wherein ground truth is binary, but model output can be negative (score below lower threshold), positive (score above upper threshold), or indeterminate (score between thresholds). For the purpose of metric calculations such as sensitivity and specificity, examples marked indeterminate do not count towards either the numerator or denominator. The budget for fraction of examples to be marked indeterminate is configurable as shown below.
```python
import numpy as np

from datasets import load_dataset
from tuning.indet_roc import BinaryLabeledScores

val = load_dataset("KintsugiHealth/dam-dataset", split="validation")
val.set_format("numpy")
test = load_dataset("KintsugiHealth/dam-dataset", split="test")
test.set_format("numpy")

data = dict(val=val, test=test)

# Associate depression model scores with binarized labels based on whether the PHQ-9 sum is >= 10
scores_labeled = {
    k: BinaryLabeledScores(
        y_score=v['scores_depression'], # Change to 'scores_anxiety' to calibrate anxiety thresholds
        y_true=(v['phq'] >= 10).astype(int) # Change to 'gad' to calibrate anxiety thresholds; optionally change cutoff
    )
    for k, v in data.items()
}

issa = scores_labeled['val'].indet_sn_sp_array() # Metrics at all possible lower, upper threshold pairs

# Compute ROC curve with 20% indeterminate budget and select a point near the diagonal
roc_at_20 = issa.roc_curve(0.2) # Pareto frontier of (sensitivity, specificity) pairs with at most 20% indeterminate fraction
print(f"Area under the ROC curve with 20% indeterminate budget: {roc_at_20.auc()=:.1%}") #
sn_eq_sp_at_20 = roc_at_20.sn_eq_sp() # Find where ROC comes closest to sensitivity = specificity diagonal
print(f"Thresholds to balance sensitivity and specificity on val set with 20% indeterminate budget: "
      f"{sn_eq_sp_at_20.lower_thresh=:.3}, {sn_eq_sp_at_20.upper_thresh=:.3}")
print(f"Performance on val set with these thresholds: {sn_eq_sp_at_20.sn=:.1%}, {sn_eq_sp_at_20.sp=:.1%}") #
test_metrics = sn_eq_sp_at_20.eval(**scores_labeled['test']._asdict()) # Thresholds evaluated on test set
print(f"Performance on test set with these thresholds: {test_metrics.sn=:.1%}, {test_metrics.sp=:.1%}") #

# Find best specificity given sensitivity and indeterminate budget constraints
constrained = issa[(issa.sn >= 0.8) & (issa.indet_frac <= 0.35)]
optimal = constrained[np.argmax(constrained.sp)]
print(f"Highest specificity achievable with sensitivity >= 80% and 35% indeterminate budget is "
      f"{optimal.sp=:.1%}, achieved at thresholds {optimal.lower_thresh=:.3}, {optimal.upper_thresh=:.3}"
)

# Collect optimal ways of achieving balanced sensitivity and specificity as a function of indeterminate fraction
sn_eq_sp = issa.sn_eq_sp_graph()
```

### Optimal Tuning for Multi-class Tasks
The depression and anxiety models were each trained with ordinal regression to predict a scalar score monotonically correlated with the underlying PHQ-9 and GAD-7 questionnaire ground truth sums. As such there are efficient dynamic programming algorithms to select optimal thresholds for multi-class numeric labels under a variety of decision criteria.

```python
from datasets import load_dataset
from tuning.optimal_ordinal import MinAbsoluteErrorOrdinalThresholding

val = load_dataset("KintsugiHealth/dam-dataset", split="validation")
val.set_format("torch")
test = load_dataset("KintsugiHealth/dam-dataset", split="test")
test.set_format("torch")

data = dict(val=val, test=test)

scores = val['scores_anxiety']  # Change to 'scores_depression' for depression threshold tuning
labels = val['gad']  # Change to 'phq' for depression threshold tuning; optionally change to quantized version for coarser prediction tuning

# Can change to any of
# `MaxAccuracyOrdinalThresholding`
# `MaxMacroRecallOrdinalThresholding`
# `MaxMacroPrecisionOrdinalThresholding`
# `MaxMacroF1OrdinalThresholding`
optimal_thresh = MinAbsoluteErrorOrdinalThresholding(num_classes=int(labels.max()) + 1)
best_constant_cost, best_constant = optimal_thresh.best_constant_output_classifier(labels)
print(f"Always predicting GAD sum = {best_constant} on val set independent of model score gives mean absolute error {best_constant_cost:.3}.")
mean_error = optimal_thresh.tune_thresholds(labels=labels, scores=scores)
print(f"Thresholds optimized on val set to predict GAD sum from anxiety score: {optimal_thresh.thresholds}")
print(f"Mean absolute error predicting GAD sum on val set based on thresholds optimized on val set: {mean_error:.3}")
test_preds = optimal_thresh(test['scores_anxiety'])
mean_error_test = optimal_thresh.mean_cost(labels=test['gad'], preds=test_preds)
print(f"Mean absolute error predicting GAD sum on test set based on thresholds optimized on val set: {mean_error_test:.3}")
```

# Acknowledgments

This model was created through equal contributions by Oleksii Abramenko, Noah Stein, and Colin Vaz during their work at Kintsugi Health. It builds on years of prior modeling, data collection, clinical research, and operational efforts by a broader team. A full list of contributors is available on the Kintsugi Health organization card at https://huggingface.co/KintsugiHealth.

# References

1. https://www.nimh.nih.gov/health/statistics/major-depression  
2. https://www.hopefordepression.org/depression-facts/  
3. https://nndc.org/facts/  
4. https://www.psychiatry.org/patients-families/stigma-and-discrimination
5. https://www.annfammed.org/content/early/2025/01/07/afm.240091
6. https://www.sciencedirect.com/science/article/pii/S1746809423004536
7. https://pmc.ncbi.nlm.nih.gov/articles/PMC3409931/
8. https://pmc.ncbi.nlm.nih.gov/articles/PMC11559157