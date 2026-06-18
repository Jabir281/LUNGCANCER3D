# LungCancer3D — Thesis Defense Speech (15 Minutes, 5 Members)

---

## MEMBER 1 

### Slide 1: Title Slide

Good morning, respected panel members. I am Sadit, and on behalf of my team, I welcome you to our thesis defense presentation titled "LungCancer3D: A Hybrid 3D CNN Framework with Explainable AI for Lung Nodule Malignancy Classification."

Lung cancer remains one of the most fatal malignancies worldwide, responsible for approximately 1.8 million deaths annually — that is nearly one in five cancer-related deaths globally. The single most important factor determining a patient's survival is the stage at which the cancer is detected. When caught early, the five-year survival rate exceeds 60 percent. However, when diagnosed at an advanced stage, that rate plummets to below 10 percent. This stark disparity is the fundamental motivation behind our work.

### Slide 2: Problem Statement & Objectives

The current clinical standard for lung cancer screening is low-dose computed tomography, or LDCT. However, radiologists face an immense challenge: a single CT scan produces hundreds of axial slices, and they must search each one for small, subtle nodules that may be early indicators of malignancy. Fatigue, cognitive overload, and the sheer volume of data inevitably lead to missed diagnoses.

Existing computer-aided diagnosis systems have made progress, but most rely on two-dimensional CNNs that analyze CT slices independently. This approach fundamentally discards the volumetric spatial information that is crucial for accurate tumour characterization. A nodule is a three-dimensional object — it grows across slices, it has volume, and its morphology in three dimensions is what determines malignancy. A 2D model cannot capture this.

Additionally, these deep learning models operate as black boxes. They produce a prediction — malignant or benign — but offer no explanation for their decision. This lack of transparency is a major barrier to clinical adoption. No radiologist will trust a diagnosis they cannot verify.

Our research therefore addresses three core problems: first, the loss of volumetric understanding in 2D CNNs; second, the lack of interpretability in existing deep learning models; and third, the high rate of false positives and false negatives that erode clinical trust.

Our objectives were fourfold: to develop 3D CNN models that utilise volumetric CT data; to enhance diagnostic precision through hybrid multi-backbone ensembles; to integrate Explainable AI using Grad-CAM++ visualisation; and to systematically evaluate Kolmogorov-Arnold Networks against traditional MLP classifiers across frozen and scratch training regimes.

### Slide 3: Methodology Overview

Let me provide a brief overview of our methodology before my colleagues dive into the details. We used the LUNA16 dataset, a curated subset of LIDC-IDRI containing 888 CT scans with expert annotations. The data was preprocessed into 64 by 64 by 64 volumetric patches at one millimetre isotropic resolution, with hard-negative sampling at a five-to-one ratio to balance the classes.

We then constructed 11 distinct model configurations: three standalone 3D CNN backbones — ResNet-18, DenseNet-121, and EfficientNet-B0 — as baselines; and eight hybrid ensemble configurations formed by concatenating features from either two or three backbones and passing them through either an MLP or a KAN classification head. Each hybrid was trained in both a frozen regime — where backbones are fixed and only the head is trained — and a scratch regime where all parameters are fine-tuned end-to-end.

Finally, we applied Grad-CAM++ with a confidence-based zeroing rule to generate three-dimensional spatial attention maps that directly reflect the model's binary decision. I will now hand over to my colleague Topu, who will walk you through the dataset and preprocessing pipeline in detail.

---

## MEMBER 2 

### Slide 4: Dataset & Preprocessing Pipeline

Thank you, Sadit. I am Topu, and I will now present the dataset and our preprocessing methodology.

We used the LUNA16 benchmark dataset, which is a curated subset of the larger LIDC-IDRI database containing 1,018 CT scans collected from seven academic institutions. The LUNA16 challenge selected 888 scans meeting strict quality criteria — slice thickness less than three millimetres and complete lung coverage. Each nodule in the dataset is annotated by four expert radiologists and assigned a malignancy score from one to five. We binarised these ratings at a consensus threshold of three or above to produce high-quality ground truth labels.

Now, let me explain the preprocessing pipeline in detail, because this is where the foundation for our results was laid.

First, raw CT scans from different institutions have varying voxel spacing — typically ranging from 0.5 to 1.5 millimetres in the axial plane and 0.6 to 5.0 millimetres in the slice thickness direction. This heterogeneity is catastrophic for 3D CNNs because convolutional kernels assume spatially consistent distances between voxels. A three-by-three-by-three kernel operating on a scan with 2.5-millimetre slice spacing covers entirely different anatomical context than the same kernel on a one-millimetre scan. To address this, we resampled all volumes to one-millimetre isotropic resolution using third-order spline interpolation. This resolution was chosen as a deliberate compromise between anatomical detail and memory footprint. At one millimetre, even nodules as small as four to six millimetres — the most clinically critical category — are represented by at least four-by-four-by-four voxels, which is sufficient for a 3D CNN to extract meaningful features.

Second, we applied Hounsfield Unit normalisation. CT intensity values were clipped to the range of negative 1000 to positive 400 Hounsfield Units, then linearly normalised to the zero-to-one range. The clipping range was chosen based on the physical properties of lung anatomy: negative 1000 HU corresponds to air — the dominant substance in healthy lung tissue — while positive 400 HU corresponds to dense cortical bone. Lung nodules typically fall in the range of negative 100 to positive 200 HU, with malignant nodules often exhibiting higher density due to irregular cell growth and micro-calcifications. By clipping to this range, we preserve the full diagnostic range while removing irrelevant outliers such as metal implants or CT table padding that could dominate the network's gradient signal.

### Slide 5: Patch Extraction, Hard-Negative Mining & Data Splitting

The third and most critical preprocessing step is patch extraction and hard-negative sampling. From each resampled volume, we extracted 64-by-64-by-64 voxel patches around the annotated nodule centroids. Why 64 cubed? We tested smaller and larger sizes. A 32-by-32-by-32 patch truncates the spatial context needed to differentiate spherical nodules from cylindrical blood vessels, while 96-by-96-by-96 exceeds GPU memory budget for a reasonable batch size. At one millimetre isotropic resolution, a 64-by-64-by-64 patch represents a physical volume of approximately 6.4 cubic centimetres, which comfortably contains nodules up to 30 millimetres in diameter plus a margin of surrounding lung parenchyma for spatial context.

Now, the hard-negative mining strategy deserves special attention. The raw LUNA16 annotation set contains approximately 1.5 million candidate centroids — most of which are false positives from a candidate detection algorithm. Of these, fewer than 2,000 are actual malignant nodules. Without hard-negative sampling, a model trained on random patches would see a ratio of approximately one positive to 750 negatives, and could trivially achieve 99.9 percent accuracy by predicting negative for every single patch — while having zero clinical utility because every malignancy would be missed. We enforced a strict five-to-one negative-to-positive ratio, independently per patient. This ensures that each training batch contains enough positive examples for the gradient to reflect malignancy-specific features, while still exposing the model to a sufficient number of negatives to learn false-positive suppression.

We also implemented spatial deduplication: negative candidates within a 10-millimetre Euclidean distance of each other were removed to prevent multiple near-identical negative patches from the same anatomical region. Without this step, the effective diversity of the training set would be artificially reduced.

Finally, we split the data at the patient level — all patches from a single patient belong to exactly one split. This is critical because nodule-level or patch-level splits can inadvertently leak information: two patches from the same patient share the same CT acquisition parameters and scanner characteristics, and a model that memorises patient-specific features can achieve artificially high validation scores while failing on held-out patients. Our final split comprised 822 positive and 5,115 negative training patches; 192 positive and 1,175 negative validation patches; and 172 positive and 1,075 negative test patches. At the patient level — which is the clinically meaningful unit — the test set contains 134 patients: 91 with malignant nodules and 43 without. Zero patient-level UID overlap was confirmed by automated audit scripts.

I will now hand over to Irtiza, who will explain our model architecture decisions in detail.

---

## MEMBER 3 

### Slide 6: Why 3D CNNs Instead of Transformers?

Thank you, Topu. I am Irtiza, and I will now present our model architecture design and the reasoning behind each key decision.

A natural question that arises is: why did we choose convolutional architectures instead of transformer-based models, which have achieved state-of-the-art results in 2D image classification and are increasingly applied to medical imaging? The answer is rooted in both theoretical reasoning and empirical evidence, and I want to present both.

We trained a 3D Vision Transformer — specifically, a ViT-Small — on our identical dataset with the same preprocessing and hyperparameters. The results were stark: the 3D ViT achieved a validation AUC of only 0.7089 and an F1 score of just 0.5797. To put that in perspective, a specificity of zero — meaning it classified all validation samples as positive. In contrast, our 3D ResNet-18 achieved an AUC of 0.9997 and an F1 of 0.9730 on the same validation set. The ViT failed for three fundamental reasons.

First, transformers require massive datasets to learn positional encoding and self-attention mechanisms from scratch. With only 822 positive training patches available, the ViT simply does not have enough data to amortise its 22 million parameters. Second, the self-attention mechanism's global receptive field is actually a disadvantage for small-object detection. A 64-by-64-by-64 patch contains approximately 262,000 voxels, and the ratio of informative tumour voxels — typically 1,000 to 8,000 for a 6 to 12 millimetre nodule — to background lung tissue is extremely low. This causes the attention weights to diffuse across irrelevant regions. Third, CNNs have an inductive bias toward local feature hierarchies — edges to textures to shapes to objects — that aligns naturally with how radiologists interpret scans. Transformers must learn this hierarchy from data, and our dataset is too small for that.

We also tested MedNext, a hybrid CNN-transformer architecture, which achieved a validation AUC of 0.9788 and F1 of 0.9405. While better than pure ViT, it still underperformed our CNN-based ensembles, which reached a perfect validation AUC of 1.0000. For all these reasons, we selected CNN-based architectures as the foundation for this study.

### Slide 7: Standalone Backbones & Hybrid Ensemble Design

We selected three pre-trained 3D CNN backbones representing different architectural families. First, 3D ResNet-18 — an 18-layer residual network with skip connections that mitigate the vanishing gradient problem. With only 11 million parameters, it is computationally efficient and has a strong track record in medical imaging. Its relatively shallow depth is actually appropriate for 64-by-64-by-64 patches, because stacking many downsampling layers would reduce spatial feature maps to just two-by-two-by-two before the classifier.

Second, 3D DenseNet-121, which connects each layer to every subsequent layer via dense skip connections. With 121 layers but only approximately 8 million parameters, it is parameter-efficient — each layer contributes only 12 new feature maps while reusing all preceding ones. This was expected to benefit our small-data regime.

Third, 3D EfficientNet-B0, which uses MBConv blocks with squeeze-and-excitation optimisation. It achieved the best validation metrics — a perfect AUC of 1.0000 and F1 of 0.9945 — during preliminary experiments. However, as we will see in the results, validation metrics can be deceptive.

Now, the central architectural contribution of our work is the hybrid ensemble design. The rationale is rooted in feature diversity. When two backbones are pre-trained on ImageNet, they develop different internal representations. ResNet-18 emphasises residual shortcuts that preserve gradient flow; DenseNet-121 focuses on dense feature reuse; EfficientNet-B0 optimises the width-depth-resolution trade-off. By concatenating their final-layer feature embeddings, our hybrid classifier receives multiple independent opinions about each patch — analogous to a radiologist seeking a second opinion. On a dataset with only 822 positive training samples, this feature diversity is more valuable than additional model capacity, because the model cannot overfit to a single backbone's spurious correlations if a second backbone disagrees.

We chose two and three backbones as our ensemble sizes for a practical reason: training three backbones from scratch approaches the memory capacity of a single A100 GPU. Four or more backbones would require distributed training. Two and three backbones represent the sweet spot where ensemble diversity is meaningful without exceeding single-GPU constraints.

### Slide 8: KAN vs MLP Heads & Frozen vs Scratch Training

The concatenated backbone features are passed to one of two classifier heads. The MLP head consists of two fully-connected layers — 512 to 256 to 1 — with ReLU activation and dropout at probability 0.3. It has approximately 130,000 trainable parameters when backbones are frozen.

The KAN head — Kolmogorov-Arnold Network — replaces the linear layers with learnable B-spline basis functions. In a KAN layer, each activation is a learned univariate spline rather than a fixed nonlinearity like ReLU. This gives KAN two theoretical advantages: it can model complex non-linear feature interactions without increasing layer width, because each spline can adapt its shape to the local data distribution; and it provides a smooth, differentiable decision boundary. The KAN head has approximately 500,000 parameters in our implementation.

Each combination was trained in two regimes. In the frozen regime, all backbone parameters have requires_grad set to False, and only the classification head is optimised. This transforms the hybrid into a feature-ensemble classifier where backbones act as fixed feature extractors. Convergence happens in one to four epochs. In the scratch regime, all parameters are jointly fine-tuned end-to-end, allowing backbones to adapt to the lung nodule domain. However, this introduces 10 to 22 million trainable parameters, dramatically increasing overfitting risk.

The decision to test both regimes was motivated by a gap in the literature: prior work on frozen backbones in medical imaging had focused on single-backbone configurations, and no study had systematically compared frozen versus scratch training for multi-backbone ensembles in 3D lung nodule classification. Our eight hybrid configurations were designed specifically to fill this gap.

I will now hand over to Monami, who will present our results.

---

## MEMBER 4 

### Slide 9: Standalone Model Results

Thank you, Irtiza. I am Monami, and I will now present the experimental results. All 11 models were evaluated on the held-out test set of 134 patients — 91 positive, 43 negative — using patient-level aggregation. This means for each patient, we took the maximum prediction probability across all patches as the scan-level score. This max-prob aggregation was chosen because it is more sensitive: a patient with even a single high-confidence positive patch should be flagged for clinical review.

Let me begin with the standalone models. 3D ResNet-18 achieved the best performance with an AUC of 0.9977, F1 of 0.9890, sensitivity of 0.9890, and specificity of 0.9767 — corresponding to only one false positive and one false negative on the entire test set. This result is notable because ResNet-18 was not the strongest model on the validation set — it ranked second behind EfficientNet-B0.

And this brings me to a critical finding. EfficientNet-B0 achieved perfect validation metrics — AUC of 1.0000, F1 of 0.9945, specificity of 0.9767 — but dropped to the worst test performance among the three standalone models: test AUC of 0.9808, F1 of 0.9333, and specificity of 0.8837. That is a 6.1 percentage point drop in F1 and a 9.3 percentage point drop in specificity from validation to test. This gap can be attributed to its MBConv architecture — depthwise separable convolutions with squeeze-and-excitation blocks introduce additional learnable parameters that allowed EfficientNet-B0 to memorise validation-set-specific patterns that did not generalise to held-out data.

DenseNet-121 achieved intermediate performance — AUC of 0.9905, F1 of 0.9560 — but with four false positives and four false negatives. Our audit revealed a deeper issue: when re-running inference on the saved checkpoint, only 23 out of 50 predictions matched the saved CSV within tolerance. This reproducibility drift suggests DenseNet-121's dense connectivity graph is numerically sensitive to mixed precision arithmetic — a critical limitation for clinical deployment.

### Slide 10: Hybrid Ensemble Results

Now, let me present the hybrid ensemble results, which contain the most important findings of our study.

The two frozen MLP configurations — 2-CNN MLP Frozen and 3-CNN MLP Frozen — tied for the best overall performance with an F1 of 0.9945, perfect specificity of 1.0000, zero false positives, and only one false negative. To emphasise the significance: on a test set of 43 negative patients, not a single one was falsely flagged as malignant.

The most striking pattern across all eight hybrids is that every frozen configuration outperforms its scratch counterpart on the test set — a complete reversal of the validation pattern where scratch variants appeared competitive. The gap is largest for 3-CNN MLP: frozen achieves an F1 of 0.9945 versus scratch at 0.9462 — a difference of 4.8 percentage points. For 3-CNN MLP Scratch, specificity falls to 0.8372 with seven false positives — the worst operating point in the entire study.

This universal frozen advantage can be explained by the parameter-to-positive-sample ratio. In the scratch regime, there are approximately 22 million trainable parameters for only 822 positive samples — a ratio of approximately 27,000 to 1, which is orders of magnitude above the rule-of-thumb limit of 10 to 1 for reliable training. In contrast, the frozen MLP regime has only approximately 130,000 trainable parameters — a ratio of approximately 158 to 1, well within the safe range. Scratch models simply memorise the training data — including scanner-specific noise and acquisition artefacts — achieving high validation scores but failing on the independent test set.

The 2-CNN version requires approximately 178 megabytes of storage versus 211 megabytes for 3-CNN, and converges in just two epochs. Both achieve identical F1 and specificity. The 2-CNN MLP Frozen is therefore our recommended best model — the Pareto-optimal point in the accuracy-cost-storage trade-off space.

### Slide 11: Comparison with Literature

Let me now place our results in the context of recent literature. Our best model — 2-CNN MLP Frozen — outperforms every comparable 2025 to 2026 paper on the LUNA16 benchmark.

MSA-Net, published in 2025, uses a 3D RTConvBlock with multi-head self-attention on 48-by-48-by-48 patches and achieves an AUC of 0.9930 and F1 of 0.9550. Our model outperforms it by 0.0044 in AUC and 0.0395 in F1. The sensitivity gap of 0.9890 versus 0.9630 means our ensemble catches approximately 2.6 more malignancies per 100 patients. The specificity gap of 1.0000 versus 0.9470 means 53 fewer false positives per 1,000 negatives.

LMLCC-Net uses a semi-supervised approach with learnable HU filters on 32-by-32-by-32 patches — one-eighth the volume of our patches. Our model outperforms it by 0.0564 in AUC, 0.0600 in sensitivity, and 0.0780 in specificity, demonstrating that fully-supervised training with appropriate hard-negative sampling is more effective than semi-supervised methods on this benchmark.

Sungheetha and colleagues use an ensemble of CNN, LSTM, and Transformer models on LIDC-IDRI, achieving an AUC of 0.9470 and sensitivity of 0.8910 with 10-fold cross-validation. Our model outperforms their ensemble by 0.0504 in AUC and 0.0980 in sensitivity, despite being a simpler architecture with frozen backbones and a small MLP head that converges in a fraction of the training time.

I will now hand over to Sakib, who will discuss the KAN versus MLP comparison, the Grad-CAM analysis, and our conclusions.

---

## MEMBER 5 —

### Slide 12: KAN vs MLP — The Regime-Dependent Reversal

Thank you, Monami. I am Sakib, and I will now present one of the most interesting findings of our study: the regime-dependent reversal between KAN and MLP heads.

In the frozen regime, MLP outperforms KAN on both F1 and specificity. For 2-CNN Frozen, MLP achieves an F1 of 0.9945 versus KAN at 0.9890, and specificity of 1.0000 versus 0.9767. The improvement is a Delta F1 of plus 0.0055 and Delta specificity of plus 0.0233. This pattern is identical for 3-CNN Frozen.

In the scratch regime, the opposite occurs. KAN outperforms MLP, and the gap widens with backbone count. For 2-CNN Scratch, KAN achieves an F1 of 0.9836 versus MLP at 0.9724 — a Delta F1 of minus 0.0112 favouring KAN. For 3-CNN Scratch, KAN's advantage grows: F1 of 0.9730 versus 0.9462 — a Delta F1 of minus 0.0268 — and specificity of 0.9070 versus 0.8372 — a Delta of minus 0.0698.

Why does this happen? The explanation lies in the properties of the features at the input to the classification head. When backbones are frozen, the feature embeddings are derived from ImageNet pre-training and are already near-linearly separable for the lung nodule task. An MLP — essentially a stack of linear transformations with fixed ReLU nonlinearities — is sufficient because the decision boundary, while not perfectly linear, does not require the complex local adaptations that KAN splines provide. KAN's extra expressivity is wasted on features that are already well-separated.

When backbones are fine-tuned from scratch, the feature embeddings shift to a new domain-specific representation that is more optimised for lung nodule features but also more non-linear in structure. Joint fine-tuning allows backbones to develop richer feature interactions — combining texture features from one backbone with shape features from another — and these interactions are better modelled by KAN's learnable spline basis functions, which can adapt to the local data manifold at finer granularity than a fixed ReLU. The effect is strongest for 3-CNN Scratch, where three fine-tuned backbones produce the richest and most non-linear feature space.

This finding reconciles conflicting results in the literature. CPLOYO reports that KAN improves detection in a 2D YOLO architecture — which operates in a scratch regime. Our study finds MLP superior for frozen features. Both findings are correct; the regime dependence means researchers should choose the head based on their training regime rather than assuming one is universally superior.

### Slide 13: Grad-CAM++ Explainable AI Analysis

We applied Grad-CAM++ to the final convolutional layer of each 2-CNN hybrid variant to generate 3D spatial attention heatmaps. For each model, we sampled true positives, true negatives, false positives, and false negatives from the test set.

A critical methodological improvement was applied to avoid what we call the colour-scale illusion. When a raw Grad-CAM heatmap is min-max normalised to the zero-to-one range, even a true negative with a patient-level probability of 0.005 can show a bright red hotspot because the raw activation — say 0.31 — is orders of magnitude smaller than a true positive activation of three to six, but normalisation erases this absolute-scale information. The visual result — a red region on the CT slice — is technically correct but clinically misleading because it suggests the model saw something when in fact it was nearly certain that no nodule was present.

To remedy this, we introduced a confidence-based zeroing rule: whenever the patient-level prediction probability falls below 0.5, the entire heatmap is set to zero. For true positive cases with probability above 0.5, a 90th percentile threshold is applied to suppress weak background noise, producing a clean hotspot localised to the nodule region.

The visual results corroborate our numerical findings. The frozen MLP variant produces crisp, spatially coherent hotspots confined to the nodule boundary. True negatives show uniformly blank heatmaps. The single false negative — probability of only 0.0078 — shows a completely blank overlay, indicating the model did not detect the nodule at all rather than misclassifying it at the boundary. This is consistent with the nodule being in a challenging anatomical position.

In contrast, scratch variants exhibit fragmented, diffuse heatmaps with activations scattered outside the nodule boundary and at patch edges — a visual signature of overfitting. The KAN Scratch model shows the most pronounced fragmentation, with the KAN spline functions adapting to boundary artefacts rather than to the nodule itself.

### Slide 14: Contributions & Recommendations

Let me summarise the key contributions of our work.

First, we conducted a systematic ablation of 11 3D CNN configurations on a common LUNA16 split with patient-level stratification, providing the first regime-dependent analysis of KAN versus MLP heads and frozen versus scratch training for volumetric lung nodule classification.

Second, we demonstrated that frozen multi-backbone ensembles achieve clinically ideal performance — zero false positives, a single false negative — with minimal training cost. The 2-CNN MLP Frozen configuration converges in just two epochs, requires only 178 megabytes of storage, and costs approximately 1.5 GPU-hours to train — representing a 6 to 56 times cost reduction compared to scratch training.

Third, we provided evidence that validation-set metrics can substantially overstate test-set performance, particularly for EfficientNet-B0 which dropped from perfect validation AUC to the worst test performance — underscoring the necessity of held-out test evaluation.

Fourth, we developed a Grad-CAM++ visualisation methodology employing confidence-based heatmap zeroing, demonstrating that heatmap spatial coherence serves as a visual proxy for generalisation quality.

### Slide 15: Conclusion & Future Work

In conclusion, our recommended deployment configuration is the 2-CNN MLP Frozen model. This choice was not our initial hypothesis — we began expecting scratch training with KAN heads to perform best. The systematic evaluation of 11 configurations revealed the opposite, leading to an evidence-based design refinement mid-study.

The frozen MLP ensemble offers several practical advantages for clinical deployment. Its 178-megabyte checkpoint fits within standard PACS workstation memory. Inference requires approximately 150 milliseconds per patch on an A100 GPU, making real-time screening feasible at approximately 300 patches per second. The model's deterministic behaviour — identical input always produces identical output — is critical for clinical auditability. And because the backbones never need retraining, deploying an updated version requires only retraining the thin MLP head in under 30 minutes on a CPU.

For future work, we recommend three directions. First, validating on larger and more diverse multi-institutional datasets such as NLST or PanCan to test generalisability. Second, integrating multi-modal data — combining CT with PET or clinical risk factors — which could further improve sensitivity for the most challenging nodules. Third, developing an interactive XAI dashboard that allows radiologists to query model decisions in real time, bridging the gap between automated diagnosis and clinical adoption.

Our ultimate goal was not to replace radiologists, but to provide them with a reliable, interpretable, and efficient partner in the critical task of early lung cancer detection. With 132 out of 134 test patients correctly classified, zero false positives, and transparent visual explanations for every decision, we believe this work represents a meaningful step toward that goal.

Thank you. We are now happy to answer any questions.
