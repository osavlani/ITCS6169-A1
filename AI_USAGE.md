# AI Usage

## Tool(s) used
Claude (Anthropic) — used as a coding/research assistant throughout the project, from architecture decisions to debugging to README/report drafting.

## Representative prompts and how AI assisted

### 1. Custom architecture design approach
**Prompt:** "So lets say I'm designing my own, what's the approach to understanding the design. How many layers in what order and which type of layers."

**Outcome:** AI explained the standard conv-block pattern (Conv→BN→ReLU→Pool, repeated with growing channel depth), the role of each layer type (BatchNorm, dropout, global average pooling vs. flatten+FC), and gave a concrete suggested skeleton (4-6 blocks, 32→64→128→256 channels) sized appropriately for a 2400-image dataset. Used to inform the decision to go with transfer learning instead, once the accuracy ceiling tradeoff was clear (see Section "Decision made independently" below).

### 2. Backbone comparison and accuracy estimates
**Prompt:** "So approach b if I take lets say resnet or any other existing backbone and train it what do you think should be the estimated accuracy of the possible arch options"

**Outcome:** AI gave a comparison table of AlexNet/VGG16/ResNet18/ResNet34/ResNet50/DenseNet121/EfficientNet-B0/MobileNetV3 with rough expected accuracy ranges for this dataset size, and reasoning for why ResNet18/EfficientNet-B0 were the likely sweet spot (strong pretrained features, small enough to not overfit on ~150 images/class). These were estimates, not guarantees — treated as a starting hypothesis, not a result, and checked against actual training runs.

### 3. Accuracy-pushing techniques (augmentation/regularization/optimizer/hyperparameter tuning)
**Prompt:** "And then I want to push accuracy, so what all can i do. Augmentation ... regularisation ... optimizer ... hyperparameter tuning ... what else do you think I can do which I am missing. No need to agree with me just tell me what seems right, wrong and what more can be done"

**Outcome:** AI reviewed each proposed lever, confirmed/corrected my assumptions (e.g., noted "different layer types" was a smaller lever than I assumed), and added items I'd missed: class imbalance check, stratified k-fold CV, ensembling, data quality/leakage audit, error analysis via confusion matrix. Several of these (confusion matrix, weight-decay ablation, model-capacity comparison) were flagged as still untested gaps in the final README rather than claimed as done — used to scope what was realistic to attempt given time budget, not followed blindly.

### 4. Environment/CUDA compatibility check
**Prompt:** Pasted an import cell and asked for "a cell to test and a test run for cuda as well" before running anything.

**Outcome:** AI provided an environment-diagnostic cell (version/CUDA-capability printout) and a separate CUDA compute test (matrix multiply on GPU with timing) to verify the GPU was actually being used, not just reported as available. This caught that the installed PyTorch build was a nightly/dev release (`2.12.0.dev...+cu128`), which was then documented in the README as a Blackwell-GPU-specific requirement, with instructions for a standard stable install on non-Blackwell GPUs.

### 5. Explaining the ResNet18 vs. TNet delta
**Prompt:** "also tell me whats new compared to basic tnet give me a list in paragraph form" (ResNet18 vs t-net)

**Outcome:** AI produced a paragraph-form summary of every pipeline change between the baseline and the ResNet18 run (RGB vs. grayscale, resolution, normalization, augmentation, architecture depth, pretrained init, two-phase training vs. single-phase). Used directly as the "Final model recipe" explanation basis in the report, cross-checked against the actual notebook cells to confirm nothing was misstated.

6. Verifying ResNet18 cells were complete before running

Prompt: "where do i get the pretrained weights from and are the coding functions given right, ready for execution?"

Outcome: AI confirmed the ResNet18 cells were functionally complete (no placeholders/TODOs). It also explained where ResNet18_Weights.IMAGENET1K_V1 actually downloads from (torchvision's model hub, cached locally) rather than assuming this was already known. This check was done before running the cells, as a sanity pass rather than discovering missing dependencies via a runtime crash.

7. Designing the hyperparameter probe cell

Prompt: Need a short 5-epoch probe runs across learning rate, Mixup/CutMix alpha, and dropout to find good hyperparameter values before committing to a final run, and also is 5 epochs a good test size?

Outcome: AI pushed back on part of the request: a flat 5-epoch probe was fine for alpha and dropout, but risky for learning rate specifically, because the cosine LR schedule needs T_max matched to the actual run length to decay properly — a 5-epoch probe with T_max left at the full schedule length would give a misleading signal. It then wrote a probe_run() function with T_max tied to probe_epochs, plus a set_head_dropout() helper (since the model had no dropout layer at all until this point) and a sweep loop that varied one parameter at a time against fixed defaults, with an explicit note that this was a sequential sweep, not a full grid search, given the time budget.
