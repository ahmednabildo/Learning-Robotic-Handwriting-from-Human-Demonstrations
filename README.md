# Learning Robotic Handwriting from Human Demonstrations

**Learning character-conditioned pen trajectories from human handwriting demonstrations using Python and PyTorch.**

This project investigates how a robot could learn handwriting motion primitives from recorded human demonstrations. It processes handwritten characters into sequences of **2D pen displacements and pen states**, compares demonstration-based trajectory baselines, and trains an **autoregressive, character-conditioned LSTM** to predict the next pen movement and whether the pen should draw, lift, or finish the character.

The repository includes the **UJI Pen Characters Version 2** data, preprocessing and training notebooks, serialized data, and saved LSTM checkpoints. Generated trajectories are visualized as character strokes. The implementation is a trajectory-learning prototype; it does **not** include a physical robot controller or deployment on a robot arm.

> **Project status:** The committed implementation uses an LSTM with separate coordinate-regression and three-class pen-state heads. Mixture Density Networks (MDNs), Transformers, and robot hardware execution are not implemented in the committed code.

## 1. Objectives

- Parse handwriting demonstrations containing multiple strokes per character.
- Normalize and resample the demonstrations into consistent trajectory representations.
- Preserve pen-up movements between strokes and an explicit end-of-character event.
- Construct simple character-level trajectory baselines.
- Train an LSTM conditioned on the requested character.
- Generate pen trajectories autoregressively and reconstruct their visible strokes.
- Evaluate the model with writer-disjoint training and validation sets.

## 2. Repository structure

```text
Learning-Robotic-Handwriting-from-Human-Demonstrations/
├── Project.ipynb                     # Parsing, preprocessing and baseline exploration
├── Training (Copy).ipynb             # PyTorch LSTM training and character generation
├── uji+pen+characters+version+2/
│   ├── ujipenchars2.txt              # Raw UJI Pen Characters v2 dataset
│   └── uji2.names                    # Dataset description
├── delta_data.pkl                    # Serialized processed trajectories
├── handwriting_lstm_best.pt          # Best checkpoint after stages 1–2
├── handwriting_lstm_stage3_best.pt   # Additional fine-tuning checkpoint
├── averaged_trajectories.pkl         # Empty placeholder in current commit
├── best_trajectories.pkl             # Empty placeholder in current commit
├── Project 1 - Handwriting.docx      # Project document
├── paper.pdf                         # Included paper
└── .gitattributes
```

The two baseline pickle files are currently **0 bytes** and cannot be loaded as valid pickle objects. The working saved data file used by the training notebook is **`delta_data.pkl`**.

## 3. Dataset

The project uses **UJI Pen Characters Version 2**, with handwritten character samples from **60 writers**. The preprocessing notebook reports **11,640 demonstrations** spanning **97 distinct characters**. The examples include letters, digits, and other symbols.

Each demonstration is initially organized as a collection of strokes, with each stroke represented by an ordered sequence of XY coordinates:

```math
S = \{(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)\}
```

A character can contain multiple strokes. Their boundaries matter because the pen must sometimes move without drawing.

### Writer-disjoint data split

The training notebook uses the dataset's existing writer identifiers (`trn_` and `tst_`), then divides the 20 `tst_` writers equally between validation and held-out test sets.

| Set | Writers | Demonstrations |
|---|---:|---:|
| Training | 40 | 7,760 |
| Validation | 10 | 1,940 |
| Test | 10 | 1,940 |
| **Total** | **60** | **11,640** |

**Why split by writer?** Examples from one writer are kept in a single split, making validation more informative about generalization to handwriting styles not seen during training. The committed notebook constructs a test loader, but it does not report a final held-out **test-set evaluation**.

## 4. Data preprocessing

Implemented in [`Project.ipynb`](Project.ipynb). The notebook processes raw demonstrations into model-ready sequences.

### 4.1 Parse and visualize strokes

The original text dataset is read into a nested structure:

```python
data[writer_id][character][repetition_index]
```

Each repetition contains a list of strokes, and each stroke contains coordinate pairs. The notebook first visualizes selected handwritten characters to inspect the raw structure.

### 4.2 Coordinate orientation

The preprocessing notebook inverts the Y coordinate to use the desired plotting orientation. The transformation is applied across all writers, characters, repetitions, and strokes.

### 4.3 Resample trajectories

Different demonstrations of the same character contain different numbers of recorded points. The notebook calculates a **character-specific target point count** by rounding the **mean** number of points across demonstrations of that character. (Although a code comment calls it a median, the implemented expression uses `np.mean`.)

Points are redistributed across strokes in proportion to each stroke's path length, followed by interpolation within the individual strokes. This retains the distinction between separate pen strokes while bringing character demonstrations to a common point count.

### 4.4 Center the character

Each resampled character is centered using the midpoint of its XY bounding box:

```math
x_c = \frac{x_{\min}+x_{\max}}{2}, \qquad
 y_c = \frac{y_{\min}+y_{\max}}{2}
```

```math
x_i' = x_i-x_c, \qquad y_i' = y_i-y_c
```

### 4.5 Convert positions to motion commands

The model works with **relative displacements**, rather than absolute XY positions:

```math
\Delta x_t = x_t-x_{t-1}, \qquad
\Delta y_t = y_t-y_{t-1}
```

Each time step is represented by:

```math
\mathbf{s}_t =
\begin{bmatrix}
\Delta x_t \\
\Delta y_t \\
 p_t
\end{bmatrix}
```

The discrete pen state `p_t` takes one of three values:

| Pen state | Meaning |
|---|---|
| `0` | **Pen down** — draw the displacement |
| `1` | **Pen up** — move without drawing |
| `2` | **End** — stop generating the character |

The preprocessing function includes an initial pen-up token, transitions between strokes, and an explicit end token.

### 4.6 Normalize and save

The processed displacement values are scaled per sequence using the largest absolute XY displacement value, preserving the relative scale between the two coordinate components; the pen-state label is kept unchanged. The result is saved to `delta_data.pkl`.

The notebook also reconstructs strokes from these sequences for consistency checks. Its recorded outputs report that all **11,640** demonstrations passed stroke-count and point-count reconstruction checks.

> **Preprocessing note:** The sequence normalization in the committed code is applied to **displacement components**. It is not the same as normalizing absolute positions to a fixed drawing canvas.

## 5. Demonstration-based trajectory baselines

`Project.ipynb` also explores two reference approaches before neural sequence learning:

- **Average trajectory:** compute a pointwise average across demonstrations of a character, with an additional mean of displacement sequences and a majority-vote treatment of the binary pen-up/down state.
- **Representative demonstration:** compare demonstrations and select a candidate with the lowest accumulated discrepancy according to the notebook's comparison procedure.

The notebook plots these trajectories for selected characters. These are exploratory baselines; the repository does **not** provide a validated numeric head-to-head comparison against the trained LSTM. The baseline pickle output statements are commented out in the current preprocessing notebook.

## 6. Character-conditioned LSTM

Implemented in [`Training (Copy).ipynb`](Training%20%28Copy%29.ipynb).

### 6.1 Model inputs and outputs

The LSTM receives a previous motion/pen-state token together with a learned embedding of the target character.

```math
\mathbf{z}_t =
\begin{bmatrix}
\Delta x_{t-1} \\
\Delta y_{t-1} \\
 p_{t-1} \\
\mathbf{e}(c)
\end{bmatrix}
```

Here, `c` is the requested character and `e(c)` is its learned embedding. The model predicts the next XY displacement and a distribution of logits over the three pen states:

```math
\hat{\mathbf{d}}_t =
\begin{bmatrix}
\widehat{\Delta x}_t \\
\widehat{\Delta y}_t
\end{bmatrix},
\qquad
\boldsymbol{\ell}_t \in \mathbb{R}^{3}
```

The same character embedding is repeated across all time steps of a sequence.

### 6.2 Architecture

| Component | Committed setting |
|---|---|
| Character vocabulary | 97 characters |
| Character embedding | 64 dimensions |
| Sequence input | 3 motion/pen features + 64 embedding features |
| Recurrent network | 3-layer LSTM |
| Hidden size | 128 |
| LSTM inter-layer dropout | 0.20 |
| Embedding dropout | 0.10 |
| Output dropout | 0.20 |
| Coordinate head | Linear layer: 128 → 2 |
| Pen-state head | Linear layer: 128 → 3 |
| Trainable parameters | 371,909 |

The output consists of **deterministic displacement predictions**, plus classification logits for the pen state. There is no probabilistic mixture output for coordinates in this version.

### 6.3 Sequence construction and padding

For teacher-forced training, the input trajectory is shifted by one time step, preceded by a start token `[0, 0, 1]`. The original trajectory is used as the target. Sequences are padded to the maximum observed length of **98 steps**, and a boolean mask identifies real tokens versus padding.

The padded training tensor has shape:

```text
(7760, 98, 3)
```

### 6.4 Loss function

Training combines a masked, coordinate-standardized mean squared error with a weighted cross-entropy loss for pen-state classification:

```math
\mathcal{L} =
\mathcal{L}_{\mathrm{xy}} +
\lambda_{\mathrm{pen}}\mathcal{L}_{\mathrm{pen}}
```

```math
\mathcal{L}_{\mathrm{xy}} =
\frac{1}{N}\sum_{t\in\mathcal{V}}
\frac{1}{2}\left[
\left(\frac{\widehat{\Delta x}_t-\Delta x_t}{\sigma_x}\right)^2
+
\left(\frac{\widehat{\Delta y}_t-\Delta y_t}{\sigma_y}\right)^2
\right]
```

Here, `V` denotes valid (non-padding) time steps and `N` is their number. The coordinate standard deviations are calculated from valid training targets only. The notebook prints approximately `0.1280` and `0.1657` for the two components.

The pen-state loss uses three-class cross entropy, with class weights inversely related to their frequency in the training targets. The configured weight is:

```math
\lambda_{\mathrm{pen}} = 0.1
```

The training code also computes a normalized coordinate-error statistic and pen-up recall internally, although the recorded training log emphasizes the **combined training and validation loss**.

## 7. Training configuration

The notebook trains on the **CPU** and sets NumPy/PyTorch random seeds to `42`. It uses **batch size 16**, gradient clipping, best-validation checkpoint tracking, and learning-rate scheduling.

| Stage | Optimizer | Initial learning rate | Early stopping patience |
|---|---|---:|---:|
| 1: Initial training | AdamW | `1e-3` | 6 epochs |
| 2: Fine-tuning | Adam | `3e-5` | 15 epochs |
| 3: Additional fine-tuning | Adam | `1e-6` | 15 epochs |

The optimizer configurations use `weight_decay=1e-4`. The notebook clamps gradient norms to **10** and uses `ReduceLROnPlateau` scheduling.

The committed notebook explains that CPU execution was selected because its local PyTorch build was incompatible with the installed NVIDIA MX230 GPU. A different compatible environment may use another device after adjusting the notebook.

## 8. Recorded results

The following values come from **saved notebook outputs**, not from a fresh training run.

| Training stage | Best recorded epoch | Best recorded validation loss |
|---|---:|---:|
| Stage 1 | 37 | 0.36767 |
| Stages 1–2 combined | 81 | 0.34270 |
| Stage 3 checkpoint | 98 | Not printed in the retained output |

The final checkpoint-loading cell confirms that `handwriting_lstm_stage3_best.pt` was loaded successfully and that it contains the **97-character** vocabulary with saved best epoch **98**.

These values represent the notebook's **combined, normalized validation objective**, not character recognition accuracy, physical position error in millimeters, or a reported test-set score.

The notebook includes visualizations of training/validation curves and generated sample characters, but the repository does not provide separate PNG result files for embedding directly in this README.

## 9. Autoregressive character generation

At inference time, the trained model starts from `[0, 0, 1]` and a requested character ID. For each new step it:

1. Runs the LSTM with the current input and recurrent state.
2. Predicts a displacement `(dx, dy)`.
3. Chooses the pen-state class with the highest logit (`argmax`).
4. Appends the predicted step to the trajectory.
5. Feeds that step back into the model.
6. Stops when the model predicts the end state (`2`) or reaches the maximum number of steps.

The notebook's generation function defaults to **97 steps** for the showcased examples. Predicted displacements are integrated into XY paths, while pen-up segments are omitted from rendered strokes.

Sample characters in the notebook include:

```python
sample_chars = ['a', 'b', 'c', 'A', 'B', 'z', '2', 'u', '8']
```

Generation is **greedy and deterministic** under fixed model weights and inputs; it does not sample from an MDN or use beam search.

## 10. Installation and usage

### Prerequisites

- Python with Jupyter Notebook or JupyterLab
- PyTorch
- NumPy
- Matplotlib

The notebooks also use Python standard-library modules such as `pickle`, `pathlib`, and `copy`. There is no pinned `requirements.txt` in the repository, so package versions should be chosen for your Python and hardware environment.

### Clone the repository

```bash
git clone https://github.com/ahmednabildo/Learning-Robotic-Handwriting-from-Human-Demonstrations.git
cd Learning-Robotic-Handwriting-from-Human-Demonstrations
```

### Install dependencies

One possible environment setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy matplotlib jupyterlab
jupyter lab
```

For CUDA acceleration, select a **PyTorch build compatible with your GPU and drivers**. The committed training notebook explicitly selects CPU.

### Run preprocessing

Open **`Project.ipynb`** and run its preprocessing cells in order. They read the bundled `ujipenchars2.txt`, parse and transform the strokes, and save `delta_data.pkl`.

The processed pickle file is already committed, so this step can be skipped when using the supplied model-training data.

### Train the LSTM

Open **`Training (Copy).ipynb`** and execute the data-loading, dataset construction, architecture, training, and checkpoint cells in order. The notebook reads `delta_data.pkl` from the **current working directory**.

It saves the following checkpoints:

```text
handwriting_lstm_best.pt
handwriting_lstm_stage3_best.pt
```

### Generate characters from a saved checkpoint (without retraining)

The notebook includes a **“Load the latest saved model (skip training)”** section. To use it, run the cells defining:

1. Imports and `HandwritingLSTM`.
2. `deltas_to_strokes` (needed for plotting).
3. The checkpoint-loading code.
4. `generate_trajectory`.
5. The sample-character plotting code.

The loader in the current notebook specifically looks for **`handwriting_lstm_stage3_best.pt`**, despite the heading referring to the “latest” model. It uses CPU loading with `torch.load(..., weights_only=True)` and restores the model, character mapping, and normalization metadata.

A trajectory can then be generated using the notebook function:

```python
trajectory = generate_trajectory(
    model,
    char='a',
    char_to_id=char_to_id,
    max_steps=97,
)

strokes = deltas_to_strokes(trajectory)
```

The notebooks are **interactive workflows**, not a packaged command-line application. Execute the prerequisite cells before using the functions shown above.

## 11. Current limitations

- **No hardware execution:** Predicted pen paths are not yet converted to a robot manipulator's kinematics, trajectories, or motor commands.
- **Deterministic XY output:** The LSTM does not represent multiple possible handwriting styles with a mixture distribution.
- **Greedy pen-state decoding:** The pen state uses `argmax`; errors can lead to missing pen lifts, early stopping, or overly long sequences.
- **Class imbalance:** Pen-up and end events may be substantially less frequent than drawing steps; weighted cross entropy addresses this only partially.
- **Accumulated trajectory drift:** Autoregressive displacement errors can compound during generation.
- **Limited evaluation reporting:** There is no final held-out test-set metric or quantitative baseline-versus-LSTM comparison in the committed results.
- **No reproducible package lockfile:** Python/PyTorch versions are not pinned, and checkpoint loading may depend on version compatibility.
- **Preprocessing split caveat:** Character-specific resampling targets are calculated from all demonstrations before the training/validation/test split; these targets therefore use statistics beyond the training writers.

## 12. Future work

Potential extensions (not currently implemented) include:

- Mixture Density Network heads for multimodal coordinate predictions.
- Transformer-based autoregressive sequence modeling.
- More robust modeling and evaluation of pen-up and end-of-character events.
- Quantitative assessment on unseen writers using trajectory and pen-state metrics.
- Distributional or geometry-aware trajectory losses.
- Physical robot handwriting through trajectory scaling, inverse kinematics, and motion control.
- A reproducible evaluation script, package manifest, and saved example figures.

## 13. Project context

This is a machine-learning and robotics project investigating how pen trajectories can be learned from human demonstrations. It combines **time-series preprocessing**, **sequential neural modeling**, **character-conditioned generation**, and **stroke reconstruction**.

For the dataset format and associated academic material, see the files included in this repository:

- [`uji+pen+characters+version+2/uji2.names`](uji+pen+characters+version+2/uji2.names)
- [`Project 1 - Handwriting.docx`](Project%201%20-%20Handwriting.docx)
- [`paper.pdf`](paper.pdf)

## Author

**Ahmed Nabil Donia**  
MSc Mechanical Engineering — Mechatronics Track  
Politecnico di Milano

[GitHub Profile](https://github.com/ahmednabildo)
