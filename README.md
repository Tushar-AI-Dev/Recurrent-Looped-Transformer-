# Recurrent Looped Transformer: A Synthetic State-Tracking Experiment

An independent experimental investigation into recurrent reasoning, sequence-length generalization, and state tracking using Recurrent Looped Transformers (RLTs).

---

## Overview

Can a neural network learn to track an evolving state and continue doing so when sequences become significantly longer than anything it encountered during training?

This experiment investigates that question by comparing a Recurrent Looped Transformer (RLT) against three alternative architectures:

- Recurrent Neural Network (GRU)
- Causal Transformer
- Token-only Merge baseline

We designed two synthetic state-tracking tasks, trained all models on sequences of length 32, and evaluated their ability to generalize to sequences up to 128 operations.

The central objective was to investigate whether recurrent latent computation provides an advantage when processing longer sequences.

**Main finding:** The RLT demonstrated partial length generalization, but the GRU achieved substantially stronger results on both tasks. This experiment highlights that recurrent architectural structure alone does not guarantee better generalization.

---

## 1. Research Question

The experiment was motivated by a fundamental question:

> Can recurrent latent computation help a model maintain and update information over sequences longer than its training distribution?

Traditional Transformers process sequences through attention layers. Recurrent architectures instead maintain a hidden state that gets updated as new information arrives.

RLTs explore another direction: repeatedly applying a decoder to a latent representation while maintaining access to encoded information.

We wanted to investigate how these architectural differences affect state tracking.

### Experimental hypothesis

We wanted to examine whether the RLT's recurrent latent computation would help it maintain state over longer sequences, particularly when compared with a standard causal Transformer.

Importantly, we did not assume that RLT would necessarily outperform recurrent neural networks such as GRUs.

---

## 2. Experimental Tasks

We created two synthetic tasks to test whether models could correctly track an evolving state.

### Task A: Parity Tracking

The model receives a sequence of binary operations and must predict the final parity.

The state changes whenever an operation toggles the current state.

For example:

| Operation | State |
|---|---|
| Initial | 0 |
| Toggle | 1 |
| Toggle | 0 |
| Toggle | 1 |
| Toggle | 0 |

The model must determine the final state after processing the entire sequence.

Random guessing gives an expected accuracy of 50%.

### Task B: Five-State Tracking

This task involves tracking a system with five possible states.

Each operation changes the current state according to the task's transition rules.

The model must predict the final state after processing all operations.

Because there are five possible output states, random guessing gives an expected accuracy of 20%.

These tasks allow us to investigate state tracking under different output complexities.

---

## 3. Architectures Compared

We evaluated four architectures under a controlled experimental setup.

### 1. Recurrent Looped Transformer (RLT)

The RLT uses a latent recurrent computation approach.

Its architecture combines:
- An encoder that processes the input sequence.
- Global key/value memory from the encoded representation.
- A recurrent decoder with sliding-window attention.
- The previous final hidden representation as input to subsequent recurrent iterations.

The objective is to investigate whether repeated latent computation can maintain useful information throughout the sequence.

### 2. GRU

A Gated Recurrent Unit processes the sequence incrementally while maintaining a hidden state.

It serves as an important recurrent baseline for comparison.

### 3. Causal Transformer

A standard Transformer with causal attention.

Each token can attend only to preceding tokens, allowing the model to process sequences autoregressively.

### 4. Token-only Merge

A baseline using token-level representations and merging operations without the same recurrent latent computation used by RLT.

It helps investigate whether the observed behavior is specifically associated with recurrent computation.

---

## 4. Experimental Setup

We maintained a controlled setup to make architectural comparisons meaningful.

| Configuration | Value |
|---|---|
| Training sequence length | 32 operations |
| Evaluation lengths | 16, 32, 64, 128 |
| Maximum extrapolation | 4× training length |
| Approximate parameter budget | 79K |
| Number of random seeds | 3 |
| Training updates | 2,000 |
| Test programs | 2,048 per task and length |
| Tasks | Parity, Five-state |
| Seeds | 42, 123, 2026 |

All architectures were evaluated using matched parameter and data budgets.

**Important:** Parameter and data budgets were matched, but computational FLOPs were not necessarily equal.

### Evaluation strategy

The models were trained on sequences of length 32.

After training, we evaluated them at progressively longer sequence lengths.

This allowed us to distinguish between:

- Performance on the training distribution.
- Generalization to unseen sequence lengths.
- Performance degradation as the sequence becomes longer.

---

## 5. Experimental Results

The following results summarize the evaluation at sequence length 128.

### Parity Tracking

| Architecture | Accuracy |
|---|---:|
| RLT | 60.8% |
| GRU | 100% |
| Causal Transformer | ≈48% |
| Token-only Merge | ≈50% |

Random guessing: 50%.

### Five-State Tracking

| Architecture | Accuracy |
|---|---:|
| RLT | 20.7% |
| GRU | 99.97% |
| Causal Transformer | ≈20% |
| Token-only Merge | ≈20% |

Random guessing: 20%.

*Results represent averages across three seeds. Transformer and Token-only Merge values are approximate readings from the experiment's plotted results.*

### Visual Interpretation

The GRU generalized exceptionally well to longer sequences in both tasks.

The RLT performed above chance on the parity task, but its performance was much weaker on the five-state task.

The causal Transformer and Token-only Merge baselines approached chance-level performance at the maximum evaluation length.

---

## 6. What Did We Learn?

### RLT showed partial length generalization

The RLT was able to learn useful state-tracking behavior, particularly on the parity task.

However, its performance degraded as the evaluation sequences became longer.

This suggests that learning recurrent latent computation does not automatically produce reliable state tracking at arbitrary sequence lengths.

### GRU was surprisingly strong

One of the most interesting observations was the GRU's performance.

It achieved:
- 100% accuracy on parity tracking.
- 99.97% accuracy on five-state tracking.

This demonstrates that a relatively simple recurrent architecture can generalize extremely well on certain structured state-tracking tasks.

### Architectural complexity does not guarantee better generalization

The experiment reinforces an important research lesson:

A more sophisticated architecture does not necessarily outperform a simpler architecture.

The effectiveness of an architecture depends on the task, training distribution, inductive biases, and how information is represented and maintained.

---

## 7. Limitations

This experiment is a small-scale synthetic proof of concept.

Several limitations should be considered:

1. Only two synthetic state-tracking tasks were investigated.
2. The evaluation focused on sequence lengths up to four times the training length.
3. The experiment does not establish general reasoning capabilities.
4. Parameter and data budgets were matched, but computational costs were not.
5. The results do not establish performance on natural language or real-world reasoning benchmarks.

The experiment should therefore be interpreted as an architectural comparison on controlled synthetic tasks, not as evidence that one architecture is universally superior.

---

## 8. Conclusion

The original motivation was to investigate whether recurrent latent computation could help models maintain information over increasingly long sequences.

Our results produced an interesting outcome.

RLT demonstrated some ability to generalize beyond its training sequence length, but GRU substantially outperformed it on both synthetic tasks.

This experiment changed our perspective on recurrent architectures: introducing additional latent computation is not sufficient by itself. What matters is whether the architecture learns a representation and update mechanism that reliably preserves the information required by the task.

**The most valuable outcome was not proving that RLT is superior. It was discovering where its generalization behavior falls short and how important strong baseline comparisons are in architectural research.**

---

## Acknowledgements

This experiment was inspired by the Recurrent Looped Transformer research and its exploration of latent recurrent computation.

Original research implementation:

https://github.com/yifanzhang-pro/recurrent-looped-tranformer

---

## Reproducibility

The experiment was conducted using Python and PyTorch in Google Colab with GPU acceleration.

The reported results are based on three random seeds.

For transparency, the numerical results should be interpreted alongside the experimental configuration and limitations documented above.

---

*Independent synthetic state-tracking experiment. This repository documents experimental findings and should not be interpreted as an official reproduction or validation of all claims made by the original RLT research.*
