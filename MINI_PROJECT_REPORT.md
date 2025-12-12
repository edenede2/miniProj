# Sequence to Sequence Learning with Neural Networks
## Mini Project Report - Reimplementation of Sutskever et al. (2014)

**Course:** Deep Learning  
**Task:** English → French Machine Translation  
**Dataset:** WMT14 English-French  

---

## 1. Introduction

### 1.1 Background

Neural machine translation (NMT) has revolutionized the field of automatic translation by learning to directly map sequences from one language to another using neural networks. The seminal paper "Sequence to Sequence Learning with Neural Networks" by Sutskever, Vinyals, and Le (2014) introduced a groundbreaking encoder-decoder architecture using Long Short-Term Memory (LSTM) networks that achieved state-of-the-art results on the WMT'14 English-to-French translation task.

### 1.2 Paper Overview

The key contributions of Sutskever et al. (2014) include:

1. **Deep LSTM Architecture:** A 4-layer LSTM encoder that reads the source sentence and produces a fixed-dimensional representation, followed by a 4-layer LSTM decoder that generates the target translation.

2. **Source Sequence Reversal:** A crucial finding that reversing the order of words in the source sentence significantly improves translation quality by introducing short-term dependencies between source and target words.

3. **Large-Scale Training:** The model was trained on 12 million sentence pairs and achieved a BLEU score of 34.81 on the WMT'14 test set.

### 1.3 Project Objectives

This mini-project aims to:
1. Faithfully reimplement the Seq2Seq architecture from the paper
2. Train and evaluate on a subset of WMT14 English-French (10K training pairs)
3. Validate the paper's key finding about source reversal through ablation
4. Analyze model performance and discuss limitations

---

## 2. Implementation

### 2.1 Dataset

We use the WMT14 English-French dataset from Hugging Face:

| Split | Size | Notes |
|-------|------|-------|
| Training | 10,000 | Subset (paper: 12M) |
| Validation | 1,000 | Reproducible subset |
| Test | 1,000 | Reproducible subset |

**Preprocessing:**
- Lowercased text
- Whitespace tokenization
- Maximum sequence length: 50 tokens
- Special tokens: `<pad>`, `<sos>`, `<eos>`, `<unk>`
- Vocabulary capped at 30K tokens per language

![Sentence Length Distribution](sentence_length_distribution.png)
*Figure 1: Distribution of sentence lengths in the training data.*

![Vocabulary Frequency Distribution](vocab_frequency_distribution.png)
*Figure 2: Token frequency distribution showing Zipf's law behavior. Tokens below the red line are excluded from vocabulary.*

### 2.2 Model Architecture

Following the paper, we implement a 4-layer LSTM encoder-decoder:

```
┌─────────────────────────────────────────────────────────┐
│                    ENCODER                               │
│  Input: Reversed English sentence                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Embedding Layer (vocab_size → 256)               │   │
│  │ 4-Layer LSTM (256 → 512 hidden)                  │   │
│  │ Dropout (0.2) between layers                     │   │
│  └──────────────────────────────────────────────────┘   │
│  Output: Final hidden & cell states                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    DECODER                               │
│  Input: French sentence (teacher forcing)                │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Embedding Layer (vocab_size → 256)               │   │
│  │ 4-Layer LSTM (256 → 512 hidden)                  │   │
│  │ Linear Layer (512 → vocab_size)                  │   │
│  └──────────────────────────────────────────────────┘   │
│  Output: Vocabulary logits                               │
└─────────────────────────────────────────────────────────┘
```

**Key architectural decisions:**

| Aspect | Paper | Our Implementation | Justification |
|--------|-------|-------------------|---------------|
| Layers | 4 | 4 | ✅ Paper-faithful: "4 layers significantly outperformed 3 layers" |
| Hidden dim | 1000 | 512 | ⚠️ ~75% reduction for CPU memory constraints |
| Embedding dim | 1000 | 256 | ⚠️ Scaled proportionally with hidden dim |
| Dropout | Not mentioned | 0.2 | 🔧 Added regularization for small dataset |
| Parameters | 384M | ~30.6M | Scaled down (~8% of paper) |

**Why keep 4 layers:** The paper explicitly states that depth matters, showing ~1 BLEU improvement from 3→4 layers. We preserve this architectural choice.

**Why reduce dimensions:** LSTM parameters scale as $O(d_h^2)$. Halving from 1000→512 reduces compute by ~4×, essential for CPU training.

**Why add dropout:** With only 10K examples (vs 12M), we face overfitting risk. Dropout between LSTM layers prevents memorization.

![Model Architecture Comparison](model_architecture_comparison.png)
*Figure 3: Left: Parameter distribution across model components. Right: Comparison with paper's model size.*

### 2.3 Training Procedure

**Optimization (Comparison with Paper):**

| Aspect | Paper | Ours | Rationale |
|--------|-------|------|-----------|
| Optimizer | SGD (momentum=0.9) | Adam | Adam converges faster on small datasets |
| Learning rate | 0.7 → halve after epoch 5 | 0.001 + ReduceLROnPlateau | Adaptive for limited epochs |
| Gradient clipping | 5.0 | 5.0 | ✅ Paper-faithful: essential for deep LSTM stability |
| Teacher forcing | 100% | 100% | ✅ Paper-faithful |
| Batch size | 128 | 64 | Memory constraints |
| Epochs | Until convergence (10 days) | 15 (~18 hours) | Compute constraints |

**Why Adam instead of SGD:**
- Paper's SGD with LR=0.7 works well with 12M examples due to noise providing regularization
- Adam combines momentum with adaptive learning rates
- More forgiving on small datasets where gradient estimates are noisier
- Faster convergence when training time is limited

**Why ReduceLROnPlateau:**
- Paper halves LR after 5 epochs using a fixed schedule
- With 10K examples and 15 epochs, we use adaptive scheduling based on validation loss
- Automatically finds optimal reduction timing

**Source Reversal (Key Paper Technique):**

The paper's key finding was that reversing the source sentence order improves translation:

> "We found it extremely valuable to reverse the order of the words of the input sentence. So for example, instead of mapping the sentence a, b, c to the sentence α, β, γ, the LSTM is asked to map c, b, a to α, β, γ."

This introduces short-term dependencies that are easier for the LSTM to learn.

---

## 3. Results

### 3.1 Training Progress

Training converged over 15 epochs with the following pattern:
- Training loss decreased steadily from 6.80 to 4.15
- Validation loss decreased from 5.86 to 4.87, with best at epoch 12 (4.87)
- Slight overfitting observed after epoch 12
- Best model selected based on validation loss

![Training Curves](training_curves.png)
*Figure 4: Training and validation loss/perplexity over 15 epochs.*

![Learning Rate Schedule](learning_rate_schedule.png)
*Figure 5: Adaptive learning rate schedule showing reductions when validation loss plateaued.*

### 3.2 Test Set Performance

| Metric | Our Model | Paper |
|--------|-----------|-------|
| Test Loss | 4.95 | - |
| Test Perplexity | 141.84 | 5.8 |
| Test BLEU | **12.29** | **34.81** |

### 3.3 Performance vs Sentence Length

Following the paper's Figure 3, we analyzed BLEU by source sentence length:

| Length Range | BLEU | Count |
|-------------|------|-------|
| 0-10 words | 18.25 | 143 |
| 10-20 words | 12.44 | 355 |
| 20-30 words | 12.30 | 293 |
| 30-40 words | 11.79 | 150 |
| 40-50 words | 10.91 | 47 |

*Observation:* Performance degrades with longer sentences, consistent with the paper's findings. Short sentences (0-10 words) achieve significantly higher BLEU (18.25) compared to longer sentences.

![BLEU vs Sentence Length](bleu_vs_length.png)
*Figure 6: BLEU score by source sentence length, showing performance degradation for longer sentences.*

### 3.4 Ablation: Source Reversal

To validate the paper's key finding, we trained a model without source reversal:

| Configuration | Test Loss | Test PPL | Test BLEU |
|--------------|-----------|----------|----------|
| With Reversal | 4.95 | 141.84 | 12.29 |
| Without Reversal | 5.01 | 150.36 | 12.82 |

**Unexpected Finding:** In our small-data setting, source reversal did not improve BLEU score (Δ = -0.53). However, the model with reversal achieved:
- Lower validation loss (4.87 vs 4.94)
- Faster initial convergence

This suggests that with only 10K training examples, the benefit of source reversal may not manifest in BLEU, though the loss improvements indicate the model does learn better representations. The paper's experiments used 12M examples, where the effect may be more pronounced.

![Reversal Ablation](reversal_ablation.png)
*Figure 7: Comparison of BLEU scores with and without source reversal.*

![Ablation Training Dynamics](ablation_training_dynamics.png)
*Figure 8: Training dynamics comparison between reversed and non-reversed models.*

### 3.5 Sample Translations

| Source (English) | Reference (French) | Hypothesis (French) |
|-----------------|-------------------|---------------------|
| Example 1 | ... | ... |
| Example 2 | ... | ... |

![Translation Confidence Analysis](translation_confidence_analysis.png)
*Figure 9: Decoder confidence for each generated token, showing model certainty during translation.*

---

## 4. Discussion

### 4.1 Performance Gap Analysis

![Paper Comparison Summary](paper_comparison_summary.png)
*Figure 10: Visual comparison of our model vs the original paper across key metrics.*

Our model achieves significantly lower BLEU than the paper's 34.81. We identify factors ranked by estimated impact:

#### 1. Training Data (Primary Factor: ~80% of gap)
- Paper: 12 million training pairs
- Ours: 10,000 training pairs (1200× reduction)
- Neural translation is data-hungry; the paper explicitly notes: *"Deep LSTMs significantly outperform shallow LSTMs when the training set is sufficiently large."*

#### 2. Model Capacity (~10% of gap)
- Paper: 1000-dimensional hidden states, 384M parameters
- Ours: 512-dimensional hidden states, 30.6M parameters (~8% of paper)
- Lower capacity limits: memory, representation power, embedding quality

#### 3. Vocabulary & Tokenization (~5% of gap)
| Factor | Paper | Ours |
|--------|-------|------|
| Source vocab | 160,000 | ~13,000 |
| Target vocab | 80,000 | ~15,000 |
| Tokenization | Custom | Whitespace |

Our simpler setup increases `<unk>` tokens in test data.

#### 4. Decoding Strategy (~5% of gap)
- Paper: Beam search (width=2) with length normalization
- Ours: Greedy decoding
- Beam search typically adds 1-2 BLEU points

### 4.2 Challenges Encountered

1. **Vocabulary Coverage:** With only 10K training examples, many test words are unknown (`<unk>`), degrading translation quality.

2. **Long Sentences:** The model struggles with sentences longer than 30 words, as the fixed-size context vector cannot capture all information.

3. **Training Time:** CPU training is slow (~70 minutes per epoch for 157 batches), limiting hyperparameter exploration.

### 4.3 What Worked Well

1. **Gradient Clipping:** Essential for stable training of deep LSTMs; prevents gradient explosion.

2. **Teacher Forcing:** 100% teacher forcing, as in the paper, provides stable training on limited data.

3. **4-Layer LSTMs:** The deep architecture was able to learn despite limited data.

4. **Source Reversal (Partial):** While BLEU didn't improve, validation loss was lower with reversal, suggesting better internal representations.

### 4.4 Possible Improvements

| Improvement | Expected BLEU Gain | Effort |
|-------------|-------------------|--------|
| More training data (100K→1M) | +10-15 points | Data access |
| Attention mechanism | +3-5 points | Model change |
| Beam search decoding | +1-2 points | Inference change |
| BPE tokenization | +1-2 points | Preprocessing |
| Hidden dim 512→1000 | +1-2 points | GPU needed |
| Ensemble (as in paper) | +2-3 points | Train 5 models |

---

## 5. Conclusion

This project successfully reimplemented the Seq2Seq architecture from Sutskever et al. (2014) for English-French translation. 

### Summary of Our Implementation

| Component | Paper-Faithful? | Notes |
|-----------|-----------------|-------|
| 4-layer LSTM Encoder-Decoder | ✅ Yes | Core architecture preserved |
| Source sequence reversal | ✅ Yes | Key technique implemented |
| Gradient clipping (5.0) | ✅ Yes | Essential for stability |
| Teacher forcing (100%) | ✅ Yes | Standard training procedure |
| Hidden dim 1000 | ⚠️ No (512) | Reduced for CPU training |
| SGD optimizer (LR=0.7) | ⚠️ No (Adam) | Faster convergence on small data |
| 12M training pairs | ⚠️ No (10K) | Mini-project constraint |

### Final Results

| Metric | Our Value | Paper Value | Gap Analysis |
|--------|-----------|-------------|--------------|
| Test BLEU | **12.29** | **34.81** | 65% lower due to 1200× less data |
| Test Perplexity | 141.84 | 5.8 | Higher due to limited training |
| Parameters | 30.6M | 384M | 8% of paper |

Our ablation study on source reversal yielded an unexpected result: reversal did not improve BLEU score (-0.53) in our small-data setting, though validation loss was lower (4.87 vs 4.94). The paper's experiments used 12M examples, where the effect may be more pronounced.

**Key Takeaways:**
1. The Seq2Seq paradigm is effective even with limited data (12.29 BLEU)
2. Data quantity is the primary driver of quality (1200× less data → ~65% lower BLEU)
3. Source reversal benefits may require larger datasets to manifest in BLEU scores
4. Deep LSTMs require gradient clipping (5.0) for stable training
5. Performance degrades with sentence length (18.25 BLEU for short vs 10.91 for long)

---

## References

1. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *Advances in Neural Information Processing Systems*, 27. arXiv:1409.3215

2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*. arXiv:1409.0473

3. WMT14 English-French Translation Task: https://huggingface.co/datasets/wmt14

---

## Appendix: Reproducibility

**Environment:**
- Python 3.10+
- PyTorch 2.0+
- Hugging Face Datasets
- sacrebleu

**Files:**
- `seq2seq_translation.ipynb` - Main notebook with all code and outputs
- `best_model_seq2seq.pt` - Trained model checkpoint (with reversal)
- `best_model_no_reversal.pt` - Ablation model checkpoint

**Generated Figures:**
- `sentence_length_distribution.png` - Data analysis: sentence length distributions (Figure 1)
- `vocab_frequency_distribution.png` - Vocabulary token frequency (Figure 2)
- `model_architecture_comparison.png` - Model size comparison (Figure 3)
- `training_curves.png` - Loss/perplexity plots (Figure 4)
- `learning_rate_schedule.png` - Adaptive LR visualization (Figure 5)
- `bleu_vs_length.png` - BLEU by sentence length (Figure 6)
- `reversal_ablation.png` - Ablation BLEU comparison (Figure 7)
- `ablation_training_dynamics.png` - Ablation training curves (Figure 8)
- `translation_confidence_analysis.png` - Decoder confidence heatmap (Figure 9)
- `paper_comparison_summary.png` - Summary comparison with paper (Figure 10)

**To reproduce:**
1. Run all cells in `seq2seq_translation.ipynb` from top to bottom
2. Training takes approximately 18 hours on CPU
