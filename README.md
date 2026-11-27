# Hindi and Hinglish Intent Classification (Anveshan DataSprint 2.0, NLP track)

A compact pipeline that classifies Hindi (Devanagari) and Hinglish (Roman script) voice-assistant queries into 56 intents. Class 55 is the out-of-scope fallback. Metric: unweighted Macro-F1.

- Public leaderboard: **0.84215** Macro-F1 (the private score is revealed after the deadline)
- Hold-out validation of the final blend: **0.8715** Macro-F1 (single 80/20 split)
- Live demo: https://svm-intent-classifier.streamlit.app

## Files in this repo

| File | What it is |
|---|---|
| `datasprint_nlp.ipynb` | Training and submission code: augmentation, TF-IDF + SVM, MuRIL fine-tune, blend, latency check |
| `model_arrays.npz` | Trained TF-IDF + SVM weights, exported from the notebook |
| `model_vocab.json.gz` | Vocabulary of the TF-IDF model, exported from the notebook |
| `app.py` | Prediction code and result cards for the demo (loads the two model files) |
| `streamlit_app.py` | The Streamlit demo page |
| `requirements.txt` | Packages for the demo app (`streamlit`, `numpy`) |
| `.devcontainer/` | Created by GitHub Codespaces, not used by the project |

## 1. Pipeline overview and architecture

```
query text
   |
   |-- lowercase
   |
   |-- Model A: TF-IDF (char 2-5 grams + word 1-2 grams) -> LinearSVC (balanced class weights)
   |-- Model B: google/muril-base-cased fine-tuned for 56 classes
   |
   v
softmax(SVM scores, T = 0.2)   and   softmax(MuRIL logits)
   |
   v
0.5 x MuRIL + 0.5 x SVM  ->  argmax  ->  class 0..55
```

Training data is the competition's `train.csv` (16,936 rows) plus generated copies (section 2). No external data and no paid APIs were used.

MuRIL settings: max length 48, batch size 64, 4 epochs, learning rate 4e-5, label smoothing 0.05, square-root class weights, mixed precision (fp16) on a Kaggle GPU.

## 2. Cross-script (Hindi and Hinglish) and Class 55 handling

**Finding.** Counting scripts, 33.3% of train queries are Latin-only versus 64.2% of test queries. Train is mostly Devanagari and test is mostly Roman-script Hinglish. The test Hinglish is also more casual. Words like "bhai", "yaar", "zara" and "jaldi" appear in 1 to 4 train rows but in 56 to 153 test rows.

**What the notebook does about it:**
1. **Transliteration.** A rule-based Devanagari-to-Roman converter makes a Roman copy of every Devanagari training row, plus a noisy copy.
2. **ASR-style noise.** Random dropped vowels, doubled letters and split words.
3. **Casual-spelling copies** (`casual()`). Short vowels, shortcuts such as "mjhe", "nhi" and "plz", and filler words such as "bhai" and "yaar" on Roman rows. The SVM trains on all of these copies. MuRIL trains on a 40% sample of them to limit GPU time.
4. Extra rows are created only from the training part when validating, never from the validation part.

**Class 55.** Train has 249 class-55 rows (1.5%). Their content is mixed (alarm and coffee requests, "next", "ludo game", playlists), so a confidence cutoff alone would not work. Class 55 is trained as an ordinary class, with balanced class weights in the SVM and square-root class weights in MuRIL. About 7.4% of our test predictions are class 55, which is higher than its share in train. We cannot check this against test labels.

## 3. Model parameter count (limit: 300M)

| Component | Parameters |
|---|---|
| google/muril-base-cased with a 56-class head | about 237.6M (BERT-base layout with a 197,285-word vocabulary, plus a 43,064-parameter head) |
| TF-IDF + LinearSVC (56 x 247,373 coefficients, 56 intercepts, 247,373 IDF weights) | about 14.1M |
| **Total at inference** | **about 251.7M, under the 300M limit** |

These figures are computed from the architecture and vocabulary sizes. The notebook's last cell prints the exact counts:

```python
mu = sum(p.numel() for p in mF.parameters())
sv = (svmF.coef_.size + svmF.intercept_.size
      + len(fF.transformer_list[0][1].idf_) + len(fF.transformer_list[1][1].idf_))
print("MuRIL:", mu, "| SVM+TF-IDF:", sv, "| total:", mu + sv)
```

## 4. Local validation score and summary

Validation is a stratified 80/20 hold-out split of `train.csv` (`random_state=0`), not k-fold. Scores come from the Kaggle run of `datasprint_nlp.ipynb`.

| Model | Macro-F1, all | Devanagari rows | Roman rows |
|---|---|---|---|
| TF-IDF + SVM (blend weight 0.0) | 0.8316 | 0.7867 | 0.9108 |
| Blend, MuRIL weight 0.3 | 0.8629 | 0.8282 | 0.9217 |
| **Blend, MuRIL weight 0.5 (used)** | **0.8715** | 0.8351 | **0.9250** |
| Blend, MuRIL weight 0.7 | 0.8715 | 0.8415 | 0.9144 |
| MuRIL alone (weight 1.0) | 0.8015 | 0.7935 | 0.8088 |

The blend beats either model alone by about 0.04 to 0.07. The weights 0.5 and 0.7 tie overall. We chose 0.5 because it is better on Roman rows, which are most of the test set.

**Effect of the augmentation** (SVM only, three split seeds, measured separately from the Kaggle run):

| Setup | Macro-F1, seeds 0 / 1 / 2 | Roman rows only |
|---|---|---|
| No augmentation | 0.771 / 0.777 / 0.769 | 0.731 / 0.739 / 0.705 |
| Transliteration + noise | 0.829 / 0.826 / 0.819 | 0.900 / 0.874 / 0.862 |
| Transliteration + noise + casual-spelling copies | 0.829 / 0.832 / 0.827 | 0.906 / 0.891 / 0.893 |

**Caveats**
- The Roman validation rows share the style of train's Roman text, which is less casual than the test's. The real gain on test is probably smaller than the validation numbers suggest. The public leaderboard score (0.84215) is the more realistic figure.
- The public leaderboard covers about 20% of the test set, so small differences are noisy.
- The blend weight was chosen on the same validation split that is reported, so 0.8715 is slightly optimistic.

## 5. Inference latency

| Model | Measured time per query |
|---|---|
| TF-IDF + SVM (scikit-learn, CPU, Kaggle notebook) | 1.66 ms |
| MuRIL (Kaggle GPU) | 7.91 ms |
| Both, one after the other | about 9.6 ms |
| Demo app model (plain NumPy, CPU) | about 0.3 ms |

All are far below the 50 ms limit.

## 6. Reproduce the final submission CSV

1. Create a Kaggle notebook for the competition and attach its data. Turn on a GPU (T4) and internet.
2. Upload `datasprint_nlp.ipynb` and run all cells in order. The packages it needs (pandas, numpy, scikit-learn, torch, transformers) are already installed on Kaggle.
3. The last training cell writes `submission.csv` with 4,912 rows and all 56 classes.

Random seeds: data split `random_state=0`, augmentation seeds 0 (validation) and 1 (final), casual-copy seeds 100 and 101, MuRIL `torch.manual_seed(0)`. Fine-tuning on a GPU with fp16 is not bit-for-bit repeatable, so a small number of predictions can differ between runs. We have not measured how many.

## 7. Live demo

A Streamlit app runs the TF-IDF + SVM model only. It shows the predicted class, the top three scores, the script of the query and the response time. The two model files in this repo were exported from the notebook.

- Run locally: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`
- Hosted: https://svm-intent-classifier.streamlit.app
- The demo does not include MuRIL, so its predictions can differ from the leaderboard submission, which used the blend.
- The scores shown in the demo are relative model scores that add up to 100%. They are not calibrated probabilities.
