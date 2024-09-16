# SwiftBERT

This repository is associated with our paper: 
Understanding, Uncovering, and Mitigating the Causes of Inference Slowdown for Language Models
https://openreview.net/forum?id=homi48OtHu

This project is organized into 4 main folders of code:
1. Attacking
2. Defending
3. Early Exit Models
4. Misc. Experiments
5. Utils

Here is a list that summarizes all of the processes that are implemented in these four main folders:
 
1. Attacking

Implementations of various slowdown attacks that we experimented with in our paper.

- Exit classification:
    - create-dataset.py
    - ExitClassifier.py
    - generate-exitclassifier-attack.py
    - train-test-exit-classifier.py
- Misc. attacks: 
    - random-replacement.py
- SlowBERT:
    - generate-slowbert-attack.py
    - slowbert-utils.py

2. Defending

Implementing our novel slowdown defense, expedited adversarial training, which we introduced in our paper.

- augment-dataset.py
- test-robust-model.py
- train-robust-model.py

3. Early Exit Models

Implementations for two BERT- and RoBERTa-based early exit models: DeeBERT [cite] and PABEE [cite]. We copied this code from [link]. Note that in our paper, we only experimented with BERT-based PABEE models.

- DeeBERT:
    - modeling_highway_bert.py
    - modeling_highway_roberta.py
- PABEE:
    - modeling-payee-albert.py
    - modeling_pabee_bert.py
- train-test-bert-mem.py

4. Misc. experiments

Implementations for three extra slowdown-related experiments that we featured in our paper.

- cloze.py
- embedding-distances.py
- paraphrase.py
