# Limitations

While our ML Ensemble Meta-Model is highly robust, there are certain limitations to our approach due to the constraints of the hackathon dataset.

### 1. Lack of Training Data (Unsupervised Ensemble)
Because the dataset does not include historical ground-truth labels (i.e., we are not given examples of "successful" vs "failed" agreements to learn from), we could not train a true Supervised Stacking Classifier (like Logistic Regression or XGBoost).
**Workaround:** We utilized an unsupervised "Soft Voting" approach where we manually defined the Meta-Aggregator weights (40% Political, 30% Economic, 30% NLP Risk). In a real-world scenario with labeled historical data, these weights would be dynamically learned by a Meta-Model.

### 2. NLP Context Nuance
Our NLP Engine utilizes `TextBlob` for sentiment analysis on objection reasons. While effective at catching explicitly negative words ("costly", "risk"), lightweight sentiment analyzers struggle with deep sarcasm or highly domain-specific political jargon. 
**Workaround:** We clamped the NLP risk score between 0 and 1 and restricted its overall ensemble weight to 30% to prevent edge-case linguistic misunderstandings from completely derailing a proposal.

### 3. Tiebreaker Scenarios
In the event that two proposals have the exact same priority, objection severity, and NLP sentiment, the engine defaults to Pandas sort order to break the tie. Advanced semantic similarity clustering (e.g., using `spaCy` or HuggingFace embeddings) could be implemented in future versions to group similar proposals and favor ones that fulfill missing categorical niches.
