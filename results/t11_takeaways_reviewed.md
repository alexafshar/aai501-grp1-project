# T11 — Exploratory Chart Takeaways

1. **Hallucination rate by model:** In this dataset, mistral-7B-instruct has the highest observed hallucination rate (67.0%, n=2,911), while gpt-3.5-turbo-0613 has the lowest (13.8%, n=2,906). These descriptive rates may be affected by differences in task composition.
2. **Hallucination rate by task:** In this dataset, Data2txt has the highest observed hallucination rate (68.6%, n=6,195), while QA has the lowest (29.6%, n=5,767).
3. **Answer length:** In this dataset, hallucinated responses have a higher median word count than faithful responses (140 versus 113 words, respectively).
4. **Feature correlations:** Among the five numeric features, the strongest absolute Pearson correlation is between missing_names and answer_length (r = 0.59); this does not establish causation.

These are descriptive findings from the cleaned RAGTruth sample, not causal conclusions.
