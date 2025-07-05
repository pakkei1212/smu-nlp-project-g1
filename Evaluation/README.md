# Evaluation Module — Personal AI Travel Agent

This module evaluates generated itineraries from the **Personal AI Travel Agent** system. It provides both **LLM-based judgment** and **language model perplexity** scoring to assess the quality, fluency, and personalization of itinerary outputs.

## Directory Contents

```
Evaluation/
├── LLM-as-a-judge.ipynb                 # Evaluation with GPT-based judgment (e.g., GPT-4/Gemini)
├── Perplexity.ipynb                     # Evaluate output fluency via perplexity
├── merged_output_with_gemini_evaluation.csv   # Judged results from Gemini
├── merged_output_with_gemini_evaluation_temp.csv
├── merged_output_with_perplexity.csv    # Perplexity scores per itinerary
├── query_responses_persona.csv          # Generated itinerary outputs with persona context
```

## Key Features

### ✅ LLM-as-a-Judge Evaluation
- Uses LLMs (e.g., GPT-4 or Gemini) to **compare two generated itineraries** and decide which is better.
- Judgment is based on:
  - Helpfulness
  - Personalization
  - Factual accuracy
  - Fluency

### Perplexity Scoring
- Measures how **fluent** or **natural** the itinerary outputs are using perplexity from a reference language model (e.g., GPT-2 or other Hugging Face models).

### Output Merging & Analysis
- Evaluation outputs are stored in CSVs for comparison and plotting:
  - `merged_output_with_gemini_evaluation.csv`
  - `merged_output_with_perplexity.csv`

## How to Run

### 1. LLM-Based Judgment

Open the notebook:
```bash
LLM-as-a-judge.ipynb
```

Steps:
- Configure your API key (OpenAI or Gemini)
- Load itinerary responses from `query_responses_persona.csv`
- Run side-by-side comparisons
- Store results in `merged_output_with_gemini_evaluation.csv`

### 2. Perplexity Evaluation

Open the notebook:
```bash
Perplexity.ipynb
```

Steps:
- Load outputs from `query_responses_persona.csv`
- Compute perplexity scores for each output
- Output results to `merged_output_with_perplexity.csv`

## 📌 Notes

- `query_responses_persona.csv` includes the generated itineraries for various user queries and personas.
- Both evaluation notebooks expect the data to be in this format:
  ```
  | user_query | persona | response_model_1 | response_model_2 |
  ```

- Gemini or OpenAI API keys must be set in the environment or notebook for LLM-based comparison to work.

## 📄 License

Specify your license here (e.g., MIT, Apache 2.0).
