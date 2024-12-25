# Deep Funding 🌱

Exploring fair open-source dependency tree funding allocation by using AI models to allocate weights and validating results with human spot-checking.

## 🤔 Why Deep Funding?

Traditional public goods funding faces scaling challenges:

- Funders must evaluate each project individually
- Large-group funding leads to public campaigning
- Small-group funding creates favoritism and limited scale

Deep Funding solves this through:

1. Value as a graph: Instead of abstract "value to humanity", we measure relative contributions between dependencies
2. Distilled human judgment: An open market of AI models allocates weights, validated by random human spot-checks

This creates a scalable, efficient system that can handle millions of dependencies while maintaining human oversight. It avoids difficult abstract questions about absolute value and instead focuses on concrete, local comparisons ("compare A and B's direct impact on C"). Also creates an open market for allocation models that can scale beyond a small number of projects.

## 📦 Data

There are several datasets available. The main data we are dealing with are GitHub repositories and their dependencies. Additionally, we have a proxy dataset with weights derived from their relative funding amounts in the past.

## 🎯 Goal

The goal is predicting each repository relative importance based on how open source maintainers and a jury would rank them. For that, we need to compare each of these repos with one another and give a relative value between them, such that the total in each case adds up to 1.

## 📊 Evaluation

A human jury will perform detailed analysis on randomly selected edges, providing a score between 0 and 1 for each comparison. Only a small fraction of predictions are actually validated by humans. Models that consistently align with human judgments earn higher weights. This creates a fast, cheap, and credibly neutral mechanism that mirrors trustworthy human judgment.

The jury will evaluate:

1. Replaceability: How much time has the dependency saved you?
2. Specificity: Is the dependency specific to EVM or a general building block like aes or hashes/nobles
3. Additionality: Is the dependency already well funded?

The models will be evaluated on how well they align with the jury's scores using Mean Squared Error (MSE). Answers given by the model must be self-consistent, ie. for any triple _a_,_b_,_c_, `c/a = c/b * b/a`. Ensure mathematical consistency in outputs given to reflect logical relationships rather than reflecting biases from the training data.

## 🚀 Quickstart

Make sure you have [`uv` installed](https://docs.astral.sh/uv/). Then run the following command to install the dependencies.

```bash
uv sync
```

### 🔐 Environment

Create a `.env` file in the root directory with the following variables:

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to a Google Cloud service account with BigQuery access. You can create one on the [Google Cloud Console](https://console.cloud.google.com/iam-admin/serviceaccounts) and store it under an `env` folder.
- `GITHUB_TOKEN`: A GitHub personal access token with rate limiting. You can create one on [GitHub Developer Settings](https://github.com/settings/tokens?type=beta).

## 📚 Resources

- [Official Website](https://deepfunding.org) - [FAQ](https://deepfunding.org/faq)
- [GitHub Repository](https://github.com/deepfunding/dependency-graph)
- [HuggingFace Competition](https://huggingface.co/spaces/DeepFunding/PredictiveFundingChallengeforOpenSourceDependencies)
- [Demo of the Voting UI](https://pairwise-df-demo.vercel.app/allocation)
- [Deep Funding podcast](https://www.youtube.com/watch?v=ygaEBHYllPU)
