# Predictive Funding Challenge 🌱

A fork of davidgasquez's repo:

- Resolved errors to enable it to run with 10GB RAM
- Added functionality in `src/train_decorrelate.py` to explicitly train models to make different errors from davidgasquez's model, with the goal of improving overall ensemble quality by explicitly increasing model diversity.

---

A machine learning challenge to predict relative funding between open source projects, aiming to create scalable and fair public goods funding allocation.

## 📦 Data

For the [contest](https://huggingface.co/spaces/DeepFunding/PredictiveFundingChallengeforOpenSourceDependencies), the dataset represents pairs of GitHub repositories and their funding amounts in historical funding rounds data.

## 🎯 Goal

The goal is predicting the relative funding received between any pair of projects. For that, we need to compare each of these repos with one another and give a relative value between them, such that the total in each case adds up to 1.

## 📊 Evaluation

*Winners* are decided based on *novelty and approach taken to predict answers for 1023 comparisons*. They are determined by their marginal contribution: how much better the final outcome is compared to if their submission (code or dataset) had never existed? That means that even if someone doesn't make a submission but provides a valuable dataset that all other contestants end up using, that would be rewarded.

The evaluation metric is **Mean Squared Error (MSE)**. The lower, the better.

Submission weights must be self-consistent, ie. for any triple _a_,_b_,_c_, `c/a = c/b * b/a`. Ensure mathematical consistency in outputs given to reflect logical relationships rather than reflecting biases from the training data.

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
- [Deep Funding Podcast](https://www.youtube.com/watch?v=ygaEBHYllPU)
