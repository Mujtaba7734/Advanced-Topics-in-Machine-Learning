# ATML Repository Instructions

This repository contains coursework for Advanced Topics in Machine Learning.

## General workflow
- Prefer reproducible Python scripts and configuration files over large stateful notebooks.
- Inspect existing code before modifying it.
- Make small, focused changes instead of broad refactors.
- Never silently alter datasets, seeds, evaluation settings, metrics, or experiment protocols.
- Keep the repository easy to audit with Git.

## Experiments
- Do not launch expensive GPU runs, large model downloads, or long experiments unless explicitly requested.
- Run cheap checks and smoke tests before expensive experiments.
- Preserve reproducibility.
- Save configurations, machine-readable metrics, and useful experiment outputs.

## Communication
- Explain mathematically meaningful changes clearly.
- Flag suspicious assumptions, inconsistencies, or possible bugs rather than silently working around them.
