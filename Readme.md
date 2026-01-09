# Lab 2 – Automated Training and Metric Reporting using GitHub Actions

## Objective
This project demonstrates CI-driven machine learning using GitHub Actions.
Each commit triggers automated training, evaluation, and artifact storage.

## Dataset
Wine Quality Dataset (UCI Repository)

## Evaluation Metrics
- Mean Squared Error (MSE)
- R² Score

## Project Structure
- `dataset/` – Wine quality datasets
- `train.py` – Training and evaluation script
- `outputs/` – Generated model and metrics
- `.github/workflows/` – GitHub Actions workflow

## How It Works
1. Push code changes to main branch
2. GitHub Actions automatically:
   - Trains the model
   - Computes metrics
   - Displays results in Job Summary
   - Stores model and results as artifacts

## Author
Druva Kumar  
Roll No: 2022BCD0018
