===============================================================================
SUBMISSION 2: submission_candidate_5seed_bagged.csv
===============================================================================

DESCRIPTION:
  5-Seed Bagged Ensemble. Same 7-component stacking pipeline as Submission 1,
  but trained with 5 different random seeds and predictions averaged.
  Seeds: 42, 123, 2026, 3407, 7777

HOW TO REPRODUCE:
  1. Place data/train.json, data/test.json in the project root under data/
  2. Place features_utils.py, markov_module.py, advanced_feature_extractors.py
     in the project root (or adjust sys.path in scripts)
  3. Run the seed deployment:
       python run_all_deployment_seeds.py
     This launches 5 parallel workers (build_seed_deployment_member.py --seed N)
     and saves per-seed test probabilities to:
       results/search_campaign/final_seed_models/seed{N}_test_probs.npy
  4. Then run the bagging aggregator:
       python build_and_evaluate_5seed_bagged_ensemble.py
  5. Output: submission_candidate_5seed_bagged.csv

FILES:
  - run_all_deployment_seeds.py               (orchestrator, launches 5 seeds)
  - build_seed_deployment_member.py           (per-seed worker, 312 lines)
  - build_and_evaluate_5seed_bagged_ensemble.py (bagging aggregator, 157 lines)
  - features_utils.py                         (34 deterministic features)
  - markov_module.py                          (Markov chain feature extraction)
  - advanced_feature_extractors.py            (additional sequence features)
  - submission_candidate_5seed_bagged.csv     (the submitted output file)

PIPELINE ARCHITECTURE:
  Same 7-component stacking as Submission 1, but:
  - Each seed uses a different StratifiedKFold random_state
  - 5 sets of test probabilities are generated independently
  - Final predictions = mean of 5 seed probabilities, threshold 0.5
