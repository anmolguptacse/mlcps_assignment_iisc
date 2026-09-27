import os, sys, time, json, argparse
os.chdir("/data2/home/budelalokesh/project _ML")
sys.path.insert(0, "/data2/home/budelalokesh/project _ML")
sys.path.insert(0, "/data2/home/budelalokesh/project _ML/scratch")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from features_utils import build_18_stats, build_8_tclb, build_rolling_entropy_features
from markov_module import fit_markov_tables, extract_markov_features
from advanced_feature_extractors import extract_static_sequence_features

def to_prob(m):
    return 1.0 / (1.0 + np.exp(-np.clip(m, -20.0, 20.0)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed for fold splitting and meta-learner")
    args = parser.parse_args()
    seed = args.seed
    
    print(f"================================================================================")
    print(f"BUILDING DEPLOYMENT MEMBER FOR SEED {seed}")
    print(f"================================================================================", flush=True)
    t_start = time.time()
    
    # 1. LOAD DATA
    print(f"[{seed}] Loading data...", flush=True)
    with open("data/train.json") as f:
        train_raw = [json.loads(line) for line in f]
    with open("data/test.json") as f:
        test_raw = [json.loads(line) for line in f]
        
    y_train = np.array([1 if r["label"] == "B" else 0 for r in train_raw], dtype=np.int64)
    n_train = len(y_train)
    n_test = len(test_raw)
    assert n_train == 10536, f"Expected 10536, got {n_train}"
    assert n_test == 3000, f"Expected 3000, got {n_test}"
    
    train_tokens = [r["text"] for r in train_raw]
    test_tokens = [r["text"] for r in test_raw]
    train_u_texts = ["".join(chr(0x1000 + t) for t in r["text"]) for r in train_raw]
    test_u_texts = ["".join(chr(0x1000 + t) for t in r["text"]) for r in test_raw]
    train_w_texts = [" ".join(str(t) for t in r["text"]) for r in train_raw]
    test_w_texts = [" ".join(str(t) for t in r["text"]) for r in test_raw]
    
    # 2. LOAD / EXTRACT DETERMINISTIC COMB75
    print(f"[{seed}] Loading Comb75 deterministic features...", flush=True)
    tr_d34 = np.hstack([build_18_stats(train_raw), build_8_tclb(train_raw), build_rolling_entropy_features(train_raw)])
    te_d34 = np.hstack([build_18_stats(test_raw), build_8_tclb(test_raw), build_rolling_entropy_features(test_raw)])
    
    tr_stat41 = np.load("scratch/static_features_41.npy")
    if os.path.exists("scratch/static_features_41_test.npy"):
        te_stat41 = np.load("scratch/static_features_41_test.npy")
    else:
        te_stat41, _ = extract_static_sequence_features([r["text"] for r in test_raw])
        np.save("scratch/static_features_41_test.npy", te_stat41)
        
    tr_comb75 = np.hstack([tr_d34, tr_stat41])
    te_comb75 = np.hstack([te_d34, te_stat41])
    
    # 3. 5-FOLD CV FOR IN-FOLD BASE OOF MATRIX
    print(f"[{seed}] Generating 5-Fold In-Fold Base OOF Matrix...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_base_matrix = np.zeros((n_train, 8), dtype=np.float64)
    test_base_preds_folds = np.zeros((5, n_test, 8), dtype=np.float64)
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_tokens, y_train), 1):
        f_t0 = time.time()
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        
        # Fit Markov tables exclusively on fold training partition
        m_tab = fit_markov_tables([train_tokens[i] for i in tr_idx], y_tr)
        f_pt_tr, f_m16_tr, f_win_tr = extract_markov_features([train_tokens[i] for i in tr_idx], m_tab)
        f_pt_va, f_m16_va, f_win_va = extract_markov_features([train_tokens[i] for i in va_idx], m_tab)
        f_pt_te, f_m16_te, f_win_te = extract_markov_features(test_tokens, m_tab)
        
        # Scalers fit on fold train
        sc75 = StandardScaler()
        d75_tr = sc75.fit_transform(tr_comb75[tr_idx])
        d75_va = sc75.transform(tr_comb75[va_idx])
        d75_te = sc75.transform(te_comb75)
        
        sc_pt = StandardScaler()
        d_pt_tr = np.hstack([d75_tr, sc_pt.fit_transform(f_pt_tr)])
        d_pt_va = np.hstack([d75_va, sc_pt.transform(f_pt_va)])
        d_pt_te = np.hstack([d75_te, sc_pt.transform(f_pt_te)])
        
        sc_m16 = StandardScaler()
        d_m16_tr = np.hstack([d75_tr, sc_m16.fit_transform(f_m16_tr)])
        d_m16_va = np.hstack([d75_va, sc_m16.transform(f_m16_va)])
        d_m16_te = np.hstack([d75_te, sc_m16.transform(f_m16_te)])
        
        sc_win = StandardScaler()
        d_win_tr = np.hstack([d75_tr, sc_win.fit_transform(f_win_tr)])
        d_win_va = np.hstack([d75_va, sc_win.transform(f_win_va)])
        d_win_te = np.hstack([d75_te, sc_win.transform(f_win_te)])
        
        # TF-IDF fit on fold train
        vc = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2, sublinear_tf=True)
        Xc_tr = vc.fit_transform([train_u_texts[i] for i in tr_idx])
        Xc_va = vc.transform([train_u_texts[i] for i in va_idx])
        Xc_te = vc.transform(test_u_texts)
        
        vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 5), min_df=2, sublinear_tf=True)
        Xw_tr = vw.fit_transform([train_w_texts[i] for i in tr_idx])
        Xw_va = vw.transform([train_w_texts[i] for i in va_idx])
        Xw_te = vw.transform(test_w_texts)
        
        # TruncatedSVD fit on fold train
        svd32 = TruncatedSVD(n_components=32, random_state=42)
        svd_tr = svd32.fit_transform(Xc_tr)
        svd_va = svd32.transform(Xc_va)
        svd_te = svd32.transform(Xc_te)
        sc_svd = StandardScaler()
        svd_tr_s = sc_svd.fit_transform(svd_tr)
        svd_va_s = sc_svd.transform(svd_va)
        svd_te_s = sc_svd.transform(svd_te)
        
        # 8 Base Models
        # Base 0: Char(1-5) + Comb75 (LinearSVC C=14)
        m0 = LinearSVC(C=14.0, max_iter=2500, tol=1e-3, random_state=42, dual="auto")
        m0.fit(sparse.hstack([Xc_tr, sparse.csr_matrix(d75_tr)]).tocsr(), y_tr)
        oof_base_matrix[va_idx, 0] = to_prob(m0.decision_function(sparse.hstack([Xc_va, sparse.csr_matrix(d75_va)]).tocsr()))
        test_base_preds_folds[fold-1, :, 0] = to_prob(m0.decision_function(sparse.hstack([Xc_te, sparse.csr_matrix(d75_te)]).tocsr()))
        
        # Base 1: Word(1-5) + Comb75 (LinearSVC C=10)
        m1 = LinearSVC(C=10.0, max_iter=2500, tol=1e-3, random_state=42, dual="auto")
        m1.fit(sparse.hstack([Xw_tr, sparse.csr_matrix(d75_tr)]).tocsr(), y_tr)
        oof_base_matrix[va_idx, 1] = to_prob(m1.decision_function(sparse.hstack([Xw_va, sparse.csr_matrix(d75_va)]).tocsr()))
        test_base_preds_folds[fold-1, :, 1] = to_prob(m1.decision_function(sparse.hstack([Xw_te, sparse.csr_matrix(d75_te)]).tocsr()))
        
        # Base 2: Pos-Trigram + Char(1-5) + Comb75 (LinearSVC C=14)
        m2 = LinearSVC(C=14.0, max_iter=2500, tol=1e-3, random_state=42, dual="auto")
        m2.fit(sparse.hstack([Xc_tr, sparse.csr_matrix(d_pt_tr)]).tocsr(), y_tr)
        oof_base_matrix[va_idx, 2] = to_prob(m2.decision_function(sparse.hstack([Xc_va, sparse.csr_matrix(d_pt_va)]).tocsr()))
        test_base_preds_folds[fold-1, :, 2] = to_prob(m2.decision_function(sparse.hstack([Xc_te, sparse.csr_matrix(d_pt_te)]).tocsr()))
        
        # Base 3: Pos-Markov + Char(1-5) + Comb75 (LinearSVC C=14)
        m3 = LinearSVC(C=14.0, max_iter=2500, tol=1e-3, random_state=42, dual="auto")
        m3.fit(sparse.hstack([Xc_tr, sparse.csr_matrix(d_m16_tr)]).tocsr(), y_tr)
        oof_base_matrix[va_idx, 3] = to_prob(m3.decision_function(sparse.hstack([Xc_va, sparse.csr_matrix(d_m16_va)]).tocsr()))
        test_base_preds_folds[fold-1, :, 3] = to_prob(m3.decision_function(sparse.hstack([Xc_te, sparse.csr_matrix(d_m16_te)]).tocsr()))
        
        # Base 4: Local-Window + Char(1-5) + Comb75 (LinearSVC C=14)
        m4 = LinearSVC(C=14.0, max_iter=2500, tol=1e-3, random_state=42, dual="auto")
        m4.fit(sparse.hstack([Xc_tr, sparse.csr_matrix(d_win_tr)]).tocsr(), y_tr)
        oof_base_matrix[va_idx, 4] = to_prob(m4.decision_function(sparse.hstack([Xc_va, sparse.csr_matrix(d_win_va)]).tocsr()))
        test_base_preds_folds[fold-1, :, 4] = to_prob(m4.decision_function(sparse.hstack([Xc_te, sparse.csr_matrix(d_win_te)]).tocsr()))
        
        # Base 5: Multi-text Fusion (Ridge a=10)
        m5 = RidgeClassifier(alpha=10.0, random_state=42)
        m5.fit(sparse.hstack([Xc_tr, Xw_tr, sparse.csr_matrix(d_pt_tr)]).tocsr(), y_tr)
        oof_base_matrix[va_idx, 5] = to_prob(m5.decision_function(sparse.hstack([Xc_va, Xw_va, sparse.csr_matrix(d_pt_va)]).tocsr()))
        test_base_preds_folds[fold-1, :, 5] = to_prob(m5.decision_function(sparse.hstack([Xc_te, Xw_te, sparse.csr_matrix(d_pt_te)]).tocsr()))
        
        # Base 6: HGB on SVD32 + Comb75 (using seed for tree stochasticity)
        m6 = HistGradientBoostingClassifier(max_depth=5, max_iter=250, random_state=seed)
        m6.fit(np.hstack([svd_tr_s, d75_tr]), y_tr)
        oof_base_matrix[va_idx, 6] = m6.predict_proba(np.hstack([svd_va_s, d75_va]))[:, 1]
        test_base_preds_folds[fold-1, :, 6] = m6.predict_proba(np.hstack([svd_te_s, d75_te]))[:, 1]
        
        # Base 7: PassiveAggressive on Char(1-5) + Comb75
        m7 = PassiveAggressiveClassifier(C=1.0, max_iter=2500, random_state=42)
        m7.fit(sparse.hstack([Xc_tr, sparse.csr_matrix(d75_tr)]).tocsr(), y_tr)
        oof_base_matrix[va_idx, 7] = to_prob(m7.decision_function(sparse.hstack([Xc_va, sparse.csr_matrix(d75_va)]).tocsr()))
        test_base_preds_folds[fold-1, :, 7] = to_prob(m7.decision_function(sparse.hstack([Xc_te, sparse.csr_matrix(d75_te)]).tocsr()))
        
        print(f"[{seed}] Fold {fold}/5 completed in {time.time()-f_t0:.1f}s", flush=True)
        
    # 4. FIT META-LEARNER ON SEED-SPECIFIC OOF BASE MATRIX
    print(f"[{seed}] Fitting Meta-Learner LogisticRegression(C=5.0)...", flush=True)
    meta_learner = LogisticRegression(C=5.0, max_iter=2000, random_state=seed)
    meta_learner.fit(oof_base_matrix, y_train)
    oof_meta_probs = meta_learner.predict_proba(oof_base_matrix)[:, 1]
    meta_oof_acc = accuracy_score(y_train, (oof_meta_probs >= 0.5).astype(int)) * 100
    meta_f1_a = f1_score(y_train, (oof_meta_probs >= 0.5).astype(int), pos_label=0)
    meta_f1_b = f1_score(y_train, (oof_meta_probs >= 0.5).astype(int), pos_label=1)
    
    print(f"[{seed}] Meta-Learner In-Fold OOF Accuracy: {meta_oof_acc:.4f}% | F1(A): {meta_f1_a:.4f} | F1(B): {meta_f1_b:.4f}", flush=True)
    print(f"[{seed}] Learned Meta Weights: {np.round(meta_learner.coef_[0], 3)} | Intercept: {meta_learner.intercept_[0]:.3f}", flush=True)
    
    # 5. REFIT ALL 8 BASE MODELS ON 100% OF TRAINING DATA
    print(f"[{seed}] Refitting 8 Base Models on 100% Training Data (N=10,536)...", flush=True)
    t_full = time.time()
    
    full_m_tab = fit_markov_tables(train_tokens, y_train)
    f_pt_full_tr, f_m16_full_tr, f_win_full_tr = extract_markov_features(train_tokens, full_m_tab)
    f_pt_full_te, f_m16_full_te, f_win_full_te = extract_markov_features(test_tokens, full_m_tab)
    
    full_sc75 = StandardScaler()
    d75_full_tr = full_sc75.fit_transform(tr_comb75)
    d75_full_te = full_sc75.transform(te_comb75)
    
    full_sc_pt = StandardScaler()
    d_pt_full_tr = np.hstack([d75_full_tr, full_sc_pt.fit_transform(f_pt_full_tr)])
    d_pt_full_te = np.hstack([d75_full_te, full_sc_pt.transform(f_pt_full_te)])
    
    full_sc_m16 = StandardScaler()
    d_m16_full_tr = np.hstack([d75_full_tr, full_sc_m16.fit_transform(f_m16_full_tr)])
    d_m16_full_te = np.hstack([d75_full_te, full_sc_m16.transform(f_m16_full_te)])
    
    full_sc_win = StandardScaler()
    d_win_full_tr = np.hstack([d75_full_tr, full_sc_win.fit_transform(f_win_full_tr)])
    d_win_full_te = np.hstack([d75_full_te, full_sc_win.transform(f_win_full_te)])
    
    full_vc = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2, sublinear_tf=True)
    Xc_full_tr = full_vc.fit_transform(train_u_texts)
    Xc_full_te = full_vc.transform(test_u_texts)
    
    full_vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 5), min_df=2, sublinear_tf=True)
    Xw_full_tr = full_vw.fit_transform(train_w_texts)
    Xw_full_te = full_vw.transform(test_w_texts)
    
    full_svd32 = TruncatedSVD(n_components=32, random_state=42)
    svd_full_tr = full_svd32.fit_transform(Xc_full_tr)
    svd_full_te = full_svd32.transform(Xc_full_te)
    full_sc_svd = StandardScaler()
    svd_full_tr_s = full_sc_svd.fit_transform(svd_full_tr)
    svd_full_te_s = full_sc_svd.transform(svd_full_te)
    
    test_base_preds_full = np.zeros((n_test, 8), dtype=np.float64)
    
    # Base 0
    m0_f = LinearSVC(C=14.0, max_iter=2500, tol=1e-3, random_state=42, dual="auto")
    m0_f.fit(sparse.hstack([Xc_full_tr, sparse.csr_matrix(d75_full_tr)]).tocsr(), y_train)
    test_base_preds_full[:, 0] = to_prob(m0_f.decision_function(sparse.hstack([Xc_full_te, sparse.csr_matrix(d75_full_te)]).tocsr()))
    
    # Base 1
    m1_f = LinearSVC(C=10.0, max_iter=2500, tol=1e-3, random_state=42, dual="auto")
    m1_f.fit(sparse.hstack([Xw_full_tr, sparse.csr_matrix(d75_full_tr)]).tocsr(), y_train)
    test_base_preds_full[:, 1] = to_prob(m1_f.decision_function(sparse.hstack([Xw_full_te, sparse.csr_matrix(d75_full_te)]).tocsr()))
    
    # Base 2
    m2_f = LinearSVC(C=14.0, max_iter=2500, tol=1e-3, random_state=42, dual="auto")
    m2_f.fit(sparse.hstack([Xc_full_tr, sparse.csr_matrix(d_pt_full_tr)]).tocsr(), y_train)
    test_base_preds_full[:, 2] = to_prob(m2_f.decision_function(sparse.hstack([Xc_full_te, sparse.csr_matrix(d_pt_full_te)]).tocsr()))
    
    # Base 3
    m3_f = LinearSVC(C=14.0, max_iter=2500, tol=1e-3, random_state=42, dual="auto")
    m3_f.fit(sparse.hstack([Xc_full_tr, sparse.csr_matrix(d_m16_full_tr)]).tocsr(), y_train)
    test_base_preds_full[:, 3] = to_prob(m3_f.decision_function(sparse.hstack([Xc_full_te, sparse.csr_matrix(d_m16_full_te)]).tocsr()))
    
    # Base 4
    m4_f = LinearSVC(C=14.0, max_iter=2500, tol=1e-3, random_state=42, dual="auto")
    m4_f.fit(sparse.hstack([Xc_full_tr, sparse.csr_matrix(d_win_full_tr)]).tocsr(), y_train)
    test_base_preds_full[:, 4] = to_prob(m4_f.decision_function(sparse.hstack([Xc_full_te, sparse.csr_matrix(d_win_full_te)]).tocsr()))
    
    # Base 5
    m5_f = RidgeClassifier(alpha=10.0, random_state=42)
    m5_f.fit(sparse.hstack([Xc_full_tr, Xw_full_tr, sparse.csr_matrix(d_pt_full_tr)]).tocsr(), y_train)
    test_base_preds_full[:, 5] = to_prob(m5_f.decision_function(sparse.hstack([Xc_full_te, Xw_full_te, sparse.csr_matrix(d_pt_full_te)]).tocsr()))
    
    # Base 6
    m6_f = HistGradientBoostingClassifier(max_depth=5, max_iter=250, random_state=seed)
    m6_f.fit(np.hstack([svd_full_tr_s, d75_full_tr]), y_train)
    test_base_preds_full[:, 6] = m6_f.predict_proba(np.hstack([svd_full_te_s, d75_full_te]))[:, 1]
    
    # Base 7
    m7_f = PassiveAggressiveClassifier(C=1.0, max_iter=2500, random_state=42)
    m7_f.fit(sparse.hstack([Xc_full_tr, sparse.csr_matrix(d75_full_tr)]).tocsr(), y_train)
    test_base_preds_full[:, 7] = to_prob(m7_f.decision_function(sparse.hstack([Xc_full_te, sparse.csr_matrix(d75_full_te)]).tocsr()))
    
    print(f"[{seed}] Full train refit completed in {time.time()-t_full:.1f}s", flush=True)
    
    # 6. INFER TEST PREDICTIONS USING META-LEARNER
    print(f"[{seed}] Generating test probabilities with meta-learner...", flush=True)
    test_probs = meta_learner.predict_proba(test_base_preds_full)[:, 1]
    
    # Also compute in-fold bagged test base prediction
    test_base_bagged = np.mean(test_base_preds_folds, axis=0)
    test_probs_infold_bagged = meta_learner.predict_proba(test_base_bagged)[:, 1]
    
    # 7. SAVE ARTIFACTS
    out_dir = "results/search_campaign/final_seed_models"
    os.makedirs(out_dir, exist_ok=True)
    
    test_prob_path = f"{out_dir}/seed{seed}_test_probs.npy"
    np.save(test_prob_path, test_probs)
    
    infold_test_path = f"{out_dir}/seed{seed}_test_probs_infold.npy"
    np.save(infold_test_path, test_probs_infold_bagged)
    
    np.save(f"{out_dir}/seed{seed}_oof_base_matrix.npy", oof_base_matrix)
    np.save(f"{out_dir}/seed{seed}_meta_oof_probs.npy", oof_meta_probs)
    
    meta_info = {
        "seed": seed,
        "meta_oof_acc": float(meta_oof_acc),
        "meta_f1_a": float(meta_f1_a),
        "meta_f1_b": float(meta_f1_b),
        "weights": [float(w) for w in meta_learner.coef_[0]],
        "intercept": float(meta_learner.intercept_[0]),
        "elapsed_sec": time.time() - t_start
    }
    with open(f"{out_dir}/seed{seed}_meta_info.json", "w") as f:
        json.dump(meta_info, f, indent=2)
        
    print(f"[{seed}] SUCCESS! Saved test probabilities to {test_prob_path}")
    print(f"[{seed}] Total elapsed time: {time.time()-t_start:.1f}s\n", flush=True)

if __name__ == "__main__":
    main()
