import math
import zlib
import gzip
import numpy as np
from collections import Counter

def extract_static_sequence_features(tokens_list):
    """
    Computes label-independent mathematical, information-theoretic, compression,
    numerical geometry, positional, and spectral features for each sequence.
    These features do NOT depend on labels, but will be scaled strictly in-fold.
    """
    n_samples = len(tokens_list)
    feature_names = []
    features = []

    for idx, tokens in enumerate(tokens_list):
        L = len(tokens)
        t_arr = np.array(tokens, dtype=np.float64)
        counts = Counter(tokens)
        u_tokens = len(counts)
        freqs = np.array(list(counts.values()), dtype=np.float64)
        probs = freqs / L

        # 1. Length & Vocabulary Richness
        f_len = float(L)
        f_log_len = math.log(L) if L > 0 else 0.0
        f_unique = float(u_tokens)
        f_ttr = u_tokens / L if L > 0 else 0.0
        f_singletons = sum(1 for c in counts.values() if c == 1)
        f_singleton_ratio = f_singletons / L if L > 0 else 0.0
        f_max_freq_ratio = float(np.max(freqs)) / L if L > 0 else 0.0

        # 2. Information Theory (Shannon, Renyi-2, Simpson, Gini)
        h_shannon = -float(np.sum(probs * np.log2(probs + 1e-12)))
        h_renyi2 = -math.log2(float(np.sum(probs ** 2)) + 1e-12)
        simpson = float(np.sum(probs ** 2))
        gini = 1.0 - simpson

        # 3. Compression & Complexity (zlib, gzip, run-length)
        raw_bytes = bytes([t % 256 for t in tokens])
        comp_zlib = len(zlib.compress(raw_bytes, level=6))
        comp_gzip = len(gzip.compress(raw_bytes, compresslevel=6))
        ratio_zlib = comp_zlib / (L + 1e-6)
        ratio_gzip = comp_gzip / (L + 1e-6)

        runs = []
        cur_run = 1
        for i in range(1, L):
            if tokens[i] == tokens[i-1]:
                cur_run += 1
            else:
                runs.append(cur_run)
                cur_run = 1
        runs.append(cur_run)
        runs_arr = np.array(runs, dtype=np.float64)
        mean_run = float(np.mean(runs_arr))
        max_run = float(np.max(runs_arr))
        adjacent_rep_rate = sum(1 for r in runs if r > 1) / (len(runs) + 1e-6)

        # 4. Token-ID Numerical Geometry
        t_mean = float(np.mean(t_arr))
        t_std = float(np.std(t_arr))
        t_min = float(np.min(t_arr))
        t_max = float(np.max(t_arr))
        t_q25 = float(np.percentile(t_arr, 25))
        t_median = float(np.median(t_arr))
        t_q75 = float(np.percentile(t_arr, 75))
        t_iqr = t_q75 - t_q25

        if L > 1:
            diffs = np.diff(t_arr)
            abs_diffs = np.abs(diffs)
            diff_mean = float(np.mean(diffs))
            diff_std = float(np.std(diffs))
            abs_diff_mean = float(np.mean(abs_diffs))
            abs_diff_max = float(np.max(abs_diffs))
            sign_changes = float(np.sum(diffs[:-1] * diffs[1:] < 0)) / (L - 1)
        else:
            diff_mean = diff_std = abs_diff_mean = abs_diff_max = sign_changes = 0.0

        even_ratio = float(np.sum(np.array(tokens) % 2 == 0)) / L
        mod10_entropy = -float(np.sum([p * math.log2(p + 1e-12) for p in np.bincount(np.array(tokens) % 10, minlength=10) / L]))

        # 5. Positional Structure
        q1_end = max(1, L // 4)
        q3_start = max(1, 3 * L // 4)
        early_tokens = tokens[:q1_end]
        late_tokens = tokens[q3_start:]

        early_ttr = len(set(early_tokens)) / len(early_tokens) if len(early_tokens) > 0 else 0.0
        late_ttr = len(set(late_tokens)) / len(late_tokens) if len(late_tokens) > 0 else 0.0
        ttr_diff = late_ttr - early_ttr

        if L > 2:
            pos_norm = np.linspace(-1.0, 1.0, L)
            slope = float(np.dot(pos_norm, t_arr - t_mean) / np.dot(pos_norm, pos_norm))
        else:
            slope = 0.0

        # 6. Spectral / Fourier Features
        fft_vals = np.abs(np.fft.rfft(t_arr - t_mean))
        if len(fft_vals) > 1:
            fft_norm = fft_vals[1:] / (np.sum(fft_vals[1:]) + 1e-12)
            spectral_entropy = -float(np.sum(fft_norm * np.log2(fft_norm + 1e-12)))
            spectral_centroid = float(np.sum(np.arange(1, len(fft_vals)) * fft_norm)) / len(fft_vals)
            low_freq_ratio = float(np.sum(fft_vals[1:max(2, len(fft_vals)//4)])) / (np.sum(fft_vals[1:]) + 1e-12)
        else:
            spectral_entropy = spectral_centroid = low_freq_ratio = 0.0

        # 7. N-gram Diversity
        bigrams = [tuple(tokens[i:i+2]) for i in range(L-1)]
        trigrams = [tuple(tokens[i:i+3]) for i in range(L-2)]
        u_bigrams = len(set(bigrams))
        u_trigrams = len(set(trigrams))
        bigram_div = u_bigrams / (L - 1) if L > 1 else 0.0
        trigram_div = u_trigrams / (L - 2) if L > 2 else 0.0

        row = [
            f_len, f_log_len, f_unique, f_ttr, f_singleton_ratio, f_max_freq_ratio,
            h_shannon, h_renyi2, simpson, gini,
            comp_zlib, comp_gzip, ratio_zlib, ratio_gzip,
            mean_run, max_run, adjacent_rep_rate,
            t_mean, t_std, t_min, t_max, t_q25, t_median, t_q75, t_iqr,
            diff_mean, diff_std, abs_diff_mean, abs_diff_max, sign_changes,
            even_ratio, mod10_entropy,
            early_ttr, late_ttr, ttr_diff, slope,
            spectral_entropy, spectral_centroid, low_freq_ratio,
            bigram_div, trigram_div
        ]
        features.append(row)

    feature_names = [
        "f_len", "f_log_len", "f_unique", "f_ttr", "f_singleton_ratio", "f_max_freq_ratio",
        "h_shannon", "h_renyi2", "simpson", "gini",
        "comp_zlib", "comp_gzip", "ratio_zlib", "ratio_gzip",
        "mean_run", "max_run", "adjacent_rep_rate",
        "t_mean", "t_std", "t_min", "t_max", "t_q25", "t_median", "t_q75", "t_iqr",
        "diff_mean", "diff_std", "abs_diff_mean", "abs_diff_max", "sign_changes",
        "even_ratio", "mod10_entropy",
        "early_ttr", "late_ttr", "ttr_diff", "slope",
        "spectral_entropy", "spectral_centroid", "low_freq_ratio",
        "bigram_div", "trigram_div"
    ]
    return np.array(features, dtype=np.float64), feature_names

class InFoldClassLM:
    """
    High-speed class-conditional language model log-likelihoods and derived features.
    Fitted strictly on training indices tr_idx.
    """
    def __init__(self, alpha=0.01, trigram_lambda=0.6, bigram_lambda=0.3, unigram_lambda=0.1):
        self.alpha = alpha
        self.trigram_lambda = trigram_lambda
        self.bigram_lambda = bigram_lambda
        self.unigram_lambda = unigram_lambda

    def fit(self, tokens_list, y_tr, tr_idx):
        self.uni_c0 = Counter()
        self.uni_c1 = Counter()
        self.bi_c0 = Counter()
        self.bi_c1 = Counter()
        self.rev_c0 = Counter()
        self.rev_c1 = Counter()
        self.pos_bi_c0 = [Counter() for _ in range(10)]
        self.pos_bi_c1 = [Counter() for _ in range(10)]
        self.tri_c0 = Counter()
        self.tri_c1 = Counter()

        self.tot0 = 0
        self.tot1 = 0
        vocab = set()

        for i, idx in enumerate(tr_idx):
            label = int(y_tr[i])
            toks = tokens_list[idx]
            L = len(toks)
            if label == 0:
                self.tot0 += L
                for pos, t in enumerate(toks):
                    vocab.add(t)
                    self.uni_c0[t] += 1
                    decile = min(9, int(10 * pos / L)) if L > 0 else 0
                    if pos > 0:
                        prev_t = toks[pos - 1]
                        self.bi_c0[(prev_t, t)] += 1
                        self.rev_c0[(t, prev_t)] += 1
                        self.pos_bi_c0[decile][(prev_t, t)] += 1
                    if pos > 1:
                        self.tri_c0[(toks[pos-2], toks[pos-1], t)] += 1
            else:
                self.tot1 += L
                for pos, t in enumerate(toks):
                    vocab.add(t)
                    self.uni_c1[t] += 1
                    decile = min(9, int(10 * pos / L)) if L > 0 else 0
                    if pos > 0:
                        prev_t = toks[pos - 1]
                        self.bi_c1[(prev_t, t)] += 1
                        self.rev_c1[(t, prev_t)] += 1
                        self.pos_bi_c1[decile][(prev_t, t)] += 1
                    if pos > 1:
                        self.tri_c1[(toks[pos-2], toks[pos-1], t)] += 1

        self.V = max(1, len(vocab))
        self.log_denom_u0 = math.log(self.tot0 + self.alpha * self.V)
        self.log_denom_u1 = math.log(self.tot1 + self.alpha * self.V)

        # Precompute prefix sums for bigrams and reverse bigrams
        self.bi_pref0 = Counter()
        for (prev, curr), cnt in self.bi_c0.items():
            self.bi_pref0[prev] += cnt
        self.bi_pref1 = Counter()
        for (prev, curr), cnt in self.bi_c1.items():
            self.bi_pref1[prev] += cnt

        self.rev_pref0 = Counter()
        for (curr, prev), cnt in self.rev_c0.items():
            self.rev_pref0[curr] += cnt
        self.rev_pref1 = Counter()
        for (curr, prev), cnt in self.rev_c1.items():
            self.rev_pref1[curr] += cnt

        self.pos_pref0 = [Counter() for _ in range(10)]
        for d in range(10):
            for (prev, curr), cnt in self.pos_bi_c0[d].items():
                self.pos_pref0[d][prev] += cnt
        self.pos_pref1 = [Counter() for _ in range(10)]
        for d in range(10):
            for (prev, curr), cnt in self.pos_bi_c1[d].items():
                self.pos_pref1[d][prev] += cnt

        self.tri_pref0 = Counter()
        for (p2, p1, curr), cnt in self.tri_c0.items():
            self.tri_pref0[(p2, p1)] += cnt
        self.tri_pref1 = Counter()
        for (p2, p1, curr), cnt in self.tri_c1.items():
            self.tri_pref1[(p2, p1)] += cnt

        return self

    def score_sequence(self, toks):
        L = len(toks)
        if L == 0:
            return np.zeros(24, dtype=np.float64)

        ll_uni_0 = 0.0
        ll_uni_1 = 0.0
        ll_bi_0 = 0.0
        ll_bi_1 = 0.0
        ll_rev_0 = 0.0
        ll_rev_1 = 0.0
        ll_pos_0 = 0.0
        ll_pos_1 = 0.0
        ll_tri_0 = 0.0
        ll_tri_1 = 0.0

        surp_diffs = []
        pos_ll_diff_early = 0.0
        pos_ll_diff_mid = 0.0
        pos_ll_diff_late = 0.0

        q1_end = max(1, L // 4)
        q3_start = max(1, 3 * L // 4)
        alpha = self.alpha
        V = self.V

        for pos, t in enumerate(toks):
            # Unigram
            c_u0 = self.uni_c0.get(t, 0)
            c_u1 = self.uni_c1.get(t, 0)
            p_u0 = (c_u0 + alpha) / (self.tot0 + alpha * V)
            p_u1 = (c_u1 + alpha) / (self.tot1 + alpha * V)
            ll_uni_0 += math.log(p_u0)
            ll_uni_1 += math.log(p_u1)

            decile = min(9, int(10 * pos / L))

            if pos > 0:
                prev_t = toks[pos - 1]
                # Bigram
                c_b0 = self.bi_c0.get((prev_t, t), 0)
                c_b1 = self.bi_c1.get((prev_t, t), 0)
                tot_b0 = self.bi_pref0.get(prev_t, 0)
                tot_b1 = self.bi_pref1.get(prev_t, 0)
                p_b0 = (c_b0 + alpha) / (tot_b0 + alpha * V)
                p_b1 = (c_b1 + alpha) / (tot_b1 + alpha * V)
                ll_bi_0 += math.log(p_b0)
                ll_bi_1 += math.log(p_b1)

                # Reverse Bigram
                c_r0 = self.rev_c0.get((t, prev_t), 0)
                c_r1 = self.rev_c1.get((t, prev_t), 0)
                tot_r0 = self.rev_pref0.get(t, 0)
                tot_r1 = self.rev_pref1.get(t, 0)
                p_r0 = (c_r0 + alpha) / (tot_r0 + alpha * V)
                p_r1 = (c_r1 + alpha) / (tot_r1 + alpha * V)
                ll_rev_0 += math.log(p_r0)
                ll_rev_1 += math.log(p_r1)

                # Positional Decile Bigram
                c_p0 = self.pos_bi_c0[decile].get((prev_t, t), 0)
                c_p1 = self.pos_bi_c1[decile].get((prev_t, t), 0)
                tot_p0 = self.pos_pref0[decile].get(prev_t, 0)
                tot_p1 = self.pos_pref1[decile].get(prev_t, 0)
                p_p0 = (c_p0 + alpha) / (tot_p0 + alpha * V)
                p_p1 = (c_p1 + alpha) / (tot_p1 + alpha * V)
                ll_pos_0 += math.log(p_p0)
                ll_pos_1 += math.log(p_p1)

                # Trigram with backoff
                if pos > 1:
                    prev2_t = (toks[pos-2], toks[pos-1])
                    c_t0 = self.tri_c0.get((toks[pos-2], toks[pos-1], t), 0)
                    c_t1 = self.tri_c1.get((toks[pos-2], toks[pos-1], t), 0)
                    tot_t0 = self.tri_pref0.get(prev2_t, 0)
                    tot_t1 = self.tri_pref1.get(prev2_t, 0)
                    p_t0_raw = (c_t0 + alpha) / (tot_t0 + alpha * V)
                    p_t1_raw = (c_t1 + alpha) / (tot_t1 + alpha * V)
                    p_tri0 = self.trigram_lambda * p_t0_raw + self.bigram_lambda * p_b0 + self.unigram_lambda * p_u0
                    p_tri1 = self.trigram_lambda * p_t1_raw + self.bigram_lambda * p_b1 + self.unigram_lambda * p_u1
                else:
                    p_tri0 = 0.7 * p_b0 + 0.3 * p_u0
                    p_tri1 = 0.7 * p_b1 + 0.3 * p_u1

                ll_tri_0 += math.log(p_tri0)
                ll_tri_1 += math.log(p_tri1)

                step_delta = math.log(p_tri1) - math.log(p_tri0)
                surp_diffs.append(step_delta)

                if pos < q1_end:
                    pos_ll_diff_early += step_delta
                elif pos >= q3_start:
                    pos_ll_diff_late += step_delta
                else:
                    pos_ll_diff_mid += step_delta

        delta_uni = ll_uni_1 - ll_uni_0
        delta_bi = ll_bi_1 - ll_bi_0
        delta_rev = ll_rev_1 - ll_rev_0
        delta_tri = ll_tri_1 - ll_tri_0
        delta_pos = ll_pos_1 - ll_pos_0

        if len(surp_diffs) > 0:
            s_arr = np.array(surp_diffs, dtype=np.float64)
            s_mean = float(np.mean(s_arr))
            s_std = float(np.std(s_arr))
            s_min = float(np.min(s_arr))
            s_max = float(np.max(s_arr))
            s_q25 = float(np.percentile(s_arr, 25))
            s_q75 = float(np.percentile(s_arr, 75))
            frac_pos = float(np.sum(s_arr > 0)) / len(s_arr)
        else:
            s_mean = s_std = s_min = s_max = s_q25 = s_q75 = frac_pos = 0.0

        return np.array([
            delta_uni, delta_uni / L,
            delta_bi, delta_bi / L,
            delta_rev, delta_rev / L,
            delta_tri, delta_tri / L,
            delta_pos, delta_pos / L,
            pos_ll_diff_early / max(1, q1_end),
            pos_ll_diff_mid / max(1, q3_start - q1_end),
            pos_ll_diff_late / max(1, L - q3_start),
            s_mean, s_std, s_min, s_max, s_q25, s_q75, frac_pos,
            ll_uni_1 / L, ll_uni_0 / L,
            ll_tri_1 / L, ll_tri_0 / L
        ], dtype=np.float64)

    def transform(self, tokens_list, idxs):
        features = [self.score_sequence(tokens_list[idx]) for idx in idxs]
        return np.array(features, dtype=np.float64)
