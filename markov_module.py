from collections import Counter, defaultdict
import numpy as np

def fit_markov_tables(tokens_sub, y_sub):
    pos_tri_A = [Counter() for _ in range(10)]
    pos_tri_B = [Counter() for _ in range(10)]
    pos_bi_A = [Counter() for _ in range(10)]
    pos_bi_B = [Counter() for _ in range(10)]
    pos_uni_A = [Counter() for _ in range(10)]
    pos_uni_B = [Counter() for _ in range(10)]
    
    glob_tri_A, glob_tri_B = Counter(), Counter()
    glob_bi_A, glob_bi_B = Counter(), Counter()
    glob_u_A, glob_u_B = Counter(), Counter()
    tot_A, tot_B = 0, 0
    vocab = set()
    
    for i in range(len(tokens_sub)):
        seq = tokens_sub[i]
        lbl = y_sub[i]
        L = len(seq)
        for t in seq: vocab.add(t)
        if lbl == 0: tot_A += L
        else: tot_B += L
        
        for j in range(L-1):
            pair = (seq[j], seq[j+1])
            d_bi = min(9, int(10 * (j+1) / max(1, L)))
            if lbl == 0:
                pos_bi_A[d_bi][pair] += 1; pos_uni_A[d_bi][seq[j]] += 1
                glob_bi_A[pair] += 1; glob_u_A[seq[j]] += 1
            else:
                pos_bi_B[d_bi][pair] += 1; pos_uni_B[d_bi][seq[j]] += 1
                glob_bi_B[pair] += 1; glob_u_B[seq[j]] += 1
                
        for j in range(L-2):
            tri = (seq[j], seq[j+1], seq[j+2])
            d_tri = min(9, int(10 * (j+2) / max(1, L)))
            if lbl == 0:
                pos_tri_A[d_tri][tri] += 1; glob_tri_A[tri] += 1
            else:
                pos_tri_B[d_tri][tri] += 1; glob_tri_B[tri] += 1
                
        if L > 0:
            if lbl == 0: glob_u_A[seq[-1]] += 1
            else: glob_u_B[seq[-1]] += 1
            
    V = len(vocab)
    total_u_A = sum(glob_u_A.values()) + V
    total_u_B = sum(glob_u_B.values()) + V
    
    return {
        "pos_tri_A": pos_tri_A, "pos_tri_B": pos_tri_B,
        "pos_bi_A": pos_bi_A, "pos_bi_B": pos_bi_B,
        "pos_uni_A": pos_uni_A, "pos_uni_B": pos_uni_B,
        "glob_tri_A": glob_tri_A, "glob_tri_B": glob_tri_B,
        "glob_bi_A": glob_bi_A, "glob_bi_B": glob_bi_B,
        "glob_u_A": glob_u_A, "glob_u_B": glob_u_B,
        "tot_A": tot_A, "tot_B": tot_B,
        "total_u_A": total_u_A, "total_u_B": total_u_B,
        "V": V
    }

def extract_markov_features(tokens_sub, tables):
    pos_tri_A, pos_tri_B = tables["pos_tri_A"], tables["pos_tri_B"]
    pos_bi_A, pos_bi_B = tables["pos_bi_A"], tables["pos_bi_B"]
    pos_uni_A, pos_uni_B = tables["pos_uni_A"], tables["pos_uni_B"]
    glob_tri_A, glob_tri_B = tables["glob_tri_A"], tables["glob_tri_B"]
    glob_bi_A, glob_bi_B = tables["glob_bi_A"], tables["glob_bi_B"]
    glob_u_A, glob_u_B = tables["glob_u_A"], tables["glob_u_B"]
    tot_A, tot_B = tables["tot_A"], tables["tot_B"]
    total_u_A, total_u_B = tables["total_u_A"], tables["total_u_B"]
    V = tables["V"]
    
    w_pt, w_gt, w_pb, w_gb, w_u = 0.40, 0.30, 0.15, 0.10, 0.05
    alpha_tri = 0.05
    alpha_bi = 0.1
    windows = [8, 16, 32, 64]
    l3, l2, l1 = 0.6, 0.3, 0.1
    
    f_pt_list, f_m16_list, f_win_list = [], [], []
    
    for seq in tokens_sub:
        L = len(seq)
        if L <= 2:
            f_pt_list.append([0.0] * 13)
            f_m16_list.append([0.0] * 16)
            f_win_list.append([0.0] * 36)
            continue
            
        # 1. Bigram pass (j = 1 to L-1)
        logp_A_bi, logp_B_bi = 0.0, 0.0
        bi_diffs = []
        for j in range(1, L):
            p_prev = seq[j-1]
            p_curr = seq[j]
            p_A_bi = (glob_bi_A.get((p_prev, p_curr), 0) + alpha_bi) / (glob_u_A.get(p_prev, 0) + alpha_bi * V)
            p_B_bi = (glob_bi_B.get((p_prev, p_curr), 0) + alpha_bi) / (glob_u_B.get(p_prev, 0) + alpha_bi * V)
            lp_A, lp_B = np.log(p_A_bi), np.log(p_B_bi)
            logp_A_bi += lp_A; logp_B_bi += lp_B
            bi_diffs.append(lp_A - lp_B)
        bi_diffs = np.array(bi_diffs)
        delta_bi = logp_A_bi - logp_B_bi
        f_bi_8 = [delta_bi, delta_bi/(L-1), np.mean(bi_diffs), np.std(bi_diffs), np.min(bi_diffs), np.max(bi_diffs), np.mean(bi_diffs>0), logp_A_bi/(logp_B_bi+1e-9)]
        
        # 2. Unified Trigram + Pos-Trigram + Window pass (i = 2 to L-1)
        logp_A_pt, logp_B_pt = 0.0, 0.0
        pt_diffs = []
        d0_diffs, d9_diffs = [], []
        
        logp_A_tri, logp_B_tri = 0.0, 0.0
        tri_diffs = []
        win_deltas = []
        
        for i in range(2, L):
            t1, t2, t3 = seq[i-2], seq[i-1], seq[i]
            d = min(9, int(10 * i / L))
            
            # Shared dictionary lookups
            c_tri_A = glob_tri_A.get((t1, t2, t3), 0)
            c_tri_B = glob_tri_B.get((t1, t2, t3), 0)
            c_bi_A  = glob_bi_A.get((t2, t3), 0)
            c_bi_B  = glob_bi_B.get((t2, t3), 0)
            c_bi_ctx_A = glob_bi_A.get((t1, t2), 0)
            c_bi_ctx_B = glob_bi_B.get((t1, t2), 0)
            c_u_prev_A = glob_u_A.get(t2, 0)
            c_u_prev_B = glob_u_B.get(t2, 0)
            c_u_curr_A = glob_u_A.get(t3, 0)
            c_u_curr_B = glob_u_B.get(t3, 0)
            
            # --- Positional Trigram ---
            p_pt_A = (pos_tri_A[d].get((t1, t2, t3), 0) + alpha_tri) / (pos_bi_A[d].get((t1, t2), 0) + alpha_tri * V)
            p_gt_A = (c_tri_A + alpha_tri) / (c_bi_ctx_A + alpha_tri * V)
            p_pb_A = (pos_bi_A[d].get((t2, t3), 0) + alpha_tri) / (pos_uni_A[d].get(t2, 0) + alpha_tri * V)
            p_gb_A = (c_bi_A + alpha_tri) / (c_u_prev_A + alpha_tri * V)
            p_u_A  = (c_u_curr_A + 1.0) / total_u_A
            interp_pt_A = w_pt * p_pt_A + w_gt * p_gt_A + w_pb * p_pb_A + w_gb * p_gb_A + w_u * p_u_A
            
            p_pt_B = (pos_tri_B[d].get((t1, t2, t3), 0) + alpha_tri) / (pos_bi_B[d].get((t1, t2), 0) + alpha_tri * V)
            p_gt_B = (c_tri_B + alpha_tri) / (c_bi_ctx_B + alpha_tri * V)
            p_pb_B = (pos_bi_B[d].get((t2, t3), 0) + alpha_tri) / (pos_uni_B[d].get(t2, 0) + alpha_tri * V)
            p_gb_B = (c_bi_B + alpha_tri) / (c_u_prev_B + alpha_tri * V)
            p_u_B  = (c_u_curr_B + 1.0) / total_u_B
            interp_pt_B = w_pt * p_pt_B + w_gt * p_gt_B + w_pb * p_pb_B + w_gb * p_gb_B + w_u * p_u_B
            
            lp_A_pt, lp_B_pt = np.log(interp_pt_A), np.log(interp_pt_B)
            logp_A_pt += lp_A_pt; logp_B_pt += lp_B_pt
            diff_pt = lp_A_pt - lp_B_pt
            pt_diffs.append(diff_pt)
            if d == 0: d0_diffs.append(diff_pt)
            elif d == 9: d9_diffs.append(diff_pt)
            
            # --- Interpolated Trigram ---
            p3_A = (c_tri_A + alpha_bi) / (c_bi_ctx_A + alpha_bi * V) if c_bi_ctx_A > 0 else (alpha_bi / (alpha_bi * V))
            p2_A = (c_bi_A + alpha_bi) / (c_u_prev_A + alpha_bi * V)
            p1_A = (c_u_curr_A + alpha_bi) / (tot_A + alpha_bi * V)
            p_tri_interp_A = l3 * p3_A + l2 * p2_A + l1 * p1_A
            
            p3_B = (c_tri_B + alpha_bi) / (c_bi_ctx_B + alpha_bi * V) if c_bi_ctx_B > 0 else (alpha_bi / (alpha_bi * V))
            p2_B = (c_bi_B + alpha_bi) / (c_u_prev_B + alpha_bi * V)
            p1_B = (c_u_curr_B + alpha_bi) / (tot_B + alpha_bi * V)
            p_tri_interp_B = l3 * p3_B + l2 * p2_B + l1 * p1_B
            
            lp_A_tri, lp_B_tri = np.log(p_tri_interp_A), np.log(p_tri_interp_B)
            logp_A_tri += lp_A_tri; logp_B_tri += lp_B_tri
            tri_diffs.append(lp_A_tri - lp_B_tri)
            
            # --- Local Window Deltas ---
            p_win_A = 0.5 * (c_tri_A + alpha_bi) / (c_bi_ctx_A + alpha_bi * V) + 0.4 * (c_bi_A + alpha_bi) / (c_u_prev_A + alpha_bi * V) + 0.1 * (c_u_curr_A + 1.0) / total_u_A
            p_win_B = 0.5 * (c_tri_B + alpha_bi) / (c_bi_ctx_B + alpha_bi * V) + 0.4 * (c_bi_B + alpha_bi) / (c_u_prev_B + alpha_bi * V) + 0.1 * (c_u_curr_B + 1.0) / total_u_B
            win_deltas.append(np.log(p_win_A) - np.log(p_win_B))
            
        pt_diffs = np.array(pt_diffs)
        delta_pt = logp_A_pt - logp_B_pt
        f_pt_list.append([
            delta_pt, delta_pt/(L-2), np.mean(pt_diffs), np.std(pt_diffs), np.min(pt_diffs), np.max(pt_diffs),
            np.mean(pt_diffs>0), logp_A_pt/(logp_B_pt+1e-9), -logp_A_pt/(L-2), -logp_B_pt/(L-2),
            (-logp_A_pt+logp_B_pt)/(L-2), np.sum(d0_diffs) if d0_diffs else 0.0, np.sum(d9_diffs) if d9_diffs else 0.0
        ])
        
        tri_diffs = np.array(tri_diffs)
        delta_tri = logp_A_tri - logp_B_tri
        f_tri_8 = [delta_tri, delta_tri/(L-2), np.mean(tri_diffs), np.std(tri_diffs), np.min(tri_diffs), np.max(tri_diffs), np.mean(tri_diffs>0), logp_A_tri/(logp_B_tri+1e-9)]
        f_m16_list.append(f_bi_8 + f_tri_8)
        
        d_arr = np.array(win_deltas, dtype=np.float64)
        N_d = len(d_arr)
        cs = np.insert(np.cumsum(d_arr), 0, 0.0)
        row_w = []
        for W in windows:
            if N_d >= W: w_vals = (cs[W:] - cs[:-W]) / W
            elif N_d > 0: w_vals = np.array([np.mean(d_arr)])
            else: w_vals = np.array([0.0])
            w_mean = float(np.mean(w_vals))
            row_w.extend([
                w_mean, float(np.std(w_vals)) if len(w_vals) > 1 else 0.0, float(np.min(w_vals)), float(np.max(w_vals)),
                float(np.median(w_vals)), float(np.percentile(w_vals, 90)), float(np.percentile(w_vals, 10)),
                float(np.max(np.abs(w_vals - w_mean))) if len(w_vals) > 0 else 0.0, float(np.mean(w_vals > 0))
            ])
        f_win_list.append(row_w)
        
    return (np.array(f_pt_list, dtype=np.float64),
            np.array(f_m16_list, dtype=np.float64),
            np.array(f_win_list, dtype=np.float64))

# ==============================================================================
