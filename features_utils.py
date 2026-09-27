import zlib
from collections import Counter, defaultdict
import numpy as np

def build_18_stats(records):
    rows = []
    for r in records:
        seq = r["text"] if isinstance(r, dict) else r
        x = np.asarray(seq, dtype=np.int64)
        n = len(x)
        if n == 0:
            rows.append([0.0]*18)
            continue
        c = Counter(seq)
        unique = len(c)
        freqs = np.array(list(c.values()))
        p = freqs / n
        ent = float(-np.sum(p * np.log2(p)))
        adj_rep = int(np.sum(x[1:] == x[:-1])) if n > 1 else 0
        bi_div = len(set(zip(seq[:-1], seq[1:]))) / (n - 1) if n > 1 else 0
        tri_div = len(set(zip(seq[:-2], seq[1:-1], seq[2:]))) / (n - 2) if n > 2 else 0

        rows.append([
            np.log1p(n), n, unique / n, ent,
            adj_rep / (n - 1) if n > 1 else 0,
            float(np.mean(freqs <= 2)), float(np.mean(x == 0)),
            float(freqs.max() / n), bi_div, tri_div,
            float(x.mean()), float(x.std()), float(np.median(x)),
            float(unique),
            float(sum(1 for v in c.values() if v==1) / unique) if unique else 0,
            float(adj_rep),
            float(sum(1 for v in c.values() if v==1)),
            float(np.sum(x == 0)),
        ])
    return np.asarray(rows, dtype=np.float64)

def build_8_tclb(records):
    rows = []
    for r in records:
        seq = r["text"] if isinstance(r, dict) else r
        x = np.asarray(seq, dtype=np.int64)
        n = len(x)
        if n == 0:
            rows.append([0.0]*8)
            continue
        seq_bytes = np.asarray(seq, dtype=np.int32).tobytes()
        comp_ratio = len(zlib.compress(seq_bytes, level=6)) / max(len(seq_bytes), 1)

        pos_dict = {}
        for idx_t, tok in enumerate(seq):
            pos_dict.setdefault(tok, []).append(idx_t)
        all_gaps = []
        for p in pos_dict.values():
            if len(p) > 1:
                all_gaps.extend(np.diff(p))
        if all_gaps:
            mean_gap = float(np.mean(all_gaps))
            fano_gap = float(np.var(all_gaps)) / (mean_gap + 1e-6)
            cv_gap   = float(np.std(all_gaps)) / (mean_gap + 1e-6)
        else:
            mean_gap = float(n)
            fano_gap = 0.0
            cv_gap   = 0.0

        if n >= 10:
            step = max(1, n // 10)
            c_cum = set()
            u_curve = []
            for i in range(step, n + 1, step):
                c_cum.update(seq[:i])
                u_curve.append(len(c_cum))
            log_n = np.log(np.arange(step, n + 1, step)[:len(u_curve)])
            log_u = np.log(np.array(u_curve))
            if len(log_n) > 1 and np.std(log_n) > 1e-5:
                heaps_b = float(np.cov(log_n, log_u)[0, 1] / np.var(log_n))
            else:
                heaps_b = 1.0
        else:
            heaps_b = 1.0

        if n >= 6:
            third = max(1, n // 3)
            c_early = Counter(seq[:third])
            c_late  = Counter(seq[-third:])
            p_e = np.array(list(c_early.values())) / third
            p_l = np.array(list(c_late.values()))  / third
            ent_early = float(-np.sum(p_e * np.log2(p_e)))
            ent_late  = float(-np.sum(p_l * np.log2(p_l)))
            ent_drift = ent_late - ent_early
            uniq_drift = (len(c_late) - len(c_early)) / third
        else:
            ent_drift = 0.0
            uniq_drift = 0.0

        rows.append([
            comp_ratio, mean_gap, fano_gap, cv_gap,
            heaps_b, ent_early if n >= 6 else 0.0, ent_drift, uniq_drift
        ])
    return np.asarray(rows, dtype=np.float64)

def build_rolling_entropy_features(records):
    rows = []
    for r in records:
        seq = r["text"] if isinstance(r, dict) else r
        n = len(seq)
        if n == 0:
            rows.append([0.0]*8)
            continue
        row = []
        for w in [16, 32, 64, 128]:
            if n < w:
                c = Counter(seq)
                p = np.array(list(c.values())) / n
                ent = float(-np.sum(p * np.log2(p)))
                row.extend([ent, 0.0])
            else:
                ents = []
                step = max(1, w // 4)
                for i in range(0, n - w + 1, step):
                    c_w = Counter(seq[i:i+w])
                    p_w = np.array(list(c_w.values())) / w
                    ents.append(float(-np.sum(p_w * np.log2(p_w))))
                row.extend([float(np.mean(ents)), float(np.std(ents))])
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)
