//! Approximate minimum-degree ordering for fill reduction.

use super::csc::CscMatrix;

/// AMD-style ordering of a symmetric pattern given by the upper triangle of `a`.
pub fn order_upper(a: &CscMatrix) -> Vec<usize> {
    let n = a.n;
    if n == 0 {
        return Vec::new();
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        for p in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[p];
            if i == j {
                continue;
            }
            adj[i].push(j);
            adj[j].push(i);
        }
    }
    for list in adj.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }

    let mut degree: Vec<usize> = adj.iter().map(|nbs| nbs.len()).collect();
    let mut alive = vec![true; n];
    let mut perm = Vec::with_capacity(n);
    let mut in_neigh = vec![false; n];

    for _ in 0..n {
        let mut best = usize::MAX;
        let mut best_deg = usize::MAX;
        for i in 0..n {
            if alive[i] && degree[i] < best_deg {
                best_deg = degree[i];
                best = i;
            }
        }
        if best == usize::MAX {
            break;
        }
        perm.push(best);
        alive[best] = false;

        let neigh: Vec<usize> = adj[best].iter().copied().filter(|&v| alive[v]).collect();
        for &v in &neigh {
            in_neigh[v] = true;
        }
        for (ii, &u) in neigh.iter().enumerate() {
            for &v in neigh.iter().skip(ii + 1) {
                if !adj[u].contains(&v) {
                    adj[u].push(v);
                    adj[v].push(u);
                }
            }
        }
        for &u in &neigh {
            adj[u].retain(|&w| w != best && alive[w]);
            adj[u].sort_unstable();
            adj[u].dedup();
            degree[u] = adj[u].len();
            in_neigh[u] = false;
        }
        adj[best].clear();
    }
    perm
}
