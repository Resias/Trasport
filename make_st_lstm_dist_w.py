import numpy as np
import pandas as pd
from collections import deque

def compute_hop_distance(adj):
    N = adj.shape[0]
    dist = np.full((N, N), np.inf)

    for start in range(N):
        dist[start, start] = 0
        queue = deque([start])

        while queue:
            cur = queue.popleft()
            for nxt in range(N):
                if adj[cur, nxt] == 1 and dist[start, nxt] == np.inf:
                    dist[start, nxt] = dist[start, cur] + 1
                    queue.append(nxt)

    return dist


def build_dist_and_W(adj_path, avg_minutes=3):
    # 1. 인접행렬 로드
    od_df = pd.read_csv(adj_path, index_col=0)
    od_df = od_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    adj = od_df.values  # (N, N)
    print(adj.shape)

    # 2. hop 기반 최단거리 계산
    dist_matrix = compute_hop_distance(adj)

    # 3. 연결 안 된 경우를 큰 값으로 처리
    finite_vals = dist_matrix[np.isfinite(dist_matrix)]
    max_val = finite_vals.max()
    dist_matrix[np.isinf(dist_matrix)] = max_val + 2

    # 4. 이동시간 행렬 생성
    W_matrix = dist_matrix * avg_minutes

    # 5. 저장
    np.save("dist_matrix.npy", dist_matrix.astype(float))
    np.save("W_matrix.npy", W_matrix.astype(float))

    print("dist_matrix.npy / W_matrix.npy 생성 완료!")
    return dist_matrix, W_matrix


# 실행
dist_matrix, W_matrix = build_dist_and_W("AD_matrix_trimmed_common.csv", avg_minutes=3)

print(np.all(dist_matrix.diagonal() == 0))
adj = pd.read_csv("AD_matrix_trimmed_common.csv", index_col=0).values
print(np.unique(dist_matrix[adj == 1]))
# 출력이 [1.] 이어야 정상
print((np.abs(W_matrix - dist_matrix * 3) < 1e-6).all())
