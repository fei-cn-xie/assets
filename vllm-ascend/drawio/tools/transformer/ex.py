"""运行示例：全套图共用同一组矩阵（数字由代码算出来，保证不出错）。

尺寸约定（遵守"不要小矩阵、不要方阵"的要求）：
  X   5×4   ← 5 个 token、每个 4 维（真实：5×4096）
  WQ  4×3   → Q = X·WQ   5×3      （5≠3，结果非方阵 ✓）
  WK  4×3   → K = X·WK   5×3      （Kᵀ 是 3×5）
  S = Q·Kᵀ = 5×3 · 3×5 → 5×5      （注意力分数天然是 token×token，**这是唯一的方阵**，
                                    因为"行=发问的 token，列=被看的 token"）
  P = softmax(S/√3) → 5×5          （必须是小数）
  V  = X·WV  5×3  → O = P·V = 5×5·5×3 → 5×3  （5≠3 ✓）
  FFN: 5×4 · 4×8 = 5×8  →  5×8 · 8×4 = 5×4   （都非方阵 ✓）
"""

import math

X = [[1, 0, 2, 1],
     [0, 2, 1, 0],
     [2, 1, 0, 1],
     [1, 1, 2, 0],
     [0, 1, 1, 2]]

WQ = [[1, 0, 1],
      [0, 1, 0],
      [1, 1, 0],
      [0, 1, 1]]

WK = [[0, 1, 1],
      [1, 0, 1],
      [0, 1, 0],
      [1, 1, 1]]

WV = [[1, 1, 0],
      [0, 1, 1],
      [1, 0, 1],
      [1, 0, 0]]

W1 = [[1, 0, 1, 0, 1, 0, 1, 0],       # up/gate 投影 4×8
      [0, 1, 0, 1, 0, 1, 0, 1],
      [1, 1, 0, 0, 1, 1, 0, 0],
      [0, 0, 1, 1, 0, 0, 1, 1]]

W2 = [[1, 0, 1, 0],                   # down 投影 8×4
      [0, 1, 0, 1],
      [1, 0, 0, 1],
      [0, 1, 1, 0],
      [1, 1, 0, 0],
      [0, 0, 1, 1],
      [1, 0, 1, 0],
      [0, 1, 0, 1]]


def mm(A, B):
    """标准矩阵乘：A(m×k) · B(k×n) → m×n。"""
    m, k, n = len(A), len(B), len(B[0])
    assert len(A[0]) == k
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
            for i in range(m)]


def T(A):
    return [list(r) for r in zip(*A)]


Q = mm(X, WQ)
K = mm(X, WK)
V = mm(X, WV)
S = mm(Q, T(K))                       # 5×5
SCALE = round(math.sqrt(3), 2)        # 1.73
S_scaled = [[round(v / math.sqrt(3), 1) for v in row] for row in S]


def softmax(row):
    mx = max(row)
    e = [math.exp(v - mx) for v in row]
    s = sum(e)
    return [x / s for x in e]


P = [[round(v, 2) for v in softmax(row)] for row in S_scaled]
O = [[round(v, 1) for v in row] for row in mm(P, V)]
H1 = mm(X, W1)                        # 5×8
H2 = mm(H1, W2)                       # 5×4

# 一行里"这些格子是怎么加出来的"手算演示（线性层 ⑤ 用）
X5 = [[1, 2, 0, 1],
      [0, 1, 3, 1],
      [2, 0, 1, 2],
      [1, 1, 1, 1],
      [3, 0, 2, 1],
      [0, 2, 1, 0]]
W5 = [[1, 0, 2, 1, 0],
      [0, 1, 1, 0, 2],
      [2, 1, 0, 1, 1],
      [1, 2, 1, 0, 1]]
R5 = mm(X5, W5)                       # 6×5
