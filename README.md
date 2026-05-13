# PyG-lowlevel

`pyg-team/pytorch_geometric`의 high-level convolution 레이어를 바로 가져다 쓰지 않고, GCN을 low-level 관점에서 직접 구현한 예제 저장소입니다.

## Pure Torch 버전

`torch_geometric` 없이, 순수 `torch`만으로 다시 구현한 버전도 포함했습니다.

- `pure_torch_spectral_gcn.py`: normalized adjacency matrix를 직접 만들어 `A_hat @ X @ W`를 계산하는 spectral GCN
- `pure_torch_spatial_gcn.py`: edge 단위 message를 만들고 `index_add_`로 aggregate하는 spatial GCN
- `pure_torch_gcn_utils.py`: toy graph 생성, mask 분할, 학습/평가 루프
- `requirements-pure-torch.txt`: PyG 없이 실행할 최소 의존성

이 버전은 외부 그래프 라이브러리 없이 GCN의 핵심 연산을 직접 보는 데 초점을 맞춥니다.

## 파일

- `spectral_gcn.py`: adjacency matrix multiplication 기반 spectral GCN
- `spatial_gcn_torch_sparse.py`: `torch_sparse` 기반 spatial GCN
- `low_level_gnn.py`: PyG `MessagePassing` 기반 GCN
- `pure_torch_spectral_gcn.py`: pure `torch` spectral GCN
- `pure_torch_spatial_gcn.py`: pure `torch` spatial GCN
- `pure_torch_gcn_utils.py`: pure `torch` toy graph/data utilities
- `gcn_utils.py`: 공통 데이터 로딩/학습/평가 유틸
- `requirements.txt`: 최소 의존성 목록

## 구현 목표

같은 GCN 수식이라도 구현 관점은 크게 둘로 볼 수 있습니다.

- spectral 관점:
  정규화된 adjacency matrix `A_hat = D^(-1/2) (A + I) D^(-1/2)`를 만든 뒤 `A_hat X W`를 직접 곱합니다.
- spatial 관점:
  각 노드가 이웃에게서 메시지를 받아 aggregate하는 연산으로 해석하고, sparse operator로 이를 수행합니다.

둘 다 결국 아래 식을 계산합니다.

```text
H^(l+1) = D^(-1/2) (A + I) D^(-1/2) H^(l) W
```

차이는 "이 식을 어떤 데이터 구조와 어떤 계산 흐름으로 구현하느냐"에 있습니다.

## 1. Spectral GCN

파일: `spectral_gcn.py`

핵심은 adjacency operator를 먼저 만든 뒤, 레이어 안에서는 `torch.sparse.mm(adj_norm, support)`만 수행하는 점입니다.

```python
support = self.lin(x)
out = torch.sparse.mm(adj_norm, support)
```

흐름은 다음과 같습니다.

1. `edge_index`에 self-loop 추가
2. sparse COO adjacency 생성
3. degree 계산
4. `D^(-1/2) A D^(-1/2)` normalization
5. 레이어에서 `A_hat @ X @ W`

이 방식은 수식이 코드에 가장 직접적으로 보입니다. 그래서 "GCN이 adjacency matrix multiplication이다"라는 감각을 익히기에 좋습니다.

## 2. Spatial GCN with `torch_sparse`

파일: `spatial_gcn_torch_sparse.py`

핵심은 adjacency를 `torch_sparse.SparseTensor`로 들고 있고, sparse matrix multiplication을 메시지 집계 관점으로 해석하는 점입니다.

```python
support = self.lin(x)
out = matmul(adj_t, support)
```

흐름은 다음과 같습니다.

1. `SparseTensor(row, col)`로 그래프 구성
2. `fill_diag(...)`로 self-loop 추가
3. `sparsesum(...)`으로 degree 계산
4. `mul(...)`로 좌우 normalization 적용
5. `matmul(adj_t, support)`로 이웃 정보 aggregate

이 방식은 대규모 그래프에서 sparse operator를 직접 다루는 감각을 익히기에 좋고, message passing 관점과도 더 가깝습니다.

## Spectral vs Spatial 차이 정리

- spectral GCN:
  그래프 convolution을 "정규화된 adjacency와의 선형대수 연산"으로 봅니다.
- spatial GCN:
  그래프 convolution을 "이웃으로부터 메시지를 모으는 local aggregation"으로 봅니다.
- spectral 구현 코드 포인트:
  `build_normalized_adjacency(...)`와 `torch.sparse.mm(...)`
- spatial 구현 코드 포인트:
  `build_normalized_sparse_tensor(...)`와 `torch_sparse.matmul(...)`
- 수학적으로는:
  여기 예제에서는 같은 GCN 식을 계산합니다.
- 실무적으로는:
  spatial 쪽 구현이 sparse 데이터 구조, 샘플링, message passing 확장으로 이어지기 더 쉽습니다.

## 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 spectral_gcn.py
python3 spatial_gcn_torch_sparse.py
python3 low_level_gnn.py
```

PyG 없이 순수 `torch` 버전만 실행하려면:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pure-torch.txt
python3 pure_torch_spectral_gcn.py
python3 pure_torch_spatial_gcn.py
```

기본값은 `Planetoid(Cora)` 데이터셋으로 학습합니다.

pure `torch` 버전은 외부 데이터셋 대신 class-wise toy graph를 직접 생성해서 학습합니다.

## 예시 옵션

```bash
python3 spectral_gcn.py --dataset CiteSeer --hidden-channels 32 --epochs 300
python3 spatial_gcn_torch_sparse.py --dataset PubMed --hidden-channels 32
python3 pure_torch_spectral_gcn.py --nodes-per-class 50 --feature-dim 32
python3 pure_torch_spatial_gcn.py --intra-class-prob 0.25 --inter-class-prob 0.02
```

## Pure Torch 코드 설명

### 1. `pure_torch_spectral_gcn.py`

spectral 구현은 먼저 정규화 adjacency를 직접 만듭니다.

```python
adj = torch.sparse_coo_tensor(
    indices=torch.stack([dst, src], dim=0),
    values=values,
    size=(num_nodes, num_nodes),
).coalesce()
out = torch.sparse.mm(adj_norm, support)
```

- `indices=[dst, src]`로 저장해서 각 row가 "메시지를 받는 노드"가 되게 했습니다.
- 그래서 `adj_norm @ support`는 `src -> dst` 방향으로 이웃 정보를 모읍니다.
- 핵심 관점은 "GCN은 정규화 adjacency와 feature matrix의 곱"입니다.

### 2. `pure_torch_spatial_gcn.py`

spatial 구현은 edge별 message를 직접 만들고 목적지 노드에 더합니다.

```python
messages = norm.unsqueeze(-1) * support[src]
out = torch.zeros_like(support)
out.index_add_(0, dst, messages)
```

- `support[src]`가 source node의 메시지입니다.
- `index_add_(0, dst, messages)`가 destination node별 aggregation입니다.
- 핵심 관점은 "GCN은 source에서 destination으로 message를 보내고 모으는 과정"입니다.

### 차이점 한 줄 요약

- spectral:
  그래프 전체를 하나의 선형 연산자 `A_hat`로 본다.
- spatial:
  edge마다 message를 만들고 node마다 aggregate하는 절차로 본다.

## MessagePassing 버전과의 관계

`low_level_gnn.py`는 같은 GCN을 PyG `MessagePassing` API로 푼 버전입니다.

- `spectral_gcn.py`:
  adjacency matrix 관점
- `spatial_gcn_torch_sparse.py`:
  sparse tensor operator 관점
- `low_level_gnn.py`:
  PyG message, aggregate, update 관점

셋을 나란히 보면 "같은 GCN이 어떤 abstraction level에서 어떻게 표현되는지" 감이 잘 잡힙니다.

## 참고

현재 저장소 환경에는 `torch`, `torch_geometric`, `torch_sparse`가 설치되어 있지 않아 여기서 학습 실행 검증은 못 했습니다. 대신 스크립트 구조와 문법은 바로 실행 가능하도록 정리해 두었습니다.
