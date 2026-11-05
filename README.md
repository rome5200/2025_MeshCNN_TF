# MeshCNN_TF

의료 3D 메쉬(정상/비정상 `.obj`)를 분류하기 위해 MeshCNN을 TensorFlow/Keras로 구현한 프로젝트입니다. 원본 [MeshCNN](https://github.com/ranahanocka/MeshCNN)은 PyTorch로 작성되어 있습니다.

## 구조

| 폴더 | 내용 |
| --- | --- |
| `models/` | `MeshCNN_TF.ipynb`, `MedMeshCNN_TF.ipynb`: MeshConv · MeshPooling 레이어와 MedMeshCNN 모델 정의, 학습 설정(콜백, 조기 종료) / `meshcnn_tf.py`: EdgeConv 기반 최소 MeshCNN 모듈 |
| `prepro/` | DICOM 전처리, `.npy` 시각화, `.obj` 메쉬 단순화(단일/전체), 캐시 생성, `summary.csv`(파일별 정점·면 수) |
| `util/` | 레이블 추출, `.obj` 시각화, 데이터 검증, 파일 정리, GPU 메모리 관리 |

## 처리 흐름

1. DICOM 영상을 전처리해 장기 메쉬(`.obj`)로 만들고, 면 수를 약 10,000개로 단순화합니다.
2. 메쉬에서 면 법선과 엣지 특성(면적, 둘레, 종횡비, 최소 내각, 원점까지 거리)을 계산하고 정규화합니다.
3. MeshConv / MeshPooling 레이어로 구성된 MedMeshCNN으로 정상/비정상을 분류합니다.

## 환경

- Python 3, Jupyter Notebook
- `tensorflow`, `numpy`, `trimesh`, `open3d`
