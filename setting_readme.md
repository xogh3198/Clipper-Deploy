# Clipper GPU 설정 및 실행 (docker/v3)

원본 Clipper 프런트엔드와 독립 GPU 워커를 사용합니다. 저장소 루트에서 실행합니다.
공통 라이브러리·CUDA는 요청 전에 초기화하고, 모델은 첫 요청에서 읽고 GPU로 이동합니다.
기본 예열은 모델 로딩마다 10회이며 첫 요청의 `load_ms`에 포함됩니다. 초기화 비용을 삭제하지 않고 별도 기록합니다.

## 1. 코드와 Docker 확인

```bash
git clone --branch docker/v3 --single-branch https://github.com/xogh3198/Clipper-Deploy.git clipper
cd clipper
docker info --format '{{.DockerRootDir}}'
nvidia-smi
```

GPU 사용이 가능한 현재 Docker를 사용합니다. 다른 실험은 먼저 종료하세요.

## 2. 개인 설정 만들기

```bash
cp -n deploy/docker/clipper-docker.env deploy/docker/clipper-docker.local.env
vim deploy/docker/clipper-docker.local.env
```

필요한 항목만 수정합니다. 주석은 별도 줄에 작성하세요.

```dotenv
GPU=2
GPU_TYPE=v100
MODEL_NAME=resnet-50
MODEL_ID=microsoft/resnet-50
MODEL_REVISION=34c2154c194f829b11125337b98c8f5f9965ff19
SLO_MS=200
BATCH_SIZE=-1
CPU_THREADS=2
WARMUP_QUERIES=10
WORKER_IMAGE=taeho-clipper-worker:0.2.0
INPUT_IMAGE=inputs/test-dog.jpg
MODEL_DIR=data/clipper/models
RESULT_DIR=data/clipper/v3/results
```

- `MODEL_NAME`은 등록 이름, `MODEL_ID`는 HF 이미지 분류 모델, `MODEL_REVISION`은 해당 모델의 40자리 commit입니다.
- `BATCH_SIZE=-1`: 원본 적응형 배치. `1`: 단일 요청 배치. `4` 등 양수: 고정 배치 크기 설정.
  요청이 적으면 실제 실행은 더 작을 수 있으므로 결과의 `batch`를 확인하세요.
- `MODEL_DIR`는 가중치 캐시, `RESULT_DIR`는 측정 결과와 실행 메타데이터입니다.
  개인 `.local.env`·가중치·측정 파일은 Git에 올리지 않습니다.
- `WARMUP_QUERIES=10`은 Proteus GPU 런타임과 같은 기본 예열 횟수입니다. `0`이면 모델 예열을 끕니다.
  GPU 전처리로 바꾸는 설정이 아닙니다. 실제 입력 전처리는 계속 CPU에서 수행합니다.

### docker/v2에서 업데이트할 때

먼저 **기존 브랜치·설정으로** 실행 중인 Clipper를 종료하세요. 개인 설정은 덮어쓰지 않습니다.

```bash
bash deploy/docker/clipper-docker.sh down
git status --short
git fetch origin refs/heads/docker/v3:refs/remotes/origin/docker/v3
git checkout --no-track -b docker/v3 refs/remotes/origin/docker/v3
```

추적 파일에 변경이 있으면 먼저 확인하고 보존하세요. `reset --hard`나 무조건적인 stash 복원은 사용하지 않습니다.
기존 `.local.env`에 이전 값이 남아 있으면 현재 터미널에서 다음처럼 명시적으로 선택할 수 있습니다.

```bash
export WORKER_IMAGE=taeho-clipper-worker:0.2.0
export WARMUP_QUERIES=10
export RESULT_DIR=/data4tb/taeho-proteus/clipper/v3/results
```

`MODEL_DIR`는 이미 다운로드한 캐시 경로를 유지합니다. `RESULT_DIR`만 새 경로로 분리하면 이전 측정은 남습니다.
설정을 계속 쓰려면 개인 `.local.env`에 같은 값을 반영하세요. Docker 데몬 선택은 변경하지 않습니다.
이후 아래 최초 실행 절차 중 `build`부터 진행합니다. 모델 리비전·캐시가 같으면 다운로드는 재사용됩니다.

ResNet-18로 교체할 때:

```dotenv
MODEL_NAME=resnet-18
MODEL_ID=microsoft/resnet-18
MODEL_REVISION=65a5785d9156231087c481e0c7dd33a5ff6f7e3e
```

## 3. 최초 실행

각 명령이 성공한 뒤 다음 단계로 진행하세요.

```bash
bash deploy/docker/clipper-docker.sh settings
bash deploy/docker/clipper-docker.sh build
bash deploy/docker/clipper-docker.sh download
bash deploy/docker/clipper-docker.sh sample
bash deploy/docker/clipper-docker.sh doctor
bash deploy/docker/clipper-docker.sh up
bash deploy/docker/clipper-docker.sh state
bash deploy/docker/clipper-docker.sh query -n 11
bash deploy/docker/clipper-docker.sh logs worker
```

`up`이 애플리케이션과 모델을 등록합니다. 별도 `register`는 필수가 아닙니다.
`sample`은 기존 INFaaS/Proteus JPEG를 복사하거나 현재 입력 파일을 확인합니다.
다른 JPEG를 쓰려면 `INPUT_IMAGE`를 기존 파일의 경로로 지정하세요.

## 4. 테스트와 종료

첫 로딩 요청 1회와 이후 반복 요청 10회:

```bash
bash deploy/docker/clipper-docker.sh cold-test -n 11
```

동시 요청으로 실제 배치 확인:

```bash
bash deploy/docker/clipper-docker.sh batching-test -n 128 --concurrency 16
bash deploy/docker/clipper-docker.sh down
```

`cold-test`는 자기 컨테이너를 재시작합니다. 첫 요청이 SLO를 넘으면 프런트엔드는
`DEFAULT`를 반환할 수 있습니다. 이때 worker completion과 클라이언트 e2e는 다른 값입니다.
최초 실행에서도 별도 `up` 없이 `cold-test`를 실행할 수 있습니다.
배치가 1보다 커야 `BATCHING VERIFIED`입니다. 실제 SLO/처리량 우위를 뜻하지 않습니다.

| 출력 | 측정 범위 |
|---|---|
| `runtime_startup_ms` | 요청 전 공통 프레임워크 import와 CUDA 초기화. 컨테이너 생성 전체 시간은 아님 |
| `cold_load_ms` | 첫 요청의 CPU 모델 준비 + 모델 H2D + 설정된 횟수의 예열 |
| `cold_load_stages.checkpoint_read_ms` | 프로세서·설정·가중치 읽기와 CPU 모델 생성. 순수 디스크 읽기만은 아님 |
| `cold_load_stages.model_h2d_ms` | 모델의 GPU 이동과 동기화 |
| `cold_load_stages.warmup_ms` | 예열 입력 준비·전송·모델 실행 |
| `cold_worker_total_ms` | 워커 배치 처리 전체. HTTP·프런트엔드 대기는 제외 |
| `cold_client_e2e_ms`, `cold_default_response` | 실제 클라이언트 응답시간과 fallback 여부 |
| `warmup_forward_delta` | 측정 도중 수행한 내부 예열 횟수. 실제 요청 배치의 `worker_forward_delta`와 별도 |

이 모드는 **모델 비상주 진단**입니다. 원본 Clipper의 모델 선로딩 완료 후 서비스 시작 방식과는 구분합니다.
OS 캐시·모델별 지연 import·프로파일링 이력까지 완전히 같다는 뜻은 아니므로 시간이 500ms가 된다고 보장하지 않습니다.
`docker/v2` 결과와 `docker/v3` 결과를 같은 cold 조건으로 묶지 마세요.

## 5. 설정 변경 후 재실험

**기존 설정으로 `down` → local.env 수정 → `settings` → `up`** 순서입니다.
특히 `PROJECT`·`RESULT_DIR`는 기존 실행을 종료한 뒤 바꾸세요.

| 변경 | 필요한 추가 작업 |
|---|---|
| 모델 | `download` → `doctor` → `up` |
| GPU·SLO·배치·CPU 스레드·예열 횟수·입력 | `up` 후 재측정 |
| 요청 수·동시성 | `-n`·`--concurrency` 변경 |
| 코드·Dockerfile·라이브러리 버전 | `build` 후 `up` |

모델·배치·GPU 변경만으로는 **재빌드가 필요 없습니다**. 이번 코드 업데이트 후에는 최초 한 번 재빌드하세요.
터미널의 `MODEL_NAME`·`GPU` 같은 변수는 local.env보다 우선하므로 `settings` 출력을 확인하세요.
포트 충돌 시 local.env의 `QUERY_PORT`·`MANAGEMENT_PORT`·`STATS_PORT`를 서로 다르게 설정합니다.

## 6. 결과

CSV/JSON과 워커 trace는 `RESULT_DIR`에 저장됩니다.
전처리·추론은 실제 GPU 워커에서 실행하며 학습하지 않습니다. 분류 정확도는 별도 검증셋으로 측정해야 합니다.
상세 구현과 오류 해결은 [docker.md](docker.md)를 참고하세요.
