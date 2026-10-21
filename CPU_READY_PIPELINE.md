# CPU 전처리 → 원본 Clipper 배치 큐 → GPU 실행

배포 브랜치: `docker/v4` (`docker/v2`에서 분기, main 변경 없음).
세 저장소의 전처리 분리 버전은 `docker/v4`로 통일했습니다. 이전 `docker/cpu-ready-v1`과 실행 코드는 같습니다.
기존 Clipper `docker/v3`의 별도 cold 초기화 수정은 이 브랜치에 자동 병합하지 않았습니다.

## GPU 서버에서 코드 받기

기존 실행은 해당 env로 `down`합니다. 아래 `git status`의 기존 수정은 보존하고,
충돌 시 강제로 덮어쓰지 마세요. 개인 `.local.env`·가중치·결과는 유지합니다.

```bash
cd /home/xsailor4/taeho/serving-repos/clipper
git status --short
git fetch origin refs/heads/docker/v4
git checkout -b docker/v4 FETCH_HEAD
```

이미 브랜치를 만들었다면 `git checkout docker/v4` 후
`git pull --ff-only origin docker/v4`로 갱신합니다.

```text
JPEG HTTP 요청 → CPU gateway(PIL RGB + HF slow processor)
 → 준비된 CPU 텐서 전달 → C++ 모델 실행/배치 큐
 → 기존 Adaptive Batching v2 → GPU 워커(결합·H2D·forward·D2H)
```

CPU gateway에는 GPU를 할당하지 않습니다. GPU 워커만 지정한 V100을 사용합니다.
JPEG 디코딩/HF 전처리는 더 이상 C++ 배치 선택 이후의 GPU 실행 구간에서 수행하지 않습니다.

전처리 전의 최초 마감시각을 C++ 큐와 타이머까지 전달하며 SLO를 다시 시작하지 않습니다.
이를 위해 Query·PredictTask·타이머·ModelQueue에 작은 deadline 전달 어댑터를 추가했습니다.
**원본 C++ 배치 선택/추정 정책은 유지하지만, 원본 코드 무수정 버전은 아닙니다.**
변경된 네 소스 파일은 빌드 시 정확한 SHA256으로 검사합니다.

## 갱신·재실행

기존 설정으로 `down`을 실행한 뒤 개인 `clipper-docker.local.env`의 아래 항목만 갱신합니다.
모델·revision·캐시·결과 경로와 GPU 번호는 기존 비교 조건을 유지하세요.

```dotenv
CPU_THREADS=2
CPU_INTEROP_THREADS=1
PREPROCESS_WORKERS=1
MAX_QUEUE_SIZE=256
QUERY_IMAGE=taeho-clipper-query:cpu-ready-v1
MANAGEMENT_IMAGE=taeho-clipper-management:cpu-ready-v1
WORKER_IMAGE=taeho-clipper-worker:cpu-ready-v1
```

`CPU_THREADS`는 intra-op, `PREPROCESS_WORKERS`는 CPU 준비 동시 작업자 수입니다.
MAX_QUEUE_SIZE는 CPU ingress의 미완료 요청과 C++ ready 큐에 각각 적용되는 한도입니다.
다른 구현의 단일 전체 미완료 요청 한도와 정확히 같은 자원 제한이라고 볼 수는 없습니다.

```bash
docker info --format '{{.DockerRootDir}}'
bash deploy/docker/clipper-docker.sh settings
bash deploy/docker/clipper-docker.sh build
bash deploy/docker/clipper-docker.sh doctor
bash deploy/docker/clipper-docker.sh up
bash deploy/docker/clipper-docker.sh state
bash deploy/docker/clipper-docker.sh cold-test -n 11
```

캐시가 없을 때만 `download --models resnet-50`를 먼저 실행합니다.
원본 C++ 프런트엔드도 재빌드해야 합니다. 이전 build manifest는 새 버전에서 거부됩니다.
새 이미지 태그를 사용하면 이전 이미지는 보존됩니다. 가중치/과거 run 결과는 삭제하지 않습니다.

## 결과 해석

- public query 포트는 그대로이고 내부에서 gateway → C++ query로 전달합니다.
- 준비된 FP32 CPU 텐서를 기존 strings RPC로 직렬화해 넘깁니다. CPU 전처리·내부 전송·텐서 복원 비용은 e2e에 포함됩니다.
  Proteus의 프로세스 내 텐서 큐와 전송 비용이 같다는 뜻은 아닙니다.
- SLO가 CPU 준비 중 만료되면 fallback, C++ 큐에서도 최초 deadline을 유지합니다.
- CPU gateway 첫 processor 초기화는 `up` 준비 단계, 모델 가중치 로딩은 기존대로 첫 GPU 요청입니다.
- 워커 기록에 `pipeline=cpu-ready-v1`, upstream `decode_ms`/`preprocess_ms`, `unpack_ms`가 남습니다.
- 첫 cold 요청은 로딩 때문에 fallback일 수 있습니다. cold 실제 완료와 클라이언트 fallback 시간을 구분합니다.

이 변경의 목적은 실행 단계 위치 정렬이며, SLO 보장이나 성능 향상을 미리 보장하지 않습니다.
