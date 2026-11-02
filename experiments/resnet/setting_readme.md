# v1: 원본 Clipper C++ 실행·계측

Clipper 기준 커밋: 9f25e3fc7f8edc891615e81c5b80d3d8aed72608.
src/ 전체와 containers/python/rpc.py는 이 커밋과 동일하게 복구했다.
추가했던 deadline_unix_us 전달·타이머 예산 변경·큐 거절 C++ 확장을 제거했다.
빌드 전 소스 차이를 검사하며 native C++ 배치/AIMD·큐·fallback·RPC를 그대로 사용한다.

요청은 원래 C++ HTTP 프런트엔드로 직접 들어간다. 별도 CPU gateway가 없다.
사용자 정의 model callback이 JPEG 디코딩·이미지 처리를 수행하고 실제 ONNX GPU 추론을 실행한다.
전처리를 native queue 앞에 옮기거나 전처리 때문에 원래 C++ deadline을 바꾸지 않는다.
내부 예측 캐시는 비교용으로 0을 설정한다. 큐 상한을 새로 추가하지 않는다.

CLIPPER_MODE=ha이면 ResNet-152 고정, ht이면 ResNet-18 고정이다. 원래 Clipper처럼 모델 선택은 사용자 설정이다.
모델은 공개 ONNX ResNet-v1 FP32이며 모델 plugin만 GPU 실행에 맞춰 연결한다.
JSON의 worker 구간은 model callback의 구간이다. C++ 프런트엔드/배치 큐를 포함하는 전체 경로는 클라이언트 E2E로 따로 기록한다.
v2는 이전 CPU-ready 어댑터이며 이번 원본 복구는 v1에만 적용했다.

## 실행 순서

기존 수정 파일이 있는 폴더는 그대로 두고 v1 커밋을 별도 폴더에서 실행한다.
기존 저장소 폴더에서 아래를 실행하며, 새 폴더 이름은 저장소별로 다르게 지정한다.

```bash
git fetch origin refs/heads/v1
git worktree add --detach ../Clipper-source-v1 FETCH_HEAD
cd ../Clipper-source-v1

systemctl --user start docker.service
unset DOCKER_CONTEXT
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
docker info --format '{{.DockerRootDir}}'
nvidia-smi -i 2
```

저장 위치는 /data4tb/taeho-proteus/docker-data. V100에 다른 작업이 있으면 해당 작업을 먼저 종료한다.
스크립트가 다른 실험을 종료하거나 Docker daemon 설정을 바꾸지 않는다.

```bash
# 실행할 구현을 아래에서 하나 지정한다.
ENGINE=clipper

bash deploy/docker/source-docker.sh "$ENGINE" settings
bash deploy/docker/source-docker.sh "$ENGINE" build &&
bash deploy/docker/source-docker.sh "$ENGINE" setup &&
bash deploy/docker/source-docker.sh "$ENGINE" up &&
bash deploy/docker/source-docker.sh "$ENGINE" register &&
bash deploy/docker/source-docker.sh "$ENGINE" state

# 프로파일링 종료 후 측정용 프로세스·정책 상태를 새로 시작한다.
bash deploy/docker/source-docker.sh "$ENGINE" down &&
bash deploy/docker/source-docker.sh "$ENGINE" up

bash deploy/docker/source-docker.sh "$ENGINE" trace --dry-run
bash deploy/docker/source-docker.sh "$ENGINE" trace --limit 0 --speed 2 --concurrency 256 --timeout 10
bash deploy/docker/source-docker.sh "$ENGINE" state
bash deploy/docker/source-docker.sh "$ENGINE" down
```

설정 변경은 experiments/resnet/local.env에 필요한 KEY=VALUE만 쓴다. 없으면 settings.env 기본값을 사용한다.
기존 local.env가 있으면 그것이 우선이므로 settings 출력으로 적용 값을 확인한다.
GPU=2, SLO_MS=200, WORKLOAD=twitter, TRACE_SPEED=2가 현재 비교 조건이다.
공개 ResNet trace의 원래 SLO는 300ms이며 200ms는 기존 실험에서 선택한 조건이다. 논문의 SLO를 그대로 재현한 값이라고 쓰지 않는다.
Twitter 목록은 3,776건이고 2배속이면 145.154초·평균 26.01 QPS다. 파일의 도착 시각·QoS·원 SLO와 파일 해시를 기록한다.
배속으로 요청 수가 늘어나지는 않는다.

## 결과 해석

JSON/CSV에 원래 클라이언트 E2E, 성공/늦은 성공/실패·거절/미전송, 요청별 모델·배치·원래 stage timings를 저장한다.
SLO 위반율은 실제 클라이언트 E2E와 실패를 기준으로 계산한다. 성공 응답 지연 분위수는 성공 응답만의 분포다.

measurement_boundaries에는 전처리 wall time, 전처리 완료 뒤 worker 결과까지의 구간, E2E−전처리 wall time을 별도로 저장한다.
DALI 전처리는 GPU 연산을 포함할 수 있으므로 전처리 wall time 전체를 CPU 연산 시간이라고 부르지 않는다.
E2E−전처리는 같은 실행의 산술 분해이며, 전처리를 실제로 제거했을 때의 결과나 원래 SLO 위반율을 대신하지 않는다.
전처리 앞에 있던 큐 대기·모델 로딩은 after_preprocessing_tail에 포함되지 않는다. 원래 배치/전처리 순서가 달라 이 값만으로 전체 서빙을 비교하지 않는다.
모델 로딩은 원래 timings/이벤트에 별도로 기록한다. 측정 구간을 맞추려고 원래 deadline이나 거절 조건을 바꾸지 않는다.

실측 정확도는 측정하지 않는다. 표시되는 모델 점수는 supplied Proteus reference이고 현재 공개 체크포인트의 정답률과 구분한다.
정규화 점수는 다섯 후보의 최고 기준 점수 79.22를 공통 분모로 사용한다. 성공 응답 전체와 SLO 내 성공 응답을 나누어 기록한다.
단일 V100·ResNet 부분 실험이며, 논문 전체 클러스터 결과의 수치 재현이라고 주장하지 않는다.
로컬 CPU 연결·소스 차이 검사는 수행하지만 Docker 빌드와 GPU 실행은 GPU 서버에서 확인해야 한다.

결과 기본 경로:

```text
/data4tb/taeho-proteus/experiments/paired-resnet/clipper/source-v1/results/
```
