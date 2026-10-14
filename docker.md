# Clipper 단일 GPU 실행 가이드

모델·GPU·배치 설정과 단계별 명령은 **[setting_readme.md](setting_readme.md)**를 먼저 참고하세요.

Clipper 원본 프런트엔드와 배치·RPC 경로에 독립 GPU 워커를 연결하는 추가 구성입니다.
INFaaS/Proteus 코드·이미지·컨테이너를 실행 백엔드로 사용하지 않습니다. 기존 Clipper 핵심 코드는 수정하지 않습니다.

기준 원본 커밋: `9f25e3fc7f8edc891615e81c5b80d3d8aed72608` (`develop`).
원본에는 논문 초기 AIMD 이후의 **Adaptive Batching v2**가 들어 있습니다.
따라서 이 브랜치를 그대로 실행했다고 해서 2017년 논문의 AIMD 재현이라고 부르면 안 됩니다.

## 1. 기본 설정

| 항목 | 기본값 |
|---|---|
| GPU | 호스트 GPU 2, V100 32GB 한 대 |
| 모델 | `microsoft/resnet-50`, FP32 |
| 가중치 리비전 | `34c2154c194f829b11125337b98c8f5f9965ff19` |
| 입력 | 기존 INFaaS 실험의 `test-dog.jpg`를 그대로 복사 |
| SLO | 200ms |
| 배치 | `BATCH_SIZE=-1`: 원본 Clipper 적응형 배치 |
| 응답 캐시 | `--prediction_cache_size=0` |
| 전처리 | PIL 디코딩 + Hugging Face CPU image processor |
| CPU 스레드 | 워커 2, 쿼리 프런트엔드 16 |
| 포트 | 호스트 `127.0.0.1:21337/21338/21339` |
| Docker 프로젝트 | `taeho-clipper` |

워커의 CUDA 장치 `cuda:0`은 컨테이너 안에서 보이는 첫 장치이며, 실제 호스트 GPU는 설정한 `GPU=2`입니다.
학습은 하지 않습니다. `download`는 사전 학습된 모델 가중치와 설정 파일을 다운로드하고,
워커는 첫 요청에서 이를 GPU에 올려 추론합니다. 등록/상태 조회는 추론을 수행하지 않습니다.

환경은 `deploy/docker/clipper-docker.env`에 있습니다.
개인 설정은 아래 파일로 덮어씁니다. 이 파일은 Git에서 제외됩니다.

```bash
cd /home/xsailor4/taeho/serving-repos/clipper
cp -n deploy/docker/clipper-docker.env deploy/docker/clipper-docker.local.env
vim deploy/docker/clipper-docker.local.env
```

개인 4TB 디스크를 사용할 때 아래 두 값을 수정하세요.

```dotenv
MODEL_DIR=/data4tb/taeho-proteus/clipper/models
RESULT_DIR=/data4tb/taeho-proteus/clipper/results
```

Docker 이미지/빌드 캐시 위치는 **현재 선택한 Docker 데몬의 data-root**로 결정됩니다.
코드 폴더를 옮겨도 이미지 저장 위치가 바뀌지 않습니다.
이 스크립트는 rootless, `DOCKER_HOST`, 시스템 Docker 설정을 변경하지 않습니다.

이미 만들어 둔 개인 rootless 데몬을 사용하려면 현재 SSH 터미널에서 선택하고 확인하세요.
다른 터미널/다른 계정의 환경변수를 변경하지 않습니다.

```bash
export DOCKER_CONTEXT=rootless
docker info --format 'root={{.DockerRootDir}} security={{json .SecurityOptions}}'
```

예상 data-root: `/data4tb/taeho-proteus/docker-data`.
`/mnt/sdb/docker`가 나오면 개인 데몬이 아닙니다. 빌드 전에 기존 rootless 연결 설정을 확인하세요.
동일 Linux 계정으로 접속한 사람은 계정의 데몬/설정 파일을 공유할 수 있습니다.
GPU·CPU·디스크는 서버 공용 자원이므로 비교 실험은 다른 작업과 동시에 실행하지 않는 것이 좋습니다.

## 2. 이미지 빌드 및 가중치 다운로드

최초 빌드는 워커만 만드는 것이 아닙니다.
원본 `QueryFrontendDockerfile`과 `ManagementFrontendDockerfile`로 C++ 프런트엔드를 직접 빌드합니다.
서버에 있는 원본 소스와 해당 커밋의 submodule을 사용합니다.
원본 소스에 변경이 있으면 빌드를 중단하며, 변경을 덮어쓰거나 초기화하지 않습니다.

```bash
bash deploy/docker/clipper-docker.sh config
bash deploy/docker/clipper-docker.sh build
bash deploy/docker/clipper-docker.sh download --models resnet-50
bash deploy/docker/clipper-docker.sh sample
bash deploy/docker/clipper-docker.sh doctor
```

`sample`은 기존 INFaaS 또는 Proteus의 JPEG를 찾아 복사합니다. 새 이미지를 인터넷에서 받지 않습니다.
다른 경로라면 직접 지정하세요. 이미 존재하는 입력은 덮어쓰지 않습니다.

```bash
bash deploy/docker/clipper-docker.sh sample --from /home/xsailor4/taeho/infaas_deploy/inputs/test-dog.jpg
```

세 시스템에서 출력되는 입력 SHA256을 대조하세요. 같은 GPU/모델명만으로 입력까지 같다고 볼 수 없습니다.

`doctor`는 V100/CUDA 연산, 고정 리비전 가중치 파일, 원본 Python RPC import를 확인합니다.
**프런트엔드 추론·SLO·배치 검증 완료를 의미하지 않습니다.**

### 원본 의존성 이미지와 로컬 빌드

실행 서버에서 `clipper/lib_base:develop`은 `manifest unknown`으로 확인됐습니다.
수정된 기본 `build`는 이 태그를 받지 않고 `deploy/docker/lib-base.Dockerfile`로
**의존성만 담긴 이미지**를 로컬에서 생성합니다. Clipper 프런트엔드/배치 구현을 대체하지 않습니다.

원본 의존성 recipe의 `git checkout -b tags/v2019.03.18.00`는 태그 checkout이 아닙니다.
새 recipe는 Folly의 해당 2019 릴리스 커밋과 CityHash 커밋을 고정합니다.
libev는 원본과 같은 4.25를 HTTPS로 받고, 파일 해시를 이미지 안에 기록합니다.
libev의 Folly 충돌 방지 처리는 유지하며, 필요한 autotools를 추가합니다.

기본 설정:

```dotenv
LIB_BASE_IMAGE=taeho-clipper/lib_base:local-v1
BUILD_LIB_BASE=1
PULL_LIB_BASE=0
BASE_BUILD_JOBS=2
```

기존 개인 env 파일이나 터미널 환경변수에 이전 값이 있으면 위 값으로 맞추세요.
`BUILD_LIB_BASE=1`일 때는 `PULL_LIB_BASE`보다 로컬 빌드가 우선합니다.
첫 빌드는 Folly 컴파일 때문에 시간이 걸리며 CPU/메모리/디스크를 사용합니다. GPU 추론이나 학습은 아닙니다.
반복 빌드는 Docker 캐시를 사용합니다.
Ubuntu 18.04 기반 레거시 의존성이므로 연구용 격리 환경에서 사용하고, 서버 빌드 결과를 별도로 확인하세요.
`lib_base`에서 실패하면 이후 download/doctor/up을 실행하지 말고 해당 빌드 오류부터 확인하세요.

이미지 ID, 원본 파일 해시, submodule 커밋은 `RESULT_DIR/build.json`에 기록합니다.
이는 빌드 provenance 기록이며 모든 오래된 외부 의존성의 완전한 bit-for-bit 재현을 보장하지는 않습니다.

## 3. deploy

```bash
bash deploy/docker/clipper-docker.sh up
bash deploy/docker/clipper-docker.sh state
```

`up`은 이 프로젝트 전용 Redis → 관리/쿼리 프런트엔드 → GPU 워커를 띄웁니다.
애플리케이션·모델 메타데이터 등록도 자동 처리합니다.
워커 HTTP 상태뿐 아니라 **원본 프런트엔드의 RPC replica 등록**까지 확인합니다.
준비 직후 `model_loaded: false`는 정상입니다. 아직 가중치를 GPU에 올리지 않았다는 뜻입니다.

별도 등록 명령도 제공하지만 `up` 후 필수로 한 번 더 할 필요는 없습니다.

```bash
bash deploy/docker/clipper-docker.sh register --models resnet-50
```

SLO/배치/모델 설정을 변경했다면 이 프로젝트만 `down` 후 `up`하세요.
이미 실행 중인 프로젝트를 `up`이 몰래 재생성하지 않습니다.
GPU에 다른 계산 프로세스가 있으면 실행을 중단하며, 다른 사람의 작업을 종료하지 않습니다.

## 4. 요청 보내기

```bash
bash deploy/docker/clipper-docker.sh query -n 11
```

실제 경로:

```text
JPEG → HTTP /imagenet/predict → 원본 Clipper 캐시/큐/적응형 배치
     → 원본 ZeroMQ Python RPC → GPU 워커의 단일 batched forward → Clipper 응답
```

JPEG는 JSON/base64 문자열 안에 담습니다. 디코딩된 이미지 바이트는 매 요청 동일합니다.
각 요청에 서로 다른 ID를 넣어 같은 이미지의 캐시/진행 중 중복 요청 결합 영향을 피합니다.
응답의 ID와 워커 기록을 대조합니다. 모델 워커에 직접 보내는 우회 추론은 하지 않습니다.

이 경로는 INFaaS의 gRPC 경로와 프로토콜이 다릅니다.
INFaaS의 DALI 전처리와도 다르므로 단순 지연시간을 시스템 정책의 성능 차이라고 해석하면 안 됩니다.

### OK와 DEFAULT의 차이

- `OK`: `default=false`이며, 실제 GPU 모델의 label과 요청 ID가 확인됨.
- `DEFAULT`: Clipper가 기한 내 모델 응답을 얻지 못해 기본 응답을 반환함. 추론 성공이 아님.
- HTTP 200만 보고 추론 성공으로 판단하지 않음.
- 기본 응답을 받았더라도 GPU에서 늦게 추론이 끝날 수 있으며, 실제 완료는 워커 JSONL로 확인.

원본 Clipper의 SLO는 프런트엔드가 요청을 받은 뒤 응답을 보내는 시간 기준입니다.
출력 `e2e`는 클라이언트 HTTP 호출의 왕복 시간입니다. 외부 전송 비용까지 같다고 간주하면 안 됩니다.
`query`에 기본 응답이 있으면 종료 코드 2를 반환합니다. 첫 cold 요청에서 예상 가능한 결과입니다.

## 5. cold/warm 실험

```bash
bash deploy/docker/clipper-docker.sh cold-test -n 11
```

이 명령은 **현재 Clipper 프로젝트만 내려서 다시 생성**합니다.
INFaaS/Proteus나 다른 프로젝트의 컨테이너는 건드리지 않습니다.
모델 다운로드 캐시, 이미지, 기존 결과는 유지합니다.

1. 첫 요청: 새 워커에서 비상주 모델 로딩을 포함한 실제 GPU 실행.
2. 첫 요청이 SLO 때문에 `DEFAULT`여도 해당 요청의 GPU 완료를 기다림.
3. 동일 JPEG로 후속 요청 10회: 실제 모델 응답, 추가 로딩 0ms, warm 평균/최소/최대 확인.

여기서 cold는 **워커 프로세스/GPU 비상주 기준**입니다.
디스크 다운로드 캐시와 OS page cache를 제거한 실험은 아닙니다.
학습, 모델 교체, 모델 선택/accuracy scaling 실험도 아닙니다.

`client-*.json`에는 요청별 e2e/기본 응답/label과 대응하는 워커 완료 기록이 저장됩니다.
`worker-*.jsonl`에는 batch size, 처리한 요청 ID, 입력 해시, 모델 준비·전처리·H2D·forward 시간이 저장됩니다.
첫 요청의 HTTP 응답 시간과 실제 GPU 완료 시간은 **서로 다른 수치**입니다.

워커 `load_ms`는 프로세서/모델 라이브러리 준비, 설정/가중치 읽기, GPU 이동을 포함합니다.
`worker_total_ms`는 워커의 해당 batch 호출 처리 시간이고 HTTP/프런트엔드 대기 시간은 제외합니다.
한 배치의 처리 시간은 요청마다 나눠 계산한 시간이 아닙니다.
GPU 단계별 측정에 동기화와 로깅 비용이 있으므로 기존 다른 시스템의 계측 없는 실행과 완전히 동일하지 않습니다.

## 6. 배치 동작 검증

먼저 `cold-test` 또는 정상 추론으로 모델을 상주시킨 뒤 실행합니다.

```bash
bash deploy/docker/clipper-docker.sh batching-test -n 128 --concurrency 16
```

동시 요청을 보내고 **해당 요청들이 포함된 실제 워커 배치 크기**를 확인합니다.
한 RPC batch를 워커가 다시 쪼개거나 자체 큐에 모으지 않습니다.
원본 Clipper가 전달한 batch에 대해 GPU forward를 한 번 실행합니다.

- `BATCHING VERIFIED`: 실제 `batch_size > 1`의 GPU forward 관측. SLO/처리량 우위를 뜻하지 않음.
- `BATCHING NOT VERIFIED`: batch 1만 관측. 종료 코드 3. 동시 요청만 보냈다고 검증 완료로 표시하지 않음.
- 기본 응답/큐에서 만료된 요청은 별도로 기록. 모든 클라이언트 요청이 실행된다고 가정하지 않음.
- 원본의 탐색용 추가 실행 때문에 워커 실행 횟수와 클라이언트 요청 수가 꼭 같지는 않음.

배치 정책 자체를 분리해 확인하려면 `BATCH_SIZE=4` 같은 고정 최대 배치로 바꾸고 `down/up`할 수 있습니다.
이 결과는 기본 적응형 배치 실험과 구분하세요. SLO를 늘린 진단 실험도 200ms 비교 결과와 섞지 마세요.

## 7. logging 및 종료

```bash
bash deploy/docker/clipper-docker.sh state --watch 2
bash deploy/docker/clipper-docker.sh logs worker
bash deploy/docker/clipper-docker.sh logs query
bash deploy/docker/clipper-docker.sh logs management
bash deploy/docker/clipper-docker.sh down
nvidia-smi -i 2
```

`state --watch`는 Ctrl+C로 끝냅니다. 워커가 떠 있으면 요청이 없을 때도 모델이 GPU 메모리에 남을 수 있습니다.
`down`은 이 프로젝트의 컨테이너/네트워크만 삭제하며, 이미지/가중치/결과는 지우지 않습니다.

레거시 `clipper_admin.metrics`는 import 시 별도 Redis 프로세스를 시작하므로 워커에서는 비활성화했습니다.
새 호환 어댑터로 해당 모듈의 import만 분리했고, 원본 `containers/python/rpc.py`의 통신/배치 코드는 그대로 사용합니다.
원본 프런트엔드 `/metrics`와 새 워커 JSONL 기록은 사용 가능합니다.
이 구성은 오래된 admin Python 패키지나 Docker socket 접근을 요구하지 않습니다.

## 8. 빌드 오류: 서브모듈의 Git 참조

프런트엔드 빌드 중 아래 오류가 나오면 가중치나 GPU 문제가 아니라 빌드 컨텍스트의 Git 메타데이터 문제입니다.

```text
fatal: not a git repository: /clipper/src/libs/spdlog/../../../.git/modules/src/libs/spdlog
```

서브모듈의 `.git`은 최상위 `.git/modules/...`를 가리키는 참조 파일입니다.
최상위 `.git`만 제외하면 컨테이너 안에는 대상이 없는 참조 파일이 남습니다.
두 프런트엔드의 `Dockerfile.dockerignore`에서 `.git`과 `**/.git`을 모두 제외합니다.
서브모듈 소스와 `src/libs/patches`는 포함하므로 원본 Dockerfile의 `git apply`는
Git 저장소 메타데이터 없이 파일에 패치를 적용합니다. 호스트의 Git 저장소와 원본 Dockerfile은 수정하지 않습니다.

수정 후 `build`를 다시 실행하세요. 기존 의존성 이미지/빌드 캐시를 삭제하거나 `--no-cache`를 사용할 필요는 없습니다.
모든 이미지 빌드가 끝나 `Build complete. Manifest:`가 나와야 `download/doctor/up`으로 진행할 수 있습니다.

## 9. 빌드 오류: GoogleTest master / CMake 버전

```text
fatal: invalid reference: master
Failed to checkout tag: 'master'
CMake 3.16 or higher is required. You are running version 3.10.2
```

원본 `CMakeLists.txt`는 GoogleTest의 `master`를 다운로드합니다.
현재 원격 브랜치/최신 테스트 라이브러리는 이 레거시 빌드 환경과 맞지 않습니다.
이는 GPU, 모델 가중치, CUDA 오류가 아니라 C++ 빌드용 테스트 의존성 문제입니다.

새 의존성 이미지에서는 GoogleTest **release-1.8.1**의 커밋
`2fe3bd994b3189899d93f1d5a881e725e046fdc2`를 고정합니다.
이미지 안의 `/opt/clipper-build-deps/googletest.git`에 호환용 `master` 참조를 만들고,
해당 이미지의 Git URL 매핑만 설정해 원본 다운로드 요청이 이 저장소를 사용하게 합니다.
이 `master`는 고정 릴리스의 로컬 별칭이지 현재 upstream master의 복원이 아닙니다.
호스트의 Git 설정, Clipper의 `CMakeLists.txt`/`configure`/원본 Dockerfile은 변경하지 않습니다.
원본 CMake 파일의 변경 여부도 빌드 전 검사합니다.

빌드 완료 후 `build.json`의 `dependency_googletest_revision`과
의존성 이미지의 `/opt/clipper-base-sources/googletest.commit`에 사용한 커밋이 기록됩니다.
기존 Folly/libev/CityHash 단계 뒤에 의존성을 추가하므로 해당 빌드 캐시를 지울 필요는 없습니다.

이 구성은 **호환 버전을 선택한 재현 환경**이며 논문 당시 모든 의존성과 동일하다는 보장은 아닙니다.

## 10. 빌드 오류: libevent / libev의 EV_READ·EV_TIMEOUT 충돌

```text
/usr/local/include/ev.h:221:3: error: expected identifier before numeric constant
EV_READ = 0x01
```

Folly의 Future 헤더는 libevent를 포함하고, Redox는 libev를 포함합니다.
libevent의 `EV_TIMEOUT/EV_READ/EV_WRITE/EV_SIGNAL` 매크로가 libev의 같은 이름의 enum을 치환해 컴파일이 실패합니다.
READ/WRITE 값도 서로 다르므로 단순 `#undef`, include 순서 변경, 경고 무시는 안전한 해결책이 아닙니다.

새 빌드 단계는 **컨테이너 이미지 안의 설치된 libevent/Folly 헤더만** 대상으로
이 네 이름에 `CLIPPER_LIBEVENT_` 접두사를 붙입니다.
libevent의 값 `TIMEOUT=1/READ=2/WRITE=4/SIGNAL=8`, libev의 값
`TIMEOUT=TIMER=256/READ=1/WRITE=2/SIGNAL=1024`는 유지합니다.
수정 전 모든 대상 헤더의 매크로 이름을 libev enum 이름과 대조합니다.
알려진 네 이름 외에 추가 충돌이 있으면 파일을 바꾸기 전에 이름과 경로를 출력하고 중단합니다.
libev와 hiredis 어댑터, 원본 Clipper/Redox 소스, 호스트의 헤더/설정은 변경하지 않습니다.
기존 libev의 libevent 호환 함수 제거는 별개의 링크 충돌 방지 조치로 계속 유지합니다.

의존성 이미지 빌드 중 실제 Folly Future와 hiredis libev 어댑터를 함께 포함해
두 include 순서에 대해 `g++ -fsyntax-only` 및 상수 값 `static_assert` 검사를 실행합니다.
성공하면 `Event header probes passed (both include orders; flag values preserved).`가 출력됩니다.
이는 헤더 컴파일 검사이며 Redis 통신/서빙 런타임 테스트 완료를 의미하지 않습니다.

변경한 의존성 헤더의 전후 SHA256은 이미지의
`/opt/clipper-base-sources/event-header-compat.json`에 기록합니다.
`build.json`의 `dependency_event_header_compatibility`에는 `libevent-prefix-v2`가 기록됩니다.
v1에서 빠졌던 `EV_TIMEOUT`도 검사 대상이며, 캐시 삭제나 `--no-cache` 없이 다시 빌드하면 됩니다.
변경 범위가 작은 빌드 호환 처리지만 전체 런타임 검증은 서버의 `up/cold-test/batching-test`로 확인해야 합니다.

이때 C++ 프런트엔드 빌드의 `DLIB WILL NOT USE CUDA`는 GPU 워커의 CUDA 실패가 아닙니다.
ResNet-50 GPU 추론은 별도의 PyTorch 워커 이미지에서 실행합니다.

## 11. 검증 범위

로컬 테스트:

```bash
python3 -m unittest discover -s tests_clipper_gpu -v
```

설정/분리/요청 ID/기본 응답/cold-warm 집계 등은 단위 테스트 대상입니다.
Windows에서 Docker/V100 실행은 하지 않았습니다.
서버의 **build → doctor → up → cold-test → batching-test**까지 성공해야 실제 실행 검증이 완료됩니다.
단일 ResNet-50의 정상 실행이 Clipper 전체 논문 재현이나 INFaaS/Proteus와의 공정한 성능 비교 완료를 의미하지 않습니다.

원본 근거:

- [원본 배치 구현](https://github.com/ucbrise/clipper/blob/9f25e3fc7f8edc891615e81c5b80d3d8aed72608/src/libclipper/src/containers.cpp)
- [원본 큐/캐시 구현](https://github.com/ucbrise/clipper/blob/9f25e3fc7f8edc891615e81c5b80d3d8aed72608/src/libclipper/src/task_executor.cpp)
- [원본 Python RPC](https://github.com/ucbrise/clipper/blob/9f25e3fc7f8edc891615e81c5b80d3d8aed72608/containers/python/rpc.py)
- [관리 API와 SLO 설명](https://github.com/ucbrise/clipper/blob/9f25e3fc7f8edc891615e81c5b80d3d8aed72608/clipper_admin/clipper_admin/clipper_admin.py)
- [원본 의존성 recipe](https://github.com/ucbrise/clipper/blob/9f25e3fc7f8edc891615e81c5b80d3d8aed72608/dockerfiles/ClipperLibBaseDockerfile)
- [Folly 2019 릴리스](https://github.com/facebook/folly/releases/tag/v2019.03.18.00)
- [고정 CityHash 커밋](https://github.com/google/cityhash/commit/8af9b8c2b889d80c22d6bc26ba0df1afb79a30db)
- [원본 GoogleTest 다운로드 설정](https://github.com/ucbrise/clipper/blob/9f25e3fc7f8edc891615e81c5b80d3d8aed72608/CMakeLists.txt)
- [GoogleTest 1.8.1 릴리스](https://github.com/google/googletest/releases/tag/release-1.8.1)
- [고정 GoogleTest의 CMake 요구 버전](https://github.com/google/googletest/blob/2fe3bd994b3189899d93f1d5a881e725e046fdc2/CMakeLists.txt)
- [고정 Folly의 libevent 헤더 포함](https://github.com/facebook/folly/blob/94a77ab3adee9f3ab184e23fe0c38d4cb5086635/folly/portability/Event.h)
- [libevent 이벤트 상수](https://github.com/libevent/libevent/blob/release-2.1.8-stable/include/event2/event.h)
- [원본이 인용한 libev/libevent 링크 충돌 논의](https://github.com/facebook/folly/issues/601#issuecomment-347100410)
