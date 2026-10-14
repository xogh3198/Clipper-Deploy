# Dependency-only compatibility recipe based on the original
# dockerfiles/ClipperLibBaseDockerfile at 9f25e3fc7f8edc891615e81c5b80d3d8aed72608.
# Does not alter or substitute Clipper's C++ frontend/queue/batching implementation.
FROM ubuntu:18.04
ARG BUILD_JOBS=2
ARG FOLLY_REVISION=94a77ab3adee9f3ab184e23fe0c38d4cb5086635
ARG CITYHASH_REVISION=8af9b8c2b889d80c22d6bc26ba0df1afb79a30db
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget git g++ cmake make pkg-config autoconf automake libtool \
    libboost-all-dev libevent-dev libdouble-conversion-dev libgoogle-glog-dev \
    libgflags-dev libiberty-dev liblz4-dev liblzma-dev libsnappy-dev \
    zlib1g-dev binutils-dev libjemalloc-dev libssl-dev libhiredis-dev libzmq5 libzmq5-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/build /opt/clipper-base-sources
# Fetch the actual release commit; the old 'git checkout -b tags/...' created
# a branch at HEAD instead of checking out the intended 2019 release.
RUN git init /opt/build/folly \
    && git -C /opt/build/folly remote add origin https://github.com/facebook/folly.git \
    && git -C /opt/build/folly fetch --depth 1 origin "${FOLLY_REVISION}" \
    && git -C /opt/build/folly checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/build/folly rev-parse HEAD)" = "${FOLLY_REVISION}" \
    && git -C /opt/build/folly rev-parse HEAD > /opt/clipper-base-sources/folly.commit \
    && mkdir /opt/build/folly/_build \
    && cd /opt/build/folly/_build \
    && CXXFLAGS=-fPIC CFLAGS=-fPIC cmake .. -DBUILD_SHARED_LIBS=ON \
       -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_TESTS=OFF \
    && make -j"${BUILD_JOBS}" && make install \
    && cd /opt/build && rm -rf /opt/build/folly
# Same libev 4.25 as the original recipe, without libevent compatibility files:
# these conflict with Folly's event.h. HTTPS certificate validation stays enabled.
RUN cd /opt/build \
    && wget --https-only -O libev-4.25.tar.gz https://dist.schmorp.de/libev/Attic/libev-4.25.tar.gz \
    && sha256sum libev-4.25.tar.gz > /opt/clipper-base-sources/libev.sha256 \
    && tar -xzf libev-4.25.tar.gz \
    && cd libev-4.25 \
    && sed -i 's/event\.[ch]//g' Makefile.am \
    && autoreconf -fi && ./configure \
    && make -j"${BUILD_JOBS}" && make install \
    && cd /opt/build && rm -rf /opt/build/libev-4.25 /opt/build/libev-4.25.tar.gz
RUN git init /opt/build/cityhash \
    && git -C /opt/build/cityhash remote add origin https://github.com/google/cityhash.git \
    && git -C /opt/build/cityhash fetch --depth 1 origin "${CITYHASH_REVISION}" \
    && git -C /opt/build/cityhash checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/build/cityhash rev-parse HEAD)" = "${CITYHASH_REVISION}" \
    && git -C /opt/build/cityhash rev-parse HEAD > /opt/clipper-base-sources/cityhash.commit \
    && cd /opt/build/cityhash && ./configure \
    && make -j"${BUILD_JOBS}" all check CXXFLAGS="-g -O3" && make install \
    && cd /opt/build && rm -rf /opt/build/cityhash \
    && ldconfig
# The original Clipper CMakeLists downloads GoogleTest's obsolete 'master'.
# Keep that file unchanged: provide a local, pinned compatibility branch and
# rewrite only the GoogleTest URL inside this image (never on the host).
# Append after the expensive dependency layers so those caches remain usable.
ARG GOOGLETEST_REVISION=2fe3bd994b3189899d93f1d5a881e725e046fdc2
RUN mkdir -p /opt/clipper-build-deps \
    && git init --bare /opt/clipper-build-deps/googletest.git \
    && git -C /opt/clipper-build-deps/googletest.git fetch --depth 1 \
       https://github.com/google/googletest.git "${GOOGLETEST_REVISION}:refs/heads/master" \
    && git -C /opt/clipper-build-deps/googletest.git symbolic-ref HEAD refs/heads/master \
    && test "$(git -C /opt/clipper-build-deps/googletest.git rev-parse master)" = "${GOOGLETEST_REVISION}" \
    && git -C /opt/clipper-build-deps/googletest.git rev-parse master \
       > /opt/clipper-base-sources/googletest.commit \
    && git config --system url."file:///opt/clipper-build-deps/googletest.git".insteadOf \
       https://github.com/google/googletest.git
# The pinned Folly headers expose libevent macros, but Redox imports libev enums
# with the same names and different values. Scope this compatibility rewrite to
# installed libevent/Folly headers in the image, preserving all flag values.
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/docker/fix-event-headers.py deploy/docker/event-header-probe.cpp /opt/clipper-build-deps/
RUN python3 /opt/clipper-build-deps/fix-event-headers.py \
    && g++ -std=c++14 -fsyntax-only -pthread -DBOOST_THREAD_VERSION=4 /opt/clipper-build-deps/event-header-probe.cpp \
    && g++ -std=c++14 -DCLIPPER_EV_FIRST -fsyntax-only -pthread -DBOOST_THREAD_VERSION=4 /opt/clipper-build-deps/event-header-probe.cpp \
    && printf '%s\n' 'Event header probes passed (both include orders; flag values preserved).'
LABEL org.clipper.gpu.dependency-recipe="local-v1" \
      org.clipper.gpu.folly="94a77ab3adee9f3ab184e23fe0c38d4cb5086635" \
      org.clipper.gpu.cityhash="8af9b8c2b889d80c22d6bc26ba0df1afb79a30db" \
      org.clipper.gpu.googletest="${GOOGLETEST_REVISION}" \
      org.clipper.gpu.event-header-compatibility="libevent-prefix-v2"
