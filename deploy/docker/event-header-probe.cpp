// Check the actual installed headers in both include orders. No GPU required.
#include <cstdlib>
#include <sys/types.h>

#ifdef CLIPPER_EV_FIRST
#include <hiredis/adapters/libev.h>
#include <folly/futures/Future.h>
#else
#include <folly/futures/Future.h>
#include <hiredis/adapters/libev.h>
#endif

#if defined(EV_TIMEOUT) || defined(EV_READ) || defined(EV_WRITE) || defined(EV_SIGNAL)
#error "libevent macros must not shadow libev enum names"
#endif

static_assert(EV_READ == 0x01, "libev READ must retain its own value");
static_assert(EV_WRITE == 0x02, "libev WRITE must retain its own value");
static_assert(EV_TIMER == 0x100, "libev TIMER must retain its own value");
static_assert(EV_TIMEOUT == EV_TIMER, "libev TIMEOUT alias changed");
static_assert(EV_SIGNAL == 0x400, "libev SIGNAL must retain its own value");
static_assert(CLIPPER_LIBEVENT_EV_TIMEOUT == 0x01, "libevent TIMEOUT value changed");
static_assert(CLIPPER_LIBEVENT_EV_READ == 0x02, "libevent READ value changed");
static_assert(CLIPPER_LIBEVENT_EV_WRITE == 0x04, "libevent WRITE value changed");
static_assert(CLIPPER_LIBEVENT_EV_SIGNAL == 0x08, "libevent SIGNAL value changed");

int main() { return 0; }
