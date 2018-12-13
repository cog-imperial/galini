#include "uid.h"
#include <atomic>

static std::atomic<galini::uid_t> _uid_counter(0);

namespace galini {

uid_t generate_uid() {
  return _uid_counter++;
}

} // namespace galini
