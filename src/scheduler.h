#include <queue>

using RequestIDType = int64_t;

class Scheduler {
 public:
  Scheduler();
  void AddRequest(RequestIDType request_id);
  void Schedule();
  void FreeFinishedRequests();

 private:
  std::queue<RequestIDType> waiting_requests_;
  std::queue<RequestIDType> running_requests_;
};
