// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"
#include "model_executor.h"
#include "scheduler.h"

/**
 * @file engine.h
 * @brief Defines the Engine class, which serves as the core component for managing
 *        model executions and request handling. It provides a way to continuously
 *        add, remove, and process requests by dynamically scheduling them.
 */

namespace Generators {

/**
 * @class Engine
 * @brief The Engine class is responsible for managing requests, executing models,
 *        and coordinating scheduling and caching mechanisms.
 *
 * The Engine class is designed to handle multiple requests concurrently, allowing
 * for efficient execution of models by dynamically batching requests.
 * It is the entry point for adding and processing requests.
 */
struct Engine : std::enable_shared_from_this<Engine>,
                LeakChecked<Engine>,
                ExternalRefCounted<Engine> {
  /**
   * @brief Constructs an Engine instance with the specified model.
   * @param model A shared pointer to the Model object to be used by the Engine
   *              and its components.
   */
  Engine(std::shared_ptr<Model> model);

  /**
   * @brief Adds a request to the Engine for processing.
   * @param request A shared pointer to the Request object to be added.
   */
  void AddRequest(std::shared_ptr<Request> request);

  /**
   * @brief Removes a previously added request from the Engine.
   * @param request A shared pointer to the Request object to be removed.
   */
  void RemoveRequest(std::shared_ptr<Request> request);

  /**
   * @brief Advances the state of a subset of the Requests the Engine is currently
   *        serving.
   *
   * This function schedules the execution of the model for the subset of requests
   * that are ready to be processed (as determined by the scheduling strategy).
   * Once these requests are scheduled, the Engine offloads the execution to the
   * model executor and updates the requests' states with the newly generated
   * tokens.
   */
  std::shared_ptr<Request> Step();

  /**
   * @brief Checks if there are any pending requests in the Engine.
   * @return True if there are pending requests; otherwise, false.
   */
  bool HasPendingRequests() const;

 private:
  std::shared_ptr<Model> model_;                         // The model used by the Engine.
  std::shared_ptr<CacheManager> cache_manager_;          // The cache manager for handling cached data.
  std::unique_ptr<Scheduler> scheduler_;                 // The scheduler responsible for managing execution order.
  std::unique_ptr<ModelExecutor> model_executor_;        // The executor responsible for running the model.
  std::queue<std::shared_ptr<Request>> ready_requests_;  // The list of requests that are ready for the application to process.
};

}  // namespace Generators
