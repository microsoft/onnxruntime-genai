// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator.h"

#include "httplib.h"
#include "json.hpp"

namespace Generators::Server {

struct PromptRequest {
  std::string prompt;
  int params;
};

struct ListenerParams {
  int32_t port = 8080;
  std::string hostname = "127.0.0.1";
};

const char* Prompt = "prompt";
const char* MaxLength = "n_predict";

struct ServerParams {
  std::string model_path;

  ServerParams(int argc, char** argv) {
    if (argc < 2) {
      throw std::invalid_argument("Usage: Path to the model is required.");
    }
    model_path = argv[1];
  }
};

}  // namespace Generators::Server

int main(int argc, char** argv) {
  using namespace Generators::Server;
  using json = nlohmann::json;

  ListenerParams params;
  ServerParams server_params(argc, argv);

  std::unique_ptr<httplib::Server> server = std::make_unique<httplib::Server>();
  std::unique_ptr<Generators::Server::Generator> generator =
      std::make_unique<Generators::Server::Generator>(server_params.model_path);

  server->Post("/generate", [&generator](const httplib::Request& req, httplib::Response& res) {
    json data = json::parse(req.body);
    if (!data.count(Prompt) || !data.count("n_predict")) {
      json error_response;
      error_response["error_message"] = "Invalid request: Missing prompt and/or num tokens to predict.";
      res.set_content(error_response.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
      return;
    }

    json response;
    std::string generated = generator->Generate(data.at("prompt"), data.at("n_predict"));
    response["content"] = generated;
    res.set_content(response.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
  });

  server->listen(params.hostname, params.port);

  return 0;
}