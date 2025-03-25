import argparse
import json
import requests

BLUE = "\033[34m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
RED = "\033[31m"
CLEAR = "\033[0m"


def launch_server(server_binary: str, model_path: str):
    import subprocess

    pid = subprocess.Popen(
        [
            str(server_binary),
            "--model_path",
            str(model_path),
            "--port_number",
            "8000",
        ]
    )

    # Wait until the server starts to listen
    started = False
    timeout_countdown = 30
    url = "http://localhost:8000"
    while not started:
        try:
            response = requests.get(url)
            json_response = json.loads(response.text)
            if json_response["response"]["status"] == "success":
                print(
                    f"{MAGENTA}Engine State: {json_response['response']['engine_state']}{CLEAR}"
                )
                started = True
        except Exception as ex:
            # Initially the server may not be ready to accept requests
            # We want to ignore and retry
            pass  # Ignore all exceptions

        # Sleep for a bit
        import time

        time.sleep(1)
        timeout_countdown = timeout_countdown - 1
        if timeout_countdown == 0:
            raise Exception("Server did not start in time")

    return pid


# This function tests the OpenAI API Interface
def run_test(url: str):

    # Test the API
    print("Testing the API with a test message")
    test_message = """
        {"messages":
            [
                {"role": "system", "content": "You are a helpful assistant. Be very brief and precise"}, 
                {"role": "user", "content": "How to make pizza in five steps?"}
            ], 
                "max_tokens": 1200
        }
    """
    json_message = json.loads(test_message)
    response = requests.post(url + "/completions", json=json_message)
    if response.status_code != 200:
        print(f"{RED}Error: {response.status_code}{CLEAR}")
        raise Exception("Error in the API")

    json_response = json.loads(response.text)
    print(f"Question: {json_response['question']}")
    print(f"Answer: {json_response['choices'][0]['message']['content']}")
    print(f"{BLUE}KPI: {json_response['kpi']}{CLEAR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SLM Engine Using HTTP API")

    # Adding arguments
    parser.add_argument(
        "-s",
        "--server_binary_path",
        help="Path to the SLM Server binary",
    )
    parser.add_argument(
        "-u",
        "--url",
        help="URL of the server to test",
    )

    parser.add_argument("-m", "--model_path", help="Path to the ONNX model")

    args = parser.parse_args()
    if args.server_binary_path is None:
        if args.url is None:
            raise Exception("Either server_binary_path or url must be provided")
        # Run the test using existing server
        run_test(args.url)
    else:
        # Launch the server and run the
        launch_server(args.server_binary_path, args.model_path)
        url = "http://localhost:8000"
        run_test(url)
