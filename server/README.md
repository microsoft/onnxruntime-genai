## ONNX Runtime Gen AI Server

### Build

Built by default when the ONNX Runtime GenAI library is built.

```bash
python build.py --skip_csharp --cuda_home <path_to_cuda_home>
```

### Run

The above build command will build the server executable. The executable requires a model to load and run which can be passed as an argument to the executable.

```bash
./build/server/server <path_to_model>
```

The server listens on host 127.0.0.1 and port 8080 and the generation url is `http://127.0.0.1:8080/generate`. Here is an example [client](client/python/client.py):

```py
import json
from urllib import request

prompt = "def is_prime(num):"
data = {"prompt": prompt, "n_predict": 200}
encoded_data = json.dumps(data).encode()

req = request.Request('http://127.0.0.1:8080/generate', data=encoded_data)
req.add_header('Content-Type', 'application/json')
response = request.urlopen(req)

response = json.loads(response.read().decode('utf-8'))
print(response["content"])

```
