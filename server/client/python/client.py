# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

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
