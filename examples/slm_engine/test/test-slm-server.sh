#!/bin/bash
curl http://localhost:8080/completions -H "Content-Type: application/json" \
 -d '{"messages":[{"role": "system", "content": "You are a helpful AI Assistant. Please answer the questions very accurately. Use emojis and markdown as appropriate"}, {"role": "user", "content": "What are the top 5 places to visit in San Diego? Be brief."}], "max_tokens": 1200, "temperature": 0.7}' -vvv
