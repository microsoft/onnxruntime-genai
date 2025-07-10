#!/bin/bash

# Original test case
echo "Original test case - San Diego travel recommendations:"
curl http://localhost:8080/completions -H "Content-Type: application/json" \
 -d '{"messages":[{"role": "system", "content": "You are a helpful AI Assistant. Please answer the questions very accurately. Use emojis and markdown as appropriate"}, {"role": "user", "content": "What are the top 5 places to visit in San Diego? Be brief."}], "max_tokens": 1200, "temperature": 0.7}' -vvv

echo -e "\n\n================================================================="
echo "Tool calling test case - Flight and Hotel booking:"

# New test case with tool calling
curl http://localhost:8080/completions -H "Content-Type: application/json" \
 -d '{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant with these tools."
        },
        {
            "role": "user",
            "content": "book flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10 , then book hotel from 2025-12-04 to 2025-12-10 in Paris"
        }
    ],
    "tools": [
        {
            "name": "booking_flight_tickets",
            "description": "booking flights",
            "parameters": {
                "origin_airport_code": {
                    "description": "The name of Departure airport code",
                    "type": "string"
                },
                "destination_airport_code": {
                    "description": "The name of Destination airport code",
                    "type": "string"
                },
                "departure_date": {
                    "description": "The date of outbound flight",
                    "type": "string"
                },
                "return_date": {
                    "description": "The date of return flight",
                    "type": "string"
                }
            }
        },
        {
            "name": "booking_hotels",
            "description": "booking hotel",
            "parameters": {
                "destination": {
                    "description": "The name of the city",
                    "type": "string"
                },
                "check_in_date": {
                    "description": "The date of check in",
                    "type": "string"
                },
                "checkout_date": {
                    "description": "The date of check out",
                    "type": "string"
                }
            }
        }
    ],
    "temperature": 0.00001,
    "max_tokens": 4096,
    "top_p": 1.0,
    "do_sample": false
}' -v
