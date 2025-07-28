#!/bin/bash

# Test case for SLM server with tool calling functionality
# This script tests the booking_flight_tickets and booking_hotels tools

echo "Testing SLM server with tool calling - Flight and Hotel booking scenario"
echo "================================================================="

# Test 1: Flight and Hotel booking with tools
echo "Test 1: Flight and Hotel booking from Beijing to Paris"
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

echo -e "\n\n"
echo "================================================================="

# Test 2: Simple tool calling test - Flight only
echo "Test 2: Flight booking only from New York to London"
curl http://localhost:8080/completions -H "Content-Type: application/json" \
 -d '{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful travel assistant."
        },
        {
            "role": "user",
            "content": "I need to book a flight from JFK to LHR on 2025-08-15, returning on 2025-08-22"
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
        }
    ],
    "temperature": 0.1,
    "max_tokens": 2048,
    "top_p": 0.9,
    "do_sample": true
}' -v

echo -e "\n\n"
echo "================================================================="

# Test 3: Hotel booking only
echo "Test 3: Hotel booking only in Tokyo"
curl http://localhost:8080/completions -H "Content-Type: application/json" \
 -d '{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful hotel booking assistant."
        },
        {
            "role": "user", 
            "content": "I need to book a hotel in Tokyo from 2025-09-01 to 2025-09-05"
        }
    ],
    "tools": [
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
    "temperature": 0.2,
    "max_tokens": 1024,
    "top_p": 0.95,
    "do_sample": true
}' -v

echo -e "\n\n"
echo "================================================================="
echo "All tool calling tests completed!"
