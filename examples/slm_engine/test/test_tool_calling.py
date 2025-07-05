#!/usr/bin/env python3
"""
Test script for SLM server with tool calling functionality
This script tests the booking_flight_tickets and booking_hotels tools
"""

import requests
import json

def test_tool_calling():
    """Test tool calling functionality with flight and hotel booking"""
    
    url = "http://localhost:8080/completions"
    headers = {"Content-Type": "application/json"}
    
    # Test case 1: Flight and Hotel booking
    print("=" * 70)
    print("Test 1: Flight and Hotel booking from Beijing to Paris")
    print("=" * 70)
    
    payload1 = {
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
        "do_sample": False
    }
    
    try:
        response1 = requests.post(url, headers=headers, json=payload1, timeout=30)
        print(f"Status Code: {response1.status_code}")
        if response1.status_code == 200:
            result = response1.json()
            print("Response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Error: {response1.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    print("\n" + "=" * 70)
    print("Test 2: Flight booking only (JFK to LHR)")
    print("=" * 70)
    
    # Test case 2: Flight only
    payload2 = {
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
        "do_sample": True
    }
    
    try:
        response2 = requests.post(url, headers=headers, json=payload2, timeout=30)
        print(f"Status Code: {response2.status_code}")
        if response2.status_code == 200:
            result = response2.json()
            print("Response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Error: {response2.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    print("\n" + "=" * 70)
    print("Test 3: Hotel booking only (Tokyo)")
    print("=" * 70)
    
    # Test case 3: Hotel only
    payload3 = {
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
        "do_sample": True
    }
    
    try:
        response3 = requests.post(url, headers=headers, json=payload3, timeout=30)
        print(f"Status Code: {response3.status_code}")
        if response3.status_code == 200:
            result = response3.json()
            print("Response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Error: {response3.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    print("\n" + "=" * 70)
    print("All tool calling tests completed!")
    print("=" * 70)

if __name__ == "__main__":
    print("Starting SLM Server Tool Calling Tests...")
    test_tool_calling()
