# SLM Engine Tool Calling Tests

This directory contains test cases for the SLM (Small Language Model) engine with tool calling functionality.

## Test Files

### 1. `test-slm-server.sh` (Updated)
- **Original test**: Basic chat completion test asking about San Diego travel recommendations
- **New test**: Tool calling test for flight and hotel booking from Beijing to Paris

### 2. `test-slm-server-tools.sh` (New)
Comprehensive tool calling test suite with three scenarios:
- **Test 1**: Combined flight and hotel booking (Beijing to Paris)
- **Test 2**: Flight booking only (JFK to LHR) 
- **Test 3**: Hotel booking only (Tokyo)

### 3. `test_tool_calling.py` (New)
Python version of the tool calling tests with better response formatting and error handling.

## Tool Definitions

The tests use two main tools:

### `booking_flight_tickets`
Parameters:
- `origin_airport_code`: Departure airport code (string)
- `destination_airport_code`: Destination airport code (string) 
- `departure_date`: Outbound flight date (string)
- `return_date`: Return flight date (string)

### `booking_hotels`
Parameters:
- `destination`: City name (string)
- `check_in_date`: Hotel check-in date (string)
- `checkout_date`: Hotel check-out date (string)

## Usage

### Prerequisites
Make sure the SLM server is running on `http://localhost:8080`

### Running the Tests

#### Bash Tests
```bash
# Run the updated original test (includes tool calling)
./test-slm-server.sh

# Run comprehensive tool calling tests
./test-slm-server-tools.sh
```

#### Python Test
```bash
# Run Python version (requires requests library)
python3 test_tool_calling.py

# Or make it executable and run directly
chmod +x test_tool_calling.py
./test_tool_calling.py
```

### Installing Python Dependencies
If running the Python test, make sure you have the `requests` library:
```bash
pip install requests
```

## Test Scenarios

### Scenario 1: Beijing to Paris Trip
- **Flight**: PEK (Beijing) → CDG (Paris), Dec 4-10, 2025
- **Hotel**: Paris, Dec 4-10, 2025
- **Parameters**: Very low temperature (0.00001), deterministic sampling

### Scenario 2: New York to London Flight
- **Flight**: JFK (New York) → LHR (London), Aug 15-22, 2025
- **Parameters**: Low temperature (0.1), sampling enabled

### Scenario 3: Tokyo Hotel Booking
- **Hotel**: Tokyo, Sep 1-5, 2025
- **Parameters**: Low temperature (0.2), sampling enabled

## Expected Response Format

The SLM should respond with tool calls in a structured format, typically including:
- Tool name identification
- Parameter extraction from user request
- Proper airport code mapping (e.g., Beijing → PEK, Paris → CDG)
- Date formatting and validation

## Troubleshooting

1. **Connection refused**: Ensure SLM server is running on port 8080
2. **Tool not recognized**: Verify the SLM model supports tool calling
3. **Parameter errors**: Check that all required parameters are provided in the tool definitions
4. **Python import errors**: Install required dependencies with `pip install requests`
