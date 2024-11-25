# Runtime Options

This file will provide details on the usage of SetRuntimeOption API. It will list all the current key value pairs which can be used as an input for this API.

## Set Terminate

Set Terminate is a runtime option to terminate the current session or continue/restart an already terminated session. The current session will throw an exception when the terminate option is enabled and the user will need to handle that scenario, examples/c/src/phi3_terminate.cpp contains an example for this.

To terminate the generation, use this key value pair: ("terminate_session", "1")

To recover from a terminated state, use this key value pair: ("terminate_session", "0")

Key: "terminate_session"

Accepted values: ("0", "1")
