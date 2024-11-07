# Runtime Options

This file will provide details on the usage of SetRuntimeOption API. It will list all the current key value pairs which can be used as an input for this API.

## Set Terminate

Set Terminate is a runtime option to terminate the current session or continue/restart an already terminated session. The current session will crash when the terminate option is enabled and the user will need to handle that scenario, examples/c/src/phi3_terminate.cpp contains an example for this.

There are two valid ways to call Set Terminate.

To enable terminate, the valid pair is: ("terminate_session", "1")

To disable terminate, the valid pair is: ("terminate_session", "0")

Key: "terminate_session"

Accepted values: ("0", "1")
