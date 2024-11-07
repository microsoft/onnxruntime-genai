# Runtime Options

This file will provide details on the usage of SetRuntimeOption API. It will list all the current key value pairs which can be used as an input for this API.

## Set Terminate

Set Terminate is a runtime option to terminate the current session or continue/restart an already terminated session. There are two valid ways to call Set Terminate.

To enable terminate, the valid pair is: ("set_terminate", "1")

To disable terminate, the valid pair is: ("set_terminate", "0")

Key: "set_terminate"

Accepted values: ("0", "1")
