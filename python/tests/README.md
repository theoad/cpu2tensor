# Python checks

Consumer/signal fixtures check framing and owned CPU/device tensors. Learning tests
check sample attribution, held-out splits and negative controls. Stdio tests check
stream boundaries, actions, cancellation and the optional Gym wrapper. Remote tests
exercise the actual operator worker; fixture tests do not replace those checks.
See [validation](../../docs/validation.md) and [the examples](../../example/README.md).
