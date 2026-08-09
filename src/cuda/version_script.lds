#_init and _fini should be local
VERS_1.0 {
  global:
    GetInterface;
    RegisterCustomOps;

  # Hide everything else.
  local:
    *;
};