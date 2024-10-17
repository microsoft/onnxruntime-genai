#_init and _fini should be local
VERS_1.0 {
  global:
    GetInterface;    

  # Hide everything else.
  local:
    *;
};