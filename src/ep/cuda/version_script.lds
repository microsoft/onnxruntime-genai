#_init and _fini should be local
VERS_1.0 {
  global:
    GetInterface;
    GetInterfaceVersion;
    qwen38_plane_*;

  # Hide everything else.
  local:
    *;
};