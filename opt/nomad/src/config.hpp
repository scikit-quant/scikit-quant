#ifdef _WIN32
#define NOMAD_UNUSED(x)
#else
#define NOMAD_UNUSED(x) x __attribute__((unused))
#endif
