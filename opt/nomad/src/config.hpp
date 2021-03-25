#ifdef _WIN32
#define NOMAD_UNUSED(x)
#define NOMAD_PRETTY_FUNCTION __FUNCSIG__
#else
#define NOMAD_UNUSED(x) x __attribute__((unused))
#define NOMAD_PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif
