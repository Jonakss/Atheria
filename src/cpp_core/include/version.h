// version.h - Versión del motor nativo C++ de Atheria 4
// Sigue Semantic Versioning (SemVer): MAJOR.MINOR.PATCH

#ifndef ATHERIA_VERSION_H
#define ATHERIA_VERSION_H

#define ATHERIA_NATIVE_VERSION_MAJOR 4
#define ATHERIA_NATIVE_VERSION_MINOR 1
#define ATHERIA_NATIVE_VERSION_PATCH 14
#define ATHERIA_NATIVE_VERSION_STRING "4.1.14"

// Helper macros para construir la versión
#define ATHERIA_NATIVE_VERSION_CODE(major, minor, patch) \
    (((major) << 16) | ((minor) << 8) | (patch))

#define ATHERIA_NATIVE_VERSION \
    ATHERIA_NATIVE_VERSION_CODE( \
        ATHERIA_NATIVE_VERSION_MAJOR, \
        ATHERIA_NATIVE_VERSION_MINOR, \
        ATHERIA_NATIVE_VERSION_PATCH \
    )

#endif // ATHERIA_VERSION_H

