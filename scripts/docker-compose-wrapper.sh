#!/bin/bash
# Wrapper script para docker-compose con compatibilidad de requests/urllib3
# Solución temporal para docker-compose 1.29.2 con versiones nuevas de requests

# Exportar variable de entorno para forzar compatibilidad
export DOCKER_HOST=unix:///var/run/docker.sock

# Ejecutar docker-compose con todos los argumentos
/usr/bin/docker-compose "$@"
