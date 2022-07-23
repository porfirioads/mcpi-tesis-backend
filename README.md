**TESIS MCPI BACKEND**
---

Proyecto de desarrollo para la tesis de la Maestría en Ciencias del Procesamiento de la Información.

- [Requisitos](#requisitos)
- [Instrucciones de instalación](#instrucciones-de-instalación)
- [Comandos útiles](#comandos-útiles)
  - [Manejo de procesos](#manejo-de-procesos)

# Requisitos

- Pipenv
- Nodejs

# Instrucciones de instalación

Instalar dependencias:

```bash
pipenv install --dev
```

# Comandos útiles

## Manejo de procesos

Obtener id del proceso iniciado por un comando:

```bash
ps aux | grep python | awk '{ print $2,$11,$12 }'
```