**TESIS MCPI BACKEND**
---

Proyecto de desarrollo para la tesis de la Maestría en Ciencias del Procesamiento de la Información.

- [Requisitos](#requisitos)
- [Instrucciones de instalación](#instrucciones-de-instalación)
- [Instrucciones de ejecución](#instrucciones-de-ejecución)
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

# Instrucciones de ejecución

Activar virtualenv:

```bash
pipenv shell
```

Correr proyecto:

```bash
python run.py
```

# Comandos útiles

## Manejo de procesos

Obtener id del proceso iniciado por un comando:

```bash
ps aux | grep python | awk '{ print $2,$11,$12 }'
```

Detener un proceso por medio de su id:

```bash
kill -9 <pid>
```
