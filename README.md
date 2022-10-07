**TESIS MCPI BACKEND**
---

Proyecto de desarrollo para la tesis de la Maestría en Ciencias del Procesamiento de la Información.

- [Requisitos](#requisitos)
- [Instrucciones de instalación](#instrucciones-de-instalación)
- [Instrucciones de ejecución](#instrucciones-de-ejecución)
- [Comandos útiles](#comandos-útiles)
  - [Manejo de procesos](#manejo-de-procesos)
- [Recursos](#recursos)
  - [Samples originales](#samples-originales)
  - [Respuestas dudosas](#respuestas-dudosas)
  - [Dataset final](#dataset-final)

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

# Recursos

- [Resultados encuesta](https://drive.google.com/drive/u/1/folders/19EUNpSZYqA0abc3yoJ8vnRST6V9SuH8E)

## Samples originales

- [Samples Dr. Celaya](https://docs.google.com/spreadsheets/d/1aeYyg5jbzB2ta6fNdsf_uYCLQvybZ-x1fdFKv8iLbDE/edit#gid=2129512129)
- [Samples Dr. Huitzy](https://docs.google.com/spreadsheets/d/10WPWgKDDN3GAEFuklFCGSz0HB26amN7o_W8dMqUSKrg/edit#gid=640831378)
- [Samples Porfirio](https://docs.google.com/spreadsheets/d/1aIOrp1windsA12OBMRop_IFkGH3Na6AYaglUJMusYjs/edit#gid=1092706847)
- [Samples Tucson](https://docs.google.com/spreadsheets/d/1lNre50rzHjvlBqz1_EJRRayfTy7T2KprSd9XnF1kqt0/edit#gid=1446544166)
## Respuestas dudosas

- [Respuestas dudosas Dr. Celaya ](https://docs.google.com/spreadsheets/d/1VOciAb0kHwfiKjMKteNTNH6qNdFurDc1gbM0lTjCRK4/edit#gid=2129512129)
- [Respuestas dudosas Dr. Huitzy ](https://docs.google.com/spreadsheets/d/1tsN1sGY0NmXvLcngHdyOIidOJagKiNlfNXo6b2fCnIc/edit#gid=2129512129)
- [Respuestas dudosas Porfirio ](https://docs.google.com/spreadsheets/d/1ryfWPxuNS3qtSYal7YoQ-ihRMXxRgvVbku7eeLaecaY/edit#gid=2129512129)

## Dataset final

- [Respuestas Tucson](https://docs.google.com/spreadsheets/d/19GgVMb-Aq1c-rAwh_9mPyeLuWu6FZqg_eoA_HWuJGDc/edit#gid=345668867)