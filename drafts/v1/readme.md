# MCPI Tucson Notebook

**Guardar requerimientos para instalación con conda:**

```bash
conda list -e > requirements.txt
```

**Crear virtualenv de conda instalando requerimientos:**

```bash
conda create --name <env> --file requirements.txt
```

**Instalar requerimientos de conda a partir de archivo:**

```bash
conda install --file requirements.txt
```

**Guardar requerimientos para instalación con pip:**

```bash
conda activate <env>
conda install pip
pip freeze > requirements.txt
```

**Crear virtualenv de pip instalando requerimientos:**

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
