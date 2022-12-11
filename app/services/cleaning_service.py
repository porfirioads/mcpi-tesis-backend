import re
from app.services.dataset_service import DatasetService
from app.services.phrase_replacer_service import PhraseReplacerService
from app.utils.singleton import SingletonMeta
from app.schemas.common_schemas import FileUpload
from textblob import TextBlob
from datetime import datetime
from app.config import logger

GENERATED_COLUMNS = [
    'length', 'cc', 'vbp', 'cd', 'dt', 'ex', 'fw', 'inn', 'jj', 'jjr', 'jjs',
    'ls', 'md', 'nn', 'nns', 'nnp', 'nnps', 'pdt', 'posss', 'prp', 'rb', 'rbr',
    'rp', 'to', 'uh', 'vb', 'vbd', 'vbg', 'vbn', 'vbz', 'wdt', 'wp', 'wps',
    'wrb', 'pos_length', 'neg_length', 'pos_acumulado', 'neg_acumulado',
    'polarity', 'cant_caracteres'
]


class CleaningService(metaclass=SingletonMeta):
    dataset_service = DatasetService()
    phrase_replacer_service = PhraseReplacerService()

    def calculate_data_fields(self, text: str) -> list:
        cc = 0  # Conjunción de coordinación
        cd = 0  # Dígito cardinal
        dt = 0  # Determinador
        ex = 0  # Existencial (allí)
        fw = 0  # Palabra extranjera
        inn = 0  # IN Preposición/junta subordinación
        jj = 0  # Adjetivo
        jjr = 0  # Adjetivo comparativo
        jjs = 0  # Adjetivo superlativo
        ls = 0  # Marcador de lista
        md = 0  # Modal (podría, podrá)
        nn = 0  # Sustantivo singular
        nns = 0  # Sustantivo plural
        nnp = 0  # Sustantivo propio
        nnps = 0  # Sustantivo propio plural
        pdt = 0  # Predeterminado
        posss = 0  # Posesivo
        prp = 0  # Pronombre personal
        rb = 0  # RB Adverbio
        rbr = 0  # Adverbio comparativo
        rp = 0  # Particular
        to = 0  # Particular de infinitivo
        uh = 0  # Interjección
        vb = 0  # Verbo
        vbd = 0  # VBD ?
        vbg = 0  # VBG ?
        vbn = 0  # Verbo participio
        vbp = 0  # Verbo presente
        vbz = 0  # Verbo tercera persona
        wdt = 0  # Wh-determiner
        wp = 0  # Pronombre de pregunta (quién, qué)
        wps = 0  # WP$ Posesivo con pronombre (cuyo)
        wrb = 0  # Wh-adverb (dónde, cuándo)
        palabras_pos = 0
        palabras_neg = 0
        acumulado_pos = 0
        acumulado_neg = 0

        # PASO 1: QUITAR URLS Y @
        url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|' + \
            '(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        url2 = '(www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|' + \
            '(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        emoticons = "['\&\-\.\/()=:;]+"
        patron = re.compile(
            '_|•|-|#|@|[¿?]|' + url + '|' + url2 + '|' + emoticons)
        entrada_m = patron.sub(' ', text)

        # PASO 2: CONVERTIR TEXTO EN MINUSCULAS
        texto = TextBlob(entrada_m)
        minusculas = texto.correct()
        minusculas = texto.lower()

        # PASO 3: ETIQUETAS TIPO DE PALABRA
        for word, pos in minusculas.tags:
            if (pos == "CC"):
                cc = cc + 1
            if (pos == "VBP"):
                vbp = vbp + 1
            if (pos == "CD"):
                cd = cd + 1
            if (pos == "DT"):
                dt = dt + 1
            if (pos == "EX"):
                ex = ex + 1
            if (pos == "FW"):
                fw = fw + 1
            if (pos == "IN"):
                inn = inn + 1
            if (pos == "JJ"):
                jj = jj + 1
            if (pos == "JJR"):
                jjr = jjr + 1
            if (pos == "JJS"):
                jjs = jjs + 1
            if (pos == "LS"):
                ls = ls + 1
            if (pos == "MD"):
                md = md + 1
            if (pos == "NN"):
                nn = nn + 1
            if (pos == "NNS"):
                nns = nns + 1
            if (pos == "NNP"):
                nnp = nnp + 1
            if (pos == "NNPS"):
                nnps = nnps + 1
            if (pos == "PDT"):
                pdt = pdt + 1
            if (pos == "POS"):
                posss = posss + 1
            if (pos == "PRP$"):
                prp = prp + 1
            if (pos == "RB"):
                rb = rb + 1
            if (pos == "RBR"):
                rbr = rbr + 1
            if (pos == "RP"):
                rp = rp + 1
            if (pos == "TO"):
                to = to + 1
            if (pos == "UH"):
                uh = uh + 1
            if (pos == "VB"):
                vb = vb + 1
            if (pos == "VBG"):
                vbd = vbd + 1
            if (pos == "VBN"):
                vbn = vbn + 1
            if (pos == "VBZ"):
                vbz = vbz + 1
            if (pos == "WDT"):
                wdt = wdt + 1
            if (pos == "WP"):
                wp = wp + 1
            if (pos == "WP$"):
                wps = wps + 1
            if (pos == "WRB"):
                wrb = wrb + 1
            verf = TextBlob(word)
            w_sent = verf.sentiment.polarity
            if (w_sent > 0):
                palabras_pos = palabras_pos + 1
                acumulado_pos = acumulado_pos + w_sent
            if (w_sent < 0):
                palabras_neg = palabras_neg + 1
                acumulado_neg = acumulado_neg + w_sent

        # PASO 4: Tokens
        palabras = minusculas.words
        cantidad = len(palabras)

        # PASO 5: A. Sentimiento
        polari = minusculas.sentiment.polarity

        # PASO 6: Cantidad de caracteres
        cant_caracteres = len(text)

        # PASO 7: Generación de fila para dataset
        ten = minusculas.lower()
        can_pala = cantidad

        return [
            ten, can_pala, cc, vbp, cd, dt, ex, fw, inn, jj, jjr, jjs, ls, md,
            nn, nns, nnp, nnps, pdt, posss, prp, rb, rbr, rp, to, uh, vb, vbd,
            vbg, vbn, vbz, wdt, wp, wps, wrb, palabras_pos, palabras_neg,
            acumulado_pos, acumulado_neg, polari, cant_caracteres
        ]

    def clean(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> FileUpload:
        # Read original dataset
        original_df = self.dataset_service.read_dataset(
            file_path=f'resources/uploads/{file_path}',
            encoding=encoding,
            delimiter=delimiter
        )

        # New dataset file name
        timestamp = int(datetime.timestamp(datetime.now()))
        file_path = f'{file_path[0: -4]}_{timestamp}.csv'

        # Open file in write mode
        file = open(
            f'resources/cleaned/{file_path}', 'w', encoding=encoding)

        # Header columns generation
        headers = f'sentiment|answer|{"|".join(GENERATED_COLUMNS)}\n'

        # Write header columns
        file.write(headers)

        # Get the columns that contains answers
        answers = original_df.columns[3:].to_list()

        # Iterate answers to generate the new dataset
        for answer in answers:
            sentiment = original_df[answer].value_counts().idxmax()
            clean_row = self.generate_clean_row(answer)
            row = f'{sentiment}|{clean_row}'
            file.write(row)
            synonym_answer = self.phrase_replacer_service.extract_synonyms(
                answer
            )
            logger.info(f'ANSWER: {answer}')
            logger.info(f'SYNONYM: {synonym_answer}')

        # End file writting
        file.close()

        return FileUpload(file_path=file_path)
