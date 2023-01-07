import re
from typing import List
from app.services.dataset_service import DatasetService
from app.utils.singleton import SingletonMeta
from textblob import TextBlob
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing
from scipy.stats import zscore

HEADERS = [
    'sentiment', 'answer', 'length', 'cc', 'vbp', 'cd', 'dt', 'ex', 'fw',
    'inn', 'jj', 'jjr', 'jjs',
    # 'ls',
    'md', 'nn', 'nns',
    # 'nnp', 'nnps',
    'pdt',
    # 'posss',
    'prp', 'rb', 'rbr', 'rp', 'to',
    # 'uh',
    'vb', 'vbd',
    # 'vbg',
    'vbn', 'vbz', 'wdt', 'wp',
    # 'wps',
    'wrb', 'pos_length', 'neg_length', 'pos_acumulado', 'neg_acumulado',
    'polarity', 'cant_caracteres'
]


class CleaningService(metaclass=SingletonMeta):
    dataset_service = DatasetService()

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
        # ls = 0  # Marcador de lista
        md = 0  # Modal (podría, podrá)
        nn = 0  # Sustantivo singular
        nns = 0  # Sustantivo plural
        # nnp = 0  # Sustantivo propio
        # nnps = 0  # Sustantivo propio plural
        pdt = 0  # Predeterminado
        # posss = 0  # Posesivo
        prp = 0  # Pronombre personal
        rb = 0  # RB Adverbio
        rbr = 0  # Adverbio comparativo
        rp = 0  # Particular
        to = 0  # Particular de infinitivo
        # uh = 0  # Interjección
        vb = 0  # Verbo
        vbd = 0  # VBD ?
        # vbg = 0  # VBG ?
        vbn = 0  # Verbo participio
        vbp = 0  # Verbo presente
        vbz = 0  # Verbo tercera persona
        wdt = 0  # Wh-determiner
        wp = 0  # Pronombre de pregunta (quién, qué)
        # wps = 0  # WP$ Posesivo con pronombre (cuyo)
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
            # if (pos == "LS"):
            #     ls = ls + 1
            if (pos == "MD"):
                md = md + 1
            if (pos == "NN"):
                nn = nn + 1
            if (pos == "NNS"):
                nns = nns + 1
            # if (pos == "NNP"):
            #     nnp = nnp + 1
            # if (pos == "NNPS"):
            #     nnps = nnps + 1
            if (pos == "PDT"):
                pdt = pdt + 1
            # if (pos == "POS"):
            #     posss = posss + 1
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
            # if (pos == "UH"):
            #     uh = uh + 1
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
            # if (pos == "WP$"):
            #     wps = wps + 1
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
        ten = f'{minusculas.lower()}'
        can_pala = cantidad

        return [
            ten, can_pala, cc, vbp, cd, dt, ex, fw, inn, jj, jjr, jjs,
            # ls,
            md, nn, nns,
            # nnp, nnps,
            pdt,
            # posss,
            prp, rb, rbr, rp, to,
            # uh,
            vb, vbd,
            # vbg,
            vbn, vbz, wdt, wp,
            # wps,
            wrb, palabras_pos, palabras_neg,
            acumulado_pos, acumulado_neg, polari, cant_caracteres
        ]

    def clean(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        # Read original dataset
        original_df = self.dataset_service.read_dataset(
            file_path=f'resources/uploads/{file_path}',
            encoding=encoding,
            delimiter=delimiter
        )

        # Get the columns that contains answers
        answers = original_df.columns[3:].to_list()

        # Variable where rows will be stored
        data = []

        # Iterate answers to generate the new dataset
        for answer in answers:
            sentiment = original_df[answer].value_counts().idxmax()
            clean_data = self.calculate_data_fields(answer)
            data.append([sentiment] + clean_data)

        new_df = pd.DataFrame(
            data,
            columns=HEADERS
        )

        # Negative and neutral answers are now considered as negative
        new_df = self.join_categories(
            df=new_df,
            source_column='sentiment',
            categories_to_join=['Neutral', 'Negativo'],
            target_category='Negativo'
        )

        return new_df

    def extract_features(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        df = self.dataset_service.read_dataset(
            file_path=f'resources/cleaned/{file_path}',
            encoding=encoding,
            delimiter=delimiter
        )

        sentiment = df.sentiment
        answer = df.answer
        df = df.drop(columns=['sentiment', 'answer'])
        df = df.apply(zscore)

        # Correlation metrics
        label_encoder = preprocessing.LabelEncoder()
        df.iloc[:, df.shape[1] - 1] = label_encoder.fit_transform(
            df.iloc[:, 0]
        ).astype('float64')
        corr = df.corr()

        # Next, we compare the correlation between features
        # and remove one of two features that have a correlation higher than
        # 0.9
        columns = np.full((corr.shape[0],), True, dtype=bool)

        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= 0.9:
                    if columns[j]:
                        columns[j] = False

        selected_columns = df.columns[columns]
        df = df[selected_columns]

        # pvalue feature selection
        selected_columns = selected_columns[1:].values

        def backward_elimination(x, Y, sl, columns):
            numVars = len(x[0])
            for i in range(0, numVars):
                regressor_OLS = sm.OLS(Y, x).fit()
                maxVar = max(regressor_OLS.pvalues).astype(float)
                if maxVar > sl:
                    for j in range(0, numVars - i):
                        if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                            x = np.delete(x, j, 1)
                            columns = np.delete(columns, j)
            regressor_OLS.summary()
            return x, columns

        data_modeled, selected_columns = backward_elimination(
            df.iloc[:, 1:].values,
            df.iloc[:, 0].values,
            0.05,
            selected_columns
        )

        # Creating a Dataframe with the columns selected using the p-value
        # and correlation
        fs_data = pd.DataFrame(data=data_modeled, columns=selected_columns)
        fs_data['sentiment'] = sentiment
        fs_data['answer'] = answer
        return fs_data

    def extract_best_features(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        df = self.dataset_service.read_dataset(
            file_path=f'resources/cleaned/{file_path}',
            encoding=encoding,
            delimiter=delimiter
        )

        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2

        x = df.drop(columns=['sentiment', 'answer'])
        y = df[['sentiment']]

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(x)
        x = scaler.transform(x)

        # N features with highest chi-squared statistics are selected
        chi2_selector = SelectKBest(chi2, k=13)
        X = chi2_selector.fit_transform(x, y)
        mask = chi2_selector.get_support(indices=True)
        k_best_features = df.columns[mask]

        selected_columns = ['sentiment', 'answer'] + k_best_features.to_list()
        fs_data = df[selected_columns]
        return fs_data

        # sentiment = df.sentiment
        # answer = df.answer
        # df = df.drop(columns=['sentiment', 'answer'])
        # df = df.apply(zscore)

        # # Correlation metrics
        # label_encoder = preprocessing.LabelEncoder()
        # df.iloc[:, df.shape[1] - 1] = label_encoder.fit_transform(
        #     df.iloc[:, 0]
        # ).astype('float64')
        # corr = df.corr()

        # # Next, we compare the correlation between features
        # # and remove one of two features that have a correlation higher than
        # # 0.9
        # columns = np.full((corr.shape[0],), True, dtype=bool)

        # for i in range(corr.shape[0]):
        #     for j in range(i + 1, corr.shape[0]):
        #         if corr.iloc[i, j] >= 0.9:
        #             if columns[j]:
        #                 columns[j] = False

        # selected_columns = df.columns[columns]
        # df = df[selected_columns]

        # # pvalue feature selection
        # selected_columns = selected_columns[1:].values

        # def backward_elimination(x, Y, sl, columns):
        #     numVars = len(x[0])
        #     for i in range(0, numVars):
        #         regressor_OLS = sm.OLS(Y, x).fit()
        #         maxVar = max(regressor_OLS.pvalues).astype(float)
        #         if maxVar > sl:
        #             for j in range(0, numVars - i):
        #                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
        #                     x = np.delete(x, j, 1)
        #                     columns = np.delete(columns, j)
        #     regressor_OLS.summary()
        #     return x, columns

        # data_modeled, selected_columns = backward_elimination(
        #     df.iloc[:, 1:].values,
        #     df.iloc[:, 0].values,
        #     0.05,
        #     selected_columns
        # )

        # # Creating a Dataframe with the columns selected using the p-value
        # # and correlation
        # fs_data = pd.DataFrame(data=data_modeled, columns=selected_columns)
        # fs_data['sentiment'] = sentiment
        # fs_data['answer'] = answer
        # return fs_data

    def join_categories(
        self,
        df: pd.DataFrame,
        source_column: str,
        categories_to_join: List[str],
        target_category: str
    ) -> pd.DataFrame:
        replacements = {}

        for category in categories_to_join:
            replacements[category] = target_category

        df[source_column] = df[source_column].replace(replacements)

        return df
