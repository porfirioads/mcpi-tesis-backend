from datetime import datetime
from fastapi import UploadFile, HTTPException
from app.utils.singleton import SingletonMeta
from app.utils.strings import Strings
import csv


class DatasetService(metaclass=SingletonMeta):
    def upload_dataset(self, file: UploadFile) -> str:
        if file.content_type != 'text/csv':
            raise HTTPException(
                status_code=400, detail='DATASET_MUST_BE_A_CSV')

        timestamp = int(datetime.timestamp(datetime.now()))
        snack_name = Strings.to_snack_case(file.filename)
        file_path = f'uploads/{timestamp}_{snack_name}'

        with open(file_path, 'wb') as f:
            f.write(file.file.read())

        return file_path

    def read_csv(self, file_path: str, encoding: str, delimiter: str) -> str:
        with open(file_path, encoding=encoding) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
            print(csv_reader.fieldnames)
            print(csv_reader.dialect)

            for row in csv_reader:
                print(csv_reader.line_num)

