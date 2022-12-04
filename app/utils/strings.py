from re import sub


class Strings:
    def to_snack_case(string: str) -> str:
        snack = '_'.join(
            sub(
                '([A-Z][a-z]+)', r' \1',
                sub(
                    '([A-Z]+)', r' \1',
                    string.replace('-', ' ')
                )
            ).split()
        ).lower()
        snack = sub('_+', '_', snack)
        return snack
