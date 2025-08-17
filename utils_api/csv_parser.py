import csv, io

def parse_csv(file_bytes):
    decoded = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(decoded))
    return list(reader)

class CSVParser:
    def parse_csv_from_bytes(self, file_bytes: bytes):
        return parse_csv(file_bytes)

    def convert_to_text(self, rows) -> str:
        return "\n".join([", ".join(row) for row in rows])
