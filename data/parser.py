
from converter.pgn_data import PGNData

if __name__ == "__main__":
    FILE_NAME = "lichess_db_standard_rated_2023-01.pgn"

    pgn_data = PGNData(FILE_NAME)
    pgn_data.export()