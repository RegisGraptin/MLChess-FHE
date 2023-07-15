
from converter.pgn_data import PGNData

if __name__ == "__main__":
    FILE_NAME = "large.pgn"

    pgn_data = PGNData(FILE_NAME)
    pgn_data.export()