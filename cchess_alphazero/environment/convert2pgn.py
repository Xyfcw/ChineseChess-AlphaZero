import codecs
import os
from env import CChessEnv
from data_helper import get_game_data_filenames, read_game_data_from_file
from lookup_tables import flip_move, Winner
from static_env import parse_onegreen_move

def load_data_from_file(filename):
    try:
        data = read_game_data_from_file(filename)
    except Exception as e:
        print ("Error when loading data ", e)
        return None
    if data is None:
        return None
    return data

def convert_to_record(data):
    cenv = CChessEnv()
    cenv.reset()

    action = None

    for item in data[1:]:
        action = item[0]
        if not cenv.red_to_move:
            action = flip_move(action)
        cenv.step(action)

    return cenv.board.record, cenv.winner

def make_pgn(record, result):
    head = '[Game "Chinese Chess"]\n'
    red = '[Red "AlphaZero"]\n'
    black = '[Black "AlphaZero"]\n'
    result = '1-0' if result == Winner.red else '0-1'
    res_line = '[Result "%s"]\n' % result
    fen = '[FEN "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR r - - 0 1"]\n'
    pgn = head + red + black + res_line + fen + record + '\n' + result
    return pgn

def save_file(data, filename):
    with codecs.open(filename, "a", encoding="utf-8") as f:
        f.write(data)

def main():
    files = get_game_data_filenames('/path/to/data/play_record')
    for file in files:
        print(file)
        data = load_data_from_file(file)
        if data is None:
            os.remove(file)
            continue
        record, result = convert_to_record(data)
        pgn = make_pgn(record, result)
        filename = file.split('/')[-1][:-5] + '.pgn'
        path = os.path.join('/path/to/data/pgn', filename)
        save_file(pgn, path)

if __name__ == '__main__':
    main()
