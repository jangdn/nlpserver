import argparse
import copy, json, os

import torch
import nltk

from torchtext import data
from models.model import BiDAF
from models.data import SQuAD

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

def model_initialize(args, squad):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = BiDAF(args, squad.WORD.vocab.vectors).to(device)

    print('loading pretrained model...')
    model.load_state_dict(torch.load(args.model_path))

    model.eval()

    print('loading complete')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--dev-file', default='dev-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--model_path', default='saved_models/BiDAF_sample.pt')

    args = parser.parse_args()

    setattr(args, 'only_build_vocab', True)
    print('Build Vocab with SQUAD...')
    squad = SQuAD(args)
    setattr(args, 'char_vocab_size', len(squad.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(squad.WORD.vocab))


    context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."

    query = "Which NFL team represented the AFC at Super Bowl 50?"

    # context, query preprocessing
    input = squad.preprocess_input(args, context, query)

    # Model Initialization
    model = model_initialize(args, squad)

    # Model 실행
    q1, q2 = model(input)

    print('q1 : ', q1, '\nq2 : ', q2)
