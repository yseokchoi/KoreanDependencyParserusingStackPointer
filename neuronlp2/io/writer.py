__author__ = 'max'


class CoNLL03Writer(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, chunk, predictions, targets, lengths):
        batch_size, _ = word.shape
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__word_alphabet.get_instance(word[i, j]).encode('utf-8')
                p = self.__pos_alphabet.get_instance(pos[i, j]).encode('utf-8')
                ch = self.__chunk_alphabet.get_instance(chunk[i, j]).encode('utf-8')
                tgt = self.__ner_alphabet.get_instance(targets[i, j]).encode('utf-8')
                pred = self.__ner_alphabet.get_instance(predictions[i, j]).encode('utf-8')
                self.__source_file.write('%d %s %s %s %s %s\n' % (j + 1, w, p, ch, tgt, pred))
            self.__source_file.write('\n')


class CoNLLXWriter(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w', encoding='utf-8')

    def close(self):
        self.__source_file.close()

    def write(self, sentences, word, pos, head, type, lengths, symbolic_root=False, symbolic_end=False):
        batch_size, _, lemma_length = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
                if len(sentences[i]) == j-start:
                    break

                t = self.__type_alphabet.get_instance(type[i, j])
                h = head[i, j]

                self.__source_file.write('%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n' % (j, sentences[i][j-start][1], sentences[i][j-start][2], sentences[i][j-start][3], \
                                                                                       sentences[i][j-start][4], sentences[i][j-start][5], h, sentences[i][j-start][7], t, sentences[i][j-start][9]))
            self.__source_file.write('\n')
