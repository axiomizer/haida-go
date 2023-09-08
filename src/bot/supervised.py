import os
import tarfile


KGS_GAMES_PATH = os.path.join('src', 'bot', 'kgs')


class SGF:
    """
    Collection = GameTree { GameTree }
    GameTree   = "(" Sequence { GameTree } ")"
    Sequence   = Node { Node }
    Node       = ";" { Property }
    Property   = PropIdent PropValue { PropValue }
    PropIdent  = UcLetter { UcLetter }
    PropValue  = "[" CValueType "]"
    CValueType = (ValueType | Compose)
    ValueType  = (None | Number | Real | Double | Color | SimpleText |
                  Text | Point  | Move | Stone)
    """

    class Node:
        def __init__(self, string):
            self.properties = {}
            end_ind = -1
            ind = string.index('[')
            while ind != -1:
                prop_ident = string[end_ind+1:ind].strip()
                prop_values = set()
                end_ind = ind - 1
                while ind != -1 and string[end_ind+1:ind].strip() == '':
                    end_ind = string.index(']', ind)
                    prop_values.add(string[ind+1:end_ind].strip())
                    ind = string.find('[', end_ind)
                self.properties[prop_ident] = prop_values

    def __init__(self, string):
        sequence = self.__main_branch(string)
        nodes = sequence.split(';')[1:]
        self.root = SGF.Node(nodes[0])
        self.moves = [SGF.Node(n) for n in nodes[1:]]
        # TODO: make sure nodes are actually move nodes
        #       only extract the move and the player properties

    def __main_branch(self, data):
        depth = 0
        start = data.find('(')
        if start == -1:
            return data
        for i in range(start, len(data)):
            if data[i] == '(':
                depth += 1
            elif data[i] == ')':
                depth -= 1
            if depth == 0:
                return data[:start] + self.__main_branch(data[start+1:i])
        raise ValueError('Unbalanced parentheses')


def train():
    # get kgs games
    archive = os.path.join(KGS_GAMES_PATH, os.listdir(KGS_GAMES_PATH)[0])
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            if member.isdir():
                continue
            f = tf.extractfile(member)
            sgf = SGF(f.read().decode('UTF-8'))
            print(sgf.root.properties)
            print(sgf.moves[4].properties)
            input('continue?')
            # TODO: continue implementing from here
