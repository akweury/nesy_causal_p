# Created by jing at 01.12.24
class Group():
    def __init__(self, id, name, input_signal, memory_signal,parents, conf):
        self.id = id
        self.name = name
        self.input = input_signal
        self.memory = memory_signal
        self.parents = parents
        self.conf = conf

    def __str__(self):
        # return self.name
        return self.name + "_" + str(self.id)

    def __hash__(self):
        return hash(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if type(other) == Group:
            return self.name == other.name
        else:
            return False

    def __lt__(self, other):
        return self.__str__() < other.__str__()
