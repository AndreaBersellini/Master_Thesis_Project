import pickle as pkl

class QTable():
    def __init__(self, path : str):
        self._path : str = path
        self._q_table : dict = {}

    def __call__(self) -> dict:
        return self._q_table

    def save(self) -> None:
        with open(self._path + '.pkl', 'wb') as file:
            pkl.dump(self._q_table, file)

    def load(self) -> bool:
        try:
            with open(self._path + '.pkl', 'rb') as file:
                self._q_table = pkl.load(file)
                return True
        except:
            return False
        
    def __len__(self) -> int:
        return len(self._q_table)