from pathlib import Path

class ResultManager:

    def __init__(self, root):
        self.root = Path(root)

    def file(self, *parts):
        path = self.root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

