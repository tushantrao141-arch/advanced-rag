import os


class Utils:

    @staticmethod
    def check_dir(directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)