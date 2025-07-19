from loguru import logger

def print_logo(color: str = "red"):
    color_start = "<" + color + ">"
    color_end = "</" + color + ">"
    logger.opt(colors=True).info(rf"""{color_start}
    __  __          __                          
   / / / /_  ______/ /________  ____ ____  ____ 
  / /_/ / / / / __  / ___/ __ \/ __ `/ _ \/ __ \ 
 / __  / /_/ / /_/ / /  / /_/ / /_/ /  __/ / / / 
/_/ /_/\__, /\__,_/_/   \____/\__, /\___/_/ /_/ 
      /____/                 /____/             
      {color_end}""")

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

if __name__ == "__main__":
    print_logo()