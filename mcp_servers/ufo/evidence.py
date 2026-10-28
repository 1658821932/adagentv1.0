"""CPU image worker for target evidence, without Agent imaging dependencies."""
import json
from pathlib import Path
import sys

from tools.tool05_instance_discovery.grounded import anchored_frame_card


if __name__ == '__main__':
    value = json.load(sys.stdin)
    card = anchored_frame_card(Path(value['source']), value['instance'], value['frame'], Path(value['output']))
    print(json.dumps(card))
