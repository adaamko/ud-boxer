import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import networkx as nx
from flask import Flask, request
from tqdm.contrib.logging import logging_redirect_tqdm

from ud_boxer.config import Config
from ud_boxer.grew_rewrite import Grew
from ud_boxer.ud import UDGraph, UDParser

app = Flask(__name__)
HOST = "localhost"
PORT = 5002
OUTPUT_DIR = "./results"

@app.route("/parse", methods=["POST"])
def parse():
    global PARSER
    global GREW

    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(exist_ok=True)

    data = request.get_json()
    ret_value = {"result": {"errors": None, "graph": None}}

    text = data["text"]

    if len(data) == 0 or text is None:
        ret_value["result"]["errors"] = "No text provided"
        return ret_value

    try:
        ud_filepath = Path(
            output_dir / f"{LANGUAGE}.ud.stanza.conll"
        )
        PARSER.parse(text, ud_filepath)
        res = GREW.run(ud_filepath)
        #res.to_sbn(OUTPUT_DIR / f"{LANGUAGE}.drs.sbn")
        graph = res.to_pydot()
        networkx_graph = nx.nx_pydot.from_pydot(graph)

        json_graph = nx.cytoscape_data(networkx_graph)
        ret_value["result"]["graph"] = json_graph

    except Exception as e:
        ret_value["result"]["errors"] = str(e)
        return ret_value

    return ret_value


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--language",
        default="en",
        type=str,
        help="Language to use for UD pipelines.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    global PARSER
    global LANGUAGE

    LANGUAGE = args.language
    PARSER = UDParser(language=args.language)

    global GREW
    GREW = Grew(language=args.language)

    app.run(host=HOST, port=PORT, debug=True)
