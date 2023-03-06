import logging
import tempfile
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path

import networkx as nx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from flask import Flask, request

from ud_boxer.sbn import SBNGraph
from ud_boxer.sbn_spec import SBNError

app = Flask(__name__)
HOST = "0.0.0.0"
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

    logging.debug(f"got this text: {text}")

    try:
        with tempfile.NamedTemporaryFile(mode='w+t') as tmp:
            tmp.write(f"{text}\tDUMMY\n")
            tmp.flush()
            for d in PREDICTOR._dataset_reader.read(tmp.name):
                pred = PREDICTOR.predict_instance(d)['predicted_tokens']
                sbn = SBNGraph().from_string(" ".join(pred))

        graph = sbn.to_pydot()
        networkx_graph = nx.nx_pydot.from_pydot(graph)

        json_graph = nx.cytoscape_data(networkx_graph)
        ret_value["result"]["graph"] = json_graph

    except SBNError:  # as e:
        print('SBNError for input:', pred)
        ret_value["result"]["graph"] = None
        ret_value["result"]["errors"] = traceback.format_exc()

    except Exception:  # as e:
        ret_value["result"]["errors"] = traceback.format_exc()
        return ret_value

    return ret_value


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    global PREDICTOR

    print('loading model archive...')
    arch = load_archive("model.tar.gz")

    print('initializing predictor...')
    PREDICTOR = Predictor.from_archive(arch, predictor_name="seq2seq")

    print('running app...')
    app.run(host=HOST, port=PORT, debug=False)
