"""
__author__: Jiaming Shen
__description__: Parse case study file (obtrained from test_fast.py) to SemEval required format
"""
import argparse

def parse_string(s):
    return s.split("||")[1].split("@@@")[0]

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Parse to SemEval format')
    args.add_argument('--input', type=str, help='input file path')
    args.add_argument('--output', type=str, help='output file path')
    args = args.parse_args()

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for lid, line in enumerate(fin):
            if lid == 0:  # skip header
                continue
            line = line.strip()
            if line:
                segs = line.split("\t")
                test_id = parse_string(segs[0])[len("test."):]
                predict_id = parse_string(segs[2].split(", ")[0])
                lemma, pos, tmp_id = predict_id.split(".")
                tmp_id = str(int(tmp_id))
                predict_id = "#".join([lemma, pos, tmp_id])
                fout.write("\t".join([test_id, predict_id, "attach", "\n"]))
