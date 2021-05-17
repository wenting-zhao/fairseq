import json
import sys
import os


def process_data(f):
    content = open(f, 'r')
    result = [json.loads(jline) for jline in content.read().splitlines()]
    ret_inputs = []
    ret_labels = []
    for i, item in enumerate(result):
        context = item['theory']
        for ques in item['questions'].keys():
            if item['questions'][ques]['QDep'] != 3: continue
            try:
                tmp = item['questions'][ques]['proofsWithIntermediates']
            except KeyError:
                continue
            if len(tmp) != 1: print("more than one proof tree"); continue
            for inte in tmp[0]['intermediates']:
                q = tmp[0]['intermediates'][inte]['text']
                ret_inputs.append(context+' <sep> '+q+' <sep>')
                ret_labels.append('1')
    return ret_inputs, ret_labels

def main():
    directory1 = sys.argv[1]
    directory2 = sys.argv[2]
    outdir = sys.argv[3]
    for split in ['train', 'dev', 'test']:
        inputs1, labels1 = process_data("%s/meta-%s.jsonl" % (directory1, split))
        inputs2, labels2 = process_data("%s/meta-%s.jsonl" % (directory2, split))
        inputs = inputs1 + inputs2
        labels = labels1 + labels2
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        f = open("%s/%s.input0" % (outdir, split), "w")
        f2 = open("%s/%s.label" % (outdir, split), "w")
        for text in inputs: f.write(text+'\n')
        for label in labels: f2.write(label+'\n')
        f.close()
        f2.close()

if __name__ == "__main__":
    main()

