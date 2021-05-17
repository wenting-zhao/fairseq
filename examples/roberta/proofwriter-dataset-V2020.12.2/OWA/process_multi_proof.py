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
            q = item['questions'][ques]['question']
            if item['questions'][ques]['QDep'] != 2: continue
            ans = item['questions'][ques]['answer']
            if ans == True: a = '1'
            elif ans == False: a = '0'
            elif ans == 'Unknown': a = '2'
            else: sys.exit("something went wrong in the label: %s" % ans)
            ret_inputs.append(context+' <sep> '+q+' <sep>')
            ret_labels.append(a)
    return ret_inputs, ret_labels

def main():
    directory1 = sys.argv[1]
    directory2 = sys.argv[2]
    directory3 = sys.argv[3]
    #directory4 = sys.argv[4]
    outdir = sys.argv[4]
    for split in ['train', 'dev', 'test']:
        inputs1, labels1 = process_data("%s/meta-%s.jsonl" % (directory1, split))
        inputs2, labels2 = process_data("%s/meta-%s.jsonl" % (directory2, split))
        inputs3, labels3 = process_data("%s/meta-%s.jsonl" % (directory3, split))
        #inputs4, labels4 = process_data("%s/meta-%s.jsonl" % (directory4, split))
        #inputs = inputs1 + inputs2 + inputs3 + inputs4
        #labels = labels1 + labels2 + labels3 + labels4
        inputs = inputs1 + inputs2 + inputs3
        labels = labels1 + labels2 + labels3
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

