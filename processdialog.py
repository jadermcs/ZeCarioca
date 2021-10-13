import json
import tqdm
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel

random.seed(42)

with open("synthetic.augmented.json") as fin:
    data = json.load(fin)
    tokens = data['ontology']['intents'] + data['ontology']['actions'] + ["<sos_u>", "<sos_b>", "<sos_a>", "<sos_r>", "<eos_u>", "<eos_b>", "<eos_a>", "<eos_r>"]
    dialogues = []
    for d in tqdm.tqdm(data['dialogs']):
        dialog = ''
        for turn in range(len(d['turns'])//2):
            t = d['turns'][turn*2:turn*2+2]
            utterance = f"<sos_u> {t[0]['utterance'].lower()} <eos_u>"
            intents = []
            for slot in t[0]['slot-values']:
                if isinstance(t[0]['slot-values'][slot], list):
                    parse = [[slot, v] for v in t[0]['slot-values'][slot]]
                    intents += [item for sublist in parse for item in sublist]
                else:
                    intents += [slot, t[0]['slot-values'][slot]]
            bs = [t[0]['intent']] + intents
            belief = "<sos_b> " + " ".join(bs).lower() + " <eos_b>"
            action = "<sos_a> " + t[1]['action'] + " <eos_a>"
            response = f"<sos_r> {t[1]['utterance_delex']} <eos_r>"
            dialog += utterance+belief+action+response
        dialogues.append({'id':d['id'], 'text':dialog})
    random.shuffle(dialogues)
    f1 = open("data/process.train.json", "w")
    f2 = open("data/process.valid.json", "w")
    f3 = open("data/ontology.json", "w")
    json.dump(tokens, f3)
    c1, c2 = 0, 0
    for i, line in enumerate(dialogues):
        if not line['id'].endswith(("1", "2", "3")):
            print(json.dumps(line), file=f1)
            c1 +=1
        else:
            print(json.dumps(line), file=f2)
            c2 +=1
    print("train size:", c1, "test size:", c2)
    f1.close()
    f2.close()
