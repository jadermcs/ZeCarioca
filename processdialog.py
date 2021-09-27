import json
import tqdm
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel

random.seed(42)
path = "models/adrenaline_multiwoz/"
checkpoint = path + "epoch56_trloss0.40_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint)

with open("data/creditos_placa_errada_completo.json") as fin:
    data = json.load(fin)
    tokens = data['ontology']['intents'] + data['ontology']['actions']
    tokenizer.add_special_tokens({'additional_special_tokens': tokens})
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(path+"connectcar_tokens")
    model.save_pretrained(path+"connectcar_tokens")
    dialogues = []
    for d in tqdm.tqdm(data['dialogs']):
        dialog = ''
        for turn in range(len(d['turns'][1:])//2):
            t = d['turns'][turn*2+1:turn*2+3]
            utterance = f"<sos_u> {t[0]['utterance']} <oes_u>"
            intents = []
            for slot in t[0]['slot-values']:
                if isinstance(t[0]['slot-values'][slot], list):
                    parse = [[slot, v] for v in t[0]['slot-values'][slot]]
                    intents += [item for sublist in parse for item in sublist]
                else:
                    intents += [slot, t[0]['slot-values'][slot]]
            bs = ["<sos_b>"] + [t[0]['intent']] + intents + ["<eos_b>"]
            belief = " ".join(bs)
            a = ["<sos_a>"] + [t[1]['action']] + ["<eos_a>"]
            action = " ".join(a)
            response = f"<sos_r> {t[1]['utterance_delex']} <oes_r>"
            dialog += utterance+belief+action+response
        dialogues.append({'id':d['id'], 'text':dialog})
    random.shuffle(dialogues)
    f1 = open("data/process.train.json", "w")
    f2 = open("data/process.valid.json", "w")
    for i, line in enumerate(dialogues):
        if i <= .7*len(dialogues):
            print(json.dumps(line), file=f1)
        else:
            print(json.dumps(line), file=f2)
    f1.close()
    f2.close()
