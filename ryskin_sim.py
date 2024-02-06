import pandas as pd
import tqdm

import shallow as sh
import lm

STIMULI_PATH = "ryskin_stimuli.csv"
EDIT_MATRIX_PATH = "/Users/canjo/data/subtlex/en_editmatrix10000_plus.csv"

def simulate(filename=STIMULI_PATH, edit_matrix_path=EDIT_MATRIX_PATH, lnps=None):
    stimuli = pd.read_csv(filename, index_col=0)
    distortion = pd.read_csv(edit_matrix_path, index_col=0)
    vocab = list(distortion.columns)
    stimuli['context'] = stimuli['sentence'].map(lambda s: " ".join(s.split()[:-1]))
    stimuli['ok'] = stimuli['target'].map(lambda x: x in vocab)
    
    # sometimes there is more than one context for item, if there is variation in a/an.
    # identify these items and mark them as bad
    num_contexts_per_item = stimuli[['item', 'context']].groupby('item').aggregate(lambda x: len(set(x))).reset_index()
    num_contexts_per_item.columns = "item num_contexts".split()
    stimuli = pd.merge(stimuli, num_contexts_per_item)
    stimuli['ok'] = stimuli['ok'] & ~(stimuli['num_contexts'] > 1)

    good_stimuli = stimuli[stimuli['ok']]
    contexts = good_stimuli[['item', 'context']].drop_duplicates()

    if lnps is None:
        lnps = {
            item : lm.conditional_logp(context, vocab)
            for _, (item, context) in tqdm.tqdm(contexts.iterrows(), total=len(contexts))
        }

    def gen():
        for i, row in tqdm.tqdm(good_stimuli.iterrows(), total=len(good_stimuli)):
            item = row['item']
            target = row['target']
            df = sh.timecourse(lnps[item], distortion[target], vocab=vocab)
            df['_item'] = item
            df['_target'] = target
            df['_context'] = row['context']
            df['_condition'] = row['condition']
            yield df

    df = pd.concat(list(gen()))
    return df
    
    
