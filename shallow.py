import numpy as np
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt

def timecourse(lnp0, d, max_lam=10, num_steps=100, vocab=None):
    """
    Give the predicted processing timecourse given K alternative interpretations for a given input, where
    
    * lnp is an array of shape K giving prior log-probabilities for the alternatives.
    * d is an array of shape K giving distortions for each of the K interpretations.

    Optionally provide an iterable of vocabulary items corresponding to entries in lnp0 and d.
    """
    lam = np.linspace(0, max_lam, num_steps) # shape L
    unnormalized = lnp0[None, :] - lam[:, None]*d[None, :]  # shape LK
    lnZ = scipy.special.logsumexp(unnormalized, -1) # shape L
    lnp = unnormalized - lnZ[:, None] # shape LK
    p = np.exp(lnp) # shape LK
    df = pd.DataFrame(p) # dataframe containing probabilities for each interpretation at each time
    if vocab is not None:
        df.columns = list(vocab)
    df['processing_time'] = lam
    df['expected_distortion'] = p @ d
    df['variance_distortion'] = p @ d**2 - (p @ d)**2
    df['kl_div'] = -lnZ - lam*df['expected_distortion']
    df['d_kl_div'] = lam * df['variance_distortion']
    return df

def hunted_example(comma_distortion=1, word_distortion=1, t_old=5, t_new=5):
    alternatives = [
        "While the hunters hunted the deer ran",
        "While the hunters hunted, the deer ran",
        "While the hunters hunted the deer they",
        "While the hunters hunted, the deer they"
    ]
    
    # from GPT-2, modified so that the third alternative isn't incredibly probable:
    lnp0 = np.array([-6.7500e+00, -6.1137e+00, -5.1715e-00, -6.3339e+00])
    
    # made up distortions:
    d_c = np.array([0, comma_distortion, 0, comma_distortion])
    d_x = np.array([0, 0, word_distortion, word_distortion])
    d_cx = d_c + d_x
    d_x_given_c = d_cx - d_c
    
    old_timecourse = timecourse(lnp0, d_c, max_lam=t_old)
    old_timecourse['phase'] = 0
    # get last distribution to use as the new default policy
    lnp_c = np.log(old_timecourse[old_timecourse['processing_time'] == old_timecourse['processing_time'].max()][range(4)].to_numpy().squeeze(0))
    new_timecourse = timecourse(lnp_c, d_cx, max_lam=t_new)
    new_timecourse['phase'] = 1
    new_timecourse['processing_time'] = new_timecourse['processing_time'] + t_old
    return pd.concat([old_timecourse, new_timecourse])

def moses_example(form_weight=0, sem_weight=1):
    import lm
    full_d_form = pd.read_csv("/Users/canjo/data/subtlex/en_editmatrix10000_plus.csv", index_col=0)
    full_vocab = set(full_d_form.columns[1:])
    full_vocab.add('Noah')
    full_vocab.add('Moses')
    vocab, d_sem = lm.cosine_distance_matrix(full_vocab)
    lower_vocab = list(map(str.lower, vocab))
    d_form = full_d_form.T[lower_vocab].T[lower_vocab]
    moses = form_weight*np.array(d_form['moses']) + sem_weight*d_sem[vocab.index('Moses')].numpy()
    lnp_bible = lm.conditional_logp_single_token("In the Bible, how many animals of each kind did", vocab)
    df_moses = timecourse(lnp_bible, moses)
    df_moses.columns = vocab + ['processing_time', 'expected_distortion', 'variance_distortion', 'kl_div', 'd_kl_div']
    return df_moses

def example_timecourses(T=2200, scale=5, prior=.1, distractor=.7, farthest=10):
    """ Nice-looking example timecourses """
    max_lam = 15
    num_steps = 1000
    t = np.linspace(0, T, num_steps)

    lnp0 = np.log([prior, (1-prior)*.5, (1-prior)*.5])
    d = np.array([0, farthest, farthest])
    df_n400 = timecourse(lnp0, d, max_lam=max_lam, num_steps=num_steps)
    df_n400['scenario'] = 'n400'

    d = np.array([0, distractor, farthest])
    df_p600 = timecourse(lnp0, d, max_lam=max_lam, num_steps=num_steps)
    df_p600['scenario'] = 'p600'

    lnp0 = np.log([prior, (1-prior)*.15, (1-prior)*(1-.15)])
    df_biphasic = timecourse(lnp0, d, max_lam=max_lam, num_steps=num_steps)
    df_biphasic['scenario'] = 'biphasic'

    df = pd.concat([df_n400, df_p600, df_biphasic])
    df['eeg'] = -df['d_kl_div']*np.sin(2*np.pi*df['t']/scale)

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    

    # plot kl divergences
    axs[0,0].set_title("Cumulative Effort")
    axs[0,0].set_ylabel("KL Divergence, D(t)")
    axs[0,0].set_xlabel("Time t (ms)")
    axs[0,0].plot(t, df[df['scenario'] == 'n400']['kl_div'], label="N400")
    axs[0,0].plot(t, df[df['scenario'] == 'p600']['kl_div'], label="P600")
    axs[0,0].plot(t, df[df['scenario'] == 'biphasic']['kl_div'], label="biphasic")
    axs[0,0].legend()

    axs[0,1].set_title("Instantaneous Effort")
    axs[0,1].set_xlabel("Time t (ms)")
    axs[0,1].set_ylabel("KL Divergence Rate, d/dt D(t)")
    axs[0,1].set_xlim(0,T/2)
    axs[0,1].plot(t, df[df['scenario'] == 'n400']['d_kl_div'], label="N400")
    axs[0,1].plot(t, df[df['scenario'] == 'p600']['d_kl_div'], label="P600")
    axs[0,1].plot(t, df[df['scenario'] == 'biphasic']['d_kl_div'], label="biphasic")
    axs[0,1].legend()
    

    axs[0,2].set_title("EEG Simulation")
    axs[0,2].set_ylabel("Voltage")
    axs[0,2].set_xlabel("Time t (ms)")
    axs[0,2].set_xlim(0,T/2)
    axs[0,2].plot(t, df[df['scenario'] == 'n400']['eeg'], label="N400")
    axs[0,2].plot(t, df[df['scenario'] == 'p600']['eeg'], label="P600")
    axs[0,2].plot(t, df[df['scenario'] == 'biphasic']['eeg'], label="biphasic")
    axs[0,2].legend()

    axs[1,0].set_title("N400 Scenario")
    axs[1,0].set_xlabel("Time t (ms)")
    axs[1,0].set_ylabel("Probability")
    axs[1,0].set_xlim(0, T/2)
    axs[1,0].plot(t, df[df['scenario'] == 'n400'][0], label="Target")
    axs[1,0].plot(t, df[df['scenario'] == 'n400'][1], label="Distractor")
    axs[1,0].plot(t, df[df['scenario'] == 'n400'][2], label="Far Distractor")
    axs[1,0].legend()

    axs[1,1].set_title("P600 Scenario")
    axs[1,1].set_xlabel("Time t (ms)")
    axs[1,1].set_ylabel("Probability")
    axs[1,1].set_xlim(0,T/2)
    axs[1,1].plot(t, df[df['scenario'] == 'p600'][0], label="Target")
    axs[1,1].plot(t, df[df['scenario'] == 'p600'][1], label="Distractor")
    axs[1,1].plot(t, df[df['scenario'] == 'p600'][2], label="Far Distractor")
    axs[1,1].legend()    

    axs[1,2].set_title("Biphasic Scenario")
    axs[1,2].set_xlabel("Time t (ms)")
    axs[1,2].set_ylabel("Probability")
    axs[1,1].set_xlim(0,T/2)    
    axs[1,2].legend(loc='upper right')    
    axs[1,2].plot(t, df[df['scenario'] == 'biphasic'][0], label="Target")
    axs[1,2].plot(t, df[df['scenario'] == 'biphasic'][1], label="Distractor")
    axs[1,2].plot(t, df[df['scenario'] == 'biphasic'][2], label="Far Distractor")
    axs[1,2].legend()

    df['region'] = df['t'].map(lambda t: 'P600' if (t/max_lam)>.1 else 'N400')
    r = df[['region', 'scenario', 'eeg']].groupby(['region', 'scenario']).sum().reset_index()
    categories = ["N400", "P600"]

    axs[2,0].set_ylabel("Sum Voltage")
    axs[2,0].set_xlabel("Time Window")
    axs[2,0].bar(categories, r[r['scenario'] == 'n400']['eeg'])

    axs[2,1].set_ylabel("Sum Voltage")
    axs[2,1].set_xlabel("Time Window")
    axs[2,1].bar(categories, r[r['scenario'] == 'p600']['eeg'])

    axs[2,2].set_ylabel("Sum Voltage")
    axs[2,2].set_xlabel("Time Window")
    axs[2,2].bar(categories, r[r['scenario'] == 'biphasic']['eeg'])        

    plt.tight_layout(pad=2.0)

    return df

                

    
    
    



    


    
    
    
    
    
