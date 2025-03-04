import numpy as np
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt

def noisy_channel_distortion(lnp_m, d_sem, d_phon):
    """
    Input:
    lnp_m: array of size M giving prior log probabilities for meanings m
    d_sem: semantic distance metric of size MS relating strings S and meanings M
    d_phon: phonological distance metric of size SS relating strings S and strings S

    Output:
    (1) a distortion matrix of shape SxS representing D[p(m | s_i) || p(m | s_j)], where
    p(m | s_sent) = 1/Z(s_sent) * exp(-d_sem[m, s_sent])
    p(s | s_sent) = 1/Z(s_sent) * exp(-d_phon[s, s_sent])
    p(m | s) = \sum_{s_sent} p(s | s_sent) p(m | s_sent)

    (2) the distribution p(m | s) on meanings given noisy strings, matrix of shape M x S.
    """
    lnp_sent_given_m = scipy.special.log_softmax(d_sem, -1) # shape MS, p(s_sent | m)
    lnp_received_given_sent = scipy.special.log_softmax(d_phon, -1) # shape SS, p(s | s_sent)
    lnp_received_given_m = scipy.special.logsumexp(lnp_sent_given_m[:, :, None] + lnp_received_given_sent[None, :, :], -2) # shape MS, p(s | m)
    lnp_m_given_received = scipy.special.log_softmax(lnp_m[:, None] + lnp_received_given_m, -2) # shape MS
    p_m_given_received = np.exp(lnp_m_given_received)    
    pairwise = lnp_m_given_received[:, :, None] - lnp_m_given_received[:, None, :] # shape MSS, log probability ratio ln p(m|s_i)/p(m|s_j)
    d = np.einsum("mi,mij->ij", p_m_given_received, pairwise) # D[p(m|s_i) || p(m|s_j)]
    return d, p_m_given_received

def timecourse(
        lnp0,
        d,
        max_t=10,
        t_steps=100,
        beta_0=1,
        beta_schedule=None,
        lnp_m=None,
        p_m_given_s=None,
        vocab=None):
    """
    Give the predicted processing timecourse given K alternative interpretations for a given input, where
    
    * lnp is an array of shape K giving prior log-probabilities for the alternatives.
    * d is an array of shape K giving distortions for each of the K interpretations.

    Optionally provide an iterable of vocabulary items corresponding to entries in lnp0 and d.
    """
    t = np.linspace(0, max_t, t_steps)
    beta_t = beta_0*t if beta_schedule is None else beta_0*beta_schedule(t)
    unnormalized = lnp0[None, :] - beta_t[:, None]*d[None, :] # shape TK
    lnZ = scipy.special.logsumexp(unnormalized, -1) # shape T
    lnp = unnormalized - lnZ[:, None] # shape TK
    p = np.exp(lnp) # shape TK
    df = pd.DataFrame(p) # dataframe containing probabilities for each interpretation at each time
    if vocab is not None:
        df.columns = list(vocab)
    df['t'] = t
    df['beta'] = beta_t
    df['expected_distortion'] = p @ d
    df['variance_distortion'] = p @ d**2 - (p @ d)**2
    df['kl_div'] = -lnZ - beta_t*df['expected_distortion']    
    df['d_kl_div'] = beta_t * df['variance_distortion']
    df['specific_heat'] = beta_t**2 * df['variance_distortion']

    if lnp_m is not None and p_m_given_s is not None:
        p_m_given_interp = p @ p_m_given_s.T
        df['m_kl_div'] = (p_m_given_interp * (np.log(p_m_given_interp) - lnp_m)).sum(-1)
        df['m_d_kl_div'] = df['m_kl_div'].diff()
    return df

def noisy_channel_anecdote_example(
        d_target_near=1,
        d_far=3,
        d_sem_near=1,
        d_sem_far=3,
        p_target=.05,
        p_probable=.4,
        p_near=.3,
        p_far=.01,
        beta_0=700,
        **kwds):
    # alternatives: antidote (target), anecdote (near), story (probable), hearse (far)
    d_phon = np.array([
        [0, d_target_near, d_far, d_far],
        [d_target_near, 0, d_far, d_far],
        [d_far, d_far, 0, d_far],
        [d_far, d_far, d_far, 0],
    ])
    d_sem = np.array([
        [0, d_sem_far, d_sem_far, d_sem_far],
        [d_sem_far, 0, d_sem_near, d_sem_far],
        [d_sem_far, d_sem_near, 0, d_sem_far],
        [d_sem_far, d_sem_far, d_sem_far, 0],
    ])
    unnorm_p = np.array([p_target, p_near, p_probable, p_far])
    lnp_m = scipy.special.log_softmax(np.log(unnorm_p))
    d, p_m_given_s = noisy_channel_distortion(lnp_m, d_sem, d_phon)
    df = timecourse(lnp_m, d[0], lnp_m=lnp_m, p_m_given_s=p_m_given_s, beta_0=beta_0, **kwds)
    return d, p_m_given_s, df

def hunted_example(comma_distortion=1, word_distortion=1, t_old=5, t_new=5, memory_weight=1):
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
    d_cx = memory_weight*d_c + d_x
    
    old_timecourse = timecourse(lnp0, d_c, max_t=t_old)
    old_timecourse['phase'] = 0
    # get last distribution to use as the new default policy
    lnp_c = np.log(old_timecourse[old_timecourse['t'] == old_timecourse['t'].max()][range(4)].to_numpy().squeeze(0))
    new_timecourse = timecourse(lnp_c, d_cx, max_t=t_new)
    new_timecourse['phase'] = 1
    new_timecourse['t'] = new_timecourse['t'] + t_old
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

def example_timecourses(T=2200, scale=5, prior=.1, distractor=.7, farthest=10, schedule=lambda x:x):
    """ Nice-looking example timecourses """
    max_lam = 15
    num_steps = 1000
    t = schedule(np.linspace(0, T, num_steps))

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
    df['eeg'] = -df['specific_heat']*np.sin(2*np.pi*df['processing_time'] / scale)

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
    axs[0,1].set_ylabel("Specific heat")
    #axs[0,1].set_xlim(0,T/2)
    axs[0,1].plot(t, df[df['scenario'] == 'n400']['specific_heat'], label="N400")
    axs[0,1].plot(t, df[df['scenario'] == 'p600']['specific_heat'], label="P600")

    axs[0,1].plot(t, df[df['scenario'] == 'biphasic']['specific_heat'], label="biphasic")
    axs[0,1].legend()
    

    axs[0,2].set_title("EEG Simulation")
    axs[0,2].set_ylabel("Voltage")
    axs[0,2].set_xlabel("Time t (ms)")
    #axs[0,2].set_xlim(0,T/2)
    axs[0,2].plot(t, df[df['scenario'] == 'n400']['eeg'], label="N400")
    axs[0,2].plot(t, df[df['scenario'] == 'p600']['eeg'], label="P600")
    axs[0,2].plot(t, df[df['scenario'] == 'biphasic']['eeg'], label="biphasic")
    axs[0,2].legend()

    axs[1,0].set_title("N400 Scenario")
    axs[1,0].set_xlabel("Time t (ms)")
    axs[1,0].set_ylabel("Probability")
    #axs[1,0].set_xlim(0, T/2)
    axs[1,0].plot(t, df[df['scenario'] == 'n400'][0], label="Target")
    axs[1,0].plot(t, df[df['scenario'] == 'n400'][1], label="Distractor")
    axs[1,0].plot(t, df[df['scenario'] == 'n400'][2], label="Far Distractor")
    axs[1,0].legend()

    axs[1,1].set_title("P600 Scenario")
    axs[1,1].set_xlabel("Time t (ms)")
    axs[1,1].set_ylabel("Probability")
    #axs[1,1].set_xlim(0,T/2)
    axs[1,1].plot(t, df[df['scenario'] == 'p600'][0], label="Target")
    axs[1,1].plot(t, df[df['scenario'] == 'p600'][1], label="Distractor")
    axs[1,1].plot(t, df[df['scenario'] == 'p600'][2], label="Far Distractor")
    axs[1,1].legend()    

    axs[1,2].set_title("Biphasic Scenario")
    axs[1,2].set_xlabel("Time t (ms)")
    axs[1,2].set_ylabel("Probability")
    #axs[1,1].set_xlim(0,T/2)    
    axs[1,2].legend(loc='upper right')    
    axs[1,2].plot(t, df[df['scenario'] == 'biphasic'][0], label="Target")
    axs[1,2].plot(t, df[df['scenario'] == 'biphasic'][1], label="Distractor")
    axs[1,2].plot(t, df[df['scenario'] == 'biphasic'][2], label="Far Distractor")
    axs[1,2].legend()

    df['region'] = df['processing_time'].map(lambda t: 'P600' if (t/max_lam)>.1 else 'N400')
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

                

    
    
    



    


    
    
    
    
    
