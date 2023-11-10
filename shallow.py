import numpy as np
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt

def timecourse(lnp0, d, max_lam=10, num_steps=100):
    """
    Give the predicted processing timecourse given K alternative interpretations for a given input, where
    
    * lnp is an array of shape K giving prior log-probabilities for the alternatives.
    * d is an array of shape K giving distortions for each of the K interpretations.
    
    Index 0 is assumed to be the "true" interpretation (lowest distortion).
    """
    lam = np.linspace(0, max_lam, num_steps) # shape L
    unnormalized = lnp0[None, :] - lam[:, None]*d[None, :]  # shape LK
    lnZ = scipy.special.logsumexp(unnormalized, -1) # shape L
    lnp = unnormalized - lnZ[:, None] # shape LK
    p = np.exp(lnp) # shape LK
    df = pd.DataFrame(p) # dataframe containing probabilities for each interpretation at each time
    df['t'] = lam
    df['expected_distortion'] = p @ d
    df['variance_distortion'] = p @ d**2 - (p @ d)**2
    df['kl_div'] = -lnZ - lam*df['expected_distortion']
    df['d_kl_div'] = lam * df['variance_distortion']
    return df

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

    fig, axs = plt.subplots(2, 3, figsize=(10, 10))

    

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

    plt.tight_layout(pad=2.0)
    

    return df

                

    
    
    



    


    
    
    
    
    
