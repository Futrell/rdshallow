setwd("~/projects/shallow")
library(tidyverse)
library(latex2exp)

LOG2 = log(2)

d = read_csv("antidote.csv") %>%
  rename(processing_time=`t...10005`)

ds = read_csv("anecdotes.csv") %>%
  rename(processing_time=`t...10005`)

dc = read_csv("anecdote.csv") %>%
  rename(processing_time=`t...10005`)

dh = read_csv("hearse.csv") %>%
  rename(processing_time=`t...10005`)

dm = read_csv("moses.csv") 

p_plot = d %>% 
  select(antidote, anecdote, story, tale, processing_time) %>%
  gather(w, p, -processing_time) %>%
  ggplot(aes(x=processing_time, y=p, color=w)) +
    geom_line() +
    theme_classic() +
    xlim(0, 7) +
    labs(x="Processing Time", y="Probability p(w | x)", color="") +
    scale_y_continuous(breaks=c(0,1)) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          #axis.text.y=element_blank(),
          #axis.ticks.y=element_blank(),
          legend.background = element_rect(fill = "transparent"),
          legend.position=c(.7, .56),
          plot.title = element_text(size = 7)) +
    ggtitle(TeX("The storyteller could turn any story into an amusing $\\textbf{antidote}$"))

d %>% 
  select(antidote, anecdote, story, tale, processing_time) %>%
  gather(w, p, -processing_time) %>%
  ggplot(aes(x=processing_time, y=log(p/(1-p)), color=w)) +
  geom_line() +
  theme_classic() +
  labs(x="Processing Time", y="Log-Odds", color="")

dh %>% 
  select(antidote, anecdote, story, tale, hearse, anecdotes, processing_time) %>%
  gather(w, p, -processing_time) %>%
  ggplot(aes(x=processing_time, y=log(p/(1-p)), color=w)) +
  geom_line() +
  theme_classic() +
  labs(x="Processing Time", y="Log-Odds", color="")


effort_plot = d %>%
  rename(`Instantaneous D'(t)`=d_kl_div, `Cumulative D(t)`=kl_div) %>%
  select(processing_time, `Instantaneous D'(t)`, `Cumulative D(t)`) %>%
  gather(key, value, -processing_time) %>%
  ggplot(aes(x=processing_time, y=value/LOG2, linetype=key)) +
    geom_line() +
    labs(x="Processing Time", y="Processing Effort (bits)", linetype="") +
    theme_classic() +
    xlim(0, 7) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          #axis.text.y=element_blank(),
          #axis.ticks.y=element_blank(),
          legend.background = element_rect(fill = "transparent"),
          legend.position=c(.7, .6),
          plot.title = element_text(size = 7))

wrap_plots(p_plot, effort_plot, ncol=1)
ggsave("antidote_timecourse.pdf", height=3.8, width=3.5)

dc %>%
  ggplot(aes(x=kl_div, y=1/d_kl_div)) +
  geom_line() +
  labs(x="Processing Time (as bits)", y="Change in Heat") +
  theme_classic() +
  ylim(0, 4)

d %>% 
  ggplot(aes(x=kl_div / LOG2, y=expected_distortion)) +
  geom_line() +
  geom_area(fill = "gray", alpha = 0.5) +
  annotate("text", x=2.5, y=1.5, label="unachievable") +
  theme_classic() +
  labs(x="Processing Depth (bits)", y="Expected Distortion") +
  theme(plot.title = element_text(size = 8)) +
  ggtitle(TeX("The storyteller could turn any story into an amusing $\\textbf{antidote}$"))

ggsave("infoplane.pdf", width=3.5, height=3.5)


d_all = d %>% 
  select(processing_time, kl_div, d_kl_div, expected_distortion, variance_distortion) %>%
  mutate(target="antidote") %>%
  bind_rows(dc %>% 
      select(processing_time, kl_div, d_kl_div, expected_distortion, variance_distortion) %>%
      mutate(target="anecdote")) %>%
  bind_rows(ds %>%
      select(processing_time, kl_div, d_kl_div, expected_distortion, variance_distortion) %>%
      mutate(target="anecdotes")) %>%
  bind_rows(dh %>%
              select(processing_time, kl_div, d_kl_div, expected_distortion, variance_distortion) %>%
              mutate(target="hearse"))

plot1 = d_all %>%
  ggplot(aes(x=processing_time, y=d_kl_div, color=target)) +
    geom_line() + 
    labs(x="Processing Time", y="Instantaneous D'(t) (bits)", color="") +
    theme_classic() +
    xlim(0, 7) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          #axis.text.y=element_blank(),
          #axis.ticks.y=element_blank(),
          legend.background = element_rect(fill = "transparent"),
          legend.position=c(.7, .7),
          plot.title = element_text(size = 7)) +
  ggtitle(TeX("The storyteller could turn any story into an amusing..."))


plot2 =  d_all %>%
  ggplot(aes(x=processing_time, y=kl_div, color=target)) +
  geom_line() + 
  labs(x="Processing Time", y="Cumulative D(t) (bits)", color="") +
  guides(color=F) +
  xlim(0, 7) +
  theme_classic() +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        #axis.text.y=element_blank(),
        #axis.ticks.y=element_blank(),
        legend.background = element_rect(fill = "transparent"),
        legend.position=c(.7, .6),
        plot.title = element_text(size = 7))

wrap_plots(plot1, plot2, ncol=1)

ggsave("four_alt.pdf", width=3.5, height=3.8)


plot3 = d_all %>%
  mutate(oscillation=-sin(processing_time - .35)) %>%
  mutate(eeg=d_kl_div*oscillation) %>%
  select(-kl_div, -d_kl_div, -oscillation) %>%
  spread(target, eeg) %>%
  gather(target, eeg, -processing_time, -anecdote) %>%
  mutate(eeg_cont=eeg - anecdote) %>%
  ggplot(aes(x=processing_time, y=eeg_cont, color=target)) +
  #geom_line(aes(y=oscillation), color="gray") +
  geom_hline(yintercept=0, linetype="dashed") +
  geom_line() + 
  labs(x="Processing Time", y=TeX("Voltage ($\\mu V$)"), color="") +
  theme_classic() +
  xlim(0, 7) +
  scale_y_continuous(breaks=c(0)) +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        #axis.text.y=element_blank(),
        #axis.ticks.y=element_blank(),
        legend.background = element_rect(fill = "transparent"),
        legend.position=c(.7, .3),
        plot.title = element_text(size = 7))

plot3
ggsave("eeg_sim_cont.pdf", width=3.5, height=2.5)


d_all %>%
  mutate(oscillation=-sin(processing_time - .5)) %>%
  mutate(eeg=d_kl_div*oscillation) %>%
  ggplot(aes(x=processing_time, y=eeg, color=target)) +
  #geom_line(aes(y=oscillation), color="gray") +
  geom_hline(yintercept=0, linetype="dashed") +
  geom_line() + 
  labs(x="Processing Time", y=TeX("Voltage ($\\mu V$)"), color="") +
  theme_classic() +
  scale_y_continuous(breaks=c(0)) +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        #axis.text.y=element_blank(),
        #axis.ticks.y=element_blank(),
        legend.background = element_rect(fill = "transparent"),
        legend.position=c(.7, .4),
        plot.title = element_text(size = 7))

ggsave("eeg_sim.pdf", width=3.5, height=2.5)

d %>%
  rename(`Instantaneous D'(t)`=d_kl_div, `Cumulative D(t)`=kl_div) %>%
  select(processing_time, `Instantaneous D'(t)`, `Cumulative D(t)`) %>%
  gather(key, value, -processing_time) %>%
  ggplot(aes(x=processing_time, y=value/LOG2, linetype=key)) +
  geom_line() +
  labs(x="Processing Time", y="Processing Effort (bits)", linetype="") +
  theme_classic() +
  xlim(0, 7) +
  ylim(0,10) +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        #axis.text.y=element_blank(),
        #axis.ticks.y=element_blank(),
        legend.background = element_rect(fill = "transparent"),
        legend.position=c(.7, .6),
        plot.title = element_text(size = 7))



dm %>% select(-expected_distortion, -variance_distortion, -kl_div, -d_kl_div, -`...1`) %>%
  gather(word, prob, -processing_time) %>% 
  group_by(processing_time) %>% 
    mutate(is_max=prob == max(prob)) %>% 
    ungroup() %>% 
  filter(is_max) %>% 
  arrange(processing_time) %>% 
  print(n=100) 


d_all %>%
  ggplot(aes(x=kl_div, y=d_kl_div, color=target)) +
  geom_line() + 
  labs(x="Processing Depth D(t)", y="Instantaneous Effort D'(t)", color="") +
  theme_classic() +
  ylim(0, 10) +
  theme(legend.position="bottom")

d_all %>%
  ggplot(aes(x=kl_div, y=variance_distortion, color=target)) +
  geom_line() + 
  labs(x="Processing Depth D(t)", y="Work", color="") +
  theme_classic() +
  theme(legend.position="bottom")


d_all %>%
  ggplot(aes(x=kl_div, y=1/(processing_time*d_kl_div), color=target)) +
  geom_line() + 
  labs(x="Processing Depth", y="Work to Increase Inverse Temperature", color="") +
  theme_classic() +
  ylim(0, 10) +
  theme(legend.position="bottom")

d_all %>%
  ggplot(aes(x=processing_time^2, y=d_kl_div, color=target)) +
  geom_line() + 
  labs(x="Processing Time", y="Work to Increase Inverse Temperature", color="") +
  theme_classic() +
  theme(legend.position="bottom")

d_all %>%
  ggplot(aes(x=processing_time, y=1/(processing_time*d_kl_div), color=target)) +
  geom_line() + 
  labs(x="Processing Time", y="Work to Increase Inverse Temperature", color="") +
  theme_classic() +
  ylim(0, 10) +
  theme(legend.position="bottom")

d_all %>%
  ggplot(aes(x=processing_time, y=variance_distortion, color=target)) +
  geom_line() + 
  labs(x="Inverse Temperature", y="Instantaneous Work", color="") +
  theme_classic() +
  ylim(0, 10) +
  theme(legend.position="bottom") 


dnine = read_csv("nine.csv") 

dnine %>%
  select(three, four, five, nine, processing_time) %>%
  gather(word, prob, -processing_time) %>%
  ggplot(aes(x=processing_time, y=prob, color=word)) +
    geom_line() +
    theme_classic() +
    labs(x="Processing Time", y="Probability p(w|x)", color="") +
    ylim(0,1) +
    #scale_y_continuous(breaks=c(0,1)) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          #axis.text.y=element_blank(),
          #axis.ticks.y=element_blank(),
          legend.background = element_rect(fill = "transparent"),
          legend.position=c(.8, .5),
          plot.title = element_text(size = 6)) +
  ggtitle(TeX("Which date only occurs in a leap year, which occurs once every $\\textbf{nine}$"))

ggsave("illusion.pdf", width=3.5, height=2.5)

