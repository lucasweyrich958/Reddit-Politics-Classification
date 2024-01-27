library(tidyverse)
library(RedditExtractoR)

dem <- get_reddit(subreddit = 'democrats', page_threshold = 1)
rep <- get_reddit(subreddit = 'Republican', page_threshold = 1)
gp <- get_reddit(subreddit = 'GreenParty', page_threshold = 1)

demrep <- full_join(dem, rep)
reddit <- full_join(demrep, gp)
reddit <- subset(reddit, select = -c(id, structure, post_date, comm_date, num_comments, upvote_prop, post_score, author, user, comment_score, controversiality, title, post_text, link, domain, URL))

setwd("/Users/Lucas/Library/Mobile Documents/com~apple~CloudDocs/Midland/Junior Spring 2021/HON300/Deep Learning Project")
write.csv(reddit, file = 'reddit.csv')

write.csv(dem, file = 'dem.csv')
write.csv(rep, file = 'rep.csv')
write.csv(gp, file = 'gp.csv')

